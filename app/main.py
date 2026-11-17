"""Face Recognition Service: RetinaFace + ArcFace (512-d) + cosine matching (NumPy)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

import base64
import cv2 as _cv2

from .config import Settings
from .db import FaceStore
from .engine import FaceEngine, FaceResult, NoFaceError
from .engine.align import AlignmentError
from .engine.image import decode_image
from .matching import EMBEDDING_DIM, batch_cosine_similarity, cosine_similarity, is_match
from .schemas import (EmbeddingResponse, EnrollResponse, FaceInfo, FaceSummary, IdentifyCandidate,
                      IdentifyResponse, MatchResponse)
from .validation.quality import validate_face_quality
log = logging.getLogger("face-service")


def _result_text(matched: bool) -> str:
    return "Match" if matched else "0 Match"


def _info(r: FaceResult) -> FaceInfo:
    return FaceInfo(bbox=r.bbox, landmarks=r.landmarks, det_score=r.det_score, face_count=r.face_count,
                    rotation_applied=r.rotation_applied, roll_degrees=r.roll_degrees)


def create_app(settings: Optional[Settings] = None, engine=None, store: Optional[FaceStore] = None) -> FastAPI:
    """`engine`/`store` có thể được truyền vào để test mà không cần model ONNX."""
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.engine = engine or FaceEngine(cfg)
        app.state.store = store or FaceStore(cfg.db_path, cfg.vector_ttl_seconds)
        log.info("Service sẵn sàng (detector=%s)", cfg.detector_backend)
        yield

    app = FastAPI(title="Face Recognition Service", version="1.0.0", lifespan=lifespan)

    # ---- dependencies -------------------------------------------------------------
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def check_key(request: Request, key_header: Optional[str] = Security(api_key_header)):
        key = request.app.state.settings.api_key
        if key and key_header != key:
            raise HTTPException(401, "API key không hợp lệ")

    def get_engine(request: Request):
        return request.app.state.engine

    def get_store(request: Request) -> FaceStore:
        return request.app.state.store

    def get_cfg(request: Request) -> Settings:
        return request.app.state.settings

    async def read_image(file: UploadFile, cfg: Settings) -> np.ndarray:
        data = await file.read()
        if not data:
            raise HTTPException(400, "File ảnh rỗng")
        if len(data) > cfg.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Ảnh vượt quá {cfg.max_upload_mb} MB")
        try:
            return decode_image(data)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    def run_extract(eng, img: np.ndarray) -> FaceResult:
        try:
            return eng.extract(img)
        except (NoFaceError, AlignmentError) as e:
            raise HTTPException(422, str(e))

    # ---- endpoints ----------------------------------------------------------------
    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "dim": EMBEDDING_DIM, "detector": cfg.detector_backend,
                "threshold": cfg.match_threshold, "stored_vectors": request.app.state.store.count()}

    @app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(check_key)])
    async def embeddings(file: UploadFile = File(...), eng=Depends(get_engine), c: Settings = Depends(get_cfg)):
        """Ảnh -> vector 512 chiều (không lưu DB). Dành cho nhóm khác tích hợp."""
        r = run_extract(eng, await read_image(file, c))
        return EmbeddingResponse(dim=EMBEDDING_DIM, embedding=r.embedding.tolist(), face=_info(r))

    @app.post("/v1/faces", response_model=EnrollResponse, dependencies=[Depends(check_key)])
    async def enroll(file: UploadFile = File(...), label: Optional[str] = Form(None),
                     return_embedding: bool = Form(False), eng=Depends(get_engine),
                     db: FaceStore = Depends(get_store), c: Settings = Depends(get_cfg)):
        """Bước 1: nhận ảnh người thứ nhất, sinh vector 512-d và lưu vào DB."""
        r = run_extract(eng, await read_image(file, c))
        rec = db.add(r.embedding, label, r.det_score)
        return EnrollResponse(id=rec.id, label=label, dim=EMBEDDING_DIM,
                              embedding=r.embedding.tolist() if return_embedding else None, face=_info(r))

    @app.get("/v1/faces", response_model=List[FaceSummary], dependencies=[Depends(check_key)])
    def list_faces(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                   db: FaceStore = Depends(get_store)):
        return [FaceSummary(id=x.id, label=x.label, det_score=x.det_score, created_at=x.created_at)
                for x in db.list(limit, offset)]

    @app.delete("/v1/faces/{face_id}", dependencies=[Depends(check_key)])
    def delete_face(face_id: str, db: FaceStore = Depends(get_store)):
        if not db.delete(face_id):
            raise HTTPException(404, "Không tìm thấy face_id")
        return {"deleted": face_id}

    @app.post("/v1/faces/{face_id}/match", response_model=MatchResponse, dependencies=[Depends(check_key)])
    async def match_with_stored(face_id: str, file: UploadFile = File(...), threshold: Optional[float] = Form(None),
                                eng=Depends(get_engine), db: FaceStore = Depends(get_store),
                                c: Settings = Depends(get_cfg)):
        """Bước 2: ảnh người thứ hai -> vector 512-d -> so khớp với vector đã lưu (face_id)."""
        ref = db.get(face_id)
        if ref is None:
            raise HTTPException(404, "Không tìm thấy face_id (có thể đã hết hạn)")
        r = run_extract(eng, await read_image(file, c))
        thr = c.match_threshold if threshold is None else threshold
        sim = cosine_similarity(ref.embedding, r.embedding)
        ok = is_match(sim, thr)
        return MatchResponse(result=_result_text(ok), match=int(ok), similarity=sim, distance=1 - sim,
                             threshold=thr, reference_id=face_id, face_b=_info(r))

    @app.post("/v1/compare", response_model=MatchResponse, dependencies=[Depends(check_key)])
    async def compare(file_a: UploadFile = File(...), file_b: UploadFile = File(...),
                      threshold: Optional[float] = Form(None), eng=Depends(get_engine),
                      c: Settings = Depends(get_cfg)):
        """So khớp trực tiếp 2 ảnh, không lưu DB."""
        ra = run_extract(eng, await read_image(file_a, c))
        rb = run_extract(eng, await read_image(file_b, c))
        thr = c.match_threshold if threshold is None else threshold
        sim = cosine_similarity(ra.embedding, rb.embedding)
        ok = is_match(sim, thr)
        return MatchResponse(result=_result_text(ok), match=int(ok), similarity=sim, distance=1 - sim,
                             threshold=thr, face_a=_info(ra), face_b=_info(rb))

    @app.post("/v1/identify", response_model=IdentifyResponse, dependencies=[Depends(check_key)])
    async def identify(file: UploadFile = File(...), top_k: int = Form(5), threshold: Optional[float] = Form(None),
                       eng=Depends(get_engine), db: FaceStore = Depends(get_store), c: Settings = Depends(get_cfg)):
        """1:N – tìm các vector giống nhất trong DB (1 phép nhân ma trận NumPy)."""
        r = run_extract(eng, await read_image(file, c))
        thr = c.match_threshold if threshold is None else threshold
        meta, mat = db.gallery()
        sims = batch_cosine_similarity(r.embedding, mat)
        order = np.argsort(-sims)[:max(1, min(top_k, 50))]
        cands = [IdentifyCandidate(id=meta[i][0], label=meta[i][1], similarity=float(sims[i]),
                                   match=int(is_match(float(sims[i]), thr))) for i in order]
        ok = bool(cands and cands[0].match)
        return IdentifyResponse(result=_result_text(ok), match=int(ok), threshold=thr, candidates=cands,
                                face=_info(r))

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        log.exception("Lỗi không mong đợi")
        return JSONResponse(status_code=500, content={"detail": "Lỗi nội bộ của service"})

    # ---- /api/v1/faces/validate ---------------------------------------------------

    _MESSAGES: dict[str, str] = {
        "OK":               "Ảnh đạt yêu cầu chất lượng khuôn mặt.",
        "INVALID_IMAGE":    "Không đọc được ảnh. Vui lòng gửi file JPEG hoặc PNG hợp lệ.",
        "IMAGE_TOO_LARGE":  "Ảnh vượt quá dung lượng cho phép.",
        "NO_FACE":          "Không phát hiện khuôn mặt trong ảnh.",
        "MULTIPLE_FACES":   "Ảnh có nhiều hơn một khuôn mặt. Vui lòng chụp đúng một người.",
        "FACE_TOO_SMALL":   "Khuôn mặt quá nhỏ hoặc quá xa camera.",
        "LOW_DET_SCORE":    "Độ tin cậy phát hiện khuôn mặt quá thấp.",
        "POSE_ROLL":        "Ảnh bị nghiêng quá nhiều. Vui lòng giữ đầu thẳng.",
        "POSE_YAW":         "Khuôn mặt quay ngang quá nhiều. Vui lòng nhìn thẳng vào camera.",
        "POSE_PITCH":       "Khuôn mặt ngửa hoặc cúi quá nhiều. Vui lòng nhìn thẳng vào camera.",
        "BLURRY":           "Ảnh bị mờ. Vui lòng chụp lại trong điều kiện ánh sáng tốt hơn.",
        "BAD_BRIGHTNESS":   "Ảnh quá tối hoặc quá sáng. Vui lòng điều chỉnh ánh sáng.",
    }

    _QUALITY_FAIL_STATUS: dict[str, int] = {
        "INVALID_IMAGE":  400,
        "IMAGE_TOO_LARGE": 413,
    }

    @app.post("/api/v1/faces/validate")
    def validate_face(
        request: Request,
        file: UploadFile = File(...),
        return_aligned: bool = Form(False),
    ):
        """Kiểm tra chất lượng khuôn mặt; trả kết quả từng check và ảnh aligned tùy chọn.

        Endpoint khai báo def (không async) vì xử lý nặng CPU (ONNX inference).
        FastAPI chạy nó trong threadpool tự động.
        """

        cfg_obj: Settings = request.app.state.settings
        detector = request.app.state.engine.detector

        # Đọc file — chạy trong sync context nên dùng file.file trực tiếp
        raw: bytes = file.file.read()
        if not raw:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "code": "INVALID_IMAGE",
                    "message": _MESSAGES["INVALID_IMAGE"],
                    "data": None,
                    "quality": None,
                },
            )
        if len(raw) > cfg_obj.max_upload_mb * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "code": "IMAGE_TOO_LARGE",
                    "message": _MESSAGES["IMAGE_TOO_LARGE"],
                    "data": None,
                    "quality": None,
                },
            )

        result = validate_face_quality(raw, detector)

        code: str = result["code"]
        passed: bool = result["passed"]
        message: str = _MESSAGES.get(code, code)

        # Ảnh aligned → JPEG base64
        aligned_b64: Optional[str] = None
        if return_aligned and result.get("aligned") is not None:
            ok, buf = _cv2.imencode(".jpg", result["aligned"])
            if ok:
                aligned_b64 = base64.b64encode(buf.tobytes()).decode()

        face_dict = result.get("face")
        data_field: Optional[dict] = None
        if face_dict is not None:
            data_field = {"face": face_dict}
            if return_aligned:
                data_field["aligned_jpeg_b64"] = aligned_b64

        http_status = _QUALITY_FAIL_STATUS.get(code, 200 if passed else 422)

        return JSONResponse(
            status_code=http_status,
            content={
                "success": passed,
                "code": code,
                "message": message,
                "data": data_field,
                "quality": {
                    "passed": result["passed"],
                    "reasons": result["reasons"],
                    "checks": result["checks"],
                },
            },
        )
    return app


app = create_app()
