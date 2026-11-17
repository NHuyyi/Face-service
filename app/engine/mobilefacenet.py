"""MobileFaceNet (mobile_face_net.tflite) -> vector 192 chiều đã L2-normalize."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from ..matching import l2_normalize

MOBILEFACENET_DIM = 192


def _get_interpreter(model_path: str):
    """Nạp trình thông dịch TFLite qua ai-edge-litert, tflite_runtime hoặc tensorflow."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError:
        pass

    try:
        from tensorflow.lite import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError as exc:
        raise ImportError(
            "Cần cài đặt 'ai-edge-litert' (hoặc 'tflite-runtime' / 'tensorflow') để chạy MobileFaceNet"
        ) from exc


class MobileFaceNetRecognizer:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model MobileFaceNet tại: {model_path}")

        self.interpreter = _get_interpreter(model_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        self.input_index = input_details["index"]
        self.output_index = output_details["index"]
        self.input_shape = tuple(input_details["shape"])  # Kỳ vọng (1, 112, 112, 3)

        out_dim = output_details["shape"][-1]
        if out_dim != MOBILEFACENET_DIM:
            raise RuntimeError(
                f"Model MobileFaceNet có output shape {output_details['shape']}, kỳ vọng chiều cuối là {MOBILEFACENET_DIM}"
            )

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        """Trích xuất vector 192 chiều từ ảnh khuôn mặt đã căn chỉnh 112x112.

        Luồng xử lý (tái hiện chính xác chuẩn Android FaceMatchHelper.java):
        1. Resize nếu chưa đúng 112x112.
        2. Chuyển hệ màu BGR sang RGB.
        3. Tiền xử lý chuẩn hoá pixel: (pixel - 127.5) / 128.0 (dải giá trị [-1.0, 1.0]).
        4. Thêm chiều batch: (112, 112, 3) -> (1, 112, 112, 3).
        5. Suy luận qua model MobileFaceNet TFLite.
        6. Lấy vector 192 chiều và chuẩn hoá L2-normalize (độ dài vector = 1.0).
        """
        if aligned_bgr_112.shape[:2] != (112, 112):
            aligned_bgr_112 = cv2.resize(aligned_bgr_112, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Chuyển BGR sang RGB
        rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Chuẩn hoá pixel theo chuẩn Android
        normalized = (rgb - 127.5) / 128.0

        # Thêm chiều batch (1, 112, 112, 3)
        input_tensor = np.expand_dims(normalized, axis=0).astype(np.float32)

        # Đưa dữ liệu vào interpreter và chạy suy luận
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()

        # Lấy vector kết quả
        raw_vec = self.interpreter.get_tensor(self.output_index)[0].astype(np.float32)

        if raw_vec.shape[0] != MOBILEFACENET_DIM:
            raise RuntimeError(f"Output nhận được {raw_vec.shape[0]} chiều, kỳ vọng {MOBILEFACENET_DIM}")

        # Chuẩn hoá L2-norm
        return l2_normalize(raw_vec)

