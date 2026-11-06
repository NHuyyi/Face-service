from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class FaceInfo(BaseModel):
    bbox: List[float]
    landmarks: List[List[float]]
    det_score: float
    face_count: int
    rotation_applied: int
    roll_degrees: float


class EmbeddingResponse(BaseModel):
    dim: int
    embedding: List[float]
    face: FaceInfo


class EnrollResponse(BaseModel):
    id: str
    label: Optional[str] = None
    dim: int
    embedding: Optional[List[float]] = None
    face: FaceInfo


class MatchResponse(BaseModel):
    result: str                 # "Match" | "0 Match"
    match: int                  # 1 | 0
    similarity: float           # cosine similarity
    distance: float             # 1 - similarity
    threshold: float
    reference_id: Optional[str] = None
    face_a: Optional[FaceInfo] = None
    face_b: Optional[FaceInfo] = None


class IdentifyCandidate(BaseModel):
    id: str
    label: Optional[str] = None
    similarity: float
    match: int


class IdentifyResponse(BaseModel):
    result: str
    match: int
    threshold: float
    candidates: List[IdentifyCandidate]
    face: FaceInfo


class FaceSummary(BaseModel):
    id: str
    label: Optional[str] = None
    det_score: float
    created_at: float
