"""So khớp vector bằng NumPy (cosine)."""
from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 512


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity ∈ [-1, 1]. Không giả định vector đã chuẩn hoá."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Kích thước vector không khớp: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Khoảng cách cosine = 1 - similarity ∈ [0, 2]."""
    return 1.0 - cosine_similarity(a, b)


def batch_cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Similarity của 1 vector với N vector (ma trận N×D) bằng một phép nhân ma trận."""
    if gallery.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = l2_normalize(query.ravel())
    g = l2_normalize(gallery)
    return g @ q


def is_match(similarity: float, threshold: float) -> bool:
    return similarity >= threshold
