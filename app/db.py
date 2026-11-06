"""Lưu vector 512 chiều vào SQLite (float32 BLOB)."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .matching import EMBEDDING_DIM


@dataclass
class FaceRecord:
    id: str
    label: Optional[str]
    embedding: np.ndarray
    det_score: float
    created_at: float


class FaceStore:
    def __init__(self, path: str, ttl_seconds: int = 0):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS faces(
                       id TEXT PRIMARY KEY,
                       label TEXT,
                       embedding BLOB NOT NULL,
                       det_score REAL,
                       created_at REAL NOT NULL)"""
            )
            self._conn.commit()
        self.purge_expired()

    @staticmethod
    def _pack(v: np.ndarray) -> bytes:
        v = np.asarray(v, dtype=np.float32).ravel()
        if v.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"Vector phải có {EMBEDDING_DIM} chiều, nhận {v.shape[0]}")
        return v.tobytes()

    @staticmethod
    def _unpack(b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.float32).copy()

    def add(self, embedding: np.ndarray, label: Optional[str] = None, det_score: float = 0.0) -> FaceRecord:
        self.purge_expired()
        rec_id = uuid.uuid4().hex
        now = time.time()
        blob = self._pack(embedding)
        with self._lock:
            self._conn.execute(
                "INSERT INTO faces(id,label,embedding,det_score,created_at) VALUES (?,?,?,?,?)",
                (rec_id, label, blob, det_score, now),
            )
            self._conn.commit()
        return FaceRecord(rec_id, label, np.asarray(embedding, np.float32), det_score, now)

    def get(self, rec_id: str) -> Optional[FaceRecord]:
        self.purge_expired()
        with self._lock:
            row = self._conn.execute(
                "SELECT id,label,embedding,det_score,created_at FROM faces WHERE id=?", (rec_id,)
            ).fetchone()
        return self._row(row) if row else None

    def delete(self, rec_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM faces WHERE id=?", (rec_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def list(self, limit: int = 100, offset: int = 0) -> List[FaceRecord]:
        self.purge_expired()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,label,embedding,det_score,created_at FROM faces "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def gallery(self) -> Tuple[List[Tuple[str, Optional[str]]], np.ndarray]:
        """Trả về ([(id,label)], ma trận N×512) phục vụ tìm kiếm 1:N."""
        self.purge_expired()
        with self._lock:
            rows = self._conn.execute("SELECT id,label,embedding FROM faces").fetchall()
        if not rows:
            return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        meta = [(r[0], r[1]) for r in rows]
        mat = np.stack([self._unpack(r[2]) for r in rows])
        return meta, mat

    def purge_expired(self) -> int:
        if self._ttl <= 0:
            return 0
        with self._lock:
            cur = self._conn.execute("DELETE FROM faces WHERE created_at < ?", (time.time() - self._ttl,))
            self._conn.commit()
        return cur.rowcount

    def _row(self, r) -> FaceRecord:
        return FaceRecord(r[0], r[1], self._unpack(r[2]), r[3], r[4])
