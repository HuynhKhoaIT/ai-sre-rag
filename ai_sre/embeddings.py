"""Embedding provider dạng cắm-rút (pluggable).

Tách riêng tầng embedding để dễ đổi nhà cung cấp:
  - hash                  : offline, deterministic, KHÔNG ngữ nghĩa thật (demo pipeline)
  - sentence-transformers : local, offline, chất lượng tốt
"""

from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (v / n).astype("float32")


class HashEmbedder:
    """Deterministic bag-of-words hashing. Chạy offline, không cần model.

    CẢNH BÁO: không có ngữ nghĩa thật — chỉ để kiểm thử pipeline end-to-end.
    Đổi sang sentence-transformers/voyage để có retrieval chất lượng.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        return _normalize(vecs)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype="float32")

def get_embedder():
    provider = os.getenv("EMBEDDING_PROVIDER", "hash").lower()

    if provider in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(os.getenv("ST_MODEL", "BAAI/bge-small-en-v1.5"))
    return HashEmbedder()
