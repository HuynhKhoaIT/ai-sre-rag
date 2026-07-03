"""Vector store in-process: FAISS (cosine qua inner-product) + BM25 cho hybrid.

PoC dùng FAISS IndexFlatIP. Production: đổi sang Weaviate (metadata filter +
hybrid native) — interface add()/save()/load() giữ nguyên là được.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[dict] = []
        self.texts: List[str] = []
        self.bm25: BM25Okapi | None = None

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        self.index.add(np.asarray(vectors, dtype="float32"))
        for c in chunks:
            self.meta.append({**asdict(c), "id": c.id()})
            self.texts.append(c.content)

    def build_bm25(self) -> None:
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        (d / "meta.json").write_text(
            json.dumps({"dim": self.dim, "meta": self.meta, "texts": self.texts}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        d = Path(directory)
        obj = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        s = cls(obj["dim"])
        s.index = faiss.read_index(str(d / "faiss.index"))
        s.meta = obj["meta"]
        s.texts = obj["texts"]
        s.build_bm25()
        return s
