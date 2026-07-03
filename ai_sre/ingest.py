"""Dựng index từ thư mục data/ -> vector store."""

from __future__ import annotations

from pathlib import Path

from .chunking import parse_deploys, parse_postmortem, parse_runbook, parse_telemetry
from .embeddings import get_embedder
from .store import VectorStore


def build_index(data_dir: str = "data", index_dir: str = "index") -> int:
    data = Path(data_dir)
    chunks = []

    pm_dir = data / "postmortems"
    if pm_dir.exists():
        for p in sorted(pm_dir.glob("*.md")):
            chunks += parse_postmortem(p)

    rb_dir = data / "runbooks"
    if rb_dir.exists():
        for p in sorted(rb_dir.glob("*.md")):
            chunks += parse_runbook(p)

    deploys = data / "deploys.jsonl"
    if deploys.exists():
        chunks += parse_deploys(deploys)

    telemetry = data / "telemetry.jsonl"
    if telemetry.exists():
        chunks += parse_telemetry(telemetry)

    if not chunks:
        raise SystemExit(f"Không tìm thấy dữ liệu trong {data_dir}/")

    embedder = get_embedder()
    vectors = embedder.embed([c.content for c in chunks])

    store = VectorStore(embedder.dim)
    store.add(vectors, chunks)
    store.build_bm25()
    store.save(index_dir)
    return len(chunks)
