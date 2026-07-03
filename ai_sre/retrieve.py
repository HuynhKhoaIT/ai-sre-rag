"""Hybrid retrieval: vector (FAISS) + keyword (BM25) hợp nhất bằng RRF,
kèm metadata filter (service, time-window) và recency boost.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import numpy as np

_RRF_K = 60  # hằng số Reciprocal Rank Fusion
_TELEMETRY = {"log", "metric", "trace"}


def _recency_boost(ts: str, half_life_days: int = 30, weight: float = 0.05) -> float:
    """Sự cố mới quan trọng hơn — cộng thêm điểm nhỏ suy giảm theo tuổi."""
    if not ts:
        return 0.0
    try:
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age = (dt.datetime.now(dt.timezone.utc) - t).days
    return weight * (0.5 ** (max(age, 0) / half_life_days))


def retrieve(
    store,
    embedder,
    query: str,
    service: Optional[str] = None,
    window_start: Optional[str] = None,
    k: int = 8,
    pool: int = 30,
) -> List[dict]:
    # --- nhánh vector ---
    qv = embedder.embed([query])
    _, idxs = store.index.search(qv, min(pool, len(store.meta)))
    vec_rank = {int(i): r for r, i in enumerate(idxs[0]) if i != -1}

    # --- nhánh keyword ---
    bm_scores = store.bm25.get_scores(query.lower().split())
    bm_top = np.argsort(bm_scores)[::-1][:pool]
    bm_rank = {int(i): r for r, i in enumerate(bm_top)}

    # --- hợp nhất RRF ---
    fused: dict[int, float] = {}
    for i, r in vec_rank.items():
        fused[i] = fused.get(i, 0.0) + 1.0 / (_RRF_K + r)
    for i, r in bm_rank.items():
        fused[i] = fused.get(i, 0.0) + 1.0 / (_RRF_K + r)

    # --- filter + recency ---
    results = []
    for i, score in fused.items():
        m = store.meta[i]
        if service and m["service"] and m["service"] != service:
            continue
        # Chỉ áp time-window cho telemetry thô; runbook/postmortem là kiến thức nền
        # nên bỏ qua cửa sổ thời gian.
        if window_start and m["timestamp"] and m["source_type"] in _TELEMETRY:
            if m["timestamp"] < window_start:
                continue
        score += _recency_boost(m["timestamp"])
        results.append((score, i))

    results.sort(reverse=True)
    return [{**store.meta[i], "score": s} for s, i in results[:k]]
