from __future__ import annotations

from typing import Iterable

import numpy as np


def quantiles(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def safe_mean(values: Iterable[float]) -> float:
    arr = [float(x) for x in values]
    if not arr:
        return 0.0
    return float(sum(arr) / len(arr))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return np.zeros(0, dtype=np.float64)

    order = np.argsort(p)
    ranked = p[order]
    n = float(p.size)

    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(ranked.size - 1, -1, -1):
        rank = float(i + 1)
        val = min(prev, ranked[i] * n / rank)
        q[i] = max(0.0, min(1.0, val))
        prev = q[i]

    out = np.empty_like(q)
    out[order] = q
    return out
