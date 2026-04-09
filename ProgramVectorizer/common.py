from __future__ import annotations

import os
from typing import Iterable

import numpy as np

_SKLEARN_OPT_IN_ENV = "PROGRAM_VECTORIZER_USE_SKLEARN"


def quantiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def jaccard(a: set[int] | set[str], b: set[int] | set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def safe_mean(values: Iterable[float]) -> float:
    arr = list(values)
    if not arr:
        return 0.0
    return float(sum(arr) / len(arr))


def should_use_sklearn() -> bool:
    return os.environ.get(_SKLEARN_OPT_IN_ENV, "0") == "1"


def get_adjusted_rand_score_func():
    if not should_use_sklearn():
        return None
    try:
        from sklearn.metrics import adjusted_rand_score  # type: ignore

        return adjusted_rand_score
    except Exception:  # noqa: BLE001
        return None


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Label-permutation-invariant ARI.
    Falls back to a lightweight NumPy implementation when sklearn is unavailable.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"labels shape mismatch: {a.shape} vs {b.shape}")
    n = int(a.size)
    if n <= 1:
        return 1.0

    sk_ari = get_adjusted_rand_score_func()
    if sk_ari is not None:
        return float(sk_ari(a, b))

    _, a_inv = np.unique(a, return_inverse=True)
    _, b_inv = np.unique(b, return_inverse=True)
    a_inv = a_inv.astype(np.int64, copy=False)
    b_inv = b_inv.astype(np.int64, copy=False)

    n_a = int(a_inv.max()) + 1
    n_b = int(b_inv.max()) + 1

    pair_codes = a_inv * np.int64(n_b) + b_inv
    nij = np.bincount(pair_codes, minlength=n_a * n_b).astype(np.int64, copy=False)
    ai = np.bincount(a_inv, minlength=n_a).astype(np.int64, copy=False)
    bj = np.bincount(b_inv, minlength=n_b).astype(np.int64, copy=False)

    def _comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) // 2

    sum_comb_nij = float(_comb2(nij).sum())
    sum_comb_ai = float(_comb2(ai).sum())
    sum_comb_bj = float(_comb2(bj).sum())
    total_comb = float(n * (n - 1) // 2)
    if total_comb <= 0:
        return 1.0

    expected = (sum_comb_ai * sum_comb_bj) / total_comb
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
    denom = max_index - expected
    if abs(denom) <= 1e-12:
        return 1.0

    ari = (sum_comb_nij - expected) / denom
    # Numerical guard.
    return float(max(-1.0, min(1.0, ari)))
