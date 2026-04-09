from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import gmean, rankdata

try:
    from .stats import quantiles
    from .schema import GSSConfig
except ImportError:
    from GssCalculator.stats import quantiles
    from GssCalculator.schema import GSSConfig


def _prepare_gss_inputs(
    expression: sp.csr_matrix,
    gene_names: np.ndarray,
    remove_mt: bool,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    keep_mask = np.ones(expression.shape[1], dtype=bool)
    if remove_mt:
        lower = np.char.lower(gene_names)
        keep_mask = ~(np.char.startswith(lower, "mt-"))

    expression = expression[:, keep_mask]
    gene_names = gene_names[keep_mask]

    bool_expression = expression.astype(bool).astype(np.float32)
    return expression, gene_names, bool_expression


def _compute_ranks(expression: sp.csr_matrix) -> np.ndarray:
    n_spots, n_genes = expression.shape
    ranks = np.zeros((n_spots, n_genes), dtype=np.float32)
    for i in range(n_spots):
        row = expression.getrow(i).toarray().ravel()
        ranks[i] = rankdata(row, method="average")
    return ranks


def _project_score(f_raw: np.ndarray, projection: str) -> np.ndarray:
    if projection == "exp2":
        return np.exp2(np.clip(f_raw, 0.0, None)) - 1.0
    if projection == "softplus":
        return np.log1p(np.exp(f_raw))
    raise ValueError(f"Unknown projection method: {projection}")


def compute_top_sets(
    expression: sp.csr_matrix,
    gene_names: np.ndarray,
    neighbor_idx: np.ndarray,
    gss_cfg: GSSConfig,
    top_ns: tuple[int, ...],
    remove_mt: bool,
) -> dict[int, list[set[int]]]:
    expression_use, _, bool_expression = _prepare_gss_inputs(expression, gene_names, remove_mt)
    ranks = _compute_ranks(expression_use)
    global_rank = gmean(ranks, axis=0) + 1e-12
    global_frac = np.asarray(bool_expression.mean(axis=0)).ravel() + 1e-12

    top_sets = {int(n): [] for n in top_ns}
    for i in range(expression_use.shape[0]):
        nb = neighbor_idx[i]
        nb = nb[nb >= 0]
        if nb.size == 0:
            for n in top_sets:
                top_sets[n].append(set())
            continue

        region_rank = gmean(ranks[nb], axis=0) + 1e-12
        f_raw = np.log2(region_rank / global_rank)

        if gss_cfg.no_expression_fraction:
            support = np.ones_like(f_raw)
        else:
            local_frac = np.asarray(bool_expression[nb].mean(axis=0)).ravel()
            support = np.clip(local_frac / global_frac, 0.0, 1.0)

        gss_row = _project_score(f_raw * support, gss_cfg.projection)

        for n in top_sets:
            n_use = min(int(n), gss_row.shape[0])
            if n_use == 0:
                top_sets[n].append(set())
                continue
            idx = np.argpartition(-gss_row, n_use - 1)[:n_use]
            top_sets[n].append(set(int(x) for x in idx.tolist()))

    return top_sets


def compute_and_sparsify_gss(
    expression: sp.csr_matrix,
    gene_names: np.ndarray,
    spot_ids: np.ndarray,
    neighbor_idx: np.ndarray,
    gss_cfg: GSSConfig,
    remove_mt: bool,
    top_n_for_qc: tuple[int, ...],
) -> dict:
    expression_use, gene_names_use, bool_expression = _prepare_gss_inputs(expression, gene_names, remove_mt)

    n_spots, n_genes = expression_use.shape
    ranks = _compute_ranks(expression_use)
    global_rank = gmean(ranks, axis=0) + 1e-12
    global_frac = np.asarray(bool_expression.mean(axis=0)).ravel() + 1e-12

    rows: list[dict] = []
    top_sets = {int(n): [] for n in top_n_for_qc}

    nonzero_counts = np.zeros(n_spots, dtype=np.int32)
    top10_share = np.zeros(n_spots, dtype=np.float32)
    gene_nonzero_counts = np.zeros(n_genes, dtype=np.int32)

    for i in range(n_spots):
        nb = neighbor_idx[i]
        nb = nb[nb >= 0]

        if nb.size == 0:
            for n in top_sets:
                top_sets[n].append(set())
            continue

        region_rank = gmean(ranks[nb], axis=0) + 1e-12
        f_raw = np.log2(region_rank / global_rank)

        if gss_cfg.no_expression_fraction:
            support = np.ones_like(f_raw)
        else:
            local_frac = np.asarray(bool_expression[nb].mean(axis=0)).ravel()
            support = np.clip(local_frac / global_frac, 0.0, 1.0)

        f_weighted = f_raw * support
        gss_row = _project_score(f_weighted, gss_cfg.projection)

        positive_idx = np.flatnonzero(gss_row > 0)
        nonzero_counts[i] = int(positive_idx.size)
        if positive_idx.size > 0:
            gene_nonzero_counts[positive_idx] += 1

        total_positive = float(gss_row[positive_idx].sum()) if positive_idx.size else 0.0
        if total_positive > 0:
            top10 = min(10, positive_idx.size)
            top10_idx = np.argpartition(-gss_row[positive_idx], top10 - 1)[:top10]
            top10_share[i] = float(gss_row[positive_idx][top10_idx].sum() / total_positive)

        for n in top_sets:
            n_use = min(int(n), gss_row.shape[0])
            if n_use == 0:
                top_sets[n].append(set())
            else:
                idx_top = np.argpartition(-gss_row, n_use - 1)[:n_use]
                top_sets[n].append(set(int(x) for x in idx_top.tolist()))

        if gss_cfg.sparsify_rule == "positive":
            keep = positive_idx
            if keep.size > 0:
                keep = keep[np.argsort(-gss_row[keep])]
        elif gss_cfg.sparsify_rule == "topM":
            m = min(max(1, gss_cfg.top_m), gss_row.shape[0])
            keep = np.argpartition(-gss_row, m - 1)[:m]
            keep = keep[np.argsort(-gss_row[keep])]
            keep = keep[gss_row[keep] > 0]
        else:
            raise ValueError(f"Unknown sparsify_rule: {gss_cfg.sparsify_rule}")

        for rank_in_spot, gene_idx in enumerate(keep, start=1):
            record = {
                "spot_id": str(spot_ids[i]),
                "gene": str(gene_names_use[gene_idx]),
                "gss": float(gss_row[gene_idx]),
                "rank_in_spot": int(rank_in_spot),
            }
            if gss_cfg.keep_f_raw:
                record["F_raw"] = float(f_raw[gene_idx])
            if gss_cfg.keep_neighbor_support:
                record["neighbor_support"] = float(support[gene_idx])
            rows.append(record)

    sparse_df = pd.DataFrame(rows)
    if sparse_df.empty:
        sparse_df = pd.DataFrame(
            columns=[
                "spot_id",
                "gene",
                "gss",
                "rank_in_spot",
                "F_raw",
                "neighbor_support",
            ]
        )

    distribution = {
        "nonzero_genes_count": quantiles(nonzero_counts.astype(np.float32)),
        "top10_share": quantiles(top10_share.astype(np.float32)),
        "gene_level_nonzero_ratio": quantiles(
            (gene_nonzero_counts / max(1, n_spots)).astype(np.float32)
        ),
    }

    return {
        "sparse_df": sparse_df,
        "top_sets": top_sets,
        "distribution": distribution,
    }
