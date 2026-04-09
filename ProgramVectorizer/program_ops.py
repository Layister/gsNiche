from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import jaccard, quantiles
from .schema import ProgramActivationConfig, ProgramQCConfig

_LOW_INFO_SCAFFOLD_PREFIX_RE = re.compile(
    r"^(AC|AL|AP|LINC|LOC|RP[0-9]|CTD|DEPRECATED_|MIR|SNORD|SNORA|RNU|C\d+orf\d+|[A-Z]\d{5,}(?:\.\d+)?)"
)
_HB_PREFIXES = ("HBA", "HBB", "HBD", "HBG", "HBM", "HBQ")


def _smooth_lower_bound_score(value: float, target: float) -> float:
    v = max(0.0, float(value))
    t = max(1e-8, float(target))
    return float(1.0 - np.exp(-(v / t)))


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    total = float(np.sum(arr))
    if total <= 0:
        if arr.size == 0:
            return arr
        arr = np.ones(arr.shape[0], dtype=np.float32)
        total = float(np.sum(arr))
    return (arr / max(total, 1e-8)).astype(np.float32, copy=False)


def _largest_connected_component_size(active_mask: np.ndarray, spot_neighbors: list[np.ndarray] | None) -> int:
    mask = np.asarray(active_mask, dtype=bool)
    active_idx = np.flatnonzero(mask)
    if active_idx.size == 0:
        return 0
    if spot_neighbors is None or len(spot_neighbors) != mask.shape[0]:
        return int(active_idx.size)

    visited = np.zeros(mask.shape[0], dtype=bool)
    best = 0
    for start in active_idx.tolist():
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        size = 0
        while stack:
            cur = int(stack.pop())
            size += 1
            for nb in np.asarray(spot_neighbors[cur], dtype=np.int64).tolist():
                if nb < 0 or nb >= mask.shape[0] or visited[nb] or (not mask[nb]):
                    continue
                visited[nb] = True
                stack.append(int(nb))
        if size > best:
            best = size
    return int(best)


def _largest_connected_component_indices(
    active_mask: np.ndarray,
    spot_neighbors: list[np.ndarray] | None,
) -> np.ndarray:
    mask = np.asarray(active_mask, dtype=bool)
    active_idx = np.flatnonzero(mask)
    if active_idx.size == 0:
        return np.empty((0,), dtype=np.int32)
    if spot_neighbors is None or len(spot_neighbors) != mask.shape[0]:
        return active_idx.astype(np.int32, copy=False)

    visited = np.zeros(mask.shape[0], dtype=bool)
    best_nodes: list[int] = []
    for start in active_idx.tolist():
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        nodes: list[int] = []
        while stack:
            cur = int(stack.pop())
            nodes.append(cur)
            for nb in np.asarray(spot_neighbors[cur], dtype=np.int64).tolist():
                if nb < 0 or nb >= mask.shape[0] or visited[nb] or (not mask[nb]):
                    continue
                visited[nb] = True
                stack.append(int(nb))
        if len(nodes) > len(best_nodes):
            best_nodes = nodes
    return np.asarray(sorted(best_nodes), dtype=np.int32)


def _connected_components_indices(
    active_mask: np.ndarray,
    spot_neighbors: list[np.ndarray] | None,
) -> list[np.ndarray]:
    mask = np.asarray(active_mask, dtype=bool)
    active_idx = np.flatnonzero(mask)
    if active_idx.size == 0:
        return []
    if spot_neighbors is None or len(spot_neighbors) != mask.shape[0]:
        return [active_idx.astype(np.int32, copy=False)]

    visited = np.zeros(mask.shape[0], dtype=bool)
    components: list[np.ndarray] = []
    for start in active_idx.tolist():
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        nodes: list[int] = []
        while stack:
            cur = int(stack.pop())
            nodes.append(cur)
            for nb in np.asarray(spot_neighbors[cur], dtype=np.int64).tolist():
                if nb < 0 or nb >= mask.shape[0] or visited[nb] or (not mask[nb]):
                    continue
                visited[nb] = True
                stack.append(int(nb))
        components.append(np.asarray(sorted(nodes), dtype=np.int32))
    components.sort(key=lambda arr: (-arr.size, int(arr[0]) if arr.size > 0 else -1))
    return components


def _activation_threshold(values: np.ndarray, cfg: ProgramActivationConfig) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    base_threshold = float(cfg.min_activation)
    pos = arr[arr > 0]
    if pos.size == 0:
        return base_threshold
    q = float(np.clip(cfg.adaptive_min_activation_quantile, 0.0, 1.0))
    if q <= 0.0:
        return base_threshold
    return float(max(base_threshold, float(np.quantile(pos, q))))


def _core_seed_threshold(values: np.ndarray, full_thr: float, cfg: ProgramActivationConfig) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    pos = arr[arr > 0]
    if pos.size == 0:
        return float(full_thr)
    q = float(np.clip(cfg.identity_view_activation_quantile, 0.0, 1.0))
    return float(max(full_thr, float(np.quantile(pos, q))))


def _bool_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.size != b.size:
        return 0.0
    return jaccard(set(np.flatnonzero(a).tolist()), set(np.flatnonzero(b).tolist()))


def _safe_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=np.float32).reshape(-1)
    xb = np.asarray(b, dtype=np.float32).reshape(-1)
    if xa.size == 0 or xb.size == 0 or xa.size != xb.size:
        return 0.0
    if float(np.std(xa)) <= 1e-8 or float(np.std(xb)) <= 1e-8:
        return 0.0
    ra = pd.Series(xa).rank(method="average").to_numpy(dtype=np.float32)
    rb = pd.Series(xb).rank(method="average").to_numpy(dtype=np.float32)
    if float(np.std(ra)) <= 1e-8 or float(np.std(rb)) <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(ra, rb)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(max(-1.0, min(1.0, corr)))


def _component_concentration_summary(
    active_mask: np.ndarray,
    spot_neighbors: list[np.ndarray] | None,
) -> dict[str, float | int]:
    components = _connected_components_indices(active_mask, spot_neighbors)
    if not components:
        return {
            "component_count": 0,
            "largest_active_component_size": 0,
            "largest_active_component_frac": 0.0,
            "top2_component_frac": 0.0,
            "top3_component_frac": 0.0,
        }
    sizes = np.asarray([int(comp.size) for comp in components], dtype=np.float32)
    total = float(np.sum(sizes))
    if total <= 0:
        return {
            "component_count": int(sizes.shape[0]),
            "largest_active_component_size": int(sizes[0]) if sizes.size > 0 else 0,
            "largest_active_component_frac": 0.0,
            "top2_component_frac": 0.0,
            "top3_component_frac": 0.0,
        }
    return {
        "component_count": int(sizes.shape[0]),
        "largest_active_component_size": int(sizes[0]) if sizes.size > 0 else 0,
        "largest_active_component_frac": float(sizes[0] / total) if sizes.size > 0 else 0.0,
        "top2_component_frac": float(np.sum(sizes[:2]) / total),
        "top3_component_frac": float(np.sum(sizes[:3]) / total),
    }


def _stretch_program_confidence(
    raw_scores: np.ndarray,
    low_anchor: float,
    high_anchor: float,
    raw_weight: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr.copy(), arr.copy()
    if arr.size == 1:
        return arr.copy(), np.asarray([0.5], dtype=np.float32)

    pct = pd.Series(arr).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    lo = float(np.clip(low_anchor, 0.0, 1.0))
    hi = float(np.clip(max(high_anchor, lo), 0.0, 1.0))
    rank_component = lo + (hi - lo) * pct
    w_raw = float(np.clip(raw_weight, 0.0, 1.0))
    stretched = w_raw * arr + (1.0 - w_raw) * rank_component
    return np.clip(stretched, 0.0, 1.0).astype(np.float32), pct.astype(np.float32)


def _template_axis_coherence(
    ranked_gene_indices: list[int],
    weights_full_map: dict[int, float],
    scaffold_gene_set: set[int],
    support_gene_set: set[int],
    context_edge_gene_set: set[int],
    top_contributing_gene_set: set[int],
    gene_activation_alignment: dict[int, float],
    gene_identity_view_alignment: dict[int, float],
    top_n: int = 12,
) -> dict[str, float]:
    ranked = [int(g) for g in ranked_gene_indices[: max(1, int(top_n))]]
    if not ranked:
        return {
            "axis_coherence_score": 0.0,
            "axis_head_mass_share": 0.0,
            "axis_scaffold_mass_share": 0.0,
            "axis_support_mass_share": 0.0,
            "axis_context_mass_share": 0.0,
            "axis_non_axis_dilution": 0.0,
            "axis_top_contributor_mass_share": 0.0,
            "axis_alignment_mean": 0.0,
        }

    weights = np.asarray([float(weights_full_map.get(int(g), 0.0)) for g in ranked], dtype=np.float32)
    if float(np.sum(weights)) <= 0:
        weights = np.ones(len(ranked), dtype=np.float32)
    weights = _normalize_weights(weights)
    head_n = min(5, len(ranked))
    head_mass_share = float(np.clip(np.sum(weights[:head_n]), 0.0, 1.0))
    scaffold_mass = 0.0
    support_mass = 0.0
    context_mass = 0.0
    other_mass = 0.0
    top_contributor_mass = 0.0
    align_vals: list[float] = []
    align_wts: list[float] = []

    for g, w in zip(ranked, weights.tolist()):
        if g in scaffold_gene_set:
            scaffold_mass += float(w)
        elif g in support_gene_set:
            support_mass += float(w)
        elif g in context_edge_gene_set:
            context_mass += float(w)
        else:
            other_mass += float(w)
        if g in top_contributing_gene_set:
            top_contributor_mass += float(w)
        align = 0.5 * float(gene_activation_alignment.get(g, 0.0)) + 0.5 * float(gene_identity_view_alignment.get(g, 0.0))
        align_vals.append(float(np.clip(align, 0.0, 1.0)))
        align_wts.append(float(w))

    scaffold_mass_share = float(np.clip(scaffold_mass, 0.0, 1.0))
    support_mass_share = float(np.clip(support_mass, 0.0, 1.0))
    context_mass_share = float(np.clip(context_mass, 0.0, 1.0))
    non_axis_dilution = float(np.clip(context_mass + other_mass, 0.0, 1.0))
    centered_mass_score = float(np.clip(scaffold_mass + 0.70 * support_mass, 0.0, 1.0))
    top_contributor_mass_share = float(np.clip(top_contributor_mass, 0.0, 1.0))
    if align_vals:
        alignment_mean = float(np.average(np.asarray(align_vals, dtype=np.float32), weights=np.asarray(align_wts, dtype=np.float32)))
    else:
        alignment_mean = 0.0
    axis_coherence_score = float(
        np.clip(
            0.30 * head_mass_share
            + 0.25 * (1.0 - non_axis_dilution)
            + 0.25 * top_contributor_mass_share
            + 0.20 * alignment_mean,
            0.0,
            1.0,
        )
    )
    return {
        "axis_coherence_score": axis_coherence_score,
        "axis_head_mass_share": head_mass_share,
        "axis_scaffold_mass_share": scaffold_mass_share,
        "axis_support_mass_share": support_mass_share,
        "axis_context_mass_share": context_mass_share,
        "axis_non_axis_dilution": non_axis_dilution,
        "axis_top_contributor_mass_share": top_contributor_mass_share,
        "axis_alignment_mean": alignment_mean,
    }

def _head_consistency_profile(
    ranked_gene_indices: list[int],
    gene_idx: np.ndarray,
    sub_strength_dense: np.ndarray,
    full_active_mask: np.ndarray,
    gene_activation_alignment: dict[int, float],
    gene_identity_view_alignment: dict[int, float],
    max_head_genes: int = 10,
) -> dict[str, float | int]:
    ranked = [int(g) for g in ranked_gene_indices[: max(2, int(max_head_genes))]]
    if len(ranked) < 2:
        return {
            "head_consistency_score": 0.0,
            "head_mean_pair_corr": 0.0,
            "head_low_pair_frac": 1.0,
            "head_component_count": int(len(ranked)),
            "head_largest_component_frac": 0.0,
            "head_alignment_dispersion": 1.0,
        }

    pos_by_gene = {int(g): i for i, g in enumerate(np.asarray(gene_idx, dtype=np.int64).tolist())}
    head_positions = [pos_by_gene[g] for g in ranked if g in pos_by_gene]
    head_genes = [g for g in ranked if g in pos_by_gene]
    if len(head_positions) < 2:
        return {
            "head_consistency_score": 0.0,
            "head_mean_pair_corr": 0.0,
            "head_low_pair_frac": 1.0,
            "head_component_count": int(len(head_positions)),
            "head_largest_component_frac": 0.0,
            "head_alignment_dispersion": 1.0,
        }

    active_mask = np.asarray(full_active_mask, dtype=bool).reshape(-1)
    if int(np.count_nonzero(active_mask)) >= 12:
        spot_mask = active_mask
    else:
        gene_activity = np.sum(np.asarray(sub_strength_dense[:, head_positions], dtype=np.float32), axis=1) > 0
        if int(np.count_nonzero(gene_activity)) >= 12:
            spot_mask = np.asarray(gene_activity, dtype=bool)
        else:
            spot_mask = np.ones(sub_strength_dense.shape[0], dtype=bool)

    head_matrix = np.asarray(sub_strength_dense[np.asarray(spot_mask, dtype=bool)][:, head_positions], dtype=np.float32)
    n_head = head_matrix.shape[1]
    if head_matrix.shape[0] < 3 or n_head < 2:
        return {
            "head_consistency_score": 0.0,
            "head_mean_pair_corr": 0.0,
            "head_low_pair_frac": 1.0,
            "head_component_count": int(n_head),
            "head_largest_component_frac": 0.0,
            "head_alignment_dispersion": 1.0,
        }

    pair_corrs: list[float] = []
    adjacency = [set() for _ in range(n_head)]
    for i in range(n_head):
        xi = head_matrix[:, i]
        for j in range(i + 1, n_head):
            xj = head_matrix[:, j]
            corr = _safe_positive_correlation(xi, xj)
            pair_corrs.append(float(corr))
            if corr >= 0.25:
                adjacency[i].add(j)
                adjacency[j].add(i)

    mean_pair_corr = float(np.mean(pair_corrs)) if pair_corrs else 0.0
    low_pair_frac = float(np.mean(np.asarray(pair_corrs, dtype=np.float32) < 0.15)) if pair_corrs else 1.0

    visited = np.zeros(n_head, dtype=bool)
    comp_sizes: list[int] = []
    for start in range(n_head):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            cur = int(stack.pop())
            size += 1
            for nb in adjacency[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(int(nb))
        comp_sizes.append(int(size))
    comp_sizes.sort(reverse=True)
    largest_component_frac = float(comp_sizes[0] / max(1, n_head)) if comp_sizes else 0.0
    component_count = int(len(comp_sizes))

    align_vals = np.asarray(
        [
            0.5 * float(gene_activation_alignment.get(int(g), 0.0))
            + 0.5 * float(gene_identity_view_alignment.get(int(g), 0.0))
            for g in head_genes
        ],
        dtype=np.float32,
    )
    if align_vals.size >= 2:
        alignment_dispersion = float(np.std(align_vals))
        alignment_cohesion = float(np.clip(1.0 - alignment_dispersion / 0.35, 0.0, 1.0))
    else:
        alignment_dispersion = 1.0
        alignment_cohesion = 0.0

    head_consistency_score = float(
        np.clip(
            0.35 * mean_pair_corr
            + 0.30 * largest_component_frac
            + 0.20 * alignment_cohesion
            + 0.15 * (1.0 - low_pair_frac),
            0.0,
            1.0,
        )
    )
    return {
        "head_consistency_score": head_consistency_score,
        "head_mean_pair_corr": mean_pair_corr,
        "head_low_pair_frac": float(np.clip(low_pair_frac, 0.0, 1.0)),
        "head_component_count": component_count,
        "head_largest_component_frac": float(np.clip(largest_component_frac, 0.0, 1.0)),
        "head_alignment_dispersion": float(max(0.0, alignment_dispersion)),
    }


def _rank_corr_by_gene_order(ordered_genes_a: list[int] | set[int], ordered_genes_b: list[int] | set[int]) -> float:
    a = [int(x) for x in ordered_genes_a]
    b = [int(x) for x in ordered_genes_b]
    if not a or not b:
        return 0.0
    rank_a = {g: i + 1 for i, g in enumerate(a)}
    rank_b = {g: i + 1 for i, g in enumerate(b)}
    common = sorted(set(rank_a.keys()) & set(rank_b.keys()))
    if len(common) < 3:
        return 0.0
    xa = np.asarray([rank_a[g] for g in common], dtype=np.float32)
    xb = np.asarray([rank_b[g] for g in common], dtype=np.float32)
    if float(np.std(xa)) <= 1e-8 or float(np.std(xb)) <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(xa, xb)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(max(-1.0, min(1.0, corr)))


def _safe_positive_correlation(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=np.float32).reshape(-1)
    xb = np.asarray(b, dtype=np.float32).reshape(-1)
    if xa.size == 0 or xb.size == 0 or xa.size != xb.size:
        return 0.0
    if float(np.std(xa)) <= 1e-8 or float(np.std(xb)) <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(xa, xb)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, 0.0, 1.0))


def _scaffold_content_quality(scaffold_gene_names: list[str]) -> dict[str, float]:
    genes = [str(g) for g in scaffold_gene_names if str(g)]
    top20 = genes[:20]
    top10 = genes[:10]
    top5 = genes[:5]
    weird_top20_frac = float(
        sum(bool(_LOW_INFO_SCAFFOLD_PREFIX_RE.match(g)) for g in top20) / max(1, len(top20))
    )
    weird_top5_frac = float(
        sum(bool(_LOW_INFO_SCAFFOLD_PREFIX_RE.match(g)) for g in top5) / max(1, len(top5))
    )
    hemoglobin_top10_frac = float(
        sum(g.startswith(_HB_PREFIXES) for g in top10) / max(1, len(top10))
    )
    score = float(
        np.clip(
            1.0 - 0.75 * weird_top20_frac - 1.00 * weird_top5_frac - 0.50 * hemoglobin_top10_frac,
            0.0,
            1.0,
        )
    )
    return {
        "scaffold_content_quality_score": score,
        "scaffold_weird_symbol_frac_top20": weird_top20_frac,
        "scaffold_weird_symbol_frac_top5": weird_top5_frac,
        "scaffold_hemoglobin_frac_top10": hemoglobin_top10_frac,
    }


def _infer_full_activation(
    sub_strength: sp.csr_matrix,
    weights_full: np.ndarray,
    spot_neighbors: list[np.ndarray] | None,
    cfg: ProgramActivationConfig,
) -> dict:
    n_spots = int(sub_strength.shape[0])
    full_values = np.asarray(sub_strength @ weights_full, dtype=np.float32).reshape(-1)
    full_thr = _activation_threshold(full_values, cfg=cfg)
    full_active_mask = full_values > full_thr
    full_active_spots = np.flatnonzero(full_active_mask)
    full_main_component = _largest_connected_component_indices(full_active_mask, spot_neighbors)
    main_component_frac = float(full_main_component.size / max(1, full_active_spots.size))
    high_thr = _core_seed_threshold(full_values, full_thr=full_thr, cfg=cfg)
    high_activation_mask = full_values >= high_thr
    high_activation_spots = np.flatnonzero(high_activation_mask)
    if high_activation_spots.size == 0 and full_values.size > 0:
        best = int(np.argmax(full_values))
        if float(full_values[best]) > 0:
            high_activation_spots = np.asarray([best], dtype=np.int32)
            high_activation_mask = np.zeros(n_spots, dtype=bool)
            high_activation_mask[best] = True
    return {
        "full_values": full_values,
        "full_threshold": float(full_thr),
        "full_active_mask": full_active_mask,
        "full_active_spots": full_active_spots,
        "main_component_frac": float(main_component_frac),
        "high_activation_mask": high_activation_mask,
        "high_activation_spots": high_activation_spots.astype(np.int32, copy=False),
    }


def _infer_identity_view_activation(
    active_strength: sp.csr_matrix,
    gene_idx: np.ndarray,
    contribution_raw: np.ndarray,
    weights_full: np.ndarray,
    attachment: np.ndarray,
    role_order: np.ndarray,
    core_count: int,
    cfg: ProgramActivationConfig,
) -> dict:
    core_pos: list[int] = []
    for pos in role_order.tolist():
        if len(core_pos) >= core_count:
            break
        if float(attachment[pos]) >= float(cfg.high_contribution_gene_min_alignment) or len(core_pos) == 0:
            core_pos.append(int(pos))
    if not core_pos and role_order.size > 0:
        core_pos = [int(role_order[0])]

    core_gene_idx = gene_idx[np.asarray(core_pos, dtype=np.int64)] if core_pos else np.empty((0,), dtype=np.int64)
    core_weight_raw = (
        contribution_raw[np.asarray(core_pos, dtype=np.int64)] if core_pos else np.zeros((0,), dtype=np.float32)
    )
    if core_gene_idx.size == 0:
        core_gene_idx = gene_idx[: min(gene_idx.size, max(1, core_count))]
        core_weight_raw = weights_full[: core_gene_idx.size]
    core_weights = _normalize_weights(core_weight_raw)
    core_strength = active_strength[:, core_gene_idx]
    identity_view_values = np.asarray(core_strength @ core_weights, dtype=np.float32).reshape(-1)
    return {
        "core_gene_idx": core_gene_idx.astype(np.int64, copy=False),
        "core_weights": core_weights,
        "identity_view_values": identity_view_values,
    }


def materialize_program_table(
    program_payload: list[dict],
    gene_ids: np.ndarray,
) -> pd.DataFrame:
    records: list[dict] = []
    gene_ids_arr = np.asarray(gene_ids).astype(str, copy=False)
    for item in program_payload:
        pid = str(item["program_id"])
        for g_idx in sorted(item["gene_indices"]):
            g_idx = int(g_idx)
            records.append(
                {
                    "program_id": pid,
                    "gene": str(gene_ids_arr[g_idx]),
                    "weight": float(item.get("weights_full", {}).get(g_idx, 0.0)),
                    "gene_weight_full": float(item.get("weights_full", {}).get(g_idx, 0.0)),
                    "gene_weight_identity_view": float(item.get("weights_identity_view", {}).get(g_idx, 0.0)),
                    "gene_role": str(item.get("role_by_gene", {}).get(g_idx, "context_edge_gene")),
                    "gene_layer": str(item.get("gene_layer_by_gene", {}).get(g_idx, item.get("role_by_gene", {}).get(g_idx, "context_edge_gene"))),
                    "gene_role_score": float(item.get("gene_role_score", {}).get(g_idx, 0.0)),
                    "attachment_score": float(item.get("gene_attachment_score", {}).get(g_idx, 0.0)),
                    "core_contribution": float(item.get("gene_core_contribution", {}).get(g_idx, 0.0)),
                    "is_core_scaffold_gene": bool(g_idx in set(item.get("template_scaffold_gene_indices", set()))),
                    "is_support_gene": bool(g_idx in set(item.get("support_gene_indices", item.get("template_support_gene_indices", set())))),
                    "is_context_edge_gene": bool(g_idx in set(item.get("context_edge_gene_indices", item.get("template_context_edge_gene_indices", set())))),
                    "is_template_scaffold_gene": bool(g_idx in set(item.get("template_scaffold_gene_indices", set()))),
                    "is_template_context_edge_gene": bool(g_idx in set(item.get("template_context_edge_gene_indices", set()))),
                    "is_top_contributing_gene": bool(g_idx in set(item.get("top_contributing_gene_indices", set()))),
                    "idf": float(item.get("idf_by_gene", {}).get(g_idx, 0.0)),
                    "support_spots": int(item.get("support_spots_by_gene", {}).get(g_idx, 0)),
                    "intra_degree": float(item.get("intra_degree_by_gene", {}).get(g_idx, 0.0)),
                    "program_size_genes": int(item.get("program_size_genes", 0)),
                    "program_gene_frac": float(item.get("program_gene_frac", 0.0)),
                    "template_run_support_frac": float(item.get("template_run_support_frac", 0.0)),
                    "template_spot_support_frac": float(item.get("template_spot_support_frac", 0.0)),
                    "template_focus_score": float(item.get("template_focus_score", 0.0)),
                    "program_template_evidence_score": float(
                        item.get("template_evidence_score", item.get("structure_confidence", 0.0))
                    ),
                    "full_active_spot_count": int(item.get("full_active_spot_count", 0)),
                    "identity_view_active_spot_count": int(item.get("identity_view_active_spot_count", 0)),
                    "core_full_consistency": float(item.get("core_full_consistency", 0.0)),
                    "activation_peakiness": float(item.get("activation_peakiness", 0.0)),
                    "activation_entropy": float(item.get("activation_entropy", 0.0)),
                    "activation_sparsity": float(item.get("activation_sparsity", 0.0)),
                    "main_component_frac": float(item.get("main_component_frac", 0.0)),
                    "high_activation_spot_count": int(item.get("high_activation_spot_count", 0)),
                    "redundancy_status": str(item.get("redundancy_status", "retained_primary")),
                    "redundancy_family_size": int(item.get("redundancy_family_size", 1)),
                    "redundancy_family_members": ",".join(
                        [str(x) for x in item.get("redundancy_family_members", [item.get("program_id", "")])]
                    ),
                    "recommended_domain_input": str(
                        item.get("recommended_domain_input", "activation_identity_view_weighted")
                    ),
                    "program_confidence": float(item.get("program_confidence", 0.0)),
                    "validity_status": str(item.get("validity_status", "invalid")),
                    "routing_status": str(item.get("routing_status", "rejected")),
                    "default_use_support_score": float(item.get("default_use_support_score", 0.0)),
                    "default_use_reason_count": int(item.get("default_use_reason_count", 0)),
                    "default_use_reasons": "|".join([str(x) for x in item.get("default_use_reasons", [])]),
                }
            )
    out = pd.DataFrame(records)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "program_id",
                "gene",
                "weight",
                "gene_weight_full",
                "gene_weight_identity_view",
                "gene_role",
                "gene_layer",
                "gene_role_score",
                "attachment_score",
                "core_contribution",
                "is_core_scaffold_gene",
                "is_support_gene",
                "is_context_edge_gene",
                "is_template_scaffold_gene",
                "is_template_context_edge_gene",
                "is_top_contributing_gene",
                "idf",
                "support_spots",
                "intra_degree",
                "template_run_support_frac",
                "template_spot_support_frac",
                "template_focus_score",
                "program_template_evidence_score",
                "program_confidence",
                "validity_status",
                "routing_status",
                "default_use_support_score",
                "default_use_reason_count",
                "default_use_reasons",
                "activation_entropy",
                "activation_sparsity",
                "high_activation_spot_count",
                "redundancy_status",
                "redundancy_family_size",
                "redundancy_family_members",
                "recommended_domain_input",
            ]
        )
    return out


def materialize_rejected_candidate_audit_table(
    program_payload: list[dict],
    gene_ids: np.ndarray,
    program_scores: list[dict],
    rejected_programs: list[dict],
) -> pd.DataFrame:
    gene_ids_arr = np.asarray(gene_ids).astype(str, copy=False)
    payload_by_pid = {str(item.get("program_id")): item for item in program_payload}
    score_by_pid = {str(item.get("program_id")): item for item in program_scores}
    reject_by_pid = {str(item.get("program_id")): item for item in rejected_programs}

    def _genes_from_indices(indices: list[int]) -> list[str]:
        out: list[str] = []
        for g_idx in indices:
            g_int = int(g_idx)
            if 0 <= g_int < gene_ids_arr.shape[0]:
                out.append(str(gene_ids_arr[g_int]))
        return out

    records: list[dict] = []
    for pid, reject_item in reject_by_pid.items():
        payload = payload_by_pid.get(pid, {})
        score_row = score_by_pid.get(pid, {})
        ranked_gene_indices = [int(g) for g in payload.get("ranked_gene_indices", [])]
        scaffold_gene_set = {int(g) for g in payload.get("template_scaffold_gene_indices", set())}
        support_gene_set = {
            int(g) for g in payload.get("support_gene_indices", payload.get("template_support_gene_indices", set()))
        }
        context_edge_gene_set = {
            int(g)
            for g in payload.get(
                "context_edge_gene_indices",
                payload.get("template_context_edge_gene_indices", set()),
            )
        }
        top_contributing_gene_set = {int(g) for g in payload.get("top_contributing_gene_indices", set())}

        scaffold_ranked = [g for g in ranked_gene_indices if g in scaffold_gene_set]
        support_ranked = [g for g in ranked_gene_indices if g in support_gene_set]
        context_ranked = [g for g in ranked_gene_indices if g in context_edge_gene_set]
        top_ranked = [g for g in ranked_gene_indices if g in top_contributing_gene_set]

        scaffold_genes = _genes_from_indices(scaffold_ranked[:20])
        support_genes = _genes_from_indices(support_ranked[:20])
        context_edge_genes = _genes_from_indices(context_ranked[:20])
        top_contributing_genes = _genes_from_indices(top_ranked[:20])
        top20_ranked_genes = _genes_from_indices(ranked_gene_indices[:20])

        reject_reasons = [str(x) for x in reject_item.get("reasons", [])]
        records.append(
            {
                "program_id": str(pid),
                "program_confidence": float(
                    score_row.get("program_confidence", reject_item.get("program_confidence", 0.0))
                ),
                "validity_status": str(score_row.get("validity_status", reject_item.get("validity_status", "invalid"))),
                "routing_status": str(score_row.get("routing_status", reject_item.get("routing_status", "rejected"))),
                "default_use_support_score": float(score_row.get("default_use_support_score", 0.0)),
                "default_use_reason_count": int(score_row.get("default_use_reason_count", 0)),
                "default_use_reasons": "|".join([str(x) for x in score_row.get("default_use_reasons", [])]),
                "reject_reasons": "|".join(reject_reasons),
                "reject_reason_count": int(len(reject_reasons)),
                "rejected_by_scaffold_content_quality": bool(
                    any(r.startswith("scaffold_content_quality<") for r in reject_reasons)
                ),
                "rejected_by_activation_shape": bool(any(r.startswith("activation_shape=") for r in reject_reasons)),
                "rejected_by_stability": bool("stability_evidence_failed" in reject_reasons),
                "template_evidence_score": float(score_row.get("template_evidence_score", 0.0)),
                "stability_score": float(score_row.get("stability_score", 0.0)),
                "activation_score": float(score_row.get("activation_score", 0.0)),
                "activation_morphology_score": float(score_row.get("activation_morphology_score", 0.0)),
                "fragmented_rescue_score": float(score_row.get("fragmented_rescue_score", 0.0)),
                "existence_score": float(score_row.get("existence_score", 0.0)),
                "activation_coverage": float(score_row.get("activation_coverage", 0.0)),
                "core_full_consistency": float(score_row.get("core_full_consistency", 0.0)),
                "activation_peakiness": float(score_row.get("activation_peakiness", 0.0)),
                "activation_entropy": float(score_row.get("activation_entropy", 0.0)),
                "activation_sparsity": float(score_row.get("activation_sparsity", 0.0)),
                "main_component_frac": float(score_row.get("main_component_frac", 0.0)),
                "high_activation_spot_count": int(score_row.get("high_activation_spot_count", 0)),
                "activation_shape_class": str(score_row.get("activation_shape_class", "")),
                "template_run_support_frac": float(score_row.get("template_run_support_frac", 0.0)),
                "template_spot_support_frac": float(score_row.get("template_spot_support_frac", 0.0)),
                "template_focus_score": float(score_row.get("template_focus_score", 0.0)),
                "scaffold_content_quality_score": float(score_row.get("scaffold_content_quality_score", 0.0)),
                "scaffold_weird_symbol_frac_top20": float(score_row.get("scaffold_weird_symbol_frac_top20", 0.0)),
                "scaffold_weird_symbol_frac_top5": float(score_row.get("scaffold_weird_symbol_frac_top5", 0.0)),
                "scaffold_hemoglobin_frac_top10": float(score_row.get("scaffold_hemoglobin_frac_top10", 0.0)),
                "n_scaffold_genes": int(len(scaffold_ranked)),
                "n_support_genes": int(len(support_ranked)),
                "n_context_edge_genes": int(len(context_ranked)),
                "scaffold_genes": "|".join(scaffold_genes),
                "support_genes": "|".join(support_genes),
                "context_edge_genes": "|".join(context_edge_genes),
                "top_contributing_genes": "|".join(top_contributing_genes),
                "top20_ranked_genes": "|".join(top20_ranked_genes),
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "program_id",
                "program_confidence",
                "validity_status",
                "routing_status",
                "default_use_support_score",
                "default_use_reason_count",
                "default_use_reasons",
                "reject_reasons",
                "reject_reason_count",
                "rejected_by_scaffold_content_quality",
                "rejected_by_activation_shape",
                "rejected_by_stability",
                "template_evidence_score",
                "stability_score",
                "activation_score",
                "activation_morphology_score",
                "fragmented_rescue_score",
                "existence_score",
                "activation_coverage",
                "core_full_consistency",
                "activation_peakiness",
                "activation_entropy",
                "activation_sparsity",
                "main_component_frac",
                "high_activation_spot_count",
                "activation_shape_class",
                "template_run_support_frac",
                "template_spot_support_frac",
                "template_focus_score",
                "scaffold_content_quality_score",
                "scaffold_weird_symbol_frac_top20",
                "scaffold_weird_symbol_frac_top5",
                "scaffold_hemoglobin_frac_top10",
                "n_scaffold_genes",
                "n_support_genes",
                "n_context_edge_genes",
                "scaffold_genes",
                "support_genes",
                "context_edge_genes",
                "top_contributing_genes",
                "top20_ranked_genes",
            ]
        )
    return out


def _program_stage_diagnostic_profile(
    item: dict,
    gene_ids: np.ndarray,
    spot_neighbors: list[np.ndarray] | None,
) -> dict[str, float | int]:
    gene_ids_arr = np.asarray(gene_ids).astype(str, copy=False)
    gene_indices = sorted(int(g) for g in item.get("gene_indices", set()))
    program_gene_count = int(len(gene_indices))
    ranked_gene_indices = [int(g) for g in item.get("ranked_gene_indices", []) if int(g) in set(gene_indices)]
    if not ranked_gene_indices:
        ranked_gene_indices = list(gene_indices)

    scaffold_gene_set = {int(g) for g in item.get("template_scaffold_gene_indices", set()) if int(g) in set(gene_indices)}
    support_gene_set = {
        int(g)
        for g in item.get("support_gene_indices", item.get("template_support_gene_indices", set()))
        if int(g) in set(gene_indices)
    }
    context_edge_gene_set = {
        int(g)
        for g in item.get("context_edge_gene_indices", item.get("template_context_edge_gene_indices", set()))
        if int(g) in set(gene_indices)
    }
    top_contributing_gene_set = {
        int(g) for g in item.get("top_contributing_gene_indices", set()) if int(g) in set(gene_indices)
    }

    scaffold_gene_count = int(len(scaffold_gene_set))
    support_gene_count = int(len(support_gene_set))
    context_edge_gene_count = int(len(context_edge_gene_set))
    denom = max(1, program_gene_count)
    scaffold_gene_frac = float(scaffold_gene_count / denom)
    support_gene_frac = float(support_gene_count / denom)
    context_edge_gene_frac = float(context_edge_gene_count / denom)
    scaffold_vs_context_ratio = float(scaffold_gene_count / max(1.0, scaffold_gene_count + context_edge_gene_count))
    scaffold_vs_support_ratio = float(scaffold_gene_count / max(1.0, scaffold_gene_count + support_gene_count))

    axis_profile = _template_axis_coherence(
        ranked_gene_indices=ranked_gene_indices,
        weights_full_map={int(k): float(v) for k, v in dict(item.get("weights_full", {})).items()},
        scaffold_gene_set=scaffold_gene_set,
        support_gene_set=support_gene_set,
        context_edge_gene_set=context_edge_gene_set,
        top_contributing_gene_set=top_contributing_gene_set,
        gene_activation_alignment={int(k): float(v) for k, v in dict(item.get("gene_activation_alignment", {})).items()},
        gene_identity_view_alignment={int(k): float(v) for k, v in dict(item.get("gene_identity_view_alignment", {})).items()},
        top_n=12,
    )

    scaffold_ranked = [g for g in ranked_gene_indices if g in scaffold_gene_set]
    scaffold_gene_names = [
        str(gene_ids_arr[g]) for g in scaffold_ranked[:20] if 0 <= int(g) < int(gene_ids_arr.shape[0])
    ]
    scaffold_quality = _scaffold_content_quality(scaffold_gene_names)

    support_mask = np.asarray(item.get("support_mask", []), dtype=bool).reshape(-1)
    support_spot_count = int(np.count_nonzero(support_mask))
    support_spot_frac = float(
        item.get("template_spot_support_frac", support_spot_count / max(1, support_mask.size if support_mask.size > 0 else 1))
    )
    support_shape = _component_concentration_summary(support_mask, spot_neighbors) if support_mask.size > 0 else {
        "component_count": 0,
        "largest_active_component_size": 0,
        "largest_active_component_frac": 0.0,
        "top2_component_frac": 0.0,
        "top3_component_frac": 0.0,
    }

    high_activation_mask = np.asarray(item.get("high_activation_mask", []), dtype=bool).reshape(-1)
    activation_shape = _component_concentration_summary(high_activation_mask, spot_neighbors) if high_activation_mask.size > 0 else {
        "component_count": 0,
        "largest_active_component_size": 0,
        "largest_active_component_frac": 0.0,
        "top2_component_frac": 0.0,
        "top3_component_frac": 0.0,
    }

    return {
        "program_size_genes": int(program_gene_count),
        "template_run_support_frac": float(item.get("template_run_support_frac", 0.0)),
        "template_spot_support_frac": float(support_spot_frac),
        "template_focus_score": float(item.get("template_focus_score", 0.0)),
        "template_evidence_score": float(item.get("template_evidence_score", item.get("structure_confidence", 0.0))),
        "scaffold_gene_count": int(scaffold_gene_count),
        "support_gene_count": int(support_gene_count),
        "context_edge_gene_count": int(context_edge_gene_count),
        "scaffold_gene_frac": float(scaffold_gene_frac),
        "support_gene_frac": float(support_gene_frac),
        "context_edge_gene_frac": float(context_edge_gene_frac),
        "scaffold_vs_context_ratio": float(scaffold_vs_context_ratio),
        "scaffold_vs_support_ratio": float(scaffold_vs_support_ratio),
        "axis_coherence_score": float(axis_profile["axis_coherence_score"]),
        "axis_head_mass_share": float(axis_profile["axis_head_mass_share"]),
        "axis_scaffold_mass_share": float(axis_profile["axis_scaffold_mass_share"]),
        "axis_support_mass_share": float(axis_profile["axis_support_mass_share"]),
        "axis_context_mass_share": float(axis_profile["axis_context_mass_share"]),
        "axis_non_axis_dilution": float(axis_profile["axis_non_axis_dilution"]),
        "axis_top_contributor_mass_share": float(axis_profile["axis_top_contributor_mass_share"]),
        "axis_alignment_mean": float(axis_profile["axis_alignment_mean"]),
        "head_consistency_score": float(item.get("head_consistency_score", 0.0)),
        "head_mean_pair_corr": float(item.get("head_mean_pair_corr", 0.0)),
        "head_low_pair_frac": float(item.get("head_low_pair_frac", 1.0)),
        "head_component_count": int(item.get("head_component_count", 0)),
        "head_largest_component_frac": float(item.get("head_largest_component_frac", 0.0)),
        "head_alignment_dispersion": float(item.get("head_alignment_dispersion", 1.0)),
        "scaffold_content_quality_score": float(scaffold_quality["scaffold_content_quality_score"]),
        "scaffold_weird_symbol_frac_top20": float(scaffold_quality["scaffold_weird_symbol_frac_top20"]),
        "scaffold_weird_symbol_frac_top5": float(scaffold_quality["scaffold_weird_symbol_frac_top5"]),
        "scaffold_hemoglobin_frac_top10": float(scaffold_quality["scaffold_hemoglobin_frac_top10"]),
        "support_spot_count": int(support_spot_count),
        "support_component_count": int(support_shape["component_count"]),
        "support_largest_component_frac": float(support_shape["largest_active_component_frac"]),
        "support_top2_component_frac": float(support_shape["top2_component_frac"]),
        "support_top3_component_frac": float(support_shape["top3_component_frac"]),
        "activation_peakiness": float(item.get("activation_peakiness", 0.0)),
        "activation_entropy": float(item.get("activation_entropy", 0.0)),
        "activation_sparsity": float(item.get("activation_sparsity", 0.0)),
        "main_component_frac": float(item.get("main_component_frac", 0.0)),
        "high_activation_spot_count": int(item.get("high_activation_spot_count", 0)),
        "spatial_locality": float(item.get("spatial_locality", 0.0)),
        "core_full_consistency": float(item.get("core_full_consistency", 0.0)),
        "activation_contrast_ratio": float(item.get("activation_contrast_ratio", 0.0)),
        "activation_dominance": float(item.get("activation_dominance", 0.0)),
        "activation_component_count": int(activation_shape["component_count"]),
        "activation_largest_component_frac": float(activation_shape["largest_active_component_frac"]),
        "activation_top2_component_frac": float(activation_shape["top2_component_frac"]),
        "activation_top3_component_frac": float(activation_shape["top3_component_frac"]),
    }


def materialize_program_stage_diagnostics_table(
    candidate_program_payload: list[dict],
    refined_program_payload: list[dict],
    refinement_table: pd.DataFrame,
    gene_ids: np.ndarray,
    spot_neighbors: list[np.ndarray] | None,
    score_by_pid: dict[str, dict] | None = None,
) -> pd.DataFrame:
    candidate_by_pid = {str(item.get("program_id")): item for item in candidate_program_payload}
    refined_by_pid = {str(item.get("program_id")): item for item in refined_program_payload}
    refinement_by_pid = (
        {
            str(row.get("program_id")): row
            for row in refinement_table.to_dict(orient="records")
        }
        if refinement_table is not None and not refinement_table.empty
        else {}
    )
    score_by_pid = score_by_pid or {}

    all_program_ids = sorted(set(candidate_by_pid.keys()) | set(refined_by_pid.keys()) | set(refinement_by_pid.keys()))
    records: list[dict] = []
    delta_keys = [
        "program_size_genes",
        "template_focus_score",
        "template_evidence_score",
        "scaffold_gene_frac",
        "support_gene_frac",
        "context_edge_gene_frac",
        "scaffold_vs_context_ratio",
        "scaffold_vs_support_ratio",
        "axis_coherence_score",
        "axis_head_mass_share",
        "axis_scaffold_mass_share",
        "axis_support_mass_share",
        "axis_context_mass_share",
        "axis_non_axis_dilution",
        "axis_top_contributor_mass_share",
        "head_consistency_score",
        "head_mean_pair_corr",
        "head_low_pair_frac",
        "head_largest_component_frac",
        "head_alignment_dispersion",
        "scaffold_content_quality_score",
        "support_largest_component_frac",
        "support_top2_component_frac",
        "support_top3_component_frac",
        "activation_peakiness",
        "activation_entropy",
        "activation_sparsity",
        "main_component_frac",
        "spatial_locality",
        "core_full_consistency",
        "activation_contrast_ratio",
        "activation_dominance",
        "activation_largest_component_frac",
        "activation_top2_component_frac",
        "activation_top3_component_frac",
    ]

    for pid in all_program_ids:
        candidate_item = candidate_by_pid.get(pid)
        refined_item = refined_by_pid.get(pid)
        refinement_row = refinement_by_pid.get(pid, {})
        score_row = score_by_pid.get(pid, {})

        record: dict[str, object] = {
            "program_id": str(pid),
            "candidate_exists": bool(candidate_item is not None),
            "refined_exists": bool(refined_item is not None),
            "refinement_status": str(refinement_row.get("refinement_status", "missing")),
            "pruned_gene_count": int(refinement_row.get("pruned_gene_count", 0)),
            "program_size_before": int(refinement_row.get("program_size_before", 0)),
            "program_size_after": int(refinement_row.get("program_size_after", 0)),
            "validity_status": str(score_row.get("validity_status", "")),
            "routing_status": str(score_row.get("routing_status", "")),
            "program_confidence": float(score_row.get("program_confidence", np.nan)),
            "program_confidence_raw": float(score_row.get("program_confidence_raw", np.nan)),
            "default_use_support_score": float(score_row.get("default_use_support_score", np.nan)),
            "default_use_reason_count": int(score_row.get("default_use_reason_count", 0)),
        }

        candidate_profile: dict[str, float | int] = {}
        refined_profile: dict[str, float | int] = {}
        if candidate_item is not None:
            candidate_profile = _program_stage_diagnostic_profile(candidate_item, gene_ids=gene_ids, spot_neighbors=spot_neighbors)
            for key, value in candidate_profile.items():
                record[f"candidate_{key}"] = value
        if refined_item is not None:
            refined_profile = _program_stage_diagnostic_profile(refined_item, gene_ids=gene_ids, spot_neighbors=spot_neighbors)
            for key, value in refined_profile.items():
                record[f"refined_{key}"] = value

        for key in delta_keys:
            c_val = candidate_profile.get(key)
            r_val = refined_profile.get(key)
            if c_val is None or r_val is None:
                record[f"delta_{key}"] = np.nan
            else:
                record[f"delta_{key}"] = float(r_val) - float(c_val)
        records.append(record)

    out = pd.DataFrame(records)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "program_id",
                "candidate_exists",
                "refined_exists",
                "refinement_status",
                "pruned_gene_count",
                "program_size_before",
                "program_size_after",
                "validity_status",
                "routing_status",
                "program_confidence",
                "program_confidence_raw",
                "default_use_support_score",
                "default_use_reason_count",
            ]
        )
    return out


def compute_activation(
    active_strength: sp.csr_matrix,
    active_mask_binary: sp.csr_matrix,
    spot_ids: np.ndarray,
    program_payload: list[dict],
    spot_neighbors: list[np.ndarray] | None,
    cfg: ProgramActivationConfig,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n_spots = int(active_strength.shape[0])
    if not program_payload:
        empty = pd.DataFrame(
            columns=[
                "program_id",
                "spot_id",
                "activation",
                "activation_identity_view",
                "activation_full",
                "activation_view",
                "rank_in_spot",
            ]
        )
        dense_activation = np.zeros((n_spots, 0), dtype=np.float32)
        summary = {
            "activation_view_for_domain": "identity_view",
            "recommended_domain_input": "activation_identity_view_weighted",
            "nonzero_programs_per_spot": quantiles(np.asarray([], dtype=np.float32)),
            "top_program_dominance": quantiles(np.asarray([], dtype=np.float32)),
            "activation_coverage_by_program": quantiles(np.asarray([], dtype=np.float32)),
            "full_activation_coverage_by_program": quantiles(np.asarray([], dtype=np.float32)),
            "effective_activation_threshold_by_program": {},
            "effective_activation_full_threshold_by_program": {},
            "high_activation_spot_count_quantiles": quantiles(np.asarray([], dtype=np.float32)),
            "activation_entropy_quantiles": quantiles(np.asarray([], dtype=np.float32)),
            "activation_sparsity_quantiles": quantiles(np.asarray([], dtype=np.float32)),
        }
        return empty, dense_activation, summary

    active_strength = active_strength.tocsr()
    active_mask_binary = active_mask_binary.tocsr()
    dense_activation_full = np.zeros((n_spots, len(program_payload)), dtype=np.float32)
    dense_activation_core = np.zeros((n_spots, len(program_payload)), dtype=np.float32)
    threshold_by_program: dict[str, float] = {}
    threshold_full_by_program: dict[str, float] = {}
    high_activation_counts: list[float] = []
    activation_entropy_values: list[float] = []
    activation_sparsity_values: list[float] = []

    for j, item in enumerate(program_payload):
        gene_idx = np.asarray(sorted(item["gene_indices"]), dtype=np.int64)
        if gene_idx.size == 0:
            continue

        sub_strength = active_strength[:, gene_idx]
        sub_binary = active_mask_binary[:, gene_idx]
        weights_full = np.asarray([item["weights_full"].get(int(g), 0.0) for g in gene_idx], dtype=np.float32)
        weights_full = _normalize_weights(weights_full)
        full_payload = _infer_full_activation(
            sub_strength=sub_strength,
            weights_full=weights_full,
            spot_neighbors=spot_neighbors,
            cfg=cfg,
        )
        full_values = np.asarray(full_payload["full_values"], dtype=np.float32)
        dense_activation_full[:, j] = full_values
        full_thr = float(full_payload["full_threshold"])
        full_active_mask = np.asarray(full_payload["full_active_mask"], dtype=bool)
        full_active_spots = np.asarray(full_payload["full_active_spots"], dtype=np.int64)
        main_component_frac = float(full_payload["main_component_frac"])
        high_activation_spots = np.asarray(full_payload["high_activation_spots"], dtype=np.int32)
        high_activation_mask = np.asarray(full_payload["high_activation_mask"], dtype=bool)

        sub_strength_dense = np.asarray(sub_strength.toarray(), dtype=np.float32)
        sub_binary_dense = np.asarray(sub_binary.toarray() > 0, dtype=np.float32)
        if high_activation_spots.size > 0:
            core_mean = np.mean(sub_strength_dense[high_activation_spots], axis=0).astype(np.float32, copy=False)
            core_active_frac = np.mean(sub_binary_dense[high_activation_spots], axis=0).astype(np.float32, copy=False)
        else:
            core_mean = np.zeros(gene_idx.shape[0], dtype=np.float32)
            core_active_frac = np.zeros(gene_idx.shape[0], dtype=np.float32)

        outside_mask = ~high_activation_mask
        if np.any(outside_mask):
            outside_mean = np.mean(sub_strength_dense[outside_mask], axis=0).astype(np.float32, copy=False)
        else:
            outside_mean = np.zeros(gene_idx.shape[0], dtype=np.float32)
        contribution_raw = weights_full * core_mean
        contribution_share = _normalize_weights(contribution_raw)
        attachment = (core_mean / (core_mean + outside_mean + 1e-8)).astype(np.float32, copy=False)
        role_score = (
            0.50 * attachment + 0.30 * core_active_frac + 0.20 * contribution_share
        ).astype(np.float32, copy=False)

        core_count = min(
            gene_idx.size,
            max(
                int(cfg.high_contribution_gene_min_count),
                int(np.ceil(float(cfg.high_contribution_gene_fraction) * gene_idx.size)),
            ),
        )
        role_order = np.argsort(-role_score)
        identity_view_payload = _infer_identity_view_activation(
            active_strength=active_strength,
            gene_idx=gene_idx,
            contribution_raw=contribution_raw,
            weights_full=weights_full,
            attachment=attachment,
            role_order=role_order,
            core_count=core_count,
            cfg=cfg,
        )
        core_gene_idx = np.asarray(identity_view_payload["core_gene_idx"], dtype=np.int64)
        core_weights = np.asarray(identity_view_payload["core_weights"], dtype=np.float32)
        identity_view_values = np.asarray(identity_view_payload["identity_view_values"], dtype=np.float32)
        dense_activation_core[:, j] = identity_view_values

        core_thr = _activation_threshold(identity_view_values, cfg=cfg)
        identity_view_mask = identity_view_values > core_thr
        seedable_component = _largest_connected_component_indices(identity_view_mask, spot_neighbors)
        high_activation_counts.append(float(np.count_nonzero(identity_view_mask)))

        identity_view_active_spots = np.flatnonzero(identity_view_mask)
        peak_value = float(np.max(identity_view_values)) if identity_view_values.size > 0 else 0.0
        baseline = float(np.median(full_values[full_active_mask])) if np.any(full_active_mask) else 0.0
        peakiness = float(max(0.0, (peak_value - baseline) / (peak_value + 1e-8)))
        core_full_consistency = float(_bool_jaccard(identity_view_mask, full_active_mask))
        spatial_locality = float(seedable_component.size / max(1, identity_view_active_spots.size))
        high_activation_spot_count = int(np.count_nonzero(identity_view_mask))
        if np.any(full_values > 0):
            mass = full_values[full_values > 0].astype(np.float64, copy=False)
            probs = mass / max(1e-8, float(np.sum(mass)))
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            max_entropy = np.log(max(2, probs.size))
            activation_entropy = float(entropy / max(max_entropy, 1e-8))
        else:
            activation_entropy = 0.0
        activation_sparsity = float(1.0 - (np.count_nonzero(full_values > 0) / max(1, full_values.size)))

        if np.any(identity_view_mask):
            active_vals = identity_view_values[identity_view_mask]
            inactive_vals = identity_view_values[~identity_view_mask]
            active_mean = float(np.mean(active_vals))
            inactive_mean = float(np.mean(inactive_vals)) if inactive_vals.size > 0 else 0.0
            activation_contrast_ratio = float(
                max(0.0, (active_mean - inactive_mean) / (active_mean + inactive_mean + 1e-8))
            )
            dominance = float(np.max(active_vals) / max(1e-8, np.sum(active_vals)))
        else:
            activation_contrast_ratio = 0.0
            dominance = 0.0

        gene_activation_alignment: dict[int, float] = {}
        gene_identity_view_alignment: dict[int, float] = {}
        gene_activation_contribution: dict[int, float] = {}
        activation_contrib_arr = np.zeros(gene_idx.shape[0], dtype=np.float32)
        for pos, g in enumerate(gene_idx.tolist()):
            g = int(g)
            gene_signal = sub_strength_dense[:, pos]
            align_full = _safe_positive_correlation(gene_signal, full_values)
            align_core = _safe_positive_correlation(gene_signal, identity_view_values)
            contrib = float(
                np.clip(
                    0.35 * float(weights_full[pos])
                    + 0.30 * float(contribution_share[pos])
                    + 0.20 * float(align_full)
                    + 0.15 * float(align_core),
                    0.0,
                    1.0,
                )
            )
            gene_activation_alignment[g] = float(align_full)
            gene_identity_view_alignment[g] = float(align_core)
            gene_activation_contribution[g] = float(contrib)
            activation_contrib_arr[pos] = float(contrib)

        role_order = np.argsort(-activation_contrib_arr)
        top_core_n = min(
            gene_idx.size,
            max(
                int(cfg.high_contribution_gene_min_count),
                int(np.ceil(float(cfg.high_contribution_gene_fraction) * gene_idx.size)),
            ),
        )
        top_core_set = set(int(gene_idx[pos]) for pos in role_order[:top_core_n].tolist())
        scaffold_gene_indices = set(int(g) for g in item.get("template_scaffold_gene_indices", set()))
        scaffold_gene_indices &= set(int(g) for g in gene_idx.tolist())
        support_gene_indices: set[int] = set()
        context_edge_gene_indices: set[int] = set()
        gene_layer_by_gene: dict[int, str] = {}
        role_by_gene: dict[int, str] = {}
        core_weight_map = {int(g): float(w) for g, w in zip(core_gene_idx.tolist(), core_weights.tolist())}
        for pos, g in enumerate(gene_idx.tolist()):
            g = int(g)
            if g in scaffold_gene_indices:
                gene_layer_by_gene[g] = "core_scaffold_gene"
                role_by_gene[g] = "core_scaffold_gene"
            elif gene_activation_contribution[g] >= max(0.12, 3.0 * float(cfg.prune_activation_contribution_max)) or max(
                gene_activation_alignment[g], gene_identity_view_alignment[g]
            ) >= float(cfg.support_gene_min_alignment):
                support_gene_indices.add(g)
                gene_layer_by_gene[g] = "support_gene"
                role_by_gene[g] = "support_gene"
            else:
                context_edge_gene_indices.add(g)
                gene_layer_by_gene[g] = "context_edge_gene"
                role_by_gene[g] = "context_edge_gene"

        if not scaffold_gene_indices:
            fallback_scaffold = [int(gene_idx[pos]) for pos in role_order[:top_core_n].tolist()]
            scaffold_gene_indices = set(fallback_scaffold)
            for g in fallback_scaffold:
                support_gene_indices.discard(g)
                context_edge_gene_indices.discard(g)
                gene_layer_by_gene[g] = "core_scaffold_gene"
                role_by_gene[g] = "core_scaffold_gene"

        ranked_scaffold = [int(gene_idx[pos]) for pos in role_order.tolist() if int(gene_idx[pos]) in scaffold_gene_indices]
        ranked_support = [int(gene_idx[pos]) for pos in role_order.tolist() if int(gene_idx[pos]) in support_gene_indices]
        ranked_context = [int(gene_idx[pos]) for pos in role_order.tolist() if int(gene_idx[pos]) in context_edge_gene_indices]
        ranked_gene_indices = ranked_scaffold + ranked_support + ranked_context
        head_profile = _head_consistency_profile(
            ranked_gene_indices=ranked_gene_indices,
            gene_idx=gene_idx,
            sub_strength_dense=sub_strength_dense,
            full_active_mask=full_active_mask,
            gene_activation_alignment=gene_activation_alignment,
            gene_identity_view_alignment=gene_identity_view_alignment,
            max_head_genes=10,
        )

        item["ranked_gene_indices"] = ranked_gene_indices
        item["weights_identity_view"] = core_weight_map
        item["role_by_gene"] = role_by_gene
        item["gene_layer_by_gene"] = gene_layer_by_gene
        item["gene_role_score"] = {int(gene_idx[pos]): float(activation_contrib_arr[pos]) for pos in range(gene_idx.size)}
        item["gene_attachment_score"] = {int(gene_idx[pos]): float(attachment[pos]) for pos in range(gene_idx.size)}
        item["gene_core_contribution"] = {int(gene_idx[pos]): float(contribution_share[pos]) for pos in range(gene_idx.size)}
        item["gene_activation_alignment"] = gene_activation_alignment
        item["gene_identity_view_alignment"] = gene_identity_view_alignment
        item["gene_activation_contribution"] = gene_activation_contribution
        item["top_contributing_gene_indices"] = set(int(g) for g in ranked_gene_indices[:top_core_n])
        item["template_scaffold_gene_indices"] = set(int(g) for g in scaffold_gene_indices)
        item["template_scaffold_gene_count"] = int(len(scaffold_gene_indices))
        item["support_gene_indices"] = set(int(g) for g in support_gene_indices)
        item["template_support_gene_indices"] = set(int(g) for g in support_gene_indices)
        item["template_support_gene_count"] = int(len(support_gene_indices))
        item["context_edge_gene_indices"] = set(int(g) for g in context_edge_gene_indices)
        item["template_context_edge_gene_indices"] = set(int(g) for g in context_edge_gene_indices)
        item["template_context_edge_gene_count"] = int(len(context_edge_gene_indices))
        item["supporting_gene_indices"] = set(int(g) for g in support_gene_indices)
        item["ambiguous_gene_indices"] = set(int(g) for g in context_edge_gene_indices)
        item["recommended_domain_input"] = "activation_identity_view_weighted"
        item["activation_threshold_full"] = float(full_thr)
        item["activation_threshold_identity_view"] = float(core_thr)
        item["full_active_spot_count"] = int(full_active_spots.size)
        item["identity_view_active_spot_count"] = int(identity_view_active_spots.size)
        item["main_component_frac"] = float(main_component_frac)
        item["core_full_consistency"] = float(core_full_consistency)
        item["activation_peakiness"] = float(peakiness)
        item["activation_entropy"] = float(activation_entropy)
        item["activation_sparsity"] = float(activation_sparsity)
        item["spatial_locality"] = float(spatial_locality)
        item["high_activation_spot_count"] = int(high_activation_spot_count)
        item["activation_contrast_ratio"] = float(activation_contrast_ratio)
        item["activation_dominance"] = float(dominance)
        item["head_consistency_score"] = float(head_profile["head_consistency_score"])
        item["head_mean_pair_corr"] = float(head_profile["head_mean_pair_corr"])
        item["head_low_pair_frac"] = float(head_profile["head_low_pair_frac"])
        item["head_component_count"] = int(head_profile["head_component_count"])
        item["head_largest_component_frac"] = float(head_profile["head_largest_component_frac"])
        item["head_alignment_dispersion"] = float(head_profile["head_alignment_dispersion"])
        item["activation_coverage_full"] = float(full_active_spots.size / max(1, n_spots))
        item["activation_coverage_identity_view"] = float(identity_view_active_spots.size / max(1, n_spots))
        item["activation_full_values"] = full_values.astype(np.float32, copy=False)
        item["activation_identity_view_values"] = identity_view_values.astype(np.float32, copy=False)
        item["high_activation_mask"] = identity_view_mask.astype(bool, copy=False)
        activation_entropy_values.append(float(activation_entropy))
        activation_sparsity_values.append(float(activation_sparsity))

        pid = str(item["program_id"])
        threshold_by_program[pid] = float(core_thr)
        threshold_full_by_program[pid] = float(full_thr)

    rows: list[dict] = []
    nonzero_counts = np.zeros(n_spots, dtype=np.float32)
    dominance = np.zeros(n_spots, dtype=np.float32)
    coverage_per_program = np.zeros(dense_activation_core.shape[1], dtype=np.float32)
    coverage_per_program_full = np.zeros(dense_activation_full.shape[1], dtype=np.float32)

    program_ids = [str(item["program_id"]) for item in program_payload]
    full_thr_vec = np.asarray([threshold_full_by_program.get(pid, 0.0) for pid in program_ids], dtype=np.float32)
    core_thr_vec = np.asarray([threshold_by_program.get(pid, 0.0) for pid in program_ids], dtype=np.float32)

    for i in range(n_spots):
        arr_core = dense_activation_core[i]
        arr_full = dense_activation_full[i]
        keep_core = np.where(arr_core > core_thr_vec)[0]
        keep_full = np.where(arr_full > full_thr_vec)[0]
        keep_union = np.asarray(sorted(set(keep_core.tolist()) | set(keep_full.tolist())), dtype=np.int64)

        nonzero_counts[i] = float(keep_core.size)
        if keep_core.size > 0:
            coverage_per_program[keep_core] += 1.0
        if keep_full.size > 0:
            coverage_per_program_full[keep_full] += 1.0
        if keep_core.size > 0:
            total = float(arr_core[keep_core].sum())
            maxv = float(arr_core[keep_core].max())
            dominance[i] = float(maxv / total) if total > 0 else 0.0
            keep_union = keep_union[np.argsort(-arr_core[keep_union])]

        for rank_in_spot, j in enumerate(keep_union.tolist(), start=1):
            rows.append(
                {
                    "program_id": str(program_ids[j]),
                    "spot_id": str(spot_ids[i]),
                    "activation": float(arr_core[j]),
                    "activation_identity_view": float(arr_core[j]),
                    "activation_full": float(arr_full[j]),
                    "activation_view": "identity_view",
                    "rank_in_spot": int(rank_in_spot),
                }
            )

    activation_df = pd.DataFrame(rows)
    if activation_df.empty:
        activation_df = pd.DataFrame(
            columns=[
                "program_id",
                "spot_id",
                "activation",
                "activation_identity_view",
                "activation_full",
                "activation_view",
                "rank_in_spot",
            ]
        )

    summary = {
        "activation_view_for_domain": "identity_view",
        "recommended_domain_input": "activation_identity_view_weighted",
        "nonzero_programs_per_spot": quantiles(nonzero_counts),
        "top_program_dominance": quantiles(dominance),
        "activation_coverage_by_program": quantiles(
            np.asarray(coverage_per_program / max(1, n_spots), dtype=np.float32)
        ),
        "full_activation_coverage_by_program": quantiles(
            np.asarray(coverage_per_program_full / max(1, n_spots), dtype=np.float32)
        ),
        "effective_activation_threshold_by_program": threshold_by_program,
        "effective_activation_full_threshold_by_program": threshold_full_by_program,
        "high_activation_spot_count_quantiles": quantiles(np.asarray(high_activation_counts, dtype=np.float32)),
        "activation_entropy_quantiles": quantiles(np.asarray(activation_entropy_values, dtype=np.float32)),
        "activation_sparsity_quantiles": quantiles(np.asarray(activation_sparsity_values, dtype=np.float32)),
    }
    return activation_df, dense_activation_core, summary


def refine_programs(
    program_payload: list[dict],
    total_gene_count: int,
    min_program_size_genes: int,
    cfg: ProgramActivationConfig,
) -> tuple[list[dict], pd.DataFrame]:
    rows: list[dict] = []
    refined: list[dict] = []
    total_gene_count = max(1, int(total_gene_count))
    min_size = max(1, int(min_program_size_genes))

    for item in program_payload:
        pid = str(item["program_id"])
        gene_indices = sorted(int(g) for g in item.get("gene_indices", set()))
        top_contributors = set(int(g) for g in item.get("top_contributing_gene_indices", set()))
        drop_genes: set[int] = set()
        for g in gene_indices:
            if g in top_contributors:
                continue
            contribution = float(item.get("gene_activation_contribution", {}).get(g, 0.0))
            align_full = float(item.get("gene_activation_alignment", {}).get(g, 0.0))
            align_core = float(item.get("gene_identity_view_alignment", {}).get(g, 0.0))
            if contribution <= float(cfg.prune_activation_contribution_max) and max(align_full, align_core) <= float(
                cfg.prune_alignment_max
            ):
                drop_genes.add(int(g))

        kept_genes = [g for g in gene_indices if g not in drop_genes]
        status = "unchanged"
        if drop_genes:
            status = "pruned"
        if len(kept_genes) < min_size:
            rows.append(
                {
                    "program_id": pid,
                    "refinement_status": "dropped_after_prune",
                    "pruned_gene_count": int(len(drop_genes)),
                    "program_size_before": int(len(gene_indices)),
                    "program_size_after": int(len(kept_genes)),
                }
            )
            continue

        new_item = dict(item)
        new_item["gene_indices"] = set(int(g) for g in kept_genes)
        kept_arr = np.asarray(kept_genes, dtype=np.int64)
        weights_full = np.asarray([float(item.get("weights_full", {}).get(int(g), 0.0)) for g in kept_arr], dtype=np.float32)
        weights_full = _normalize_weights(weights_full)
        new_item["weights_full"] = {int(g): float(w) for g, w in zip(kept_arr.tolist(), weights_full.tolist())}
        for key in [
            "weights_identity_view",
            "role_by_gene",
            "gene_role_score",
            "gene_attachment_score",
            "gene_core_contribution",
            "gene_activation_alignment",
            "gene_identity_view_alignment",
            "gene_activation_contribution",
            "idf_by_gene",
            "support_spots_by_gene",
            "intra_degree_by_gene",
        ]:
            source = dict(item.get(key, {}))
            new_item[key] = {int(g): source[int(g)] for g in kept_genes if int(g) in source}
        new_item["template_scaffold_gene_indices"] = set(
            int(g) for g in item.get("template_scaffold_gene_indices", set()) if int(g) in new_item["gene_indices"]
        )
        new_item["template_support_gene_indices"] = set(
            int(g)
            for g in item.get("template_support_gene_indices", item.get("support_gene_indices", set()))
            if int(g) in new_item["gene_indices"]
        )
        new_item["template_context_edge_gene_indices"] = set(
            int(g) for g in item.get("template_context_edge_gene_indices", set()) if int(g) in new_item["gene_indices"]
        )
        new_item["template_scaffold_gene_count"] = int(len(new_item["template_scaffold_gene_indices"]))
        new_item["template_support_gene_count"] = int(len(new_item["template_support_gene_indices"]))
        new_item["template_context_edge_gene_count"] = int(len(new_item["template_context_edge_gene_indices"]))
        new_item["support_gene_indices"] = set(
            int(g) for g in item.get("support_gene_indices", set()) if int(g) in new_item["gene_indices"]
        )
        new_item["context_edge_gene_indices"] = set(
            int(g) for g in item.get("context_edge_gene_indices", set()) if int(g) in new_item["gene_indices"]
        )
        new_item["supporting_gene_indices"] = set(
            int(g) for g in item.get("supporting_gene_indices", set()) if int(g) in new_item["gene_indices"]
        )
        new_item["ambiguous_gene_indices"] = set(
            int(g) for g in item.get("ambiguous_gene_indices", set()) if int(g) in new_item["gene_indices"]
        )
        new_item["ranked_gene_indices"] = [int(g) for g in item.get("ranked_gene_indices", []) if int(g) in new_item["gene_indices"]]
        new_item["program_size_genes"] = int(len(kept_genes))
        new_item["program_gene_frac"] = float(len(kept_genes) / total_gene_count)
        new_item["refinement_status"] = status
        new_item["pruned_gene_count"] = int(len(drop_genes))
        new_item["top_contributing_gene_indices"] = set(int(g) for g in top_contributors if int(g) in new_item["gene_indices"])
        refined.append(new_item)
        rows.append(
            {
                "program_id": pid,
                "refinement_status": status,
                "pruned_gene_count": int(len(drop_genes)),
                "program_size_before": int(len(gene_indices)),
                "program_size_after": int(len(kept_genes)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "program_id",
                "refinement_status",
                "pruned_gene_count",
                "program_size_before",
                "program_size_after",
            ]
        )
    return refined, out


def resolve_program_redundancy(
    program_payload: list[dict],
    score_by_pid: dict[str, dict],
    qc_cfg: ProgramQCConfig,
) -> tuple[list[dict], pd.DataFrame]:
    if not program_payload:
        empty = pd.DataFrame(
            columns=[
                "program_id",
                "redundancy_status",
                "primary_program_id",
                "redundancy_family_size",
                "redundancy_family_members",
                "scaffold_match",
                "scaffold_gene_overlap",
                "scaffold_gene_rank_corr",
                "instance_metric_hit_count",
                "activation_identity_view_overlap",
                "high_activation_overlap",
                "activation_full_rank_corr",
                "top_contributing_gene_overlap",
            ]
        )
        return [], empty

    ordered = sorted(
        program_payload,
        key=lambda item: (
            -float(score_by_pid.get(str(item["program_id"]), {}).get("program_confidence", item.get("program_confidence", 0.0))),
            -int(item.get("high_activation_spot_count", 0)),
            str(item["program_id"]),
        ),
    )
    annotated_by_pid: dict[str, dict] = {}
    rows: list[dict] = []
    consumed: set[str] = set()
    min_instance_hits_for_family = max(1, int(qc_cfg.redundancy_min_instance_metric_hits_for_family))
    min_instance_hits_for_duplicate = max(
        min_instance_hits_for_family,
        int(qc_cfg.redundancy_min_instance_metric_hits_for_duplicate),
    )

    for i, primary in enumerate(ordered):
        pid = str(primary["program_id"])
        if pid in consumed:
            continue
        primary["redundancy_status"] = "retained_primary"
        primary["redundant_with"] = ""
        primary["primary_program_id"] = pid
        consumed.add(pid)
        family_members = [pid]
        family_rows: list[dict] = []

        for secondary in ordered[i + 1 :]:
            sid = str(secondary["program_id"])
            if sid in consumed:
                continue
            core_i = np.asarray(primary.get("activation_identity_view_values", []), dtype=np.float32)
            core_j = np.asarray(secondary.get("activation_identity_view_values", []), dtype=np.float32)
            full_i = np.asarray(primary.get("activation_full_values", []), dtype=np.float32)
            full_j = np.asarray(secondary.get("activation_full_values", []), dtype=np.float32)
            scaffold_i = list(primary.get("ranked_gene_indices", []))
            scaffold_i = [int(g) for g in scaffold_i if int(g) in set(primary.get("template_scaffold_gene_indices", set()))]
            scaffold_j = list(secondary.get("ranked_gene_indices", []))
            scaffold_j = [int(g) for g in scaffold_j if int(g) in set(secondary.get("template_scaffold_gene_indices", set()))]
            scaffold_overlap = jaccard(set(scaffold_i), set(scaffold_j))
            scaffold_rank_corr = _rank_corr_by_gene_order(scaffold_i, scaffold_j)
            scaffold_match = bool(
                scaffold_overlap >= float(qc_cfg.redundancy_scaffold_overlap_threshold)
                or scaffold_rank_corr >= float(qc_cfg.redundancy_scaffold_rank_corr_threshold)
            )
            if not scaffold_match:
                continue
            act_overlap = _bool_jaccard(
                core_i > float(primary.get("activation_threshold_identity_view", 0.0)),
                core_j > float(secondary.get("activation_threshold_identity_view", 0.0)),
            )
            high_activation_overlap = jaccard(
                set(np.flatnonzero(np.asarray(primary.get("high_activation_mask", []), dtype=bool)).tolist()),
                set(np.flatnonzero(np.asarray(secondary.get("high_activation_mask", []), dtype=bool)).tolist()),
            )
            full_rank_corr = _safe_rank_correlation(full_i, full_j)
            top_contributing_gene_overlap = jaccard(
                set(int(g) for g in primary.get("top_contributing_gene_indices", set())),
                set(int(g) for g in secondary.get("top_contributing_gene_indices", set())),
            )
            instance_metric_hits = int(
                (act_overlap >= float(qc_cfg.redundancy_activation_overlap_threshold))
                + (high_activation_overlap >= float(qc_cfg.redundancy_high_activation_overlap_threshold))
                + (full_rank_corr >= float(qc_cfg.redundancy_full_rank_corr_threshold))
                + (
                    top_contributing_gene_overlap
                    >= float(qc_cfg.redundancy_top_contributing_gene_overlap_threshold)
                )
            )
            if instance_metric_hits < min_instance_hits_for_family:
                continue

            if instance_metric_hits >= min_instance_hits_for_duplicate:
                status = "merged_duplicate"
            else:
                status = "redundant_variant"
            secondary["redundancy_status"] = status
            secondary["redundant_with"] = pid
            secondary["primary_program_id"] = pid
            consumed.add(sid)
            family_members.append(sid)
            family_rows.append(
                {
                    "program_id": sid,
                    "redundancy_status": status,
                    "primary_program_id": pid,
                    "redundancy_family_size": 0,
                    "redundancy_family_members": "",
                    "scaffold_match": bool(scaffold_match),
                    "scaffold_gene_overlap": float(scaffold_overlap),
                    "scaffold_gene_rank_corr": float(scaffold_rank_corr),
                    "instance_metric_hit_count": int(instance_metric_hits),
                    "activation_identity_view_overlap": float(act_overlap),
                    "high_activation_overlap": float(high_activation_overlap),
                    "activation_full_rank_corr": float(full_rank_corr),
                    "top_contributing_gene_overlap": float(top_contributing_gene_overlap),
                }
            )

        primary["redundancy_family_size"] = int(len(family_members))
        primary["redundancy_family_members"] = list(family_members)
        primary["primary_program_id"] = pid
        annotated_by_pid[pid] = primary
        rows.append(
            {
                "program_id": pid,
                "redundancy_status": "retained_primary",
                "primary_program_id": pid,
                "redundancy_family_size": int(len(family_members)),
                "redundancy_family_members": ",".join(family_members),
                "scaffold_match": True,
                "scaffold_gene_overlap": 0.0,
                "scaffold_gene_rank_corr": 0.0,
                "instance_metric_hit_count": 0,
                "activation_identity_view_overlap": 0.0,
                "high_activation_overlap": 0.0,
                "activation_full_rank_corr": 0.0,
                "top_contributing_gene_overlap": 0.0,
            }
        )
        for row in family_rows:
            row["redundancy_family_size"] = int(len(family_members))
            row["redundancy_family_members"] = ",".join(family_members)
            rows.append(row)
            sid = str(row["program_id"])
            annotated_secondary = next((item for item in ordered if str(item["program_id"]) == sid), None)
            if annotated_secondary is not None:
                annotated_secondary["redundancy_family_size"] = int(len(family_members))
                annotated_secondary["redundancy_family_members"] = list(family_members)
                annotated_by_pid[sid] = annotated_secondary

    annotated_payload: list[dict] = []
    for item in program_payload:
        pid = str(item["program_id"])
        annotated = annotated_by_pid.get(pid, item)
        if "redundancy_status" not in annotated:
            annotated["redundancy_status"] = "retained_primary"
            annotated["redundant_with"] = ""
            annotated["primary_program_id"] = pid
            annotated["redundancy_family_size"] = 1
            annotated["redundancy_family_members"] = [pid]
        annotated_payload.append(annotated)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "program_id",
                "redundancy_status",
                "primary_program_id",
                "redundancy_family_size",
                "redundancy_family_members",
                "scaffold_match",
                "scaffold_gene_overlap",
                "scaffold_gene_rank_corr",
                "instance_metric_hit_count",
                "activation_identity_view_overlap",
                "high_activation_overlap",
                "activation_full_rank_corr",
                "top_contributing_gene_overlap",
            ]
        )
    return annotated_payload, out


def compute_high_contribution_gene_stability(
    final_program_payload: list[dict],
    repeat_program_summaries: list[list[dict]],
    top_ns: tuple[int, ...],
    gene_ids: np.ndarray,
    stable_high_contribution_gene_min_frequency: float,
) -> tuple[dict, list[dict]]:
    top_ns = tuple(sorted({int(n) for n in top_ns if int(n) > 0}))
    if not top_ns:
        top_ns = (20, 50)
    elif 20 not in top_ns:
        top_ns = tuple(sorted(set(top_ns) | {20}))

    gene_ids_arr = np.asarray(gene_ids).astype(str, copy=False)

    def _to_gene_id_set(values) -> set[str]:
        out: set[str] = set()
        for v in values:
            if isinstance(v, (int, np.integer)):
                idx = int(v)
                if 0 <= idx < gene_ids_arr.shape[0]:
                    out.add(str(gene_ids_arr[idx]))
                    continue
            out.add(str(v))
        return out

    def _rank_corr_by_gene_ids(
        final_ranked_gene_ids: list[str],
        cand_ranked_gene_ids: list[str],
        max_rank_n: int,
    ) -> float:
        f = [str(x) for x in final_ranked_gene_ids[: max(1, int(max_rank_n))]]
        c = [str(x) for x in cand_ranked_gene_ids[: max(1, int(max_rank_n))]]
        if not f or not c:
            return 0.0
        f_rank = {g: i + 1 for i, g in enumerate(f)}
        c_rank = {g: i + 1 for i, g in enumerate(c)}
        common = sorted(set(f_rank.keys()) & set(c_rank.keys()))
        if len(common) < 3:
            return 0.0
        x = np.asarray([f_rank[g] for g in common], dtype=np.float32)
        y = np.asarray([c_rank[g] for g in common], dtype=np.float32)
        if float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
            return 0.0
        corr = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return float(max(-1.0, min(1.0, corr)))

    top_ref_n = 20 if 20 in top_ns else int(min(top_ns))
    rank_ref_n = int(max(top_ns))
    freq_min_frac = max(0.0, min(1.0, float(stable_high_contribution_gene_min_frequency)))

    final_sets = []
    for item in final_program_payload:
        ranked = item.get("ranked_gene_indices", [])
        top_contributing_genes = sorted(item.get("top_contributing_gene_indices", set()))
        ranked_gene_ids = [str(gene_ids_arr[int(x)]) for x in ranked if 0 <= int(x) < gene_ids_arr.shape[0]]
        top_contributing_gene_ids = [
            str(gene_ids_arr[int(x)]) for x in top_contributing_genes if 0 <= int(x) < gene_ids_arr.shape[0]
        ]
        top_sets: dict[int, set[str]] = {}
        for n in top_ns:
            if top_contributing_gene_ids:
                top_sets[int(n)] = set(top_contributing_gene_ids[: min(len(top_contributing_gene_ids), n)])
            else:
                top_sets[int(n)] = set(ranked_gene_ids[: min(len(ranked_gene_ids), n)])
        final_sets.append(
            {
                "program_id": item["program_id"],
                "gene_set": _to_gene_id_set(item["gene_indices"]),
                "ranked_gene_ids": ranked_gene_ids,
                "top_sets": top_sets,
            }
        )

    per_program_records: list[dict] = []
    for final in final_sets:
        record = {"program_id": final["program_id"], "by_topN": {}}
        best_cands: list[dict | None] = []
        final_top_ref = set(final["top_sets"].get(top_ref_n, set()))
        for repeat in repeat_program_summaries:
            if not repeat:
                best_cands.append(None)
                continue
            best = None
            best_top_overlap = -1.0
            for cand in repeat:
                cand_top_ref = _to_gene_id_set(cand.get("top_sets", {}).get(top_ref_n, set()))
                ov = jaccard(final_top_ref, cand_top_ref)
                if ov > best_top_overlap:
                    best_top_overlap = ov
                    best = cand
            best_cands.append(best)

        for n in top_ns:
            scores = []
            for best in best_cands:
                if best is None:
                    scores.append(0.0)
                    continue
                final_top = set(final["top_sets"][n])
                cand_top = _to_gene_id_set(best.get("top_sets", {}).get(n, set()))
                scores.append(float(jaccard(final_top, cand_top)))
            record["by_topN"][str(n)] = {
                "quantiles": quantiles(np.asarray(scores, dtype=np.float32)),
                "mean": float(np.mean(scores)) if scores else 0.0,
            }

        rank_corr_scores: list[float] = []
        gene_freq: dict[str, int] = {}
        matched_rounds = 0
        for best in best_cands:
            if best is None:
                rank_corr_scores.append(0.0)
                continue
            cand_ranked_gene_ids = [str(x) for x in best.get("ranked_gene_ids", [])]
            rank_corr_scores.append(
                _rank_corr_by_gene_ids(
                    final_ranked_gene_ids=list(final["ranked_gene_ids"]),
                    cand_ranked_gene_ids=cand_ranked_gene_ids,
                    max_rank_n=rank_ref_n,
                )
            )
            cand_top_ref = _to_gene_id_set(best.get("top_sets", {}).get(top_ref_n, set()))
            if cand_top_ref:
                matched_rounds += 1
                for g in cand_top_ref:
                    gene_freq[g] = gene_freq.get(g, 0) + 1

        if matched_rounds > 0:
            stable_high_contribution_genes = sorted(
                [g for g, c in gene_freq.items() if (c / float(matched_rounds)) >= freq_min_frac]
            )
        else:
            stable_high_contribution_genes = []
        record["rank_corr"] = {
            "quantiles": quantiles(np.asarray(rank_corr_scores, dtype=np.float32)),
            "mean": float(np.mean(rank_corr_scores)) if rank_corr_scores else 0.0,
        }
        record["stable_high_contribution_gene_set_size"] = int(len(stable_high_contribution_genes))
        record["stable_high_contribution_gene_min_frequency"] = float(freq_min_frac)
        record["stable_high_contribution_genes"] = stable_high_contribution_genes
        per_program_records.append(record)

    summary = {
        "matching_mode": "topN_direct_gene_id",
        "top_ref_n": int(top_ref_n),
        "rank_ref_n": int(rank_ref_n),
        "stable_high_contribution_gene_min_frequency": float(freq_min_frac),
        "topN_metrics": {
            str(n): {
                "per_program_p50_quantiles": quantiles(
                    np.asarray([r["by_topN"][str(n)]["quantiles"]["p50"] for r in per_program_records], dtype=np.float32)
                ),
            }
            for n in top_ns
        },
        "records": per_program_records,
    }
    return summary, []


def subset_high_contribution_gene_stability_summary(
    high_contribution_gene_stability_summary: dict,
    keep_program_ids: set[str],
) -> dict:
    records_all = list(high_contribution_gene_stability_summary.get("records", []))
    records = [r for r in records_all if str(r.get("program_id")) in keep_program_ids]
    if not records:
        return {"topN_metrics": {}, "records": []}

    top_ns = sorted({int(k) for r in records for k in r.get("by_topN", {}).keys()})
    topn_metrics = {}
    for n in top_ns:
        n_str = str(n)
        vals = [float(r.get("by_topN", {}).get(n_str, {}).get("quantiles", {}).get("p50", 0.0)) for r in records]
        topn_metrics[n_str] = {"per_program_p50_quantiles": quantiles(np.asarray(vals, dtype=np.float32))}

    return {"topN_metrics": topn_metrics, "records": records}


def decide_observable_programs(
    program_payload: list[dict],
    dense_activation: np.ndarray,
    gene_ids: np.ndarray,
    confounder_flags: dict,
    high_contribution_gene_stability_summary: dict,
    bootstrap_enabled: bool,
    spot_neighbors: list[np.ndarray] | None = None,
    activation_thresholds_by_program: dict[str, float] | None = None,
    qc_cfg: ProgramQCConfig | None = None,
) -> dict:
    qc = qc_cfg or ProgramQCConfig()
    gene_ids_arr = np.asarray(gene_ids).astype(str, copy=False)

    hk_set = {str(x.get("program_id")) for x in confounder_flags.get("housekeeping_like_programs", [])}
    bl_set = {str(x.get("program_id")) for x in confounder_flags.get("blacklist_enriched_programs", [])}

    top20_p50_by_pid: dict[str, float] = {}
    rank_corr_p50_by_pid: dict[str, float] = {}
    stable_gene_set_size_by_pid: dict[str, int] = {}
    for r in list(high_contribution_gene_stability_summary.get("records", [])):
        pid = str(r.get("program_id"))
        top20_p50_by_pid[pid] = float(r.get("by_topN", {}).get("20", {}).get("quantiles", {}).get("p50", 0.0))
        rank_corr_p50_by_pid[pid] = float(r.get("rank_corr", {}).get("quantiles", {}).get("p50", 0.0))
        stable_gene_set_size_by_pid[pid] = int(r.get("stable_high_contribution_gene_set_size", 0))

    kept_ids: list[str] = []
    default_use_ids: list[str] = []
    review_ids: list[str] = []
    rejected: list[dict] = []
    score_rows: list[dict] = []
    reason_counts: dict[str, int] = {}
    default_use_reason_counts: dict[str, int] = {}
    pending_rows: list[dict] = []

    hard_fail_cov = float(max(0.0, qc.hard_fail_min_activation_coverage))
    soft_cov_max = min(1.0, max(0.0, float(qc.soft_max_activation_coverage)))
    hard_fail_top20 = float(max(0.0, qc.hard_fail_min_top20_jaccard_p50))
    hard_fail_rank_corr = float(max(-1.0, min(1.0, qc.hard_fail_min_rank_corr_p50)))
    hard_fail_stable_gene_set_size = int(max(1, qc.hard_fail_min_stable_high_contribution_gene_set_size))
    hard_fail_template_run_support = float(np.clip(qc.hard_fail_min_template_run_support_frac, 0.0, 1.0))
    hard_fail_high_activation_spots = max(1, int(qc.hard_fail_min_high_activation_spots))
    hard_fail_scaffold_quality = float(np.clip(qc.hard_fail_min_scaffold_content_quality, 0.0, 1.0))
    hard_fail_hb_top10 = float(np.clip(qc.hard_fail_max_scaffold_hemoglobin_frac_top10, 0.0, 1.0))
    hard_fail_noise_main = float(np.clip(qc.hard_fail_noise_max_main_component_frac, 0.0, 1.0))
    hard_fail_noise_top3 = float(np.clip(qc.hard_fail_noise_max_top3_component_frac, 0.0, 1.0))
    hard_fail_noise_peakiness = float(np.clip(qc.hard_fail_noise_max_activation_peakiness, 0.0, 1.0))
    hard_fail_noise_consistency = float(np.clip(qc.hard_fail_noise_max_activation_view_consistency, 0.0, 1.0))

    good_run_support = float(np.clip(qc.good_template_run_support_frac, 0.0, 1.0))
    good_spot_support = float(np.clip(qc.good_template_spot_support_frac, 0.0, 1.0))
    good_focus = float(np.clip(qc.good_template_focus_score, 0.0, 1.0))
    good_top20 = max(1e-8, float(qc.good_top20_jaccard_p50))
    good_rank_corr = max(1e-8, float(qc.good_rank_corr_p50))
    good_stable_gene_set_size = max(1, int(qc.good_stable_high_contribution_gene_set_size))
    good_scaffold_quality = float(np.clip(qc.good_scaffold_content_quality, 0.0, 1.0))
    good_scaffold_gene_frac = max(1e-8, float(qc.good_scaffold_gene_frac))
    good_activation_coverage = max(1e-8, float(qc.good_activation_coverage))
    good_high_activation_spots = max(1, int(qc.good_high_activation_spots))
    good_peakiness = float(np.clip(qc.good_activation_peakiness, 0.0, 1.0))
    good_consistency = float(np.clip(qc.good_activation_view_consistency, 0.0, 1.0))
    min_main_component_frac = float(np.clip(qc.min_main_component_frac, 0.0, 1.0))
    single_main_component_frac = float(np.clip(qc.single_main_block_min_main_component_frac, 0.0, 1.0))
    good_top2_component_frac = float(np.clip(qc.good_top2_component_frac, 0.0, 1.0))
    good_top3_component_frac = float(np.clip(qc.good_top3_component_frac, 0.0, 1.0))
    threshold_map = activation_thresholds_by_program or {}
    default_use_min_program_confidence = float(np.clip(qc.default_use_min_program_confidence, 0.0, 1.0))
    default_use_min_support_score = float(np.clip(qc.default_use_min_support_score, 0.0, 1.0))
    default_use_min_validity = float(np.clip(qc.default_use_min_validity_score, 0.0, 1.0))
    default_use_min_activation_presence = float(np.clip(qc.default_use_min_activation_presence_score, 0.0, 1.0))
    default_use_min_structure = float(np.clip(qc.default_use_min_structure_score, 0.0, 1.0))
    default_use_min_scaffold_quality = float(np.clip(qc.default_use_min_scaffold_content_quality, 0.0, 1.0))
    high_thr = float(np.clip(qc.high_program_confidence_threshold, 0.0, 1.0))
    dense_activation = np.asarray(dense_activation, dtype=np.float32)

    for j, item in enumerate(program_payload):
        pid = str(item["program_id"])
        hard_fail_reasons: list[str] = []
        act_thr = float(threshold_map.get(pid, 0.0))
        values = (
            dense_activation[:, j]
            if dense_activation.ndim == 2 and dense_activation.shape[1] > j
            else np.zeros((dense_activation.shape[0],), dtype=np.float32)
        )
        active_mask = values > act_thr
        coverage = float(np.mean(active_mask)) if active_mask.size > 0 else 0.0
        component_summary = _component_concentration_summary(active_mask, spot_neighbors)
        largest_active_component_size = int(component_summary["largest_active_component_size"])
        largest_active_component_frac = float(component_summary["largest_active_component_frac"])
        top2_component_frac = float(component_summary["top2_component_frac"])
        top3_component_frac = float(component_summary["top3_component_frac"])
        component_count = int(component_summary["component_count"])
        component_active_spot_count = int(np.count_nonzero(active_mask))

        top20_p50 = float(top20_p50_by_pid.get(pid, 0.0))
        rank_corr_p50 = float(rank_corr_p50_by_pid.get(pid, 0.0))
        core_size = int(stable_gene_set_size_by_pid.get(pid, 0))

        core_full_consistency = float(item.get("core_full_consistency", 0.0))
        activation_peakiness = float(item.get("activation_peakiness", 0.0))
        activation_entropy = float(item.get("activation_entropy", 0.0))
        activation_sparsity = float(item.get("activation_sparsity", 0.0))
        main_component_frac = float(item.get("main_component_frac", largest_active_component_frac))
        high_activation_spot_count = int(item.get("high_activation_spot_count", largest_active_component_size))
        spatial_locality = float(item.get("spatial_locality", largest_active_component_frac))
        activation_contrast_ratio = float(item.get("activation_contrast_ratio", 0.0))
        activation_dominance = float(item.get("activation_dominance", 0.0))
        ranked_gene_indices = [int(g) for g in item.get("ranked_gene_indices", [])]
        scaffold_gene_set = set(int(g) for g in item.get("template_scaffold_gene_indices", set()))
        scaffold_gene_indices = [int(g) for g in ranked_gene_indices if int(g) in scaffold_gene_set]
        if not scaffold_gene_indices:
            top_contributing_set = set(int(g) for g in item.get("top_contributing_gene_indices", set()))
            scaffold_gene_indices = [int(g) for g in ranked_gene_indices if int(g) in top_contributing_set]
        scaffold_gene_names = [
            str(gene_ids_arr[g]) for g in scaffold_gene_indices if 0 <= int(g) < gene_ids_arr.shape[0]
        ]
        template_support_gene_set = set(
            int(g) for g in item.get("template_support_gene_indices", item.get("support_gene_indices", set()))
        )
        template_context_edge_gene_set = set(
            int(g) for g in item.get("template_context_edge_gene_indices", item.get("context_edge_gene_indices", set()))
        )
        scaffold_content = _scaffold_content_quality(scaffold_gene_names)
        scaffold_content_quality_score = float(scaffold_content["scaffold_content_quality_score"])
        scaffold_weird_symbol_frac_top20 = float(scaffold_content["scaffold_weird_symbol_frac_top20"])
        scaffold_weird_symbol_frac_top5 = float(scaffold_content["scaffold_weird_symbol_frac_top5"])
        scaffold_hemoglobin_frac_top10 = float(scaffold_content["scaffold_hemoglobin_frac_top10"])
        program_gene_count = max(1, int(len(item.get("gene_indices", []))))
        scaffold_gene_frac = float(len(scaffold_gene_indices) / program_gene_count)
        support_gene_count = int(item.get("template_support_gene_count", len(item.get("template_support_gene_indices", []))))
        context_edge_gene_count = int(
            item.get("template_context_edge_gene_count", len(item.get("template_context_edge_gene_indices", [])))
        )
        support_gene_frac = float(support_gene_count / program_gene_count)
        context_edge_gene_frac = float(context_edge_gene_count / program_gene_count)
        scaffold_vs_context_ratio = float(len(scaffold_gene_indices) / max(1.0, len(scaffold_gene_indices) + context_edge_gene_count))
        scaffold_vs_support_ratio = float(len(scaffold_gene_indices) / max(1.0, len(scaffold_gene_indices) + support_gene_count))
        context_purity_score = float(np.clip(1.0 - context_edge_gene_frac, 0.0, 1.0))
        support_compactness_score = float(np.clip(scaffold_vs_support_ratio, 0.0, 1.0))
        axis_profile = _template_axis_coherence(
            ranked_gene_indices=ranked_gene_indices,
            weights_full_map={int(k): float(v) for k, v in dict(item.get("weights_full", {})).items()},
            scaffold_gene_set=scaffold_gene_set,
            support_gene_set=template_support_gene_set,
            context_edge_gene_set=template_context_edge_gene_set,
            top_contributing_gene_set=set(int(g) for g in item.get("top_contributing_gene_indices", set())),
            gene_activation_alignment={int(k): float(v) for k, v in dict(item.get("gene_activation_alignment", {})).items()},
            gene_identity_view_alignment={int(k): float(v) for k, v in dict(item.get("gene_identity_view_alignment", {})).items()},
            top_n=12,
        )
        axis_coherence_score = float(axis_profile["axis_coherence_score"])
        axis_head_mass_share = float(axis_profile["axis_head_mass_share"])
        axis_scaffold_mass_share = float(axis_profile["axis_scaffold_mass_share"])
        axis_support_mass_share = float(axis_profile["axis_support_mass_share"])
        axis_context_mass_share = float(axis_profile["axis_context_mass_share"])
        axis_non_axis_dilution = float(axis_profile["axis_non_axis_dilution"])
        axis_top_contributor_mass_share = float(axis_profile["axis_top_contributor_mass_share"])
        axis_alignment_mean = float(axis_profile["axis_alignment_mean"])
        head_consistency_score = float(item.get("head_consistency_score", 0.0))
        head_mean_pair_corr = float(item.get("head_mean_pair_corr", 0.0))
        head_low_pair_frac = float(item.get("head_low_pair_frac", 1.0))
        head_component_count = int(item.get("head_component_count", 0))
        head_largest_component_frac = float(item.get("head_largest_component_frac", 0.0))
        head_alignment_dispersion = float(item.get("head_alignment_dispersion", 1.0))

        if bool(qc.require_rerun_for_validity) and not bootstrap_enabled:
            hard_fail_reasons.append("rerun_stability_disabled")
        if bool(qc.drop_housekeeping_or_blacklist) and pid in hk_set:
            hard_fail_reasons.append("housekeeping_like")
        if bool(qc.drop_housekeeping_or_blacklist) and pid in bl_set:
            hard_fail_reasons.append("blacklist_enriched")

        template_run_support_frac = float(item.get("template_run_support_frac", 0.0))
        template_spot_support_frac = float(item.get("template_spot_support_frac", 0.0))
        template_focus_score = float(item.get("template_focus_score", 0.0))
        if template_run_support_frac < hard_fail_template_run_support:
            hard_fail_reasons.append(f"template_run_support<{hard_fail_template_run_support}")
        if coverage < hard_fail_cov:
            hard_fail_reasons.append(f"activation_coverage<{hard_fail_cov}")
        if high_activation_spot_count < hard_fail_high_activation_spots:
            hard_fail_reasons.append(f"high_activation_spots<{hard_fail_high_activation_spots}")
        if scaffold_content_quality_score < hard_fail_scaffold_quality:
            hard_fail_reasons.append(f"scaffold_content_quality<{hard_fail_scaffold_quality}")
        if scaffold_hemoglobin_frac_top10 > hard_fail_hb_top10:
            hard_fail_reasons.append(f"scaffold_hemoglobin_frac_top10>{hard_fail_hb_top10}")
        if bootstrap_enabled and (
            top20_p50 < hard_fail_top20
            and rank_corr_p50 < hard_fail_rank_corr
            and core_size < hard_fail_stable_gene_set_size
        ):
            hard_fail_reasons.append("validity_discovery_collapsed")

        run_support_score = _smooth_lower_bound_score(template_run_support_frac, good_run_support)
        spot_support_score = _smooth_lower_bound_score(template_spot_support_frac, good_spot_support)
        focus_score = _smooth_lower_bound_score(template_focus_score, good_focus)
        top20_score = _smooth_lower_bound_score(top20_p50, good_top20)
        rank_corr_score = _smooth_lower_bound_score(max(rank_corr_p50, 0.0), good_rank_corr)
        core_size_score = _smooth_lower_bound_score(float(core_size), float(good_stable_gene_set_size))
        scaffold_gene_frac_score = _smooth_lower_bound_score(scaffold_gene_frac, good_scaffold_gene_frac)
        coverage_score = _smooth_lower_bound_score(coverage, good_activation_coverage)
        if coverage > soft_cov_max:
            coverage_score *= max(0.0, 1.0 - (coverage - soft_cov_max) / max(1e-8, 1.0 - soft_cov_max))
        seedability_score = _smooth_lower_bound_score(float(high_activation_spot_count), float(good_high_activation_spots))
        top2_component_score = _smooth_lower_bound_score(float(top2_component_frac), good_top2_component_frac)
        top3_component_score = _smooth_lower_bound_score(float(top3_component_frac), good_top3_component_frac)

        if main_component_frac >= single_main_component_frac:
            activation_shape_class = "single_main_block"
        elif (
            top2_component_frac >= good_top2_component_frac
            and activation_peakiness >= max(0.75, good_peakiness * 0.85)
            and core_full_consistency >= max(0.35, good_consistency * 0.75)
        ):
            activation_shape_class = "multi_focal_supported"
        elif (
            top3_component_frac >= good_top3_component_frac
            and activation_peakiness >= max(0.80, good_peakiness * 0.90)
            and core_full_consistency >= max(0.30, good_consistency * 0.65)
        ):
            activation_shape_class = "distributed_supported"
        elif (
            main_component_frac <= hard_fail_noise_main
            and top3_component_frac <= hard_fail_noise_top3
            and activation_peakiness <= hard_fail_noise_peakiness
            and core_full_consistency <= hard_fail_noise_consistency
        ):
            activation_shape_class = "fragmented_noise_like"
        else:
            activation_shape_class = "fragmented_borderline"

        structure_nonrandom_score = float(
            np.clip(
                0.25 * top2_component_score
                + 0.25 * top3_component_score
                + 0.20 * activation_peakiness
                + 0.15 * core_full_consistency
                + 0.15 * spatial_locality,
                0.0,
                1.0,
            )
        )
        shape_risk = float(
            np.clip(
                1.0
                - structure_nonrandom_score
                + (0.15 if activation_shape_class == "fragmented_borderline" else 0.35 if activation_shape_class == "fragmented_noise_like" else 0.0),
                0.0,
                1.0,
            )
        )
        if activation_shape_class == "fragmented_noise_like":
            hard_fail_reasons.append(f"activation_shape={activation_shape_class}")

        validity_discovery_support_score = float(
            np.clip(
                0.35 * run_support_score
                + 0.25 * top20_score
                + 0.25 * rank_corr_score
                + 0.15 * core_size_score,
                0.0,
                1.0,
            )
        )
        validity_content_integrity_score = float(
            np.clip(
                0.70 * scaffold_content_quality_score
                + 0.15 * max(0.0, 1.0 - scaffold_hemoglobin_frac_top10)
                + 0.15 * max(0.0, 1.0 - scaffold_weird_symbol_frac_top5),
                0.0,
                1.0,
            )
        )
        validity_activation_presence_score = float(
            np.clip(
                0.30 * coverage_score
                + 0.30 * seedability_score
                + 0.20 * activation_contrast_ratio
                + 0.10 * activation_dominance
                + 0.10 * max(0.0, 1.0 - activation_entropy),
                0.0,
                1.0,
            )
        )
        validity_score = float(
            np.clip(
                0.40 * validity_discovery_support_score
                + 0.25 * validity_activation_presence_score
                + 0.20 * validity_content_integrity_score
                + 0.15 * structure_nonrandom_score,
                0.0,
                1.0,
            )
        )
        axis_center_score = float(
            np.clip(
                0.30 * axis_coherence_score
                + 0.25 * focus_score
                + 0.15 * axis_head_mass_share
                + 0.15 * scaffold_vs_context_ratio
                + 0.15 * support_compactness_score,
                0.0,
                1.0,
            )
        )
        scaffold_definition_score = float(
            np.clip(
                0.25 * scaffold_content_quality_score
                + 0.20 * scaffold_gene_frac_score
                + 0.15 * top20_score
                + 0.15 * rank_corr_score
                + 0.15 * axis_top_contributor_mass_share
                + 0.10 * support_compactness_score,
                0.0,
                1.0,
            )
        )
        activation_clarity_score = float(
            np.clip(
                0.20 * activation_contrast_ratio
                + 0.15 * activation_dominance
                + 0.20 * activation_peakiness
                + 0.20 * core_full_consistency
                + 0.10 * seedability_score
                + 0.15 * max(0.0, 1.0 - activation_entropy),
                0.0,
                1.0,
            )
        )
        structure_quality_score = float(
            np.clip(
                0.40 * axis_center_score
                + 0.30 * scaffold_definition_score
                + 0.20 * activation_clarity_score
                + 0.10 * structure_nonrandom_score,
                0.0,
                1.0,
            )
        )
        mixed_template_support_score = float(
            np.clip(
                0.45 * axis_center_score
                + 0.30 * validity_activation_presence_score
                + 0.25 * structure_nonrandom_score,
                0.0,
                1.0,
            )
        )
        program_conf_raw = float(
            np.clip(
                0.55 * validity_score
                + 0.45 * structure_quality_score,
                0.0,
                1.0,
            )
        )
        default_use_support_score = float(
            np.clip(
                0.40 * validity_score
                + 0.20 * validity_activation_presence_score
                + 0.20 * structure_nonrandom_score
                + 0.20 * scaffold_content_quality_score,
                0.0,
                1.0,
            )
        )
        shape_label_bonus = {
            "single_main_block": 1.0,
            "multi_focal_supported": 0.90,
            "distributed_supported": 0.85,
            "fragmented_borderline": 0.65,
            "fragmented_noise_like": 0.15,
        }.get(activation_shape_class, 0.50)

        score_row = {
            "program_id": str(pid),
            "program_confidence_raw": float(program_conf_raw),
            "template_evidence_score": float(validity_score),
            "structure_confidence": float(validity_score),
            "structure_confidence_baseline": float(validity_discovery_support_score),
            "stability_score": float(validity_discovery_support_score),
            "activation_score": float(validity_activation_presence_score),
            "activation_morphology_score": float(structure_nonrandom_score),
            "existence_score": float(validity_score),
            "template_existence_evidence": float(validity_discovery_support_score),
            "scaffold_quality_evidence": float(scaffold_definition_score),
            "activation_evidence": float(validity_activation_presence_score),
            "focus_evidence": float(axis_center_score),
            "morphology_evidence": float(structure_nonrandom_score),
            "validity_score": float(validity_score),
            "validity_discovery_support_score": float(validity_discovery_support_score),
            "validity_activation_presence_score": float(validity_activation_presence_score),
            "validity_content_integrity_score": float(validity_content_integrity_score),
            "validity_structure_score": float(structure_nonrandom_score),
            "structure_quality_score": float(structure_quality_score),
            "axis_center_score": float(axis_center_score),
            "scaffold_definition_score": float(scaffold_definition_score),
            "activation_clarity_score": float(activation_clarity_score),
            "mixed_template_support_score": float(mixed_template_support_score),
            "default_use_support_score": float(default_use_support_score),
            "activation_coverage": float(coverage),
            "activation_threshold": float(act_thr),
            "largest_active_component_size": int(largest_active_component_size),
            "largest_active_component_frac": float(largest_active_component_frac),
            "component_count": int(component_count),
            "top2_component_frac": float(top2_component_frac),
            "top3_component_frac": float(top3_component_frac),
            "component_active_spot_count": int(component_active_spot_count),
            "core_full_consistency": float(core_full_consistency),
            "activation_peakiness": float(activation_peakiness),
            "activation_entropy": float(activation_entropy),
            "activation_sparsity": float(activation_sparsity),
            "main_component_frac": float(main_component_frac),
            "high_activation_spot_count": int(high_activation_spot_count),
            "spatial_locality": float(spatial_locality),
            "activation_contrast_ratio": float(activation_contrast_ratio),
            "activation_dominance": float(activation_dominance),
            "activation_shape_class": str(activation_shape_class),
            "shape_evidence_score": float(structure_nonrandom_score),
            "shape_risk": float(shape_risk),
            "scaffold_content_quality_score": float(scaffold_content_quality_score),
            "scaffold_weird_symbol_frac_top20": float(scaffold_weird_symbol_frac_top20),
            "scaffold_weird_symbol_frac_top5": float(scaffold_weird_symbol_frac_top5),
            "scaffold_hemoglobin_frac_top10": float(scaffold_hemoglobin_frac_top10),
            "scaffold_gene_frac": float(scaffold_gene_frac),
            "support_gene_frac": float(support_gene_frac),
            "context_edge_gene_frac": float(context_edge_gene_frac),
            "scaffold_vs_context_ratio": float(scaffold_vs_context_ratio),
            "scaffold_vs_support_ratio": float(scaffold_vs_support_ratio),
            "context_purity_score": float(context_purity_score),
            "support_compactness_score": float(support_compactness_score),
            "axis_coherence_score": float(axis_coherence_score),
            "axis_head_mass_share": float(axis_head_mass_share),
            "axis_scaffold_mass_share": float(axis_scaffold_mass_share),
            "axis_support_mass_share": float(axis_support_mass_share),
            "axis_context_mass_share": float(axis_context_mass_share),
            "axis_non_axis_dilution": float(axis_non_axis_dilution),
            "axis_top_contributor_mass_share": float(axis_top_contributor_mass_share),
            "axis_alignment_mean": float(axis_alignment_mean),
            "head_consistency_score": float(head_consistency_score),
            "head_mean_pair_corr": float(head_mean_pair_corr),
            "head_low_pair_frac": float(head_low_pair_frac),
            "head_component_count": int(head_component_count),
            "head_largest_component_frac": float(head_largest_component_frac),
            "head_alignment_dispersion": float(head_alignment_dispersion),
            "top20_jaccard_p50": float(top20_p50),
            "rank_corr_p50": float(rank_corr_p50),
            "stable_high_contribution_gene_set_size": int(core_size),
            "top20_score": float(top20_score),
            "rank_corr_score": float(rank_corr_score),
            "core_size_score": float(core_size_score),
            "stability_core_score": float(validity_discovery_support_score),
            "activation_coverage_score": float(coverage_score),
            "seedability_score": float(seedability_score),
            "main_component_floor_pass": bool(main_component_frac >= min_main_component_frac),
            "multi_focal_support_score": float(top2_component_score),
            "fragmented_rescue_score": float(top3_component_score),
            "component_shape_score": float(structure_nonrandom_score),
            "shape_label_bonus": float(shape_label_bonus),
            "template_run_support_frac": float(template_run_support_frac),
            "template_spot_support_frac": float(template_spot_support_frac),
            "template_focus_score": float(template_focus_score),
            "hard_gate_failed": bool(len(hard_fail_reasons) > 0),
            "hard_fail_reason_count": int(len(hard_fail_reasons)),
        }
        pending_rows.append(
            {
                "program_id": str(pid),
                "item": item,
                "score_row": score_row,
                "hard_fail_reasons": list(hard_fail_reasons),
                "activation_coverage": float(coverage),
                "axis_center_score": float(axis_center_score),
                "validity_activation_presence_score": float(validity_activation_presence_score),
                "structure_nonrandom_score": float(structure_nonrandom_score),
            }
        )

    conf_raw_vals = np.asarray([float(x["score_row"]["program_confidence_raw"]) for x in pending_rows], dtype=np.float32)
    conf_vals, conf_rank_pct = _stretch_program_confidence(
        conf_raw_vals,
        low_anchor=0.35,
        high_anchor=min(0.95, high_thr + 0.10),
        raw_weight=0.60,
    )

    for idx, pending in enumerate(pending_rows):
        pid = str(pending["program_id"])
        item = pending["item"]
        score_row = dict(pending["score_row"])
        hard_fail_reasons = list(pending["hard_fail_reasons"])
        program_conf = float(conf_vals[idx]) if idx < conf_vals.shape[0] else float(score_row["program_confidence_raw"])
        rank_pct = float(conf_rank_pct[idx]) if idx < conf_rank_pct.shape[0] else 0.5
        if hard_fail_reasons:
            validity_status = "invalid"
            routing_status = "rejected"
            reasons = list(hard_fail_reasons)
            default_use_reasons: list[str] = []
        else:
            validity_status = "valid"
            reasons = []
            default_use_reasons = []
            if program_conf < default_use_min_program_confidence:
                default_use_reasons.append(f"program_confidence<{default_use_min_program_confidence:.2f}")
            if float(score_row["default_use_support_score"]) < default_use_min_support_score:
                default_use_reasons.append(f"default_use_support<{default_use_min_support_score:.2f}")
            if float(score_row["validity_score"]) < default_use_min_validity:
                default_use_reasons.append(f"validity_score<{default_use_min_validity:.2f}")
            if float(score_row["validity_activation_presence_score"]) < default_use_min_activation_presence:
                default_use_reasons.append(f"activation_presence<{default_use_min_activation_presence:.2f}")
            if float(score_row["validity_structure_score"]) < default_use_min_structure:
                default_use_reasons.append(f"structure_score<{default_use_min_structure:.2f}")
            if float(score_row["scaffold_content_quality_score"]) < default_use_min_scaffold_quality:
                default_use_reasons.append(f"scaffold_content_quality<{default_use_min_scaffold_quality:.2f}")

            routing_status = "review_only" if default_use_reasons else "default_use"

        score_row["program_confidence"] = float(program_conf)
        score_row["program_confidence_rank_pct"] = float(rank_pct)
        score_row["validity_status"] = str(validity_status)
        score_row["routing_status"] = str(routing_status)
        score_row["default_use_reason_count"] = int(len(default_use_reasons))
        score_row["default_use_reasons"] = list(default_use_reasons)
        score_rows.append(score_row)

        item["program_confidence_raw"] = float(score_row["program_confidence_raw"])
        item["program_confidence"] = float(program_conf)
        item["program_confidence_rank_pct"] = float(rank_pct)
        item["validity_status"] = str(validity_status)
        item["routing_status"] = str(routing_status)
        item["default_use_support_score"] = float(score_row["default_use_support_score"])
        item["default_use_reason_count"] = int(len(default_use_reasons))
        item["default_use_reasons"] = list(default_use_reasons)
        item["existence_score"] = float(score_row["existence_score"])
        item["activation_morphology_score"] = float(score_row["activation_morphology_score"])
        item["activation_shape_class"] = str(score_row["activation_shape_class"])
        item["multi_focal_support_score"] = float(score_row["multi_focal_support_score"])
        item["fragmented_rescue_score"] = float(score_row["fragmented_rescue_score"])
        item["component_shape_score"] = float(score_row["component_shape_score"])
        item["shape_risk"] = float(score_row["shape_risk"])
        item["template_existence_evidence"] = float(score_row["template_existence_evidence"])
        item["scaffold_quality_evidence"] = float(score_row["scaffold_quality_evidence"])
        item["activation_evidence"] = float(score_row["activation_evidence"])
        item["focus_evidence"] = float(score_row["focus_evidence"])
        item["morphology_evidence"] = float(score_row["morphology_evidence"])
        item["scaffold_content_quality_score"] = float(score_row["scaffold_content_quality_score"])
        item["scaffold_weird_symbol_frac_top20"] = float(score_row["scaffold_weird_symbol_frac_top20"])
        item["scaffold_weird_symbol_frac_top5"] = float(score_row["scaffold_weird_symbol_frac_top5"])
        item["scaffold_hemoglobin_frac_top10"] = float(score_row["scaffold_hemoglobin_frac_top10"])
        item["head_consistency_score"] = float(score_row["head_consistency_score"])
        item["head_mean_pair_corr"] = float(score_row["head_mean_pair_corr"])
        item["head_low_pair_frac"] = float(score_row["head_low_pair_frac"])
        item["head_component_count"] = int(score_row["head_component_count"])
        item["head_largest_component_frac"] = float(score_row["head_largest_component_frac"])
        item["head_alignment_dispersion"] = float(score_row["head_alignment_dispersion"])

        if routing_status == "rejected":
            rejected.append(
                {
                    "program_id": str(pid),
                    "reasons": reasons,
                    "program_confidence": float(program_conf),
                    "validity_status": str(validity_status),
                    "routing_status": str(routing_status),
                }
            )
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        else:
            kept_ids.append(str(pid))
            if routing_status == "default_use":
                default_use_ids.append(str(pid))
            else:
                review_ids.append(str(pid))
                for r in default_use_reasons:
                    default_use_reason_counts[r] = int(default_use_reason_counts.get(r, 0) + 1)

    conf_vals = np.asarray([float(r["program_confidence"]) for r in score_rows], dtype=np.float32)
    validity_counts: dict[str, int] = {}
    routing_counts: dict[str, int] = {}
    for r in score_rows:
        validity_status = str(r.get("validity_status", "invalid"))
        routing_status = str(r.get("routing_status", "rejected"))
        validity_counts[validity_status] = int(validity_counts.get(validity_status, 0) + 1)
        routing_counts[routing_status] = int(routing_counts.get(routing_status, 0) + 1)

    return {
        "candidate_program_count": int(len(program_payload)),
        "kept_program_count": int(len(kept_ids)),
        "default_use_program_count": int(len(default_use_ids)),
        "review_program_count": int(len(review_ids)),
        "rejected_program_count": int(len(rejected)),
        "valid_program_count": int(validity_counts.get("valid", 0)),
        "rerun_stability_enabled": bool(bootstrap_enabled),
        "require_rerun_for_validity": bool(qc.require_rerun_for_validity),
        "hard_fail_min_activation_coverage": float(hard_fail_cov),
        "soft_max_activation_coverage": float(soft_cov_max),
        "hard_fail_min_template_run_support_frac": float(hard_fail_template_run_support),
        "hard_fail_min_high_activation_spots": int(hard_fail_high_activation_spots),
        "hard_fail_min_scaffold_content_quality": float(hard_fail_scaffold_quality),
        "hard_fail_max_scaffold_hemoglobin_frac_top10": float(hard_fail_hb_top10),
        "hard_fail_min_top20_jaccard_p50": float(hard_fail_top20),
        "hard_fail_min_rank_corr_p50": float(hard_fail_rank_corr),
        "hard_fail_min_stable_high_contribution_gene_set_size": int(hard_fail_stable_gene_set_size),
        "hard_fail_noise_max_main_component_frac": float(hard_fail_noise_main),
        "hard_fail_noise_max_top3_component_frac": float(hard_fail_noise_top3),
        "hard_fail_noise_max_activation_peakiness": float(hard_fail_noise_peakiness),
        "hard_fail_noise_max_activation_view_consistency": float(hard_fail_noise_consistency),
        "good_template_run_support_frac": float(good_run_support),
        "good_template_spot_support_frac": float(good_spot_support),
        "good_template_focus_score": float(good_focus),
        "good_top20_jaccard_p50": float(good_top20),
        "good_rank_corr_p50": float(good_rank_corr),
        "good_stable_high_contribution_gene_set_size": int(good_stable_gene_set_size),
        "good_scaffold_content_quality": float(good_scaffold_quality),
        "good_scaffold_gene_frac": float(good_scaffold_gene_frac),
        "good_activation_coverage": float(good_activation_coverage),
        "good_high_activation_spots": int(good_high_activation_spots),
        "good_activation_peakiness": float(good_peakiness),
        "good_activation_view_consistency": float(good_consistency),
        "min_main_component_frac": float(min_main_component_frac),
        "single_main_block_min_main_component_frac": float(single_main_component_frac),
        "good_top2_component_frac": float(good_top2_component_frac),
        "good_top3_component_frac": float(good_top3_component_frac),
        "default_use_min_program_confidence": float(default_use_min_program_confidence),
        "default_use_min_support_score": float(default_use_min_support_score),
        "default_use_min_validity_score": float(default_use_min_validity),
        "default_use_min_activation_presence_score": float(default_use_min_activation_presence),
        "default_use_min_structure_score": float(default_use_min_structure),
        "default_use_min_scaffold_content_quality": float(default_use_min_scaffold_quality),
        "high_program_confidence_threshold": float(high_thr),
        "drop_housekeeping_or_blacklist": bool(qc.drop_housekeeping_or_blacklist),
        "gating_mode": "validity_default_use_filter",
        "program_confidence_quantiles": quantiles(conf_vals),
        "program_confidence_components": {
            "validity_weight": 0.55,
            "structure_quality_weight": 0.45,
            "rank_stretch_raw_weight": 0.60,
            "rank_stretch_rank_weight": 0.40,
        },
        "validity_counts": validity_counts,
        "routing_counts": routing_counts,
        "default_use_reason_counts": default_use_reason_counts,
        "program_scores": score_rows,
        "kept_program_ids": kept_ids,
        "default_use_program_ids": default_use_ids,
        "review_program_ids": review_ids,
        "rejected_programs": rejected,
        "rejection_reason_counts": reason_counts,
    }


def build_confounder_flags(
    program_payload: list[dict],
    dense_activation: np.ndarray,
    support_frac: np.ndarray,
    blacklist_mask: np.ndarray,
    qc_cfg: ProgramQCConfig,
    activation_thresholds_by_program: dict[str, float] | None = None,
) -> tuple[dict, pd.DataFrame]:
    housekeeping_like: list[dict] = []
    blacklist_enriched: list[dict] = []
    rows = []
    threshold_map = activation_thresholds_by_program or {}
    for j, item in enumerate(program_payload):
        genes = sorted(item["gene_indices"])
        if not genes:
            continue
        pid = str(item["program_id"])
        act_thr = float(threshold_map.get(pid, 0.0))
        coverage = float(np.mean(dense_activation[:, j] > act_thr))
        mean_support = float(np.mean(support_frac[genes]))
        blacklist_frac = float(np.mean(blacklist_mask[genes]))

        rows.append(
            {
                "program_id": pid,
                "activation_coverage": coverage,
                "activation_threshold": act_thr,
                "mean_gene_support_frac": mean_support,
                "blacklist_gene_frac": blacklist_frac,
                "n_genes": int(len(genes)),
            }
        )

        if (
            coverage >= float(qc_cfg.housekeeping_activation_coverage_threshold)
            and mean_support >= float(qc_cfg.housekeeping_mean_gene_support_frac_threshold)
        ):
            housekeeping_like.append(
                {
                    "program_id": pid,
                    "activation_coverage": coverage,
                    "activation_threshold": act_thr,
                    "mean_gene_support_frac": mean_support,
                }
            )
        if blacklist_frac >= float(qc_cfg.blacklist_enrichment_threshold):
            blacklist_enriched.append(
                {
                    "program_id": pid,
                    "blacklist_gene_frac": blacklist_frac,
                }
            )

    confounders = {
        "housekeeping_like_programs": housekeeping_like,
        "blacklist_enriched_programs": blacklist_enriched,
    }
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "program_id",
                "activation_coverage",
                "activation_threshold",
                "mean_gene_support_frac",
                "blacklist_gene_frac",
                "n_genes",
            ]
        )
    return confounders, table
