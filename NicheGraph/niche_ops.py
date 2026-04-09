from __future__ import annotations

from collections import defaultdict
import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .common import quantiles
from .schema import (
    BasicNicheFilterConfig,
    DomainEdgeConfig,
    DomainReliabilityConfig,
    InteractionDedupConfig,
    InteractionDiscoveryConfig,
    NichePipelineConfig,
    RandomBaselineConfig,
)


def _empty_edges_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "domain_key_i",
            "domain_key_j",
            "domain_id_i",
            "domain_id_j",
            "program_id_i",
            "program_id_j",
            "spot_count_i",
            "spot_count_j",
            "shared_boundary_edges",
            "spatial_overlap",
            "boundary_distance",
            "proximity_distance",
            "boundary_size_i",
            "boundary_size_j",
            "domain_reliability_i",
            "domain_reliability_j",
            "edge_reliability",
            "edge_reliability_factor",
            "edge_weight_raw",
            "boundary_contact",
            "overlap_contact",
            "proximity_contact",
            "relation_types",
            "contact_ratio",
            "contact_strength",
            "contact_strength_eff",
            "overlap_strength",
            "overlap_strength_eff",
            "soft_strength",
            "soft_strength_eff",
            "edge_strength",
            "edge_strength_raw",
            "is_strong_contact",
            "is_strong_overlap",
            "is_strong_soft",
            "is_structural_contact",
            "is_structural_overlap",
            "is_structural_edge",
            "is_support_edge",
            "is_strong_edge",
            "is_spatial_neighbor",
            "domain_pair_key",
        ]
    )


def _empty_interaction_structures_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "niche_id",
            "canonical_pattern_id",
            "component_id",
            "member_count",
            "backbone_node_count",
            "program_count",
            "program_ids",
            "backbone_program_pairs",
            "backbone_program_pair_count",
            "cross_program_edge_count",
            "strong_edge_count",
            "backbone_edge_count",
            "contact_fraction",
            "overlap_fraction",
            "proximity_fraction",
            "mean_edge_strength",
            "mean_edge_reliability",
            "basic_qc_pass",
            "basic_qc_fail_reason",
            "interaction_confidence",
            "random_occurrence_rate",
            "non_random_score",
            "random_qc_pass",
            "duplicate_collapsed_from_count",
            "backbone_node_keys",
            "member_node_keys",
            "seed_source_keys",
            "_backbone_edge_keys",
            "_strong_edge_signature_keys",
            "_backbone_program_pair_signature_keys",
            "_member_domain_keys",
            "_internal_edge_count",
            "_same_program_edge_count",
        ]
    )


def _empty_interaction_membership_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "niche_id",
            "domain_key",
            "program_id",
            "is_backbone_member",
            "is_structure_member",
            "joined_via_proximity",
            "seed_provenance",
        ]
    )


def _safe_percentile_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(arr.size, dtype=np.float64)
    mask = np.isfinite(arr)
    if int(np.count_nonzero(mask)) == 0:
        return out
    out[mask] = pd.Series(arr[mask]).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def _signature_tokens(value: object) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {tok for tok in text.split(";") if tok}


def _sorted_join(values: set[str] | list[str] | tuple[str, ...]) -> str:
    vals = sorted(set(str(v) for v in values if str(v)))
    return ";".join(vals) if vals else ""


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _program_pair_key(program_i: str, program_j: str) -> str:
    a = str(program_i)
    b = str(program_j)
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def _relation_edge_key(domain_i: str, domain_j: str) -> str:
    a = str(domain_i)
    b = str(domain_j)
    return f"{a}||{b}" if a <= b else f"{b}||{a}"


def _estimate_spot_spacing(
    spot_coords: np.ndarray | None,
    neighbors_idx: np.ndarray | None,
    q: float,
) -> float:
    if spot_coords is None:
        return float("nan")
    coords = np.asarray(spot_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] < 2:
        return float("nan")

    finite_mask = np.isfinite(coords).all(axis=1)
    if int(np.count_nonzero(finite_mask)) < 2:
        return float("nan")

    qq = float(np.clip(q, 0.0, 1.0))
    if neighbors_idx is not None and neighbors_idx.ndim == 2 and neighbors_idx.shape[0] == coords.shape[0]:
        nn_vals: list[float] = []
        for i in range(neighbors_idx.shape[0]):
            if not finite_mask[i]:
                continue
            nbs = np.asarray(neighbors_idx[i], dtype=np.int64)
            valid = (nbs >= 0) & (nbs < coords.shape[0]) & (nbs != i)
            if not np.any(valid):
                continue
            nbs = nbs[valid]
            nbs = nbs[finite_mask[nbs]]
            if nbs.size == 0:
                continue
            d = np.linalg.norm(coords[nbs, :2] - coords[i, :2], axis=1)
            d = d[np.isfinite(d) & (d > 0)]
            if d.size > 0:
                nn_vals.append(float(np.min(d)))
        if nn_vals:
            return float(np.quantile(np.asarray(nn_vals, dtype=np.float64), qq))

    coords_f = coords[finite_mask, :2]
    tree = cKDTree(coords_f)
    try:
        d, _ = tree.query(coords_f, k=2, workers=-1)
    except TypeError:
        d, _ = tree.query(coords_f, k=2)
    if np.asarray(d).ndim != 2 or d.shape[1] < 2:
        return float("nan")
    nn = np.asarray(d[:, 1], dtype=np.float64)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        return float("nan")
    return float(np.quantile(nn, qq))


def _resolve_epsilon(
    distances: np.ndarray,
    cfg: DomainEdgeConfig,
    spot_spacing: float,
) -> tuple[float, str]:
    finite_dist = np.asarray(distances, dtype=np.float64)
    finite_dist = finite_dist[np.isfinite(finite_dist) & (finite_dist >= 0)]
    if finite_dist.size == 0:
        return 0.0, "empty_distance"

    mode = str(cfg.epsilon_mode)
    if mode == "spot_spacing":
        if not np.isfinite(spot_spacing) or spot_spacing <= 0:
            raise ValueError("epsilon_mode='spot_spacing' requires valid spot spacing from raw coordinates.")
        return float(max(0.0, float(cfg.epsilon_spacing_multiplier) * float(spot_spacing))), "spot_spacing"
    if mode == "fixed":
        if cfg.epsilon_distance is None:
            return float(np.quantile(finite_dist, 0.25)), "fixed_default_q25"
        return float(max(0.0, cfg.epsilon_distance)), "fixed"
    q = float(np.clip(cfg.epsilon_distance_quantile, 0.0, 1.0))
    return float(np.quantile(finite_dist, q)), "distance_quantile"


def _build_domain_spot_index_map(
    membership_df: pd.DataFrame,
    spot_ids: np.ndarray | None,
    spot_coords: np.ndarray | None,
    domain_keys: set[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if spot_ids is None or spot_coords is None:
        raise ValueError("Raw spot_ids/spot_coords are required for proximity edge construction.")
    coords = np.asarray(spot_coords, dtype=np.float64)
    ids = np.asarray(spot_ids).astype(str)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != ids.shape[0]:
        raise ValueError("Invalid raw spot coordinate payload.")

    spot_to_idx = {s: i for i, s in enumerate(ids.tolist())}
    mem = membership_df.loc[membership_df["domain_key"].astype(str).isin(domain_keys), ["domain_key", "spot_id"]].copy()
    if mem.empty:
        raise ValueError("No domain membership entries available for requested domain keys.")

    unknown_spots = sorted(set(mem.loc[~mem["spot_id"].astype(str).isin(spot_to_idx.keys()), "spot_id"].astype(str).tolist()))
    if unknown_spots:
        raise ValueError(
            "domain_spot_membership contains spot_id values missing in raw spatial order; "
            f"examples={unknown_spots[:5]}"
        )

    domain_map: dict[str, np.ndarray] = {}
    for dkey, sub in mem.groupby("domain_key"):
        idx = sorted(set(int(spot_to_idx[str(s)]) for s in sub["spot_id"].astype(str).tolist()))
        domain_map[str(dkey)] = np.asarray(idx, dtype=np.int64)
    return domain_map, np.asarray(coords[:, :2], dtype=np.float64)


def _extract_boundary_spot_indices(
    domain_to_spot_idx: dict[str, np.ndarray],
    neighbors_idx: np.ndarray | None,
    use_boundary_spots: bool,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    out: dict[str, np.ndarray] = {}
    mode: dict[str, str] = {}
    use_neighbors = bool(use_boundary_spots) and neighbors_idx is not None and neighbors_idx.ndim == 2

    for dkey, idx in domain_to_spot_idx.items():
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            out[dkey] = idx
            mode[dkey] = "empty"
            continue
        if not use_neighbors:
            out[dkey] = idx
            mode[dkey] = "all_spots"
            continue

        domain_set = set(int(x) for x in idx.tolist())
        boundary: list[int] = []
        for s in idx.tolist():
            if s < 0 or s >= neighbors_idx.shape[0]:
                continue
            nbs = np.asarray(neighbors_idx[s], dtype=np.int64)
            valid = (nbs >= 0) & (nbs < neighbors_idx.shape[0]) & (nbs != s)
            if not np.any(valid):
                continue
            nbs = nbs[valid]
            if any(int(nb) not in domain_set for nb in nbs.tolist()):
                boundary.append(int(s))

        if len(boundary) < min(3, idx.size):
            out[dkey] = idx
            mode[dkey] = "fallback_all_spots"
        else:
            out[dkey] = np.asarray(sorted(set(boundary)), dtype=np.int64)
            mode[dkey] = "boundary_spots"
    return out, mode


def _directed_trimmed_nn_distance(
    src_coords: np.ndarray,
    tgt_tree: cKDTree,
    trim_fraction: float,
    trim_k_min: int,
) -> float:
    if src_coords.size == 0:
        return float("nan")
    try:
        d, _ = tgt_tree.query(src_coords, k=1, workers=-1)
    except TypeError:
        d, _ = tgt_tree.query(src_coords, k=1)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    frac = float(np.clip(trim_fraction, 0.0, 1.0))
    k = max(int(trim_k_min), int(math.ceil(frac * d.size)))
    k = min(k, int(d.size))
    vals = np.partition(d, k - 1)[:k] if k > 0 else d
    return float(np.mean(vals))


def _robust_boundary_distance(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    tree_a: cKDTree,
    tree_b: cKDTree,
    cfg: DomainEdgeConfig,
) -> float:
    d_ab = _directed_trimmed_nn_distance(
        src_coords=coords_a,
        tgt_tree=tree_b,
        trim_fraction=float(cfg.robust_boundary_distance_trim_fraction),
        trim_k_min=int(cfg.robust_boundary_distance_trim_k_min),
    )
    d_ba = _directed_trimmed_nn_distance(
        src_coords=coords_b,
        tgt_tree=tree_a,
        trim_fraction=float(cfg.robust_boundary_distance_trim_fraction),
        trim_k_min=int(cfg.robust_boundary_distance_trim_k_min),
    )
    if np.isfinite(d_ab) and np.isfinite(d_ba):
        if str(cfg.robust_boundary_distance_symmetry) == "mean":
            return float(0.5 * (d_ab + d_ba))
        return float(min(d_ab, d_ba))
    if np.isfinite(d_ab):
        return float(d_ab)
    if np.isfinite(d_ba):
        return float(d_ba)
    return float("nan")


def _positive_quantile_threshold(values: np.ndarray, quantile: float, floor: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return float("inf")
    return float(max(float(max(0.0, floor)), float(np.quantile(arr, float(np.clip(quantile, 0.0, 1.0))))))


def build_domain_adjacency_edges(
    domains_df: pd.DataFrame,
    domain_graph_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    spot_ids: np.ndarray | None,
    spot_coords: np.ndarray | None,
    neighbors_idx: np.ndarray | None,
    cfg: DomainEdgeConfig,
    rel_cfg: DomainReliabilityConfig,
) -> tuple[pd.DataFrame, dict]:
    use_domain_cols = [
        "domain_id",
        "domain_key",
        "program_seed_id",
        "spot_count",
        "domain_reliability",
        "boundary_edge_count",
    ]
    present_domain_cols = [c for c in use_domain_cols if c in domains_df.columns]
    domains_keep = domains_df.loc[domains_df["qc_pass"], present_domain_cols].copy()
    if domains_keep.empty:
        return _empty_edges_table(), {
            "epsilon_distance": 0.0,
            "epsilon_source": "empty_domains",
            "spot_spacing_estimate": float("nan"),
            "proximity_distance_source": "boundary",
            "strong_contact_threshold": float("inf"),
            "strong_overlap_threshold": float("inf"),
            "strong_soft_threshold": 0.0,
            "domain_reliability_enabled": bool(rel_cfg.enabled),
            "domain_reliability_pair_mode": str(rel_cfg.pair_mode),
        }

    if "boundary_edge_count" not in domains_keep.columns:
        k = max(1e-8, float(cfg.boundary_fallback_perimeter_scale))
        area = np.maximum(domains_keep["spot_count"].to_numpy(dtype=np.float64), 1.0)
        domains_keep["boundary_edge_count"] = np.maximum(1.0, k * np.sqrt(area))
    if "domain_reliability" not in domains_keep.columns:
        domains_keep["domain_reliability"] = 1.0

    key_to_domain = domains_keep.set_index("domain_key").to_dict(orient="index")
    keep_keys = set(str(x) for x in domains_keep["domain_key"].astype(str).tolist())

    edges = domain_graph_df.copy()
    edges = edges[
        edges["domain_key_i"].astype(str).isin(keep_keys)
        & edges["domain_key_j"].astype(str).isin(keep_keys)
    ].reset_index(drop=True)
    if edges.empty:
        return _empty_edges_table(), {
            "epsilon_distance": 0.0,
            "epsilon_source": "empty_edges",
            "spot_spacing_estimate": float("nan"),
            "proximity_distance_source": "boundary",
            "strong_contact_threshold": float("inf"),
            "strong_overlap_threshold": float("inf"),
            "strong_soft_threshold": 0.0,
            "domain_reliability_enabled": bool(rel_cfg.enabled),
            "domain_reliability_pair_mode": str(rel_cfg.pair_mode),
        }

    edges["domain_key_i"] = edges["domain_key_i"].astype(str)
    edges["domain_key_j"] = edges["domain_key_j"].astype(str)
    shared = np.maximum(edges["shared_boundary_edges"].to_numpy(dtype=np.float64), 0.0)
    overlap = np.clip(edges["spatial_overlap"].to_numpy(dtype=np.float64), 0.0, 1.0)

    boundary_dist = np.full(edges.shape[0], np.nan, dtype=np.float64)
    domain_spot_idx_map, coords_xy = _build_domain_spot_index_map(
        membership_df=membership_df,
        spot_ids=spot_ids,
        spot_coords=spot_coords,
        domain_keys=keep_keys,
    )
    boundary_idx_map, boundary_mode_map = _extract_boundary_spot_indices(
        domain_to_spot_idx=domain_spot_idx_map,
        neighbors_idx=neighbors_idx,
        use_boundary_spots=bool(cfg.robust_boundary_distance_use_boundary_spots),
    )
    boundary_coord_map: dict[str, np.ndarray] = {}
    tree_cache: dict[str, cKDTree | None] = {}
    for dkey, idx_arr in boundary_idx_map.items():
        arr = np.asarray(coords_xy[np.asarray(idx_arr, dtype=np.int64)], dtype=np.float64) if len(idx_arr) > 0 else np.zeros((0, 2), dtype=np.float64)
        arr = arr[np.isfinite(arr).all(axis=1)]
        boundary_coord_map[dkey] = arr
        tree_cache[dkey] = cKDTree(arr) if arr.shape[0] > 0 else None

    for idx, r in enumerate(edges.itertuples(index=False)):
        if float(getattr(r, "spatial_overlap", 0.0)) > 0 or float(getattr(r, "shared_boundary_edges", 0.0)) > 0:
            boundary_dist[idx] = 0.0
            continue
        ki = str(r.domain_key_i)
        kj = str(r.domain_key_j)
        ci = boundary_coord_map.get(ki, np.zeros((0, 2), dtype=np.float64))
        cj = boundary_coord_map.get(kj, np.zeros((0, 2), dtype=np.float64))
        ti = tree_cache.get(ki)
        tj = tree_cache.get(kj)
        if ci.size == 0 or cj.size == 0 or ti is None or tj is None:
            continue
        boundary_dist[idx] = _robust_boundary_distance(ci, cj, ti, tj, cfg)

    proximity_dist = np.asarray(boundary_dist, dtype=np.float64)
    if np.any(~np.isfinite(proximity_dist)):
        raise ValueError("Boundary distance contains non-finite values.")

    boundary_contact = shared >= int(cfg.min_shared_boundary_edges)
    overlap_contact = overlap > float(cfg.min_spatial_overlap)
    if bool(cfg.suppress_contact_when_overlap):
        boundary_contact = boundary_contact & (~overlap_contact)

    spot_spacing = _estimate_spot_spacing(spot_coords=spot_coords, neighbors_idx=neighbors_idx, q=float(cfg.epsilon_spacing_quantile))
    epsilon, epsilon_source = _resolve_epsilon(proximity_dist, cfg, spot_spacing)
    if cfg.use_proximity_edges and epsilon > 0:
        proximity_contact = np.isfinite(proximity_dist) & (proximity_dist <= epsilon)
        soft = np.zeros_like(proximity_dist, dtype=np.float64)
        soft[proximity_contact] = (epsilon - proximity_dist[proximity_contact]) / max(1e-8, epsilon)
    else:
        proximity_contact = np.zeros(edges.shape[0], dtype=bool)
        soft = np.zeros(edges.shape[0], dtype=np.float64)
    if bool(cfg.soft_redundant_when_overlap_or_contact):
        redundant_soft = boundary_contact | overlap_contact
        proximity_contact = proximity_contact & (~redundant_soft)
        soft[redundant_soft] = 0.0

    keep_mask = boundary_contact | overlap_contact | proximity_contact
    edges = edges.loc[keep_mask].reset_index(drop=True)
    if edges.empty:
        return _empty_edges_table(), {
            "epsilon_distance": float(epsilon),
            "epsilon_source": str(epsilon_source),
            "spot_spacing_estimate": float(spot_spacing),
            "proximity_distance_source": "boundary",
            "strong_contact_threshold": float("inf"),
            "strong_overlap_threshold": float("inf"),
            "strong_soft_threshold": 0.0,
            "domain_reliability_enabled": bool(rel_cfg.enabled),
            "domain_reliability_pair_mode": str(rel_cfg.pair_mode),
        }

    shared = shared[keep_mask]
    overlap = overlap[keep_mask]
    boundary_dist = boundary_dist[keep_mask]
    proximity_dist = proximity_dist[keep_mask]
    boundary_contact = boundary_contact[keep_mask]
    overlap_contact = overlap_contact[keep_mask]
    proximity_contact = proximity_contact[keep_mask]
    soft = soft[keep_mask]

    fallback_k = max(1e-8, float(cfg.boundary_fallback_perimeter_scale))
    fallback_used = 0

    def _boundary_size_for_domain(dom: dict) -> float:
        nonlocal fallback_used
        b = float(dom.get("boundary_edge_count", float("nan")))
        if np.isfinite(b) and b > 0:
            return float(max(1.0, b))
        fallback_used += 1
        return float(max(1.0, fallback_k * math.sqrt(max(1.0, float(dom.get("spot_count", 1.0))))))

    boundary_size_i = np.asarray([_boundary_size_for_domain(key_to_domain[str(k)]) for k in edges["domain_key_i"].astype(str).tolist()], dtype=np.float64)
    boundary_size_j = np.asarray([_boundary_size_for_domain(key_to_domain[str(k)]) for k in edges["domain_key_j"].astype(str).tolist()], dtype=np.float64)
    contact_ratio = np.clip(shared / np.maximum(1.0, np.minimum(boundary_size_i, boundary_size_j)), 0.0, 1.0)

    map_mode = str(cfg.contact_strength_mapping)
    sat_c = float(max(1e-8, cfg.contact_strength_saturation_c))
    if map_mode == "identity":
        contact_strength = contact_ratio
    elif map_mode == "tanh":
        contact_strength = np.tanh(contact_ratio / sat_c)
    else:
        contact_strength = contact_ratio / (contact_ratio + sat_c)
    contact_strength = np.clip(contact_strength, 0.0, 1.0)
    overlap_strength = overlap
    soft_strength = np.clip(soft, 0.0, 1.0)

    reliability_i = np.asarray([
        np.clip(float(key_to_domain[str(k)].get("domain_reliability", 1.0)), float(rel_cfg.min_node_reliability), float(rel_cfg.max_node_reliability))
        for k in edges["domain_key_i"].astype(str).tolist()
    ], dtype=np.float64)
    reliability_j = np.asarray([
        np.clip(float(key_to_domain[str(k)].get("domain_reliability", 1.0)), float(rel_cfg.min_node_reliability), float(rel_cfg.max_node_reliability))
        for k in edges["domain_key_j"].astype(str).tolist()
    ], dtype=np.float64)

    pair_mode = str(rel_cfg.pair_mode)
    if pair_mode == "mean":
        edge_reliability = 0.5 * (reliability_i + reliability_j)
    elif pair_mode == "min":
        edge_reliability = np.minimum(reliability_i, reliability_j)
    else:
        edge_reliability = np.sqrt(np.maximum(0.0, reliability_i * reliability_j))
    if float(rel_cfg.pair_power) > 0:
        edge_reliability = np.power(edge_reliability, float(rel_cfg.pair_power))
    edge_reliability = np.clip(edge_reliability, float(rel_cfg.min_node_reliability), float(rel_cfg.max_node_reliability))
    edge_reliability_factor = (1.0 - float(np.clip(rel_cfg.edge_reliability_blend, 0.0, 1.0))) + float(np.clip(rel_cfg.edge_reliability_blend, 0.0, 1.0)) * edge_reliability

    contact_strength_eff = np.clip(contact_strength * edge_reliability_factor, 0.0, 1.0)
    overlap_strength_eff = np.clip(overlap_strength * edge_reliability_factor, 0.0, 1.0)
    soft_strength_eff = np.clip(soft_strength * edge_reliability_factor, 0.0, 1.0)

    weights_sum = float(cfg.shared_boundary_scale + cfg.spatial_overlap_scale + cfg.proximity_scale)
    if weights_sum <= 0:
        weights_sum = 1.0
    edge_strength_raw = (
        float(cfg.shared_boundary_scale) * contact_strength
        + float(cfg.spatial_overlap_scale) * overlap_strength
        + float(cfg.proximity_scale) * soft_strength
    ) / weights_sum
    edge_strength = (
        float(cfg.shared_boundary_scale) * contact_strength_eff
        + float(cfg.spatial_overlap_scale) * overlap_strength_eff
        + float(cfg.proximity_scale) * soft_strength_eff
    ) / weights_sum

    strong_contact_thr = _positive_quantile_threshold(contact_strength_eff[boundary_contact], float(cfg.strong_contact_quantile), float(cfg.strong_contact_min_strength))
    strong_overlap_thr = _positive_quantile_threshold(overlap_strength_eff[overlap_contact], float(cfg.strong_overlap_quantile), float(cfg.strong_overlap_min_strength))
    is_strong_contact = boundary_contact & (contact_strength_eff >= strong_contact_thr)
    is_strong_overlap = overlap_contact & (overlap_strength_eff >= strong_overlap_thr)
    is_structural_contact = is_strong_contact
    is_structural_overlap = is_strong_overlap
    is_structural_edge = is_structural_contact | is_structural_overlap
    is_support_edge = is_structural_edge | proximity_contact
    is_strong_edge = is_structural_edge.copy()

    relation_types: list[str] = []
    for i in range(edges.shape[0]):
        tags: list[str] = []
        if boundary_contact[i]:
            tags.append("boundary_contact")
        if overlap_contact[i]:
            tags.append("co_localization")
        if proximity_contact[i]:
            tags.append("proximity")
        relation_types.append(";".join(tags) if tags else "none")

    rows: list[dict] = []
    for idx, r in enumerate(edges.itertuples(index=False)):
        ki = str(r.domain_key_i)
        kj = str(r.domain_key_j)
        di = key_to_domain[ki]
        dj = key_to_domain[kj]
        rows.append(
            {
                "domain_key_i": ki,
                "domain_key_j": kj,
                "domain_id_i": str(di["domain_id"]),
                "domain_id_j": str(dj["domain_id"]),
                "program_id_i": str(di["program_seed_id"]),
                "program_id_j": str(dj["program_seed_id"]),
                "spot_count_i": int(di["spot_count"]),
                "spot_count_j": int(dj["spot_count"]),
                "shared_boundary_edges": int(r.shared_boundary_edges),
                "spatial_overlap": float(r.spatial_overlap),
                "boundary_distance": float(boundary_dist[idx]),
                "proximity_distance": float(proximity_dist[idx]),
                "boundary_size_i": float(boundary_size_i[idx]),
                "boundary_size_j": float(boundary_size_j[idx]),
                "domain_reliability_i": float(reliability_i[idx]),
                "domain_reliability_j": float(reliability_j[idx]),
                "edge_reliability": float(edge_reliability[idx]),
                "edge_reliability_factor": float(edge_reliability_factor[idx]),
                "edge_weight_raw": float(r.edge_weight),
                "boundary_contact": bool(boundary_contact[idx]),
                "overlap_contact": bool(overlap_contact[idx]),
                "proximity_contact": bool(proximity_contact[idx]),
                "relation_types": relation_types[idx],
                "contact_ratio": float(contact_ratio[idx]),
                "contact_strength": float(contact_strength[idx]),
                "contact_strength_eff": float(contact_strength_eff[idx]),
                "overlap_strength": float(overlap_strength[idx]),
                "overlap_strength_eff": float(overlap_strength_eff[idx]),
                "soft_strength": float(soft_strength[idx]),
                "soft_strength_eff": float(soft_strength_eff[idx]),
                "edge_strength": float(edge_strength[idx]),
                "edge_strength_raw": float(edge_strength_raw[idx]),
                "is_strong_contact": bool(is_strong_contact[idx]),
                "is_strong_overlap": bool(is_strong_overlap[idx]),
                "is_strong_soft": False,
                "is_structural_contact": bool(is_structural_contact[idx]),
                "is_structural_overlap": bool(is_structural_overlap[idx]),
                "is_structural_edge": bool(is_structural_edge[idx]),
                "is_support_edge": bool(is_support_edge[idx]),
                "is_strong_edge": bool(is_strong_edge[idx]),
                "is_spatial_neighbor": True,
                "domain_pair_key": _relation_edge_key(ki, kj),
            }
        )

    out = pd.DataFrame(rows)
    meta = {
        "epsilon_distance": float(epsilon),
        "epsilon_source": str(epsilon_source),
        "spot_spacing_estimate": float(spot_spacing),
        "proximity_distance_source": "boundary",
        "boundary_distance_method": "trimmed_nn",
        "boundary_trim_fraction": float(cfg.robust_boundary_distance_trim_fraction),
        "boundary_trim_k_min": int(cfg.robust_boundary_distance_trim_k_min),
        "boundary_symmetry": str(cfg.robust_boundary_distance_symmetry),
        "boundary_spot_mode_used": ",".join(sorted(set(boundary_mode_map.values()))) if boundary_mode_map else "none",
        "boundary_fallback_perimeter_scale": float(fallback_k),
        "boundary_fallback_used_count": int(fallback_used),
        "contact_strength_mapping": str(map_mode),
        "contact_strength_saturation_c": float(sat_c),
        "strong_contact_threshold": float(strong_contact_thr),
        "strong_overlap_threshold": float(strong_overlap_thr),
        "strong_soft_threshold": 0.0,
        "strong_contact_mode": "quantile_floor",
        "strong_overlap_mode": "quantile_floor",
        "strong_soft_mode": "disabled",
        "strong_contact_quantile": float(cfg.strong_contact_quantile),
        "strong_overlap_quantile": float(cfg.strong_overlap_quantile),
        "strong_contact_min_strength": float(cfg.strong_contact_min_strength),
        "strong_overlap_min_strength": float(cfg.strong_overlap_min_strength),
        "domain_reliability_enabled": bool(rel_cfg.enabled),
        "domain_reliability_pair_mode": str(pair_mode),
        "domain_reliability_mean": float(np.mean(edge_reliability)) if edge_reliability.size > 0 else float("nan"),
        "domain_reliability_factor_mean": float(np.mean(edge_reliability_factor)) if edge_reliability_factor.size > 0 else float("nan"),
    }
    return out, meta


def _prepare_interaction_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    out = edges_df.copy()
    if out.empty:
        out["edge_key"] = pd.Series(dtype=object)
        out["program_pair_key"] = pd.Series(dtype=object)
        out["is_cross_program"] = pd.Series(dtype=bool)
        out["is_strong_relation"] = pd.Series(dtype=bool)
        out["is_weak_relation"] = pd.Series(dtype=bool)
        out["seed_score"] = pd.Series(dtype=np.float64)
        return out

    out["edge_key"] = out["domain_pair_key"].astype(str)
    out["program_pair_key"] = [
        _program_pair_key(pi, pj)
        for pi, pj in zip(out["program_id_i"].astype(str).tolist(), out["program_id_j"].astype(str).tolist())
    ]
    out["is_cross_program"] = out["program_id_i"].astype(str).to_numpy() != out["program_id_j"].astype(str).to_numpy()
    out["is_strong_relation"] = out["is_strong_contact"].to_numpy(dtype=bool) | out["is_strong_overlap"].to_numpy(dtype=bool)
    out["is_weak_relation"] = out["proximity_contact"].to_numpy(dtype=bool)
    out["seed_score"] = np.zeros(out.shape[0], dtype=np.float64)

    seed_mask = out["is_cross_program"].to_numpy(dtype=bool) & out["is_strong_relation"].to_numpy(dtype=bool)
    if int(np.count_nonzero(seed_mask)) > 0:
        strength_rank = _safe_percentile_rank(out.loc[seed_mask, "edge_strength"].to_numpy(dtype=np.float64))
        reliability_rank = _safe_percentile_rank(out.loc[seed_mask, "edge_reliability"].to_numpy(dtype=np.float64))
        out.loc[seed_mask, "seed_score"] = np.clip(0.65 * strength_rank + 0.35 * reliability_rank, 0.0, 1.0)
    return out


def _discover_interaction_seeds(edges_df: pd.DataFrame, discovery_cfg: InteractionDiscoveryConfig) -> pd.DataFrame:
    if edges_df.empty:
        return edges_df.iloc[0:0].copy()
    eligible = edges_df.loc[
        edges_df["is_cross_program"].to_numpy(dtype=bool) & edges_df["is_strong_relation"].to_numpy(dtype=bool)
    ].copy()
    if eligible.empty:
        return eligible

    threshold = float(
        np.quantile(
            eligible["seed_score"].to_numpy(dtype=np.float64),
            float(np.clip(discovery_cfg.seed_score_quantile, 0.0, 1.0)),
        )
    )
    seeds = eligible.loc[eligible["seed_score"].to_numpy(dtype=np.float64) >= threshold].copy()
    if seeds.empty:
        seeds = eligible.nlargest(1, columns=["seed_score", "edge_strength", "edge_reliability"]).copy()

    g = nx.Graph()
    for r in eligible.itertuples(index=False):
        g.add_edge(str(r.domain_key_i), str(r.domain_key_j))
    node_to_component: dict[str, int] = {}
    for comp_id, comp in enumerate(nx.connected_components(g), start=1):
        for node in comp:
            node_to_component[str(node)] = comp_id
    seeds["component_id"] = [int(node_to_component.get(str(u), 0)) for u in seeds["domain_key_i"].astype(str).tolist()]
    seeds = seeds.sort_values(
        by=["component_id", "seed_score", "edge_strength", "edge_reliability", "edge_key"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    return seeds.groupby("component_id", group_keys=False).head(int(max(1, discovery_cfg.max_seeds_per_component))).reset_index(drop=True)


def _component_edge_rank(edges_df: pd.DataFrame) -> np.ndarray:
    if edges_df.empty:
        return np.asarray([], dtype=np.float64)
    strength_rank = _safe_percentile_rank(edges_df["edge_strength"].to_numpy(dtype=np.float64))
    reliability_rank = _safe_percentile_rank(edges_df["edge_reliability"].to_numpy(dtype=np.float64))
    return np.clip(0.65 * strength_rank + 0.35 * reliability_rank, 0.0, 1.0)


def _select_diverse_seeds_for_component(
    comp_seeds: pd.DataFrame,
    discovery_cfg: InteractionDiscoveryConfig,
) -> pd.DataFrame:
    if comp_seeds.empty:
        return comp_seeds.copy()
    ranked = comp_seeds.sort_values(
        by=["seed_score", "edge_strength", "edge_reliability", "edge_key"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    max_keep = 3
    if discovery_cfg.max_patterns_per_component is not None:
        max_keep = min(max_keep, int(max(1, discovery_cfg.max_patterns_per_component)))
    selected_rows: list[int] = []
    seen_pair_keys: set[str] = set()
    seen_endpoint_sets: set[tuple[str, str]] = set()
    for i in range(ranked.shape[0]):
        row = ranked.iloc[i]
        pair_key = str(row["program_pair_key"])
        endpoint_key = tuple(sorted((str(row["domain_key_i"]), str(row["domain_key_j"]))))
        if pair_key in seen_pair_keys:
            continue
        if endpoint_key in seen_endpoint_sets:
            continue
        selected_rows.append(i)
        seen_pair_keys.add(pair_key)
        seen_endpoint_sets.add(endpoint_key)
        if len(selected_rows) >= max_keep:
            break
    if not selected_rows:
        selected_rows = [0]
    return ranked.iloc[selected_rows].reset_index(drop=True)


def _extract_local_region(
    component_edges: pd.DataFrame,
    seed_row: pd.Series,
) -> dict[str, object]:
    if component_edges.empty:
        return {
            "backbone_region_nodes": set(),
            "context_region_nodes": set(),
            "backbone_region_edges": component_edges.copy(),
        }
    strong_cross = component_edges.loc[
        component_edges["is_cross_program"].to_numpy(dtype=bool)
        & component_edges["is_strong_relation"].to_numpy(dtype=bool)
    ].copy()
    if strong_cross.empty:
        return {
            "backbone_region_nodes": set(),
            "context_region_nodes": set(),
            "backbone_region_edges": strong_cross,
        }
    seed_u = str(seed_row["domain_key_i"])
    seed_v = str(seed_row["domain_key_j"])
    seed_nodes = {seed_u, seed_v}
    seed_pair_key = str(seed_row["program_pair_key"])
    seed_programs = {str(seed_row["program_id_i"]), str(seed_row["program_id_j"])}
    neighbor_edges = strong_cross.loc[
        strong_cross["domain_key_i"].astype(str).isin(seed_nodes)
        | strong_cross["domain_key_j"].astype(str).isin(seed_nodes)
    ].copy()
    base_region_nodes = set(seed_nodes)
    if not neighbor_edges.empty:
        base_region_nodes |= set(neighbor_edges["domain_key_i"].astype(str).tolist())
        base_region_nodes |= set(neighbor_edges["domain_key_j"].astype(str).tolist())

    def _collect_region_context_nodes(base_region_nodes: set[str]) -> set[str]:
        context_nodes: set[str] = set()
        base_programs: set[str] = set()
        strong_to_base: dict[str, set[str]] = defaultdict(set)
        weak_to_base: dict[str, set[str]] = defaultdict(set)
        strong_programs: dict[str, set[str]] = defaultdict(set)
        weak_programs: dict[str, set[str]] = defaultdict(set)
        candidate_programs: dict[str, set[str]] = defaultdict(set)

        for row in component_edges.itertuples(index=False):
            u = str(row.domain_key_i)
            v = str(row.domain_key_j)
            pu = str(row.program_id_i)
            pv = str(row.program_id_j)
            if u in base_region_nodes:
                base_programs.add(pu)
            if v in base_region_nodes:
                base_programs.add(pv)
            u_in = u in base_region_nodes
            v_in = v in base_region_nodes
            if u_in == v_in:
                continue
            base_node = u if u_in else v
            cand_node = v if u_in else u
            base_prog = pu if u_in else pv
            cand_prog = pv if u_in else pu
            candidate_programs[cand_node].add(cand_prog)
            if bool(row.is_cross_program) and bool(row.is_strong_relation):
                strong_to_base[cand_node].add(base_node)
                strong_programs[cand_node].add(base_prog)
            elif bool(row.is_weak_relation):
                weak_to_base[cand_node].add(base_node)
                weak_programs[cand_node].add(base_prog)

        core_related_programs = set(seed_programs) | set(base_programs)
        for node, node_programs in candidate_programs.items():
            strong_links = len(strong_to_base.get(node, set()))
            weak_links = len(weak_to_base.get(node, set()))
            strong_prog_cov = len(strong_programs.get(node, set()))
            weak_prog_cov = len(weak_programs.get(node, set()))
            consistent = bool(node_programs & core_related_programs)
            if strong_links >= 2:
                context_nodes.add(node)
                continue
            if weak_links >= 2 and weak_prog_cov >= 2:
                context_nodes.add(node)
                continue
            if consistent and max(strong_prog_cov, weak_prog_cov) >= 2:
                context_nodes.add(node)
        return context_nodes

    context_region_nodes = _collect_region_context_nodes(base_region_nodes)
    region_all = strong_cross.loc[
        strong_cross["domain_key_i"].astype(str).isin(base_region_nodes)
        & strong_cross["domain_key_j"].astype(str).isin(base_region_nodes)
    ].copy()
    if region_all.empty:
        region_all = strong_cross.loc[
            strong_cross["edge_key"].astype(str).eq(str(seed_row["edge_key"]))
        ].copy()

    seed_neighbor_nodes = set(base_region_nodes) - set(seed_nodes)
    seed_context_nodes = set(seed_nodes) | set(seed_neighbor_nodes)

    adjacency_to_seed_context: dict[str, set[str]] = defaultdict(set)
    pair_context_by_node: dict[str, set[str]] = defaultdict(set)
    for row in region_all.itertuples(index=False):
        u = str(row.domain_key_i)
        v = str(row.domain_key_j)
        pair = str(row.program_pair_key)
        if u in seed_context_nodes and v in base_region_nodes:
            adjacency_to_seed_context[v].add(u)
            pair_context_by_node[v].add(pair)
        if v in seed_context_nodes and u in base_region_nodes:
            adjacency_to_seed_context[u].add(v)
            pair_context_by_node[u].add(pair)

    def _edge_is_seed_semantically_consistent(row: pd.Series) -> bool:
        pair = str(row["program_pair_key"])
        if pair == seed_pair_key:
            return True
        pair_programs = _pair_programs(pair)
        if pair_programs & seed_programs:
            return True
        u = str(row["domain_key_i"])
        v = str(row["domain_key_j"])
        u_links = adjacency_to_seed_context.get(u, set()) - {v}
        v_links = adjacency_to_seed_context.get(v, set()) - {u}
        local_pairs = set(pair_context_by_node.get(u, set())) | set(pair_context_by_node.get(v, set())) | {pair}
        return bool(u_links and v_links and len(local_pairs) >= 2)

    keep_mask = [_edge_is_seed_semantically_consistent(region_all.iloc[i]) for i in range(region_all.shape[0])]
    backbone_region_edges = region_all.loc[np.asarray(keep_mask, dtype=bool)].copy()
    if backbone_region_edges.empty:
        backbone_region_edges = region_all.loc[
            region_all["edge_key"].astype(str).eq(str(seed_row["edge_key"]))
        ].copy()
    return {
        "backbone_region_nodes": set(base_region_nodes),
        "context_region_nodes": set(context_region_nodes) - set(base_region_nodes),
        "backbone_region_edges": backbone_region_edges.reset_index(drop=True),
    }


def _count_core_neighbors(candidate_edges: pd.DataFrame, candidate_node: str, core_nodes: set[str], relation_col: str) -> int:
    neighbors: set[str] = set()
    for r in candidate_edges.loc[candidate_edges[relation_col].to_numpy(dtype=bool)].itertuples(index=False):
        if str(r.domain_key_i) == str(candidate_node):
            other = str(r.domain_key_j)
        else:
            other = str(r.domain_key_i)
        if other in core_nodes:
            neighbors.add(other)
    return int(len(neighbors))


def _is_informative_new_pair(
    pair_key: str,
    existing_programs: set[str],
    existing_program_pairs: set[str],
    pair_support_count: int,
) -> bool:
    if int(pair_support_count) < 2:
        return False
    pair_txt = str(pair_key)
    if not pair_txt or pair_txt in existing_program_pairs:
        return False
    parts = pair_txt.split("|", 1)
    if len(parts) != 2:
        return False
    p1, p2 = parts
    return bool((p1 not in existing_programs) or (p2 not in existing_programs))


def _backbone_local_pair_support_count(
    component_edges: pd.DataFrame,
    pair_key: str,
    core_nodes: set[str],
    candidate_nodes: set[str],
) -> int:
    if component_edges.empty:
        return 0
    sub = component_edges.loc[
        component_edges["is_cross_program"].to_numpy(dtype=bool)
        & component_edges["is_strong_relation"].to_numpy(dtype=bool)
        & component_edges["program_pair_key"].astype(str).eq(str(pair_key)).to_numpy(dtype=bool)
    ].copy()
    if sub.empty:
        return 0
    frontier_nodes: set[str] = set()
    strong_sub = component_edges.loc[
        component_edges["is_cross_program"].to_numpy(dtype=bool)
        & component_edges["is_strong_relation"].to_numpy(dtype=bool)
    ].copy()
    if not strong_sub.empty:
        for row in strong_sub.itertuples(index=False):
            u = str(row.domain_key_i)
            v = str(row.domain_key_j)
            if u in core_nodes and v not in core_nodes:
                frontier_nodes.add(v)
            if v in core_nodes and u not in core_nodes:
                frontier_nodes.add(u)
    neighborhood = set(core_nodes) | set(candidate_nodes) | frontier_nodes
    keep_mask = []
    for row in sub.itertuples(index=False):
        u = str(row.domain_key_i)
        v = str(row.domain_key_j)
        keep_mask.append(
            (u in neighborhood)
            and (v in neighborhood)
            and (
                (u in core_nodes)
                or (v in core_nodes)
                or (u in candidate_nodes)
                or (v in candidate_nodes)
                or (u in frontier_nodes)
                or (v in frontier_nodes)
            )
        )
    return int(np.count_nonzero(np.asarray(keep_mask, dtype=bool)))


def _backbone_forms_minimal_multipair_skeleton(
    component_edges: pd.DataFrame,
    pair_key: str,
    core_nodes: set[str],
    candidate_nodes: set[str],
    core_program_pairs: set[str],
) -> bool:
    if component_edges.empty:
        return False
    strong_sub = component_edges.loc[
        component_edges["is_cross_program"].to_numpy(dtype=bool)
        & component_edges["is_strong_relation"].to_numpy(dtype=bool)
    ].copy()
    if strong_sub.empty:
        return False

    frontier_nodes: set[str] = set()
    for row in strong_sub.itertuples(index=False):
        u = str(row.domain_key_i)
        v = str(row.domain_key_j)
        if u in core_nodes and v not in core_nodes:
            frontier_nodes.add(v)
        if v in core_nodes and u not in core_nodes:
            frontier_nodes.add(u)

    neighborhood = set(core_nodes) | set(candidate_nodes) | frontier_nodes
    local = strong_sub.loc[
        strong_sub["domain_key_i"].astype(str).isin(neighborhood)
        & strong_sub["domain_key_j"].astype(str).isin(neighborhood)
        & (
            strong_sub["domain_key_i"].astype(str).isin(candidate_nodes)
            | strong_sub["domain_key_j"].astype(str).isin(candidate_nodes)
        )
    ].copy()
    if local.empty:
        return False

    local_pairs = set(local["program_pair_key"].astype(str).tolist())
    combined_pairs = set(core_program_pairs) | local_pairs
    return (
        str(pair_key) in local_pairs
        and len(combined_pairs) >= 2
        and int(local.shape[0]) >= 2
        and any(pair not in core_program_pairs for pair in local_pairs)
    )


def _canonical_pattern_id(component_id: int, core_nodes: set[str], core_pairs: set[str], backbone_edge_keys: set[str]) -> str:
    return "::".join(
        [
            f"C{int(component_id)}",
            _sorted_join(core_nodes),
            _sorted_join(core_pairs),
            _sorted_join(backbone_edge_keys),
        ]
    )


def _pair_programs(pair_key: str) -> set[str]:
    txt = str(pair_key)
    parts = txt.split("|", 1)
    if len(parts) != 2:
        return set()
    return {parts[0], parts[1]}


def _candidate_forms_local_multipair_skeleton(
    candidate_row: pd.Series,
    selected_edge_keys: set[str],
    component_edges: pd.DataFrame,
    core_nodes: set[str],
    core_program_pairs: set[str],
) -> bool:
    pair_key = str(candidate_row["program_pair_key"])
    if pair_key in core_program_pairs:
        return False
    pair_programs = _pair_programs(pair_key)
    if not pair_programs:
        return False
    existing_pair_programs = [_pair_programs(pair) for pair in core_program_pairs]
    if not any(pair_programs & programs for programs in existing_pair_programs):
        return False

    edge_key = str(candidate_row["edge_key"])
    candidate_nodes = {str(candidate_row["domain_key_i"]), str(candidate_row["domain_key_j"])}
    neighborhood_nodes = set(core_nodes) | set(candidate_nodes)
    local_edges = component_edges.loc[
        component_edges["edge_key"].astype(str).isin(set(selected_edge_keys) | {edge_key})
        & component_edges["is_cross_program"].to_numpy(dtype=bool)
        & component_edges["is_strong_relation"].to_numpy(dtype=bool)
        & component_edges["domain_key_i"].astype(str).isin(neighborhood_nodes)
        & component_edges["domain_key_j"].astype(str).isin(neighborhood_nodes)
    ].copy()
    if local_edges.empty:
        return False

    local_pairs = set(local_edges["program_pair_key"].astype(str).tolist())
    if len(local_pairs) < 2 or pair_key not in local_pairs:
        return False

    core_incident_edges = local_edges.loc[
        local_edges["domain_key_i"].astype(str).isin(core_nodes)
        | local_edges["domain_key_j"].astype(str).isin(core_nodes)
    ].copy()
    return int(core_incident_edges["program_pair_key"].astype(str).nunique()) >= 2


def _candidate_adds_anchored_new_program(
    candidate_row: pd.Series,
    component_edges: pd.DataFrame,
    core_nodes: set[str],
    core_programs: set[str],
    core_program_pairs: set[str],
) -> bool:
    u = str(candidate_row["domain_key_i"])
    v = str(candidate_row["domain_key_j"])
    pu = str(candidate_row["program_id_i"])
    pv = str(candidate_row["program_id_j"])
    candidate_programs = {pu, pv}
    new_programs = candidate_programs - set(core_programs)
    if not new_programs:
        return False

    neighborhood_nodes = set(core_nodes) | {u, v}
    local = component_edges.loc[
        component_edges["is_cross_program"].to_numpy(dtype=bool)
        & component_edges["is_strong_relation"].to_numpy(dtype=bool)
        & (
            component_edges["domain_key_i"].astype(str).isin(neighborhood_nodes)
            | component_edges["domain_key_j"].astype(str).isin(neighborhood_nodes)
        )
    ].copy()
    if local.empty:
        return False

    for new_program in new_programs:
        support_nodes: set[str] = set()
        support_existing_pairs: set[str] = set()
        for row in local.itertuples(index=False):
            ru = str(row.domain_key_i)
            rv = str(row.domain_key_j)
            rpu = str(row.program_id_i)
            rpv = str(row.program_id_j)
            rpair = str(row.program_pair_key)
            if rpu == new_program and rv in core_nodes:
                support_nodes.add(rv)
                if rpair in core_program_pairs:
                    support_existing_pairs.add(rpair)
            elif rpv == new_program and ru in core_nodes:
                support_nodes.add(ru)
                if rpair in core_program_pairs:
                    support_existing_pairs.add(rpair)
        if len(support_nodes) >= 2:
            return True
        if len(support_existing_pairs) >= 1 and len(core_program_pairs) >= 1:
            return True
    return False


def _local_backbone_pair_edges(
    component_edges: pd.DataFrame,
    pair_key: str,
    edge_keys: set[str],
    neighborhood_nodes: set[str],
) -> pd.DataFrame:
    if not edge_keys:
        return component_edges.iloc[0:0].copy()
    return component_edges.loc[
        component_edges["edge_key"].astype(str).isin(edge_keys)
        & component_edges["program_pair_key"].astype(str).eq(str(pair_key))
        & component_edges["domain_key_i"].astype(str).isin(neighborhood_nodes)
        & component_edges["domain_key_j"].astype(str).isin(neighborhood_nodes)
    ].copy()


def _pair_support_state_from_edges(
    pair_edges: pd.DataFrame,
    backbone_nodes: set[str],
) -> str:
    if pair_edges.empty:
        return "weak"
    anchor_nodes: set[str] = set()
    for row in pair_edges.itertuples(index=False):
        u0 = str(row.domain_key_i)
        v0 = str(row.domain_key_j)
        if u0 in backbone_nodes:
            anchor_nodes.add(u0)
        if v0 in backbone_nodes:
            anchor_nodes.add(v0)
    if pair_edges.shape[0] >= 2 and len(anchor_nodes) >= 2:
        return "stable"
    return "weak"


def _compute_pair_redundancy_penalty(
    component_edges: pd.DataFrame,
    pair_key: str,
    selected_edge_keys: set[str],
    core_nodes: set[str],
    candidate_nodes: set[str],
    core_program_pairs: set[str],
) -> float:
    if pair_key not in core_program_pairs:
        return 0.0
    neighborhood_nodes = set(core_nodes) | set(candidate_nodes)
    pair_edges = _local_backbone_pair_edges(
        component_edges=component_edges,
        pair_key=pair_key,
        edge_keys=selected_edge_keys,
        neighborhood_nodes=neighborhood_nodes,
    )
    if pair_edges.empty:
        return 0.0
    state = _pair_support_state_from_edges(pair_edges=pair_edges, backbone_nodes=set(core_nodes))
    if state != "stable":
        return 0.0
    if pair_edges.shape[0] >= 3:
        return 1.0
    return 0.5


def _compute_dense_repeat_penalty(
    candidate_row: pd.Series,
    component_edges: pd.DataFrame,
    core_nodes: set[str],
    core_program_pairs: set[str],
    new_pair_novelty: float,
    new_program_novelty: float,
) -> float:
    if new_pair_novelty > 0.0 or new_program_novelty > 0.0:
        return 0.0
    candidate_nodes = {str(candidate_row["domain_key_i"]), str(candidate_row["domain_key_j"])}
    neighborhood_nodes = set(core_nodes) | set(candidate_nodes)
    local = component_edges.loc[
        component_edges["is_cross_program"].to_numpy(dtype=bool)
        & component_edges["is_strong_relation"].to_numpy(dtype=bool)
        & (
            component_edges["domain_key_i"].astype(str).isin(candidate_nodes)
            | component_edges["domain_key_j"].astype(str).isin(candidate_nodes)
        )
        & (
            component_edges["domain_key_i"].astype(str).isin(neighborhood_nodes)
            | component_edges["domain_key_j"].astype(str).isin(neighborhood_nodes)
        )
    ].copy()
    if local.empty:
        return 0.0
    repeated = local["program_pair_key"].astype(str).isin(core_program_pairs)
    repeated_count = int(repeated.sum())
    if repeated_count >= 3:
        return 1.0
    if repeated_count >= 2:
        return 0.5
    return 0.0


def _evaluate_backbone_completion_gain(
    candidate_row: pd.Series,
    selected_edge_keys: set[str],
    component_edges: pd.DataFrame,
    core_nodes: set[str],
    core_programs: set[str],
    core_program_pairs: set[str],
) -> tuple[float, dict[str, object]]:
    edge_key = str(candidate_row["edge_key"])
    if edge_key in selected_edge_keys:
        return 0.0, {}

    u = str(candidate_row["domain_key_i"])
    v = str(candidate_row["domain_key_j"])
    pu = str(candidate_row["program_id_i"])
    pv = str(candidate_row["program_id_j"])
    pair_key = str(candidate_row["program_pair_key"])

    next_edge_keys = set(selected_edge_keys)
    next_edge_keys.add(edge_key)
    next_core_nodes = set(core_nodes) | {u, v}
    next_core_programs = set(core_programs) | {pu, pv}
    next_core_program_pairs = set(core_program_pairs) | {pair_key}
    candidate_nodes = {u, v}

    new_pair_novelty = 1.0 if _candidate_forms_local_multipair_skeleton(
        candidate_row=candidate_row,
        selected_edge_keys=selected_edge_keys,
        component_edges=component_edges,
        core_nodes=core_nodes,
        core_program_pairs=core_program_pairs,
    ) else 0.0
    new_program_novelty = 1.0 if _candidate_adds_anchored_new_program(
        candidate_row=candidate_row,
        component_edges=component_edges,
        core_nodes=core_nodes,
        core_programs=core_programs,
        core_program_pairs=core_program_pairs,
    ) and new_pair_novelty > 0.0 else 0.0

    pair_support_upgrade = 0.0
    if pair_key in core_program_pairs:
        current_state = _pair_support_state_from_edges(
            pair_edges=_local_backbone_pair_edges(
                component_edges=component_edges,
                pair_key=pair_key,
                edge_keys=selected_edge_keys,
                neighborhood_nodes=set(core_nodes) | candidate_nodes,
            ),
            backbone_nodes=set(core_nodes),
        )
        next_state = _pair_support_state_from_edges(
            pair_edges=_local_backbone_pair_edges(
                component_edges=component_edges,
                pair_key=pair_key,
                edge_keys=next_edge_keys,
                neighborhood_nodes=set(next_core_nodes),
            ),
            backbone_nodes=set(next_core_nodes),
        )
        pair_support_upgrade = 1.0 if (current_state == "weak" and next_state == "stable") else 0.0

    pair_redundancy_penalty = _compute_pair_redundancy_penalty(
        component_edges=component_edges,
        pair_key=pair_key,
        selected_edge_keys=selected_edge_keys,
        core_nodes=core_nodes,
        candidate_nodes=candidate_nodes,
        core_program_pairs=core_program_pairs,
    )
    dense_repeat_penalty = _compute_dense_repeat_penalty(
        candidate_row=candidate_row,
        component_edges=component_edges,
        core_nodes=core_nodes,
        core_program_pairs=core_program_pairs,
        new_pair_novelty=new_pair_novelty,
        new_program_novelty=new_program_novelty,
    )

    completion_gain = (
        1.0 * new_pair_novelty
        + 0.8 * new_program_novelty
        + 0.4 * pair_support_upgrade
        - 0.5 * pair_redundancy_penalty
        - 0.3 * dense_repeat_penalty
    )
    return float(completion_gain), {
        "edge_key": edge_key,
        "new_core_nodes": next_core_nodes,
        "new_core_programs": next_core_programs,
        "new_core_program_pairs": next_core_program_pairs,
        "distinct_program_pair_gain": float(new_pair_novelty),
        "new_program_gain": float(new_program_novelty),
        "pair_support_upgrade": float(pair_support_upgrade),
        "pair_redundancy_penalty": float(pair_redundancy_penalty),
        "dense_repeat_penalty": float(dense_repeat_penalty),
    }


def _candidate_advances_pattern_completion(
    candidate_row: pd.Series,
    selected_edge_keys: set[str],
    component_edges: pd.DataFrame,
    core_nodes: set[str],
    core_program_pairs: set[str],
    update: dict[str, object],
) -> bool:
    if float(update.get("new_program_gain", 0.0)) > 0.0:
        return True
    if float(update.get("pair_support_upgrade", 0.0)) > 0.0:
        return True
    if float(update.get("distinct_program_pair_gain", 0.0)) <= 0.0:
        return False

    edge_key = str(candidate_row["edge_key"])
    u = str(candidate_row["domain_key_i"])
    v = str(candidate_row["domain_key_j"])
    candidate_nodes = {u, v}
    next_edge_keys = set(selected_edge_keys) | {edge_key}
    neighborhood_nodes = set(core_nodes) | set(candidate_nodes)

    local_edges = component_edges.loc[
        component_edges["edge_key"].astype(str).isin(next_edge_keys)
        & component_edges["is_cross_program"].to_numpy(dtype=bool)
        & component_edges["is_strong_relation"].to_numpy(dtype=bool)
        & component_edges["domain_key_i"].astype(str).isin(neighborhood_nodes)
        & component_edges["domain_key_j"].astype(str).isin(neighborhood_nodes)
    ].copy()
    if local_edges.empty:
        return False

    local_pairs = set(local_edges["program_pair_key"].astype(str).tolist())
    if len(local_pairs) < 2:
        return False

    anchored_core_nodes: set[str] = set()
    existing_pair_seen = False
    for row in local_edges.itertuples(index=False):
        ru = str(row.domain_key_i)
        rv = str(row.domain_key_j)
        rpair = str(row.program_pair_key)
        touches_candidate = (ru in candidate_nodes) or (rv in candidate_nodes)
        if not touches_candidate:
            continue
        if ru in core_nodes:
            anchored_core_nodes.add(ru)
        if rv in core_nodes:
            anchored_core_nodes.add(rv)
        if rpair in core_program_pairs:
            existing_pair_seen = True

    return bool(existing_pair_seen and len(anchored_core_nodes) >= 2)


def _assemble_canonical_backbone(
    seed_row: pd.Series,
    component_edges_df: pd.DataFrame,
    discovery_cfg: InteractionDiscoveryConfig,
) -> dict[str, object]:
    component_edges = component_edges_df.copy()
    component_edges["_component_rank"] = _component_edge_rank(component_edges)
    selected_edge_keys = {str(seed_row["edge_key"])}
    core_nodes = {str(seed_row["domain_key_i"]), str(seed_row["domain_key_j"])}
    core_programs = {str(seed_row["program_id_i"]), str(seed_row["program_id_j"])}
    core_program_pairs = {str(seed_row["program_pair_key"])}

    while True:
        candidates = component_edges.loc[
            (~component_edges["edge_key"].astype(str).isin(selected_edge_keys))
            & (
                component_edges["domain_key_i"].astype(str).isin(core_nodes)
                | component_edges["domain_key_j"].astype(str).isin(core_nodes)
            )
        ].copy()
        if candidates.empty:
            break

        best_gain = 0.0
        best_update: dict[str, object] = {}
        best_edge_key = ""
        best_rank = -1.0
        best_row: pd.Series | None = None

        for i in range(candidates.shape[0]):
            row = candidates.iloc[i]
            pair_key = str(row["program_pair_key"])
            is_new_pair = pair_key not in core_program_pairs
            support_to_core = int(str(row["domain_key_i"]) in core_nodes) + int(str(row["domain_key_j"]) in core_nodes)

            if is_new_pair:
                candidate_nodes = {str(row["domain_key_i"]), str(row["domain_key_j"])}
                informative = (
                    _is_informative_new_pair(
                        pair_key=pair_key,
                        existing_programs=core_programs,
                        existing_program_pairs=core_program_pairs,
                        pair_support_count=_backbone_local_pair_support_count(
                            component_edges=component_edges,
                            pair_key=pair_key,
                            core_nodes=core_nodes,
                            candidate_nodes=candidate_nodes,
                        ),
                    )
                    or _backbone_forms_minimal_multipair_skeleton(
                        component_edges=component_edges,
                        pair_key=pair_key,
                        core_nodes=core_nodes,
                        candidate_nodes=candidate_nodes,
                        core_program_pairs=core_program_pairs,
                    )
                )
                if not informative:
                    continue
            else:
                if support_to_core < 2:
                    continue

            gain, update = _evaluate_backbone_completion_gain(
                candidate_row=row,
                selected_edge_keys=selected_edge_keys,
                component_edges=component_edges,
                core_nodes=core_nodes,
                core_programs=core_programs,
                core_program_pairs=core_program_pairs,
            )
            if gain <= 0.0:
                continue
            rank_val = float(row.get("_component_rank", 0.0))
            edge_key = str(row["edge_key"])
            if (
                gain > best_gain
                or (gain == best_gain and rank_val > best_rank)
                or (gain == best_gain and rank_val == best_rank and edge_key < best_edge_key)
            ):
                best_gain = float(gain)
                best_update = update
                best_edge_key = edge_key
                best_rank = rank_val
                best_row = row

        if not best_update or best_gain <= 0.0 or best_row is None:
            break
        if not _candidate_advances_pattern_completion(
            candidate_row=best_row,
            selected_edge_keys=selected_edge_keys,
            component_edges=component_edges,
            core_nodes=core_nodes,
            core_program_pairs=core_program_pairs,
            update=best_update,
        ):
            break

        selected_edge_keys.add(str(best_update["edge_key"]))
        core_nodes = set(best_update["new_core_nodes"])
        core_programs = set(best_update["new_core_programs"])
        core_program_pairs = set(best_update["new_core_program_pairs"])

    backbone_edges = component_edges.loc[
        component_edges["edge_key"].astype(str).isin(selected_edge_keys)
    ].copy()
    backbone_edges = backbone_edges.sort_values(
        by=["_component_rank", "seed_score", "edge_key"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return {
        "seed_edge_key": str(seed_row["edge_key"]),
        "seed_nodes": {str(seed_row["domain_key_i"]), str(seed_row["domain_key_j"])},
        "core_nodes": set(core_nodes),
        "core_program_pairs": set(core_program_pairs),
        "backbone_edges": backbone_edges,
    }


def _collect_structure_members(
    backbone_nodes: set[str],
    backbone_program_pairs: set[str],
    backbone_region_nodes: set[str],
    member_candidate_nodes: set[str],
    local_region_edges: pd.DataFrame,
    domain_to_program: dict[str, str],
    max_rounds: int = 2,
) -> dict[str, object]:
    member_nodes = set(backbone_nodes)
    joined_via_proximity: dict[str, bool] = {}
    backbone_programs = {domain_to_program.get(str(node), "") for node in backbone_nodes if domain_to_program.get(str(node), "")}

    for _ in range(int(max(1, max_rounds))):
        newly_added: set[str] = set()
        for node in sorted(set(member_candidate_nodes) - set(member_nodes)):
            incident = local_region_edges.loc[
                (
                    local_region_edges["domain_key_i"].astype(str).eq(str(node))
                    & local_region_edges["domain_key_j"].astype(str).isin(member_nodes)
                )
                | (
                    local_region_edges["domain_key_j"].astype(str).eq(str(node))
                    & local_region_edges["domain_key_i"].astype(str).isin(member_nodes)
                )
            ].copy()
            if incident.empty:
                continue

            strong_incident = incident.loc[incident["is_strong_relation"].to_numpy(dtype=bool)].copy()
            weak_incident = incident.loc[incident["is_weak_relation"].to_numpy(dtype=bool)].copy()
            anchor_nodes = {
                str(r.domain_key_j if str(r.domain_key_i) == str(node) else r.domain_key_i)
                for r in incident.itertuples(index=False)
            }
            anchor_programs = {
                domain_to_program.get(anchor, "")
                for anchor in anchor_nodes
                if domain_to_program.get(anchor, "")
            }
            strong_backbone_anchors = {
                str(r.domain_key_j if str(r.domain_key_i) == str(node) else r.domain_key_i)
                for r in strong_incident.itertuples(index=False)
                if (str(r.domain_key_j if str(r.domain_key_i) == str(node) else r.domain_key_i)) in backbone_nodes
            }
            weak_anchor_nodes = {
                str(r.domain_key_j if str(r.domain_key_i) == str(node) else r.domain_key_i)
                for r in weak_incident.itertuples(index=False)
            }
            node_program = str(domain_to_program.get(str(node), ""))
            strong_pairs = {
                str(r.program_pair_key)
                for r in strong_incident.itertuples(index=False)
                if bool(r.is_cross_program)
            }
            unsupported_strong_pairs = {
                pair for pair in strong_pairs
                if pair not in backbone_program_pairs and _pair_programs(pair).isdisjoint(backbone_programs)
            }
            if unsupported_strong_pairs:
                continue

            has_attachment = bool(anchor_nodes)
            multi_anchor = (
                bool(strong_backbone_anchors)
                or len(weak_anchor_nodes) >= 2
                or len(anchor_programs) >= 2
            )
            if not has_attachment or not multi_anchor:
                continue
            if strong_incident.empty and (len(weak_anchor_nodes) < 2 or len(anchor_programs) < 2):
                continue
            if weak_incident.shape[0] > 0 and strong_incident.empty and len(anchor_nodes) == 1:
                continue
            if node_program and node_program not in backbone_programs and len(anchor_programs) < 2 and len(strong_backbone_anchors) == 0:
                continue

            newly_added.add(str(node))
            joined_via_proximity[str(node)] = bool(strong_incident.empty and weak_incident.shape[0] > 0)

        if not newly_added:
            break
        member_nodes |= newly_added

    return {
        "member_nodes": set(member_nodes),
        "joined_via_proximity": joined_via_proximity,
    }


def discover_interaction_structures(
    edges_df: pd.DataFrame,
    domains_df: pd.DataFrame,
    discovery_cfg: InteractionDiscoveryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prepared_edges = _prepare_interaction_edges(edges_df)
    if prepared_edges.empty:
        return _empty_interaction_structures_table(), _empty_interaction_membership_table(), {"seed_count": 0, "raw_structure_count": 0}

    seeds_df = _discover_interaction_seeds(prepared_edges, discovery_cfg)
    if seeds_df.empty:
        return _empty_interaction_structures_table(), _empty_interaction_membership_table(), {"seed_count": 0, "raw_structure_count": 0}

    domain_cols = domains_df.loc[domains_df["qc_pass"].to_numpy(dtype=bool), ["domain_key", "program_seed_id"]].copy()
    domain_to_program = dict(zip(domain_cols["domain_key"].astype(str).tolist(), domain_cols["program_seed_id"].astype(str).tolist()))

    strong_cross_edges = prepared_edges.loc[
        prepared_edges["is_cross_program"].to_numpy(dtype=bool) & prepared_edges["is_strong_relation"].to_numpy(dtype=bool)
    ].copy()
    component_to_edges: dict[int, pd.DataFrame] = {}
    if not strong_cross_edges.empty:
        g = nx.Graph()
        for r in strong_cross_edges.itertuples(index=False):
            g.add_edge(str(r.domain_key_i), str(r.domain_key_j))
        node_to_component: dict[str, int] = {}
        for comp_id, comp in enumerate(nx.connected_components(g), start=1):
            for node in comp:
                node_to_component[str(node)] = comp_id
        strong_cross_edges["component_id"] = [int(node_to_component.get(str(u), 0)) for u in strong_cross_edges["domain_key_i"].astype(str).tolist()]
        for comp_id, sub in strong_cross_edges.groupby("component_id"):
            component_to_edges[int(comp_id)] = sub.reset_index(drop=True)

    adjacency: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(prepared_edges.itertuples(index=False)):
        adjacency[str(r.domain_key_i)].append(idx)
        adjacency[str(r.domain_key_j)].append(idx)

    canonical_rows: dict[str, dict] = {}
    canonical_membership: dict[str, dict[str, dict]] = {}
    niche_idx = 0
    merged_seed_duplicates = 0

    selected_seed_count = 0
    for comp_id, comp_seeds in seeds_df.groupby("component_id"):
        component_edges_df = component_to_edges.get(int(comp_id), strong_cross_edges)
        if component_edges_df.empty:
            continue
        comp_seeds = _select_diverse_seeds_for_component(comp_seeds.reset_index(drop=True), discovery_cfg)
        selected_seed_count += int(comp_seeds.shape[0])
        produced = 0
        for seed in comp_seeds.itertuples(index=False):
            if discovery_cfg.max_patterns_per_component is not None and produced >= int(discovery_cfg.max_patterns_per_component):
                break
            seed_row = pd.Series(seed._asdict())
            region = _extract_local_region(component_edges_df, seed_row)
            backbone_region_nodes = {str(x) for x in region.get("backbone_region_nodes", set())}
            context_region_nodes = {str(x) for x in region.get("context_region_nodes", set())}
            region_edges_df = region.get("backbone_region_edges", component_edges_df.iloc[0:0].copy())
            if region_edges_df.empty or not backbone_region_nodes:
                continue
            if not bool(region_edges_df["edge_key"].astype(str).eq(str(seed_row["edge_key"])).any()):
                continue

            backbone = _assemble_canonical_backbone(seed_row, region_edges_df, discovery_cfg)
            backbone_nodes = set(str(x) for x in backbone["core_nodes"])
            backbone_program_pairs = set(str(x) for x in backbone["core_program_pairs"])
            local_region_nodes = set(backbone_region_nodes) | set(context_region_nodes)
            local_region_edges = prepared_edges.loc[
                prepared_edges["domain_key_i"].astype(str).isin(local_region_nodes)
                & prepared_edges["domain_key_j"].astype(str).isin(local_region_nodes)
            ].copy()
            member_candidate_nodes = set(local_region_nodes) - set(backbone_nodes)
            member_collection = _collect_structure_members(
                backbone_nodes=backbone_nodes,
                backbone_program_pairs=backbone_program_pairs,
                backbone_region_nodes=set(backbone_region_nodes),
                member_candidate_nodes=member_candidate_nodes,
                local_region_edges=local_region_edges,
                domain_to_program=domain_to_program,
            )
            member_nodes = set(str(x) for x in member_collection.get("member_nodes", set()))
            joined_via_proximity = {
                str(k): bool(v)
                for k, v in dict(member_collection.get("joined_via_proximity", {})).items()
            }

            internal_edges = prepared_edges.loc[
                prepared_edges["domain_key_i"].astype(str).isin(member_nodes)
                & prepared_edges["domain_key_j"].astype(str).isin(member_nodes)
            ].copy()
            if internal_edges.empty:
                continue
            strong_internal_edges = internal_edges.loc[internal_edges["is_strong_relation"].to_numpy(dtype=bool)].copy()
            backbone_edge_keys = set(backbone["backbone_edges"]["edge_key"].astype(str).tolist())
            backbone_program_pairs = {str(x) for x in backbone["backbone_edges"]["program_pair_key"].astype(str).tolist() if str(x)}
            canonical_id = _canonical_pattern_id(
                component_id=int(comp_id),
                core_nodes=backbone_nodes,
                core_pairs=backbone_program_pairs,
                backbone_edge_keys=backbone_edge_keys,
            )

            if canonical_id in canonical_rows:
                merged_seed_duplicates += 1
                existing = canonical_rows[canonical_id]
                seed_union = _signature_tokens(existing.get("seed_source_keys", "")) | {str(backbone["seed_edge_key"])}
                existing["seed_source_keys"] = _sorted_join(seed_union)
                existing["duplicate_collapsed_from_count"] = int(existing.get("duplicate_collapsed_from_count", 0)) + 1
                if (
                    int(len(member_nodes)) > int(existing.get("member_count", 0))
                    or int(len(backbone_nodes)) > int(existing.get("backbone_node_count", 0))
                ):
                    existing.update(
                        {
                            "member_count": int(len(member_nodes)),
                            "backbone_node_count": int(len(backbone_nodes)),
                            "program_count": int(len({domain_to_program.get(str(node), "") for node in member_nodes if domain_to_program.get(str(node), "")})),
                            "program_ids": _sorted_join({domain_to_program.get(str(node), "") for node in member_nodes if domain_to_program.get(str(node), "")}),
                            "backbone_program_pairs": _sorted_join(backbone_program_pairs),
                            "backbone_program_pair_count": int(len(backbone_program_pairs)),
                            "cross_program_edge_count": int(np.count_nonzero(internal_edges["is_cross_program"].to_numpy(dtype=bool))),
                            "strong_edge_count": int(strong_internal_edges.shape[0]),
                            "backbone_edge_count": int(len(backbone_edge_keys)),
                            "contact_fraction": float(np.mean(internal_edges["boundary_contact"].to_numpy(dtype=np.float64))),
                            "overlap_fraction": float(np.mean(internal_edges["overlap_contact"].to_numpy(dtype=np.float64))),
                            "proximity_fraction": float(np.mean(internal_edges["proximity_contact"].to_numpy(dtype=np.float64))),
                            "mean_edge_strength": float(np.mean(internal_edges["edge_strength"].to_numpy(dtype=np.float64))),
                            "mean_edge_reliability": float(np.mean(internal_edges["edge_reliability"].to_numpy(dtype=np.float64))),
                            "backbone_node_keys": _sorted_join(backbone_nodes),
                            "member_node_keys": _sorted_join(member_nodes),
                            "_backbone_edge_keys": _sorted_join(backbone_edge_keys),
                            "_strong_edge_signature_keys": _sorted_join(set(strong_internal_edges["edge_key"].astype(str).tolist())),
                            "_backbone_program_pair_signature_keys": _sorted_join(backbone_program_pairs),
                            "_member_domain_keys": _sorted_join(member_nodes),
                            "_internal_edge_count": int(internal_edges.shape[0]),
                            "_same_program_edge_count": int(internal_edges.shape[0] - np.count_nonzero(internal_edges["is_cross_program"].to_numpy(dtype=bool))),
                        }
                    )
                    canonical_membership[canonical_id] = {}
                    for domain_key in sorted(member_nodes):
                        is_backbone = str(domain_key) in backbone_nodes
                        canonical_membership[canonical_id][str(domain_key)] = {
                            "niche_id": existing["niche_id"],
                            "domain_key": str(domain_key),
                            "program_id": str(domain_to_program.get(str(domain_key), "")),
                            "is_backbone_member": bool(is_backbone),
                            "is_structure_member": True,
                            "joined_via_proximity": bool(joined_via_proximity.get(str(domain_key), False)),
                            "seed_provenance": "backbone" if is_backbone else ("member_collection:mixed" if joined_via_proximity.get(str(domain_key), False) else "member_collection:attached"),
                        }
                continue

            niche_idx += 1
            produced += 1
            niche_id = f"N{niche_idx:06d}"
            program_ids = {domain_to_program.get(str(node), "") for node in member_nodes if domain_to_program.get(str(node), "")}
            row = {
                "niche_id": niche_id,
                "canonical_pattern_id": canonical_id,
                "component_id": int(comp_id),
                "member_count": int(len(member_nodes)),
                "backbone_node_count": int(len(backbone_nodes)),
                "program_count": int(len(program_ids)),
                "program_ids": _sorted_join(program_ids),
                "backbone_program_pairs": _sorted_join(backbone_program_pairs),
                "backbone_program_pair_count": int(len(backbone_program_pairs)),
                "cross_program_edge_count": int(np.count_nonzero(internal_edges["is_cross_program"].to_numpy(dtype=bool))),
                "strong_edge_count": int(strong_internal_edges.shape[0]),
                "backbone_edge_count": int(len(backbone_edge_keys)),
                "contact_fraction": float(np.mean(internal_edges["boundary_contact"].to_numpy(dtype=np.float64))),
                "overlap_fraction": float(np.mean(internal_edges["overlap_contact"].to_numpy(dtype=np.float64))),
                "proximity_fraction": float(np.mean(internal_edges["proximity_contact"].to_numpy(dtype=np.float64))),
                "mean_edge_strength": float(np.mean(internal_edges["edge_strength"].to_numpy(dtype=np.float64))),
                "mean_edge_reliability": float(np.mean(internal_edges["edge_reliability"].to_numpy(dtype=np.float64))),
                "basic_qc_pass": False,
                "basic_qc_fail_reason": "",
                "interaction_confidence": 0.0,
                "duplicate_collapsed_from_count": 0,
                "backbone_node_keys": _sorted_join(backbone_nodes),
                "member_node_keys": _sorted_join(member_nodes),
                "seed_source_keys": str(backbone["seed_edge_key"]),
                "_backbone_edge_keys": _sorted_join(backbone_edge_keys),
                "_strong_edge_signature_keys": _sorted_join(set(strong_internal_edges["edge_key"].astype(str).tolist())),
                "_backbone_program_pair_signature_keys": _sorted_join(backbone_program_pairs),
                "_member_domain_keys": _sorted_join(member_nodes),
                "_internal_edge_count": int(internal_edges.shape[0]),
                "_same_program_edge_count": int(internal_edges.shape[0] - np.count_nonzero(internal_edges["is_cross_program"].to_numpy(dtype=bool))),
            }
            canonical_rows[canonical_id] = row
            canonical_membership[canonical_id] = {}
            seed_nodes = set(str(x) for x in backbone["seed_nodes"])
            for domain_key in sorted(member_nodes):
                is_backbone = str(domain_key) in backbone_nodes
                if str(domain_key) in seed_nodes:
                    provenance = f"seed:{backbone['seed_edge_key']}"
                elif is_backbone:
                    provenance = "backbone"
                else:
                    provenance = "member_collection:mixed" if joined_via_proximity.get(str(domain_key), False) else "member_collection:attached"
                canonical_membership[canonical_id][str(domain_key)] = {
                    "niche_id": niche_id,
                    "domain_key": str(domain_key),
                    "program_id": str(domain_to_program.get(str(domain_key), "")),
                    "is_backbone_member": bool(is_backbone),
                    "is_structure_member": True,
                    "joined_via_proximity": bool(joined_via_proximity.get(str(domain_key), False)),
                    "seed_provenance": str(provenance),
                }

    rows = list(canonical_rows.values())
    mem_rows = [member for members in canonical_membership.values() for member in members.values()]
    structures_df = pd.DataFrame(rows) if rows else _empty_interaction_structures_table()
    membership_df = pd.DataFrame(mem_rows) if mem_rows else _empty_interaction_membership_table()
    return structures_df, membership_df, {
        "seed_count": int(selected_seed_count),
        "raw_structure_count": int(structures_df.shape[0]),
        "seed_duplicate_merge_count": int(merged_seed_duplicates),
    }


def _randomize_component_edges_strong_cross(prepared_edges: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    strong_cross = prepared_edges.loc[
        prepared_edges["is_cross_program"].to_numpy(dtype=bool)
        & prepared_edges["is_strong_relation"].to_numpy(dtype=bool)
    ].copy()
    if strong_cross.empty:
        return strong_cross

    domain_to_program: dict[str, str] = {}
    domain_to_id: dict[str, object] = {}
    for row in prepared_edges.itertuples(index=False):
        ui = str(row.domain_key_i)
        uj = str(row.domain_key_j)
        domain_to_program[ui] = str(row.program_id_i)
        domain_to_program[uj] = str(row.program_id_j)
        domain_to_id[ui] = getattr(row, "domain_id_i", ui)
        domain_to_id[uj] = getattr(row, "domain_id_j", uj)

    j_nodes = strong_cross["domain_key_j"].astype(str).to_numpy()
    if j_nodes.size <= 1:
        return strong_cross.reset_index(drop=True)
    permuted = rng.permutation(j_nodes)
    randomized = strong_cross.copy().reset_index(drop=True)
    randomized["domain_key_j"] = permuted
    randomized["domain_id_j"] = [domain_to_id.get(str(k), str(k)) for k in randomized["domain_key_j"].astype(str).tolist()]
    randomized["program_id_i"] = [domain_to_program.get(str(k), "") for k in randomized["domain_key_i"].astype(str).tolist()]
    randomized["program_id_j"] = [domain_to_program.get(str(k), "") for k in randomized["domain_key_j"].astype(str).tolist()]
    randomized["domain_pair_key"] = [
        _relation_edge_key(ki, kj)
        for ki, kj in zip(randomized["domain_key_i"].astype(str).tolist(), randomized["domain_key_j"].astype(str).tolist())
    ]
    randomized["edge_key"] = randomized["domain_pair_key"].astype(str)
    randomized["program_pair_key"] = [
        _program_pair_key(pi, pj)
        for pi, pj in zip(randomized["program_id_i"].astype(str).tolist(), randomized["program_id_j"].astype(str).tolist())
    ]
    randomized["is_cross_program"] = randomized["program_id_i"].astype(str).to_numpy() != randomized["program_id_j"].astype(str).to_numpy()
    randomized["is_strong_relation"] = True
    randomized = randomized.loc[
        randomized["domain_key_i"].astype(str).to_numpy() != randomized["domain_key_j"].astype(str).to_numpy()
    ].copy()
    randomized = randomized.drop_duplicates(subset=["edge_key"], keep="first").reset_index(drop=True)
    return randomized


def _structure_random_signature(row: pd.Series) -> dict:
    return {
        "backbone_pair_sig": set(_signature_tokens(row.get("_backbone_program_pair_signature_keys", ""))),
        "backbone_edge_count": int(row.get("backbone_edge_count", 0)),
        "program_count": int(row.get("program_count", 0)),
    }


def _random_structure_matches(observed_sig: dict, random_row: pd.Series, cfg: RandomBaselineConfig) -> bool:
    random_sig = _structure_random_signature(random_row)
    pair_overlap = _jaccard(
        set(observed_sig.get("backbone_pair_sig", set())),
        set(random_sig.get("backbone_pair_sig", set())),
    )
    return bool(
        pair_overlap >= float(np.clip(cfg.min_pair_signature_overlap, 0.0, 1.0))
        and abs(int(observed_sig.get("backbone_edge_count", 0)) - int(random_sig.get("backbone_edge_count", 0)))
        <= int(max(0, cfg.max_backbone_edge_count_diff))
        and abs(int(observed_sig.get("program_count", 0)) - int(random_sig.get("program_count", 0)))
        <= int(max(0, cfg.max_program_count_diff))
    )


def apply_random_baseline_filter(
    passing_structures_df: pd.DataFrame,
    passing_membership_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    domains_df: pd.DataFrame,
    discovery_cfg: InteractionDiscoveryConfig,
    basic_filter_cfg: BasicNicheFilterConfig,
    random_cfg: RandomBaselineConfig,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    input_count = int(passing_structures_df.shape[0])
    passthrough_structures = passing_structures_df.copy()
    if "random_occurrence_rate" not in passthrough_structures.columns:
        passthrough_structures["random_occurrence_rate"] = np.full(passthrough_structures.shape[0], np.nan, dtype=np.float64)
    if "non_random_score" not in passthrough_structures.columns:
        passthrough_structures["non_random_score"] = np.full(passthrough_structures.shape[0], np.nan, dtype=np.float64)
    if "random_qc_pass" not in passthrough_structures.columns:
        passthrough_structures["random_qc_pass"] = np.ones(passthrough_structures.shape[0], dtype=bool)
    if passing_structures_df.empty:
        return passthrough_structures, passing_membership_df.copy(), {
            "n_iter": int(random_cfg.n_iter),
            "input_structure_count": 0,
            "passed_random_filter_count": 0,
            "failed_random_filter_count": 0,
            "mean_non_random_score": float("nan"),
            "median_non_random_score": float("nan"),
        }

    observed = passing_structures_df.copy().reset_index(drop=True)
    observed_ids = observed["niche_id"].astype(str).tolist()
    observed_sigs = {str(row["niche_id"]): _structure_random_signature(row) for _, row in observed.iterrows()}
    hit_counts = {nid: 0 for nid in observed_ids}
    prepared_edges = _prepare_interaction_edges(edges_df)
    strong_cross_mask = prepared_edges["is_cross_program"].to_numpy(dtype=bool) & prepared_edges["is_strong_relation"].to_numpy(dtype=bool)
    other_edges = prepared_edges.loc[~strong_cross_mask].copy()
    rng = np.random.default_rng(int(random_seed) + int(random_cfg.rng_seed_offset))

    for _ in range(int(max(0, random_cfg.n_iter))):
        randomized_strong = _randomize_component_edges_strong_cross(prepared_edges, rng)
        random_edges = pd.concat([other_edges, randomized_strong], ignore_index=True, sort=False)
        random_structures, _, _ = discover_interaction_structures(
            edges_df=random_edges,
            domains_df=domains_df,
            discovery_cfg=discovery_cfg,
        )
        random_filtered = apply_basic_niche_filter(
            structures_df=random_structures,
            filter_cfg=basic_filter_cfg,
        )
        random_passing = random_filtered.loc[
            random_filtered["basic_qc_pass"].to_numpy(dtype=bool)
        ].reset_index(drop=True) if not random_filtered.empty else random_filtered.iloc[0:0].copy()
        if random_passing.empty:
            continue
        for nid in observed_ids:
            sig = observed_sigs[nid]
            matched = False
            for _, random_row in random_passing.iterrows():
                if _random_structure_matches(sig, random_row, random_cfg):
                    matched = True
                    break
            if matched:
                hit_counts[nid] += 1

    denom = float(max(1, int(random_cfg.n_iter)))
    random_occurrence_rate = np.asarray([float(hit_counts[nid]) / denom for nid in observed_ids], dtype=np.float64)
    non_random_score = np.clip(1.0 - random_occurrence_rate, 0.0, 1.0)
    random_qc_pass = non_random_score >= float(np.clip(random_cfg.min_non_random_score, 0.0, 1.0))
    observed["random_occurrence_rate"] = random_occurrence_rate
    observed["non_random_score"] = non_random_score
    observed["random_qc_pass"] = random_qc_pass
    kept_ids = set(observed.loc[observed["random_qc_pass"].to_numpy(dtype=bool), "niche_id"].astype(str).tolist())
    filtered_structures = observed.loc[observed["niche_id"].astype(str).isin(kept_ids)].reset_index(drop=True)
    filtered_membership = passing_membership_df.loc[
        passing_membership_df["niche_id"].astype(str).isin(kept_ids)
    ].reset_index(drop=True) if kept_ids else passing_membership_df.iloc[0:0].copy()
    return filtered_structures, filtered_membership, {
        "n_iter": int(random_cfg.n_iter),
        "input_structure_count": int(input_count),
        "passed_random_filter_count": int(filtered_structures.shape[0]),
        "failed_random_filter_count": int(input_count - filtered_structures.shape[0]),
        "mean_non_random_score": float(np.mean(non_random_score)) if non_random_score.size > 0 else float("nan"),
        "median_non_random_score": float(np.median(non_random_score)) if non_random_score.size > 0 else float("nan"),
    }


def apply_basic_niche_filter(structures_df: pd.DataFrame, filter_cfg: BasicNicheFilterConfig) -> pd.DataFrame:
    out = structures_df.copy()
    if out.empty:
        out["basic_qc_pass"] = pd.Series(dtype=bool)
        out["basic_qc_fail_reason"] = pd.Series(dtype=object)
        out["interaction_confidence"] = pd.Series(dtype=np.float64)
        return out

    strong_backbone_present = out["backbone_node_count"].to_numpy(dtype=np.int64) >= 2
    program_count = out["program_count"].to_numpy(dtype=np.int64)
    cross_program_edge_count = out["cross_program_edge_count"].to_numpy(dtype=np.int64)
    strong_edge_count = out["strong_edge_count"].to_numpy(dtype=np.int64)
    proximity_fraction = np.clip(out["proximity_fraction"].to_numpy(dtype=np.float64), 0.0, 1.0)
    mean_edge_strength = np.clip(out["mean_edge_strength"].to_numpy(dtype=np.float64), 0.0, 1.0)
    mean_edge_reliability = np.clip(out["mean_edge_reliability"].to_numpy(dtype=np.float64), 0.0, 1.0)
    backbone_pair_count = np.asarray([len(_signature_tokens(v)) for v in out["backbone_program_pairs"].astype(str).tolist()], dtype=np.float64)
    backbone_edge_count = out["backbone_edge_count"].to_numpy(dtype=np.int64) if "backbone_edge_count" in out.columns else out["strong_edge_count"].to_numpy(dtype=np.int64)
    internal_edge_series = out["_internal_edge_count"] if "_internal_edge_count" in out.columns else pd.Series(np.zeros(out.shape[0]), index=out.index)
    same_program_edge_series = out["_same_program_edge_count"] if "_same_program_edge_count" in out.columns else pd.Series(np.zeros(out.shape[0]), index=out.index)
    internal_edge_count = pd.to_numeric(internal_edge_series, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    same_program_edge_count = pd.to_numeric(same_program_edge_series, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    same_program_edge_fraction = np.divide(
        same_program_edge_count,
        np.maximum(1.0, internal_edge_count),
        out=np.zeros_like(same_program_edge_count, dtype=np.float64),
        where=np.maximum(1.0, internal_edge_count) > 0,
    )
    pair_member_ratio = np.divide(
        backbone_pair_count,
        np.maximum(1.0, out["member_count"].to_numpy(dtype=np.float64)),
        out=np.zeros_like(backbone_pair_count, dtype=np.float64),
        where=np.maximum(1.0, out["member_count"].to_numpy(dtype=np.float64)) > 0,
    )

    confidence = np.clip(
        0.30 * mean_edge_strength
        + 0.30 * mean_edge_reliability
        + 0.15 * np.clip(cross_program_edge_count / 3.0, 0.0, 1.0)
        + 0.15 * np.clip(backbone_pair_count / np.maximum(1.0, out["program_count"].to_numpy(dtype=np.float64)), 0.0, 1.0)
        + 0.10 * np.clip(backbone_edge_count / 3.0, 0.0, 1.0),
        0.0,
        1.0,
    )

    basic_pass = np.ones(out.shape[0], dtype=bool)
    reasons: list[str] = []
    for i in range(out.shape[0]):
        fail: list[str] = []
        if int(program_count[i]) < int(max(2, filter_cfg.min_program_count)):
            fail.append("insufficient_program_count")
        if (not bool(strong_backbone_present[i])) or int(strong_edge_count[i]) <= 0:
            fail.append("missing_strong_backbone")
        if int(cross_program_edge_count[i]) < int(max(1, filter_cfg.min_cross_program_edges)):
            fail.append("cross_program_support_insufficient")
        if int(backbone_pair_count[i]) < int(max(1, filter_cfg.min_core_program_pair_count)):
            fail.append("backbone_program_pair_richness_too_low")
        if float(mean_edge_reliability[i]) < float(np.clip(filter_cfg.min_mean_edge_reliability, 0.0, 1.0)):
            fail.append("mean_edge_reliability_too_low")
        if int(strong_edge_count[i]) <= 0 and float(proximity_fraction[i]) >= 1.0:
            fail.append("proximity_only_pattern")
        if float(same_program_edge_fraction[i]) > float(np.clip(filter_cfg.max_same_program_edge_fraction, 0.0, 1.0)):
            fail.append("same_program_dominance_too_high")
        if int(out["member_count"].iloc[i]) >= 4 and float(pair_member_ratio[i]) < float(max(0.0, filter_cfg.min_core_pair_member_ratio)):
            fail.append("large_structure_backbone_pair_thin")
        if float(confidence[i]) < float(np.clip(filter_cfg.min_interaction_confidence, 0.0, 1.0)):
            fail.append("interaction_confidence_too_low")
        basic_pass[i] = len(fail) == 0
        reasons.append(";".join(fail))

    out["basic_qc_pass"] = basic_pass
    out["basic_qc_fail_reason"] = np.asarray(reasons, dtype=object)
    out["interaction_confidence"] = confidence
    return out


def deduplicate_interaction_structures(
    structures_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    dedup_cfg: InteractionDedupConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if structures_df.empty:
        return structures_df.copy(), membership_df.copy(), {
            "input_structure_count": 0,
            "retained_structure_count": 0,
            "dropped_structure_count": 0,
            "dropped_niche_ids": [],
            "dropped_against": {},
        }

    scored = structures_df.copy()
    pair_completeness = np.divide(
        pd.to_numeric(scored.get("backbone_program_pair_count", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float64),
        np.maximum(1.0, pd.to_numeric(scored.get("program_count", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float64)),
        out=np.zeros(scored.shape[0], dtype=np.float64),
        where=np.maximum(1.0, pd.to_numeric(scored.get("program_count", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float64)) > 0,
    )
    scored["_representative_score"] = np.clip(
        0.55 * pd.to_numeric(scored.get("interaction_confidence", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        + 0.30 * pd.to_numeric(scored.get("mean_edge_reliability", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        + 0.15 * np.clip(pair_completeness, 0.0, 1.0),
        0.0,
        1.0,
    )
    ordered = scored.sort_values(
        by=["_representative_score", "interaction_confidence", "strong_edge_count", "member_count", "niche_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    kept_rows: list[pd.Series] = []
    dropped_ids: list[str] = []
    dropped_against: dict[str, str] = {}
    duplicate_counts: dict[str, int] = defaultdict(int)

    for _, row in ordered.iterrows():
        row_backbone = _signature_tokens(row.get("_backbone_edge_keys", ""))
        row_core = _signature_tokens(row.get("backbone_node_keys", ""))
        row_strong_sig = _signature_tokens(row.get("_strong_edge_signature_keys", ""))
        row_pair_sig = _signature_tokens(row.get("_backbone_program_pair_signature_keys", ""))
        row_rep = float(row.get("_representative_score", 0.0))
        drop = False
        for kept in kept_rows:
            kept_backbone = _signature_tokens(kept.get("_backbone_edge_keys", ""))
            kept_core = _signature_tokens(kept.get("backbone_node_keys", ""))
            kept_strong_sig = _signature_tokens(kept.get("_strong_edge_signature_keys", ""))
            kept_pair_sig = _signature_tokens(kept.get("_backbone_program_pair_signature_keys", ""))

            backbone_overlap = _jaccard(row_backbone, kept_backbone)
            core_overlap = _jaccard(row_core, kept_core)
            strong_sig_overlap = _jaccard(row_strong_sig, kept_strong_sig)
            pair_sig_overlap = _jaccard(row_pair_sig, kept_pair_sig)
            exact_canonical = str(row.get("canonical_pattern_id", "")) and str(row.get("canonical_pattern_id", "")) == str(kept.get("canonical_pattern_id", ""))
            near_duplicate = (
                backbone_overlap >= float(np.clip(dedup_cfg.backbone_overlap_threshold, 0.0, 1.0))
                and core_overlap >= float(np.clip(dedup_cfg.core_overlap_threshold, 0.0, 1.0))
                and (
                    strong_sig_overlap >= float(np.clip(dedup_cfg.strong_edge_signature_overlap_threshold, 0.0, 1.0))
                    or pair_sig_overlap >= float(np.clip(dedup_cfg.core_program_pair_overlap_threshold, 0.0, 1.0))
                )
            )
            exact_duplicate = (
                core_overlap >= 0.999
                and pair_sig_overlap >= 0.999
                and (strong_sig_overlap >= 0.95 or backbone_overlap >= 0.95)
            )
            if (exact_canonical or exact_duplicate) or (
                near_duplicate and (
                    row_rep + float(max(0.0, dedup_cfg.representative_margin)) < float(kept.get("_representative_score", 0.0))
                )
            ):
                drop = True
                dropped_ids.append(str(row["niche_id"]))
                dropped_against[str(row["niche_id"])] = str(kept["niche_id"])
                duplicate_counts[str(kept["niche_id"])] += int(1 + int(row.get("duplicate_collapsed_from_count", 0)))
                break
        if not drop:
            kept_rows.append(row)

    deduped_structures = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else structures_df.iloc[0:0].copy()
    if not deduped_structures.empty:
        deduped_structures["duplicate_collapsed_from_count"] = [
            int(row.get("duplicate_collapsed_from_count", 0)) + int(duplicate_counts.get(str(row.get("niche_id", "")), 0))
            for _, row in deduped_structures.iterrows()
        ]
        if "_representative_score" in deduped_structures.columns:
            deduped_structures = deduped_structures.drop(columns=["_representative_score"])
    keep_ids = set(deduped_structures["niche_id"].astype(str).tolist()) if not deduped_structures.empty else set()
    deduped_membership = membership_df.loc[membership_df["niche_id"].astype(str).isin(keep_ids)].reset_index(drop=True)
    return deduped_structures, deduped_membership, {
        "input_structure_count": int(structures_df.shape[0]),
        "retained_structure_count": int(deduped_structures.shape[0]),
        "dropped_structure_count": int(len(dropped_ids)),
        "dropped_niche_ids": dropped_ids,
        "dropped_against": dropped_against,
        "duplicate_collapsed_from_count": {nid: int(cnt) for nid, cnt in duplicate_counts.items()},
    }


def finalize_interaction_structure_outputs(
    structures_df: pd.DataFrame,
    membership_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal_cols = [
        c
        for c in structures_df.columns
        if c.startswith("_signature")
        or c.startswith("_internal")
        or c in {
            "_backbone_edge_keys",
            "_strong_edge_signature_keys",
            "_backbone_program_pair_signature_keys",
            "_member_domain_keys",
            "_same_program_edge_count",
            "_representative_score",
        }
    ]
    trimmed_structures = structures_df.drop(columns=internal_cols, errors="ignore").copy()
    structure_cols = [
        "niche_id",
        "canonical_pattern_id",
        "component_id",
        "member_count",
        "backbone_node_count",
        "program_count",
        "program_ids",
        "backbone_program_pairs",
        "backbone_program_pair_count",
        "cross_program_edge_count",
        "strong_edge_count",
        "backbone_edge_count",
        "contact_fraction",
        "overlap_fraction",
        "proximity_fraction",
        "mean_edge_strength",
        "mean_edge_reliability",
        "basic_qc_pass",
        "basic_qc_fail_reason",
        "interaction_confidence",
        "random_occurrence_rate",
        "non_random_score",
        "random_qc_pass",
        "duplicate_collapsed_from_count",
        "backbone_node_keys",
        "member_node_keys",
        "seed_source_keys",
    ]
    membership_cols = [
        "niche_id",
        "domain_key",
        "program_id",
        "is_backbone_member",
        "is_structure_member",
        "joined_via_proximity",
        "seed_provenance",
    ]
    out_struct = trimmed_structures.loc[:, [c for c in structure_cols if c in trimmed_structures.columns]].copy() if not trimmed_structures.empty else _empty_interaction_structures_table().loc[:, structure_cols].copy()
    out_mem = membership_df.loc[:, [c for c in membership_cols if c in membership_df.columns]].copy() if not membership_df.empty else _empty_interaction_membership_table().copy()
    return out_struct, out_mem


def build_niche_report(
    sample_id: str,
    edges_df: pd.DataFrame,
    structures_df: pd.DataFrame,
    edge_meta: dict,
    cfg: NichePipelineConfig,
    discovery_summary: dict | None = None,
    random_summary: dict | None = None,
    dedup_summary: dict | None = None,
) -> dict:
    edge_contact = edges_df["is_strong_contact"].to_numpy(dtype=bool) if (not edges_df.empty and "is_strong_contact" in edges_df.columns) else np.zeros(edges_df.shape[0], dtype=bool)
    edge_overlap = edges_df["is_strong_overlap"].to_numpy(dtype=bool) if (not edges_df.empty and "is_strong_overlap" in edges_df.columns) else np.zeros(edges_df.shape[0], dtype=bool)
    edge_strong = edge_contact | edge_overlap
    edge_weak = edges_df["proximity_contact"].to_numpy(dtype=bool) if (not edges_df.empty and "proximity_contact" in edges_df.columns) else np.zeros(edges_df.shape[0], dtype=bool)
    edge_cross = (
        edges_df["program_id_i"].astype(str).to_numpy() != edges_df["program_id_j"].astype(str).to_numpy()
    ) if (not edges_df.empty and "program_id_i" in edges_df.columns and "program_id_j" in edges_df.columns) else np.zeros(edges_df.shape[0], dtype=bool)

    return {
        "sample_id": str(sample_id),
        "inputs_summary": {
            "domain_edge_count": int(edges_df.shape[0]),
            "strong_relation_edge_count": int(np.count_nonzero(edge_strong)),
            "weak_relation_edge_count": int(np.count_nonzero(edge_weak)),
            "cross_program_strong_edge_count": int(np.count_nonzero(edge_strong & edge_cross)),
        },
        "interaction_structure_summary": {
            "discovered_structure_count": int((discovery_summary or {}).get("raw_structure_count", structures_df.shape[0])),
            "retained_structure_count": int(structures_df.shape[0]),
            "member_count_quantiles": quantiles(structures_df["member_count"].to_numpy(dtype=np.float64) if (not structures_df.empty and "member_count" in structures_df.columns) else np.asarray([], dtype=np.float64)),
            "backbone_node_count_quantiles": quantiles(structures_df["backbone_node_count"].to_numpy(dtype=np.float64) if (not structures_df.empty and "backbone_node_count" in structures_df.columns) else np.asarray([], dtype=np.float64)),
            "program_count_quantiles": quantiles(structures_df["program_count"].to_numpy(dtype=np.float64) if (not structures_df.empty and "program_count" in structures_df.columns) else np.asarray([], dtype=np.float64)),
            "backbone_program_pair_count_quantiles": quantiles(structures_df["backbone_program_pair_count"].to_numpy(dtype=np.float64) if (not structures_df.empty and "backbone_program_pair_count" in structures_df.columns) else np.asarray([], dtype=np.float64)),
            "backbone_edge_count_quantiles": quantiles(structures_df["backbone_edge_count"].to_numpy(dtype=np.float64) if (not structures_df.empty and "backbone_edge_count" in structures_df.columns) else np.asarray([], dtype=np.float64)),
            "strong_edge_count_quantiles": quantiles(structures_df["strong_edge_count"].to_numpy(dtype=np.float64) if (not structures_df.empty and "strong_edge_count" in structures_df.columns) else np.asarray([], dtype=np.float64)),
            "interaction_confidence_quantiles": quantiles(structures_df["interaction_confidence"].to_numpy(dtype=np.float64) if (not structures_df.empty and "interaction_confidence" in structures_df.columns) else np.asarray([], dtype=np.float64)),
            "mean_edge_reliability_quantiles": quantiles(structures_df["mean_edge_reliability"].to_numpy(dtype=np.float64) if (not structures_df.empty and "mean_edge_reliability" in structures_df.columns) else np.asarray([], dtype=np.float64)),
        },
        "random_baseline_summary": random_summary or {},
        "edge_graph_summary": {
            "epsilon_distance": float(edge_meta.get("epsilon_distance", 0.0)),
            "epsilon_source": str(edge_meta.get("epsilon_source", "")),
            "spot_spacing_estimate": float(edge_meta.get("spot_spacing_estimate", float("nan"))),
            "proximity_distance_source": str(edge_meta.get("proximity_distance_source", "")),
            "strong_contact_threshold": float(edge_meta.get("strong_contact_threshold", 0.0)),
            "strong_overlap_threshold": float(edge_meta.get("strong_overlap_threshold", 0.0)),
            "domain_reliability_enabled": bool(edge_meta.get("domain_reliability_enabled", False)),
            "domain_reliability_pair_mode": str(edge_meta.get("domain_reliability_pair_mode", "")),
            "edge_reliability_mean": float(edge_meta.get("domain_reliability_mean", float("nan"))),
        },
        "deduplication_summary": dedup_summary or {},
        "filters": {
            "seed_score_quantile": float(cfg.discovery.seed_score_quantile),
            "backbone_rank_quantile": float(cfg.discovery.backbone_rank_quantile),
            "expansion_rank_quantile": float(cfg.discovery.expansion_rank_quantile),
            "min_program_count": int(cfg.basic_filter.min_program_count),
            "min_cross_program_edges": int(cfg.basic_filter.min_cross_program_edges),
            "min_core_program_pair_count": int(cfg.basic_filter.min_core_program_pair_count),
            "min_mean_edge_reliability": float(cfg.basic_filter.min_mean_edge_reliability),
            "min_interaction_confidence": float(cfg.basic_filter.min_interaction_confidence),
            "max_same_program_edge_fraction": float(cfg.basic_filter.max_same_program_edge_fraction),
            "min_core_pair_member_ratio": float(cfg.basic_filter.min_core_pair_member_ratio),
            "dedup_backbone_overlap_threshold": float(cfg.dedup.backbone_overlap_threshold),
            "dedup_core_overlap_threshold": float(cfg.dedup.core_overlap_threshold),
            "dedup_strong_edge_signature_overlap_threshold": float(cfg.dedup.strong_edge_signature_overlap_threshold),
            "dedup_core_program_pair_overlap_threshold": float(cfg.dedup.core_program_pair_overlap_threshold),
            "dedup_representative_margin": float(cfg.dedup.representative_margin),
            "random_baseline_n_iter": int(cfg.random_baseline.n_iter),
            "random_baseline_min_non_random_score": float(cfg.random_baseline.min_non_random_score),
        },
    }
