import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def _distance_to_weight(distances: np.ndarray) -> np.ndarray:
    """Convert distances to smooth positive edge weights."""
    if distances.size == 0:
        return np.array([], dtype=np.float32)

    positive = distances[distances > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-6)
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    return weights.astype(np.float32, copy=False)


def _cosine_distance_to_weight(distances: np.ndarray) -> np.ndarray:
    """Map cosine distance [0,2] to similarity-like weight [0,1]."""
    sims = 1.0 - distances
    return np.clip((sims + 1.0) / 2.0, 0.0, 1.0).astype(np.float32, copy=False)


def _build_spatial_knn(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cells = int(coords.shape[0])
    if n_cells <= 1:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            np.empty((n_cells, 0), dtype=np.int64),
        )

    k_query = max(2, min(int(k) + 1, n_cells))
    nbrs = NearestNeighbors(n_neighbors=k_query).fit(coords)
    dists, idx = nbrs.kneighbors(coords)

    src = np.repeat(np.arange(n_cells), idx.shape[1])
    dst = idx.reshape(-1)
    dist = dists.reshape(-1)

    mask = src != dst
    src = src[mask].astype(np.int64, copy=False)
    dst = dst[mask].astype(np.int64, copy=False)
    dist = dist[mask].astype(np.float32, copy=False)

    # Candidate list per node (exclude self).
    knn_candidates = []
    for i in range(n_cells):
        row = idx[i]
        row = row[row != i]
        knn_candidates.append(row.astype(np.int64, copy=False))

    max_len = max((len(x) for x in knn_candidates), default=0)
    padded = np.full((n_cells, max_len), -1, dtype=np.int64)
    for i, row in enumerate(knn_candidates):
        padded[i, : len(row)] = row

    return src, dst, dist, padded


def _build_spatial_radius(coords: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cells = int(coords.shape[0])
    if n_cells <= 1:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    nbrs = NearestNeighbors().fit(coords)
    dist_list, idx_list = nbrs.radius_neighbors(coords, radius=radius, return_distance=True)

    src_parts = []
    dst_parts = []
    dist_parts = []
    for i in range(n_cells):
        idx = idx_list[i]
        dist = dist_list[i]
        mask = idx != i
        idx = idx[mask]
        dist = dist[mask]
        if idx.size == 0:
            continue
        src_parts.append(np.full(idx.shape[0], i, dtype=np.int64))
        dst_parts.append(idx.astype(np.int64, copy=False))
        dist_parts.append(dist.astype(np.float32, copy=False))

    if not src_parts:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    return np.concatenate(src_parts), np.concatenate(dst_parts), np.concatenate(dist_parts)


def _estimate_radius_from_knn(coords: np.ndarray, k_ref: int) -> float:
    n_cells = int(coords.shape[0])
    if n_cells <= 1:
        return 0.0

    k_query = max(2, min(int(k_ref) + 1, n_cells))
    nbrs = NearestNeighbors(n_neighbors=k_query).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    valid = dists[:, 1:].reshape(-1)
    valid = valid[valid > 0]
    if valid.size == 0:
        return 0.0
    return float(np.percentile(valid, 90))


def _resolve_gating_sets(
    coords: np.ndarray,
    params,
    spatial_knn_padded: np.ndarray,
) -> tuple[list[set[int]], float | None, int | None]:
    n_cells = int(coords.shape[0])
    mode = getattr(params, "expr_gating_mode", "spatial_knn")

    gating_knn = None
    gating_k = None
    if mode in {"spatial_knn", "hybrid"}:
        gating_k = int(getattr(params, "expr_gating_k", 0) or getattr(params, "n_neighbors", 11))
        if spatial_knn_padded.shape[1] == 0 or gating_k > spatial_knn_padded.shape[1]:
            _, _, _, gating_knn = _build_spatial_knn(coords, gating_k)
        else:
            gating_knn = spatial_knn_padded[:, :gating_k]

    gating_radius = None
    if mode in {"radius", "hybrid"}:
        gating_radius = getattr(params, "expr_gating_radius", None)
        if gating_radius is None or gating_radius <= 0:
            spatial_radius = getattr(params, "spatial_radius", None)
            if spatial_radius is not None and spatial_radius > 0:
                gating_radius = float(spatial_radius)
            else:
                k_ref = int(gating_k or getattr(params, "n_neighbors", 11))
                gating_radius = _estimate_radius_from_knn(coords, k_ref)

    gating_sets: list[set[int]] = []
    for i in range(n_cells):
        s = set()
        if gating_knn is not None and gating_knn.shape[1] > 0:
            vals = gating_knn[i]
            s.update(int(x) for x in vals[vals >= 0])
        gating_sets.append(s)

    return gating_sets, gating_radius, gating_k


def _passes_gating(
    i: int,
    j: int,
    mode: str,
    gating_sets: list[set[int]],
    coords: np.ndarray,
    gating_radius: float | None,
) -> bool:
    if mode == "spatial_knn":
        return j in gating_sets[i]

    if mode == "radius":
        if gating_radius is None or gating_radius <= 0:
            return False
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        return dist <= gating_radius

    if mode == "hybrid":
        in_knn = j in gating_sets[i]
        in_radius = False
        if gating_radius is not None and gating_radius > 0:
            in_radius = float(np.linalg.norm(coords[i] - coords[j])) <= gating_radius
        return in_knn or in_radius

    raise ValueError(f"Unsupported expr_gating_mode: {mode}")


def _build_expr_edges(expr_embedding: np.ndarray, params, coords: np.ndarray, gating_sets, gating_radius):
    n_cells = int(expr_embedding.shape[0])
    if n_cells <= 1:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            0,
        )

    k_expr = max(1, min(int(getattr(params, "k_expr", 15)), n_cells - 1))
    metric = getattr(params, "expr_metric", "cosine")
    nn = NearestNeighbors(n_neighbors=k_expr + 1, metric=metric).fit(expr_embedding)
    dists, idx = nn.kneighbors(expr_embedding)

    src_parts = []
    dst_parts = []
    dist_parts = []
    total_candidates = 0
    kept = 0

    gating_mode = getattr(params, "expr_gating_mode", "spatial_knn")

    for i in range(n_cells):
        cand = idx[i]
        cand_dist = dists[i]
        mask_self = cand != i
        cand = cand[mask_self]
        cand_dist = cand_dist[mask_self]

        total_candidates += int(cand.shape[0])
        keep_idx = []
        keep_dist = []
        for j, dij in zip(cand, cand_dist, strict=False):
            j = int(j)
            if _passes_gating(i, j, gating_mode, gating_sets, coords, gating_radius):
                keep_idx.append(j)
                keep_dist.append(float(dij))

        if not keep_idx:
            continue

        kept += len(keep_idx)
        src_parts.append(np.full(len(keep_idx), i, dtype=np.int64))
        dst_parts.append(np.asarray(keep_idx, dtype=np.int64))
        dist_parts.append(np.asarray(keep_dist, dtype=np.float32))

    if not src_parts:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            total_candidates,
        )

    return (
        np.concatenate(src_parts),
        np.concatenate(dst_parts),
        np.concatenate(dist_parts),
        total_candidates,
    )


def _reduce(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    if mode == "mean":
        return float(np.mean(arr))
    return float(np.max(arr))


def _build_union_edges(spatial_edges, expr_edges, reduce_mode: str):
    s_src, s_dst, s_w = spatial_edges
    e_src, e_dst, e_w = expr_edges

    edge_map: dict[tuple[int, int], dict[str, list[float]]] = {}

    def add(src, dst, w_spatial=None, w_expr=None):
        key = (int(src), int(dst))
        payload = edge_map.setdefault(key, {"s": [], "e": []})
        if w_spatial is not None:
            payload["s"].append(float(w_spatial))
        if w_expr is not None:
            payload["e"].append(float(w_expr))

    for src, dst, w in zip(s_src, s_dst, s_w, strict=False):
        add(src, dst, w_spatial=w)
        add(dst, src, w_spatial=w)

    for src, dst, w in zip(e_src, e_dst, e_w, strict=False):
        add(src, dst, w_expr=w)
        add(dst, src, w_expr=w)

    if not edge_map:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
            {"spatial_only": 0, "expr_only": 0, "both": 0},
        )

    src_list = []
    dst_list = []
    attr_list = []
    type_counter = {"spatial_only": 0, "expr_only": 0, "both": 0}

    for (src, dst), payload in edge_map.items():
        w_s = _reduce(payload["s"], reduce_mode)
        w_e = _reduce(payload["e"], reduce_mode)

        src_list.append(src)
        dst_list.append(dst)
        attr_list.append([w_s, w_e])

        has_s = w_s > 0
        has_e = w_e > 0
        if has_s and has_e:
            type_counter["both"] += 1
        elif has_s:
            type_counter["spatial_only"] += 1
        elif has_e:
            type_counter["expr_only"] += 1

    return (
        np.asarray(src_list, dtype=np.int64),
        np.asarray(dst_list, dtype=np.int64),
        np.asarray(attr_list, dtype=np.float32),
        type_counter,
    )


def _undirected_count(src: np.ndarray, dst: np.ndarray) -> int:
    if src.size == 0:
        return 0
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    pairs = set(zip(a.tolist(), b.tolist(), strict=False))
    return len(pairs)


def construct_adjacency_matrix(adata, params, expr_embedding=None, verbose=True):
    """Build Hybrid Graph payload for GAT training.

    Hybrid graph = union(E_spatial, E_expr_gated)
    edge_attr = [w_spatial, w_expr]
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    n_cells = int(coords.shape[0])

    spatial_mode = getattr(params, "spatial_graph_mode", "knn")
    k_spatial = int(getattr(params, "n_neighbors", 11))

    if spatial_mode == "radius":
        radius = getattr(params, "spatial_radius", None)
        if radius is None or radius <= 0:
            raise ValueError("spatial_radius must be > 0 when spatial_graph_mode='radius'")
        s_src, s_dst, s_dist = _build_spatial_radius(coords, float(radius))
        # Still build knn candidates for gating defaults.
        _, _, _, knn_padded = _build_spatial_knn(coords, k_spatial)
    else:
        s_src, s_dst, s_dist, knn_padded = _build_spatial_knn(coords, k_spatial)

    if getattr(params, "weighted_adj", False):
        s_w = _distance_to_weight(s_dist)
    else:
        s_w = np.ones_like(s_dist, dtype=np.float32)

    gating_sets, gating_radius, gating_k = _resolve_gating_sets(
        coords,
        params,
        knn_padded,
    )

    if expr_embedding is None:
        raise ValueError(
            "expr_embedding is required for Hybrid Graph construction. "
            "Pass a controlled-size embedding (e.g., HVG/latent) to avoid densifying adata.X."
        )
    expr_embedding = np.asarray(expr_embedding, dtype=np.float32)

    e_src, e_dst, e_dist, expr_total_candidates = _build_expr_edges(
        expr_embedding,
        params,
        coords,
        gating_sets,
        gating_radius,
    )

    if getattr(params, "expr_metric", "cosine") == "cosine":
        e_w = _cosine_distance_to_weight(e_dist)
    else:
        e_w = _distance_to_weight(e_dist)

    u_src, u_dst, edge_attr, edge_type_counts = _build_union_edges(
        (s_src, s_dst, s_w),
        (e_src, e_dst, e_w),
        reduce_mode=getattr(params, "union_reduce", "max"),
    )

    edge_index = torch.tensor(np.vstack([u_src, u_dst]), dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    graph_meta = {
        "spatial": {
            "mode": spatial_mode,
            "k": k_spatial,
            "radius": getattr(params, "spatial_radius", None),
            "weighted": bool(getattr(params, "weighted_adj", False)),
            "edge_count_directed": int(s_src.size),
            "edge_count_undirected": int(_undirected_count(s_src, s_dst)),
        },
        "expression": {
            "k_expr": int(getattr(params, "k_expr", 15)),
            "embedding_source": getattr(params, "expr_embedding_source", "unknown"),
            "embedding_pca_dim": getattr(params, "expr_pca_dim", None),
            "metric": getattr(params, "expr_metric", "cosine"),
            "gating_mode": getattr(params, "expr_gating_mode", "spatial_knn"),
            "gating_k": gating_k,
            "gating_radius": gating_radius,
            "edge_count_directed": int(e_src.size),
            "edge_count_undirected": int(_undirected_count(e_src, e_dst)),
            "candidate_total": int(expr_total_candidates),
            "candidate_kept_ratio": float(e_src.size / max(1, expr_total_candidates)),
            "expr_dim": int(expr_embedding.shape[1]),
        },
        "union": {
            "edge_count_directed": int(u_src.size),
            "edge_count_undirected": int(_undirected_count(u_src, u_dst)),
            "edge_type_counts": edge_type_counts,
            "edge_attr_schema": ["w_spatial", "w_expr"],
            "edge_attr_dim": 2,
            "reduce_mode": getattr(params, "union_reduce", "max"),
        },
    }

    if verbose:
        num_edges = int(edge_index.shape[1])
        avg_degree = num_edges / max(1, n_cells)
        print(f"Hybrid graph contains {num_edges} directed edges, {n_cells} nodes.")
        print(f"Average degree: {avg_degree:.2f}")

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr_t,
        "num_nodes": n_cells,
        "graph_stats": {
            "num_edges": int(edge_index.shape[1]),
            "avg_degree": float(edge_index.shape[1] / max(1, n_cells)),
        },
        "graph_meta": graph_meta,
    }
