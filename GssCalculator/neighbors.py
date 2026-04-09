from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from .stats import quantiles
    from .schema import NeighborsConfig
except ImportError:
    from GssCalculator.stats import quantiles
    from GssCalculator.schema import NeighborsConfig


def build_spatial_candidates(
    coords: np.ndarray,
    cfg: NeighborsConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[set[int]]]:
    n_spots = int(coords.shape[0])
    if n_spots < 2:
        return (
            [np.array([], dtype=np.int32) for _ in range(n_spots)],
            [np.array([], dtype=np.float32) for _ in range(n_spots)],
            [set() for _ in range(n_spots)],
        )

    k_query = max(2, min(cfg.k_spatial + 1, n_spots))
    nbrs = NearestNeighbors(n_neighbors=k_query)
    nbrs.fit(coords)

    distances_knn, indices_knn = nbrs.kneighbors(coords, return_distance=True)
    radius_neighbors = None
    radius_distances = None

    if cfg.candidate_mode in {"radius", "hybrid"}:
        if cfg.spatial_radius is None or cfg.spatial_radius <= 0:
            raise ValueError("spatial_radius must be > 0 when candidate_mode is radius or hybrid")
        radius_distances, radius_neighbors = nbrs.radius_neighbors(
            coords,
            radius=cfg.spatial_radius,
            return_distance=True,
        )

    candidates: list[np.ndarray] = []
    candidate_distances: list[np.ndarray] = []
    adjacency: list[set[int]] = [set() for _ in range(n_spots)]

    for i in range(n_spots):
        dist_map: dict[int, float] = {}

        for idx, dist in zip(indices_knn[i], distances_knn[i], strict=False):
            if idx == i:
                continue
            dist_map[int(idx)] = float(dist)
            adjacency[i].add(int(idx))
            adjacency[int(idx)].add(i)

        if radius_neighbors is not None:
            for idx, dist in zip(radius_neighbors[i], radius_distances[i], strict=False):
                if idx == i:
                    continue
                current = dist_map.get(int(idx))
                if current is None or dist < current:
                    dist_map[int(idx)] = float(dist)

        ordered = sorted(dist_map.items(), key=lambda x: x[1])
        candidates.append(np.array([x[0] for x in ordered], dtype=np.int32))
        candidate_distances.append(np.array([x[1] for x in ordered], dtype=np.float32))

    return candidates, candidate_distances, adjacency


def _nodes_within_hops(adjacency: list[set[int]], source: int, max_hops: int) -> set[int]:
    if max_hops <= 0:
        return set(adjacency[source])

    visited = {source}
    frontier = {source}
    within = set()

    for _ in range(max_hops):
        next_frontier: set[int] = set()
        for node in frontier:
            for nb in adjacency[node]:
                if nb in visited:
                    continue
                visited.add(nb)
                within.add(nb)
                next_frontier.add(nb)
        if not next_frontier:
            break
        frontier = next_frontier

    return within


def build_neighbors(
    latent: np.ndarray,
    candidate_idx: list[np.ndarray],
    candidate_dist: list[np.ndarray],
    spatial_adjacency: list[set[int]],
    cfg: NeighborsConfig,
) -> tuple[np.ndarray, np.ndarray, dict, list[set[int]]]:
    del candidate_dist

    n_spots = latent.shape[0]
    k_effective = max(0, min(cfg.k, n_spots - 1))

    idx_out = np.full((n_spots, k_effective), fill_value=-1, dtype=np.int32)
    sim_out = np.zeros((n_spots, k_effective), dtype=np.float32)

    latent_norm = np.linalg.norm(latent, axis=1, keepdims=True)
    latent_norm[latent_norm == 0] = 1.0
    latent_unit = latent / latent_norm

    connectivity_sets = [
        _nodes_within_hops(spatial_adjacency, i, cfg.connectivity_hops) for i in range(n_spots)
    ]

    actual_ks = []
    for i in range(n_spots):
        base_candidates = candidate_idx[i]
        if base_candidates.size == 0:
            actual_ks.append(0)
            continue

        connected = connectivity_sets[i]
        mask_connected = np.array([c in connected for c in base_candidates], dtype=bool)
        filtered_candidates = base_candidates[mask_connected]

        # Fallback semantics:
        # - decrease_k: strictly keep connectivity-filtered candidates (can be empty).
        # - spatial: allow fallback to spatial candidate pool if connectivity pool is empty.
        if filtered_candidates.size == 0 and cfg.fallback == "spatial":
            filtered_candidates = base_candidates

        if filtered_candidates.size == 0:
            actual_ks.append(0)
            continue

        sims = latent_unit[filtered_candidates] @ latent_unit[i]
        order = np.argsort(-sims)
        selected = filtered_candidates[order]
        selected_sim = sims[order]

        if selected.size < k_effective and cfg.fallback == "spatial":
            seen = set(selected.tolist())
            fill_nodes = []
            fill_sims = []
            for node in base_candidates:
                if int(node) in seen:
                    continue
                fill_nodes.append(int(node))
                fill_sims.append(float(latent_unit[int(node)] @ latent_unit[i]))
                if len(fill_nodes) + len(seen) >= k_effective:
                    break

            if fill_nodes:
                selected = np.concatenate([selected, np.array(fill_nodes, dtype=np.int32)])
                selected_sim = np.concatenate([selected_sim, np.array(fill_sims, dtype=np.float32)])

        take = min(k_effective, selected.size)

        idx_out[i, :take] = selected[:take]
        sim_out[i, :take] = selected_sim[:take]
        actual_ks.append(int(take))

    missing_ratio = float(np.mean(idx_out < 0)) if k_effective > 0 else 0.0

    neighbors_meta = {
        "k": cfg.k,
        "k_effective": k_effective,
        "k_spatial": cfg.k_spatial,
        "candidate_mode": cfg.candidate_mode,
        "spatial_radius": cfg.spatial_radius,
        "similarity_metric": cfg.similarity_metric,
        "connectivity_hops": cfg.connectivity_hops,
        "fallback": cfg.fallback,
        "actual_k_quantiles": quantiles(np.asarray(actual_ks, dtype=np.float32)),
        "actual_k_min": int(min(actual_ks) if actual_ks else 0),
        "actual_k_max": int(max(actual_ks) if actual_ks else 0),
        "missing_ratio": missing_ratio,
        "valid_neighbor_ratio": float(np.mean(idx_out >= 0)) if k_effective > 0 else 1.0,
    }

    return idx_out, sim_out, neighbors_meta, connectivity_sets
