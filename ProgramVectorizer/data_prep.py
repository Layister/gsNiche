from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .bundle_io import read_json
from .common import quantiles
from .schema import ProgramInputConfig, ProgramPipelineConfig, ProgramPreprocessConfig


def load_gss_inputs(gss_bundle_path: Path, cfg: ProgramPipelineConfig) -> tuple[pd.DataFrame, dict, dict]:
    gss_sparse_path = gss_bundle_path / cfg.input.gss_sparse_relpath
    if not gss_sparse_path.exists():
        raise FileNotFoundError(f"Missing GSS sparse file: {gss_sparse_path}")

    gss_df = pd.read_parquet(gss_sparse_path)
    required_cols = {"spot_id", "gene", "gss"}
    missing = required_cols - set(gss_df.columns)
    if missing:
        raise ValueError(
            f"GSS sparse parquet missing required columns {sorted(missing)}. "
            f"Found columns: {list(gss_df.columns)}"
        )

    gss_df = gss_df.loc[:, ["spot_id", "gene", "gss"]].copy()
    gss_df["spot_id"] = gss_df["spot_id"].astype(str)
    gss_df["gene"] = gss_df["gene"].astype(str)
    gss_df["gss"] = gss_df["gss"].astype(np.float32)
    gss_df = gss_df[gss_df["gss"] > 0].reset_index(drop=True)

    gss_meta_path = gss_bundle_path / cfg.input.gss_meta_relpath
    gss_meta = read_json(gss_meta_path) if gss_meta_path.exists() else {}

    gss_manifest_path = gss_bundle_path / cfg.input.gss_manifest_relpath
    gss_manifest = read_json(gss_manifest_path) if gss_manifest_path.exists() else {}
    return gss_df, gss_meta, gss_manifest


def build_spot_gene_matrix(gss_df: pd.DataFrame) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    if gss_df.empty:
        raise ValueError("Input gss_sparse dataframe is empty after filtering gss > 0.")

    spot_ids = np.sort(gss_df["spot_id"].unique().astype(str))
    gene_ids = np.sort(gss_df["gene"].unique().astype(str))

    spot_to_idx = {s: i for i, s in enumerate(spot_ids)}
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}

    row = gss_df["spot_id"].map(spot_to_idx).to_numpy(dtype=np.int64)
    col = gss_df["gene"].map(gene_to_idx).to_numpy(dtype=np.int64)
    data = gss_df["gss"].to_numpy(dtype=np.float32)

    matrix = sp.csr_matrix((data, (row, col)), shape=(len(spot_ids), len(gene_ids)), dtype=np.float32)
    matrix.eliminate_zeros()
    return matrix, spot_ids, gene_ids


def _build_blacklist_mask(
    gene_ids: np.ndarray,
    cfg: ProgramPreprocessConfig,
) -> np.ndarray:
    prefixes = tuple(p.upper() for p in cfg.blacklist_prefixes)
    exact_genes = {g.upper() for g in cfg.blacklist_genes}
    mask = np.zeros(gene_ids.shape[0], dtype=bool)
    for i, gene in enumerate(gene_ids):
        gu = str(gene).upper()
        if gu in exact_genes:
            mask[i] = True
            continue
        if prefixes and gu.startswith(prefixes):
            mask[i] = True
    return mask


def _resolve_support_thresholds(
    n_spots: int,
    cfg: ProgramPreprocessConfig,
) -> tuple[int, int]:
    min_support_auto = max(5, int(math.ceil(cfg.min_gene_support_frac * n_spots)))
    min_support = cfg.min_gene_support_spots if cfg.min_gene_support_spots is not None else min_support_auto
    min_support = max(1, int(min_support))

    max_support_auto = int(math.floor(cfg.max_gene_support_frac * n_spots))
    max_support = cfg.max_gene_support_spots if cfg.max_gene_support_spots is not None else max_support_auto
    max_support = min(max(1, int(max_support)), n_spots)
    if max_support < min_support:
        max_support = min_support
    return min_support, max_support


def infer_gene_activity(
    matrix: sp.csr_matrix,
    gene_ids: np.ndarray,
    cfg: ProgramPreprocessConfig,
) -> dict:
    n_spots, n_genes = matrix.shape
    csc = matrix.tocsc(copy=True)

    thresholds = np.zeros(n_genes, dtype=np.float32)
    active_support_spots = np.zeros(n_genes, dtype=np.int32)
    local_contrast = np.zeros(n_genes, dtype=np.float32)
    gss_mean = np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32, copy=False)
    blacklist_mask = _build_blacklist_mask(gene_ids, cfg)

    rows_binary: list[np.ndarray] = []
    cols_binary: list[np.ndarray] = []
    data_binary: list[np.ndarray] = []
    rows_strength: list[np.ndarray] = []
    cols_strength: list[np.ndarray] = []
    data_strength: list[np.ndarray] = []

    q = float(np.clip(cfg.gene_active_quantile, 0.0, 1.0))
    abs_floor = float(max(0.0, cfg.gene_active_min_gss))

    for j in range(n_genes):
        start, end = csc.indptr[j], csc.indptr[j + 1]
        idx = csc.indices[start:end]
        vals = csc.data[start:end].astype(np.float32, copy=False)
        if vals.size == 0:
            thresholds[j] = np.float32(np.inf)
            continue

        thr = float(max(abs_floor, np.quantile(vals, q)))
        thresholds[j] = np.float32(thr)

        keep = vals >= thr
        if not np.any(keep):
            continue

        active_idx = idx[keep].astype(np.int64, copy=False)
        active_vals = vals[keep].astype(np.float32, copy=False)
        active_support_spots[j] = int(active_idx.size)

        rows_binary.append(active_idx)
        cols_binary.append(np.full(active_idx.shape[0], j, dtype=np.int64))
        data_binary.append(np.ones(active_idx.shape[0], dtype=np.float32))

        rows_strength.append(active_idx)
        cols_strength.append(np.full(active_idx.shape[0], j, dtype=np.int64))
        data_strength.append(active_vals)

        active_mean = float(np.mean(active_vals)) if active_vals.size > 0 else 0.0
        inactive_vals = vals[~keep]
        inactive_mean = float(np.mean(inactive_vals)) if inactive_vals.size > 0 else 0.0
        local_contrast[j] = float(
            max(0.0, (active_mean - inactive_mean) / (active_mean + inactive_mean + 1e-8))
        )

    if rows_binary:
        row_bin = np.concatenate(rows_binary)
        col_bin = np.concatenate(cols_binary)
        dat_bin = np.concatenate(data_binary)
        active_mask_binary = sp.csr_matrix((dat_bin, (row_bin, col_bin)), shape=matrix.shape, dtype=np.float32)
    else:
        active_mask_binary = sp.csr_matrix(matrix.shape, dtype=np.float32)

    if rows_strength:
        row_strength = np.concatenate(rows_strength)
        col_strength = np.concatenate(cols_strength)
        dat_strength = np.concatenate(data_strength)
        active_strength = sp.csr_matrix(
            (dat_strength, (row_strength, col_strength)),
            shape=matrix.shape,
            dtype=np.float32,
        )
    else:
        active_strength = sp.csr_matrix(matrix.shape, dtype=np.float32)

    support_frac = active_support_spots.astype(np.float32) / max(1, n_spots)
    return {
        "active_mask_binary": active_mask_binary.tocsr(),
        "active_strength": active_strength.tocsr(),
        "active_thresholds": thresholds,
        "support_spots": active_support_spots,
        "support_frac": support_frac.astype(np.float32, copy=False),
        "gss_mean": gss_mean,
        "local_contrast": local_contrast,
        "blacklist_mask": blacklist_mask,
        "summary": {
            "active_threshold_quantiles": quantiles(thresholds[np.isfinite(thresholds)]),
            "active_support_spots_quantiles": quantiles(active_support_spots.astype(np.float32)),
            "active_support_frac_quantiles": quantiles(support_frac.astype(np.float32)),
            "local_contrast_quantiles": quantiles(local_contrast.astype(np.float32)),
            "gene_gss_mean_quantiles": quantiles(gss_mean.astype(np.float32)),
        },
    }


def apply_gene_filters(
    matrix: sp.csr_matrix,
    gene_ids: np.ndarray,
    activity_payload: dict,
    cfg: ProgramPreprocessConfig,
) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    n_spots = int(matrix.shape[0])
    support_spots = np.asarray(activity_payload["support_spots"], dtype=np.int32)
    support_frac = np.asarray(activity_payload["support_frac"], dtype=np.float32)
    gss_mean = np.asarray(activity_payload["gss_mean"], dtype=np.float32)
    local_contrast = np.asarray(activity_payload["local_contrast"], dtype=np.float32)
    blacklist_mask = np.asarray(activity_payload["blacklist_mask"], dtype=bool)

    min_support, max_support = _resolve_support_thresholds(n_spots, cfg)
    keep = (support_spots >= min_support) & (support_spots <= max_support)
    relaxed_max_support = False
    relaxed_min_support = False

    dropped_by_blacklist = np.zeros_like(keep)
    if cfg.blacklist_action == "drop":
        dropped_by_blacklist = blacklist_mask.copy()
        keep &= ~blacklist_mask

    if int(np.count_nonzero(keep)) == 0:
        relaxed_max_support = True
        keep = support_spots >= min_support
        if cfg.blacklist_action == "drop":
            keep &= ~blacklist_mask
    if int(np.count_nonzero(keep)) == 0:
        relaxed_min_support = True
        keep = support_spots >= 1
        if cfg.blacklist_action == "drop":
            keep &= ~blacklist_mask
    if int(np.count_nonzero(keep)) == 0:
        raise ValueError("All genes were filtered out. Relax support thresholds or blacklist settings.")

    gss_q = float(np.clip(cfg.min_gene_gss_mean_quantile, 0.0, 1.0))
    gss_cutoff = 0.0
    filtered_low_gss = 0
    relaxed_gss_quantile = False
    if gss_q > 0.0 and int(np.count_nonzero(keep)) > 0:
        candidate_vals = gss_mean[keep]
        gss_cutoff = float(np.quantile(candidate_vals, gss_q))
        keep_q = keep & (gss_mean >= gss_cutoff)
        if int(np.count_nonzero(keep_q)) > 0:
            filtered_low_gss = int(np.count_nonzero(keep)) - int(np.count_nonzero(keep_q))
            keep = keep_q
        else:
            relaxed_gss_quantile = True

    contrast_q = float(np.clip(cfg.min_gene_local_contrast_quantile, 0.0, 1.0))
    contrast_cutoff = 0.0
    filtered_low_contrast = 0
    relaxed_contrast_quantile = False
    if contrast_q > 0.0 and int(np.count_nonzero(keep)) > 0:
        candidate_vals = local_contrast[keep]
        contrast_cutoff = float(np.quantile(candidate_vals, contrast_q))
        keep_q = keep & (local_contrast >= contrast_cutoff)
        if int(np.count_nonzero(keep_q)) > 0:
            filtered_low_contrast = int(np.count_nonzero(keep)) - int(np.count_nonzero(keep_q))
            keep = keep_q
        else:
            relaxed_contrast_quantile = True

    matrix_f = matrix[:, keep].tocsr()
    gene_ids_f = gene_ids[keep]
    active_mask_binary_f = activity_payload["active_mask_binary"][:, keep].tocsr()
    active_strength_f = activity_payload["active_strength"][:, keep].tocsr()
    thresholds_f = np.asarray(activity_payload["active_thresholds"], dtype=np.float32)[keep]
    support_spots_f = support_spots[keep]
    support_frac_f = support_frac[keep]
    gss_mean_f = gss_mean[keep]
    local_contrast_f = local_contrast[keep]
    blacklist_mask_f = blacklist_mask[keep]

    idf = np.log1p(n_spots / (1.0 + support_spots_f.astype(np.float64))).astype(np.float32)

    blacklist_factor = np.ones_like(idf, dtype=np.float32)
    if cfg.blacklist_action == "downweight":
        factor = float(np.clip(cfg.blacklist_downweight_factor, 1e-6, 1.0))
        blacklist_factor[blacklist_mask_f] = factor

    summary = {
        "n_genes_before": int(gene_ids.shape[0]),
        "n_genes_after": int(gene_ids_f.shape[0]),
        "min_gene_support_spots": int(min_support),
        "max_gene_support_spots": int(max_support),
        "filtered_low_support": int(np.count_nonzero(support_spots < min_support)),
        "filtered_high_support": int(np.count_nonzero(support_spots > max_support)),
        "min_gene_gss_mean_quantile": float(gss_q),
        "gene_gss_mean_cutoff": float(gss_cutoff),
        "filtered_low_gss_mean": int(filtered_low_gss),
        "min_gene_local_contrast_quantile": float(contrast_q),
        "gene_local_contrast_cutoff": float(contrast_cutoff),
        "filtered_low_local_contrast": int(filtered_low_contrast),
        "relaxed_gss_quantile": bool(relaxed_gss_quantile),
        "relaxed_contrast_quantile": bool(relaxed_contrast_quantile),
        "relaxed_max_support": bool(relaxed_max_support),
        "relaxed_min_support": bool(relaxed_min_support),
        "blacklisted_total": int(np.count_nonzero(blacklist_mask)),
        "blacklisted_dropped": int(np.count_nonzero(dropped_by_blacklist)),
        "active_threshold_quantiles": quantiles(thresholds_f[np.isfinite(thresholds_f)]),
        "gene_gss_mean_quantiles": quantiles(gss_mean.astype(np.float32)),
        "local_contrast_quantiles": quantiles(local_contrast.astype(np.float32)),
        "support_spots_quantiles": quantiles(support_spots.astype(np.float32)),
        "support_frac_quantiles": quantiles(support_frac.astype(np.float32)),
    }

    payload = {
        "matrix": matrix_f,
        "gene_ids": gene_ids_f,
        "active_mask_binary": active_mask_binary_f,
        "active_strength": active_strength_f,
        "active_thresholds": thresholds_f,
        "support_spots": support_spots_f,
        "support_frac": support_frac_f,
        "gss_mean": gss_mean_f,
        "local_contrast": local_contrast_f,
        "idf": idf,
        "blacklist_mask": blacklist_mask_f,
        "blacklist_factor": blacklist_factor,
        "summary": summary,
    }
    return matrix_f, gene_ids_f, payload


def _resolve_h5ad_path(gss_manifest: dict, input_cfg: ProgramInputConfig) -> Path:
    if input_cfg.h5ad_path_override:
        raw = str(input_cfg.h5ad_path_override).strip()
    else:
        raw = str(gss_manifest.get("inputs", {}).get("h5ad_path", "")).strip()
    if not raw:
        raise ValueError("Missing h5ad path. Set ProgramInputConfig.h5ad_path_override or provide it in gss manifest.")
    path = Path(raw)
    if (not path.exists()) or (not path.is_file()):
        raise FileNotFoundError(f"h5ad file not found: {path}")
    return path


def _load_spot_ids_and_coords_from_h5ad(
    gss_manifest: dict,
    input_cfg: ProgramInputConfig,
) -> tuple[np.ndarray, np.ndarray]:
    h5ad_path = _resolve_h5ad_path(gss_manifest=gss_manifest, input_cfg=input_cfg)
    spot_field = str(gss_manifest.get("inputs", {}).get("spot_id_field", "obs_names")).strip() or "obs_names"

    try:
        import scanpy as sc

        adata = sc.read_h5ad(h5ad_path)
        if spot_field == "obs_names":
            spot_ids = adata.obs_names.astype(str).to_numpy()
        elif spot_field in adata.obs.columns:
            raw = adata.obs[spot_field]
            if raw.isna().any():
                raise ValueError(f"Spot id field contains NaN values: {spot_field}")
            spot_ids = raw.astype(str).to_numpy()
        else:
            spot_ids = adata.obs_names.astype(str).to_numpy()
        if "spatial" not in adata.obsm:
            raise ValueError("Raw h5ad is missing adata.obsm['spatial'].")
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    except Exception:  # noqa: BLE001
        if spot_field != "obs_names":
            raise
        import h5py

        with h5py.File(h5ad_path, "r") as f:
            if "obs" not in f:
                raise ValueError("Raw h5ad missing obs group.")
            obs = f["obs"]
            key = "_index" if "_index" in obs else "index" if "index" in obs else None
            if key is None:
                raise ValueError("Raw h5ad missing obs index for spot alignment.")
            raw = obs[key][:]
            if raw.dtype.kind in {"S", "O"}:
                vals = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in raw.tolist()]
            else:
                vals = [str(x) for x in raw.tolist()]
            spot_ids = np.asarray(vals, dtype=str)
            if "obsm" not in f or "spatial" not in f["obsm"]:
                raise ValueError("Raw h5ad missing obsm/spatial.")
            coords = np.asarray(f["obsm"]["spatial"][:], dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != spot_ids.shape[0]:
        raise ValueError("Invalid raw spatial coordinates; expected shape (n_spots, >=2) aligned with spot ids.")
    if len(spot_ids) != len(set(spot_ids.tolist())):
        raise ValueError("Spot IDs in raw h5ad are not unique.")
    return spot_ids.astype(str), coords[:, :2].copy()


def build_spot_neighbors_from_raw(
    spot_ids: np.ndarray,
    gss_manifest: dict,
    input_cfg: ProgramInputConfig,
) -> tuple[list[np.ndarray], dict]:
    raw_spot_ids, raw_coords = _load_spot_ids_and_coords_from_h5ad(
        gss_manifest=gss_manifest,
        input_cfg=input_cfg,
    )
    raw_index = {sid: i for i, sid in enumerate(raw_spot_ids.tolist())}
    row = np.array([raw_index.get(str(s), -1) for s in spot_ids], dtype=np.int64)
    if np.any(row < 0):
        miss = spot_ids[row < 0][:5].tolist()
        raise ValueError(f"Cannot align some Program spot_ids to raw h5ad spatial coordinates: {miss}")

    coords = np.asarray(raw_coords[row, :2], dtype=np.float64)
    finite_mask = np.isfinite(coords).all(axis=1)
    if int(np.count_nonzero(finite_mask)) < 2:
        raise ValueError("Too few valid spatial coordinates to build spot neighbors.")

    coords_f = coords[finite_mask, :2]
    tree = cKDTree(coords_f)
    try:
        d, _ = tree.query(coords_f, k=2, workers=-1)
    except TypeError:
        d, _ = tree.query(coords_f, k=2)
    nn = np.asarray(d[:, 1], dtype=np.float64)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        raise ValueError("Cannot estimate spot spacing from raw spatial coordinates.")
    spot_spacing = float(np.median(nn))
    radius = float(max(1e-8, 1.2 * spot_spacing))

    finite_idx = np.flatnonzero(finite_mask)
    balls = tree.query_ball_point(coords_f, r=radius)
    neighbors: list[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in range(coords.shape[0])]
    for local_i, members in enumerate(balls):
        global_i = int(finite_idx[local_i])
        mapped = [int(finite_idx[j]) for j in members if int(finite_idx[j]) != global_i]
        if mapped:
            neighbors[global_i] = np.asarray(sorted(set(mapped)), dtype=np.int64)

    return neighbors, {
        "source": "raw_h5ad_spatial",
        "spot_spacing": float(spot_spacing),
        "neighbor_radius": float(radius),
        "finite_spot_frac": float(np.mean(finite_mask)),
    }

