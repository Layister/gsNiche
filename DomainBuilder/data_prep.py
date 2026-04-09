from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .bundle_io import read_json
from .domain_ops import build_spot_graph_from_coords_knn, build_spot_graph_from_neighbors
from .schema import DomainPipelineConfig


def _resolve_program_allowlist(cfg: DomainPipelineConfig) -> tuple[set[str], dict]:
    selected: list[str] = []
    inline = list(getattr(cfg.input, "program_allowlist", ()) or ())
    selected.extend([str(x).strip() for x in inline if str(x).strip()])
    file_path_raw = str(getattr(cfg.input, "program_allowlist_file", "") or "").strip()
    if file_path_raw:
        path = Path(file_path_raw)
        if not path.exists():
            raise FileNotFoundError(f"Configured program_allowlist_file does not exist: {path}")
        selected.extend([line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    allow = {str(x) for x in selected if str(x)}
    return allow, {
        "enabled": bool(allow),
        "requested_count": int(len(selected)),
        "unique_count": int(len(allow)),
        "file": file_path_raw or None,
    }


def _resolve_program_qc_selection(
    programs_df: pd.DataFrame,
    activation_df: pd.DataFrame,
    program_qc_df: pd.DataFrame,
    cfg: DomainPipelineConfig,
) -> tuple[set[str], dict]:
    known_programs = {str(x) for x in programs_df["program_id"].astype(str).unique().tolist()}
    if not bool(getattr(cfg.input, "program_use_qc_selection", True)):
        return known_programs, {
            "enabled": False,
            "selected_count": int(len(known_programs)),
            "dropped_count": 0,
        }

    if program_qc_df.empty or "program_id" not in program_qc_df.columns:
        return known_programs, {
            "enabled": True,
            "selected_count": int(len(known_programs)),
            "dropped_count": 0,
            "fallback": "missing_program_qc",
        }

    qc = program_qc_df.copy()
    qc["program_id"] = qc["program_id"].astype(str)
    qc = qc.drop_duplicates(subset=["program_id"], keep="first")

    selected = qc["program_id"].astype(str).isin(known_programs)
    summary: dict[str, object] = {
        "enabled": True,
        "fallback": None,
    }

    validity_allowed = {str(x).strip() for x in getattr(cfg.input, "allowed_validity_statuses", ()) if str(x).strip()}
    if validity_allowed and "validity_status" in qc.columns:
        selected &= qc["validity_status"].astype(str).isin(validity_allowed)
        summary["allowed_validity_statuses"] = sorted(validity_allowed)

    routing_allowed = {str(x).strip() for x in getattr(cfg.input, "allowed_routing_statuses", ()) if str(x).strip()}
    if routing_allowed and "routing_status" in qc.columns:
        selected &= qc["routing_status"].astype(str).isin(routing_allowed)
        summary["allowed_routing_statuses"] = sorted(routing_allowed)

    redundancy_allowed = {
        str(x).strip() for x in getattr(cfg.input, "allowed_redundancy_statuses", ()) if str(x).strip()
    }
    if redundancy_allowed and "redundancy_status" in qc.columns:
        selected &= qc["redundancy_status"].astype(str).isin(redundancy_allowed)
        summary["allowed_redundancy_statuses"] = sorted(redundancy_allowed)

    selected_programs = set(qc.loc[selected, "program_id"].astype(str).tolist())
    if not selected_programs:
        selected_programs = known_programs
        summary["fallback"] = "empty_after_qc_filter"

    dropped_programs = sorted(known_programs - selected_programs)
    summary["selected_count"] = int(len(selected_programs))
    summary["dropped_count"] = int(len(dropped_programs))
    summary["dropped_examples"] = dropped_programs[:10]
    summary["activation_rows_before"] = int(activation_df.shape[0])
    return selected_programs, summary


def _load_h5ad_spot_ids_and_coords(gss_manifest: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    h5ad_path = str(gss_manifest.get("inputs", {}).get("h5ad_path", "")).strip()
    if not h5ad_path:
        return None, None

    path = Path(h5ad_path)
    if not path.exists():
        return None, None

    try:
        import scanpy as sc
    except Exception:  # noqa: BLE001
        return None, None

    try:
        adata = sc.read_h5ad(path)
    except Exception:  # noqa: BLE001
        return None, None

    spot_field = str(gss_manifest.get("inputs", {}).get("spot_id_field", "obs_names"))
    if spot_field == "obs_names":
        spot_ids = adata.obs_names.astype(str).to_numpy()
    elif spot_field in adata.obs.columns:
        raw = adata.obs[spot_field]
        if raw.isna().any():
            return None, None
        spot_ids = raw.astype(str).to_numpy()
    else:
        return None, None

    if len(spot_ids) != len(set(spot_ids.tolist())):
        return None, None

    coords = None
    if "spatial" in adata.obsm:
        try:
            coords_arr = np.asarray(adata.obsm["spatial"], dtype=np.float64)
            if coords_arr.ndim == 2 and coords_arr.shape[1] >= 2 and coords_arr.shape[0] == spot_ids.shape[0]:
                coords = coords_arr[:, :2].copy()
        except Exception:  # noqa: BLE001
            coords = None
    return spot_ids.astype(str), coords


def _resolve_gss_bundle_path(program_bundle_path: Path, cfg: DomainPipelineConfig, program_manifest: dict) -> Path:
    if cfg.input.gss_bundle_path_override:
        path = Path(cfg.input.gss_bundle_path_override)
        if not path.exists():
            raise FileNotFoundError(f"Configured gss_bundle_path_override does not exist: {path}")
        return path

    inputs = program_manifest.get("inputs", {})
    gss_path = inputs.get("gss_bundle_path", None)
    if gss_path:
        path = Path(str(gss_path))
        if path.exists():
            return path

    # Fallback for local layout where program_bundle sits next to gss_bundle.
    sibling = program_bundle_path.parent / "gss_bundle"
    if sibling.exists():
        return sibling

    raise FileNotFoundError(
        "Unable to locate gss_bundle path from program manifest. "
        "Set DomainPipelineConfig.input.gss_bundle_path_override explicitly."
    )


def _read_required_program_inputs(
    program_bundle_path: Path,
    cfg: DomainPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, Path]:
    manifest_path = program_bundle_path / cfg.input.program_manifest_relpath
    programs_path = program_bundle_path / cfg.input.programs_relpath
    activation_path = program_bundle_path / cfg.input.program_activation_relpath
    program_qc_path = program_bundle_path / cfg.input.program_qc_relpath

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing program manifest: {manifest_path}")
    if not programs_path.exists():
        raise FileNotFoundError(f"Missing programs parquet: {programs_path}")
    if not activation_path.exists():
        raise FileNotFoundError(f"Missing program activation parquet: {activation_path}")

    manifest = read_json(manifest_path)
    programs_df = pd.read_parquet(programs_path)
    activation_df = pd.read_parquet(activation_path)
    if bool(cfg.program_confidence.enabled):
        if not program_qc_path.exists():
            raise FileNotFoundError(
                f"Missing program QC table required by program_confidence.enabled=True: {program_qc_path}"
            )
        program_qc_df = pd.read_parquet(program_qc_path)
    else:
        program_qc_df = pd.read_parquet(program_qc_path) if program_qc_path.exists() else pd.DataFrame()
    return programs_df, activation_df, manifest, program_qc_df, program_qc_path


def _resolve_spot_order(
    gss_bundle_path: Path,
    gss_manifest: dict,
    activation_df: pd.DataFrame,
    cfg: DomainPipelineConfig,
    n_spots_neighbors: int,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    spot_ids_path = gss_bundle_path / cfg.input.neighbors_spot_ids_relpath
    if spot_ids_path.exists():
        spot_ids = np.load(spot_ids_path, allow_pickle=False).astype(str)
        if int(spot_ids.shape[0]) != int(n_spots_neighbors):
            raise ValueError(
                f"spot_ids length mismatch with neighbors rows: {spot_ids.shape[0]} vs {n_spots_neighbors}"
            )

        coords = None
        if cfg.input.try_load_coords_from_h5ad:
            spot_h5ad, coords_h5ad = _load_h5ad_spot_ids_and_coords(gss_manifest)
            if spot_h5ad is not None and coords_h5ad is not None:
                idx = {sid: i for i, sid in enumerate(spot_h5ad)}
                if set(spot_ids.tolist()).issubset(set(idx.keys())):
                    coords = np.full((spot_ids.shape[0], 2), np.nan, dtype=np.float64)
                    for i, sid in enumerate(spot_ids):
                        coords[i] = coords_h5ad[idx[sid]]
        return spot_ids, coords, "neighbors_spot_ids"

    spot_h5ad = None
    coords_h5ad = None
    if cfg.input.try_load_coords_from_h5ad:
        spot_h5ad, coords_h5ad = _load_h5ad_spot_ids_and_coords(gss_manifest)
    if spot_h5ad is not None:
        if int(spot_h5ad.shape[0]) != int(n_spots_neighbors):
            raise ValueError(
                f"h5ad spot count mismatch with neighbors rows: {spot_h5ad.shape[0]} vs {n_spots_neighbors}"
            )
        return spot_h5ad, coords_h5ad, "h5ad"

    raise ValueError(
        "Cannot recover canonical spot order. "
        "Provide neighbors/spot_ids.npy (recommended) or ensure h5ad spot order can be resolved."
    )


def _build_activation_matrix(
    activation_df: pd.DataFrame,
    spot_ids: np.ndarray,
    program_ids: np.ndarray,
    activation_col: str = "activation",
) -> sp.csr_matrix:
    spot_to_idx = {str(s): i for i, s in enumerate(spot_ids)}
    prog_to_idx = {str(p): i for i, p in enumerate(program_ids)}

    row = activation_df["spot_id"].map(spot_to_idx)
    col = activation_df["program_id"].map(prog_to_idx)
    if row.isna().any():
        unknown = activation_df.loc[row.isna(), "spot_id"].astype(str).unique().tolist()
        raise ValueError(f"Activation contains unknown spot_id entries: {unknown[:5]}")
    if col.isna().any():
        unknown = activation_df.loc[col.isna(), "program_id"].astype(str).unique().tolist()
        raise ValueError(f"Activation contains unknown program_id entries: {unknown[:5]}")

    data = activation_df[activation_col].to_numpy(dtype=np.float32)
    mat = sp.csr_matrix(
        (data, (row.to_numpy(dtype=np.int64), col.to_numpy(dtype=np.int64))),
        shape=(spot_ids.shape[0], program_ids.shape[0]),
        dtype=np.float32,
    )
    mat.eliminate_zeros()
    return mat


def _resolve_program_confidence_weighting(
    program_ids: np.ndarray,
    program_qc_df: pd.DataFrame,
    cfg: DomainPipelineConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict]:
    pids = [str(pid) for pid in np.asarray(program_ids, dtype=object).tolist()]
    rows: list[dict] = []
    if not pids:
        empty = pd.DataFrame(
            columns=[
                "program_id",
                "program_confidence_raw",
                "program_confidence_used",
                "program_confidence_weight",
            ]
        )
        return (
            empty,
            {},
            {
                "enabled": bool(cfg.program_confidence.enabled),
                "program_count": 0,
                "missing_program_confidence_count": 0,
            },
        )

    enabled = bool(cfg.program_confidence.enabled)
    conf_col = str(cfg.program_confidence.confidence_col)
    strict = bool(cfg.program_confidence.strict)
    min_conf = float(np.clip(float(cfg.program_confidence.min_confidence), 0.0, 1.0))
    gamma = float(max(0.0, cfg.program_confidence.gamma))

    raw_map: dict[str, float] = {}
    if enabled:
        required = {"program_id", conf_col}
        missing_cols = required - set(program_qc_df.columns)
        if missing_cols:
            raise ValueError(
                f"program_qc table missing required columns for confidence weighting: {sorted(missing_cols)}"
            )
        qc = program_qc_df.loc[:, ["program_id", conf_col]].copy()
        qc["program_id"] = qc["program_id"].astype(str)
        qc[conf_col] = pd.to_numeric(qc[conf_col], errors="coerce")
        qc = qc.dropna(subset=[conf_col]).groupby("program_id", as_index=False)[conf_col].max()
        raw_map = {str(pid): float(val) for pid, val in zip(qc["program_id"].tolist(), qc[conf_col].tolist())}

    missing_pids = sorted([pid for pid in pids if pid not in raw_map]) if enabled else []
    if enabled and strict and missing_pids:
        raise ValueError(
            "Cannot resolve program confidence for all program_ids. "
            f"Missing examples: {missing_pids[:5]}"
        )

    weight_info: dict[str, dict[str, float]] = {}
    for pid in pids:
        raw_conf = float(raw_map.get(pid, 1.0)) if enabled else 1.0
        used_conf = float(np.clip(raw_conf, min_conf, 1.0)) if enabled else 1.0
        weight = float(used_conf**gamma) if enabled else 1.0
        weight_info[pid] = {
            "program_confidence_raw": raw_conf,
            "program_confidence_used": used_conf,
            "program_confidence_weight": weight,
        }
        rows.append({"program_id": pid, **weight_info[pid]})

    out_df = pd.DataFrame(rows).sort_values("program_id").reset_index(drop=True)
    weights = out_df["program_confidence_weight"].to_numpy(dtype=np.float64)
    used = out_df["program_confidence_used"].to_numpy(dtype=np.float64)
    raw = out_df["program_confidence_raw"].to_numpy(dtype=np.float64)
    summary = {
        "enabled": enabled,
        "confidence_col": conf_col,
        "min_confidence": min_conf,
        "gamma": gamma,
        "strict": strict,
        "program_count": int(out_df.shape[0]),
        "missing_program_confidence_count": int(len(missing_pids)),
        "weight_quantiles": {
            "p10": float(np.quantile(weights, 0.10)),
            "p50": float(np.quantile(weights, 0.50)),
            "p90": float(np.quantile(weights, 0.90)),
        },
        "confidence_used_quantiles": {
            "p10": float(np.quantile(used, 0.10)),
            "p50": float(np.quantile(used, 0.50)),
            "p90": float(np.quantile(used, 0.90)),
        },
        "confidence_raw_quantiles": {
            "p10": float(np.quantile(raw, 0.10)),
            "p50": float(np.quantile(raw, 0.50)),
            "p90": float(np.quantile(raw, 0.90)),
        },
    }
    return out_df, weight_info, summary


def load_domain_inputs(
    program_bundle_path: Path,
    cfg: DomainPipelineConfig,
) -> dict:
    programs_df, activation_df, program_manifest, program_qc_df, program_qc_path = _read_required_program_inputs(
        program_bundle_path,
        cfg,
    )
    program_allowlist, allowlist_summary = _resolve_program_allowlist(cfg)

    required_activation_cols = {"program_id", "spot_id", "activation"}
    missing = required_activation_cols - set(activation_df.columns)
    if missing:
        raise ValueError(
            f"program_activation.parquet missing required columns: {sorted(missing)}; "
            f"found={list(activation_df.columns)}"
        )

    if "activation_identity_view_weighted" in activation_df.columns:
        activation_value_col = "activation_identity_view_weighted"
    elif "activation_identity_view" in activation_df.columns:
        activation_value_col = "activation_identity_view"
    else:
        activation_value_col = "activation"
    base_cols = ["program_id", "spot_id", "activation"]
    extra_cols = [
        c
        for c in [
            "activation_identity_view",
            "activation_full",
            "activation_identity_view_weighted",
            "activation_weighted",
            "activation_view",
            "recommended_domain_input",
            "program_confidence",
            "rank_in_spot",
        ]
        if c in activation_df.columns
    ]
    ordered_cols = []
    for col in base_cols + extra_cols:
        if col in activation_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    activation_df = activation_df.loc[:, ordered_cols].copy()
    activation_df["program_id"] = activation_df["program_id"].astype(str)
    activation_df["spot_id"] = activation_df["spot_id"].astype(str)
    if activation_value_col != "activation":
        activation_df["activation"] = activation_df[activation_value_col].to_numpy(dtype=np.float32)
    else:
        activation_df["activation"] = activation_df["activation"].to_numpy(dtype=np.float32)
    if "activation_identity_view" in activation_df.columns:
        activation_df["activation_identity_view"] = activation_df["activation_identity_view"].to_numpy(dtype=np.float32)
    if "activation_full" in activation_df.columns:
        activation_df["activation_full"] = activation_df["activation_full"].to_numpy(dtype=np.float32)
    if "activation_identity_view_weighted" in activation_df.columns:
        activation_df["activation_identity_view_weighted"] = activation_df["activation_identity_view_weighted"].to_numpy(
            dtype=np.float32
        )
    activation_df = activation_df[activation_df["activation"] > 0].reset_index(drop=True)

    if "program_id" not in programs_df.columns:
        raise ValueError("programs.parquet must include 'program_id' column")
    programs_df = programs_df.copy()
    programs_df["program_id"] = programs_df["program_id"].astype(str)
    if "gene" in programs_df.columns:
        programs_df["gene"] = programs_df["gene"].astype(str)
    qc_selected_programs, qc_selection_summary = _resolve_program_qc_selection(
        programs_df=programs_df,
        activation_df=activation_df,
        program_qc_df=program_qc_df,
        cfg=cfg,
    )
    programs_df = programs_df.loc[programs_df["program_id"].isin(qc_selected_programs)].reset_index(drop=True)
    activation_df = activation_df.loc[
        activation_df["program_id"].astype(str).isin(qc_selected_programs)
    ].reset_index(drop=True)
    if not program_qc_df.empty and "program_id" in program_qc_df.columns:
        program_qc_df = program_qc_df.loc[
            program_qc_df["program_id"].astype(str).isin(qc_selected_programs)
        ].reset_index(drop=True)
    if program_allowlist:
        known_programs = set(programs_df["program_id"].unique().tolist())
        missing_programs = sorted(program_allowlist - known_programs)
        if missing_programs:
            raise ValueError(
                f"Requested program_allowlist contains unknown program_id: {missing_programs[:10]}"
            )
        programs_df = programs_df.loc[programs_df["program_id"].isin(program_allowlist)].reset_index(drop=True)
        activation_df = activation_df.loc[
            activation_df["program_id"].astype(str).isin(program_allowlist)
        ].reset_index(drop=True)
        if not program_qc_df.empty and "program_id" in program_qc_df.columns:
            program_qc_df = program_qc_df.loc[
                program_qc_df["program_id"].astype(str).isin(program_allowlist)
            ].reset_index(drop=True)

    program_ids = np.sort(programs_df["program_id"].unique().astype(str))
    if program_ids.size == 0:
        # Keep matrix shape valid for no-program samples.
        program_ids = np.asarray([], dtype=object)

    # Activation program_id must be subset of program list.
    extra_programs = set(activation_df["program_id"].unique().tolist()) - set(program_ids.tolist())
    if extra_programs:
        raise ValueError(f"Activation contains program_id not in programs.parquet: {sorted(extra_programs)[:5]}")
    program_conf_df, program_weight_info, program_conf_summary = _resolve_program_confidence_weighting(
        program_ids=program_ids,
        program_qc_df=program_qc_df,
        cfg=cfg,
    )
    raw_conf_map = {
        str(r["program_id"]): float(r["program_confidence_raw"])
        for r in program_conf_df.to_dict(orient="records")
    }
    used_conf_map = {
        str(r["program_id"]): float(r["program_confidence_used"])
        for r in program_conf_df.to_dict(orient="records")
    }
    weight_map = {
        str(r["program_id"]): float(r["program_confidence_weight"])
        for r in program_conf_df.to_dict(orient="records")
    }
    activation_preweighted = activation_value_col.endswith("_weighted")
    if activation_preweighted and "activation_identity_view" in activation_df.columns:
        activation_df["activation_raw"] = activation_df["activation_identity_view"].to_numpy(dtype=np.float32)
    else:
        activation_df["activation_raw"] = activation_df["activation"].to_numpy(dtype=np.float32)
    activation_df["program_confidence_raw"] = activation_df["program_id"].map(raw_conf_map).fillna(1.0).to_numpy(
        dtype=np.float32
    )
    activation_df["program_confidence_used"] = activation_df["program_id"].map(used_conf_map).fillna(1.0).to_numpy(
        dtype=np.float32
    )
    activation_df["program_confidence_weight"] = activation_df["program_id"].map(weight_map).fillna(1.0).to_numpy(
        dtype=np.float32
    )
    if activation_preweighted:
        activation_df["activation_effective"] = activation_df[activation_value_col].to_numpy(dtype=np.float32)
    else:
        activation_df["activation_effective"] = (
            activation_df["activation_raw"].to_numpy(dtype=np.float32)
            * activation_df["program_confidence_weight"].to_numpy(dtype=np.float32)
        )
    activation_df["activation"] = activation_df["activation_effective"].to_numpy(dtype=np.float32)

    gss_bundle_path = _resolve_gss_bundle_path(program_bundle_path, cfg, program_manifest)
    gss_manifest_path = gss_bundle_path / cfg.input.gss_manifest_relpath
    if not gss_manifest_path.exists():
        raise FileNotFoundError(f"Missing gss manifest: {gss_manifest_path}")
    gss_manifest = read_json(gss_manifest_path)

    neighbors_idx_path = gss_bundle_path / cfg.input.neighbors_idx_relpath
    neighbors_meta_path = gss_bundle_path / cfg.input.neighbors_meta_relpath
    if not neighbors_idx_path.exists():
        raise FileNotFoundError(f"Missing neighbors idx: {neighbors_idx_path}")
    if not neighbors_meta_path.exists():
        raise FileNotFoundError(f"Missing neighbors meta: {neighbors_meta_path}")

    neighbors_idx = np.load(neighbors_idx_path, allow_pickle=False)
    if neighbors_idx.ndim != 2:
        raise ValueError(f"neighbors.idx.npy must be 2D, got shape={neighbors_idx.shape}")
    neighbors_meta = read_json(neighbors_meta_path)

    spot_ids, coords, spot_order_source = _resolve_spot_order(
        gss_bundle_path=gss_bundle_path,
        gss_manifest=gss_manifest,
        activation_df=activation_df,
        cfg=cfg,
        n_spots_neighbors=int(neighbors_idx.shape[0]),
    )

    activation_mat_raw = _build_activation_matrix(
        activation_df,
        spot_ids=spot_ids,
        program_ids=program_ids,
        activation_col="activation_raw",
    )
    activation_mat = _build_activation_matrix(
        activation_df,
        spot_ids=spot_ids,
        program_ids=program_ids,
        activation_col="activation_effective",
    )
    dense_activation_raw = np.asarray(activation_mat_raw.toarray(), dtype=np.float32)
    dense_activation = np.asarray(activation_mat.toarray(), dtype=np.float32)

    gss_adjacency, gss_edge_array = build_spot_graph_from_neighbors(neighbor_idx=neighbors_idx)

    flow_graph_mode = str(cfg.potential.flow_graph_mode)
    if flow_graph_mode == "gss":
        adjacency, edge_array = gss_adjacency, gss_edge_array
        graph_source = "gss_neighbors_idx"
    elif flow_graph_mode == "spatial":
        if coords is None:
            raise ValueError(
                "flow_graph_mode='spatial' requires coords, but no spatial coordinates were resolved "
                "(check h5ad path / spot order mapping)."
            )
        adjacency, edge_array = build_spot_graph_from_coords_knn(
            coords=coords,
            k=int(cfg.potential.spatial_graph_k),
        )
        graph_source = "spatial_knn_from_coords"
    else:
        raise ValueError(f"Unsupported flow_graph_mode: {flow_graph_mode}")

    return {
        "programs_df": programs_df,
        "activation_df": activation_df,
        "program_manifest": program_manifest,
        "gss_manifest": gss_manifest,
        "neighbors_meta": neighbors_meta,
        "gss_bundle_path": gss_bundle_path,
        "neighbors_idx_path": neighbors_idx_path,
        "neighbors_meta_path": neighbors_meta_path,
        "spot_ids": spot_ids,
        "program_ids": program_ids,
        "coords": coords,
        "spot_order_source": spot_order_source,
        "activation_matrix": activation_mat,
        "activation_matrix_raw": activation_mat_raw,
        "dense_activation": dense_activation,
        "dense_activation_raw": dense_activation_raw,
        "program_confidence_table": program_conf_df,
        "program_weight_info": program_weight_info,
        "program_confidence_summary": program_conf_summary,
        "program_allowlist_summary": allowlist_summary,
        "program_qc_selection_summary": qc_selection_summary,
        "program_qc_path": program_qc_path,
        "adjacency": adjacency,
        "spot_edges": edge_array,
        "graph_source": graph_source,
        "flow_graph_mode": flow_graph_mode,
    }


def load_domain_visualization_inputs(
    program_bundle_path: Path,
    cfg: DomainPipelineConfig,
) -> dict:
    cfg_vis = deepcopy(cfg)
    cfg_vis.input.program_use_qc_selection = False
    return load_domain_inputs(program_bundle_path=program_bundle_path, cfg=cfg_vis)
