from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bundle_io import read_json
from .schema import NichePipelineConfig


def _resolve_program_allowlist(cfg: NichePipelineConfig) -> tuple[set[str], dict]:
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
    program_bundle_path: Path,
    cfg: NichePipelineConfig,
    known_program_ids: set[str],
) -> tuple[set[str], dict]:
    if not bool(getattr(cfg.input, "program_use_qc_selection", True)):
        return known_program_ids, {
            "enabled": False,
            "selected_count": int(len(known_program_ids)),
            "dropped_count": 0,
        }

    program_qc_path = program_bundle_path / cfg.input.program_qc_relpath
    if not program_qc_path.exists():
        return known_program_ids, {
            "enabled": True,
            "selected_count": int(len(known_program_ids)),
            "dropped_count": 0,
            "fallback": "missing_program_qc",
        }

    qc = pd.read_parquet(program_qc_path)
    if qc.empty or "program_id" not in qc.columns:
        return known_program_ids, {
            "enabled": True,
            "selected_count": int(len(known_program_ids)),
            "dropped_count": 0,
            "fallback": "empty_program_qc",
        }

    qc["program_id"] = qc["program_id"].astype(str)
    qc = qc.drop_duplicates(subset=["program_id"], keep="first")
    selected = qc["program_id"].astype(str).isin(known_program_ids)
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
        selected_programs = known_program_ids
        summary["fallback"] = "empty_after_qc_filter"

    dropped_programs = sorted(known_program_ids - selected_programs)
    summary["selected_count"] = int(len(selected_programs))
    summary["dropped_count"] = int(len(dropped_programs))
    summary["dropped_examples"] = dropped_programs[:10]
    return selected_programs, summary


def _resolve_domain_reliability(domains_df: pd.DataFrame, cfg: NichePipelineConfig) -> tuple[pd.DataFrame, dict]:
    out = domains_df.copy()

    required = {"domain_reliability"}
    if bool(cfg.domain_reliability.require_domain_fields):
        required |= {
            "domain_confidence_component",
            "domain_prominence_component",
            "domain_density_component",
        }

    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(
            "domains.parquet missing DomainBuilder-provided reliability fields: "
            f"{missing}. Rerun DomainBuilder before Niche."
        )

    rel = pd.to_numeric(out["domain_reliability"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    rel = np.clip(rel, 0.0, 1.0)
    out["domain_reliability"] = rel.astype(np.float32)

    comp_cols = [
        "domain_confidence_component",
        "domain_prominence_component",
        "domain_density_component",
    ]
    comp_means: dict[str, float] = {}
    for col in comp_cols:
        if col in out.columns:
            vv = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            vv = np.clip(vv, 0.0, 1.0)
            out[col] = vv.astype(np.float32)
            comp_means[f"{col}_mean"] = float(np.mean(vv)) if vv.size > 0 else float("nan")

    return out, {
        "source": "domain_bundle",
        "enabled": bool(cfg.domain_reliability.enabled),
        "missing_required_fields": [],
        "reliability_min": float(np.min(rel)) if rel.size > 0 else float("nan"),
        "reliability_max": float(np.max(rel)) if rel.size > 0 else float("nan"),
        "reliability_mean": float(np.mean(rel)) if rel.size > 0 else float("nan"),
        "low_reliability_frac": float(np.mean(rel <= 0.05)) if rel.size > 0 else 0.0,
        **comp_means,
    }


def _try_load_spot_ids_and_coords_from_h5ad(gss_bundle_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    manifest_path = gss_bundle_path / "manifest.json"
    if not manifest_path.exists():
        return None, None

    try:
        manifest = read_json(manifest_path)
    except Exception:  # noqa: BLE001
        return None, None

    h5ad_path_raw = str(manifest.get("inputs", {}).get("h5ad_path", "")).strip()
    if not h5ad_path_raw:
        return None, None

    h5ad_path = Path(h5ad_path_raw)
    if not h5ad_path.exists():
        return None, None

    spot_field = str(manifest.get("inputs", {}).get("spot_id_field", "obs_names")).strip()
    if not spot_field:
        spot_field = "obs_names"

    spot_ids: np.ndarray | None = None
    coords: np.ndarray | None = None

    # Primary path: scanpy/anndata (handles arbitrary obs field robustly).
    try:
        import scanpy as sc

        adata = sc.read_h5ad(h5ad_path)
        if spot_field == "obs_names":
            spot_ids = adata.obs_names.astype(str).to_numpy()
        elif spot_field in adata.obs.columns:
            raw = adata.obs[spot_field]
            if raw.isna().any():
                return None, None
            spot_ids = raw.astype(str).to_numpy()

        if "spatial" in adata.obsm:
            coords_arr = np.asarray(adata.obsm["spatial"], dtype=np.float64)
            if coords_arr.ndim == 2 and coords_arr.shape[1] >= 2 and coords_arr.shape[0] == spot_ids.shape[0]:
                coords = coords_arr[:, :2].copy()
    except Exception:  # noqa: BLE001
        spot_ids = None

    # Secondary path: lightweight h5py fallback for obs_names + obsm/spatial.
    if spot_ids is None and spot_field == "obs_names":
        try:
            import h5py

            with h5py.File(h5ad_path, "r") as f:
                if "obs" not in f:
                    return None, None
                obs = f["obs"]
                key = "_index" if "_index" in obs else "index" if "index" in obs else None
                if key is None:
                    return None, None
                raw = obs[key][:]
                if raw.dtype.kind in {"S", "O"}:
                    vals = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in raw.tolist()]
                else:
                    vals = [str(x) for x in raw.tolist()]
                spot_ids = np.asarray(vals, dtype=str)

                if "obsm" in f and "spatial" in f["obsm"]:
                    coords_arr = np.asarray(f["obsm"]["spatial"][:], dtype=np.float64)
                    if coords_arr.ndim == 2 and coords_arr.shape[1] >= 2 and coords_arr.shape[0] == spot_ids.shape[0]:
                        coords = coords_arr[:, :2].copy()
        except Exception:  # noqa: BLE001
            spot_ids = None

    if spot_ids is None or spot_ids.size == 0:
        return None, None
    if int(np.unique(spot_ids).shape[0]) != int(spot_ids.shape[0]):
        return None, None
    return spot_ids.astype(str), coords


def _resolve_program_bundle_path(
    domain_bundle_path: Path,
    cfg: NichePipelineConfig,
    domain_manifest: dict,
) -> Path:
    if cfg.input.program_bundle_path_override:
        path = Path(cfg.input.program_bundle_path_override)
        if not path.exists():
            raise FileNotFoundError(f"Configured program_bundle_path_override does not exist: {path}")
        return path

    inputs = domain_manifest.get("inputs", {})
    program_path = inputs.get("program_bundle_path", None)
    if program_path:
        path = Path(str(program_path))
        if path.exists():
            return path

    sibling = domain_bundle_path.parent / "program_bundle"
    if sibling.exists():
        return sibling

    raise FileNotFoundError(
        "Unable to locate program_bundle path from domain manifest. "
        "Set NichePipelineConfig.input.program_bundle_path_override explicitly."
    )


def _resolve_gss_bundle_path(
    domain_bundle_path: Path,
    program_bundle_path: Path,
    cfg: NichePipelineConfig,
    domain_manifest: dict,
) -> Path:
    if cfg.input.gss_bundle_path_override:
        path = Path(cfg.input.gss_bundle_path_override)
        if not path.exists():
            raise FileNotFoundError(f"Configured gss_bundle_path_override does not exist: {path}")
        return path

    inputs = domain_manifest.get("inputs", {})
    gss_path = inputs.get("gss_bundle_path", None)
    if gss_path:
        path = Path(str(gss_path))
        if path.exists():
            return path

    manifest_path = program_bundle_path / "manifest.json"
    if manifest_path.exists():
        program_manifest = read_json(manifest_path)
        prog_inputs = program_manifest.get("inputs", {})
        gss_path = prog_inputs.get("gss_bundle_path", None)
        if gss_path:
            path = Path(str(gss_path))
            if path.exists():
                return path

    sibling = domain_bundle_path.parent / "gss_bundle"
    if sibling.exists():
        return sibling

    raise FileNotFoundError(
        "Unable to locate gss_bundle path from domain/program manifest. "
        "Set NichePipelineConfig.input.gss_bundle_path_override explicitly."
    )


def _resolve_spot_ids_and_coords_from_raw(
    gss_bundle_path: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    # Canonical strategy (v3): always recover spot order from raw h5ad referenced in gss manifest.
    spot_ids_h5ad, coords_h5ad = _try_load_spot_ids_and_coords_from_h5ad(gss_bundle_path=gss_bundle_path)
    if spot_ids_h5ad is None:
        raise ValueError(
            "Cannot recover canonical spot order from raw data. "
            "Ensure gss_bundle/manifest.json contains a readable inputs.h5ad_path "
            "and valid spot_id_field (or obs_names)."
        )
    if coords_h5ad is None:
        raise ValueError(
            "Cannot recover spatial coordinates from raw h5ad. "
            "Ensure raw h5ad contains adata.obsm['spatial'] with shape (n_spots, >=2)."
        )
    return spot_ids_h5ad, np.asarray(coords_h5ad, dtype=np.float64), "gss_manifest_h5ad"


def _read_domain_bundle_inputs(domain_bundle_path: Path, cfg: NichePipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    manifest_path = domain_bundle_path / cfg.input.domain_manifest_relpath
    domains_path = domain_bundle_path / cfg.input.domains_relpath
    membership_path = domain_bundle_path / cfg.input.domain_membership_relpath
    graph_path = domain_bundle_path / cfg.input.domain_graph_relpath
    meta_path = domain_bundle_path / cfg.input.domain_meta_relpath

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing domain manifest: {manifest_path}")
    if not domains_path.exists():
        raise FileNotFoundError(f"Missing domains parquet: {domains_path}")
    if not membership_path.exists():
        raise FileNotFoundError(f"Missing domain membership parquet: {membership_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing domain graph parquet: {graph_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing domain meta json: {meta_path}")

    domain_manifest = read_json(manifest_path)
    domains_df = pd.read_parquet(domains_path)
    membership_df = pd.read_parquet(membership_path)
    domain_graph_df = pd.read_parquet(graph_path)
    domain_meta = read_json(meta_path)
    return domains_df, membership_df, domain_graph_df, domain_manifest, domain_meta


def _read_program_activation(program_bundle_path: Path, cfg: NichePipelineConfig) -> pd.DataFrame:
    activation_path = program_bundle_path / cfg.input.program_activation_relpath
    if not activation_path.exists():
        return pd.DataFrame(columns=["program_id", "spot_id", "activation", "rank_in_spot"])

    activation_df = pd.read_parquet(activation_path)
    required = {"program_id", "spot_id", "activation"}
    missing = required - set(activation_df.columns)
    if missing:
        raise ValueError(
            f"program_activation missing required columns: {sorted(missing)}; found={list(activation_df.columns)}"
        )

    out = activation_df.loc[:, ["program_id", "spot_id", "activation"]].copy()
    out["program_id"] = out["program_id"].astype(str)
    out["spot_id"] = out["spot_id"].astype(str)
    out["activation"] = out["activation"].to_numpy(dtype=np.float32)
    out = out[out["activation"] > 0].reset_index(drop=True)
    return out


def _read_neighbors(
    gss_bundle_path: Path,
    cfg: NichePipelineConfig,
    required: bool,
) -> tuple[np.ndarray | None, Path | None]:
    neighbors_idx_path = gss_bundle_path / cfg.input.neighbors_idx_relpath
    if not neighbors_idx_path.exists():
        if required:
            raise FileNotFoundError(f"Missing neighbors idx: {neighbors_idx_path}")
        return None, None

    neighbors_idx = np.load(neighbors_idx_path, allow_pickle=False)
    if neighbors_idx.ndim != 2:
        raise ValueError(f"neighbors.idx.npy must be 2D, got shape={neighbors_idx.shape}")

    return neighbors_idx, neighbors_idx_path


def load_niche_inputs(
    domain_bundle_path: Path,
    cfg: NichePipelineConfig,
    require_neighbors: bool,
) -> dict:
    program_allowlist, allowlist_summary = _resolve_program_allowlist(cfg)
    domains_df, membership_df, domain_graph_df, domain_manifest, domain_meta = _read_domain_bundle_inputs(
        domain_bundle_path=domain_bundle_path,
        cfg=cfg,
    )

    required_domain_cols = {"domain_id", "domain_key", "program_seed_id", "spot_count", "qc_pass"}
    required_domain_cols |= {"domain_reliability"}
    if bool(cfg.domain_reliability.require_domain_fields):
        required_domain_cols |= {
            "domain_confidence_component",
            "domain_prominence_component",
            "domain_density_component",
        }
    missing = required_domain_cols - set(domains_df.columns)
    if missing:
        raise ValueError(f"domains.parquet missing required columns: {sorted(missing)}")

    domains_df = domains_df.copy()
    domains_df["domain_id"] = domains_df["domain_id"].astype(str)
    domains_df["domain_key"] = domains_df["domain_key"].astype(str)
    domains_df["program_seed_id"] = domains_df["program_seed_id"].astype(str)
    domains_df["qc_pass"] = domains_df["qc_pass"].astype(bool)
    if program_allowlist:
        known_programs = set(domains_df["program_seed_id"].unique().tolist())
        missing_programs = sorted(program_allowlist - known_programs)
        if missing_programs:
            raise ValueError(
                f"Requested program_allowlist contains unknown program_id in domain bundle: {missing_programs[:10]}"
            )
        keep_domain_keys = set(
            domains_df.loc[domains_df["program_seed_id"].isin(program_allowlist), "domain_key"].astype(str).tolist()
        )
        domains_df = domains_df.loc[domains_df["program_seed_id"].isin(program_allowlist)].reset_index(drop=True)
        membership_df = membership_df.loc[membership_df["domain_key"].astype(str).isin(keep_domain_keys)].reset_index(drop=True)
        domain_graph_df = domain_graph_df.loc[
            domain_graph_df["domain_key_i"].astype(str).isin(keep_domain_keys)
            & domain_graph_df["domain_key_j"].astype(str).isin(keep_domain_keys)
        ].reset_index(drop=True)
    domains_df, domain_reliability_summary = _resolve_domain_reliability(domains_df=domains_df, cfg=cfg)

    required_membership_cols = {"domain_key", "spot_id"}
    missing = required_membership_cols - set(membership_df.columns)
    if missing:
        raise ValueError(f"domain_spot_membership.parquet missing required columns: {sorted(missing)}")
    membership_df = membership_df.copy()
    membership_df["domain_key"] = membership_df["domain_key"].astype(str)
    membership_df["spot_id"] = membership_df["spot_id"].astype(str)
    if "spot_idx" in membership_df.columns:
        membership_df["spot_idx"] = membership_df["spot_idx"].to_numpy(dtype=np.int64)

    graph_required = {
        "domain_key_i",
        "domain_key_j",
        "shared_boundary_edges",
        "spatial_overlap",
        "edge_weight",
    }
    missing = graph_required - set(domain_graph_df.columns)
    if missing:
        raise ValueError(f"domain_graph.parquet missing required columns: {sorted(missing)}")

    domain_graph_df = domain_graph_df.copy()
    domain_graph_df["domain_key_i"] = domain_graph_df["domain_key_i"].astype(str)
    domain_graph_df["domain_key_j"] = domain_graph_df["domain_key_j"].astype(str)

    program_bundle_path = _resolve_program_bundle_path(
        domain_bundle_path=domain_bundle_path,
        cfg=cfg,
        domain_manifest=domain_manifest,
    )
    gss_bundle_path = _resolve_gss_bundle_path(
        domain_bundle_path=domain_bundle_path,
        program_bundle_path=program_bundle_path,
        cfg=cfg,
        domain_manifest=domain_manifest,
    )

    spot_ids, spot_coords, spot_order_source = _resolve_spot_ids_and_coords_from_raw(
        gss_bundle_path=gss_bundle_path,
    )

    activation_df = _read_program_activation(program_bundle_path=program_bundle_path, cfg=cfg)
    known_programs = set(domains_df["program_seed_id"].astype(str).unique().tolist())
    qc_selected_programs, qc_selection_summary = _resolve_program_qc_selection(
        program_bundle_path=program_bundle_path,
        cfg=cfg,
        known_program_ids=known_programs,
    )
    keep_domain_keys = set(
        domains_df.loc[domains_df["program_seed_id"].isin(qc_selected_programs), "domain_key"].astype(str).tolist()
    )
    domains_df = domains_df.loc[domains_df["program_seed_id"].isin(qc_selected_programs)].reset_index(drop=True)
    membership_df = membership_df.loc[membership_df["domain_key"].astype(str).isin(keep_domain_keys)].reset_index(drop=True)
    domain_graph_df = domain_graph_df.loc[
        domain_graph_df["domain_key_i"].astype(str).isin(keep_domain_keys)
        & domain_graph_df["domain_key_j"].astype(str).isin(keep_domain_keys)
    ].reset_index(drop=True)
    activation_df = activation_df.loc[
        activation_df["program_id"].astype(str).isin(qc_selected_programs)
    ].reset_index(drop=True)
    if program_allowlist:
        activation_df = activation_df.loc[
            activation_df["program_id"].astype(str).isin(program_allowlist)
        ].reset_index(drop=True)
    neighbors_idx, neighbors_idx_path = _read_neighbors(
        gss_bundle_path=gss_bundle_path,
        cfg=cfg,
        required=bool(require_neighbors),
    )
    if neighbors_idx is not None and int(neighbors_idx.shape[0]) != int(spot_ids.shape[0]):
        raise ValueError(
            "Raw h5ad spot count mismatch with neighbors rows "
            f"({spot_ids.shape[0]} vs {neighbors_idx.shape[0]})."
        )

    return {
        "domains_df": domains_df,
        "membership_df": membership_df,
        "domain_graph_df": domain_graph_df,
        "domain_manifest": domain_manifest,
        "domain_meta": domain_meta,
        "program_bundle_path": program_bundle_path,
        "gss_bundle_path": gss_bundle_path,
        "activation_df": activation_df,
        "neighbors_idx": neighbors_idx,
        "spot_ids": spot_ids,
        "spot_coords": spot_coords,
        "spot_order_source": spot_order_source,
        "neighbors_idx_path": neighbors_idx_path,
        "domain_reliability_summary": domain_reliability_summary,
        "program_allowlist_summary": allowlist_summary,
        "program_qc_selection_summary": qc_selection_summary,
    }
