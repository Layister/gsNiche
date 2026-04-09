from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import RepresentationInputBundle, RepresentationPipelineConfig


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_cancer_type(program_bundle_path: Path, explicit: str | None, cfg: RepresentationPipelineConfig) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip().upper()
    parts = program_bundle_path.resolve().parts
    if len(parts) >= 4 and parts[-2] != "ST" and parts[-3] == "ST":
        return str(parts[-4]).strip().upper()
    if len(parts) >= 3 and parts[-3] not in {"", "ST"}:
        candidate = str(parts[-3]).strip().upper()
        if candidate and candidate != "ST":
            return candidate
    return str(cfg.default_cancer_type).strip().upper()


def _resolve_sample_id(program_manifest: dict, program_bundle_path: Path) -> str:
    sample_id = str(program_manifest.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return program_bundle_path.parent.name


def _resolve_total_spots(program_meta: dict, activation_df: pd.DataFrame) -> int:
    for key in ("n_spots",):
        try:
            value = int(program_meta.get(key, 0))
        except Exception:
            value = 0
        if value > 0:
            return value
    return int(activation_df["spot_id"].astype(str).nunique()) if "spot_id" in activation_df.columns else 0


def _resolve_activation_col(
    activation_df: pd.DataFrame,
    program_meta: dict,
    program_manifest: dict,
    cfg: RepresentationPipelineConfig,
) -> str:
    requested = str(cfg.input.prefer_activation_col).strip()
    if requested and requested in activation_df.columns:
        return requested
    meta_rec = str(program_meta.get("recommended_domain_input", "")).strip()
    if meta_rec and meta_rec in activation_df.columns:
        return meta_rec
    manifest_rec = (
        program_manifest.get("output_semantics", {})
        .get("program_activation", {})
        .get("recommended_domain_input", "")
    )
    manifest_rec = str(manifest_rec).strip()
    if manifest_rec and manifest_rec in activation_df.columns:
        return manifest_rec
    for col in cfg.input.fallback_activation_cols:
        if col in activation_df.columns:
            return str(col)
    raise ValueError(
        "program_activation.parquet does not contain any usable activation column. "
        f"Checked: {[requested, meta_rec, manifest_rec, *cfg.input.fallback_activation_cols]}"
    )


def _load_annotation_map(program_bundle_path: Path, cfg: RepresentationPipelineConfig) -> tuple[dict[str, dict], dict]:
    annotation_path = program_bundle_path / cfg.input.annotation_summary_relpath
    if not annotation_path.exists():
        if not bool(cfg.input.allow_missing_annotation):
            raise FileNotFoundError(f"Missing program annotation summary: {annotation_path}")
        return {}, {
            "available": False,
            "path": str(annotation_path),
            "reason": "missing_annotation_summary",
            "missing_policy": "annotation evidence contributes zero score and gene/topology evidence remain active",
        }

    payload = _read_json(annotation_path)
    if not isinstance(payload, list):
        raise ValueError(f"program annotation summary must be a list of records: {annotation_path}")
    out: dict[str, dict] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("program_id", "")).strip()
        if pid:
            out[pid] = dict(item)
    return out, {
        "available": True,
        "path": str(annotation_path),
        "reason": "loaded",
        "missing_policy": "not_used",
    }


def _resolve_gss_bundle_path(program_bundle_path: Path, program_manifest: dict) -> Path | None:
    raw = str(program_manifest.get("inputs", {}).get("gss_bundle_path", "")).strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    sibling = program_bundle_path.parent / "gss_bundle"
    if sibling.exists():
        return sibling
    return None


def _resolve_domain_bundle_path(program_bundle_path: Path, program_manifest: dict) -> Path | None:
    raw = str(program_manifest.get("inputs", {}).get("domain_bundle_path", "")).strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    sibling = program_bundle_path.parent / "domain_bundle"
    if sibling.exists():
        return sibling
    return None


def _resolve_niche_bundle_path(program_bundle_path: Path, program_manifest: dict) -> Path | None:
    raw = str(program_manifest.get("inputs", {}).get("niche_bundle_path", "")).strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    sibling = program_bundle_path.parent / "niche_bundle"
    if sibling.exists():
        return sibling
    return None


def _load_h5ad_spot_ids_from_gss_manifest(gss_bundle_path: Path, cfg: RepresentationPipelineConfig) -> np.ndarray | None:
    manifest_path = gss_bundle_path / cfg.input.gss_manifest_relpath
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    h5ad_path_raw = str(manifest.get("inputs", {}).get("h5ad_path", "")).strip()
    if not h5ad_path_raw:
        return None
    h5ad_path = Path(h5ad_path_raw)
    if not h5ad_path.exists():
        return None
    spot_field = str(manifest.get("inputs", {}).get("spot_id_field", "obs_names")).strip() or "obs_names"
    try:
        import scanpy as sc

        adata = sc.read_h5ad(h5ad_path)
        if spot_field == "obs_names":
            return adata.obs_names.astype(str).to_numpy()
        if spot_field in adata.obs.columns:
            raw = adata.obs[spot_field]
            if raw.isna().any():
                return None
            return raw.astype(str).to_numpy()
    except Exception:
        pass
    if spot_field == "obs_names":
        try:
            import h5py

            with h5py.File(h5ad_path, "r") as handle:
                if "obs" not in handle:
                    return None
                obs = handle["obs"]
                key = "_index" if "_index" in obs else "index" if "index" in obs else None
                if key is None:
                    return None
                raw = obs[key][:]
                if getattr(raw, "dtype", None) is not None and raw.dtype.kind in {"S", "O"}:
                    values = [
                        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                        for x in raw.tolist()
                    ]
                else:
                    values = [str(x) for x in raw.tolist()]
                return np.asarray(values, dtype=str)
        except Exception:
            return None
    return None


def _load_topology_inputs(
    gss_bundle_path: Path | None,
    cfg: RepresentationPipelineConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    if gss_bundle_path is None:
        return None, None, {
            "available": False,
            "reason": "missing_gss_bundle",
            "missing_policy": "role axes retain weak annotation/gene hints but topology-led evidence contributes zero support",
        }

    neighbors_path = gss_bundle_path / cfg.input.neighbors_idx_relpath
    spot_ids_path = gss_bundle_path / cfg.input.neighbors_spot_ids_relpath
    if not neighbors_path.exists():
        return None, None, {
            "available": False,
            "reason": "missing_neighbors_inputs",
            "neighbors_path": str(neighbors_path),
            "spot_ids_path": str(spot_ids_path),
            "missing_policy": "role axes retain weak annotation/gene hints but topology-led evidence contributes zero support",
        }

    neighbor_idx = np.load(neighbors_path, allow_pickle=False)
    if spot_ids_path.exists():
        spot_ids = np.load(spot_ids_path, allow_pickle=False).astype(str)
        spot_source = "neighbors_spot_ids"
    else:
        spot_ids = _load_h5ad_spot_ids_from_gss_manifest(gss_bundle_path, cfg)
        spot_source = "gss_manifest_h5ad"
        if spot_ids is None:
            return None, None, {
                "available": False,
                "reason": "missing_spot_ids_and_h5ad_fallback",
                "neighbors_path": str(neighbors_path),
                "spot_ids_path": str(spot_ids_path),
                "missing_policy": "role axes retain weak annotation/gene hints but topology-led evidence contributes zero support",
            }
    if neighbor_idx.ndim != 2 or spot_ids.ndim != 1 or neighbor_idx.shape[0] != spot_ids.shape[0]:
        return None, None, {
            "available": False,
            "reason": "invalid_neighbors_shape",
            "neighbors_shape": list(neighbor_idx.shape),
            "spot_ids_shape": list(spot_ids.shape),
            "missing_policy": "role axes retain weak annotation/gene hints but topology-led evidence contributes zero support",
        }
    return spot_ids.astype(str, copy=False), neighbor_idx.astype(np.int64, copy=False), {
        "available": True,
        "reason": "loaded",
        "neighbors_path": str(neighbors_path),
        "spot_ids_path": str(spot_ids_path),
        "spot_id_source": str(spot_source),
        "missing_policy": "not_used",
    }


def _ensure_required_program_columns(programs_df: pd.DataFrame) -> pd.DataFrame:
    if programs_df.empty:
        return programs_df
    required = {"program_id", "gene"}
    missing = sorted(required - set(programs_df.columns))
    if missing:
        raise ValueError(f"programs.parquet missing required columns: {missing}")
    out = programs_df.copy()
    out["program_id"] = out["program_id"].astype(str)
    out["gene"] = out["gene"].astype(str)
    return out


def _ensure_required_activation_columns(activation_df: pd.DataFrame) -> pd.DataFrame:
    if activation_df.empty:
        return activation_df
    required = {"program_id", "spot_id"}
    missing = sorted(required - set(activation_df.columns))
    if missing:
        raise ValueError(f"program_activation.parquet missing required columns: {missing}")
    out = activation_df.copy()
    out["program_id"] = out["program_id"].astype(str)
    out["spot_id"] = out["spot_id"].astype(str)
    return out


def load_representation_inputs(
    program_bundle_path: str | Path,
    cancer_type: str | None,
    cfg: RepresentationPipelineConfig,
) -> RepresentationInputBundle:
    bundle_path = Path(program_bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"program_bundle not found: {bundle_path}")

    manifest_path = bundle_path / cfg.input.program_manifest_relpath
    programs_path = bundle_path / cfg.input.programs_relpath
    activation_path = bundle_path / cfg.input.program_activation_relpath
    meta_path = bundle_path / cfg.input.program_meta_relpath
    qc_path = bundle_path / cfg.input.program_qc_relpath

    for path in (manifest_path, programs_path, activation_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Representation input: {path}")

    program_manifest = _read_json(manifest_path)
    program_meta = _read_json(meta_path)
    sample_id = _resolve_sample_id(program_manifest, bundle_path)
    resolved_cancer_type = infer_cancer_type(bundle_path, cancer_type, cfg)
    programs_df = _ensure_required_program_columns(pd.read_parquet(programs_path))
    activation_df = _ensure_required_activation_columns(pd.read_parquet(activation_path))
    program_qc_df = pd.read_parquet(qc_path) if qc_path.exists() else pd.DataFrame()
    if not program_qc_df.empty and "program_id" in program_qc_df.columns:
        program_qc_df = program_qc_df.copy()
        program_qc_df["program_id"] = program_qc_df["program_id"].astype(str)

    annotation_map, annotation_status = _load_annotation_map(bundle_path, cfg)
    gss_bundle_path = _resolve_gss_bundle_path(bundle_path, program_manifest)
    spot_ids, neighbor_idx, topology_status = _load_topology_inputs(gss_bundle_path, cfg)
    total_spots = _resolve_total_spots(program_meta, activation_df)
    activation_col = _resolve_activation_col(activation_df, program_meta, program_manifest, cfg)

    return RepresentationInputBundle(
        sample_id=sample_id,
        cancer_type=resolved_cancer_type,
        program_bundle_path=str(bundle_path.resolve()),
        gss_bundle_path=str(gss_bundle_path.resolve()) if gss_bundle_path else None,
        program_manifest=program_manifest,
        program_meta=program_meta,
        programs_df=programs_df,
        activation_df=activation_df,
        program_qc_df=program_qc_df,
        annotation_map=annotation_map,
        annotation_status=annotation_status,
        topology_status=topology_status,
        spot_ids=spot_ids,
        neighbor_idx=neighbor_idx,
        total_spots=int(total_spots),
        activation_col=activation_col,
    )


def build_eligibility_table(bundle: RepresentationInputBundle, cfg: RepresentationPipelineConfig) -> pd.DataFrame:
    base = pd.DataFrame({"program_id": sorted(bundle.programs_df["program_id"].astype(str).unique().tolist())})
    qc = bundle.program_qc_df.copy()
    if qc.empty:
        keep = [
            "program_id",
            "validity_status",
            "routing_status",
            "program_confidence",
            "default_use_support_score",
            "program_template_evidence_score",
            "redundancy_status",
        ]
        keep = [c for c in keep if c in bundle.programs_df.columns]
        qc = bundle.programs_df.loc[:, keep].drop_duplicates(subset=["program_id"]) if keep else base.copy()
    else:
        keep = [
            "program_id",
            "validity_status",
            "routing_status",
            "program_confidence",
            "default_use_support_score",
            "template_evidence_score",
            "program_template_evidence_score",
            "redundancy_status",
        ]
        keep = [c for c in keep if c in qc.columns]
        qc = qc.loc[:, keep].drop_duplicates(subset=["program_id"])

    out = base.merge(qc, on="program_id", how="left")
    out["validity_status"] = out.get("validity_status", pd.Series(dtype=object)).fillna("unknown").astype(str)
    out["routing_status"] = out.get("routing_status", pd.Series(dtype=object)).fillna("unknown").astype(str)
    out["program_confidence"] = pd.to_numeric(out.get("program_confidence", 0.0), errors="coerce").fillna(0.0)
    out["default_use_support_score"] = pd.to_numeric(
        out.get("default_use_support_score", 0.0), errors="coerce"
    ).fillna(0.0)
    if "template_evidence_score" not in out.columns:
        out["template_evidence_score"] = pd.to_numeric(
            out.get("program_template_evidence_score", 0.0), errors="coerce"
        ).fillna(0.0)
    else:
        out["template_evidence_score"] = pd.to_numeric(out["template_evidence_score"], errors="coerce").fillna(0.0)
    out["redundancy_status"] = out.get("redundancy_status", pd.Series(dtype=object)).fillna("unknown").astype(str)

    valid_allowed = {str(x).strip() for x in cfg.eligibility.allowed_validity_statuses if str(x).strip()}
    default_allowed = {str(x).strip() for x in cfg.eligibility.default_routing_statuses if str(x).strip()}
    optional_allowed = {str(x).strip() for x in cfg.eligibility.optional_routing_statuses if str(x).strip()}

    eligibility_status: list[str] = []
    eligible_mask: list[bool] = []
    for row in out.itertuples(index=False):
        validity_status = str(row.validity_status)
        routing_status = str(row.routing_status)
        confidence = float(row.program_confidence)
        if validity_status not in valid_allowed or routing_status in {"rejected", "invalid"}:
            eligibility_status.append("excluded_invalid_or_rejected")
            eligible_mask.append(False)
        elif routing_status in default_allowed:
            eligibility_status.append("eligible_default_use")
            eligible_mask.append(True)
        elif routing_status in optional_allowed:
            if bool(cfg.eligibility.include_high_confidence_review_only) and (
                confidence >= float(cfg.eligibility.high_confidence_review_only_threshold)
            ):
                eligibility_status.append("eligible_review_only_high_confidence")
                eligible_mask.append(True)
            else:
                eligibility_status.append("excluded_review_only_low_confidence")
                eligible_mask.append(False)
        else:
            eligibility_status.append("excluded_other")
            eligible_mask.append(False)

    out["eligibility_status"] = eligibility_status
    out["eligible_for_burden"] = np.asarray(eligible_mask, dtype=bool)
    return out.sort_values("program_id").reset_index(drop=True)


def load_domain_inputs(
    program_bundle_path: str | Path,
    cfg: RepresentationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    bundle_path = Path(program_bundle_path)
    manifest_path = bundle_path / cfg.input.program_manifest_relpath
    if not manifest_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "available": False,
            "status": "missing_program_manifest",
        }
    program_manifest = _read_json(manifest_path)
    domain_bundle_path = _resolve_domain_bundle_path(bundle_path, program_manifest)
    if domain_bundle_path is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "available": False,
            "status": "missing_domain_bundle",
        }
    domains_path = domain_bundle_path / cfg.input.domains_relpath
    domain_program_map_path = domain_bundle_path / cfg.input.domain_program_map_relpath
    domain_graph_path = domain_bundle_path / cfg.input.domain_graph_relpath
    missing = [str(p) for p in (domains_path, domain_program_map_path, domain_graph_path) if not p.exists()]
    if missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "available": False,
            "status": "missing_domain_inputs",
            "missing_paths": missing,
            "domain_bundle_path": str(domain_bundle_path),
        }
    domains_df = pd.read_parquet(domains_path)
    program_map_df = pd.read_parquet(domain_program_map_path)
    graph_df = pd.read_parquet(domain_graph_path)
    for df, key in ((domains_df, "domain_id"), (program_map_df, "domain_id")):
        if key in df.columns:
            df[key] = df[key].astype(str)
    for col in ("domain_key", "program_seed_id", "sample_id"):
        if col in domains_df.columns:
            domains_df[col] = domains_df[col].astype(str)
        if col in program_map_df.columns:
            program_map_df[col] = program_map_df[col].astype(str)
    for col in ("domain_key_i", "domain_key_j"):
        if col in graph_df.columns:
            graph_df[col] = graph_df[col].astype(str)
    return domains_df, program_map_df, graph_df, {
        "available": True,
        "status": "loaded",
        "domain_bundle_path": str(domain_bundle_path),
        "domains_path": str(domains_path),
        "domain_program_map_path": str(domain_program_map_path),
        "domain_graph_path": str(domain_graph_path),
    }


def load_niche_inputs(
    program_bundle_path: str | Path,
    cfg: RepresentationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    bundle_path = Path(program_bundle_path)
    manifest_path = bundle_path / cfg.input.program_manifest_relpath
    if not manifest_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "available": False,
            "status": "missing_program_manifest",
        }
    program_manifest = _read_json(manifest_path)
    niche_bundle_path = _resolve_niche_bundle_path(bundle_path, program_manifest)
    if niche_bundle_path is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "available": False,
            "status": "missing_niche_bundle",
        }
    structures_path = niche_bundle_path / cfg.input.niche_structures_relpath
    membership_path = niche_bundle_path / cfg.input.niche_membership_relpath
    edges_path = niche_bundle_path / cfg.input.niche_edges_relpath
    missing = [str(p) for p in (structures_path, membership_path) if not p.exists()]
    if missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "available": False,
            "status": "missing_niche_inputs",
            "missing_paths": missing,
            "niche_bundle_path": str(niche_bundle_path),
        }
    structures_df = pd.read_parquet(structures_path)
    membership_df = pd.read_parquet(membership_path)
    edges_df = pd.read_parquet(edges_path) if edges_path.exists() else pd.DataFrame()
    for df, key in ((structures_df, "niche_id"), (membership_df, "niche_id")):
        if key in df.columns:
            df[key] = df[key].astype(str)
    for col in ("domain_key", "program_id", "canonical_pattern_id"):
        if col in structures_df.columns:
            structures_df[col] = structures_df[col].astype(str)
        if col in membership_df.columns:
            membership_df[col] = membership_df[col].astype(str)
    for col in ("domain_key_i", "domain_key_j", "domain_id_i", "domain_id_j"):
        if col in edges_df.columns:
            edges_df[col] = edges_df[col].astype(str)
    return structures_df, membership_df, edges_df, {
        "available": True,
        "status": "loaded",
        "niche_bundle_path": str(niche_bundle_path),
        "niche_structures_path": str(structures_path),
        "niche_membership_path": str(membership_path),
        "niche_edges_path": str(edges_path),
    }


def load_cross_sample_program_profiles(
    out_root: str | Path,
    current_sample_id: str,
    cancer_type: str | None,
    cfg: RepresentationPipelineConfig,
) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    root = Path(out_root)
    if not root.exists():
        return pd.DataFrame(), {}, {
            "available": False,
            "status": "missing_out_root",
            "out_root": str(root),
        }

    frames: list[pd.DataFrame] = []
    summary_map: dict[str, dict] = {}
    loaded_samples: list[str] = []
    errors: list[dict] = []
    target_cancer = str(cancer_type or "").strip().upper()
    for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        sample_id = str(sample_dir.name)
        if sample_id == str(current_sample_id):
            continue
        bundle_dir = sample_dir / cfg.input.representation_bundle_dirname
        profile_path = bundle_dir / cfg.input.representation_program_profile_relpath
        if not profile_path.exists():
            continue
        try:
            df = pd.read_parquet(profile_path)
        except Exception as exc:
            errors.append({"sample_id": sample_id, "path": str(profile_path), "error": str(exc)})
            continue
        if df.empty:
            continue
        df = df.copy()
        if "sample_id" in df.columns:
            df["sample_id"] = df["sample_id"].astype(str)
        else:
            df["sample_id"] = sample_id
        if "program_id" in df.columns:
            df["program_id"] = df["program_id"].astype(str)
        if target_cancer and "cancer_type" in df.columns:
            df = df.loc[df["cancer_type"].astype(str).str.upper() == target_cancer].copy()
        if df.empty:
            continue
        frames.append(df)
        loaded_samples.append(sample_id)
        summary_path = bundle_dir / cfg.input.representation_program_summary_relpath
        if summary_path.exists():
            try:
                summary_map[sample_id] = _read_json(summary_path)
            except Exception as exc:
                errors.append({"sample_id": sample_id, "path": str(summary_path), "error": str(exc)})

    if not frames:
        return pd.DataFrame(), summary_map, {
            "available": False,
            "status": "no_reference_samples",
            "out_root": str(root),
            "loaded_samples": loaded_samples,
            "errors": errors,
        }

    return pd.concat(frames, ignore_index=True), summary_map, {
        "available": True,
        "status": "loaded",
        "out_root": str(root),
        "loaded_samples": loaded_samples,
        "reference_sample_count": len(loaded_samples),
        "errors": errors,
    }


def load_cross_sample_domain_profiles(
    out_root: str | Path,
    current_sample_id: str,
    cancer_type: str | None,
    cfg: RepresentationPipelineConfig,
) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    root = Path(out_root)
    if not root.exists():
        return pd.DataFrame(), {}, {
            "available": False,
            "status": "missing_out_root",
            "out_root": str(root),
        }

    frames: list[pd.DataFrame] = []
    summary_map: dict[str, dict] = {}
    loaded_samples: list[str] = []
    errors: list[dict] = []
    target_cancer = str(cancer_type or "").strip().upper()
    for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        sample_id = str(sample_dir.name)
        if sample_id == str(current_sample_id):
            continue
        bundle_dir = sample_dir / cfg.input.representation_bundle_dirname
        profile_path = bundle_dir / cfg.input.representation_domain_profile_relpath
        if not profile_path.exists():
            continue
        try:
            df = pd.read_parquet(profile_path)
        except Exception as exc:
            errors.append({"sample_id": sample_id, "path": str(profile_path), "error": str(exc)})
            continue
        if df.empty:
            continue
        df = df.copy()
        if "sample_id" in df.columns:
            df["sample_id"] = df["sample_id"].astype(str)
        else:
            df["sample_id"] = sample_id
        if "domain_id" in df.columns:
            df["domain_id"] = df["domain_id"].astype(str)
        if target_cancer and "cancer_type" in df.columns:
            df = df.loc[df["cancer_type"].astype(str).str.upper() == target_cancer].copy()
        if df.empty:
            continue
        frames.append(df)
        loaded_samples.append(sample_id)
        summary_path = bundle_dir / cfg.input.representation_domain_summary_relpath
        if summary_path.exists():
            try:
                summary_map[sample_id] = _read_json(summary_path)
            except Exception as exc:
                errors.append({"sample_id": sample_id, "path": str(summary_path), "error": str(exc)})

    if not frames:
        return pd.DataFrame(), summary_map, {
            "available": False,
            "status": "no_reference_samples",
            "out_root": str(root),
            "loaded_samples": loaded_samples,
            "errors": errors,
        }

    return pd.concat(frames, ignore_index=True), summary_map, {
        "available": True,
        "status": "loaded",
        "out_root": str(root),
        "loaded_samples": loaded_samples,
        "reference_sample_count": len(loaded_samples),
        "errors": errors,
    }


def load_cross_sample_niche_profiles(
    out_root: str | Path,
    current_sample_id: str,
    cancer_type: str | None,
    cfg: RepresentationPipelineConfig,
) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    root = Path(out_root)
    if not root.exists():
        return pd.DataFrame(), {}, {
            "available": False,
            "status": "missing_out_root",
            "out_root": str(root),
        }

    frames: list[pd.DataFrame] = []
    summary_map: dict[str, dict] = {}
    loaded_samples: list[str] = []
    errors: list[dict] = []
    target_cancer = str(cancer_type or "").strip().upper()
    for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        sample_id = str(sample_dir.name)
        if sample_id == str(current_sample_id):
            continue
        bundle_dir = sample_dir / cfg.input.representation_bundle_dirname
        profile_path = bundle_dir / cfg.input.representation_niche_profile_relpath
        if not profile_path.exists():
            continue
        try:
            df = pd.read_parquet(profile_path)
        except Exception as exc:
            errors.append({"sample_id": sample_id, "path": str(profile_path), "error": str(exc)})
            continue
        if df.empty:
            continue
        df = df.copy()
        if "sample_id" in df.columns:
            df["sample_id"] = df["sample_id"].astype(str)
        else:
            df["sample_id"] = sample_id
        if "niche_id" in df.columns:
            df["niche_id"] = df["niche_id"].astype(str)
        if target_cancer and "cancer_type" in df.columns:
            df = df.loc[df["cancer_type"].astype(str).str.upper() == target_cancer].copy()
        if df.empty:
            continue
        frames.append(df)
        loaded_samples.append(sample_id)
        summary_path = bundle_dir / cfg.input.representation_niche_summary_relpath
        if summary_path.exists():
            try:
                summary_map[sample_id] = _read_json(summary_path)
            except Exception as exc:
                errors.append({"sample_id": sample_id, "path": str(summary_path), "error": str(exc)})

    if not frames:
        return pd.DataFrame(), summary_map, {
            "available": False,
            "status": "no_reference_samples",
            "out_root": str(root),
            "loaded_samples": loaded_samples,
            "errors": errors,
        }

    return pd.concat(frames, ignore_index=True), summary_map, {
        "available": True,
        "status": "loaded",
        "out_root": str(root),
        "loaded_samples": loaded_samples,
        "reference_sample_count": len(loaded_samples),
        "errors": errors,
    }
