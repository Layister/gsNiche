from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CohortReportingConfig


@dataclass
class SampleReportingBundle:
    sample_id: str
    cancer_type: str
    representation_bundle_path: Path
    program_bundle_path: Path
    domain_bundle_path: Path | None
    niche_bundle_path: Path | None
    axis_definition: dict
    manifest: dict
    program_profile_df: pd.DataFrame
    program_burden_df: pd.DataFrame
    program_summary: dict
    program_pairs_df: pd.DataFrame
    program_cross_sample_summary: dict
    domain_profile_df: pd.DataFrame
    domain_burden_df: pd.DataFrame
    domain_summary: dict
    domain_pairs_df: pd.DataFrame
    domain_cross_sample_summary: dict
    niche_profile_df: pd.DataFrame
    niche_burden_df: pd.DataFrame
    niche_summary: dict
    niche_pairs_df: pd.DataFrame
    niche_cross_sample_summary: dict
    sample_summary: dict
    domains_df: pd.DataFrame
    domain_spot_membership_df: pd.DataFrame
    spot_coords_df: pd.DataFrame
    domain_program_map_df: pd.DataFrame
    domain_graph_df: pd.DataFrame
    niche_membership_df: pd.DataFrame
    niche_structures_df: pd.DataFrame


@dataclass
class CohortReportingInputs:
    out_root: Path
    sample_bundles: list[SampleReportingBundle]
    component_axes: list[str]
    role_axes: list[str]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_parquet_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df.copy()


def _resolve_domain_bundle_path(program_bundle_path: Path, program_manifest: dict) -> Path | None:
    raw = str(program_manifest.get("inputs", {}).get("domain_bundle_path", "")).strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    sibling = program_bundle_path.parent / "domain_bundle"
    return sibling if sibling.exists() else None


def _resolve_niche_bundle_path(program_bundle_path: Path, program_manifest: dict) -> Path | None:
    raw = str(program_manifest.get("inputs", {}).get("niche_bundle_path", "")).strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    sibling = program_bundle_path.parent / "niche_bundle"
    return sibling if sibling.exists() else None


def _iter_representation_bundles(out_root: Path, sample_ids: list[str] | tuple[str, ...] | None) -> list[Path]:
    if not out_root.exists():
        return []
    sample_filter = {str(x) for x in sample_ids} if sample_ids else None
    bundle_dirs: list[Path] = []
    for sample_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        if sample_filter and sample_dir.name not in sample_filter:
            continue
        bundle_dir = sample_dir / "representation_bundle"
        if (bundle_dir / "manifest.json").exists():
            bundle_dirs.append(bundle_dir)
    return bundle_dirs


def _normalize_string_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _resolve_gss_manifest_path(domain_bundle_path: Path | None, program_bundle_path: Path, representation_manifest: dict) -> Path | None:
    raw = ""
    if domain_bundle_path and (domain_bundle_path / "manifest.json").exists():
        try:
            raw = str(_read_json(domain_bundle_path / "manifest.json").get("inputs", {}).get("gss_manifest_path", "")).strip()
        except Exception:
            raw = ""
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    raw = str(representation_manifest.get("inputs", {}).get("gss_bundle_path", "")).strip()
    if raw:
        p = Path(raw) / "manifest.json"
        if p.exists():
            return p
    sibling = program_bundle_path.parent / "gss_bundle" / "manifest.json"
    return sibling if sibling.exists() else None


def _load_spot_coords_from_gss_manifest(gss_manifest_path: Path | None) -> pd.DataFrame:
    if gss_manifest_path is None or not gss_manifest_path.exists():
        return pd.DataFrame(columns=["spot_id", "x", "y"])
    try:
        manifest = _read_json(gss_manifest_path)
    except Exception:
        return pd.DataFrame(columns=["spot_id", "x", "y"])
    h5ad_path_raw = str(manifest.get("inputs", {}).get("h5ad_path", "")).strip()
    if not h5ad_path_raw:
        return pd.DataFrame(columns=["spot_id", "x", "y"])
    h5ad_path = Path(h5ad_path_raw)
    if not h5ad_path.exists():
        return pd.DataFrame(columns=["spot_id", "x", "y"])
    try:
        import h5py
    except Exception:
        return pd.DataFrame(columns=["spot_id", "x", "y"])
    try:
        with h5py.File(h5ad_path, "r") as f:
            if "obs" not in f or "obsm" not in f or "spatial" not in f["obsm"]:
                return pd.DataFrame(columns=["spot_id", "x", "y"])
            obs = f["obs"]
            key = "_index" if "_index" in obs else "index" if "index" in obs else None
            if key is None:
                return pd.DataFrame(columns=["spot_id", "x", "y"])
            raw = obs[key][:]
            if raw.dtype.kind in {"S", "O"}:
                spot_ids = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in raw.tolist()]
            else:
                spot_ids = [str(x) for x in raw.tolist()]
            coords = np.asarray(f["obsm"]["spatial"][:], dtype=np.float64)
            if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != len(spot_ids):
                return pd.DataFrame(columns=["spot_id", "x", "y"])
            return pd.DataFrame({"spot_id": np.asarray(spot_ids, dtype=str), "x": coords[:, 0], "y": coords[:, 1]})
    except Exception:
        return pd.DataFrame(columns=["spot_id", "x", "y"])


def _load_sample_bundle(bundle_dir: Path) -> SampleReportingBundle:
    manifest = _read_json(bundle_dir / "manifest.json")
    sample_id = str(manifest.get("sample_id") or bundle_dir.parent.name)
    cancer_type = str(manifest.get("cancer_type") or "")
    axis_definition = _read_json(bundle_dir / "axis_definition.json")
    program_bundle_path = Path(str(manifest.get("inputs", {}).get("program_bundle_path", "")))
    program_manifest = _read_json(program_bundle_path / "manifest.json")
    domain_bundle_path = _resolve_domain_bundle_path(program_bundle_path, program_manifest)
    niche_bundle_path = _resolve_niche_bundle_path(program_bundle_path, program_manifest)
    gss_manifest_path = _resolve_gss_manifest_path(domain_bundle_path, program_bundle_path, manifest)

    program_profile_df = _normalize_string_columns(
        _load_parquet_or_empty(bundle_dir / "program" / "macro_profile.parquet"),
        ("sample_id", "program_id", "component_primary_axis"),
    )
    program_burden_df = _load_parquet_or_empty(bundle_dir / "program" / "sample_burden.parquet")
    program_summary = _read_json(bundle_dir / "program" / "macro_summary.json")
    program_pairs_df = _normalize_string_columns(
        _load_parquet_or_empty(bundle_dir / "program" / "cross_sample_comparability.parquet"),
        ("sample_id_a", "sample_id_b", "program_id_a", "program_id_b"),
    )
    program_cross_sample_summary = _read_json(bundle_dir / "program" / "cross_sample_summary.json")

    domain_profile_df = _normalize_string_columns(
        _load_parquet_or_empty(bundle_dir / "domain" / "macro_profile.parquet"),
        ("sample_id", "domain_id", "domain_key", "source_program_id"),
    )
    domain_burden_df = _load_parquet_or_empty(bundle_dir / "domain" / "sample_burden.parquet")
    domain_summary = _read_json(bundle_dir / "domain" / "macro_summary.json")
    domain_pairs_df = _normalize_string_columns(
        _load_parquet_or_empty(bundle_dir / "domain" / "cross_sample_comparability.parquet"),
        ("sample_id_a", "sample_id_b", "domain_id_a", "domain_id_b", "source_program_id_a", "source_program_id_b"),
    )
    domain_cross_sample_summary = _read_json(bundle_dir / "domain" / "cross_sample_summary.json")

    niche_profile_df = _normalize_string_columns(
        _load_parquet_or_empty(bundle_dir / "niche" / "macro_profile.parquet"),
        ("sample_id", "niche_id", "canonical_pattern_id"),
    )
    niche_burden_df = _load_parquet_or_empty(bundle_dir / "niche" / "sample_burden.parquet")
    niche_summary = _read_json(bundle_dir / "niche" / "macro_summary.json")
    niche_pairs_df = _normalize_string_columns(
        _load_parquet_or_empty(bundle_dir / "niche" / "cross_sample_comparability.parquet"),
        ("sample_id_a", "sample_id_b", "niche_id_a", "niche_id_b"),
    )
    niche_cross_sample_summary = _read_json(bundle_dir / "niche" / "cross_sample_summary.json")

    sample_summary = _read_json(bundle_dir / "sample" / "macro_summary.json")
    domains_df = _normalize_string_columns(
        _load_parquet_or_empty(domain_bundle_path / "domains.parquet") if domain_bundle_path else pd.DataFrame(),
        ("sample_id", "domain_id", "domain_key", "program_seed_id"),
    )
    domain_spot_membership_df = _normalize_string_columns(
        _load_parquet_or_empty(domain_bundle_path / "domain_spot_membership.parquet") if domain_bundle_path else pd.DataFrame(),
        ("domain_key", "spot_id"),
    )
    spot_coords_df = _normalize_string_columns(
        _load_spot_coords_from_gss_manifest(gss_manifest_path),
        ("spot_id",),
    )
    domain_program_map_df = _normalize_string_columns(
        _load_parquet_or_empty(domain_bundle_path / "domain_program_map.parquet") if domain_bundle_path else pd.DataFrame(),
        ("domain_id", "domain_key", "program_seed_id"),
    )
    domain_graph_df = _normalize_string_columns(
        _load_parquet_or_empty(domain_bundle_path / "domain_graph.parquet") if domain_bundle_path else pd.DataFrame(),
        ("domain_key_i", "domain_key_j"),
    )
    niche_membership_df = _normalize_string_columns(
        _load_parquet_or_empty(niche_bundle_path / "niche_membership.parquet") if niche_bundle_path else pd.DataFrame(),
        ("niche_id", "domain_key", "program_id"),
    )
    niche_structures_df = _normalize_string_columns(
        _load_parquet_or_empty(niche_bundle_path / "niche_structures.parquet") if niche_bundle_path else pd.DataFrame(),
        ("niche_id", "canonical_pattern_id", "program_ids"),
    )

    return SampleReportingBundle(
        sample_id=sample_id,
        cancer_type=cancer_type,
        representation_bundle_path=bundle_dir,
        program_bundle_path=program_bundle_path,
        domain_bundle_path=domain_bundle_path,
        niche_bundle_path=niche_bundle_path,
        axis_definition=axis_definition,
        manifest=manifest,
        program_profile_df=program_profile_df,
        program_burden_df=program_burden_df,
        program_summary=program_summary,
        program_pairs_df=program_pairs_df,
        program_cross_sample_summary=program_cross_sample_summary,
        domain_profile_df=domain_profile_df,
        domain_burden_df=domain_burden_df,
        domain_summary=domain_summary,
        domain_pairs_df=domain_pairs_df,
        domain_cross_sample_summary=domain_cross_sample_summary,
        niche_profile_df=niche_profile_df,
        niche_burden_df=niche_burden_df,
        niche_summary=niche_summary,
        niche_pairs_df=niche_pairs_df,
        niche_cross_sample_summary=niche_cross_sample_summary,
        sample_summary=sample_summary,
        domains_df=domains_df,
        domain_spot_membership_df=domain_spot_membership_df,
        spot_coords_df=spot_coords_df,
        domain_program_map_df=domain_program_map_df,
        domain_graph_df=domain_graph_df,
        niche_membership_df=niche_membership_df,
        niche_structures_df=niche_structures_df,
    )


def load_cohort_reporting_inputs(
    out_root: str | Path,
    sample_ids: list[str] | tuple[str, ...] | None = None,
    cancer_type: str | None = None,
    config: CohortReportingConfig | None = None,
) -> CohortReportingInputs:
    _ = config or CohortReportingConfig()
    root = Path(out_root)
    bundle_dirs = _iter_representation_bundles(root, sample_ids)
    sample_bundles = [_load_sample_bundle(bundle_dir) for bundle_dir in bundle_dirs]
    if cancer_type:
        target = str(cancer_type).strip().upper()
        sample_bundles = [bundle for bundle in sample_bundles if str(bundle.cancer_type).upper() == target]
    if not sample_bundles:
        return CohortReportingInputs(out_root=root, sample_bundles=[], component_axes=[], role_axes=[])

    axis_definition = sample_bundles[0].axis_definition
    component_axes = [str(item.get("axis_name") or item.get("axis_id") or "") for item in axis_definition.get("component_axes", [])]
    role_axes = [str(item.get("axis_name") or item.get("axis_id") or "") for item in axis_definition.get("role_axes", [])]
    return CohortReportingInputs(
        out_root=root,
        sample_bundles=sample_bundles,
        component_axes=[axis for axis in component_axes if axis],
        role_axes=[axis for axis in role_axes if axis],
    )
