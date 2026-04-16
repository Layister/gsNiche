from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .schema import CohortReportingConfig
from .cross_sample_synthesis import TRIAGE_CLASSES


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_output_dirs(root: Path, cfg: CohortReportingConfig) -> dict[str, Path]:
    figures = root / cfg.output.figures_dirname
    tables = root / cfg.output.tables_dirname
    dashboard = root / cfg.output.dashboard_dirname
    data = dashboard / cfg.output.dashboard_data_dirname
    for path in (root, figures, tables, dashboard, data):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "figures": figures,
        "tables": tables,
        "dashboard": dashboard,
        "data": data,
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_contents(src_root: Path, dst_root: Path) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for path in src_root.rglob("*"):
        rel = path.relative_to(src_root)
        dst = dst_root / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: Path,
    cfg: CohortReportingConfig,
    sample_ids: list[str],
    sample_order: list[str],
    figures: list[str],
    tables: list[str],
    dashboard_files: list[str],
    sample_atlas_files: list[str],
    default_sample_id: str | None,
    *,
    axis_order: dict[str, list[str]] | None = None,
    sample_order_source: str = "computed",
) -> dict:
    _ = root
    return {
        "schema_version": cfg.schema_version,
        "created_at": iso_now(),
        "sample_ids": sample_ids,
        "sample_order": sample_order,
        "sample_order_source": sample_order_source,
        "default_sample_id": default_sample_id or "",
        "primary_reporting_mode": "sample_atlas_first",
        "primary_observation_unit": "within_sample_objects",
        "cross_sample_page": "Cross-sample Synthesis",
        "cross_sample_research_mode": "structure_synthesis",
        "cross_sample_result_classes": TRIAGE_CLASSES,
        "program_cross_sample_umap_enabled": True,
        "program_cross_sample_umap_role": "program_comparability_distribution_view",
        "program_cross_sample_umap_distance_source": "program_cross_sample_comparability",
        "program_cross_sample_umap_distance_imputation": "missing_pairs_set_to_1.0",
        "dashboard_mode": "read_only_report_browser",
        "dashboard_description": "Read-only report browser. Sample Atlas is the main reading path; cross-sample synthesis is a secondary result table view.",
        "axis_order": axis_order or {"component": [], "role": []},
        "supported_axis_threshold": {
            "component": float(cfg.display.supported_axis_threshold_component),
            "role": float(cfg.display.supported_axis_threshold_role),
        },
        "program_static_figure_name": "program_composition_overview",
        "domain_visualization_primary_mode": "one_static_matrix_plus_one_interactive_viewer",
        "domain_static_figure_name": "program_domain_deployment_matrix",
        "domain_spatial_viewer_enabled": True,
        "domain_spatial_viewer_min_programs": int(cfg.display.domain_spatial_viewer_min_programs),
        "domain_spatial_viewer_max_programs": int(cfg.display.domain_spatial_viewer_max_programs),
        "domain_spatial_viewer_default_program_count": int(cfg.display.domain_spatial_viewer_default_program_count),
        "domain_spatial_viewer_dense_selection_threshold": int(cfg.display.domain_spatial_viewer_dense_selection_threshold),
        "domain_spatial_viewer_dense_selection_note": cfg.display.domain_spatial_viewer_dense_selection_note,
        "domain_spatial_viewer_default_view_mode": cfg.display.domain_spatial_viewer_default_view_mode,
        "domain_spatial_viewer_view_modes": list(cfg.display.domain_spatial_viewer_view_modes),
        "domain_spatial_viewer_footprint_note": (
            "Viewer contours are generated from precomputed Domain spot memberships and spatial coordinates to show relative position, clustering, separation, and overlap trends. "
            "They are overlap-view contours for reporting, not exact boundary-survey reconstructions."
        ),
        "niche_visualization_primary_mode": "single_assembly_matrix_only",
        "niche_static_figure_name": "sample_level_niche_assembly_matrix",
        "niche_spatial_viewer_enabled": True,
        "niche_spatial_viewer_role": "interactive_niche_member_spatial_viewer",
        "niche_spatial_viewer_selection_mode": "single_niche_only",
        "niche_spatial_viewer_default_view_mode": cfg.display.niche_spatial_viewer_default_view_mode,
        "niche_spatial_viewer_view_modes": list(cfg.display.niche_spatial_viewer_view_modes),
        "niche_spatial_viewer_geometry_mode": "member_domain_spot_membership_contours",
        "niche_spatial_viewer_niche_order": "representative_then_confidence_then_member_count_then_niche_id",
        "niche_spatial_viewer_program_color_mode": "sample_stable_and_domain_viewer_aligned_when_available",
        "niche_spatial_viewer_footprint_note": (
            "Viewer contours are generated from precomputed member-Domain spot memberships and spatial coordinates to show relative position, separation, adjacency, and local overlap trends. "
            "They are local-assembly footprints for reporting, not exact boundary reconstructions."
        ),
        "outputs": {
            "figures": figures,
            "tables": tables,
            "dashboard": dashboard_files,
            "sample_atlas": sample_atlas_files,
        },
    }
