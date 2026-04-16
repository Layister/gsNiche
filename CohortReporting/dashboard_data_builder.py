from __future__ import annotations

from .cross_sample_synthesis import TRIAGE_CLASSES
from .loaders import CohortReportingInputs
from .schema import CohortReportingConfig


def build_dashboard_payloads(
    inputs: CohortReportingInputs,
    cfg: CohortReportingConfig,
    sample_order: list[str],
    sample_atlas_payloads: dict[str, dict],
    cross_sample_synthesis: dict,
    cross_sample_figures: dict | None = None,
) -> dict:
    _ = cfg
    return {
        "sample_atlas": sample_atlas_payloads,
        "cross_sample_synthesis": cross_sample_synthesis,
        "cross_sample_figures": cross_sample_figures or {},
        "app_config": {
            "sample_order": sample_order,
            "default_sample_id": sample_order[0] if sample_order else "",
            "home_page": "sample_atlas",
            "primary_reporting_mode": "sample_atlas_first",
            "primary_observation_unit": "within_sample_objects",
            "cross_sample_page": "Cross-sample Synthesis",
            "cross_sample_research_mode": "structure_synthesis",
            "cross_sample_result_classes": TRIAGE_CLASSES,
            "program_cross_sample_umap_enabled": True,
            "program_cross_sample_umap_role": "program_comparability_distribution_view",
            "program_cross_sample_umap_distance_source": "program_cross_sample_comparability",
            "program_cross_sample_umap_distance_imputation": "missing_pairs_set_to_1.0",
            "axis_order": {
                "component": inputs.component_axes,
                "role": inputs.role_axes,
            },
            "supported_axis_threshold": {
                "component": float(cfg.display.supported_axis_threshold_component),
                "role": float(cfg.display.supported_axis_threshold_role),
            },
            "program_static_figure_name": "program_composition_overview",
            "domain_static_figure_name": "program_domain_deployment_matrix",
            "domain_visualization_primary_mode": "one_static_matrix_plus_one_interactive_viewer",
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
            "niche_static_figure_name": "sample_level_niche_assembly_matrix",
            "niche_visualization_primary_mode": "single_assembly_matrix_only",
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
            "report_browser_mode": True,
            "description": "Read-only report browser. Sample Atlas is the main reading path; cross-sample synthesis is a secondary result table view.",
        },
    }
