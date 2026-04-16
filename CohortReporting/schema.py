from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CohortReportingOutputConfig:
    cohort_dirname: str = "representation_cohort_presentation"
    sample_dirname: str = "presentation"
    figures_dirname: str = "figures"
    tables_dirname: str = "tables"
    sample_atlas_dirname: str = "sample_atlas"
    dashboard_dirname: str = "dashboard"
    dashboard_data_dirname: str = "data"
    manifest_filename: str = "manifest.json"


@dataclass(frozen=True)
class CohortReportingDisplayConfig:
    comparability_top_k: int = 3
    top_chains_per_pattern: int = 5
    top_panel_chains_per_pattern: int = 3
    figure_dpi: int = 220
    point_size_min: float = 40.0
    point_size_max: float = 220.0
    supported_axis_threshold_component: float = 0.15
    supported_axis_threshold_role: float = 0.15
    recompute_ordering: bool = False
    domain_spatial_viewer_min_programs: int = 1
    domain_spatial_viewer_max_programs: int = 5
    domain_spatial_viewer_default_program_count: int = 3
    domain_spatial_viewer_dense_selection_threshold: int = 4
    domain_spatial_viewer_default_view_mode: str = "overlay"
    domain_spatial_viewer_view_modes: tuple[str, ...] = ("fill", "boundary", "overlay")
    domain_spatial_viewer_default_selection: str = "recommended_or_top_burden"
    domain_spatial_viewer_dense_selection_note: str = "Current selection is dense; overlap readability may be reduced."
    niche_spatial_viewer_default_view_mode: str = "overlay"
    niche_spatial_viewer_view_modes: tuple[str, ...] = ("fill", "boundary", "overlay")


@dataclass(frozen=True)
class CohortReportingConfig:
    schema_version: str = "cohortreporting.v1"
    output: CohortReportingOutputConfig = field(default_factory=CohortReportingOutputConfig)
    display: CohortReportingDisplayConfig = field(default_factory=CohortReportingDisplayConfig)
