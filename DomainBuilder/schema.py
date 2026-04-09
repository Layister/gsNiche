from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DomainInputConfig:
    program_manifest_relpath: str = "manifest.json"
    programs_relpath: str = "programs.parquet"
    program_activation_relpath: str = "program_activation.parquet"
    program_qc_relpath: str = "qc_tables/program_qc.parquet"
    gss_manifest_relpath: str = "manifest.json"
    neighbors_idx_relpath: str = "neighbors/neighbors.idx.npy"
    neighbors_meta_relpath: str = "neighbors/neighbors_meta.json"
    neighbors_spot_ids_relpath: str = "neighbors/spot_ids.npy"
    gss_bundle_path_override: Optional[str] = None
    try_load_coords_from_h5ad: bool = True
    program_allowlist: tuple[str, ...] = ()
    program_allowlist_file: Optional[str] = None
    program_use_qc_selection: bool = True
    allowed_validity_statuses: tuple[str, ...] = ("valid",)
    allowed_routing_statuses: tuple[str, ...] = ("default_use", "review_only")
    allowed_redundancy_statuses: tuple[str, ...] = ("retained_primary", "redundant_variant")


@dataclass
class PotentialConfig:
    flow_graph_mode: Literal["spatial", "gss"] = "spatial"
    spatial_graph_k: int = 18
    smoothing_enabled: bool = True
    smoothing_alpha: float = 0.60
    smoothing_steps: int = 5
    flow_epsilon: float = 1e-8
    enforce_spatial_entity: bool = True
    bridge_gap_spots: int = 2
    active_floor_abs: float = 1e-6
    active_floor_scale_factor: float = 0.01
    min_active_spots_per_program: int = 20
    merge_small_basins_enabled: bool = True
    merge_small_basin_max_spots: int = 3
    merge_small_basin_scale_factor: float = 1.2


@dataclass
class DomainFilterConfig:
    min_domain_spots: Optional[int] = 20
    min_domain_spots_frac: float = 0.002
    prominence_outside_quantile: float = 0.80
    min_domain_internal_density: float = 0.01


@dataclass
class DomainQCConfig:
    keep_rejected_domains_in_table: bool = True


@dataclass
class DomainAdjacencyConfig:
    mode: Literal["shared_boundary", "inverse_centroid_distance"] = "shared_boundary"
    min_shared_boundary_edges: int = 1
    centroid_distance_eps: float = 1e-6


@dataclass
class DomainMergeConfig:
    enabled: bool = False
    min_shared_boundary_edges: int = 2
    max_centroid_distance: float = 3.0
    max_peak_ratio: float = 1.30
    max_prominence_ratio: float = 1.50


@dataclass
class ProgramConfidenceConfig:
    enabled: bool = True
    confidence_col: str = "program_confidence"
    min_confidence: float = 0.05
    gamma: float = 1.0
    strict: bool = True


@dataclass
class DomainReliabilityConfig:
    enabled: bool = True
    confidence_exponent: float = 1.0
    prominence_exponent: float = 0.5
    density_exponent: float = 0.5
    prominence_scale_quantile: float = 0.90
    density_scale: float = 0.15
    min_node_reliability: float = 0.05
    max_node_reliability: float = 1.0


@dataclass
class DomainPipelineConfig:
    schema_version: str = "domainbundle.v1"
    random_seed: int = 2024
    code_version_override: Optional[str] = None
    input: DomainInputConfig = field(default_factory=DomainInputConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    filter: DomainFilterConfig = field(default_factory=DomainFilterConfig)
    qc: DomainQCConfig = field(default_factory=DomainQCConfig)
    adjacency: DomainAdjacencyConfig = field(default_factory=DomainAdjacencyConfig)
    merge: DomainMergeConfig = field(default_factory=DomainMergeConfig)
    program_confidence: ProgramConfidenceConfig = field(default_factory=ProgramConfidenceConfig)
    domain_reliability: DomainReliabilityConfig = field(default_factory=DomainReliabilityConfig)
