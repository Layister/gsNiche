from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class NicheInputConfig:
    domain_manifest_relpath: str = "manifest.json"
    domains_relpath: str = "domains.parquet"
    domain_membership_relpath: str = "domain_spot_membership.parquet"
    domain_graph_relpath: str = "domain_graph.parquet"
    domain_meta_relpath: str = "domain_meta.json"
    program_bundle_path_override: Optional[str] = None
    program_activation_relpath: str = "program_activation.parquet"
    program_qc_relpath: str = "qc_tables/program_qc.parquet"
    gss_bundle_path_override: Optional[str] = None
    neighbors_idx_relpath: str = "neighbors/neighbors.idx.npy"
    neighbors_spot_ids_relpath: str = "neighbors/spot_ids.npy"
    program_allowlist: tuple[str, ...] = ()
    program_allowlist_file: Optional[str] = None
    program_use_qc_selection: bool = True
    allowed_validity_statuses: tuple[str, ...] = ("valid",)
    allowed_routing_statuses: tuple[str, ...] = ("default_use", "review_only")
    allowed_redundancy_statuses: tuple[str, ...] = ("retained_primary", "redundant_variant")


@dataclass
class DomainReliabilityConfig:
    enabled: bool = True
    require_domain_fields: bool = True
    confidence_exponent: float = 1.0
    prominence_exponent: float = 0.5
    density_exponent: float = 0.5
    prominence_scale_quantile: float = 0.90
    density_scale: float = 0.15
    min_node_reliability: float = 0.05
    max_node_reliability: float = 1.0
    pair_mode: Literal["geometric_mean", "mean", "min"] = "geometric_mean"
    pair_power: float = 1.0
    edge_reliability_blend: float = 0.35


@dataclass
class DomainEdgeConfig:
    min_shared_boundary_edges: int = 1
    min_spatial_overlap: float = 0.0
    use_proximity_edges: bool = True
    epsilon_mode: Literal["spot_spacing", "quantile", "fixed"] = "spot_spacing"
    epsilon_distance: Optional[float] = None
    epsilon_distance_quantile: float = 0.25
    epsilon_spacing_multiplier: float = 1.5
    epsilon_spacing_quantile: float = 0.50
    robust_boundary_distance_trim_fraction: float = 0.10
    robust_boundary_distance_trim_k_min: int = 3
    robust_boundary_distance_symmetry: Literal["min", "mean"] = "min"
    robust_boundary_distance_use_boundary_spots: bool = True
    soft_redundant_when_overlap_or_contact: bool = True
    suppress_contact_when_overlap: bool = True
    boundary_fallback_perimeter_scale: float = 4.0
    contact_strength_mapping: Literal["identity", "rational", "tanh"] = "rational"
    contact_strength_saturation_c: float = 0.25
    shared_boundary_scale: float = 1.0
    spatial_overlap_scale: float = 2.0
    proximity_scale: float = 1.0
    strong_contact_quantile: float = 0.80
    strong_overlap_quantile: float = 0.80
    strong_contact_min_strength: float = 0.08
    strong_overlap_min_strength: float = 0.08


@dataclass
class InteractionDiscoveryConfig:
    seed_score_quantile: float = 0.60
    max_seeds_per_component: int = 12
    max_patterns_per_component: Optional[int] = None
    backbone_rank_quantile: float = 0.80
    expansion_rank_quantile: float = 0.70
    max_expansion_steps: Optional[int] = None
    proximity_join_min_core_links: int = 2
    proximity_join_min_program_neighbors: int = 2


@dataclass
class BasicNicheFilterConfig:
    min_program_count: int = 2
    min_cross_program_edges: int = 1
    min_core_program_pair_count: int = 1
    min_mean_edge_reliability: float = 0.20
    min_interaction_confidence: float = 0.20
    max_same_program_edge_fraction: float = 0.60
    min_core_pair_member_ratio: float = 0.20


@dataclass
class InteractionDedupConfig:
    backbone_overlap_threshold: float = 0.80
    core_overlap_threshold: float = 0.80
    strong_edge_signature_overlap_threshold: float = 0.75
    core_program_pair_overlap_threshold: float = 0.75
    representative_margin: float = 0.03


@dataclass
class RandomBaselineConfig:
    enabled: bool = False
    n_iter: int = 24
    min_non_random_score: float = 0.70
    min_pair_signature_overlap: float = 0.50
    max_backbone_edge_count_diff: int = 1
    max_program_count_diff: int = 1
    rng_seed_offset: int = 1000


@dataclass
class NichePipelineConfig:
    schema_version: str = "nichebundle.v1"
    random_seed: int = 2024
    code_version_override: Optional[str] = None
    input: NicheInputConfig = field(default_factory=NicheInputConfig)
    domain_reliability: DomainReliabilityConfig = field(default_factory=DomainReliabilityConfig)
    edge: DomainEdgeConfig = field(default_factory=DomainEdgeConfig)
    discovery: InteractionDiscoveryConfig = field(default_factory=InteractionDiscoveryConfig)
    basic_filter: BasicNicheFilterConfig = field(default_factory=BasicNicheFilterConfig)
    random_baseline: RandomBaselineConfig = field(default_factory=RandomBaselineConfig)
    dedup: InteractionDedupConfig = field(default_factory=InteractionDedupConfig)
