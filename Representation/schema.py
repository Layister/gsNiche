from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AxisDefinition:
    axis_id: str
    axis_type: Literal["component", "role"]
    description: str
    evidence_sources: tuple[str, ...]
    axis_name: str = ""
    axis_description: str = ""
    score_range: tuple[float, float] = (0.0, 1.0)
    score_semantics: str = (
        "Independent support strength for this macro axis. "
        "It is not a probability, not a class membership, and not normalized across axes."
    )
    primary_evidence_sources: tuple[str, ...] = ()
    positive_gene_markers: tuple[str, ...] = ()
    positive_annotation_terms: tuple[str, ...] = ()
    topology_hints: tuple[str, ...] = ()
    negative_hints: tuple[str, ...] = ()
    gene_markers: tuple[str, ...] = ()
    annotation_keywords: tuple[str, ...] = ()
    topology_preferences: dict[str, float] = field(default_factory=dict)
    gating: dict[str, object] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.axis_name:
            object.__setattr__(self, "axis_name", self.axis_id)
        if not self.axis_description:
            object.__setattr__(self, "axis_description", self.description)
        if not self.primary_evidence_sources:
            object.__setattr__(self, "primary_evidence_sources", self.evidence_sources)
        if not self.positive_gene_markers:
            object.__setattr__(self, "positive_gene_markers", self.gene_markers)
        if not self.positive_annotation_terms:
            object.__setattr__(self, "positive_annotation_terms", self.annotation_keywords)
        if not self.topology_hints and self.topology_preferences:
            object.__setattr__(self, "topology_hints", tuple(self.topology_preferences.keys()))


@dataclass
class RepresentationInputConfig:
    program_manifest_relpath: str = "manifest.json"
    program_meta_relpath: str = "program_meta.json"
    programs_relpath: str = "programs.parquet"
    program_activation_relpath: str = "program_activation.parquet"
    program_qc_relpath: str = "qc_tables/program_qc.parquet"
    annotation_summary_relpath: str = "program_annotation/program_annotation_summary.json"
    domain_manifest_relpath: str = "manifest.json"
    domains_relpath: str = "domains.parquet"
    domain_program_map_relpath: str = "domain_program_map.parquet"
    domain_graph_relpath: str = "domain_graph.parquet"
    domain_qc_relpath: str = "qc_tables/domain_qc.parquet"
    niche_manifest_relpath: str = "manifest.json"
    niche_structures_relpath: str = "niche_structures.parquet"
    niche_membership_relpath: str = "niche_membership.parquet"
    niche_edges_relpath: str = "domain_adjacency_edges.parquet"
    representation_bundle_dirname: str = "representation_bundle"
    representation_program_profile_relpath: str = "program/macro_profile.parquet"
    representation_program_summary_relpath: str = "program/macro_summary.json"
    representation_domain_profile_relpath: str = "domain/macro_profile.parquet"
    representation_domain_summary_relpath: str = "domain/macro_summary.json"
    representation_niche_profile_relpath: str = "niche/macro_profile.parquet"
    representation_niche_summary_relpath: str = "niche/macro_summary.json"
    gss_manifest_relpath: str = "manifest.json"
    neighbors_idx_relpath: str = "neighbors/neighbors.idx.npy"
    neighbors_spot_ids_relpath: str = "neighbors/spot_ids.npy"
    allow_missing_annotation: bool = True
    prefer_activation_col: str = "activation_identity_view_weighted"
    fallback_activation_cols: tuple[str, ...] = ("activation_weighted", "activation_identity_view", "activation")


@dataclass
class RepresentationEligibilityConfig:
    allowed_validity_statuses: tuple[str, ...] = ("valid",)
    default_routing_statuses: tuple[str, ...] = ("default_use",)
    optional_routing_statuses: tuple[str, ...] = ("review_only",)
    include_high_confidence_review_only: bool = True
    high_confidence_review_only_threshold: float = 0.80


@dataclass
class RepresentationScoringConfig:
    top_gene_limit: int = 20
    scaffold_gene_limit: int = 12
    component_gene_weight: float = 0.75
    component_annotation_weight: float = 0.25
    role_topology_weight: float = 0.75
    role_annotation_weight: float = 0.15
    role_gene_weight: float = 0.10
    annotation_keyword_saturation: int = 3
    hotspot_top_fraction: float = 0.20
    activation_mass_log_scale: float = 2.5
    low_confidence_profile_threshold: float = 0.45
    low_information_gene_depth_threshold: float = 0.25
    low_information_annotation_threshold: float = 0.20
    low_information_topology_depth_threshold: float = 0.25
    representative_program_min_confidence: float = 0.45
    supported_axis_score_threshold: float = 0.15
    high_confidence_profile_threshold: float = 0.60
    component_representation_mass_weight: float = 0.35
    component_representation_primary_count_weight: float = 0.20
    component_representation_high_conf_count_weight: float = 0.20
    component_representation_representative_count_weight: float = 0.15
    component_representation_diversity_weight: float = 0.10
    component_representation_axis_score_threshold: float = 0.12
    component_dominant_representation_margin: float = 0.03
    component_dominant_burden_fallback_margin: float = 0.05
    node_like_sample_burden_attenuation: float = 0.72
    node_like_dominant_min_margin: float = 0.08
    node_like_dominant_min_confidence_weighted_burden: float = 0.40
    node_like_dominant_max_competing_role_burden: float = 0.36
    dominant_axis_top_k: int = 3
    top_programs_per_axis: int = 3
    cross_sample_component_similarity_weight: float = 0.65
    cross_sample_role_similarity_weight: float = 0.25
    cross_sample_optional_stats_weight: float = 0.10
    cross_sample_comparable_top_k: int = 3
    cross_sample_summary_top_pairs: int = 5
    cross_sample_nearest_samples_top_k: int = 3
    cross_sample_low_information_penalty: float = 0.85
    cross_sample_missing_annotation_penalty: float = 0.95
    cross_sample_missing_topology_penalty: float = 0.95
    domain_cross_sample_component_similarity_weight: float = 0.55
    domain_cross_sample_role_similarity_weight: float = 0.30
    domain_cross_sample_structure_similarity_weight: float = 0.15
    domain_cross_sample_comparable_top_k: int = 3
    domain_cross_sample_summary_top_pairs: int = 5
    domain_cross_sample_nearest_samples_top_k: int = 3
    niche_cross_sample_component_similarity_weight: float = 0.55
    niche_cross_sample_role_similarity_weight: float = 0.30
    niche_cross_sample_structure_similarity_weight: float = 0.15
    niche_cross_sample_comparable_top_k: int = 3
    niche_cross_sample_summary_top_pairs: int = 5
    niche_cross_sample_nearest_samples_top_k: int = 3


@dataclass
class RepresentationPipelineConfig:
    schema_version: str = "representationbundle.v1"
    code_version_override: Optional[str] = None
    default_cancer_type: str = "COAD"
    input: RepresentationInputConfig = field(default_factory=RepresentationInputConfig)
    eligibility: RepresentationEligibilityConfig = field(default_factory=RepresentationEligibilityConfig)
    scoring: RepresentationScoringConfig = field(default_factory=RepresentationScoringConfig)


@dataclass
class RepresentationInputBundle:
    sample_id: str
    cancer_type: str
    program_bundle_path: str
    gss_bundle_path: Optional[str]
    program_manifest: dict
    program_meta: dict
    programs_df: pd.DataFrame
    activation_df: pd.DataFrame
    program_qc_df: pd.DataFrame
    annotation_map: dict[str, dict]
    annotation_status: dict
    topology_status: dict
    spot_ids: Optional[np.ndarray]
    neighbor_idx: Optional[np.ndarray]
    total_spots: int
    activation_col: str


@dataclass
class ProgramEvidence:
    sample_id: str
    cancer_type: str
    program_id: str
    validity_status: str
    routing_status: str
    eligibility_status: str
    eligible_for_burden: bool
    program_confidence: float
    template_evidence_score: float
    default_use_support_score: float
    redundancy_status: str
    top_genes: tuple[str, ...]
    scaffold_genes: tuple[str, ...]
    annotation_term_ids: tuple[str, ...]
    annotation_summary_text: str
    annotation_confidence: float
    annotation_available: bool
    activation_mass: float
    activation_coverage: float
    active_spot_count: int
    activation_mean_active: float
    activation_hotspot_share: float
    activation_peakiness: float
    activation_entropy: float
    activation_sparsity: float
    main_component_frac: float
    high_activation_spot_count: int
    topology_available: bool
    topology_boundary_fraction: float
    topology_local_purity: float
    topology_component_count: int
    topology_component_density: float
