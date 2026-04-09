from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ProgramInputConfig:
    gss_sparse_relpath: str = "gss/gss_sparse.parquet"
    gss_meta_relpath: str = "gss/gss_meta.json"
    gss_manifest_relpath: str = "manifest.json"
    h5ad_path_override: Optional[str] = None


@dataclass
class ProgramPreprocessConfig:
    min_gene_support_spots: Optional[int] = None
    min_gene_support_frac: float = 0.01
    max_gene_support_spots: Optional[int] = None
    max_gene_support_frac: float = 0.80
    min_gene_gss_mean_quantile: float = 0.20
    min_gene_local_contrast_quantile: float = 0.10
    gene_active_quantile: float = 0.80
    gene_active_min_gss: float = 0.001
    blacklist_prefixes: tuple[str, ...] = ("MT-", "RPL", "RPS")
    blacklist_genes: tuple[str, ...] = ("MALAT1", "NEAT1", "XIST")
    blacklist_action: Literal["downweight", "drop", "none"] = "downweight"
    blacklist_downweight_factor: float = 0.20

@dataclass
class ProgramActivationConfig:
    min_activation: float = 0.001
    adaptive_min_activation_quantile: float = 0.60
    identity_view_activation_quantile: float = 0.85
    high_contribution_gene_fraction: float = 0.35
    high_contribution_gene_min_count: int = 4
    high_contribution_gene_min_alignment: float = 0.55
    support_gene_min_alignment: float = 0.30
    prune_alignment_max: float = 0.15
    prune_activation_contribution_max: float = 0.02


@dataclass
class ProgramTemplateConfig:
    evidence_row_quantile: float = 0.80
    evidence_min_value: float = 1e-4
    evidence_min_genes_per_spot: int = 8

    nmf_k_grid: tuple[int, ...] = (12, 16, 24)
    nmf_max_iter: int = 400
    nmf_alpha_w: float = 0.0
    nmf_alpha_h: float = 1e-4
    nmf_l1_ratio: float = 0.3

    candidate_gene_weight_quantile: float = 0.80
    candidate_gene_fraction: float = 0.20
    candidate_gene_min_count: int = 15
    candidate_max_gene_count: int = 300

    candidate_activation_quantile: float = 0.75
    candidate_min_support_spots: int = 20
    candidate_min_support_frac: float = 0.01

    merge_activation_corr: float = 0.85
    merge_gene_jaccard: float = 0.30
    merge_activation_corr_strict: float = 0.92
    consensus_core_gene_min_run_frac: float = 0.60
    consensus_support_gene_min_run_frac: float = 0.40
    consensus_edge_gene_min_run_frac: float = 0.25
    min_template_run_support_frac: float = 0.20
    min_program_size_genes: int = 20
    max_program_gene_frac_warn: float = 0.20
    

@dataclass
class ProgramBootstrapConfig:
    enabled: bool = True
    bootstrap_B: int = 8
    early_stop_enabled: bool = True
    early_stop_min_rounds: int = 6
    early_stop_consecutive_rounds: int = 2
    early_stop_label_match: float = 0.990
    top_ns: tuple[int, ...] = (20, 50)


@dataclass
class ProgramQCConfig:
    housekeeping_activation_coverage_threshold: float = 0.70
    housekeeping_mean_gene_support_frac_threshold: float = 0.50
    blacklist_enrichment_threshold: float = 0.30
    drop_housekeeping_or_blacklist: bool = True
    dominance_warn_p50: float = 0.80

    require_rerun_for_validity: bool = True
    stable_high_contribution_gene_min_frequency: float = 0.50
    soft_max_activation_coverage: float = 0.70

    hard_fail_min_template_run_support_frac: float = 0.20
    hard_fail_min_activation_coverage: float = 0.02
    hard_fail_min_high_activation_spots: int = 12
    hard_fail_min_scaffold_content_quality: float = 0.60
    hard_fail_max_scaffold_hemoglobin_frac_top10: float = 0.35
    hard_fail_min_top20_jaccard_p50: float = 0.08
    hard_fail_min_rank_corr_p50: float = 0.30
    hard_fail_min_stable_high_contribution_gene_set_size: int = 3
    hard_fail_noise_max_main_component_frac: float = 0.20
    hard_fail_noise_max_top3_component_frac: float = 0.55
    hard_fail_noise_max_activation_peakiness: float = 0.88
    hard_fail_noise_max_activation_view_consistency: float = 0.35

    good_template_run_support_frac: float = 0.80
    good_template_spot_support_frac: float = 0.08
    good_template_focus_score: float = 0.08
    good_top20_jaccard_p50: float = 0.25
    good_rank_corr_p50: float = 0.60
    good_stable_high_contribution_gene_set_size: int = 8
    good_scaffold_content_quality: float = 0.95
    good_scaffold_gene_frac: float = 0.25
    good_activation_coverage: float = 0.10
    good_high_activation_spots: int = 64
    good_activation_peakiness: float = 0.90
    good_activation_view_consistency: float = 0.50
    good_head_consistency_score: float = 0.60
    min_main_component_frac: float = 0.20
    single_main_block_min_main_component_frac: float = 0.50
    good_top2_component_frac: float = 0.75
    good_top3_component_frac: float = 0.90

    default_use_min_program_confidence: float = 0.60
    default_use_min_support_score: float = 0.58
    default_use_min_validity_score: float = 0.58
    default_use_min_activation_presence_score: float = 0.56
    default_use_min_structure_score: float = 0.48
    default_use_min_scaffold_content_quality: float = 0.68
    high_program_confidence_threshold: float = 0.80

    redundancy_scaffold_overlap_threshold: float = 0.20
    redundancy_scaffold_rank_corr_threshold: float = 0.65
    redundancy_activation_overlap_threshold: float = 0.55
    redundancy_high_activation_overlap_threshold: float = 0.35
    redundancy_full_rank_corr_threshold: float = 0.70
    redundancy_top_contributing_gene_overlap_threshold: float = 0.60
    redundancy_min_instance_metric_hits_for_family: int = 1
    redundancy_min_instance_metric_hits_for_duplicate: int = 2


@dataclass
class ProgramPipelineConfig:
    schema_version: str = "programbundle.v2"
    random_seed: int = 2024
    code_version_override: Optional[str] = None
    input: ProgramInputConfig = field(default_factory=ProgramInputConfig)
    preprocess: ProgramPreprocessConfig = field(default_factory=ProgramPreprocessConfig)
    template: ProgramTemplateConfig = field(default_factory=ProgramTemplateConfig)
    activation: ProgramActivationConfig = field(default_factory=ProgramActivationConfig)
    bootstrap: ProgramBootstrapConfig = field(default_factory=ProgramBootstrapConfig)
    qc: ProgramQCConfig = field(default_factory=ProgramQCConfig)
