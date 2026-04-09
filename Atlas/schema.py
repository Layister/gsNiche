from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AtlasInputConfig:
    niche_manifest_relpath: str = "manifest.json"
    niche_annotation_dirname: str = "niche_annotation"
    interpretation_relpath: str = "niche_annotation/niche_interpretation.tsv"
    top_interfaces_relpath: str = "niche_annotation/niche_top_interfaces.tsv"
    allow_missing_annotation: bool = True


@dataclass
class AtlasFeatureConfig:
    local_top_interface_count: int = 6
    global_top_interface_count: int = 24
    global_top_term_count: int = 24
    interface_group_weight: float = 1.00
    term_group_weight: float = 0.70
    structural_group_weight: float = 0.40
    interaction_confidence_weight: float = 1.00
    backbone_size_weight: float = 0.80
    member_size_weight: float = 0.45
    edge_mix_weight: float = 0.40
    core_profile_weight: float = 0.35
    context_unresolved_weight: float = 0.20


@dataclass
class AtlasClusteringConfig:
    k_values: tuple[int, ...] = (3, 4, 5, 6)
    distance_metric: str = "cosine"
    linkage_method: str = "average"


@dataclass
class AtlasConfig:
    schema_version: str = "nicheatlas.v2"
    random_seed: int = 2024
    input: AtlasInputConfig = field(default_factory=AtlasInputConfig)
    feature: AtlasFeatureConfig = field(default_factory=AtlasFeatureConfig)
    clustering: AtlasClusteringConfig = field(default_factory=AtlasClusteringConfig)
