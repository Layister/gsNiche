from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class PreprocessConfig:
    data_layer: str = "X"
    min_cells: int = 1
    normalize_target_sum: float = 1e4
    hvg: int = 5000
    hvg_flavor: str = "seurat_v3"
    use_hvg_for_latent: bool = True
    remove_mt_genes: bool = True
    spot_id_field: Optional[str] = None
    gene_id_type: str = "symbol"


@dataclass
class LatentConfig:
    epochs: int = 300
    feat_hidden1: int = 256
    feat_hidden2: int = 128
    gat_hidden1: int = 64
    gat_hidden2: int = 30
    p_drop: float = 0.1
    gat_lr: float = 0.001
    weight_decay: float = 0.01
    n_neighbors: int = 11
    spatial_graph_mode: Literal["knn", "radius"] = "knn"
    spatial_radius: Optional[float] = None
    label_w: float = 1.0
    rec_w: float = 1.0
    weighted_adj: bool = False
    k_expr: int = 15
    expr_embedding_source: Literal["pca", "hvg"] = "pca"
    expr_pca_dim: int = 50
    expr_metric: Literal["cosine", "euclidean"] = "cosine"
    expr_gating_mode: Literal["spatial_knn", "radius", "hybrid"] = "spatial_knn"
    expr_gating_k: Optional[int] = None
    expr_gating_radius: Optional[float] = None
    union_reduce: Literal["max", "mean"] = "max"
    nheads: int = 3
    var: bool = False
    convergence_threshold: float = 1e-4
    anticollapse_enabled: bool = True
    anticollapse_var_weight: float = 1.0
    anticollapse_cov_weight: float = 0.01
    anticollapse_var_target: float = 1.0
    anticollapse_start_epoch: int = 20
    annotation_field: Optional[str] = None


@dataclass
class NeighborsConfig:
    k: int = 20
    k_spatial: int = 101
    candidate_mode: Literal["spatial_knn", "radius", "hybrid"] = "spatial_knn"
    spatial_radius: Optional[float] = None
    similarity_metric: Literal["cosine"] = "cosine"
    connectivity_hops: int = 2
    fallback: Literal["spatial", "decrease_k"] = "spatial"


@dataclass
class GSSConfig:
    no_expression_fraction: bool = False
    projection: Literal["exp2", "softplus"] = "exp2"
    sparsify_rule: Literal["topM", "positive"] = "topM"
    top_m: int = 3000
    keep_f_raw: bool = True
    keep_neighbor_support: bool = True


@dataclass
class QCConfig:
    bootstrap_repeats: int = 5
    bootstrap_latent_noise_std: float = 0.02
    bootstrap_top_n: tuple[int, ...] = (50, 100)
    neighbor_stability_gate_p50: float = 0.60
    gss_stability_gate_p50: float = 0.50


@dataclass
class GSSPipelineConfig:
    schema_version: str = "gssbundle.v1"
    random_seed: int = 2024
    code_version_override: Optional[str] = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    neighbors: NeighborsConfig = field(default_factory=NeighborsConfig)
    gss: GSSConfig = field(default_factory=GSSConfig)
    qc: QCConfig = field(default_factory=QCConfig)
