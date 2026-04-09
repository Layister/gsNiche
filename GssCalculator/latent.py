from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA

try:
    from .schema import LatentConfig, PreprocessConfig
except ImportError:
    from GssCalculator.schema import LatentConfig, PreprocessConfig

logger = logging.getLogger(__name__)


def resolve_spot_ids(adata: sc.AnnData, spot_id_field: str | None) -> tuple[np.ndarray, str, bool]:
    if spot_id_field and spot_id_field in adata.obs.columns:
        raw_values = adata.obs[spot_id_field]
        candidate = raw_values.astype(str).to_numpy()
        if not raw_values.isna().any() and len(candidate) == len(set(candidate)):
            return candidate, spot_id_field, False

    obs_names = adata.obs_names.astype(str).to_numpy()
    if len(obs_names) == len(set(obs_names)):
        return obs_names, "obs_names", False

    generated = np.array([f"spot_{i:07d}" for i in range(adata.n_obs)], dtype=object)
    return generated, "generated", True


def prepare_expression_and_latent_input(
    adata: sc.AnnData,
    cfg: PreprocessConfig,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray | None]:
    if cfg.data_layer != "X":
        if cfg.data_layer not in adata.layers:
            raise ValueError(f"Data layer {cfg.data_layer} is not found in adata.layers")
        adata.X = adata.layers[cfg.data_layer].copy()

    if adata.n_vars == 0:
        raise ValueError("No genes available after filtering.")

    n_top_genes = max(1, min(cfg.hvg, adata.n_vars))
    try:
        sc.pp.highly_variable_genes(adata, flavor=cfg.hvg_flavor, n_top_genes=n_top_genes)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to compute HVG with flavor %s, fallback to seurat. reason=%s",
            cfg.hvg_flavor,
            exc,
        )
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)

    sc.pp.normalize_total(adata, target_sum=cfg.normalize_target_sum)
    sc.pp.log1p(adata)

    expression = adata.X
    if not sp.issparse(expression):
        expression = sp.csr_matrix(np.asarray(expression, dtype=np.float32))
    elif not isinstance(expression, sp.csr_matrix):
        expression = expression.tocsr()

    expression = expression.astype(np.float32)

    hvg_mask = adata.var["highly_variable"].to_numpy() if "highly_variable" in adata.var else None
    if cfg.use_hvg_for_latent and hvg_mask is not None and np.any(hvg_mask):
        latent_input = expression[:, hvg_mask]
    else:
        latent_input = expression

    latent_input_arr = latent_input.toarray() if sp.issparse(latent_input) else np.asarray(latent_input)
    latent_input_arr = latent_input_arr.astype(np.float32, copy=False)

    return expression, latent_input_arr, hvg_mask


def compute_latent(
    adata: sc.AnnData,
    latent_input: np.ndarray,
    latent_cfg: LatentConfig,
    random_seed: int,
) -> tuple[np.ndarray, dict]:
    latent, meta = _compute_gat_latent(adata, latent_input, latent_cfg, random_seed)
    return latent, meta


def _compute_gat_latent(
    adata: sc.AnnData,
    latent_input: np.ndarray,
    latent_cfg: LatentConfig,
    random_seed: int,
) -> tuple[np.ndarray, dict]:
    try:
        from .GNN.adjacency_matrix import construct_adjacency_matrix
        from .GNN.train import ModelTrainer
    except ImportError:
        from GssCalculator.GNN.adjacency_matrix import construct_adjacency_matrix
        from GssCalculator.GNN.train import ModelTrainer

    args = SimpleNamespace(
        random_seed=random_seed,
        epochs=latent_cfg.epochs,
        feat_hidden1=latent_cfg.feat_hidden1,
        feat_hidden2=latent_cfg.feat_hidden2,
        feat_cell=latent_input.shape[1],
        gat_hidden1=latent_cfg.gat_hidden1,
        gat_hidden2=latent_cfg.gat_hidden2,
        p_drop=latent_cfg.p_drop,
        gat_lr=latent_cfg.gat_lr,
        weight_decay=latent_cfg.weight_decay,
        n_neighbors=latent_cfg.n_neighbors,
        spatial_graph_mode=latent_cfg.spatial_graph_mode,
        spatial_radius=latent_cfg.spatial_radius,
        label_w=latent_cfg.label_w,
        rec_w=latent_cfg.rec_w,
        weighted_adj=latent_cfg.weighted_adj,
        k_expr=latent_cfg.k_expr,
        expr_embedding_source=latent_cfg.expr_embedding_source,
        expr_pca_dim=latent_cfg.expr_pca_dim,
        expr_metric=latent_cfg.expr_metric,
        expr_gating_mode=latent_cfg.expr_gating_mode,
        expr_gating_k=latent_cfg.expr_gating_k,
        expr_gating_radius=latent_cfg.expr_gating_radius,
        union_reduce=latent_cfg.union_reduce,
        nheads=latent_cfg.nheads,
        var=latent_cfg.var,
        convergence_threshold=latent_cfg.convergence_threshold,
        anticollapse_enabled=latent_cfg.anticollapse_enabled,
        anticollapse_var_weight=latent_cfg.anticollapse_var_weight,
        anticollapse_cov_weight=latent_cfg.anticollapse_cov_weight,
        anticollapse_var_target=latent_cfg.anticollapse_var_target,
        anticollapse_start_epoch=latent_cfg.anticollapse_start_epoch,
    )

    label = None
    if latent_cfg.annotation_field and latent_cfg.annotation_field in adata.obs.columns:
        label = pd.factorize(adata.obs[latent_cfg.annotation_field].astype(str))[0]

    node_x = latent_input

    expr_embedding = _build_expr_embedding(latent_input, latent_cfg, random_seed)

    graph_dict = construct_adjacency_matrix(
        adata,
        args,
        expr_embedding=expr_embedding,
        verbose=False,
    )
    trainer = ModelTrainer(node_x, graph_dict, args, label)
    trainer.run_train()
    latent = trainer.get_latent().astype(np.float32, copy=False)

    return latent, {
        "method": "gat_ae",
        "epochs": latent_cfg.epochs,
        "training": trainer.get_training_summary(),
        "graph": graph_dict.get("graph_meta", {}),
    }


def _build_expr_embedding(
    latent_input: np.ndarray,
    latent_cfg: LatentConfig,
    random_seed: int,
) -> np.ndarray:
    source = latent_cfg.expr_embedding_source
    if source == "hvg":
        return latent_input.astype(np.float32, copy=False)

    if source == "pca":
        max_dim = min(latent_input.shape[0] - 1, latent_input.shape[1])
        max_dim = max(1, max_dim)
        n_comp = max(1, min(int(latent_cfg.expr_pca_dim), max_dim))
        pca = PCA(n_components=n_comp, random_state=random_seed)
        emb = pca.fit_transform(latent_input)
        return emb.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported expr_embedding_source: {source}")
