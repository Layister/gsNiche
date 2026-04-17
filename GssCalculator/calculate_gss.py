from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# Ensure package-style import works even when running this file directly:
# `python /.../GssCalculator/calculate_gss.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GssCalculator.pipeline import GSSPipelineConfig, run_gss_pipeline
from utils.dataset_registry import get_dataset
from utils.h5ad_schema import require_gss_h5ad_schema

warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


dataset = get_dataset("DLPFC")
sample_ids = dataset.sample_ids


pipeline_config = GSSPipelineConfig()
pipeline_config.preprocess.data_layer = dataset.data_layer
pipeline_config.preprocess.spot_id_field = dataset.spot_id_field
pipeline_config.latent.epochs = 300
pipeline_config.latent.n_neighbors = 11
pipeline_config.latent.spatial_graph_mode = "knn"
pipeline_config.latent.k_expr = 15
pipeline_config.latent.expr_embedding_source = "pca"
pipeline_config.latent.expr_pca_dim = 50
pipeline_config.latent.expr_metric = "cosine"
pipeline_config.latent.expr_gating_mode = "spatial_knn"
pipeline_config.latent.expr_gating_k = 20
pipeline_config.latent.anticollapse_enabled = True
pipeline_config.latent.anticollapse_var_weight = 1.0
pipeline_config.latent.anticollapse_cov_weight = 0.01
pipeline_config.latent.anticollapse_var_target = 1.0
pipeline_config.latent.anticollapse_start_epoch = 20
pipeline_config.neighbors.k = 20
pipeline_config.neighbors.k_spatial = 101
pipeline_config.gss.sparsify_rule = "topM"
pipeline_config.gss.top_m = 1000
pipeline_config.qc.bootstrap_repeats = 5


def main() -> None:
    for sample_id in sample_ids:
        print(f"Processing {dataset.dataset_id} / {sample_id} ...")
        h5ad_path = dataset.h5ad_path(sample_id)
        out_root = dataset.out_root()
        schema_report = require_gss_h5ad_schema(
            h5ad_path=h5ad_path,
            dataset=dataset,
            sample_id=sample_id,
        )
        print(schema_report.format_report())

        bundle_path = run_gss_pipeline(
            h5ad_path=h5ad_path,
            out_root=out_root,
            sample_id=sample_id,
            config=pipeline_config,
        )
        print(f"Done: {bundle_path}")


if __name__ == "__main__":
    main()
