from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# Ensure package-style import works even when running this file directly:
# `python /.../GssCalcu/calculate_gss.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GssCalculator.pipeline import GSSPipelineConfig, run_gss_pipeline

warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


# Source data root.
work_dir = Path("/Users/wuyang/Documents/SC-ST data")
cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]

"""
cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]

cancer = "PAAD"
sample_ids = ["NCBI569", "NCBI570", "NCBI571", "NCBI572"]

cancer = "IDC"
sample_ids = ["NCBI681", "NCBI682", "NCBI683", "TENX13", "TENX14"]

cancer = "PRAD"
sample_ids = ["INT25", "INT26", "INT27", "INT28", "TENX40", "TENX46"]

cancer = "EPM"
sample_ids = ["NCBI629", "NCBI630", "NCBI631", "NCBI632", "NCBI633"]

"""


pipeline_config = GSSPipelineConfig()
pipeline_config.preprocess.data_layer = "X"
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
        print(f"Processing {cancer} / {sample_id} ...")
        h5ad_path = work_dir / cancer / "ST" / f"{sample_id}.h5ad"
        out_root = work_dir / cancer / "ST"

        bundle_path = run_gss_pipeline(
            h5ad_path=h5ad_path,
            out_root=out_root,
            sample_id=sample_id,
            config=pipeline_config,
        )
        print(f"Done: {bundle_path}")


if __name__ == "__main__":
    main()
