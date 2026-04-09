from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

try:
    from .schema import (
        GSSConfig,
        GSSPipelineConfig,
        LatentConfig,
        NeighborsConfig,
        PreprocessConfig,
        QCConfig,
    )
    from .bundle_io import (
        ensure_bundle_dirs,
        get_code_version,
        hash_file,
        iso_now,
        promote_bundle,
        set_random_seed,
        write_json,
        write_parquet,
    )
    from .neighbors import build_neighbors, build_spatial_candidates
    from .metrics import compute_neighbor_qc, compute_stability_qc
    from .latent import compute_latent, prepare_expression_and_latent_input, resolve_spot_ids
    from .gss_compute import compute_and_sparsify_gss
except ImportError:
    from GssCalculator.schema import (
        GSSConfig,
        GSSPipelineConfig,
        LatentConfig,
        NeighborsConfig,
        PreprocessConfig,
        QCConfig,
    )
    from GssCalculator.bundle_io import (
        ensure_bundle_dirs,
        get_code_version,
        hash_file,
        iso_now,
        promote_bundle,
        set_random_seed,
        write_json,
        write_parquet,
    )
    from GssCalculator.neighbors import build_neighbors, build_spatial_candidates
    from GssCalculator.metrics import compute_neighbor_qc, compute_stability_qc
    from GssCalculator.latent import (
        compute_latent,
        prepare_expression_and_latent_input,
        resolve_spot_ids,
    )
    from GssCalculator.gss_compute import compute_and_sparsify_gss

logger = logging.getLogger(__name__)


__all__ = [
    "PreprocessConfig",
    "LatentConfig",
    "NeighborsConfig",
    "GSSConfig",
    "QCConfig",
    "GSSPipelineConfig",
    "run_gss_pipeline",
]


def run_gss_pipeline(
    h5ad_path: str | Path,
    out_root: str | Path,
    sample_id: str,
    config: GSSPipelineConfig | None = None,
) -> Path:
    cfg = config or GSSPipelineConfig()
    set_random_seed(cfg.random_seed)

    h5ad_path = Path(h5ad_path)
    out_root = Path(out_root)
    sample_dir = out_root / sample_id
    final_bundle = sample_dir / "gss_bundle"
    tmp_bundle = sample_dir / f"gss_bundle.__tmp__{int(time.time())}_{os.getpid()}"

    if tmp_bundle.exists():
        shutil.rmtree(tmp_bundle)

    ensure_bundle_dirs(tmp_bundle)

    try:
        adata = sc.read_h5ad(h5ad_path)
        n_spots_raw = int(adata.n_obs)
        n_genes_raw = int(adata.n_vars)
        adata.var_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=cfg.preprocess.min_cells)

        spot_ids, spot_id_field, spot_id_generated = resolve_spot_ids(
            adata, cfg.preprocess.spot_id_field
        )
        expression, latent_input, hvg_mask = prepare_expression_and_latent_input(adata, cfg.preprocess)
        gene_names = adata.var_names.to_numpy().astype(str)

        latent, latent_meta = compute_latent(adata, latent_input, cfg.latent, cfg.random_seed)
        np.save(tmp_bundle / "latent" / "latent.gat_ae.npy", latent)

        if "spatial" not in adata.obsm:
            raise ValueError("Missing adata.obsm['spatial']; required for neighbors building.")

        candidate_idx, candidate_dist, spatial_adj = build_spatial_candidates(
            adata.obsm["spatial"],
            cfg.neighbors,
        )

        neighbor_idx, neighbor_sim, neighbors_meta, connectivity_sets = build_neighbors(
            latent,
            candidate_idx,
            candidate_dist,
            spatial_adj,
            cfg.neighbors,
        )
        np.save(tmp_bundle / "neighbors" / "neighbors.idx.npy", neighbor_idx)
        np.save(tmp_bundle / "neighbors" / "neighbors.sim.npy", neighbor_sim)

        gss_payload = compute_and_sparsify_gss(
            expression=expression,
            gene_names=gene_names,
            spot_ids=spot_ids,
            neighbor_idx=neighbor_idx,
            gss_cfg=cfg.gss,
            remove_mt=cfg.preprocess.remove_mt_genes,
            top_n_for_qc=cfg.qc.bootstrap_top_n,
        )
        write_parquet(gss_payload["sparse_df"], tmp_bundle / "gss" / "gss_sparse.parquet")

        neighbor_qc_payload = compute_neighbor_qc(
            coords=adata.obsm["spatial"],
            neighbor_idx=neighbor_idx,
            neighbor_sim=neighbor_sim,
            connectivity_sets=connectivity_sets,
            radius=cfg.neighbors.spatial_radius,
            return_per_spot=True,
        )
        neighbor_qc = neighbor_qc_payload["summary"]
        neighbor_per_spot = neighbor_qc_payload["per_spot"]

        stability_qc = compute_stability_qc(
            latent=latent,
            expression=expression,
            base_neighbor_idx=neighbor_idx,
            base_top_sets=gss_payload["top_sets"],
            candidate_idx=candidate_idx,
            candidate_dist=candidate_dist,
            spatial_adj=spatial_adj,
            neighbors_cfg=cfg.neighbors,
            gss_cfg=cfg.gss,
            top_ns=cfg.qc.bootstrap_top_n,
            repeats=cfg.qc.bootstrap_repeats,
            noise_std=cfg.qc.bootstrap_latent_noise_std,
            seed=cfg.random_seed,
            remove_mt=cfg.preprocess.remove_mt_genes,
            gene_names=gene_names,
            return_per_spot=True,
        )
        stability_per_spot = stability_qc.pop("per_spot")

        neighbor_per_spot_df = pd.DataFrame(
            {
                "spot_id": spot_ids,
                "neighbor_locality": neighbor_per_spot["neighbor_locality"],
                "neighbor_homogeneity_mean": neighbor_per_spot["neighbor_homogeneity_mean"],
                "neighbor_homogeneity_var": neighbor_per_spot["neighbor_homogeneity_var"],
                "neighbor_connectivity": neighbor_per_spot["neighbor_connectivity"],
                "within_radius_ratio": neighbor_per_spot["within_radius_ratio"],
            }
        )
        write_parquet(
            neighbor_per_spot_df,
            tmp_bundle / "qc" / "qc_tables" / "neighbor_per_spot.parquet",
        )

        stability_table_payload = {
            "spot_id": spot_ids,
            "neighbor_stability_jaccard": stability_per_spot["neighbor_stability_jaccard"],
        }
        for top_n, values in stability_per_spot["gss_topN_stability_jaccard"].items():
            stability_table_payload[f"gss_top{top_n}_stability_jaccard"] = values
        stability_per_spot_df = pd.DataFrame(stability_table_payload)
        write_parquet(
            stability_per_spot_df,
            tmp_bundle / "qc" / "qc_tables" / "stability_per_spot.parquet",
        )

        qc_report = {
            "sample_id": sample_id,
            "summary": {
                "n_spots": int(adata.n_obs),
                "n_genes": int(expression.shape[1]),
                "bootstrap_repeats": cfg.qc.bootstrap_repeats,
            },
            "neighbor_quality": neighbor_qc,
            "neighbor_stability_jaccard": stability_qc["neighbor_stability_jaccard"],
            "gss_topN_stability_jaccard": stability_qc["gss_topN_stability_jaccard"],
            "gss_distribution": gss_payload["distribution"],
            "acceptance": {
                "neighbor_stability_p50_ge": cfg.qc.neighbor_stability_gate_p50,
                "neighbor_stability_p50": stability_qc["neighbor_stability_jaccard"]["quantiles"][
                    "p50"
                ],
                "neighbor_stability_pass": stability_qc["neighbor_stability_jaccard"]["quantiles"][
                    "p50"
                ]
                >= cfg.qc.neighbor_stability_gate_p50,
                "gss_top100_stability_p50_ge": cfg.qc.gss_stability_gate_p50,
                "gss_top100_stability_p50": stability_qc["gss_topN_stability_jaccard"]
                .get("100", {})
                .get("quantiles", {})
                .get("p50", 0.0),
                "gss_top100_stability_pass": stability_qc["gss_topN_stability_jaccard"]
                .get("100", {})
                .get("quantiles", {})
                .get("p50", 0.0)
                >= cfg.qc.gss_stability_gate_p50,
            },
        }

        latent_meta.update(
            {
                "shape": [int(latent.shape[0]), int(latent.shape[1])],
                "hvg_used": bool(cfg.preprocess.use_hvg_for_latent),
                "hvg_count": int(np.count_nonzero(hvg_mask))
                if hvg_mask is not None
                else int(expression.shape[1]),
                "random_seed": cfg.random_seed,
            }
        )
        neighbors_meta.update(
            {
                "shape": [int(neighbor_idx.shape[0]), int(neighbor_idx.shape[1])],
                "spot_id_field": spot_id_field,
                "spot_id_generated": spot_id_generated,
            }
        )

        gss_meta = {
            "projection": cfg.gss.projection,
            "sparsify_rule": cfg.gss.sparsify_rule,
            "top_m": cfg.gss.top_m,
            "remove_mt_genes": cfg.preprocess.remove_mt_genes,
            "spot_count": int(adata.n_obs),
            "gene_count": int(expression.shape[1]),
            "sparse_rows": int(gss_payload["sparse_df"].shape[0]),
            "columns": list(gss_payload["sparse_df"].columns),
            "distribution": gss_payload["distribution"],
        }

        write_json(tmp_bundle / "latent" / "latent_meta.json", latent_meta)
        write_json(tmp_bundle / "neighbors" / "neighbors_meta.json", neighbors_meta)
        write_json(tmp_bundle / "gss" / "gss_meta.json", gss_meta)
        write_json(tmp_bundle / "qc" / "qc_report.json", qc_report)

        manifest = {
            "schema_version": cfg.schema_version,
            "created_at": iso_now(),
            "sample_id": sample_id,
            "code_version": get_code_version(
                repo_root=Path(__file__).resolve().parents[1],
                override=cfg.code_version_override,
            ),
            "random_seed": cfg.random_seed,
            "inputs": {
                "h5ad_path": str(h5ad_path.resolve()),
                "n_spots": n_spots_raw,
                "n_genes": n_genes_raw,
                "gene_id_type": cfg.preprocess.gene_id_type,
                "spot_id_field": spot_id_field,
                "input_hash": hash_file(h5ad_path),
            },
            "params": asdict(cfg),
            "outputs": {
                "latent": "latent/latent.gat_ae.npy",
                "latent_meta": "latent/latent_meta.json",
                "neighbors_idx": "neighbors/neighbors.idx.npy",
                "neighbors_sim": "neighbors/neighbors.sim.npy",
                "neighbors_meta": "neighbors/neighbors_meta.json",
                "gss_sparse": "gss/gss_sparse.parquet",
                "gss_meta": "gss/gss_meta.json",
                "qc_report": "qc/qc_report.json",
                "qc_neighbor_per_spot": "qc/qc_tables/neighbor_per_spot.parquet",
                "qc_stability_per_spot": "qc/qc_tables/stability_per_spot.parquet",
            },
            "timestamps": {
                "finished_at": iso_now(),
            },
        }
        write_json(tmp_bundle / "manifest.json", manifest)

        promote_bundle(tmp_bundle, final_bundle)
        logger.info("GSS bundle generated at %s", final_bundle)
        return final_bundle
    except Exception:
        if tmp_bundle.exists():
            shutil.rmtree(tmp_bundle, ignore_errors=True)
        raise
