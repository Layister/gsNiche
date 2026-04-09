from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle_io import (
    ensure_bundle_dirs,
    get_code_version,
    hash_array,
    hash_file,
    iso_now,
    promote_bundle,
    set_random_seed,
    write_json,
    write_parquet,
)
from .data_prep import load_niche_inputs
from .niche_ops import (
    apply_basic_niche_filter,
    apply_random_baseline_filter,
    build_domain_adjacency_edges,
    build_niche_report,
    deduplicate_interaction_structures,
    discover_interaction_structures,
    finalize_interaction_structure_outputs,
)
from .schema import (
    BasicNicheFilterConfig,
    DomainEdgeConfig,
    DomainReliabilityConfig,
    InteractionDedupConfig,
    InteractionDiscoveryConfig,
    NicheInputConfig,
    NichePipelineConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NicheInputConfig",
    "DomainReliabilityConfig",
    "DomainEdgeConfig",
    "InteractionDiscoveryConfig",
    "BasicNicheFilterConfig",
    "InteractionDedupConfig",
    "NichePipelineConfig",
    "run_niche_pipeline",
]


def run_niche_pipeline(
    domain_bundle_path: str | Path,
    out_root: str | Path,
    sample_id: str,
    config: NichePipelineConfig | None = None,
) -> Path:
    cfg = config or NichePipelineConfig()
    set_random_seed(cfg.random_seed)

    domain_bundle_path = Path(domain_bundle_path)
    out_root = Path(out_root)

    sample_dir = out_root / sample_id
    final_bundle = sample_dir / "niche_bundle"
    tmp_bundle = sample_dir / f"niche_bundle.__tmp__{int(time.time())}_{os.getpid()}"

    if tmp_bundle.exists():
        shutil.rmtree(tmp_bundle)
    ensure_bundle_dirs(tmp_bundle)

    try:
        t0 = time.time()
        payload = load_niche_inputs(
            domain_bundle_path=domain_bundle_path,
            cfg=cfg,
            require_neighbors=False,
        )
        logger.info("Niche load_niche_inputs finished in %.2fs", time.time() - t0)
        domains_df = payload["domains_df"]
        membership_df_raw = payload["membership_df"]
        domain_graph_df = payload["domain_graph_df"]

        t0 = time.time()
        edges_df, edge_meta = build_domain_adjacency_edges(
            domains_df=domains_df,
            domain_graph_df=domain_graph_df,
            membership_df=membership_df_raw,
            spot_ids=payload["spot_ids"],
            spot_coords=payload["spot_coords"],
            neighbors_idx=payload["neighbors_idx"],
            cfg=cfg.edge,
            rel_cfg=cfg.domain_reliability,
        )
        logger.info("Niche build_domain_adjacency_edges finished in %.2fs", time.time() - t0)

        t0 = time.time()
        raw_structures_df, raw_membership_df, discovery_summary = discover_interaction_structures(
            edges_df=edges_df,
            domains_df=domains_df,
            discovery_cfg=cfg.discovery,
        )
        logger.info("Niche discover_interaction_structures finished in %.2fs", time.time() - t0)
        logger.info("Niche raw discovered structures: %d", int(raw_structures_df.shape[0]))

        t0 = time.time()
        filtered_structures_df = apply_basic_niche_filter(
            structures_df=raw_structures_df,
            filter_cfg=cfg.basic_filter,
        )
        logger.info("Niche apply_basic_niche_filter finished in %.2fs", time.time() - t0)

        passing_ids = set(
            filtered_structures_df.loc[
                filtered_structures_df["basic_qc_pass"].to_numpy(dtype=bool),
                "niche_id",
            ].astype(str).tolist()
        ) if not filtered_structures_df.empty else set()
        passing_structures_df = filtered_structures_df.loc[
            filtered_structures_df["niche_id"].astype(str).isin(passing_ids)
        ].reset_index(drop=True) if passing_ids else filtered_structures_df.iloc[0:0].copy()
        passing_membership_df = raw_membership_df.loc[
            raw_membership_df["niche_id"].astype(str).isin(passing_ids)
        ].reset_index(drop=True) if passing_ids else raw_membership_df.iloc[0:0].copy()
        discovery_summary = {
            **(discovery_summary or {}),
            "post_basic_filter_structure_count": int(passing_structures_df.shape[0]),
            "basic_filter_fail_reason_counts": (
                filtered_structures_df.loc[
                    ~filtered_structures_df["basic_qc_pass"].to_numpy(dtype=bool),
                    "basic_qc_fail_reason",
                ].value_counts().to_dict()
                if (not filtered_structures_df.empty and "basic_qc_fail_reason" in filtered_structures_df.columns)
                else {}
            ),
        }
        logger.info("Niche post-basic-filter structures: %d", int(passing_structures_df.shape[0]))

        if cfg.random_baseline.enabled:
            t0 = time.time()
            random_filtered_structures_df, random_filtered_membership_df, random_summary = apply_random_baseline_filter(
                passing_structures_df=passing_structures_df,
                passing_membership_df=passing_membership_df,
                edges_df=edges_df,
                domains_df=domains_df,
                discovery_cfg=cfg.discovery,
                basic_filter_cfg=cfg.basic_filter,
                random_cfg=cfg.random_baseline,
                random_seed=cfg.random_seed,
            )
            logger.info("Niche apply_random_baseline_filter finished in %.2fs", time.time() - t0)
        else:
            random_filtered_structures_df = passing_structures_df.copy()
            random_filtered_membership_df = passing_membership_df.copy()
            random_summary = {
                "enabled": False,
                "n_iter": int(cfg.random_baseline.n_iter),
                "input_structure_count": int(passing_structures_df.shape[0]),
                "passed_random_filter_count": int(passing_structures_df.shape[0]),
                "failed_random_filter_count": 0,
                "mean_non_random_score": float("nan"),
                "median_non_random_score": float("nan"),
            }
        logger.info("Niche post-random-filter structures: %d", int(random_filtered_structures_df.shape[0]))

        t0 = time.time()
        dedup_structures_df, dedup_membership_df, dedup_summary = deduplicate_interaction_structures(
            structures_df=random_filtered_structures_df,
            membership_df=random_filtered_membership_df,
            dedup_cfg=cfg.dedup,
        )
        logger.info("Niche deduplicate_interaction_structures finished in %.2fs", time.time() - t0)
        logger.info("Niche post-dedup structures: %d", int(dedup_structures_df.shape[0]))

        structures_out, membership_out = finalize_interaction_structure_outputs(
            structures_df=dedup_structures_df,
            membership_df=dedup_membership_df,
        )

        write_parquet(edges_df, tmp_bundle / "domain_adjacency_edges.parquet")
        write_parquet(structures_out, tmp_bundle / "niche_structures.parquet")
        write_parquet(membership_out, tmp_bundle / "niche_membership.parquet")

        edge_thresholds_df = pd.DataFrame(
            [
                {
                    "epsilon_distance": float(edge_meta.get("epsilon_distance", 0.0)),
                    "epsilon_source": str(edge_meta.get("epsilon_source", "")),
                    "spot_spacing_estimate": float(edge_meta.get("spot_spacing_estimate", float("nan"))),
                    "proximity_distance_source": str(edge_meta.get("proximity_distance_source", "")),
                    "boundary_distance_method": str(edge_meta.get("boundary_distance_method", "")),
                    "boundary_trim_fraction": float(edge_meta.get("boundary_trim_fraction", float("nan"))),
                    "boundary_trim_k_min": float(edge_meta.get("boundary_trim_k_min", float("nan"))),
                    "boundary_symmetry": str(edge_meta.get("boundary_symmetry", "")),
                    "boundary_spot_mode_used": str(edge_meta.get("boundary_spot_mode_used", "")),
                    "boundary_fallback_perimeter_scale": float(edge_meta.get("boundary_fallback_perimeter_scale", float("nan"))),
                    "boundary_fallback_used_count": int(edge_meta.get("boundary_fallback_used_count", 0)),
                    "contact_strength_mapping": str(edge_meta.get("contact_strength_mapping", "")),
                    "contact_strength_saturation_c": float(edge_meta.get("contact_strength_saturation_c", float("nan"))),
                    "domain_reliability_enabled": bool(edge_meta.get("domain_reliability_enabled", False)),
                    "domain_reliability_pair_mode": str(edge_meta.get("domain_reliability_pair_mode", "")),
                    "edge_reliability_mean": float(edge_meta.get("domain_reliability_mean", float("nan"))),
                    "edge_reliability_factor_mean": float(edge_meta.get("domain_reliability_factor_mean", float("nan"))),
                    "strong_contact_threshold": float(edge_meta.get("strong_contact_threshold", 0.0)),
                    "strong_overlap_threshold": float(edge_meta.get("strong_overlap_threshold", 0.0)),
                }
            ]
        )
        write_parquet(edge_thresholds_df, tmp_bundle / "qc_tables" / "edge_thresholds.parquet")

        niche_report = build_niche_report(
            sample_id=sample_id,
            edges_df=edges_df,
            structures_df=structures_out,
            edge_meta=edge_meta,
            cfg=cfg,
            discovery_summary=discovery_summary,
            random_summary=random_summary,
            dedup_summary=dedup_summary,
        )
        write_json(tmp_bundle / "niche_report.json", niche_report)

        niche_meta = {
            "sample_id": sample_id,
            "domain_count_qc_pass": int(np.count_nonzero(domains_df["qc_pass"].to_numpy(dtype=bool))),
            "adjacency_edge_count": int(edges_df.shape[0]),
            "strong_relation_edge_count": int(
                np.count_nonzero(
                    edges_df["is_strong_contact"].to_numpy(dtype=bool) | edges_df["is_strong_overlap"].to_numpy(dtype=bool)
                )
            ) if (not edges_df.empty and "is_strong_contact" in edges_df.columns and "is_strong_overlap" in edges_df.columns) else 0,
            "weak_relation_edge_count": int(np.count_nonzero(edges_df["proximity_contact"].to_numpy(dtype=bool)))
            if (not edges_df.empty and "proximity_contact" in edges_df.columns)
            else 0,
            "discovered_structure_count": int((discovery_summary or {}).get("raw_structure_count", 0)),
            "retained_structure_count": int(structures_out.shape[0]),
            "basic_filter_fail_reason_counts": (discovery_summary or {}).get("basic_filter_fail_reason_counts", {}),
            "random_baseline_summary": random_summary,
            "deduplication_summary": dedup_summary,
            "epsilon_distance": float(edge_meta.get("epsilon_distance", 0.0)),
            "epsilon_source": str(edge_meta.get("epsilon_source", "")),
            "spot_spacing_estimate": float(edge_meta.get("spot_spacing_estimate", float("nan"))),
            "proximity_distance_source": str(edge_meta.get("proximity_distance_source", "")),
            "domain_reliability_enabled": bool(edge_meta.get("domain_reliability_enabled", False)),
            "domain_reliability_pair_mode": str(edge_meta.get("domain_reliability_pair_mode", "")),
            "domain_reliability_summary": payload.get("domain_reliability_summary", {}),
            "program_qc_selection_summary": payload.get("program_qc_selection_summary", {}),
            "spot_order_source": payload["spot_order_source"],
            "spot_ids_hash": hash_array(payload["spot_ids"].astype(str)) if payload["spot_ids"] is not None else None,
        }
        write_json(tmp_bundle / "niche_meta.json", niche_meta)

        domain_manifest_path = domain_bundle_path / cfg.input.domain_manifest_relpath
        domains_path = domain_bundle_path / cfg.input.domains_relpath
        domain_membership_path = domain_bundle_path / cfg.input.domain_membership_relpath
        domain_graph_path = domain_bundle_path / cfg.input.domain_graph_relpath
        program_activation_path = payload["program_bundle_path"] / cfg.input.program_activation_relpath

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
                "domain_bundle_path": str(domain_bundle_path.resolve()),
                "program_bundle_path": str(payload["program_bundle_path"].resolve()),
                "gss_bundle_path": str(payload["gss_bundle_path"].resolve()),
                "domain_manifest_path": str(domain_manifest_path.resolve()),
                "domains_path": str(domains_path.resolve()),
                "domain_membership_path": str(domain_membership_path.resolve()),
                "domain_graph_path": str(domain_graph_path.resolve()),
                "program_activation_path": str(program_activation_path.resolve()),
                "neighbors_idx_path": str(payload["neighbors_idx_path"].resolve()) if payload["neighbors_idx_path"] is not None else None,
                "domain_manifest_hash": hash_file(domain_manifest_path),
                "domains_hash": hash_file(domains_path),
                "domain_membership_hash": hash_file(domain_membership_path),
                "domain_graph_hash": hash_file(domain_graph_path),
                "program_activation_hash": hash_file(program_activation_path) if program_activation_path.exists() else None,
                "neighbors_idx_hash": hash_file(payload["neighbors_idx_path"]) if payload["neighbors_idx_path"] is not None else None,
                "domain_count_qc_pass": int(np.count_nonzero(domains_df["qc_pass"].to_numpy(dtype=bool))),
                "spot_order_source": payload["spot_order_source"],
                "spot_order_hash": hash_array(payload["spot_ids"].astype(str)) if payload["spot_ids"] is not None else None,
                "program_qc_selection": payload.get("program_qc_selection_summary", {}),
            },
            "params": asdict(cfg),
            "outputs": {
                "domain_adjacency_edges": "domain_adjacency_edges.parquet",
                "niche_structures": "niche_structures.parquet",
                "niche_membership": "niche_membership.parquet",
                "niche_report": "niche_report.json",
                "niche_meta": "niche_meta.json",
                "qc_edge_thresholds": "qc_tables/edge_thresholds.parquet",
            },
            "timestamps": {"finished_at": iso_now()},
        }
        write_json(tmp_bundle / "manifest.json", manifest)

        promote_bundle(tmp_bundle, final_bundle)
        logger.info("Niche bundle generated at %s", final_bundle)
        return final_bundle
    except Exception:
        if tmp_bundle.exists():
            shutil.rmtree(tmp_bundle, ignore_errors=True)
        raise
