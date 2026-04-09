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
    hash_file,
    iso_now,
    promote_bundle,
    set_random_seed,
    write_json,
    write_parquet,
)
from .common import quantiles
from .data_prep import (
    apply_gene_filters,
    build_spot_gene_matrix,
    build_spot_neighbors_from_raw,
    infer_gene_activity,
    load_gss_inputs,
)
from .program_ops import (
    build_confounder_flags,
    compute_activation,
    compute_high_contribution_gene_stability,
    decide_observable_programs,
    materialize_program_table,
    materialize_rejected_candidate_audit_table,
    materialize_program_stage_diagnostics_table,
    refine_programs,
    resolve_program_redundancy,
    subset_high_contribution_gene_stability_summary,
)
from .schema import (
    ProgramActivationConfig,
    ProgramBootstrapConfig,
    ProgramPipelineConfig,
    ProgramPreprocessConfig,
    ProgramQCConfig,
    ProgramTemplateConfig,
)
from .template_ops import TemplateDiscoveryConfig, discover_template_programs

logger = logging.getLogger(__name__)

__all__ = [
    "ProgramPipelineConfig",
    "ProgramPreprocessConfig",
    "ProgramTemplateConfig",
    "TemplateDiscoveryConfig",
    "ProgramActivationConfig",
    "ProgramBootstrapConfig",
    "ProgramQCConfig",
    "run_program_pipeline",
]


def run_program_pipeline(
    gss_bundle_path: str | Path,
    out_root: str | Path,
    sample_id: str,
    config: ProgramPipelineConfig | None = None,
) -> Path:
    cfg = config or ProgramPipelineConfig()
    set_random_seed(cfg.random_seed)

    gss_bundle_path = Path(gss_bundle_path)
    out_root = Path(out_root)

    sample_dir = out_root / sample_id
    final_bundle = sample_dir / "program_bundle"
    tmp_bundle = sample_dir / f"program_bundle.__tmp__{int(time.time())}_{os.getpid()}"

    if tmp_bundle.exists():
        shutil.rmtree(tmp_bundle)
    ensure_bundle_dirs(tmp_bundle)

    try:
        gss_df, gss_meta, gss_manifest = load_gss_inputs(gss_bundle_path, cfg)
        spot_by_gene, spot_ids, gene_ids = build_spot_gene_matrix(gss_df)
        spot_neighbors, spot_neighbor_meta = build_spot_neighbors_from_raw(
            spot_ids=spot_ids,
            gss_manifest=gss_manifest,
            input_cfg=cfg.input,
        )

        activity_payload = infer_gene_activity(spot_by_gene, gene_ids=gene_ids, cfg=cfg.preprocess)
        gss_filtered_matrix, filtered_gene_ids, filter_payload = apply_gene_filters(
            spot_by_gene,
            gene_ids=gene_ids,
            activity_payload=activity_payload,
            cfg=cfg.preprocess,
        )

        active_mask_binary = filter_payload["active_mask_binary"]
        active_strength = filter_payload["active_strength"]
        support_spots = filter_payload["support_spots"]
        support_frac = filter_payload["support_frac"]
        idf = filter_payload["idf"]
        blacklist_mask = filter_payload["blacklist_mask"]
        blacklist_factor = filter_payload["blacklist_factor"]
        template_cfg = TemplateDiscoveryConfig(**asdict(cfg.template))
        template_discovery = discover_template_programs(
            active_strength=active_strength,
            gene_ids=filtered_gene_ids,
            support_spots=support_spots,
            idf=idf,
            blacklist_factor=blacklist_factor,
            cfg=template_cfg,
            bootstrap_cfg=cfg.bootstrap,
            random_seed=cfg.random_seed,
        )
        candidate_program_payload = list(template_discovery.get("candidate_program_payload", []))
        bootstrap_payload = {
            "enabled": bool(len(template_discovery.get("repeat_program_summaries", [])) >= 2),
            "assignments": [],
            "program_summaries": template_discovery.get("repeat_program_summaries", []),
            "consensus_program_sets": None,
            "assignment_label_stability": template_discovery.get(
                "assignment_label_stability",
                {"metric": "ARI", "quantiles": quantiles(np.asarray([], dtype=np.float32))},
            ),
            "rerun_consensus_mode": "template_rerun_consensus",
            "rerun_build_stats": {
                "template_runs": int(template_discovery.get("template_summary", {}).get("runs_done", 0)),
                "reruns_run": int(template_discovery.get("template_summary", {}).get("runs_done", 0)),
            },
            "template_methods_used": ["sparse_nmf_template_learning"],
            "template_method_used": "sparse_nmf_template_learning",
            "rerun_program_set_summary": {},
        }
        template_summary = dict(template_discovery.get("template_summary", {}))

        _, candidate_dense_activation, candidate_activation_summary = compute_activation(
            active_strength=active_strength,
            active_mask_binary=active_mask_binary,
            spot_ids=spot_ids,
            program_payload=candidate_program_payload,
            spot_neighbors=spot_neighbors,
            cfg=cfg.activation,
        )
        refined_candidate_payload, refinement_table = refine_programs(
            candidate_program_payload,
            total_gene_count=int(filtered_gene_ids.shape[0]),
            min_program_size_genes=int(cfg.template.min_program_size_genes),
            cfg=cfg.activation,
        )

        _, refined_candidate_dense_activation, refined_candidate_activation_summary = compute_activation(
            active_strength=active_strength,
            active_mask_binary=active_mask_binary,
            spot_ids=spot_ids,
            program_payload=refined_candidate_payload,
            spot_neighbors=spot_neighbors,
            cfg=cfg.activation,
        )

        candidate_high_contribution_gene_stability_summary = {"topN_metrics": {}, "records": []}
        candidate_low_stability_programs: list[dict] = []
        if bootstrap_payload["enabled"]:
            candidate_high_contribution_gene_stability_summary, candidate_low_stability_programs = (
                compute_high_contribution_gene_stability(
                    final_program_payload=refined_candidate_payload,
                    repeat_program_summaries=bootstrap_payload["program_summaries"],
                    top_ns=cfg.bootstrap.top_ns,
                    gene_ids=filtered_gene_ids,
                    stable_high_contribution_gene_min_frequency=cfg.qc.stable_high_contribution_gene_min_frequency,
                )
            )

        candidate_confounder_flags, candidate_confounder_table = build_confounder_flags(
            program_payload=refined_candidate_payload,
            dense_activation=refined_candidate_dense_activation,
            support_frac=support_frac,
            blacklist_mask=blacklist_mask,
            qc_cfg=cfg.qc,
            activation_thresholds_by_program=refined_candidate_activation_summary.get(
                "effective_activation_threshold_by_program",
                {},
            ),
        )

        existence_decision = decide_observable_programs(
            program_payload=refined_candidate_payload,
            dense_activation=refined_candidate_dense_activation,
            gene_ids=filtered_gene_ids,
            confounder_flags=candidate_confounder_flags,
            high_contribution_gene_stability_summary=candidate_high_contribution_gene_stability_summary,
            bootstrap_enabled=bool(bootstrap_payload["enabled"]),
            spot_neighbors=spot_neighbors,
            activation_thresholds_by_program=refined_candidate_activation_summary.get(
                "effective_activation_threshold_by_program",
                {},
            ),
            qc_cfg=cfg.qc,
        )
        score_rows = list(existence_decision.get("program_scores", []))
        valid_program_ids = {
            str(r.get("program_id"))
            for r in score_rows
            if str(r.get("validity_status", "invalid")) == "valid"
        }
        score_by_pid = {str(r.get("program_id")): dict(r) for r in score_rows}
        rejected_candidate_audit_table = materialize_rejected_candidate_audit_table(
            program_payload=refined_candidate_payload,
            gene_ids=filtered_gene_ids,
            program_scores=score_rows,
            rejected_programs=list(existence_decision.get("rejected_programs", [])),
        )
        program_stage_diagnostics_table = materialize_program_stage_diagnostics_table(
            candidate_program_payload=candidate_program_payload,
            refined_program_payload=refined_candidate_payload,
            refinement_table=refinement_table,
            gene_ids=filtered_gene_ids,
            spot_neighbors=spot_neighbors,
            score_by_pid=score_by_pid,
        )
        valid_candidate_payload = [p for p in refined_candidate_payload if str(p["program_id"]) in valid_program_ids]
        for item in valid_candidate_payload:
            pid = str(item["program_id"])
            item["program_confidence"] = float(score_by_pid.get(pid, {}).get("program_confidence", 0.0))
            item["validity_status"] = str(score_by_pid.get(pid, {}).get("validity_status", "invalid"))
            item["routing_status"] = str(score_by_pid.get(pid, {}).get("routing_status", "rejected"))
            item["default_use_support_score"] = float(score_by_pid.get(pid, {}).get("default_use_support_score", 0.0))
            item["default_use_reason_count"] = int(score_by_pid.get(pid, {}).get("default_use_reason_count", 0))
            item["default_use_reasons"] = list(score_by_pid.get(pid, {}).get("default_use_reasons", []))
        program_payload, redundancy_table = resolve_program_redundancy(
            valid_candidate_payload,
            score_by_pid=score_by_pid,
            qc_cfg=cfg.qc,
        )

        activation_df, dense_activation, activation_summary = compute_activation(
            active_strength=active_strength,
            active_mask_binary=active_mask_binary,
            spot_ids=spot_ids,
            program_payload=program_payload,
            spot_neighbors=spot_neighbors,
            cfg=cfg.activation,
        )
        conf_map = {
            str(item["program_id"]): float(score_by_pid.get(str(item["program_id"]), {}).get("program_confidence", 0.0))
            for item in program_payload
        }
        activation_df["program_confidence"] = activation_df["program_id"].astype(str).map(conf_map).fillna(0.0).to_numpy(
            dtype=np.float32
        )
        activation_df["activation_identity_view_weighted"] = (
            activation_df["activation_identity_view"].to_numpy(dtype=np.float32)
            * activation_df["program_confidence"].to_numpy(dtype=np.float32)
        )
        activation_df["activation_weighted"] = activation_df["activation_identity_view_weighted"].to_numpy(dtype=np.float32)
        activation_df["recommended_domain_input"] = "activation_identity_view_weighted"
        programs_df = materialize_program_table(program_payload, filtered_gene_ids)
        if not programs_df.empty:
            programs_df["program_confidence"] = programs_df["program_id"].astype(str).map(
                lambda pid: float(score_by_pid.get(str(pid), {}).get("program_confidence", 0.0))
            )
            programs_df["validity_status"] = programs_df["program_id"].astype(str).map(
                lambda pid: str(score_by_pid.get(str(pid), {}).get("validity_status", "invalid"))
            )
            programs_df["routing_status"] = programs_df["program_id"].astype(str).map(
                lambda pid: str(score_by_pid.get(str(pid), {}).get("routing_status", "rejected"))
            )
            programs_df["default_use_support_score"] = programs_df["program_id"].astype(str).map(
                lambda pid: float(score_by_pid.get(str(pid), {}).get("default_use_support_score", 0.0))
            )
            programs_df["default_use_reason_count"] = programs_df["program_id"].astype(str).map(
                lambda pid: int(score_by_pid.get(str(pid), {}).get("default_use_reason_count", 0))
            )
            programs_df["default_use_reasons"] = programs_df["program_id"].astype(str).map(
                lambda pid: "|".join([str(x) for x in score_by_pid.get(str(pid), {}).get("default_use_reasons", [])])
            )

        confounder_flags, confounder_table = build_confounder_flags(
            program_payload=program_payload,
            dense_activation=dense_activation,
            support_frac=support_frac,
            blacklist_mask=blacklist_mask,
            qc_cfg=cfg.qc,
            activation_thresholds_by_program=activation_summary.get(
                "effective_activation_threshold_by_program",
                {},
            ),
        )

        high_contribution_gene_stability_summary = {"topN_metrics": {}, "records": []}
        low_stability_programs: list[dict] = []
        final_program_ids = {str(item["program_id"]) for item in program_payload}
        if bootstrap_payload["enabled"]:
            high_contribution_gene_stability_summary = subset_high_contribution_gene_stability_summary(
                candidate_high_contribution_gene_stability_summary,
                keep_program_ids=final_program_ids,
            )
            low_stability_programs = [
                item for item in candidate_low_stability_programs if str(item.get("program_id")) in final_program_ids
            ]

        program_sizes = np.array([len(item["gene_indices"]) for item in program_payload], dtype=np.float32)
        max_program_gene_frac = (
            float(program_sizes.max() / max(1, filtered_gene_ids.shape[0])) if program_sizes.size > 0 else 0.0
        )
        kept_conf = np.asarray(
            [float(score_by_pid.get(str(item["program_id"]), {}).get("program_confidence", 0.0)) for item in program_payload],
            dtype=np.float32,
        )
        program_summary = {
            "candidate_program_count": int(len(candidate_program_payload)),
            "refined_candidate_program_count": int(len(refined_candidate_payload)),
            "valid_after_existence_count": int(len(valid_candidate_payload)),
            "kept_after_existence_count": int(len(valid_candidate_payload)),
            "default_use_after_validity_count": int(existence_decision.get("default_use_program_count", 0)),
            "review_after_validity_count": int(existence_decision.get("review_program_count", 0)),
            "program_count": int(len(program_payload)),
            "size_distribution": quantiles(program_sizes),
            "max_program_gene_frac": max_program_gene_frac,
            "program_confidence_quantiles": quantiles(kept_conf),
            "candidate_program_confidence_quantiles": existence_decision.get("program_confidence_quantiles", {}),
            "validity_counts": existence_decision.get("validity_counts", {}),
            "routing_counts": existence_decision.get("routing_counts", {}),
            "default_use_reason_counts": existence_decision.get("default_use_reason_counts", {}),
            "redundancy_status_counts": (
                redundancy_table["redundancy_status"].value_counts().to_dict() if not redundancy_table.empty else {}
            ),
            "discovery_mode": "gss_sparse_template_learning",
            "template_discovery": template_summary,
            "no_program_discovered": bool(len(program_payload) == 0),
            "identity_mode": "gss_template_learning_plus_activation_field",
            "existence_filter": existence_decision,
        }

        score_table = pd.DataFrame(score_rows)
        if score_table.empty:
            score_table = pd.DataFrame(
                columns=[
                    "program_id",
                    "program_confidence",
                    "validity_status",
                    "routing_status",
                    "default_use_support_score",
                    "default_use_reason_count",
                    "template_evidence_score",
                    "stability_score",
                    "activation_score",
                    "activation_morphology_score",
                    "existence_score",
                ]
            )
        program_struct_rows = [
            {
                "program_id": str(item["program_id"]),
                "program_size_genes": int(item.get("program_size_genes", 0)),
                "program_gene_frac": float(item.get("program_gene_frac", 0.0)),
                "template_run_support_frac": float(item.get("template_run_support_frac", 0.0)),
                "template_spot_support_frac": float(item.get("template_spot_support_frac", 0.0)),
                "template_focus_score": float(item.get("template_focus_score", 0.0)),
                "template_evidence_score": float(
                    item.get("template_evidence_score", item.get("structure_confidence", 0.0))
                ),
                "core_full_consistency": float(item.get("core_full_consistency", 0.0)),
                "activation_peakiness": float(item.get("activation_peakiness", 0.0)),
                "activation_entropy": float(item.get("activation_entropy", 0.0)),
                "activation_sparsity": float(item.get("activation_sparsity", 0.0)),
                "main_component_frac": float(item.get("main_component_frac", 0.0)),
                "high_activation_spot_count": int(item.get("high_activation_spot_count", 0)),
                "recommended_domain_input": str(
                    item.get("recommended_domain_input", "activation_identity_view_weighted")
                ),
            }
            for item in refined_candidate_payload
        ]
        program_struct_table = pd.DataFrame(program_struct_rows)
        if program_struct_table.empty:
            program_struct_table = pd.DataFrame(
                columns=[
                    "program_id",
                    "program_size_genes",
                    "program_gene_frac",
                    "template_run_support_frac",
                    "template_spot_support_frac",
                    "template_focus_score",
                    "template_evidence_score",
                    "core_full_consistency",
                    "activation_peakiness",
                    "activation_entropy",
                    "activation_sparsity",
                    "main_component_frac",
                    "high_activation_spot_count",
                    "recommended_domain_input",
                ]
            )
        program_qc_table = candidate_confounder_table.merge(program_struct_table, on="program_id", how="outer")
        overlap_cols = sorted((set(program_qc_table.columns) & set(score_table.columns)) - {"program_id"})
        if overlap_cols:
            program_qc_table = program_qc_table.drop(columns=overlap_cols, errors="ignore")
        program_qc_table = program_qc_table.merge(score_table, on="program_id", how="outer")
        if not refinement_table.empty:
            overlap_cols = sorted((set(program_qc_table.columns) & set(refinement_table.columns)) - {"program_id"})
            if overlap_cols:
                program_qc_table = program_qc_table.drop(columns=overlap_cols, errors="ignore")
            program_qc_table = program_qc_table.merge(refinement_table, on="program_id", how="outer")
        if not redundancy_table.empty:
            overlap_cols = sorted((set(program_qc_table.columns) & set(redundancy_table.columns)) - {"program_id"})
            if overlap_cols:
                program_qc_table = program_qc_table.drop(columns=overlap_cols, errors="ignore")
            program_qc_table = program_qc_table.merge(redundancy_table, on="program_id", how="outer")
        write_parquet(program_qc_table, tmp_bundle / "qc_tables" / "program_qc.parquet")
        write_parquet(redundancy_table, tmp_bundle / "qc_tables" / "program_redundancy.parquet")
        write_parquet(rejected_candidate_audit_table, tmp_bundle / "qc_tables" / "rejected_candidate_audit.parquet")
        write_parquet(program_stage_diagnostics_table, tmp_bundle / "qc_tables" / "program_stage_diagnostics.parquet")

        write_parquet(programs_df, tmp_bundle / "programs.parquet")
        write_parquet(activation_df, tmp_bundle / "program_activation.parquet")

        program_meta = {
            "sample_id": sample_id,
            "n_spots": int(gss_filtered_matrix.shape[0]),
            "n_genes_before_filter": int(gene_ids.shape[0]),
            "n_genes_after_filter": int(filtered_gene_ids.shape[0]),
            "candidate_program_count": int(len(candidate_program_payload)),
            "program_count": int(len(program_payload)),
            "discovery_backend": "gss_sparse_template_learning",
            "program_ids": [item["program_id"] for item in program_payload],
            "recommended_domain_input": "activation_identity_view_weighted",
            "full_activation_for_reporting": True,
            "template_discovery": template_summary,
            "program_existence_filter": {
                "kept_program_count": int(existence_decision.get("kept_program_count", 0)),
                "default_use_program_count": int(existence_decision.get("default_use_program_count", 0)),
                "review_program_count": int(existence_decision.get("review_program_count", 0)),
                "rejected_program_count": int(existence_decision.get("rejected_program_count", 0)),
                "rejection_reason_counts": existence_decision.get("rejection_reason_counts", {}),
                "program_confidence_quantiles": existence_decision.get("program_confidence_quantiles", {}),
                "validity_counts": existence_decision.get("validity_counts", {}),
                "routing_counts": existence_decision.get("routing_counts", {}),
                "default_use_reason_counts": existence_decision.get("default_use_reason_counts", {}),
            },
            "activation_storage": {
                "is_sparse": True,
                "format": "long",
                "default_view": "identity_view",
                "has_activation_identity_view": True,
                "has_activation_full": True,
                "has_activation_identity_view_weighted": True,
                "min_activation": float(cfg.activation.min_activation),
                "adaptive_min_activation_quantile": float(cfg.activation.adaptive_min_activation_quantile),
                "has_rank_in_spot": True,
                "retained_rows": int(activation_df.shape[0]),
            },
        }
        write_json(tmp_bundle / "program_meta.json", program_meta)

        qc_report = {
            "sample_id": sample_id,
            "inputs_summary": {
                "n_spots": int(gss_filtered_matrix.shape[0]),
                "n_genes": int(gene_ids.shape[0]),
                "topM": gss_meta.get("top_m", None),
                "gss_sparse_rows": int(gss_df.shape[0]),
            },
            "gene_support_summary": {
                **filter_payload["summary"],
                "filtered_ratio": float(
                    1.0 - (filter_payload["summary"]["n_genes_after"] / max(1, filter_payload["summary"]["n_genes_before"]))
                ),
            },
            "gene_activity_summary": activity_payload["summary"],
            "template_discovery": template_summary,
            "program_summary": program_summary,
            "activation_summary": activation_summary,
            "candidate_activation_summary": candidate_activation_summary,
            "refined_candidate_activation_summary": refined_candidate_activation_summary,
            "stability_summary": {
                "rerun_target": int(cfg.bootstrap.bootstrap_B if cfg.bootstrap.enabled else 0),
                "rerun_enabled": bool(bootstrap_payload["enabled"]),
                "reruns_run": int(
                    bootstrap_payload.get("rerun_build_stats", {}).get("reruns_run", 0)
                ),
                "stability_mode": bootstrap_payload.get("rerun_consensus_mode", None),
                "rerun_build_stats": bootstrap_payload.get("rerun_build_stats", {}),
                "assignment_label_stability": bootstrap_payload.get("assignment_label_stability", {}),
                "high_contribution_genes_stability": high_contribution_gene_stability_summary,
                "low_stability_programs": low_stability_programs,
                "candidate_high_contribution_genes_stability": candidate_high_contribution_gene_stability_summary,
                "candidate_low_stability_programs": candidate_low_stability_programs,
            },
            "program_refinement": {
                "refinement_status_counts": (
                    refinement_table["refinement_status"].value_counts().to_dict() if not refinement_table.empty else {}
                ),
            },
            "program_stage_diagnostics": {
                "record_count": int(program_stage_diagnostics_table.shape[0]),
                "refined_exists_count": int(program_stage_diagnostics_table.get("refined_exists", pd.Series(dtype=bool)).sum())
                if not program_stage_diagnostics_table.empty
                else 0,
                "dropped_after_refinement_count": int(
                    (~program_stage_diagnostics_table.get("refined_exists", pd.Series(dtype=bool))).sum()
                )
                if not program_stage_diagnostics_table.empty
                else 0,
            },
            "program_redundancy": (
                redundancy_table.to_dict(orient="records") if not redundancy_table.empty else []
            ),
            "confounder_flags": confounder_flags,
            "candidate_confounder_flags": candidate_confounder_flags,
            "acceptance": {
                "observability_mode": "validity_default_use_filter",
                "default_use_min_program_confidence": float(cfg.qc.default_use_min_program_confidence),
                "default_use_min_support_score": float(cfg.qc.default_use_min_support_score),
                "default_use_min_validity_score": float(cfg.qc.default_use_min_validity_score),
                "default_use_min_activation_presence_score": float(cfg.qc.default_use_min_activation_presence_score),
                "default_use_min_structure_score": float(cfg.qc.default_use_min_structure_score),
                "default_use_min_scaffold_content_quality": float(cfg.qc.default_use_min_scaffold_content_quality),
                "high_program_confidence_threshold": float(cfg.qc.high_program_confidence_threshold),
                "drop_housekeeping_or_blacklist": bool(cfg.qc.drop_housekeeping_or_blacklist),
                "spot_neighbor_graph": spot_neighbor_meta,
                "max_program_gene_frac_warn_threshold": cfg.template.max_program_gene_frac_warn,
                "max_program_gene_frac_warn": bool(
                    max_program_gene_frac > float(cfg.template.max_program_gene_frac_warn)
                ),
                "dominance_warn_p50_threshold": cfg.qc.dominance_warn_p50,
                "dominance_warn_p50": bool(
                    activation_summary["top_program_dominance"]["p50"] > float(cfg.qc.dominance_warn_p50)
                ),
            },
        }
        write_json(tmp_bundle / "qc_report.json", qc_report)

        gss_sparse_input = gss_bundle_path / cfg.input.gss_sparse_relpath
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
                "gss_bundle_path": str(gss_bundle_path.resolve()),
                "gss_sparse_path": str(gss_sparse_input.resolve()),
                "gss_sparse_hash": hash_file(gss_sparse_input),
                "n_spots": int(spot_by_gene.shape[0]),
                "n_genes": int(spot_by_gene.shape[1]),
                "topM": gss_meta.get("top_m", None),
                "gss_schema_version": gss_manifest.get("schema_version", None),
                "program_input_source": "gss_template_evidence_matrix",
            },
            "params": asdict(cfg),
            "outputs": {
                "programs": "programs.parquet",
                "program_activation": "program_activation.parquet",
                "program_meta": "program_meta.json",
                "qc_report": "qc_report.json",
                "qc_program_table": "qc_tables/program_qc.parquet",
                "qc_program_redundancy": "qc_tables/program_redundancy.parquet",
                "qc_rejected_candidate_audit": "qc_tables/rejected_candidate_audit.parquet",
            },
            "output_semantics": {
                "program_activation": {
                    "is_sparse": True,
                    "format": "long",
                    "default_view": "identity_view",
                    "activation_column_for_domain": "activation_identity_view_weighted",
                    "has_activation_identity_view": True,
                    "has_activation_full": True,
                    "has_activation_identity_view_weighted": True,
                    "min_activation": float(cfg.activation.min_activation),
                    "adaptive_min_activation_quantile": float(cfg.activation.adaptive_min_activation_quantile),
                    "has_rank_in_spot": True,
                    "recommended_domain_input": "activation_identity_view_weighted",
                    "full_activation_for_reporting": True,
                },
            },
            "resolved_params": {
                "template": {
                    "template_discovery": template_summary,
                }
            },
            "timestamps": {"finished_at": iso_now()},
        }
        write_json(tmp_bundle / "manifest.json", manifest)

        promote_bundle(tmp_bundle, final_bundle)
        logger.info("Program bundle written: %s", final_bundle)
        return final_bundle
    except Exception:
        shutil.rmtree(tmp_bundle, ignore_errors=True)
        raise
