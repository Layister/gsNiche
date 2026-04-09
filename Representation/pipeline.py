from __future__ import annotations

import shutil
import time
from pathlib import Path

from .config import resolve_axis_catalog
from .loaders import build_eligibility_table, load_representation_inputs
from .loaders import load_domain_inputs, load_niche_inputs
from .schema import RepresentationPipelineConfig
from .scoring.component_axes import score_component_axes
from .scoring.evidence_extractors import extract_program_evidence
from .scoring.role_axes import score_role_axes
from .summarize import (
    build_axis_definition_payload,
    build_domain_burden_table,
    build_domain_profile_table,
    build_domain_summary_markdown,
    build_domain_summary_payload,
    build_future_layer_meta,
    build_niche_burden_table,
    build_niche_profile_table,
    build_niche_summary_markdown,
    build_niche_summary_payload,
    build_program_profile_table,
    build_program_summary_markdown,
    build_qc_report,
    build_sample_burden_table,
    build_sample_summary_payload,
    build_sample_summary_markdown,
    build_sample_summary_view,
)
from .writers import build_manifest, ensure_bundle_dirs, promote_bundle, write_json, write_parquet, write_text


def run_representation_pipeline(
    program_bundle_path: str | Path,
    out_root: str | Path,
    sample_id: str | None = None,
    cancer_type: str | None = None,
    config: RepresentationPipelineConfig | None = None,
) -> Path:
    cfg = config or RepresentationPipelineConfig()
    program_bundle_path = Path(program_bundle_path)
    out_root = Path(out_root)
    resolved_sample_id = str(sample_id).strip() if sample_id else program_bundle_path.parent.name

    sample_dir = out_root / resolved_sample_id
    final_bundle = sample_dir / "representation_bundle"
    tmp_bundle = sample_dir / f"representation_bundle.__tmp__{int(time.time())}"
    if tmp_bundle.exists():
        shutil.rmtree(tmp_bundle)
    ensure_bundle_dirs(tmp_bundle)

    try:
        bundle = load_representation_inputs(
            program_bundle_path=program_bundle_path,
            cancer_type=cancer_type,
            cfg=cfg,
        )
        eligibility_df = build_eligibility_table(bundle, cfg)
        axis_catalog = resolve_axis_catalog(bundle.cancer_type, cfg)
        component_axes = axis_catalog["component_axes"]
        role_axes = axis_catalog["role_axes"]
        evidences = extract_program_evidence(bundle=bundle, eligibility_df=eligibility_df, cfg=cfg)

        component_score_map: dict[str, dict[str, float]] = {}
        component_detail_map: dict[str, dict[str, dict]] = {}
        role_score_map: dict[str, dict[str, float]] = {}
        role_detail_map: dict[str, dict[str, dict]] = {}
        for evidence in evidences:
            component_scores, component_details = score_component_axes(evidence, component_axes, cfg)
            role_scores, role_details = score_role_axes(evidence, role_axes, cfg)
            component_score_map[evidence.program_id] = component_scores
            component_detail_map[evidence.program_id] = component_details
            role_score_map[evidence.program_id] = role_scores
            role_detail_map[evidence.program_id] = role_details

        axis_definition = build_axis_definition_payload(
            cancer_type=bundle.cancer_type,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        program_profile_df = build_program_profile_table(
            evidences=evidences,
            component_axes=component_axes,
            role_axes=role_axes,
            component_score_map=component_score_map,
            component_detail_map=component_detail_map,
            role_score_map=role_score_map,
            role_detail_map=role_detail_map,
            cfg=cfg,
        )
        sample_burden_df = build_sample_burden_table(
            bundle=bundle,
            program_profile_df=program_profile_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        sample_summary = build_sample_summary_payload(
            bundle=bundle,
            program_profile_df=program_profile_df,
            sample_burden_df=sample_burden_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        domains_df, domain_program_map_df, domain_graph_df, domain_status = load_domain_inputs(program_bundle_path=program_bundle_path, cfg=cfg)
        domain_profile_df = build_domain_profile_table(
            bundle=bundle,
            domains_df=domains_df,
            domain_program_map_df=domain_program_map_df,
            domain_graph_df=domain_graph_df,
            program_profile_df=program_profile_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        domain_burden_df = build_domain_burden_table(
            bundle=bundle,
            domain_profile_df=domain_profile_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        domain_summary = build_domain_summary_payload(
            bundle=bundle,
            domain_profile_df=domain_profile_df,
            domain_burden_df=domain_burden_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        domain_summary_md = build_domain_summary_markdown(domain_summary)
        niche_structures_df, niche_membership_df, niche_edges_df, niche_status = load_niche_inputs(program_bundle_path=program_bundle_path, cfg=cfg)
        niche_profile_df = build_niche_profile_table(
            bundle=bundle,
            niche_structures_df=niche_structures_df,
            niche_membership_df=niche_membership_df,
            niche_edges_df=niche_edges_df,
            domain_profile_df=domain_profile_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        niche_burden_df = build_niche_burden_table(
            bundle=bundle,
            niche_profile_df=niche_profile_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        niche_summary = build_niche_summary_payload(
            bundle=bundle,
            niche_profile_df=niche_profile_df,
            niche_burden_df=niche_burden_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        niche_summary_md = build_niche_summary_markdown(niche_summary)
        program_summary_md = build_program_summary_markdown(sample_summary)
        sample_summary_view = build_sample_summary_view(
            sample_summary,
            domain_summary=domain_summary if domain_status.get("available") else None,
            niche_summary=niche_summary if niche_status.get("available") else None,
        )
        sample_summary_md = build_sample_summary_markdown(sample_summary_view)
        qc_report = build_qc_report(
            bundle=bundle,
            eligibility_df=eligibility_df,
            program_profile_df=program_profile_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )

        axis_definition_path = tmp_bundle / "axis_definition.json"
        program_profile_path = tmp_bundle / "program" / "macro_profile.parquet"
        sample_burden_path = tmp_bundle / "program" / "sample_burden.parquet"
        program_summary_json_path = tmp_bundle / "program" / "macro_summary.json"
        program_summary_md_path = tmp_bundle / "program" / "macro_summary.md"
        domain_profile_path = tmp_bundle / "domain" / "macro_profile.parquet"
        domain_burden_path = tmp_bundle / "domain" / "sample_burden.parquet"
        domain_summary_json_path = tmp_bundle / "domain" / "macro_summary.json"
        domain_summary_md_path = tmp_bundle / "domain" / "macro_summary.md"
        niche_profile_path = tmp_bundle / "niche" / "macro_profile.parquet"
        niche_burden_path = tmp_bundle / "niche" / "sample_burden.parquet"
        niche_summary_json_path = tmp_bundle / "niche" / "macro_summary.json"
        niche_summary_md_path = tmp_bundle / "niche" / "macro_summary.md"
        sample_summary_json_path = tmp_bundle / "sample" / "macro_summary.json"
        sample_summary_md_path = tmp_bundle / "sample" / "macro_summary.md"
        qc_report_path = tmp_bundle / "qc_report.json"

        write_json(axis_definition_path, axis_definition)
        write_parquet(program_profile_path, program_profile_df)
        write_parquet(sample_burden_path, sample_burden_df)
        write_json(program_summary_json_path, sample_summary)
        write_text(program_summary_md_path, program_summary_md)
        write_parquet(domain_profile_path, domain_profile_df)
        write_parquet(domain_burden_path, domain_burden_df)
        write_json(domain_summary_json_path, domain_summary)
        write_text(domain_summary_md_path, domain_summary_md)
        write_parquet(niche_profile_path, niche_profile_df)
        write_parquet(niche_burden_path, niche_burden_df)
        write_json(niche_summary_json_path, niche_summary)
        write_text(niche_summary_md_path, niche_summary_md)
        write_json(sample_summary_json_path, sample_summary_view)
        write_text(sample_summary_md_path, sample_summary_md)
        write_json(qc_report_path, qc_report)

        manifest = build_manifest(
            bundle=bundle,
            cfg=cfg,
            axis_definition_path=axis_definition_path,
            program_profile_path=program_profile_path,
            sample_burden_path=sample_burden_path,
            program_summary_json_path=program_summary_json_path,
            program_summary_md_path=program_summary_md_path,
            domain_profile_path=domain_profile_path,
            domain_burden_path=domain_burden_path,
            domain_summary_json_path=domain_summary_json_path,
            domain_summary_md_path=domain_summary_md_path,
            niche_profile_path=niche_profile_path,
            niche_burden_path=niche_burden_path,
            niche_summary_json_path=niche_summary_json_path,
            niche_summary_md_path=niche_summary_md_path,
            sample_summary_json_path=sample_summary_json_path,
            sample_summary_md_path=sample_summary_md_path,
            qc_report_path=qc_report_path,
        )
        write_json(tmp_bundle / "manifest.json", manifest)
        promote_bundle(tmp_bundle, final_bundle)
        return final_bundle
    except Exception:
        shutil.rmtree(tmp_bundle, ignore_errors=True)
        raise
