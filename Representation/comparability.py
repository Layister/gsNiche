from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .config import resolve_axis_catalog
from .loaders import (
    load_cross_sample_domain_profiles,
    load_cross_sample_niche_profiles,
    load_cross_sample_program_profiles,
)
from .schema import RepresentationPipelineConfig
from .summarize import (
    build_domain_cross_sample_comparability_table,
    build_domain_cross_sample_summary_markdown,
    build_domain_cross_sample_summary_payload,
    build_niche_cross_sample_comparability_table,
    build_niche_cross_sample_summary_markdown,
    build_niche_cross_sample_summary_payload,
    build_program_cross_sample_comparability_table,
    build_program_cross_sample_summary_markdown,
    build_program_cross_sample_summary_payload,
    build_sample_summary_markdown,
    build_sample_summary_view,
)
from .writers import enrich_manifest_with_cross_sample_outputs, write_json, write_parquet, write_text


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_representation_bundles(
    out_root: str | Path,
    cfg: RepresentationPipelineConfig,
    sample_ids: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    root = Path(out_root)
    if not root.exists():
        return []
    sample_filter = {str(x) for x in sample_ids} if sample_ids else None
    bundle_dirs: list[Path] = []
    for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if sample_filter and sample_dir.name not in sample_filter:
            continue
        bundle_dir = sample_dir / cfg.input.representation_bundle_dirname
        if (bundle_dir / "program" / "macro_profile.parquet").exists():
            bundle_dirs.append(bundle_dir)
    return bundle_dirs


def _bundle_context(bundle_dir: Path, override_cancer_type: str | None = None) -> SimpleNamespace:
    manifest = _read_json(bundle_dir / "manifest.json")
    sample_id = str(manifest.get("sample_id") or bundle_dir.parent.name)
    cancer_type = str(override_cancer_type or manifest.get("cancer_type") or "")
    return SimpleNamespace(sample_id=sample_id, cancer_type=cancer_type)


def run_representation_comparability(
    out_root: str | Path,
    sample_ids: list[str] | tuple[str, ...] | None = None,
    cancer_type: str | None = None,
    config: RepresentationPipelineConfig | None = None,
) -> list[Path]:
    cfg = config or RepresentationPipelineConfig()
    bundle_dirs = _iter_representation_bundles(out_root, cfg, sample_ids=sample_ids)
    written_bundles: list[Path] = []

    for bundle_dir in bundle_dirs:
        context = _bundle_context(bundle_dir, override_cancer_type=cancer_type)
        axis_catalog = resolve_axis_catalog(context.cancer_type, cfg)
        component_axes = axis_catalog["component_axes"]
        role_axes = axis_catalog["role_axes"]

        program_profile_df = pd.read_parquet(bundle_dir / "program" / "macro_profile.parquet")
        program_summary = _read_json(bundle_dir / "program" / "macro_summary.json")
        domain_profile_df = pd.read_parquet(bundle_dir / "domain" / "macro_profile.parquet")
        domain_summary = _read_json(bundle_dir / "domain" / "macro_summary.json")
        niche_profile_df = pd.read_parquet(bundle_dir / "niche" / "macro_profile.parquet")
        niche_summary = _read_json(bundle_dir / "niche" / "macro_summary.json")

        reference_program_df, reference_program_summary_map, _ = load_cross_sample_program_profiles(
            out_root=out_root,
            current_sample_id=context.sample_id,
            cancer_type=context.cancer_type,
            cfg=cfg,
        )
        program_pairs_df = build_program_cross_sample_comparability_table(
            current_program_profile_df=program_profile_df,
            reference_program_profile_df=reference_program_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        program_cross_sample_summary = build_program_cross_sample_summary_payload(
            bundle=context,
            current_program_profile_df=program_profile_df,
            cross_sample_pairs_df=program_pairs_df,
            reference_program_profile_df=reference_program_df,
            reference_program_summary_map=reference_program_summary_map,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        program_cross_sample_md = build_program_cross_sample_summary_markdown(program_cross_sample_summary)

        reference_domain_df, reference_domain_summary_map, _ = load_cross_sample_domain_profiles(
            out_root=out_root,
            current_sample_id=context.sample_id,
            cancer_type=context.cancer_type,
            cfg=cfg,
        )
        domain_pairs_df = build_domain_cross_sample_comparability_table(
            current_domain_profile_df=domain_profile_df,
            reference_domain_profile_df=reference_domain_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        domain_cross_sample_summary = build_domain_cross_sample_summary_payload(
            bundle=context,
            current_domain_profile_df=domain_profile_df,
            cross_sample_pairs_df=domain_pairs_df,
            reference_domain_profile_df=reference_domain_df,
            reference_domain_summary_map=reference_domain_summary_map,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        domain_cross_sample_md = build_domain_cross_sample_summary_markdown(domain_cross_sample_summary)

        reference_niche_df, reference_niche_summary_map, _ = load_cross_sample_niche_profiles(
            out_root=out_root,
            current_sample_id=context.sample_id,
            cancer_type=context.cancer_type,
            cfg=cfg,
        )
        niche_pairs_df = build_niche_cross_sample_comparability_table(
            current_niche_profile_df=niche_profile_df,
            reference_niche_profile_df=reference_niche_df,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        niche_cross_sample_summary = build_niche_cross_sample_summary_payload(
            bundle=context,
            current_niche_profile_df=niche_profile_df,
            cross_sample_pairs_df=niche_pairs_df,
            reference_niche_profile_df=reference_niche_df,
            reference_niche_summary_map=reference_niche_summary_map,
            component_axes=component_axes,
            role_axes=role_axes,
            cfg=cfg,
        )
        niche_cross_sample_md = build_niche_cross_sample_summary_markdown(niche_cross_sample_summary)

        program_comp_path = bundle_dir / "program" / "cross_sample_comparability.parquet"
        program_comp_json_path = bundle_dir / "program" / "cross_sample_summary.json"
        program_comp_md_path = bundle_dir / "program" / "cross_sample_summary.md"
        domain_comp_path = bundle_dir / "domain" / "cross_sample_comparability.parquet"
        domain_comp_json_path = bundle_dir / "domain" / "cross_sample_summary.json"
        domain_comp_md_path = bundle_dir / "domain" / "cross_sample_summary.md"
        niche_comp_path = bundle_dir / "niche" / "cross_sample_comparability.parquet"
        niche_comp_json_path = bundle_dir / "niche" / "cross_sample_summary.json"
        niche_comp_md_path = bundle_dir / "niche" / "cross_sample_summary.md"

        write_parquet(program_comp_path, program_pairs_df)
        write_json(program_comp_json_path, program_cross_sample_summary)
        write_text(program_comp_md_path, program_cross_sample_md)
        write_parquet(domain_comp_path, domain_pairs_df)
        write_json(domain_comp_json_path, domain_cross_sample_summary)
        write_text(domain_comp_md_path, domain_cross_sample_md)
        write_parquet(niche_comp_path, niche_pairs_df)
        write_json(niche_comp_json_path, niche_cross_sample_summary)
        write_text(niche_comp_md_path, niche_cross_sample_md)

        sample_summary_view = build_sample_summary_view(
            program_summary,
            domain_summary=domain_summary,
            niche_summary=niche_summary,
            program_comparability_summary=program_cross_sample_summary,
            domain_comparability_summary=domain_cross_sample_summary,
            niche_comparability_summary=niche_cross_sample_summary,
        )
        sample_summary_md = build_sample_summary_markdown(sample_summary_view)
        write_json(bundle_dir / "sample" / "macro_summary.json", sample_summary_view)
        write_text(bundle_dir / "sample" / "macro_summary.md", sample_summary_md)

        manifest = _read_json(bundle_dir / "manifest.json")
        manifest = enrich_manifest_with_cross_sample_outputs(
            manifest,
            bundle_root=bundle_dir,
            program_paths=(program_comp_path, program_comp_json_path, program_comp_md_path),
            domain_paths=(domain_comp_path, domain_comp_json_path, domain_comp_md_path),
            niche_paths=(niche_comp_path, niche_comp_json_path, niche_comp_md_path),
        )
        write_json(bundle_dir / "manifest.json", manifest)
        written_bundles.append(bundle_dir)

    return written_bundles
