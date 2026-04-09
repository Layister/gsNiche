from __future__ import annotations

from pathlib import Path

from .domain_annotation.interpret_domains import DomainAnnotationConfig, run_domain_annotation
from .niche_annotation.interpret_niches import NicheAnnotationConfig, run_niche_annotation
from .program_annotation.interpret_programs import ProgramAnnotationConfig, run_program_annotation
from .schema import ProgramAnnotationResourceConfig


def run_biology_annotation_pipeline(
    program_bundle_path: str | Path,
    out_dir: str | Path | None = None,
    config: ProgramAnnotationConfig | None = None,
    resources: ProgramAnnotationResourceConfig | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    res = resources or ProgramAnnotationResourceConfig()
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    source_profile_yaml = root / res.source_profile_yaml_relpath

    return run_program_annotation(
        program_bundle_path=program_bundle_path,
        annotation_source=res.annotation_source,
        source_profile_yaml=source_profile_yaml,
        out_dir=out_dir,
        config=config,
    )


def run_domain_biology_annotation_pipeline(
    domain_bundle_path: str | Path,
    program_annotation_summary: str | Path | None = None,
    out_dir: str | Path | None = None,
    config: DomainAnnotationConfig | None = None,
) -> Path:
    return run_domain_annotation(
        domain_bundle_path=domain_bundle_path,
        program_annotation_summary=program_annotation_summary,
        out_dir=out_dir,
        config=config,
    )


def run_niche_biology_annotation_pipeline(
    niche_bundle_path: str | Path,
    out_dir: str | Path | None = None,
    config: NicheAnnotationConfig | None = None,
) -> Path:
    return run_niche_annotation(
        niche_bundle_path=niche_bundle_path,
        out_dir=out_dir,
        config=config,
    )
