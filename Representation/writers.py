from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import RepresentationInputBundle, RepresentationPipelineConfig


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")


def ensure_bundle_dirs(bundle_root: Path) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    for name in ("program", "domain", "niche", "sample"):
        (bundle_root / name).mkdir(parents=True, exist_ok=True)


def promote_bundle(tmp_bundle: Path, final_bundle: Path) -> None:
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    backup = final_bundle.parent / f"{final_bundle.name}.__bak__{int(time.time())}"
    if final_bundle.exists():
        if backup.exists():
            shutil.rmtree(backup)
        final_bundle.rename(backup)
    tmp_bundle.rename(final_bundle)
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_code_version(repo_root: Path, override: Optional[str]) -> str:
    if override:
        return str(override)
    cmd = ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    fallback = hashlib.sha1(f"{time.time()}-{repo_root}".encode("utf-8")).hexdigest()[:12]
    return f"local-{fallback}"


def build_manifest(
    bundle: RepresentationInputBundle,
    cfg: RepresentationPipelineConfig,
    axis_definition_path: Path,
    program_profile_path: Path,
    sample_burden_path: Path,
    program_summary_json_path: Path,
    program_summary_md_path: Path,
    domain_profile_path: Path,
    domain_burden_path: Path,
    domain_summary_json_path: Path,
    domain_summary_md_path: Path,
    niche_profile_path: Path,
    niche_burden_path: Path,
    niche_summary_json_path: Path,
    niche_summary_md_path: Path,
    sample_summary_json_path: Path,
    sample_summary_md_path: Path,
    qc_report_path: Path,
    cross_sample_comparability_path: Path | None = None,
    cross_sample_summary_json_path: Path | None = None,
    cross_sample_summary_md_path: Path | None = None,
    domain_cross_sample_comparability_path: Path | None = None,
    domain_cross_sample_summary_json_path: Path | None = None,
    domain_cross_sample_summary_md_path: Path | None = None,
    niche_cross_sample_comparability_path: Path | None = None,
    niche_cross_sample_summary_json_path: Path | None = None,
    niche_cross_sample_summary_md_path: Path | None = None,
) -> dict:
    program_bundle_path = Path(bundle.program_bundle_path)
    manifest_path = program_bundle_path / cfg.input.program_manifest_relpath
    inputs = {
        "program_bundle_path": str(program_bundle_path),
        "program_manifest_path": str(manifest_path),
        "program_manifest_hash": hash_file(manifest_path),
        "programs_path": str(program_bundle_path / cfg.input.programs_relpath),
        "program_activation_path": str(program_bundle_path / cfg.input.program_activation_relpath),
        "program_qc_path": str(program_bundle_path / cfg.input.program_qc_relpath),
        "program_annotation_summary_path": str(program_bundle_path / cfg.input.annotation_summary_relpath),
        "gss_bundle_path": bundle.gss_bundle_path,
    }
    annotation_path = program_bundle_path / cfg.input.annotation_summary_relpath
    if annotation_path.exists():
        inputs["program_annotation_summary_hash"] = hash_file(annotation_path)
    bundle_root = axis_definition_path.parent
    rel = lambda p: str(p.relative_to(bundle_root))

    outputs_program = {
        "macro_profile": rel(program_profile_path),
        "sample_burden": rel(sample_burden_path),
        "macro_summary_json": rel(program_summary_json_path),
        "macro_summary_md": rel(program_summary_md_path),
    }
    if cross_sample_comparability_path and cross_sample_summary_json_path and cross_sample_summary_md_path:
        outputs_program.update(
            {
                "cross_sample_comparability": rel(cross_sample_comparability_path),
                "cross_sample_summary_json": rel(cross_sample_summary_json_path),
                "cross_sample_summary_md": rel(cross_sample_summary_md_path),
            }
        )

    outputs_domain = {
        "macro_profile": rel(domain_profile_path),
        "sample_burden": rel(domain_burden_path),
        "macro_summary_json": rel(domain_summary_json_path),
        "macro_summary_md": rel(domain_summary_md_path),
    }
    if (
        domain_cross_sample_comparability_path
        and domain_cross_sample_summary_json_path
        and domain_cross_sample_summary_md_path
    ):
        outputs_domain.update(
            {
                "cross_sample_comparability": rel(domain_cross_sample_comparability_path),
                "cross_sample_summary_json": rel(domain_cross_sample_summary_json_path),
                "cross_sample_summary_md": rel(domain_cross_sample_summary_md_path),
            }
        )

    outputs_niche = {
        "macro_profile": rel(niche_profile_path),
        "sample_burden": rel(niche_burden_path),
        "macro_summary_json": rel(niche_summary_json_path),
        "macro_summary_md": rel(niche_summary_md_path),
    }
    if (
        niche_cross_sample_comparability_path
        and niche_cross_sample_summary_json_path
        and niche_cross_sample_summary_md_path
    ):
        outputs_niche.update(
            {
                "cross_sample_comparability": rel(niche_cross_sample_comparability_path),
                "cross_sample_summary_json": rel(niche_cross_sample_summary_json_path),
                "cross_sample_summary_md": rel(niche_cross_sample_summary_md_path),
            }
        )

    program_files = {
        "macro_profile.parquet": "Program object layer: one row per Program with macro scores, evidence, and confidence.",
        "sample_burden.parquet": "Program sample-statistics layer: sample-level burden and representation support aggregated from Program results.",
        "macro_summary.json": "Program structured reading layer for machine and human quick consumption.",
        "macro_summary.md": "Program human-reading layer and current main readable entry for Program-level Representation.",
    }
    if cross_sample_comparability_path and cross_sample_summary_json_path and cross_sample_summary_md_path:
        program_files.update(
            {
                "cross_sample_comparability.parquet": "Program cross-sample comparability layer: high-similarity Program pairs anchored on the current sample and compared against other samples.",
                "cross_sample_summary.json": "Program cross-sample structured summary of nearest samples, shared tendencies, and sample-specific emphases.",
                "cross_sample_summary.md": "Program cross-sample human-reading summary page for comparable Program tendencies across samples.",
            }
        )

    domain_files = {
        "macro_profile.parquet": "Domain object layer: one row per Domain with inherited Program component scores, role refinements, morphology, and confidence.",
        "sample_burden.parquet": "Domain sample-statistics layer: Domain-level component and role burdens aggregated within sample.",
        "macro_summary.json": "Domain structured reading layer summarizing dominant component/role and representative domains.",
        "macro_summary.md": "Domain human-reading layer and entry point for spatial entity-level macro interpretation.",
    }
    if (
        domain_cross_sample_comparability_path
        and domain_cross_sample_summary_json_path
        and domain_cross_sample_summary_md_path
    ):
        domain_files.update(
            {
                "cross_sample_comparability.parquet": "Domain cross-sample comparability layer: high-similarity Domain pairs anchored on the current sample and compared against other samples.",
                "cross_sample_summary.json": "Domain cross-sample structured summary of nearest samples, shared tendencies, and sample-specific entity emphases.",
                "cross_sample_summary.md": "Domain cross-sample human-reading summary page for comparable Domain tendencies across samples.",
            }
        )

    niche_files = {
        "macro_profile.parquet": "Niche assembly layer: one row per niche with Domain component/role composition and lightweight structure fields.",
        "sample_burden.parquet": "Niche sample-statistics layer: Niche-level component and role burdens aggregated within sample.",
        "macro_summary.json": "Niche structured reading layer summarizing dominant component/role composition and representative niches.",
        "macro_summary.md": "Niche human-reading layer and entry point for local assembly-level macro interpretation.",
    }
    if (
        niche_cross_sample_comparability_path
        and niche_cross_sample_summary_json_path
        and niche_cross_sample_summary_md_path
    ):
        niche_files.update(
            {
                "cross_sample_comparability.parquet": "Niche cross-sample comparability layer: high-similarity Niche pairs anchored on the current sample and compared against other samples.",
                "cross_sample_summary.json": "Niche cross-sample structured summary of nearest samples, shared local organization tendencies, and sample-specific Niche emphases.",
                "cross_sample_summary.md": "Niche cross-sample human-reading summary page for comparable Niche tendencies across samples.",
            }
        )

    return {
        "schema_version": cfg.schema_version,
        "created_at": iso_now(),
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "code_version": get_code_version(Path(__file__).resolve().parents[1], cfg.code_version_override),
        "inputs": inputs,
        "params": {
            "default_cancer_type": cfg.default_cancer_type,
            "input": cfg.input.__dict__,
            "eligibility": cfg.eligibility.__dict__,
            "scoring": cfg.scoring.__dict__,
        },
        "outputs": {
            "axis_definition": rel(axis_definition_path),
            "qc_report": rel(qc_report_path),
            "program": outputs_program,
            "domain": outputs_domain,
            "niche": outputs_niche,
            "sample": {
                "macro_summary_json": rel(sample_summary_json_path),
                "macro_summary_md": rel(sample_summary_md_path),
            },
        },
        "bundle_layout": {
            "axis_definition.json": {
                "layer": "bundle",
                "role": "definition_layer",
                "description": "Semantic contract for macro axes across the whole Representation bundle.",
            },
            "qc_report.json": {
                "layer": "bundle",
                "role": "audit_layer",
                "description": "Cross-layer QC, reliability, missingness, and diagnostics.",
            },
            "program": {
                "status": "implemented",
                "files": program_files,
            },
            "domain": {
                "status": "implemented",
                "files": domain_files,
            },
            "niche": {
                "status": "implemented",
                "files": niche_files,
            },
            "sample": {
                "status": "implemented_cross_layer",
                "files": {
                    "macro_summary.json": "Current final sample summary layer spanning Program, Domain, and Niche summaries.",
                    "macro_summary.md": "Current human-readable bundle entry point spanning Program, Domain, and Niche summaries.",
                },
            },
        },
        "timestamps": {"finished_at": iso_now()},
    }


def enrich_manifest_with_cross_sample_outputs(
    manifest: dict,
    bundle_root: Path,
    *,
    program_paths: tuple[Path, Path, Path] | None = None,
    domain_paths: tuple[Path, Path, Path] | None = None,
    niche_paths: tuple[Path, Path, Path] | None = None,
) -> dict:
    out = dict(manifest)
    outputs = dict(out.get("outputs", {}) or {})
    bundle_layout = dict(out.get("bundle_layout", {}) or {})
    rel = lambda p: str(p.relative_to(bundle_root))

    if program_paths:
        comp_path, json_path, md_path = program_paths
        program_outputs = dict(outputs.get("program", {}) or {})
        program_outputs.update(
            {
                "cross_sample_comparability": rel(comp_path),
                "cross_sample_summary_json": rel(json_path),
                "cross_sample_summary_md": rel(md_path),
            }
        )
        outputs["program"] = program_outputs
        program_layout = dict(bundle_layout.get("program", {}) or {})
        program_files = dict(program_layout.get("files", {}) or {})
        program_files.update(
            {
                "cross_sample_comparability.parquet": "Program cross-sample comparability layer: high-similarity Program pairs anchored on the current sample and compared against other samples.",
                "cross_sample_summary.json": "Program cross-sample structured summary of nearest samples, shared tendencies, and sample-specific emphases.",
                "cross_sample_summary.md": "Program cross-sample human-reading summary page for comparable Program tendencies across samples.",
            }
        )
        program_layout["files"] = program_files
        bundle_layout["program"] = program_layout

    if domain_paths:
        comp_path, json_path, md_path = domain_paths
        domain_outputs = dict(outputs.get("domain", {}) or {})
        domain_outputs.update(
            {
                "cross_sample_comparability": rel(comp_path),
                "cross_sample_summary_json": rel(json_path),
                "cross_sample_summary_md": rel(md_path),
            }
        )
        outputs["domain"] = domain_outputs
        domain_layout = dict(bundle_layout.get("domain", {}) or {})
        domain_files = dict(domain_layout.get("files", {}) or {})
        domain_files.update(
            {
                "cross_sample_comparability.parquet": "Domain cross-sample comparability layer: high-similarity Domain pairs anchored on the current sample and compared against other samples.",
                "cross_sample_summary.json": "Domain cross-sample structured summary of nearest samples, shared tendencies, and sample-specific entity emphases.",
                "cross_sample_summary.md": "Domain cross-sample human-reading summary page for comparable Domain tendencies across samples.",
            }
        )
        domain_layout["files"] = domain_files
        bundle_layout["domain"] = domain_layout

    if niche_paths:
        comp_path, json_path, md_path = niche_paths
        niche_outputs = dict(outputs.get("niche", {}) or {})
        niche_outputs.update(
            {
                "cross_sample_comparability": rel(comp_path),
                "cross_sample_summary_json": rel(json_path),
                "cross_sample_summary_md": rel(md_path),
            }
        )
        outputs["niche"] = niche_outputs
        niche_layout = dict(bundle_layout.get("niche", {}) or {})
        niche_files = dict(niche_layout.get("files", {}) or {})
        niche_files.update(
            {
                "cross_sample_comparability.parquet": "Niche cross-sample comparability layer: high-similarity Niche pairs anchored on the current sample and compared against other samples.",
                "cross_sample_summary.json": "Niche cross-sample structured summary of nearest samples, shared local organization tendencies, and sample-specific Niche emphases.",
                "cross_sample_summary.md": "Niche cross-sample human-reading summary page for comparable Niche tendencies across samples.",
            }
        )
        niche_layout["files"] = niche_files
        bundle_layout["niche"] = niche_layout

    out["outputs"] = outputs
    out["bundle_layout"] = bundle_layout
    timestamps = dict(out.get("timestamps", {}) or {})
    timestamps["finished_at"] = iso_now()
    out["timestamps"] = timestamps
    return out
