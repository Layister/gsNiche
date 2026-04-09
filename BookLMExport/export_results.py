from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TEXT_EXTENSIONS = {".json", ".tsv", ".txt", ".csv", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
BINARY_TABLE_EXTENSION = ".parquet"
SUPPORTED_STAGE_BUNDLES = ("program_bundle", "domain_bundle", "niche_bundle")
DEFAULT_SOURCE_ROOT = Path("/Users/wuyang/Documents/gsNiche典型结果/results")
DEFAULT_OUTPUT_ROOT = Path("/Users/wuyang/Documents/gsNiche典型结果/exports")

EXCLUDED_PATH_PARTS = {"qc", "qc_tables", "__pycache__"}
EXCLUDED_FILE_NAMES = {
    ".ds_store",
    "manifest.json",
    "qc_report.json",
}
EXCLUDED_FILE_PREFIXES = ("qc_",)
CORE_STAGE_RULES = {
    "program": {
        "tables": {
            "programs.parquet",
            "program_activation.parquet",
            "program_annotation/program_annotation_hits.tsv",
        },
        "json_md": {
            "program_annotation/program_annotation_summary.json",
            "program_annotation/annotation_meta.json",
            "program_meta.json",
        },
        "figures": {
            "plot/program_annotation_hclust_heatmap.png",
            "plot/program_program_cosine_distance_heatmap.png",
            "plot/program_activation_overlap.png",
        },
        "figure_prefixes": ("plot/program_activation_overlap.",),
    },
    "domain": {
        "tables": {
            "domains.parquet",
            "domain_program_map.parquet",
            "domain_spot_membership.parquet",
            "domain_graph.parquet",
            "domain_annotation/domain_annotation_table.tsv",
        },
        "json_md": {
            "domain_annotation/annotation_meta.json",
            "domain_annotation/domain_objective_data.json",
            "domain_meta.json",
        },
        "figures": set(),
        "figure_prefixes": ("plot/qc_plot.",),
    },
    "niche": {
        "tables": {
            "niche_structures.parquet",
            "niche_membership.parquet",
            "domain_adjacency_edges.parquet",
            "niche_annotation/niche_interpretation.tsv",
            "niche_annotation/niche_top_interfaces.tsv",
        },
        "json_md": {
            "niche_annotation/niche_summary.json",
            "niche_annotation/annotation_visualization/annotation_visualization_summary.json",
            "niche_meta.json",
            "niche_report.json",
            "plot/niche_summary.json",
        },
        "figures": {
            "plot/niche_structure_overview.png",
            "plot/niche_program_composition.png",
            "plot/niche_footprint_panels.png",
            "plot/niche_edge_graph.png",
            "niche_annotation/annotation_visualization/annotation_structure_overview.png",
            "niche_annotation/annotation_visualization/annotation_interface_panels.png",
            "niche_annotation/annotation_visualization/annotation_term_evidence_heatmap.png",
        },
        "figure_prefixes": (),
    },
}


@dataclass(frozen=True)
class ExportTask:
    source: Path
    destination: Path
    stage: str
    kind: str


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert gsNiche result bundles into NotebookLM-friendly assets. "
            "Non-QC program/domain/niche outputs are reorganized into Markdown, "
            "CSV/TSV/JSON, and copied figures."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=(
            "Root directory like /path/to/results containing cancer/sample folders. "
            f"Default: {DEFAULT_SOURCE_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory for NotebookLM-friendly exports. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--cancers",
        nargs="*",
        default=None,
        help="Optional list of cancer directories to export. Default: all cancers under source root.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional list of sample ids to export across selected cancers. Default: all samples.",
    )
    parser.add_argument(
        "--parquet-format",
        choices=("csv", "tsv"),
        default="csv",
        help="Text format used when converting parquet files. Default: csv.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap applied when converting parquet files.",
    )
    return parser


def _iter_cancer_dirs(source_root: Path, selected: set[str] | None) -> Iterable[Path]:
    for path in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if selected and path.name not in selected:
            continue
        yield path


def _iter_sample_dirs(cancer_dir: Path, selected: set[str] | None) -> Iterable[Path]:
    for path in sorted(p for p in cancer_dir.iterdir() if p.is_dir()):
        if selected and path.name not in selected:
            continue
        yield path


def _is_visualization_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def _is_parquet_file(path: Path) -> bool:
    return path.suffix.lower() == BINARY_TABLE_EXTENSION


def _should_exclude(path: Path) -> bool:
    parts_lower = {part.lower() for part in path.parts}
    if parts_lower & EXCLUDED_PATH_PARTS:
        return True
    if path.name.lower() in EXCLUDED_FILE_NAMES:
        return True
    if not _is_visualization_file(path) and any(
        path.name.lower().startswith(prefix) for prefix in EXCLUDED_FILE_PREFIXES
    ):
        return True
    return False


def _classify_task(bundle_dir: Path, file_path: Path) -> ExportTask | None:
    if _should_exclude(file_path):
        return None
    if not (
        _is_visualization_file(file_path)
        or _is_text_file(file_path)
        or _is_parquet_file(file_path)
    ):
        return None

    stage = bundle_dir.name.removesuffix("_bundle")
    relpath = file_path.relative_to(bundle_dir)
    relpath_posix = relpath.as_posix()
    stage_rules = CORE_STAGE_RULES.get(stage)
    if stage_rules is None:
        return None

    allowed = False
    if _is_visualization_file(file_path):
        allowed = relpath_posix in stage_rules["figures"] or any(
            relpath_posix.startswith(prefix) for prefix in stage_rules["figure_prefixes"]
        )
    elif _is_parquet_file(file_path) or file_path.suffix.lower() in {".tsv", ".csv"}:
        allowed = relpath_posix in stage_rules["tables"]
    elif file_path.suffix.lower() == ".json":
        allowed = relpath_posix in stage_rules["json_md"]
    elif file_path.suffix.lower() == ".md":
        allowed = relpath_posix in stage_rules["json_md"]

    if not allowed:
        return None

    export_relpath = relpath
    kind = "document"

    if _is_parquet_file(file_path):
        export_relpath = relpath.with_suffix("")
        kind = "table"
    elif _is_visualization_file(file_path):
        kind = "figure"
    elif file_path.suffix.lower() in {".json", ".tsv", ".csv"}:
        kind = "table"

    return ExportTask(
        source=file_path,
        destination=export_relpath,
        stage=stage,
        kind=kind,
    )


def _collect_export_tasks(sample_dir: Path) -> list[ExportTask]:
    tasks: list[ExportTask] = []
    for bundle_name in SUPPORTED_STAGE_BUNDLES:
        bundle_dir = sample_dir / bundle_name
        if not bundle_dir.is_dir():
            continue
        for file_path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
            task = _classify_task(bundle_dir, file_path)
            if task is not None:
                tasks.append(task)
    return tasks


def _copy_or_convert_file(
    task: ExportTask,
    destination_root: Path,
    cancer: str,
    sample: str,
    parquet_format: str,
    max_rows: int | None,
) -> Path:
    prefix = f"{cancer}.{sample}."
    flattened_name = task.destination.as_posix().replace("/", "__")
    destination = destination_root / task.stage / flattened_name
    destination.parent.mkdir(parents=True, exist_ok=True)

    if _is_parquet_file(task.source):
        df = pd.read_parquet(task.source)
        if max_rows is not None:
            df = df.head(max_rows)
        if parquet_format == "csv":
            destination = destination.with_suffix(".csv")
        else:
            destination = destination.with_suffix(".tsv")
        destination = destination.with_name(f"{prefix}{destination.name}")
        df.to_csv(destination, index=False, sep="\t" if parquet_format == "tsv" else ",")
        return destination

    if task.source.suffix.lower() == ".tsv":
        df = pd.read_csv(task.source, sep="\t")
        if max_rows is not None:
            df = df.head(max_rows)
        destination = destination.with_suffix(".csv")
        destination = destination.with_name(f"{prefix}{destination.name}")
        df.to_csv(destination, index=False)
        return destination

    if task.source.suffix.lower() == ".json":
        payload = _load_json_if_exists(task.source)
        destination = destination.with_suffix(".md")
        destination = destination.with_name(f"{prefix}{destination.name}")
        destination.write_text(_json_to_markdown(task.source.stem, payload), encoding="utf-8")
        return destination

    destination = destination.with_name(f"{prefix}{destination.name}")
    shutil.copy2(task.source, destination)
    return destination


def _load_json_if_exists(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_table_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return None


def _json_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _json_to_markdown(title: str, payload: dict | list | None) -> str:
    lines = [f"# {title}", ""]
    lines.extend(_render_json_markdown(payload, level=2))
    if len(lines) == 2:
        lines.append("No content.")
    lines.append("")
    return "\n".join(lines)


def _render_json_markdown(payload: object, level: int) -> list[str]:
    heading = "#" * max(level, 2)
    if isinstance(payload, dict):
        if not payload:
            return ["Empty object."]
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{heading} {key}")
                lines.append("")
                lines.extend(_render_json_markdown(value, level + 1))
                lines.append("")
            else:
                lines.append(f"- **{key}**: {_json_scalar(value)}")
        return lines
    if isinstance(payload, list):
        if not payload:
            return ["Empty list."]
        if all(isinstance(item, dict) for item in payload):
            frames = [pd.json_normalize(item) for item in payload]
            table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not table.empty:
                return [table.to_markdown(index=False)]
        lines = []
        for idx, item in enumerate(payload, start=1):
            if isinstance(item, (dict, list)):
                lines.append(f"{heading} Item {idx}")
                lines.append("")
                lines.extend(_render_json_markdown(item, level + 1))
                lines.append("")
            else:
                lines.append(f"- {_json_scalar(item)}")
        return lines
    return [_json_scalar(payload)]


def _format_float(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return str(value)


def _top_program_lines(program_summary: list[dict] | None, limit: int = 8) -> list[str]:
    if not program_summary:
        return ["No program annotation summary was exported."]
    lines: list[str] = []
    for item in program_summary[:limit]:
        lines.append(
            (
                f"- `{item.get('program_id', 'NA')}`: {item.get('summary_text', 'No summary')} "
                f"(confidence={_format_float(item.get('annotation_confidence'))}, "
                f"program_confidence={_format_float(item.get('program_confidence'))})"
            )
        )
    return lines


def _top_domain_lines(domain_table: pd.DataFrame | None, limit: int = 8) -> list[str]:
    if domain_table is None or domain_table.empty:
        return ["No domain annotation table was exported."]
    ordered = domain_table.sort_values(
        by=["domain_reliability", "spot_count"],
        ascending=[False, False],
    ).head(limit)
    lines: list[str] = []
    for row in ordered.itertuples(index=False):
        lines.append(
            (
                f"- `{row.domain_id}` / `{row.program_seed_id}`: {row.domain_annotation_label}, "
                f"morphology={row.morphology_label}, salience={row.salience_label}, "
                f"reliability={_format_float(row.domain_reliability)}, spots={row.spot_count}"
            )
        )
    return lines


def _top_niche_lines(niche_summary: dict | None, limit: int = 8) -> list[str]:
    if not niche_summary or not niche_summary.get("niches"):
        return ["No niche summary was exported."]
    lines: list[str] = []
    for item in niche_summary["niches"][:limit]:
        lines.append(
            (
                f"- `{item.get('niche_id', 'NA')}`: {item.get('summary_text', 'No summary')} "
                f"(members={len(item.get('members', []))}, "
                f"contact={_format_float(item.get('contact_fraction'))}, "
                f"overlap={_format_float(item.get('overlap_fraction'))})"
            )
        )
    return lines


def _summarize_stage_assets(exported_files: list[Path], stage_root: Path) -> list[str]:
    relpaths = [path.relative_to(stage_root) for path in exported_files if path.is_file()]
    if not relpaths:
        return ["No files exported."]
    lines: list[str] = []
    counts = Counter(path.suffix.lower() for path in relpaths)
    counts_text = ", ".join(f"{ext or '[no ext]'}={count}" for ext, count in sorted(counts.items()))
    lines.append(f"Exported {len(relpaths)} files: {counts_text}.")
    return lines


def _write_sample_overview(
    source_sample_dir: Path,
    output_sample_dir: Path,
    cancer: str,
    sample: str,
    exported_files: list[Path],
) -> None:
    program_summary = _load_json_if_exists(
        source_sample_dir / "program_bundle" / "program_annotation" / "program_annotation_summary.json"
    )
    domain_table = _load_table_if_exists(
        source_sample_dir / "domain_bundle" / "domain_annotation" / "domain_annotation_table.tsv"
    )
    if domain_table is None:
        raw_domain_tsv = source_sample_dir / "domain_bundle" / "domain_annotation" / "domain_annotation_table.tsv"
        if raw_domain_tsv.exists():
            domain_table = pd.read_csv(raw_domain_tsv, sep="\t")
    niche_summary = _load_json_if_exists(
        source_sample_dir / "niche_bundle" / "niche_annotation" / "niche_summary.json"
    )

    stage_to_files: dict[str, list[Path]] = {"program": [], "domain": [], "niche": []}
    for path in exported_files:
        relparts = path.relative_to(output_sample_dir).parts
        if relparts:
            stage_to_files.setdefault(relparts[0], []).append(path)

    lines = [
        f"# gsNiche NotebookLM Export: {cancer} / {sample}",
        "",
        "This folder is a NotebookLM-friendly export of non-QC gsNiche outputs.",
        "Binary parquet files were converted into CSV tables, JSON files into Markdown, and figures were copied directly.",
        "",
        "## Stage Inventory",
    ]

    for stage in ("program", "domain", "niche"):
        lines.append(f"### {stage.title()}")
        lines.extend(_summarize_stage_assets(stage_to_files.get(stage, []), output_sample_dir))
        lines.append("")

    lines.extend(
        [
            "## Program Highlights",
            *(_top_program_lines(program_summary if isinstance(program_summary, list) else None)),
            "",
            "## Domain Highlights",
            *(_top_domain_lines(domain_table)),
            "",
            "## Niche Highlights",
            *(_top_niche_lines(niche_summary if isinstance(niche_summary, dict) else None)),
            "",
            "## Exported Figures",
        ]
    )

    figure_paths = sorted(
        path.relative_to(output_sample_dir).as_posix()
        for path in exported_files
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if figure_paths:
        lines.extend(f"- `{path}`" for path in figure_paths)
    else:
        lines.append("- No figures exported.")
    lines.append("")

    lines.extend(
        [
            "## Notes",
            "- This export keeps object-complete core files rather than every non-QC file.",
            f"- Exported filenames are prefixed with `{cancer}.{sample}.` for uniqueness.",
            "- All domain `qc_plot.*` images were retained by request.",
            "- `program_annotation_umap.png` was intentionally excluded.",
            "- `domain_biological_annotation` was intentionally excluded.",
            "- `manifest.json` files were excluded because this export generates its own summary documents.",
            "- Converted tables preserve the original relative stage structure where possible.",
            "",
        ]
    )

    overview_path = output_sample_dir / "sample_overview.md"
    overview_path.write_text("\n".join(lines), encoding="utf-8")


def _write_catalog(output_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "export_catalog.tsv", sep="\t", index=False)


def export_results_tree(
    source_root: Path,
    output_root: Path,
    cancers: Iterable[str] | None = None,
    samples: Iterable[str] | None = None,
    parquet_format: str = "csv",
    max_rows: int | None = None,
) -> list[Path]:
    selected_cancers = set(cancers) if cancers else None
    selected_samples = set(samples) if samples else None
    exported_sample_dirs: list[Path] = []
    catalog_rows: list[dict[str, object]] = []

    output_root.mkdir(parents=True, exist_ok=True)
    catalog_path = output_root / "export_catalog.tsv"
    if catalog_path.exists():
        catalog_path.unlink()

    for cancer_dir in _iter_cancer_dirs(source_root, selected_cancers):
        for sample_dir in _iter_sample_dirs(cancer_dir, selected_samples):
            tasks = _collect_export_tasks(sample_dir)
            if not tasks:
                continue

            output_sample_dir = output_root / cancer_dir.name / sample_dir.name
            if output_sample_dir.exists():
                shutil.rmtree(output_sample_dir)
            exported_files: list[Path] = []
            for task in tasks:
                exported_path = _copy_or_convert_file(
                    task=task,
                    destination_root=output_sample_dir,
                    cancer=cancer_dir.name,
                    sample=sample_dir.name,
                    parquet_format=parquet_format,
                    max_rows=max_rows,
                )
                exported_files.append(exported_path)
                catalog_rows.append(
                    {
                        "cancer": cancer_dir.name,
                        "sample": sample_dir.name,
                        "stage": task.stage,
                        "kind": task.kind,
                        "source_path": str(task.source),
                        "export_path": str(exported_path),
                    }
                )

            _write_sample_overview(
                source_sample_dir=sample_dir,
                output_sample_dir=output_sample_dir,
                cancer=cancer_dir.name,
                sample=sample_dir.name,
                exported_files=exported_files,
            )
            exported_sample_dirs.append(output_sample_dir)

    _write_catalog(output_root=output_root, rows=catalog_rows)
    return exported_sample_dirs


def main() -> int:
    args = _build_cli().parse_args()
    export_results_tree(
        source_root=args.source_root,
        output_root=args.output_root,
        cancers=args.cancers,
        samples=args.samples,
        parquet_format=args.parquet_format,
        max_rows=args.max_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
