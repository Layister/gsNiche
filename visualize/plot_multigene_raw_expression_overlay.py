from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(PROJECT_ROOT / ".cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DEFAULT_H5AD_ROOT = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "COAD"
DEFAULT_SAMPLE = "TENX89"
DEFAULT_GENES = ["PLA2G2A", "TNXB", "CXCL12"]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "multigene_raw_expression"

# 高区分度颜色：蓝、红、绿、紫、橙、青。前三个就是默认三基因。
DEFAULT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _normalize_gene_name(gene: object) -> str:
    text = str(gene).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.upper()


def _deduplicate_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _load_gene_list(genes_arg: str | None, gene_file: str | None) -> list[str]:
    values: list[str] = []
    if genes_arg:
        values.extend(part.strip() for part in str(genes_arg).split(","))
    if gene_file:
        path = Path(gene_file)
        if not path.exists():
            raise FileNotFoundError(f"Gene file not found: {path}")
        values.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    genes = _deduplicate_keep_order(_normalize_gene_name(x) for x in values if _normalize_gene_name(x))
    if not genes:
        raise ValueError("No valid genes were provided. Use --genes or --gene-file.")
    return genes


def _resolve_h5ad_path(cancer: str, sample: str, h5ad_path: str | None) -> Path:
    if h5ad_path:
        path = Path(h5ad_path)
        if not path.exists():
            raise FileNotFoundError(f"h5ad file not found: {path}")
        return path

    path = DEFAULT_H5AD_ROOT / cancer / "ST" / f"{sample}.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"h5ad file not found: {path}")
    return path


def _to_dense_1d(x: object) -> np.ndarray:
    if hasattr(x, "toarray"):
        arr = x.toarray()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            raise ValueError(f"Expected a vector but got shape {arr.shape}")
    return arr.astype(np.float64, copy=False)


def _extract_gene_expression(adata: object, gene: str) -> tuple[np.ndarray, str]:
    gene_norm = _normalize_gene_name(gene)

    raw = getattr(adata, "raw", None)
    if raw is not None:
        raw_names = np.asarray(getattr(raw, "var_names", []), dtype=str)
        raw_map = {_normalize_gene_name(name): idx for idx, name in enumerate(raw_names)}
        if gene_norm in raw_map:
            idx = int(raw_map[gene_norm])
            values = _to_dense_1d(raw.X[:, idx])
            return values, "adata.raw.X"

    var_names = np.asarray(getattr(adata, "var_names", []), dtype=str)
    var_map = {_normalize_gene_name(name): idx for idx, name in enumerate(var_names)}
    if gene_norm in var_map:
        idx = int(var_map[gene_norm])
        values = _to_dense_1d(adata.X[:, idx])
        return values, "adata.X"

    raise KeyError(gene)


def _resolve_output_path(output: str | None, cancer: str, sample: str, genes: list[str]) -> Path:
    if output:
        return Path(output)
    gene_label = "_".join(genes[:6])
    if len(genes) > 6:
        gene_label = f"{gene_label}_plus{len(genes) - 6}"
    return DEFAULT_OUTPUT_ROOT / cancer / f"{sample}.raw_multigene_overlay.{gene_label}.png"


def _load_h5ad(path: Path) -> object:
    try:
        import anndata as ad
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("anndata is required to read h5ad for raw expression plotting.") from exc
    return ad.read_h5ad(path)


def _normalize_for_overlay(expr: np.ndarray, vmax_quantile: float) -> np.ndarray:
    expr = np.asarray(expr, dtype=np.float64)
    expr = np.where(np.isfinite(expr), expr, 0.0)
    positive = expr[expr > 0]
    if positive.size == 0:
        return np.zeros_like(expr, dtype=np.float64)
    q = float(np.clip(vmax_quantile, 0.1, 1.0))
    vmax = float(np.quantile(positive, q))
    vmax = max(vmax, float(positive.max()), 1e-8) if vmax <= 0 else vmax
    return np.clip(expr / vmax, 0.0, 1.0)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay multiple raw gene expression distributions in one spatial plot. "
            "Each spot is colored by the strongest selected gene after per-gene normalization."
        ),
    )
    parser.add_argument("--cancer", default=DEFAULT_CANCER, type=str, help=f"Cancer type (default: {DEFAULT_CANCER}).")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE, type=str, help=f"Sample id (default: {DEFAULT_SAMPLE}).")
    parser.add_argument("--h5ad", default=None, type=str, help="Optional explicit raw h5ad path.")
    parser.add_argument(
        "--genes",
        default=",".join(DEFAULT_GENES),
        type=str,
        help=f"Comma-separated genes (default: {','.join(DEFAULT_GENES)}).",
    )
    parser.add_argument("--gene-file", default=None, type=str, help="Optional one-gene-per-line file.")
    parser.add_argument("--out", default=None, type=str, help="Output PNG path.")
    parser.add_argument(
        "--colors",
        default=",".join(DEFAULT_COLORS),
        type=str,
        help="Comma-separated matplotlib colors. Reused cyclically if fewer than genes.",
    )
    parser.add_argument(
        "--min-normalized",
        default=0.15,
        type=float,
        help="Only color spots whose strongest normalized gene expression is at least this value.",
    )
    parser.add_argument("--spot-size", default=16.0, type=float, help="Colored scatter marker size.")
    parser.add_argument("--background-size", default=7.0, type=float, help="Gray background marker size.")
    parser.add_argument("--alpha", default=0.88, type=float, help="Colored scatter alpha.")
    parser.add_argument("--background-alpha", default=0.28, type=float, help="Background scatter alpha.")
    parser.add_argument("--vmax-quantile", default=0.98, type=float, help="Per-gene upper clipping quantile.")
    parser.add_argument("--dpi", default=240, type=int, help="Figure DPI.")
    parser.add_argument("--fig-width", default=6.2, type=float, help="Figure width.")
    parser.add_argument("--fig-height", default=5.8, type=float, help="Figure height.")
    parser.add_argument("--title", default=None, type=str, help="Optional figure title.")
    parser.add_argument("--summary-json", default=None, type=str, help="Optional JSON summary output path.")
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Vertically flip the spatial plot. Default keeps the native orientation.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    genes = _load_gene_list(args.genes, args.gene_file)
    h5ad_path = _resolve_h5ad_path(cancer=str(args.cancer), sample=str(args.sample), h5ad_path=args.h5ad)
    adata = _load_h5ad(h5ad_path)

    if "spatial" not in adata.obsm:
        raise ValueError(f"adata.obsm['spatial'] is missing in: {h5ad_path}")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid spatial coordinates shape: {coords.shape}")
    coords = coords[:, :2]

    expr_by_gene: dict[str, np.ndarray] = {}
    norm_by_gene: dict[str, np.ndarray] = {}
    source_map: dict[str, str] = {}
    missing_genes: list[str] = []
    for gene in genes:
        try:
            expr, source_label = _extract_gene_expression(adata=adata, gene=gene)
        except KeyError:
            missing_genes.append(gene)
            continue
        expr_by_gene[gene] = expr
        norm_by_gene[gene] = _normalize_for_overlay(expr, vmax_quantile=float(args.vmax_quantile))
        source_map[gene] = source_label

    plotted_genes = [gene for gene in genes if gene in norm_by_gene]
    if not plotted_genes:
        raise ValueError(f"No requested genes were found in {h5ad_path}: {genes}")

    norm_matrix = np.column_stack([norm_by_gene[gene] for gene in plotted_genes])
    dominant_idx = np.argmax(norm_matrix, axis=1)
    dominant_value = np.max(norm_matrix, axis=1)
    keep = dominant_value >= float(args.min_normalized)

    colors = [part.strip() for part in str(args.colors).split(",") if part.strip()]
    if not colors:
        colors = DEFAULT_COLORS
    gene_colors = {gene: colors[idx % len(colors)] for idx, gene in enumerate(plotted_genes)}

    fig, ax = plt.subplots(1, 1, figsize=(float(args.fig_width), float(args.fig_height)), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c="#D0D0D0",
        s=float(args.background_size),
        alpha=float(args.background_alpha),
        linewidths=0.0,
    )

    # 先画较弱点，再画较强点，强表达点留在最上层。
    order = np.argsort(dominant_value)
    for gene_idx, gene in enumerate(plotted_genes):
        mask = keep & (dominant_idx == gene_idx)
        mask_order = order[mask[order]]
        if mask_order.size == 0:
            continue
        ax.scatter(
            coords[mask_order, 0],
            coords[mask_order, 1],
            c=gene_colors[gene],
            s=float(args.spot_size),
            alpha=float(args.alpha),
            linewidths=0.0,
            label=gene,
        )

    title = args.title or f"{args.cancer} / {args.sample} raw gene overlay"
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if args.flip_y:
        ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=gene_colors[gene], markersize=8, label=gene)
        for gene in plotted_genes
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=9,
        title="Dominant gene",
        title_fontsize=9,
    )
    fig.tight_layout()

    out_path = _resolve_output_path(output=args.out, cancer=str(args.cancer), sample=str(args.sample), genes=plotted_genes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "cancer": str(args.cancer),
        "sample": str(args.sample),
        "h5ad": str(h5ad_path),
        "output": str(out_path),
        "genes_requested": genes,
        "genes_plotted": plotted_genes,
        "missing_genes": missing_genes,
        "expression_source": source_map,
        "colors": gene_colors,
        "min_normalized": float(args.min_normalized),
        "vmax_quantile": float(args.vmax_quantile),
        "colored_spot_count": int(keep.sum()),
        "dominant_gene_spot_count": {
            gene: int(np.sum(keep & (dominant_idx == idx))) for idx, gene in enumerate(plotted_genes)
        },
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] cancer={args.cancer}")
    print(f"[ok] sample={args.sample}")
    print(f"[ok] h5ad={h5ad_path}")
    print(f"[ok] output={out_path}")
    print(f"[ok] summary={summary_path}")
    print(f"[ok] genes_plotted={','.join(plotted_genes)}")
    if missing_genes:
        print(f"[warn] missing_genes={','.join(missing_genes)}")


if __name__ == "__main__":
    main()
