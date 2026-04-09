from __future__ import annotations

import argparse
import json
import math
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
import pandas as pd


DEFAULT_RESULTS_ROOT = Path("/Users/wuyang/Documents/gsNiche典型结果/results")
DEFAULT_H5AD_ROOT = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "COAD"
DEFAULT_SAMPLE = "TENX89"
DEFAULT_GENES = ["COMP", "COL11A1", "IL6", "CXCL5"]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "raw_gene_expression"


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


def _resolve_h5ad_path(
    cancer: str,
    sample: str,
    h5ad_path: str | None,
) -> Path:
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
    gene_label = "_".join(genes[:4])
    if len(genes) > 4:
        gene_label = f"{gene_label}_plus{len(genes) - 4}"
    return DEFAULT_OUTPUT_ROOT / cancer / f"{sample}.raw_expression.{gene_label}.png"


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize raw spatial gene expression for selected genes in one cancer sample.",
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), type=str, help="Results root directory.")
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
    parser.add_argument("--spot-size", default=12.0, type=float, help="Scatter marker size.")
    parser.add_argument("--alpha", default=0.95, type=float, help="Scatter alpha.")
    parser.add_argument("--cmap", default="Reds", type=str, help="Matplotlib colormap.")
    parser.add_argument("--vmax-quantile", default=0.99, type=float, help="Upper clipping quantile for color scale.")
    parser.add_argument("--dpi", default=220, type=int, help="Figure DPI.")
    parser.add_argument("--fig-width", default=4.8, type=float, help="Per-panel width.")
    parser.add_argument("--fig-height", default=4.6, type=float, help="Per-panel height.")
    parser.add_argument("--ncols", default=2, type=int, help="Number of panel columns.")
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Vertically flip the spatial plot. Default behavior keeps the native orientation used by program/domain plots.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    genes = _load_gene_list(args.genes, args.gene_file)
    h5ad_path = _resolve_h5ad_path(
        cancer=str(args.cancer),
        sample=str(args.sample),
        h5ad_path=args.h5ad,
    )

    try:
        import scanpy as sc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scanpy is required to read h5ad for raw expression plotting.") from exc

    adata = sc.read_h5ad(h5ad_path)
    if "spatial" not in adata.obsm:
        raise ValueError(f"adata.obsm['spatial'] is missing in: {h5ad_path}")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid spatial coordinates shape: {coords.shape}")
    coords = coords[:, :2]

    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(float(args.fig_width) * ncols, float(args.fig_height) * nrows),
        squeeze=False,
    )

    missing_genes: list[str] = []
    source_map: dict[str, str] = {}
    for idx, gene in enumerate(genes):
        ax = axes[idx // ncols][idx % ncols]
        try:
            expr, source_label = _extract_gene_expression(adata=adata, gene=gene)
        except KeyError:
            missing_genes.append(gene)
            ax.axis("off")
            ax.set_title(f"{gene}\nmissing", fontsize=11)
            continue

        source_map[gene] = source_label
        positive = expr[np.isfinite(expr) & (expr > 0)]
        vmax = float(np.quantile(positive, float(np.clip(args.vmax_quantile, 0.0, 1.0)))) if positive.size else 0.0
        vmax = max(vmax, float(expr.max()) if expr.size else 0.0, 1e-8)

        order = np.argsort(expr)
        sc_plot = ax.scatter(
            coords[order, 0],
            coords[order, 1],
            c=expr[order],
            s=float(args.spot_size),
            alpha=float(args.alpha),
            cmap=str(args.cmap),
            vmin=0.0,
            vmax=vmax,
            linewidths=0.0,
        )
        ax.set_title(f"{gene} ({source_label})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if args.flip_y:
            ax.invert_yaxis()
        cbar = fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    for idx in range(len(genes), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"{args.cancer} / {args.sample} raw gene expression", fontsize=14)
    fig.tight_layout()

    out_path = _resolve_output_path(
        output=args.out,
        cancer=str(args.cancer),
        sample=str(args.sample),
        genes=genes,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] cancer={args.cancer}")
    print(f"[ok] sample={args.sample}")
    print(f"[ok] h5ad={h5ad_path}")
    print(f"[ok] output={out_path}")
    print(f"[ok] genes_plotted={len(genes) - len(missing_genes)}")
    if source_map:
        source_summary = ", ".join(f"{gene}:{source}" for gene, source in source_map.items())
        print(f"[ok] expression_source={source_summary}")
    if missing_genes:
        print(f"[warn] missing_genes={','.join(missing_genes)}")


if __name__ == "__main__":
    main()
