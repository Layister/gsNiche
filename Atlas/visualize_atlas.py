from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram


DEFAULT_ATLAS_DIR = Path("/Users/wuyang/Documents/SC-ST data/IDC/ST/atlas")


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot an Atlas dendrogram plus annotation-interface heatmap."
    )
    parser.add_argument("--atlas-dir", type=str, default=str(DEFAULT_ATLAS_DIR))
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Default: <atlas-dir>/atlas_dendrogram_heatmap.png",
    )
    parser.add_argument(
        "--include-other-interfaces",
        action="store_true",
        help="Include if__other_interfaces as the last heatmap column.",
    )
    parser.add_argument(
        "--label-size",
        type=float,
        default=7.0,
        help="Font size for row labels.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def _load_inputs(atlas_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    manifest_path = atlas_dir / "manifest.json"
    feature_path = atlas_dir / "atlas_features.tsv"
    order_path = atlas_dir / "dendrogram_order.tsv"
    assign_path = atlas_dir / "archetype_assignments.tsv"
    linkage_path = atlas_dir / "linkage_matrix.tsv"

    missing = [str(path) for path in [manifest_path, feature_path, order_path, assign_path, linkage_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing atlas visualization inputs: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = pd.read_csv(feature_path, sep="\t")
    order_df = pd.read_csv(order_path, sep="\t")
    assignments = pd.read_csv(assign_path, sep="\t")
    linkage_df = pd.read_csv(linkage_path, sep="\t")
    linkage_mat = linkage_df.loc[:, ["left", "right", "distance", "leaf_count"]].to_numpy(dtype=np.float64)

    for df in (features, order_df, assignments):
        df["sample_id"] = df["sample_id"].astype(str)
        df["niche_id"] = df["niche_id"].astype(str)
        df["niche_key"] = df["niche_key"].astype(str)

    return manifest, features, order_df, assignments, linkage_mat


def _column_order(manifest: dict, features: pd.DataFrame, include_other: bool) -> list[str]:
    selected_pairs = [str(x) for x in manifest.get("selected_interface_features", [])]
    cols = [f"if__{pair}" for pair in selected_pairs if f"if__{pair}" in features.columns]
    if include_other and "if__other_interfaces" in features.columns:
        cols.append("if__other_interfaces")
    return cols


def _sample_palette(sample_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    uniq = list(dict.fromkeys(sample_ids))
    return {sample_id: cmap(i % cmap.N) for i, sample_id in enumerate(uniq)}


def _archetype_palette(archetype_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("Set2")
    uniq = list(dict.fromkeys(archetype_ids))
    return {archetype_id: cmap(i % cmap.N) for i, archetype_id in enumerate(uniq)}


def _short_interface_label(label: str) -> str:
    text = str(label)
    if text == "other_interfaces":
        return "other"
    if "|" in text:
        pair, relation = text.split("|", 1)
    else:
        pair, relation = text, "mixed"
    relation_short = {"contact": "ct", "overlap": "ov", "soft": "sf", "mixed": "mx"}.get(
        relation.strip().lower(),
        relation[:2].lower(),
    )
    return f"{pair.replace('<->', '–')}:{relation_short}"


def _make_strip(values: list[str], palette: dict[str, tuple[float, float, float, float]]) -> tuple[np.ndarray, list[str]]:
    uniq = list(dict.fromkeys(values))
    cmap = ListedColormap([palette[key] for key in uniq])
    lookup = {key: i for i, key in enumerate(uniq)}
    arr = np.asarray([[lookup[value]] for value in values], dtype=np.int64)
    return arr, uniq


def _interface_heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "atlas_interface",
        ["#fffdf5", "#fee8a8", "#fdb366", "#ef6a32", "#b30000"],
        N=256,
    )


def _sample_boundaries(sample_ids: list[str]) -> list[int]:
    boundaries: list[int] = []
    for idx in range(1, len(sample_ids)):
        if sample_ids[idx] != sample_ids[idx - 1]:
            boundaries.append(idx)
    return boundaries


def main() -> None:
    args = _build_cli().parse_args()
    atlas_dir = Path(args.atlas_dir)
    manifest, features, order_df, assignments, linkage_mat = _load_inputs(atlas_dir=atlas_dir)

    recommended_k = int(manifest.get("recommended_k", 0))
    rec_assign = assignments.loc[assignments["k"].astype(int) == recommended_k].copy()
    if rec_assign.empty:
        raise ValueError(f"No assignments found for recommended_k={recommended_k}")

    ordered = order_df.sort_values("leaf_order").reset_index(drop=True)
    plot_df = ordered.merge(features, on=["sample_id", "niche_id", "niche_key"], how="left", validate="one_to_one")
    plot_df = plot_df.merge(
        rec_assign.loc[:, ["sample_id", "niche_id", "niche_key", "archetype_id", "archetype_status"]],
        on=["sample_id", "niche_id", "niche_key"],
        how="left",
        validate="one_to_one",
    )

    heatmap_cols = _column_order(
        manifest=manifest,
        features=plot_df,
        include_other=bool(args.include_other_interfaces),
    )
    if not heatmap_cols:
        raise ValueError("No interface feature columns found for heatmap plotting.")

    heatmap = plot_df.loc[:, heatmap_cols].to_numpy(dtype=np.float64)
    col_labels = [_short_interface_label(col.replace("if__", "", 1)) for col in heatmap_cols]

    sample_ids = plot_df["sample_id"].astype(str).tolist()
    archetype_ids = plot_df["archetype_id"].fillna("unassigned").astype(str).tolist()
    sample_colors = _sample_palette(sample_ids=sample_ids)
    archetype_colors = _archetype_palette(archetype_ids=archetype_ids)
    sample_strip, sample_levels = _make_strip(sample_ids, sample_colors)
    archetype_strip, archetype_levels = _make_strip(archetype_ids, archetype_colors)
    sample_breaks = _sample_boundaries(sample_ids)

    n_rows = plot_df.shape[0]
    fig_height = max(8.0, 0.19 * n_rows + 2.4)
    fig_width = max(14.0, 0.34 * len(col_labels) + 6.8)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=int(args.dpi))
    gs = GridSpec(
        nrows=2,
        ncols=5,
        height_ratios=[0.14, 1.0],
        width_ratios=[1.6, 6.8, 0.18, 0.18, 1.15],
        left=0.055,
        right=0.985,
        bottom=0.08,
        top=0.92,
        wspace=0.06,
        hspace=0.08,
    )

    ax_d = fig.add_subplot(gs[1, 0])
    dendrogram(
        linkage_mat,
        orientation="left",
        no_labels=True,
        color_threshold=0.0,
        above_threshold_color="#4a4a4a",
        ax=ax_d,
    )
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_title("Dendrogram", fontsize=10, pad=8)
    for spine in ax_d.spines.values():
        spine.set_visible(False)

    y_min, y_max = ax_d.get_ylim()
    row_extent = [0.0, float(10 * n_rows)]
    if y_max < y_min:
        row_extent = [float(10 * n_rows), 0.0]

    ax_h = fig.add_subplot(gs[1, 1])
    im = ax_h.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap=_interface_heatmap_cmap(),
        vmin=0.0,
        extent=[-0.5, len(col_labels) - 0.5, row_extent[0], row_extent[1]],
    )
    ax_h.set_ylim(y_min, y_max)
    ax_h.set_yticks([])
    ax_h.set_xticks(np.arange(len(col_labels)))
    ax_h.set_xticklabels(col_labels, rotation=50, ha="right", fontsize=7)
    ax_h.set_title("Interface Feature Weights", fontsize=10, pad=8)
    ax_h.set_xlabel("Interaction interface feature", fontsize=8)
    ax_h.tick_params(axis="y", length=0)
    ax_h.tick_params(axis="x", length=0)
    for boundary in sample_breaks:
        boundary_y = boundary * 10.0
        ax_h.axhline(boundary_y, color="#d7d7d7", lw=0.7, zorder=3)
        ax_d.axhline(boundary_y, color="#d7d7d7", lw=0.7, zorder=3)

    cax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("normalized feature weight", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)
    cax.xaxis.set_label_position("top")
    cax.xaxis.tick_top()
    cax.set_title("Heatmap scale", fontsize=8, pad=3)

    ax_sample = fig.add_subplot(gs[1, 2])
    ax_sample.imshow(
        sample_strip,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([sample_colors[key] for key in sample_levels]),
        extent=[-0.5, 0.5, row_extent[0], row_extent[1]],
    )
    ax_sample.set_ylim(y_min, y_max)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])
    ax_sample.tick_params(length=0)
    ax_sample.set_title("S", fontsize=8, pad=8)
    for boundary in sample_breaks:
        ax_sample.axhline(boundary * 10.0, color="#ffffff", lw=0.7, zorder=3)

    ax_arch = fig.add_subplot(gs[1, 3])
    ax_arch.imshow(
        archetype_strip,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([archetype_colors[key] for key in archetype_levels]),
        extent=[-0.5, 0.5, row_extent[0], row_extent[1]],
    )
    ax_arch.set_ylim(y_min, y_max)
    ax_arch.set_xticks([])
    ax_arch.set_yticks([])
    ax_arch.tick_params(length=0)
    ax_arch.set_title(f"A\nk={recommended_k}", fontsize=8, pad=4)
    for boundary in sample_breaks:
        ax_arch.axhline(boundary * 10.0, color="#ffffff", lw=0.7, zorder=3)

    for ax in (ax_h, ax_sample, ax_arch):
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#bdbdbd")

    ax_leg = fig.add_subplot(gs[:, 4])
    ax_leg.axis("off")
    sample_handles = [Patch(facecolor=sample_colors[key], edgecolor="none", label=key) for key in sample_levels]
    arch_handles = [Patch(facecolor=archetype_colors[key], edgecolor="none", label=key) for key in archetype_levels]
    leg1 = ax_leg.legend(
        handles=sample_handles,
        loc="upper left",
        frameon=False,
        fontsize=7,
        title="Samples",
        title_fontsize=8,
        handlelength=0.9,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    ax_leg.add_artist(leg1)
    ax_leg.legend(
        handles=arch_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.47),
        frameon=False,
        fontsize=7,
        title="Archetypes",
        title_fontsize=8,
        handlelength=0.9,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    cohort_id = str(manifest.get("cohort_id", "cohort"))
    fig.suptitle(
        f"{cohort_id} Atlas: Dendrogram and Interface Feature Map",
        fontsize=13,
        y=0.975,
    )

    out_path = Path(args.out) if args.out else atlas_dir / "atlas_dendrogram_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=int(args.dpi))
    plt.close(fig)
    print(f"[ok] atlas_dir={atlas_dir}")
    print(f"[ok] recommended_k={recommended_k}")
    print(f"[ok] output={out_path}")


if __name__ == "__main__":
    main()
