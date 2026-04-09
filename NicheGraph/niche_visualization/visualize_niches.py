from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NicheGraph.data_prep import load_niche_inputs
from NicheGraph.schema import NichePipelineConfig

DEFAULT_WORK_DIR = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "IDC"
DEFAULT_SAMPLE_ID = "TENX14"
DEFAULT_PANEL_COLS = 4


def _read_bundle_paths(niche_bundle: Path) -> tuple[Path, str]:
    manifest_path = niche_bundle / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_id = str(manifest.get("sample_id", niche_bundle.parent.name))
    domain_bundle_path = Path(
        manifest.get("inputs", {}).get("domain_bundle_path", str(niche_bundle.parent / "domain_bundle"))
    )
    return domain_bundle_path, sample_id


def _load_inputs(niche_bundle: Path) -> dict:
    domain_bundle_path, sample_id = _read_bundle_paths(niche_bundle=niche_bundle)
    structures = pd.read_parquet(niche_bundle / "niche_structures.parquet")
    membership = pd.read_parquet(niche_bundle / "niche_membership.parquet")
    domains = pd.read_parquet(domain_bundle_path / "domains.parquet")
    domain_spot_membership = pd.read_parquet(domain_bundle_path / "domain_spot_membership.parquet")
    adjacency_edges = pd.read_parquet(niche_bundle / "domain_adjacency_edges.parquet")

    structures["niche_id"] = structures["niche_id"].astype(str)
    membership["niche_id"] = membership["niche_id"].astype(str)
    membership["domain_key"] = membership["domain_key"].astype(str)
    membership["program_id"] = membership["program_id"].astype(str)
    domains["domain_key"] = domains["domain_key"].astype(str)
    domains["program_seed_id"] = domains["program_seed_id"].astype(str)
    domain_spot_membership["domain_key"] = domain_spot_membership["domain_key"].astype(str)
    domain_spot_membership["spot_id"] = domain_spot_membership["spot_id"].astype(str)
    for col in ["domain_key_i", "domain_key_j", "program_id_i", "program_id_j"]:
        adjacency_edges[col] = adjacency_edges[col].astype(str)

    return {
        "domain_bundle_path": domain_bundle_path,
        "sample_id": sample_id,
        "structures": structures,
        "membership": membership,
        "domains": domains,
        "domain_spot_membership": domain_spot_membership,
        "adjacency_edges": adjacency_edges,
    }


def _load_spot_table(domain_bundle_path: Path) -> pd.DataFrame:
    cfg = NichePipelineConfig()
    payload = load_niche_inputs(domain_bundle_path=domain_bundle_path, cfg=cfg, require_neighbors=False)
    spot_ids = np.asarray(payload["spot_ids"]).astype(str)
    coords = np.asarray(payload["spot_coords"], dtype=np.float64)
    out = pd.DataFrame({"spot_id": spot_ids, "x": coords[:, 0], "y": coords[:, 1]})
    return out[np.isfinite(out["x"]) & np.isfinite(out["y"])].reset_index(drop=True)


def _resolve_niche_ids(structures: pd.DataFrame, explicit_ids: list[str] | None) -> list[str]:
    if structures.empty:
        return []
    if explicit_ids:
        keep = {str(x) for x in explicit_ids if str(x).strip()}
        if keep:
            return [
                str(x)
                for x in structures.loc[structures["niche_id"].astype(str).isin(keep), "niche_id"].astype(str).tolist()
            ]
    return structures.sort_values(
        ["interaction_confidence", "member_count", "backbone_edge_count", "niche_id"],
        ascending=[False, False, False, True],
    )["niche_id"].astype(str).tolist()


def _program_colors(program_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    uniq = sorted({str(x) for x in program_ids if str(x)})
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(1, len(uniq)))
    return {pid: cmap(i) for i, pid in enumerate(uniq)}


def _niche_colors(niche_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    uniq = sorted({str(x) for x in niche_ids if str(x)})
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(1, len(uniq)))
    return {nid: cmap(i) for i, nid in enumerate(uniq)}


def _overlay_niche_colors(member_spots: pd.DataFrame, niche_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    uniq = [str(x) for x in niche_ids if str(x)]
    if not uniq:
        return {}
    overlap_niche_ids: list[str] = []
    if not member_spots.empty:
        overlap_spot_ids = (
            member_spots.groupby("spot_id")["niche_id"].nunique().rename("niche_overlap").reset_index()
        )
        overlap_spot_ids = set(
            overlap_spot_ids.loc[overlap_spot_ids["niche_overlap"].fillna(0).astype(int) > 1, "spot_id"].astype(str)
        )
        if overlap_spot_ids:
            overlap_niche_ids = sorted(
                member_spots.loc[member_spots["spot_id"].astype(str).isin(overlap_spot_ids), "niche_id"].astype(str).unique().tolist()
            )
    non_overlap_ids = [nid for nid in uniq if nid not in set(overlap_niche_ids)]

    colors: dict[str, tuple[float, float, float, float]] = {}
    if overlap_niche_ids:
        overlap_palette = [
            "#E53935",
            "#1E88E5",
            "#FDD835",
            "#00897B",
            "#8E24AA",
            "#FB8C00",
            "#3949AB",
            "#D81B60",
            "#43A047",
            "#00ACC1",
        ]
        for i, nid in enumerate(overlap_niche_ids):
            colors[nid] = overlap_palette[i % len(overlap_palette)]
    if non_overlap_ids:
        base_cmap = plt.colormaps.get_cmap("Pastel1").resampled(max(3, len(non_overlap_ids)))
        for i, nid in enumerate(non_overlap_ids):
            colors[nid] = base_cmap(i)
    return colors


def _grid(n_items: int, panel_cols: int) -> tuple[plt.Figure, np.ndarray]:
    cols = int(max(1, min(8, panel_cols)))
    rows = int(math.ceil(max(1, n_items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), squeeze=False)
    return fig, axes


def _padded_limits(x: pd.Series, y: pd.Series, pad_ratio: float = 0.10) -> tuple[float, float, float, float]:
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    pad_x = max(1.0, span_x * pad_ratio)
    pad_y = max(1.0, span_y * pad_ratio)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def _prepare_member_domains(membership: pd.DataFrame, domains: pd.DataFrame, niche_ids: list[str]) -> pd.DataFrame:
    if not niche_ids:
        return pd.DataFrame()
    mem = membership.loc[membership["niche_id"].isin(niche_ids)].copy()
    dom_cols = [
        "domain_key",
        "domain_id",
        "program_seed_id",
        "geo_centroid_x",
        "geo_centroid_y",
        "spot_count",
        "domain_reliability",
        "geo_area_est",
        "geo_compactness",
    ]
    dom = domains.loc[:, [c for c in dom_cols if c in domains.columns]].copy()
    dom = dom.rename(columns={"program_seed_id": "domain_program_id"})
    out = mem.merge(dom, on="domain_key", how="left", validate="many_to_one")
    out["program_id"] = out["program_id"].fillna(out.get("domain_program_id", "")).astype(str)
    out["is_backbone_member"] = out["is_backbone_member"].fillna(False).astype(bool)
    out["is_structure_member"] = out["is_structure_member"].fillna(False).astype(bool)
    out["geo_centroid_x"] = pd.to_numeric(out["geo_centroid_x"], errors="coerce")
    out["geo_centroid_y"] = pd.to_numeric(out["geo_centroid_y"], errors="coerce")
    return out[np.isfinite(out["geo_centroid_x"]) & np.isfinite(out["geo_centroid_y"])].reset_index(drop=True)


def _prepare_member_spots(
    membership: pd.DataFrame,
    domain_spot_membership: pd.DataFrame,
    spot_table: pd.DataFrame,
    niche_ids: list[str],
) -> pd.DataFrame:
    mem = membership.loc[membership["niche_id"].isin(niche_ids)].copy()
    if mem.empty:
        return pd.DataFrame()
    joined = mem.merge(domain_spot_membership.loc[:, ["domain_key", "spot_id"]], on="domain_key", how="inner")
    if joined.empty:
        return pd.DataFrame()
    joined = joined.merge(spot_table, on="spot_id", how="inner")
    joined["is_backbone_member"] = joined["is_backbone_member"].fillna(False).astype(bool)
    return joined.reset_index(drop=True)


def _prepare_internal_edges(
    structures: pd.DataFrame,
    member_domains: pd.DataFrame,
    adjacency_edges: pd.DataFrame,
    niche_ids: list[str],
) -> pd.DataFrame:
    if not niche_ids or member_domains.empty or adjacency_edges.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    struct_lookup = {str(r["niche_id"]): dict(r) for r in structures.to_dict(orient="records")}
    for niche_id, sub in member_domains.groupby("niche_id"):
        member_lookup = {str(r["domain_key"]): dict(r) for r in sub.to_dict(orient="records")}
        keys = set(member_lookup.keys())
        edge_sub = adjacency_edges.loc[
            adjacency_edges["domain_key_i"].isin(keys)
            & adjacency_edges["domain_key_j"].isin(keys)
            & adjacency_edges["is_strong_edge"].fillna(False).astype(bool)
        ].copy()
        if edge_sub.empty:
            continue
        struct = struct_lookup.get(str(niche_id), {})
        for edge in edge_sub.to_dict(orient="records"):
            ki = str(edge["domain_key_i"])
            kj = str(edge["domain_key_j"])
            li = member_lookup.get(ki, {})
            lj = member_lookup.get(kj, {})
            edge_type = "mixed" if bool(edge.get("is_strong_contact", False)) and bool(edge.get("is_strong_overlap", False)) else ("overlap" if bool(edge.get("is_strong_overlap", False)) else "contact")
            rows.append(
                {
                    "niche_id": str(niche_id),
                    "canonical_pattern_id": str(struct.get("canonical_pattern_id", "")),
                    "domain_key_i": ki,
                    "domain_key_j": kj,
                    "program_id_i": str(edge.get("program_id_i", li.get("program_id", ""))),
                    "program_id_j": str(edge.get("program_id_j", lj.get("program_id", ""))),
                    "x_i": float(li.get("geo_centroid_x", np.nan)),
                    "y_i": float(li.get("geo_centroid_y", np.nan)),
                    "x_j": float(lj.get("geo_centroid_x", np.nan)),
                    "y_j": float(lj.get("geo_centroid_y", np.nan)),
                    "edge_type": edge_type,
                    "edge_strength": float(edge.get("edge_strength", 0.0)),
                    "edge_reliability": float(edge.get("edge_reliability", 0.0)),
                    "is_backbone_edge": bool(li.get("is_backbone_member", False) and lj.get("is_backbone_member", False)),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out[np.isfinite(out["x_i"]) & np.isfinite(out["x_j"]) & np.isfinite(out["y_i"]) & np.isfinite(out["y_j"])].reset_index(drop=True)


def _write_summary(structures: pd.DataFrame, niche_ids: list[str], out_path: Path) -> None:
    cols = [
        "niche_id",
        "canonical_pattern_id",
        "component_id",
        "member_count",
        "backbone_node_count",
        "program_count",
        "program_ids",
        "backbone_program_pairs",
        "backbone_program_pair_count",
        "backbone_edge_count",
        "strong_edge_count",
        "contact_fraction",
        "overlap_fraction",
        "proximity_fraction",
        "mean_edge_strength",
        "mean_edge_reliability",
        "interaction_confidence",
        "duplicate_collapsed_from_count",
    ]
    out = structures.loc[structures["niche_id"].isin(niche_ids), [c for c in cols if c in structures.columns]].copy()
    out = out.sort_values(["interaction_confidence", "member_count", "niche_id"], ascending=[False, False, True]).reset_index(drop=True)
    out.to_json(out_path, orient="records", indent=2, force_ascii=False)


def _plot_structure_overview(structures: pd.DataFrame, niche_ids: list[str], out_path: Path) -> None:
    sub = structures.loc[structures["niche_id"].isin(niche_ids)].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)

    sc = axes[0].scatter(
        sub["backbone_edge_count"].to_numpy(dtype=float),
        sub["member_count"].to_numpy(dtype=float),
        s=45 + 18 * sub["program_count"].to_numpy(dtype=float),
        c=sub["interaction_confidence"].to_numpy(dtype=float),
        cmap="viridis",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )
    axes[0].set_xlabel("Backbone Edge Count")
    axes[0].set_ylabel("Member Count")
    axes[0].set_title("Niche Structure Overview")
    axes[0].grid(alpha=0.2, linewidth=0.6)
    for row in sub.sort_values("interaction_confidence", ascending=False).head(min(8, len(sub))).to_dict(orient="records"):
        axes[0].text(
            float(row["backbone_edge_count"]) + 0.1,
            float(row["member_count"]) + 0.1,
            str(row["niche_id"]),
            fontsize=7,
        )
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("Interaction Confidence")

    frac = sub.loc[:, ["niche_id", "contact_fraction", "overlap_fraction", "proximity_fraction"]].copy()
    frac = frac.sort_values(["contact_fraction", "overlap_fraction", "niche_id"], ascending=[False, False, True]).reset_index(drop=True)
    x = np.arange(frac.shape[0], dtype=float)
    axes[1].bar(x, frac["contact_fraction"].to_numpy(dtype=float), color="#d55e00", label="contact")
    axes[1].bar(
        x,
        frac["overlap_fraction"].to_numpy(dtype=float),
        bottom=frac["contact_fraction"].to_numpy(dtype=float),
        color="#0072b2",
        label="overlap",
    )
    axes[1].bar(
        x,
        frac["proximity_fraction"].to_numpy(dtype=float),
        bottom=(frac["contact_fraction"] + frac["overlap_fraction"]).to_numpy(dtype=float),
        color="#999999",
        label="proximity",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(frac["niche_id"].astype(str).tolist(), rotation=45, ha="right", fontsize=8)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Edge-Type Composition")
    axes[1].legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    axes[1].grid(axis="y", alpha=0.2, linewidth=0.6)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_program_composition(
    member_domains: pd.DataFrame,
    niche_ids: list[str],
    out_path: Path,
    max_programs: int = 10,
) -> None:
    if not niche_ids or member_domains.empty:
        return

    base = member_domains.loc[:, ["niche_id", "domain_key", "program_id", "is_backbone_member"]].drop_duplicates().copy()
    base["program_id"] = base["program_id"].fillna("unresolved").replace("", "unresolved").astype(str)
    base["is_backbone_member"] = base["is_backbone_member"].fillna(False).astype(bool)

    total_counts = (
        base.groupby("program_id", as_index=False)["domain_key"]
        .nunique()
        .sort_values(["domain_key", "program_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    keep_programs = total_counts["program_id"].astype(str).head(int(max(3, max_programs))).tolist()
    if not keep_programs:
        return

    base["program_group"] = np.where(base["program_id"].isin(keep_programs), base["program_id"], "other")
    program_order = keep_programs + (["other"] if "other" in set(base["program_group"].tolist()) else [])
    program_palette = _program_colors(program_order)
    if "other" in program_order:
        program_palette["other"] = "#bdbdbd"

    base["layer"] = np.where(base["is_backbone_member"].to_numpy(dtype=bool), "backbone", "members")
    layer_frames = {
        "backbone": base.loc[base["layer"] == "backbone"].copy(),
        "members": base.copy(),
    }

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(11.5, 0.78 * len(niche_ids) + 4.6), 6.8),
        sharex=True,
        constrained_layout=True,
    )

    for ax, (layer_name, layer_df) in zip(axes, layer_frames.items()):
        if layer_df.empty:
            ax.set_axis_off()
            continue
        counts = (
            layer_df.groupby(["niche_id", "program_group"], as_index=False)["domain_key"]
            .nunique()
            .rename(columns={"domain_key": "domain_count"})
        )
        pivot = (
            counts.pivot_table(index="niche_id", columns="program_group", values="domain_count", aggfunc="sum", fill_value=0.0)
            .reindex(index=niche_ids, fill_value=0.0)
        )
        for program in program_order:
            if program not in pivot.columns:
                pivot[program] = 0.0
        pivot = pivot.loc[:, program_order]
        denom = pivot.sum(axis=1).to_numpy(dtype=float)
        denom = np.where(denom > 0.0, denom, 1.0)[:, None]
        frac = pivot.to_numpy(dtype=float) / denom

        x = np.arange(len(niche_ids), dtype=float)
        bottom = np.zeros(len(niche_ids), dtype=float)
        for idx, program in enumerate(program_order):
            values = frac[:, idx]
            if not np.any(values > 0.0):
                continue
            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.78,
                color=program_palette.get(program, "#9e9e9e"),
                edgecolor="white",
                linewidth=0.35,
                label=program,
            )
            bottom += values

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction", fontsize=9)
        ax.set_title(
            "Backbone Program Composition" if layer_name == "backbone" else "Member Program Composition",
            fontsize=11,
            pad=8,
        )
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("#bdbdbd")
            spine.set_linewidth(0.7)

    axes[-1].set_xticks(np.arange(len(niche_ids), dtype=float))
    axes[-1].set_xticklabels([str(x) for x in niche_ids], rotation=45, ha="right", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=program_palette[p], edgecolor="white", linewidth=0.35, label=p)
        for p in program_order
        if p in program_palette
    ]
    axes[0].legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        ncol=min(4, max(1, int(math.ceil(len(handles) / 3)))),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.01),
        borderaxespad=0.0,
        title="Programs",
        title_fontsize=9,
    )
    fig.suptitle("NicheGraph Program Composition", fontsize=14, y=1.02)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_edge_graph(
    structures: pd.DataFrame,
    member_domains: pd.DataFrame,
    internal_edges: pd.DataFrame,
    niche_ids: list[str],
    out_path: Path,
) -> None:
    if not niche_ids or member_domains.empty or internal_edges.empty:
        return
    colors = _niche_colors(niche_ids)
    struct_lookup = {str(r["niche_id"]): dict(r) for r in structures.to_dict(orient="records")}
    fig, ax = plt.subplots(figsize=(14.8, 10.0), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.76, top=0.93, bottom=0.05)

    all_nodes = member_domains.drop_duplicates(subset=["domain_key"])
    ax.scatter(
        all_nodes["geo_centroid_x"].to_numpy(dtype=float),
        all_nodes["geo_centroid_y"].to_numpy(dtype=float),
        s=9,
        color="#d0d0d0",
        alpha=0.20,
        linewidths=0,
        zorder=0,
    )

    legend_handles = []
    for niche_id in niche_ids:
        nodes = member_domains.loc[member_domains["niche_id"] == niche_id].copy()
        edges_sub = internal_edges.loc[internal_edges["niche_id"] == niche_id].copy()
        if nodes.empty or edges_sub.empty:
            continue
        color = colors.get(str(niche_id), "#4d4d4d")
        for edge in edges_sub.to_dict(orient="records"):
            line_style = "-" if edge["edge_type"] in {"contact", "mixed"} else (0, (2.2, 1.2))
            ax.plot(
                [edge["x_i"], edge["x_j"]],
                [edge["y_i"], edge["y_j"]],
                color=color,
                linewidth=2.6 if edge["is_backbone_edge"] else 0.95,
                linestyle=line_style,
                alpha=0.88 if edge["is_backbone_edge"] else 0.16,
                zorder=1,
            )
        member_only = nodes.loc[~nodes["is_backbone_member"]].copy()
        if not member_only.empty:
            ax.scatter(
                member_only["geo_centroid_x"],
                member_only["geo_centroid_y"],
                s=34,
                color=color,
                alpha=0.18,
                linewidths=0.35,
                edgecolors="white",
                zorder=2,
            )
        backbone = nodes.loc[nodes["is_backbone_member"]].copy()
        if not backbone.empty:
            ax.scatter(
                backbone["geo_centroid_x"],
                backbone["geo_centroid_y"],
                s=92,
                color=color,
                alpha=0.96,
                linewidths=1.0,
                edgecolors="black",
                zorder=3,
            )
            cx = float(backbone["geo_centroid_x"].median())
            cy = float(backbone["geo_centroid_y"].median())
            struct = struct_lookup.get(str(niche_id), {})
            ax.text(
                cx,
                cy,
                f"{niche_id}\nB={int(struct.get('backbone_node_count', 0))}",
                fontsize=7.5,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.92),
                zorder=4,
            )
            legend_handles.append(
                plt.Line2D([0], [0], color=color, linewidth=2.6, marker="o", markersize=6, markerfacecolor=color, markeredgecolor="black", label=str(niche_id))
            )

    edge_style_handles = [
        plt.Line2D([0], [0], color="#4d4d4d", linewidth=2.0, linestyle="-", label="contact / mixed"),
        plt.Line2D([0], [0], color="#4d4d4d", linewidth=2.0, linestyle=(0, (2.2, 1.2)), label="overlap"),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markeredgecolor="black", markersize=7, linewidth=0, label="backbone node"),
        plt.Line2D([0], [0], marker="o", color="#7f7f7f", markerfacecolor="#7f7f7f", markeredgecolor="white", markersize=6, linewidth=0, alpha=0.45, label="member node"),
    ]
    ax.set_title("NicheGraph Interaction Graph", fontsize=15, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(
        handles=edge_style_handles,
        frameon=False,
        fontsize=8,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.07),
        bbox_transform=fig.transFigure,
        title="Graph Key",
        title_fontsize=9,
    )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=8,
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.93),
            bbox_transform=fig.transFigure,
            title="Niches",
            title_fontsize=9,
        )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_footprint_panels(
    member_spots: pd.DataFrame,
    spot_table: pd.DataFrame,
    niche_ids: list[str],
    out_path: Path,
    panel_cols: int,
) -> None:
    if not niche_ids or member_spots.empty:
        return
    colors = _overlay_niche_colors(member_spots, niche_ids)
    struct_lookup = {
        str(r["niche_id"]): {"backbone_node_count": int(r.get("backbone_node_count", 0)), "member_count": int(r.get("member_count", 0))}
        for r in member_spots.loc[:, ["niche_id"]].drop_duplicates().to_dict(orient="records")
    }
    cols = int(max(1, min(6, panel_cols)))
    rows = int(math.ceil(max(1, len(niche_ids)) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.0 * cols, 3.2 * rows),
        squeeze=False,
        subplot_kw={"box_aspect": 0.88},
        constrained_layout=True,
    )
    for ax in axes.ravel():
        ax.axis("off")
    for idx, niche_id in enumerate(niche_ids):
        ax = axes.ravel()[idx]
        ax.axis("on")
        sub = member_spots.loc[member_spots["niche_id"] == niche_id].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        color = colors.get(str(niche_id), "#4d4d4d")
        x0, x1, y0, y1 = _padded_limits(sub["x"], sub["y"], pad_ratio=0.14)
        local_bg = spot_table.loc[
            (spot_table["x"] >= x0)
            & (spot_table["x"] <= x1)
            & (spot_table["y"] >= y0)
            & (spot_table["y"] <= y1)
        ]
        if not local_bg.empty:
            ax.scatter(
                local_bg["x"],
                local_bg["y"],
                s=5,
                color="#d9d9d9",
                alpha=0.30,
                linewidths=0,
                zorder=0,
            )
        member_only = sub.loc[~sub["is_backbone_member"]].copy()
        if not member_only.empty:
            ax.scatter(
                member_only["x"],
                member_only["y"],
                s=12,
                color=color,
                alpha=0.30,
                linewidths=0.25,
                edgecolors="white",
                zorder=1,
            )
        backbone = sub.loc[sub["is_backbone_member"]].copy()
        if not backbone.empty:
            ax.scatter(
                backbone["x"],
                backbone["y"],
                s=28,
                color=color,
                alpha=0.95,
                linewidths=0.55,
                edgecolors="black",
                zorder=2,
            )
        niche_backbone_n = int(sub["is_backbone_member"].fillna(False).astype(bool).sum())
        niche_member_n = int(sub["spot_id"].astype(str).nunique())
        ax.set_title(f"{niche_id}\nB={niche_backbone_n}  M={niche_member_n}", fontsize=8.5, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_color("#bdbdbd")
            spine.set_linewidth(0.7)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_footprint(
    spot_table: pd.DataFrame,
    member_spots: pd.DataFrame,
    niche_ids: list[str],
    out_path: Path,
) -> None:
    if member_spots.empty or not niche_ids:
        return
    fig, ax = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
    ax.scatter(
        spot_table["x"].to_numpy(dtype=float),
        spot_table["y"].to_numpy(dtype=float),
        s=2.2,
        color="#e0e0e0",
        alpha=0.22,
        linewidths=0,
        zorder=0,
    )
    niche_colors = _overlay_niche_colors(member_spots, niche_ids)
    overlap_counts = (
        member_spots.groupby("spot_id")["niche_id"].nunique().rename("niche_overlap").reset_index()
        if not member_spots.empty else pd.DataFrame(columns=["spot_id", "niche_overlap"])
    )
    plot_spots = member_spots.merge(overlap_counts, on="spot_id", how="left")
    for niche_id in niche_ids:
        sub = plot_spots.loc[plot_spots["niche_id"] == niche_id].copy()
        if sub.empty:
            continue
        member_only = sub.loc[~sub["is_backbone_member"]].copy()
        backbone = sub.loc[sub["is_backbone_member"]].copy()
        color = niche_colors.get(str(niche_id), "#555555")
        if not member_only.empty:
            ax.scatter(
                member_only["x"].to_numpy(dtype=float),
                member_only["y"].to_numpy(dtype=float),
                s=11,
                color=color,
                alpha=0.22,
                linewidths=0.25,
                edgecolors="white",
                zorder=1,
            )
        if not backbone.empty:
            ax.scatter(
                backbone["x"].to_numpy(dtype=float),
                backbone["y"].to_numpy(dtype=float),
                s=19,
                color=color,
                alpha=0.78,
                linewidths=0.35,
                edgecolors="white",
                zorder=2,
                label=str(niche_id),
            )
    overlap_spots = plot_spots.loc[plot_spots["niche_overlap"].fillna(0).astype(int) > 1].drop_duplicates("spot_id")
    if not overlap_spots.empty:
        ax.scatter(
            overlap_spots["x"].to_numpy(dtype=float),
            overlap_spots["y"].to_numpy(dtype=float),
            s=34,
            facecolors="none",
            edgecolors="black",
            linewidths=0.85,
            alpha=0.95,
            zorder=3,
        )
    for niche_id in niche_ids:
        sub = plot_spots.loc[(plot_spots["niche_id"] == niche_id) & (plot_spots["is_backbone_member"].fillna(False))].copy()
        if sub.empty:
            sub = plot_spots.loc[plot_spots["niche_id"] == niche_id].copy()
        if sub.empty:
            continue
        ax.text(
            float(sub["x"].median()),
            float(sub["y"].median()),
            str(niche_id),
            fontsize=7.5,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor=niche_colors.get(str(niche_id), "#555555"),
                linewidth=0.7,
                alpha=0.82,
            ),
            zorder=4,
        )
    ax.set_title("Multi-Niche Footprint Overlay", fontsize=15, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("#6f6f6f")
        spine.set_linewidth(0.9)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_visualization(
    niche_bundle: str | Path,
    outdir: str | Path | None = None,
    explicit_niche_ids: list[str] | None = None,
    with_spot_footprint: bool = True,
    panel_cols: int = DEFAULT_PANEL_COLS,
) -> dict[str, str]:
    niche_bundle_path = Path(niche_bundle)
    out_root = Path(outdir) if outdir is not None else (niche_bundle_path / "plot")
    out_root.mkdir(parents=True, exist_ok=True)

    payload = _load_inputs(niche_bundle=niche_bundle_path)
    structures = payload["structures"]
    niche_ids = _resolve_niche_ids(structures=structures, explicit_ids=explicit_niche_ids)
    member_domains = _prepare_member_domains(payload["membership"], payload["domains"], niche_ids)
    internal_edges = _prepare_internal_edges(structures, member_domains, payload["adjacency_edges"], niche_ids)

    outputs: dict[str, str] = {}
    summary_path = out_root / "niche_summary.json"
    _write_summary(structures, niche_ids, summary_path)
    outputs["summary_json"] = str(summary_path)

    overview_path = out_root / "niche_structure_overview.png"
    _plot_structure_overview(structures, niche_ids, overview_path)
    outputs["structure_overview_png"] = str(overview_path)

    edge_graph_path = out_root / "niche_edge_graph.png"
    _plot_edge_graph(structures, member_domains, internal_edges, niche_ids, edge_graph_path)
    outputs["edge_graph_png"] = str(edge_graph_path)

    program_comp_path = out_root / "niche_program_composition.png"
    _plot_program_composition(member_domains, niche_ids, program_comp_path)
    outputs["program_composition_png"] = str(program_comp_path)

    if with_spot_footprint:
        spot_table = _load_spot_table(payload["domain_bundle_path"])
        member_spots = _prepare_member_spots(
            membership=payload["membership"],
            domain_spot_membership=payload["domain_spot_membership"],
            spot_table=spot_table,
            niche_ids=niche_ids,
        )
        footprint_panel_path = out_root / "niche_footprint_panels.png"
        _plot_footprint_panels(member_spots, spot_table, niche_ids, footprint_panel_path, panel_cols)
        outputs["footprint_panels_png"] = str(footprint_panel_path)

        overlay_path = out_root / "niche_footprint_overlay.png"
        _plot_overlay_footprint(spot_table, member_spots, niche_ids, overlay_path)
        outputs["footprint_overlay_png"] = str(overlay_path)

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize current NicheGraph structure outputs.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--cancer", type=str, default=DEFAULT_CANCER)
    parser.add_argument("--sample-id", type=str, default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--niche-bundle", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--niche-ids", type=str, default="")
    parser.add_argument("--no-spot-footprint", action="store_true")
    parser.add_argument("--panel-cols", type=int, default=DEFAULT_PANEL_COLS)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    niche_bundle = (
        Path(args.niche_bundle).resolve()
        if args.niche_bundle is not None
        else Path(args.work_dir) / str(args.cancer) / "ST" / str(args.sample_id) / "niche_bundle"
    )
    outdir = Path(args.outdir).resolve() if args.outdir is not None else niche_bundle / "plot"
    explicit_niche_ids = [x.strip() for x in str(args.niche_ids).split(",") if x.strip()]
    outputs = run_visualization(
        niche_bundle=niche_bundle,
        outdir=outdir,
        explicit_niche_ids=explicit_niche_ids,
        with_spot_footprint=not bool(args.no_spot_footprint),
        panel_cols=int(np.clip(int(args.panel_cols), 1, 8)),
    )
    print("Visualization generated:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
