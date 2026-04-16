from __future__ import annotations

import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gsniche-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gsniche-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from .sample_atlas import DOMAIN_STATIC_FIGURE_NAME, NICHE_STATIC_FIGURE_NAME, PROGRAM_STATIC_FIGURE_NAME
from .schema import CohortReportingConfig


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_program_cross_sample_umap(umap_payload: dict, out_dir: Path) -> list[Path]:
    stem = "program_cross_sample_umap"
    if str(umap_payload.get("status", "skipped")) != "ok":
        return []
    points_df = pd.DataFrame(umap_payload.get("points", []) or [])
    if points_df.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6), dpi=220, sharex=True, sharey=True)
    component_values = sorted(points_df["leading_component_anchor"].astype(str).fillna("").unique().tolist())
    role_values = sorted(points_df["leading_role_anchor"].astype(str).fillna("").unique().tolist())
    sample_values = sorted(points_df["sample_id"].astype(str).fillna("").unique().tolist())
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#FF9DA6"]
    sample_colors = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#8C564B", "#17BECF"]
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    sample_color_map = {value: sample_colors[idx % len(sample_colors)] for idx, value in enumerate(sample_values)}
    component_marker_map = {value: markers[idx % len(markers)] for idx, value in enumerate(component_values)}
    role_marker_map = {value: markers[idx % len(markers)] for idx, value in enumerate(role_values)}

    def _style_umap_axis(ax: plt.Axes, title: str) -> None:
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    component_ax, role_ax = axes
    for component_value in component_values:
        component_df = points_df.loc[points_df["leading_component_anchor"].astype(str) == component_value].copy()
        if component_df.empty:
            continue
        component_ax.scatter(
            component_df["x"].astype(float),
            component_df["y"].astype(float),
            s=56,
            c=[sample_color_map.get(str(v), "#666666") for v in component_df["sample_id"].astype(str)],
            marker=component_marker_map.get(component_value, "o"),
            alpha=0.9,
            linewidths=0.35,
            edgecolors="#303030",
        )
    for role_value in role_values:
        role_df = points_df.loc[points_df["leading_role_anchor"].astype(str) == role_value].copy()
        if role_df.empty:
            continue
        role_ax.scatter(
            role_df["x"].astype(float),
            role_df["y"].astype(float),
            s=56,
            c=[sample_color_map.get(str(v), "#666666") for v in role_df["sample_id"].astype(str)],
            marker=role_marker_map.get(role_value, "o"),
            alpha=0.9,
            linewidths=0.35,
            edgecolors="#303030",
        )
    _style_umap_axis(component_ax, "Sample color + component shape")
    _style_umap_axis(role_ax, "Sample color + role shape")
    fig.suptitle("Program cross-sample UMAP", fontsize=12, y=0.99)
    fig.text(
        0.5,
        0.952,
        "Same layout in both panels: color shows sample; left shape shows component, right shape shows role.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.930,
        "Nearby points indicate higher cross-sample comparability between Program objects; recurrence classes are interpreted within the current cohort scale.",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )
    sample_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=sample_color_map[val], markeredgecolor="#303030", linestyle="none", markersize=6.5, label=val)
        for val in sample_values
    ]
    component_handles = [
        Line2D([0], [0], marker=component_marker_map[val], color="#666666", markerfacecolor="white", markeredgecolor="#666666", linestyle="none", markersize=6.5, label=val or "unanchored")
        for val in component_values
    ]
    role_handles = [
        Line2D([0], [0], marker=role_marker_map[val], color="#666666", markerfacecolor="white", markeredgecolor="#666666", linestyle="none", markersize=6.5, label=val or "unanchored")
        for val in role_values
    ]
    if sample_handles:
        legend0 = role_ax.legend(handles=sample_handles, title="Sample color", frameon=False, fontsize=7.5, title_fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        role_ax.add_artist(legend0)
    if component_handles:
        legend1 = component_ax.legend(handles=component_handles, title="Component shape", frameon=False, fontsize=7.5, title_fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
        component_ax.add_artist(legend1)
    if role_handles:
        role_ax.legend(handles=role_handles, title="Role shape", frameon=False, fontsize=7.5, title_fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 0.55))
    fig.subplots_adjust(top=0.84, right=0.82, left=0.035, bottom=0.04, wspace=0.06)
    return _save_figure(fig, out_dir, stem)


def plot_cross_sample_triage_overview(triage_overview_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=220)
    if triage_overview_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No cross-sample triage summary available", ha="center", va="center")
        return _save_figure(fig, out_dir, "cross_sample_triage_overview")
    level_order = ["program", "domain", "niche", "chain"]
    class_order = ["stable_recurrent", "conditional_variant", "sample_specific"]
    color_map = {
        "stable_recurrent": "#4C78A8",
        "conditional_variant": "#F58518",
        "sample_specific": "#B8B8B8",
    }
    pivot = (
        triage_overview_df.pivot_table(index="result_level", columns="stability_class", values="count", aggfunc="sum", fill_value=0)
        .reindex(index=level_order, columns=class_order, fill_value=0)
    )
    x = np.arange(len(level_order))
    bottom = np.zeros(len(level_order), dtype=float)
    for cls in class_order:
        vals = pivot[cls].to_numpy(dtype=float)
        bars = ax.bar(x, vals, bottom=bottom, color=color_map[cls], width=0.62, label=cls)
        for rect, value, btm in zip(bars, vals, bottom):
            if value >= 2:
                ax.text(rect.get_x() + rect.get_width() / 2.0, btm + value / 2.0, f"{int(value)}", ha="center", va="center", fontsize=8, color="white")
        bottom = bottom + vals
    totals = pivot.sum(axis=1).to_numpy(dtype=float)
    for idx, total in enumerate(totals):
        ax.text(x[idx], total + max(0.15, total * 0.03), str(int(total)), ha="center", va="bottom", fontsize=8.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([lvl.title() for lvl in level_order], fontsize=9)
    ax.set_ylabel("Pattern count")
    ax.set_title("Cross-sample triage overview\nstructure synthesis summary; recurrence classes are interpreted within the current cohort scale", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return _save_figure(fig, out_dir, "cross_sample_triage_overview")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _parse_strip_map(encoded: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in str(encoded or "").split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            values[str(key)] = float(value)
        except Exception:
            values[str(key)] = 0.0
    return values


def _recipe_strip(ax: plt.Axes, encoded: str, axis_order: list[str], x0: float, y: float, width: float, height: float, cmap_name: str) -> None:
    cmap = plt.get_cmap(cmap_name)
    values = _parse_strip_map(encoded)
    cell_w = width / max(1, len(axis_order))
    for idx, axis_id in enumerate(axis_order):
        value = float(values.get(axis_id, 0.0))
        ax.add_patch(
            Rectangle(
                (x0 + idx * cell_w, y),
                cell_w * 0.96,
                height,
                facecolor=cmap(max(0.0, min(1.0, value))),
                edgecolor="#ffffff",
                linewidth=0.25,
            )
        )


def _plot_program_composition_overview(sample_id: str, section_payload: dict, out_dir: Path) -> list[Path]:
    figure_payload = dict(section_payload.get("program_composition_overview", {}) or {})
    records = pd.DataFrame(figure_payload.get("records", []) or [])
    if records.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=220)
        ax.axis("off")
        ax.text(0.5, 0.5, "No Program composition overview available", ha="center", va="center")
        return _save_figure(fig, out_dir, PROGRAM_STATIC_FIGURE_NAME)
    component_order = []
    if "component_strip" in records.columns and not records["component_strip"].empty:
        component_order = [item.split("=", 1)[0] for item in str(records.iloc[0]["component_strip"]).split(";") if "=" in item]
    n_rows = int(records.shape[0])
    row_step = 1.24
    fig, ax = plt.subplots(figsize=(12.9, max(5.3, 0.80 * n_rows + 2.4)), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows * row_step + 1.15)
    ax.axis("off")
    fig.suptitle(f"{sample_id} Program composition overview", y=0.975, fontsize=13.8)
    fig.text(0.5, 0.95, "Read this as a compact program ingredient list: component composition is primary, role is auxiliary.", ha="center", va="top", fontsize=9.4)
    x_id = 0.05
    x_comp = 0.28
    x_role = 0.76
    x_conf = 0.89
    header_y = n_rows * row_step + 0.36
    ax.text(x_id, header_y, "Program", fontsize=11.0, weight="bold")
    ax.text(x_comp, header_y, "Component composition", fontsize=11.0, weight="bold")
    ax.text(x_role, header_y, "Aux role", fontsize=11.0, weight="bold")
    ax.text(x_conf, header_y, "Conf", fontsize=11.0, weight="bold")
    for idx, (_, row) in enumerate(records.iterrows()):
        y = n_rows * row_step - idx * row_step - 0.12
        ax.add_patch(Rectangle((0.02, y - 0.84), 0.96, 0.96, facecolor="white", edgecolor="#e8e8e8", linewidth=0.6))
        ax.text(x_id, y - 0.24, str(row.get("program_id", "")), fontsize=10.2, va="center", ha="left")
        ax.text(x_id, y - 0.52, str(row.get("leading_component_anchor", "")), fontsize=8.2, va="center", ha="left", color="#666666")
        _recipe_strip(ax, str(row.get("component_strip", "")), component_order, x_comp, y - 0.66, 0.34, 0.36, "magma")
        ax.text(x_role, y - 0.38, str(row.get("leading_role_anchor", "")), fontsize=8.8, va="center", ha="left", color="#555555")
        ax.text(x_conf, y - 0.38, f"{_safe_float(row.get('confidence', 0.0)):.2f}", fontsize=8.8, va="center", ha="left", color="#555555")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.04)
    return _save_figure(fig, out_dir, PROGRAM_STATIC_FIGURE_NAME)


def _microbar(ax: plt.Axes, x: float, y: float, width: float, height: float, value: float, color: str, vmax: float) -> None:
    ax.add_patch(Rectangle((x, y), width, height, facecolor="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.3))
    frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, float(value) / float(vmax)))
    ax.add_patch(Rectangle((x, y), width * frac, height, facecolor=color, edgecolor="none"))


def _heat_cells(ax: plt.Axes, values: list[float], labels: list[str], x0: float, y: float, width: float, height: float, cmap_name: str = "Blues") -> None:
    cmap = plt.get_cmap(cmap_name)
    cell_w = width / max(1, len(values))
    for idx, (label, value) in enumerate(zip(labels, values)):
        face = cmap(max(0.0, min(1.0, float(value))))
        ax.add_patch(Rectangle((x0 + idx * cell_w, y), cell_w * 0.96, height, facecolor=face, edgecolor="#ffffff", linewidth=0.4))
        ax.text(x0 + idx * cell_w + cell_w * 0.48, y + height + 0.012, label, ha="center", va="bottom", fontsize=7.1, color="#444444")


def _plot_program_domain_bridge_matrix(sample_id: str, bridge_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    if bridge_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=220)
        ax.axis("off")
        ax.text(0.5, 0.5, "No Program -> Domain deployment matrix available", ha="center", va="center")
        return _save_figure(fig, out_dir, DOMAIN_STATIC_FIGURE_NAME)
    plot_df = bridge_df.copy()
    plot_df["source_program_id"] = plot_df["source_program_id"].astype(str)
    n_rows = int(plot_df.shape[0])
    row_step = 1.28
    fig, ax = plt.subplots(figsize=(17.2, max(6.0, 0.62 * n_rows + 2.5)), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows * row_step + 1.2)
    ax.axis("off")
    fig.suptitle(f"{sample_id} Program-Domain deployment matrix", y=0.972, fontsize=14.0)
    fig.text(0.5, 0.947, "Read as a compact spatial landing overview: how each Program is split, deployed, and what broad shape it tends to form.", ha="center", va="top", fontsize=9.2)
    x_program = 0.03
    x_split = 0.24
    x_deploy = 0.49
    x_morph = 0.73
    header_y = n_rows * row_step + 0.38
    ax.text(x_program, header_y, "Program", fontsize=11.0, weight="bold")
    ax.text(x_split, header_y, "Domain split", fontsize=11.0, weight="bold")
    ax.text(x_deploy, header_y, "Deployment", fontsize=11.0, weight="bold")
    ax.text(x_morph, header_y, "Morphology", fontsize=11.0, weight="bold")
    max_domain_count = max(1.0, float(pd.to_numeric(plot_df["domain_count"], errors="coerce").fillna(0.0).max()))
    max_total_spots = max(1.0, float(pd.to_numeric(plot_df["total_spot_count"], errors="coerce").fillna(0.0).max()))
    max_spot_count_mean = max(1.0, float(pd.to_numeric(plot_df["spot_count_mean"], errors="coerce").fillna(0.0).max()))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y = n_rows * row_step - idx * row_step - 0.12
        ax.add_patch(Rectangle((0.01, y - 0.90), 0.98, 1.02, facecolor="white", edgecolor="#e6e6e6", linewidth=0.6))
        ax.text(x_program, y - 0.28, str(row.get("source_program_id", "")), fontsize=10.2, va="center", ha="left")
        ax.text(x_program, y - 0.56, str(row.get("leading_component_anchor", "")), fontsize=8.2, va="center", ha="left", color="#666666")
        _microbar(ax, x_split, y - 0.18, 0.12, 0.10, _safe_float(row.get("domain_count", 0.0)), "#4C78A8", max_domain_count)
        _microbar(ax, x_split, y - 0.36, 0.12, 0.10, _safe_float(row.get("total_spot_count", 0.0)), "#59A14F", max_total_spots)
        _microbar(ax, x_split, y - 0.54, 0.12, 0.10, _safe_float(row.get("largest_domain_share_by_spots", 0.0)), "#E15759", 1.0)
        ax.text(x_split + 0.13, y - 0.13, "count", fontsize=8.0, va="center")
        ax.text(x_split + 0.13, y - 0.31, "spots", fontsize=8.0, va="center")
        ax.text(x_split + 0.13, y - 0.49, "largest", fontsize=8.0, va="center")
        deploy_vals = [
            _safe_float(row.get("spot_count_mean", 0.0)) / max_spot_count_mean,
            _safe_float(row.get("geo_boundary_ratio_mean", 0.0)),
            min(1.0, _safe_float(row.get("geo_elongation_mean", 0.0)) / 3.0),
            _safe_float(row.get("mixed_neighbor_fraction_mean", 0.0)),
        ]
        _heat_cells(ax, deploy_vals, ["spot", "edge", "elong", "mixed"], x_deploy, y - 0.70, 0.16, 0.38, "Blues")
        morph_vals = [
            _safe_float(row.get("largest_domain_share_by_spots", 0.0)),
            _safe_float(row.get("geo_boundary_ratio_mean", 0.0)),
            _safe_float(row.get("mixed_neighbor_fraction_mean", 0.0)),
        ]
        _heat_cells(ax, morph_vals, ["largest", "edge", "mixed"], x_morph, y - 0.70, 0.14, 0.38, "Purples")
        ax.text(x_morph + 0.15, y - 0.44, str(row.get("program_split_pattern", row.get("morphology_hint", ""))), fontsize=8.2, va="center", ha="left", color="#444444")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.935, bottom=0.03)
    return _save_figure(fig, out_dir, DOMAIN_STATIC_FIGURE_NAME)


def _role_bar_strip(ax: plt.Axes, encoded: str, axis_order: list[str], x0: float, y: float, width: float, height: float) -> None:
    values = _parse_strip_map(encoded)
    raw = np.asarray([max(0.0, float(values.get(axis_id, 0.0))) for axis_id in axis_order], dtype=float)
    total = float(raw.sum())
    norm = raw / total if total > 0 else np.zeros_like(raw)
    palette = {
        "scaffold_like": "#4C78A8",
        "interface_like": "#F58518",
        "node_like": "#54A24B",
        "companion_like": "#B279A2",
    }
    cell_w = width / max(1, len(axis_order))
    for idx, axis_id in enumerate(axis_order):
        base_x = x0 + idx * cell_w
        ax.add_patch(
            Rectangle(
                (base_x, y),
                cell_w * 0.92,
                height,
                facecolor="#f4f4f4",
                edgecolor="#d8d8d8",
                linewidth=0.25,
            )
        )
        bar_h = height * float(norm[idx])
        color = palette.get(axis_id, "#777777")
        ax.add_patch(
            Rectangle(
                (base_x + cell_w * 0.14, y + height - bar_h),
                cell_w * 0.64,
                bar_h,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            base_x + cell_w * 0.46,
            y - 0.018,
            axis_id.split("_", 1)[0][0].upper(),
            ha="center",
            va="top",
            fontsize=5.7,
            color=color,
        )
    if norm.size:
        max_idx = int(np.argmax(norm))
        base_x = x0 + max_idx * cell_w
        ax.add_patch(
            Rectangle(
                (base_x, y),
                cell_w * 0.92,
                height,
                fill=False,
                edgecolor="#333333",
                linewidth=0.8,
            )
        )


def _mini_value_bar(ax: plt.Axes, x: float, y: float, width: float, height: float, value: float, color: str) -> None:
    ax.add_patch(Rectangle((x, y), width, height, facecolor="#f3f3f3", edgecolor="#d0d0d0", linewidth=0.25))
    frac = max(0.0, min(1.0, float(value)))
    ax.add_patch(Rectangle((x, y), width * frac, height, facecolor=color, edgecolor="none"))


def _support_panel(
    ax: plt.Axes,
    x0: float,
    y0: float,
    panel_w: float,
    size_value: float,
    conf_value: float,
    cohes_value: float,
) -> None:
    bar_x = x0
    bar_w = panel_w * 0.72
    label_x = x0 + panel_w * 0.80
    bar_h = 0.055
    y_positions = [y0, y0 - 0.12, y0 - 0.24]
    labels = ["size", "conf", "cohes"]
    values = [size_value, conf_value, cohes_value]
    colors = ["#4C78A8", "#59A14F", "#9C755F"]
    for label, value, color, y in zip(labels, values, colors, y_positions):
        _mini_value_bar(ax, bar_x, y, bar_w, bar_h, value, color)
        ax.text(label_x, y + bar_h / 2.0, label, fontsize=5.8, va="center", ha="left", color="#555555")


def _wrapped_label(value: object, width: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)) or text


def _strip_axis_note(axis_order: list[str], max_items: int = 6) -> str:
    if not axis_order:
        return ""
    if len(axis_order) <= max_items:
        return " | ".join(axis_order)
    head = " | ".join(axis_order[:max_items])
    return f"{head} | ..."


def _plot_sample_level_niche_assembly_matrix(sample_id: str, niche_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    if niche_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=220)
        ax.axis("off")
        ax.text(0.5, 0.5, "No niche assembly matrix available", ha="center", va="center")
        return _save_figure(fig, out_dir, NICHE_STATIC_FIGURE_NAME)
    plot_df = niche_df.copy()
    program_order = [x for x in str(plot_df.iloc[0].get("program_axis_order", "")).split(",") if x]
    role_order = [x for x in str(plot_df.iloc[0].get("role_axis_order", "")).split(",") if x]
    n_rows = int(plot_df.shape[0])
    row_step = 1.32
    fig, ax = plt.subplots(figsize=(18.2, max(7.8, 0.72 * n_rows + 3.1)), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows * row_step + 1.25)
    ax.axis("off")
    fig.suptitle(f"{sample_id} Sample-level niche assembly matrix", y=0.978, fontsize=14.0)
    fig.text(
        0.5,
        0.954,
        "Read as a local-assembly checklist: who participates, how roles divide, and how the main contact structure is organized.",
        ha="center",
        va="top",
        fontsize=9.4,
    )
    x_id = 0.03
    x_program = 0.18
    x_role = 0.42
    x_contact_left = 0.60
    x_contact_right = 0.71
    x_support = 0.85
    header_y = n_rows * row_step + 0.42
    ax.text(x_id, header_y, "Niche", fontsize=11.0, weight="bold")
    ax.text(x_program, header_y, "Member program\ncomposition", fontsize=11.0, weight="bold", va="center")
    ax.text(x_role, header_y, "Member role\ncomposition", fontsize=11.0, weight="bold", va="center")
    ax.text(x_contact_left, header_y, "Main contact\nstructure", fontsize=11.0, weight="bold", va="center")
    ax.text(x_support, header_y, "Support", fontsize=11.0, weight="bold")
    max_member_count = max(1.0, float(pd.to_numeric(plot_df["member_count"], errors="coerce").fillna(0.0).max()))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y = n_rows * row_step - idx * row_step - 0.12
        ax.add_patch(Rectangle((0.01, y - 0.94), 0.98, 1.08, facecolor="white", edgecolor="#e6e6e6", linewidth=0.6))
        ax.text(x_id, y - 0.28, str(row.get("niche_id", "")), fontsize=10.0, va="center", ha="left")
        ax.text(
            x_id,
            y - 0.52,
            f"members={int(float(row.get('member_count', 0) or 0))}  conf={_safe_float(row.get('niche_confidence', 0.0)):.2f}",
            fontsize=7.9,
            va="center",
            ha="left",
            color="#666666",
        )
        _recipe_strip(ax, str(row.get("member_program_composition", "")), program_order, x_program, y - 0.70, 0.20, 0.44, "magma")
        _role_bar_strip(ax, str(row.get("member_role_composition", "")), role_order, x_role, y - 0.68, 0.12, 0.38)
        dominant_pair = _wrapped_label(str(row.get("dominant_contact_pair", "") or "unresolved"), 20)
        secondary_pair = _wrapped_label(str(row.get("secondary_contact_pair", "") or "secondary: none"), 20)
        ax.text(x_contact_left, y - 0.18, dominant_pair, fontsize=8.0, va="top", ha="left", linespacing=1.0)
        ax.text(x_contact_left, y - 0.42, secondary_pair, fontsize=7.0, va="top", ha="left", color="#666666", linespacing=1.0)
        ax.text(
            x_contact_right,
            y - 0.38,
            _wrapped_label(str(row.get("contact_structure_hint", "")), 16),
            fontsize=7.2,
            va="center",
            ha="left",
            bbox=dict(facecolor="#eef3ff", edgecolor="#d7def5", linewidth=0.35, pad=1.2),
            linespacing=1.0,
        )
        _support_panel(
            ax,
            x_support,
            y - 0.22,
            0.11,
            _safe_float(row.get("member_count", 0.0)) / max_member_count,
            _safe_float(row.get("niche_confidence", 0.0)),
            _safe_float(row.get("organizational_cohesion", 0.0)),
        )
    fig.text(
        0.5,
        0.03,
        "Program strip is the semantic recipe proxy. Role strip keeps fixed slots. Contact hint is the primary structure-reading cue.",
        ha="center",
        va="bottom",
        fontsize=8.8,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.05)
    return _save_figure(fig, out_dir, NICHE_STATIC_FIGURE_NAME)


def build_sample_atlas_figures(
    sample_atlas_payloads: dict[str, dict],
    cfg: CohortReportingConfig,
    out_dir: Path,
) -> list[Path]:
    atlas_root = out_dir / cfg.output.sample_atlas_dirname
    paths: list[Path] = []
    for sample_id, payload in sample_atlas_payloads.items():
        sample_dir = atlas_root / str(sample_id)
        sections = dict(payload.get("sections", {}) or {})
        program_section = dict(sections.get("program", {}) or {})
        domain_section = dict(sections.get("domain", {}) or {})
        niche_df = pd.DataFrame(payload.get("niche_assembly_matrix_records", []) or [])
        paths.extend(_plot_program_composition_overview(str(sample_id), program_section, sample_dir))
        bridge_df = pd.DataFrame((domain_section.get("program_domain_deployment_matrix", {}) or {}).get("records", []) or [])
        paths.extend(_plot_program_domain_bridge_matrix(str(sample_id), bridge_df, sample_dir))
        paths.extend(_plot_sample_level_niche_assembly_matrix(str(sample_id), niche_df, sample_dir))
    return paths
