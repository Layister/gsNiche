from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_WORK_DIR = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "IDC"
DEFAULT_SAMPLE_ID = "TENX14"
DEFAULT_PANEL_COLS = 4
MAX_INTERFACES_PER_NICHE = 5
MAX_HEATMAP_TERMS = 30


def _load_sample_id(annotation_dir: Path) -> str:
    meta_path = annotation_dir / "annotation_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_id = str(meta.get("sample_id", "")).strip()
        if sample_id:
            return sample_id
    return annotation_dir.parent.parent.name


def _read_inputs(annotation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], str]:
    interp_path = annotation_dir / "niche_interpretation.tsv"
    topif_path = annotation_dir / "niche_top_interfaces.tsv"
    summary_path = annotation_dir / "niche_summary.json"
    if not interp_path.exists():
        raise FileNotFoundError(f"Missing interpretation table: {interp_path}")
    if not topif_path.exists():
        raise FileNotFoundError(f"Missing top interface table: {topif_path}")
    interpretation = pd.read_csv(interp_path, sep="\t")
    top_interfaces = pd.read_csv(topif_path, sep="\t")
    summary_records = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    sample_id = _load_sample_id(annotation_dir=annotation_dir)
    return interpretation, top_interfaces, summary_records, sample_id


def _resolve_niche_order(interpretation: pd.DataFrame) -> list[str]:
    interp = interpretation.copy()
    interp["niche_id"] = interp["niche_id"].astype(str)
    interp["interaction_confidence"] = pd.to_numeric(interp["interaction_confidence"], errors="coerce").fillna(0.0)
    interp["backbone_edge_count"] = pd.to_numeric(interp["backbone_edge_count"], errors="coerce").fillna(0.0)
    interp["member_count"] = pd.to_numeric(interp["member_count"], errors="coerce").fillna(0.0)
    return (
        interp.sort_values(
            ["interaction_confidence", "backbone_edge_count", "member_count", "niche_id"],
            ascending=[False, False, False, True],
        )["niche_id"]
        .astype(str)
        .tolist()
    )


def _safe_json_dict(raw: object) -> dict[str, float]:
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    txt = str(raw).strip()
    if not txt or txt in {"nan", "None", "null", "{}"}:
        return {}
    try:
        obj = json.loads(txt)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in obj.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def _term_label(term_id: str) -> str:
    txt = str(term_id).strip()
    upper = txt.upper()
    for prefix in (
        "HALLMARK_",
        "GO_BIOLOGICAL_PROCESS_",
        "GOBP_",
        "REACTOME_",
        "KEGG_",
        "KEGG_MEDICUS_",
    ):
        if upper.startswith(prefix):
            txt = txt[len(prefix):]
            break
    return " ".join(txt.replace("_", " ").split()) or "unresolved"


def _display_term_label(term_label: str, max_words: int = 3) -> str:
    words = str(term_label).split()
    if len(words) <= max_words:
        return str(term_label)
    return " ".join(words[:max_words]) + "..."


def _compact_side_label(program_id: str, morphology: str) -> str:
    prog = str(program_id).strip()
    morph = str(morphology).strip()
    if morph and morph.lower() != "nan":
        return f"{prog}  {morph}"
    return prog


def _short_interface_label(interface_label: str) -> str:
    txt = str(interface_label)
    parts = txt.split(":")
    if len(parts) >= 3:
        niche_id = parts[0]
        pair = parts[1]
        edge_type = parts[2]
        return f"{niche_id}  {pair}  {edge_type}"
    return txt


def _prepare_top_interfaces(top_interfaces: pd.DataFrame, niche_order: list[str]) -> pd.DataFrame:
    topif = top_interfaces.copy()
    topif["niche_id"] = topif["niche_id"].astype(str)
    topif["edge_strength"] = pd.to_numeric(topif["edge_strength"], errors="coerce").fillna(0.0)
    topif["edge_reliability"] = pd.to_numeric(topif["edge_reliability"], errors="coerce").fillna(0.0)
    topif["interface_rank_score"] = topif["edge_strength"] * topif["edge_reliability"]
    topif["left_program_term_scores"] = topif["left_program_term_scores_json"].map(_safe_json_dict)
    topif["right_program_term_scores"] = topif["right_program_term_scores_json"].map(_safe_json_dict)
    topif["left_term_count"] = topif["left_program_term_scores"].map(len)
    topif["right_term_count"] = topif["right_program_term_scores"].map(len)
    topif["interface_key"] = (
        topif["domain_i"].astype(str)
        + "||"
        + topif["domain_j"].astype(str)
        + "||"
        + topif["edge_type"].astype(str)
    )
    selected: list[pd.DataFrame] = []
    for niche_id in niche_order:
        sub = topif.loc[topif["niche_id"] == niche_id].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["edge_reliability", "edge_strength", "left_term_count", "right_term_count", "interface_key"],
            ascending=[False, False, False, False, True],
        ).head(MAX_INTERFACES_PER_NICHE)
        selected.append(sub)
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def _plot_structure_overview(
    interpretation: pd.DataFrame,
    niche_order: list[str],
    sample_id: str,
    out_path: Path,
) -> None:
    if not niche_order:
        return
    cols = [
        "interaction_confidence",
        "contact_fraction",
        "overlap_fraction",
        "core_profiled_fraction",
        "context_unresolved_fraction",
    ]
    label_map = {
        "interaction_confidence": "confidence",
        "contact_fraction": "contact",
        "overlap_fraction": "overlap",
        "core_profiled_fraction": "core profiled",
        "context_unresolved_fraction": "context unresolved",
    }
    interp = interpretation.copy()
    interp["niche_id"] = interp["niche_id"].astype(str)
    for col in cols + ["backbone_edge_count", "member_count"]:
        interp[col] = pd.to_numeric(interp[col], errors="coerce").fillna(0.0)
    interp = interp.set_index("niche_id").reindex(niche_order)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(14.5, max(6.2, 0.42 * len(niche_order) + 2.0)),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    fig.subplots_adjust(left=0.07, right=0.88, top=0.92, bottom=0.08, wspace=0.20)

    y = np.arange(len(niche_order), dtype=float)
    metrics = interp.loc[:, cols].to_numpy(dtype=float)
    n_cols = len(cols)
    for j, col in enumerate(cols):
        ax0.scatter(
            np.full(len(niche_order), j, dtype=float),
            y,
            s=90 + 280 * np.clip(metrics[:, j], 0.0, 1.0),
            c=metrics[:, j],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            edgecolors="black",
            linewidths=0.35,
        )
    ax0.set_xticks(np.arange(n_cols))
    ax0.set_xticklabels([label_map[c] for c in cols], rotation=35, ha="right")
    ax0.set_yticks(y)
    ax0.set_yticklabels(niche_order)
    ax0.invert_yaxis()
    ax0.set_title(f"{sample_id}: Annotation Structure Overview")
    ax0.grid(axis="x", alpha=0.14, linewidth=0.6)
    ax0.grid(axis="y", alpha=0.08, linewidth=0.5)

    edge_counts = interp["backbone_edge_count"].to_numpy(dtype=float)
    member_counts = interp["member_count"].to_numpy(dtype=float)
    ax1.barh(y - 0.18, edge_counts, height=0.34, color="#4C78A8", label="backbone edges")
    ax1.barh(y + 0.18, member_counts, height=0.34, color="#F58518", label="members")
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlabel("Count")
    ax1.set_title("Backbone vs Member Scale")
    ax1.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax1.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_interface_panels(
    interfaces: pd.DataFrame,
    niche_order: list[str],
    sample_id: str,
    out_path: Path,
    panel_cols: int,
) -> None:
    if interfaces.empty or not niche_order:
        return
    niches_with_data = [nid for nid in niche_order if nid in set(interfaces["niche_id"].astype(str))]
    if not niches_with_data:
        return
    cols = int(max(1, min(4, panel_cols)))
    rows = int(math.ceil(len(niches_with_data) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 3.1 * rows), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")

    edge_color = {"contact": "#D55E00", "overlap": "#0072B2", "mixed": "#6A3D9A"}
    for idx, niche_id in enumerate(niches_with_data):
        ax = axes.ravel()[idx]
        ax.axis("on")
        sub = interfaces.loc[interfaces["niche_id"] == niche_id].copy().reset_index(drop=True)
        n = len(sub)
        y = np.arange(n, dtype=float)[::-1]
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.5, max(0.55, n - 0.1))
        for i, row in sub.iterrows():
            yy = y[i]
            color = edge_color.get(str(row["edge_type"]), "#5f5f5f")
            strength = float(row["edge_strength"])
            reliability = float(row["edge_reliability"])
            line_w = 1.0 + 4.0 * np.clip(reliability, 0.0, 1.0)
            if i % 2 == 0:
                ax.axhspan(yy - 0.35, yy + 0.35, color="#f7f7f7", zorder=0)
            ax.plot([0.24, 0.76], [yy, yy], color=color, linewidth=line_w, alpha=0.92, solid_capstyle="round", zorder=1)
            ax.scatter([0.24, 0.76], [yy, yy], s=22 + 60 * np.clip(strength, 0.0, 1.0), color=color, edgecolors="black", linewidths=0.35, zorder=3)
            ax.text(0.02, yy, _compact_side_label(row["program_i"], row["left_morphology"]), ha="left", va="center", fontsize=6.8)
            ax.text(0.98, yy, _compact_side_label(row["program_j"], row["right_morphology"]), ha="right", va="center", fontsize=6.8)
            ax.text(0.50, yy + 0.17, f'{row["edge_type"]}  r={reliability:.2f}', ha="center", va="bottom", fontsize=6.6, color="#4a4a4a")
        mean_rel = float(sub["edge_reliability"].mean()) if not sub.empty else 0.0
        ax.set_title(f"{niche_id}  (n={n}, mean r={mean_rel:.2f})", fontsize=8.7, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c8c8c8")
            spine.set_linewidth(0.7)

    fig.suptitle(f"{sample_id}: Interface Evidence Panels", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _prepare_interface_term_matrix(interfaces: pd.DataFrame, niche_order: list[str]) -> pd.DataFrame:
    if interfaces.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for row in interfaces.itertuples(index=False):
        niche_id = str(row.niche_id)
        interface_label = f"{niche_id}:{row.program_i}-{row.program_j}:{row.edge_type}"
        left_scores = row.left_program_term_scores if isinstance(row.left_program_term_scores, dict) else {}
        right_scores = row.right_program_term_scores if isinstance(row.right_program_term_scores, dict) else {}
        term_keys = set(left_scores.keys()) | set(right_scores.keys())
        for term_id in term_keys:
            score = float(left_scores.get(term_id, 0.0)) + float(right_scores.get(term_id, 0.0))
            rows.append(
                {
                    "niche_id": niche_id,
                    "interface_label": interface_label,
                    "term_id": str(term_id),
                    "term_label": _term_label(str(term_id)),
                    "score": score,
                }
            )
    if not rows:
        return pd.DataFrame()
    long_df = pd.DataFrame(rows)
    term_order = (
        long_df.groupby(["term_id", "term_label"], as_index=False)["score"]
        .sum()
        .sort_values(["score", "term_label"], ascending=[False, True])
        .head(MAX_HEATMAP_TERMS)
    )
    keep = set(term_order["term_id"].astype(str))
    use = long_df.loc[long_df["term_id"].astype(str).isin(keep)].copy()
    interface_order: list[str] = []
    seen: set[str] = set()
    for niche_id in niche_order:
        labels = (
            use.loc[use["niche_id"].astype(str) == str(niche_id), "interface_label"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        for label in labels:
            if label not in seen:
                seen.add(label)
                interface_order.append(label)
    mat = (
        use.pivot_table(index="interface_label", columns="term_label", values="score", aggfunc="sum", fill_value=0.0)
        .reindex(interface_order)
        .fillna(0.0)
    )
    return mat


def _plot_term_evidence_heatmap(
    term_matrix: pd.DataFrame,
    sample_id: str,
    out_path: Path,
) -> None:
    if term_matrix.empty:
        return
    fig_w = max(12.0, 0.28 * term_matrix.shape[0] + 4.5)
    fig_h = max(6.5, 0.34 * term_matrix.shape[1] + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.93, top=0.88, bottom=0.28)
    data = np.log1p(term_matrix.to_numpy(dtype=float)).T
    im = ax.imshow(data, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(np.arange(term_matrix.shape[0]))
    ax.set_xticklabels([_short_interface_label(x) for x in term_matrix.index.tolist()], rotation=45, ha="right", fontsize=6.6)
    ax.set_yticks(np.arange(term_matrix.shape[1]))
    ax.set_yticklabels([_display_term_label(c) for c in term_matrix.columns.tolist()], fontsize=8)
    ax.set_title(
        f"{sample_id}: Interface-Level Term Evidence\nColumns are interfaces; brighter cells indicate stronger combined term evidence across both interface ends.",
        fontsize=12,
    )
    ax.set_xticks(np.arange(-0.5, term_matrix.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, term_matrix.shape[1], 1), minor=True)
    ax.grid(which="minor", color=(1, 1, 1, 0.08), linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    col_labels = term_matrix.index.tolist()
    for idx in range(1, len(col_labels)):
        prev_niche = str(col_labels[idx - 1]).split(":")[0]
        cur_niche = str(col_labels[idx]).split(":")[0]
        if prev_niche != cur_niche:
            ax.axvline(idx - 0.5, color="white", linewidth=1.4, alpha=0.92)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=0, pad=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log1p(term score)")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_visualization(annotation_dir: Path, outdir: Path | None = None, panel_cols: int = DEFAULT_PANEL_COLS) -> dict[str, str]:
    annotation_dir = Path(annotation_dir).expanduser().resolve()
    if outdir is None:
        outdir = annotation_dir / "annotation_visualization"
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    interpretation, top_interfaces, summary_records, sample_id = _read_inputs(annotation_dir=annotation_dir)
    niche_order = _resolve_niche_order(interpretation)
    interfaces = _prepare_top_interfaces(top_interfaces, niche_order)
    term_matrix = _prepare_interface_term_matrix(interfaces, niche_order)

    structure_overview_path = outdir / "annotation_structure_overview.png"
    interface_panels_path = outdir / "annotation_interface_panels.png"
    term_heatmap_path = outdir / "annotation_term_evidence_heatmap.png"
    summary_path = outdir / "annotation_visualization_summary.json"

    _plot_structure_overview(interpretation, niche_order, sample_id, structure_overview_path)
    _plot_interface_panels(interfaces, niche_order, sample_id, interface_panels_path, panel_cols=panel_cols)
    _plot_term_evidence_heatmap(term_matrix, sample_id, term_heatmap_path)

    summary = {
        "sample_id": sample_id,
        "niche_count": int(len(niche_order)),
        "visualized_interface_count": int(interfaces.shape[0]),
        "term_heatmap_row_count": int(term_matrix.shape[0]),
        "term_heatmap_col_count": int(term_matrix.shape[1]),
        "source_files": {
            "interpretation_tsv": str(annotation_dir / "niche_interpretation.tsv"),
            "top_interfaces_tsv": str(annotation_dir / "niche_top_interfaces.tsv"),
            "summary_json": str(annotation_dir / "niche_summary.json"),
        },
        "outputs": {
            "structure_overview_png": str(structure_overview_path),
            "interface_panels_png": str(interface_panels_path),
            "term_evidence_heatmap_png": str(term_heatmap_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = dict(summary["outputs"])
    outputs["summary_json"] = str(summary_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize edge-first niche annotation outputs.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--cancer", type=str, default=DEFAULT_CANCER)
    parser.add_argument("--sample-id", type=str, default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--annotation-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--panel-cols", type=int, default=DEFAULT_PANEL_COLS)
    args = parser.parse_args()

    outputs = run_visualization(
        annotation_dir=(
            Path(args.annotation_dir)
            if args.annotation_dir
            else Path(args.work_dir) / str(args.cancer) / "ST" / str(args.sample_id) / "niche_bundle" / "niche_annotation"
        ),
        outdir=Path(args.outdir).expanduser() if args.outdir else None,
        panel_cols=int(np.clip(int(args.panel_cols), 1, 4)),
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
