from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
import matplotlib.tri as mtri
import numpy as np
from matplotlib.lines import Line2D

from DomainBuilder.bundle_io import read_json
from DomainBuilder.data_prep import load_domain_visualization_inputs
from DomainBuilder.schema import DomainPipelineConfig

DEFAULT_WORK_DIR = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "COAD"
DEFAULT_SAMPLE_ID = "TENX89"
DEFAULT_PROGRAM_IDS = ["P0004", "P0007", "P0021"]


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay activation boundaries for multiple programs on one spatial plot. "
            "All spots are drawn in light gray; each program is rendered as contour lines only."
        )
    )
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), type=str)
    parser.add_argument("--cancer", default=DEFAULT_CANCER, type=str)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID, type=str)
    parser.add_argument(
        "--program-bundle",
        default=None,
        type=str,
        help="Optional explicit path to program_bundle. Overrides --work-dir/--cancer/--sample-id.",
    )
    parser.add_argument(
        "--program-id",
        default=DEFAULT_PROGRAM_IDS,
        nargs="+",
        type=str,
        help="Program ID(s) to overlay. Supports repeated values or comma-separated chunks.",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="Output path. Default: <program_bundle>/plot/program_activation_overlap.<sample_id>.png",
    )
    parser.add_argument(
        "--activation-source",
        default="effective",
        choices=["effective", "raw"],
        help="Activation values used to compute boundaries.",
    )
    parser.add_argument(
        "--threshold-mode",
        default="qc",
        choices=["qc", "quantile", "absolute"],
        help="How to choose the contour threshold for each program.",
    )
    parser.add_argument(
        "--threshold-quantile",
        default=0.85,
        type=float,
        help="Used only when --threshold-mode quantile.",
    )
    parser.add_argument(
        "--absolute-threshold",
        default=None,
        type=float,
        help="Used when --threshold-mode absolute.",
    )
    parser.add_argument("--spot-size", default=10.0, type=float, help="Background spot size.")
    parser.add_argument("--spot-alpha", default=0.75, type=float, help="Background spot alpha.")
    parser.add_argument(
        "--render-mode",
        default="soft-fill",
        choices=["soft-fill", "outline"],
        help="soft-fill draws low-alpha filled regions plus borders; outline draws borders only.",
    )
    parser.add_argument(
        "--fill-alpha",
        default=0.14,
        type=float,
        help="Fill alpha used in soft-fill mode.",
    )
    parser.add_argument("--line-width", default=2.0, type=float, help="Contour line width.")
    parser.add_argument("--dpi", default=220, type=int)
    parser.add_argument("--fig-width", default=8.5, type=float)
    parser.add_argument("--fig-height", default=8.0, type=float)
    parser.add_argument(
        "--max-edge-quantile",
        default=0.995,
        type=float,
        help="Mask extremely long triangulation edges to avoid artificial contour bridges.",
    )
    parser.add_argument("--seed", default=2024, type=int)
    return parser


def _normalize_program_id_args(raw_values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for chunk in str(value).split(","):
            pid = chunk.strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            out.append(pid)
    return out


def _resolve_program_bundle(args: argparse.Namespace) -> Path:
    if args.program_bundle:
        return Path(args.program_bundle)
    return Path(args.work_dir) / str(args.cancer) / "ST" / str(args.sample_id) / "program_bundle"


def _resolve_output_path(program_bundle: Path, sample_id: str, out_arg: str | None) -> Path:
    if out_arg:
        return Path(out_arg)
    return program_bundle / "plot" / f"program_activation_overlap.{sample_id}.png"


def _load_qc_threshold_map(program_bundle: Path) -> dict[str, float]:
    qc_report_path = program_bundle / "qc_report.json"
    if not qc_report_path.exists():
        return {}
    payload = read_json(qc_report_path)
    activation_summary = payload.get("activation_summary", {})
    raw_map = activation_summary.get("effective_activation_threshold_by_program", {})
    if not isinstance(raw_map, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw_map.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def _resolve_threshold(
    program_id: str,
    values: np.ndarray,
    threshold_mode: str,
    qc_threshold_map: dict[str, float],
    threshold_quantile: float,
    absolute_threshold: float | None,
) -> tuple[float, str]:
    positive = np.asarray(values[np.isfinite(values) & (values > 0)], dtype=np.float32)
    if threshold_mode == "absolute":
        if absolute_threshold is None:
            raise ValueError("--absolute-threshold is required when --threshold-mode absolute.")
        return float(absolute_threshold), "absolute"

    if threshold_mode == "qc":
        qc_value = qc_threshold_map.get(str(program_id), None)
        if qc_value is not None and np.isfinite(qc_value):
            return float(qc_value), "qc"
        raise ValueError(
            f"Missing QC activation threshold for program_id={program_id}. "
            "Use --threshold-mode quantile or --threshold-mode absolute only when you intentionally want a manual override."
        )

    if positive.size == 0:
        return float("nan"), "quantile"

    q = float(np.clip(threshold_quantile, 0.0, 1.0))
    return float(np.quantile(positive, q)), "quantile"


def _build_triangulation(
    coords: np.ndarray,
    max_edge_quantile: float,
) -> mtri.Triangulation:
    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1])
    if triangulation.triangles.size == 0:
        raise ValueError("Unable to build triangulation from spatial coordinates.")

    analyzer = mtri.TriAnalyzer(triangulation)
    mask = analyzer.get_flat_tri_mask(min_circle_ratio=0.01)

    triangles = triangulation.triangles
    p0 = coords[triangles[:, 0]]
    p1 = coords[triangles[:, 1]]
    p2 = coords[triangles[:, 2]]
    edge_len = np.stack(
        [
            np.linalg.norm(p0 - p1, axis=1),
            np.linalg.norm(p1 - p2, axis=1),
            np.linalg.norm(p2 - p0, axis=1),
        ],
        axis=1,
    )
    max_edge = edge_len.max(axis=1)
    finite_max_edge = max_edge[np.isfinite(max_edge)]
    if finite_max_edge.size > 0:
        q = float(np.clip(max_edge_quantile, 0.0, 1.0))
        edge_cutoff = float(np.quantile(finite_max_edge, q))
        mask = np.asarray(mask | (max_edge > edge_cutoff), dtype=bool)

    triangulation.set_mask(mask)
    return triangulation


def main() -> None:
    args = _build_cli().parse_args()
    program_bundle = _resolve_program_bundle(args)
    manifest_path = program_bundle / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing program manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_id = str(manifest.get("sample_id", args.sample_id))
    cfg = DomainPipelineConfig(random_seed=int(args.seed))
    payload = load_domain_visualization_inputs(program_bundle_path=program_bundle, cfg=cfg)

    coords = np.asarray(payload["coords"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates are required and must have shape [n_spots, 2+].")

    if str(args.activation_source) == "raw":
        dense_activation = np.asarray(payload["dense_activation_raw"], dtype=np.float32)
        activation_label = "raw_activation"
    else:
        dense_activation = np.asarray(payload["dense_activation"], dtype=np.float32)
        activation_label = "effective_activation"

    program_ids = np.asarray(payload["program_ids"]).astype(str)
    pid_to_col = {pid: idx for idx, pid in enumerate(program_ids.tolist())}
    selected_program_ids = _normalize_program_id_args(args.program_id)
    missing_program_ids = [pid for pid in selected_program_ids if pid not in pid_to_col]
    if missing_program_ids:
        raise ValueError(f"program_id not found: {missing_program_ids[:10]}")

    triangulation = _build_triangulation(coords=coords, max_edge_quantile=float(args.max_edge_quantile))
    qc_threshold_map = _load_qc_threshold_map(program_bundle=program_bundle)
    color_map = plt.colormaps.get_cmap("tab10").resampled(max(3, len(selected_program_ids)))

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(float(args.fig_width), float(args.fig_height)),
        constrained_layout=True,
    )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=float(args.spot_size),
        c="#d9d9d9",
        alpha=float(np.clip(args.spot_alpha, 0.0, 1.0)),
        linewidths=0,
        zorder=1,
    )

    legend_handles = []
    summary_rows: list[dict[str, float | str | int]] = []
    for idx, program_id in enumerate(selected_program_ids):
        values = np.asarray(dense_activation[:, pid_to_col[program_id]], dtype=np.float32)
        threshold, threshold_source = _resolve_threshold(
            program_id=program_id,
            values=values,
            threshold_mode=str(args.threshold_mode),
            qc_threshold_map=qc_threshold_map,
            threshold_quantile=float(args.threshold_quantile),
            absolute_threshold=args.absolute_threshold,
        )
        if not np.isfinite(threshold):
            print(f"[warn] skip {program_id}: no positive activation values available.")
            continue
        if float(np.nanmax(values)) <= float(threshold):
            print(f"[warn] skip {program_id}: max activation <= threshold ({threshold:.6g}).")
            continue

        color = color_map(idx)
        if str(args.render_mode) == "soft-fill":
            filled = ax.tricontourf(
                triangulation,
                values,
                levels=[threshold, float(np.nanmax(values)) + 1e-8],
                colors=[color],
                alpha=float(np.clip(args.fill_alpha, 0.0, 1.0)),
                zorder=2,
            )
            if len(filled.allsegs[0]) == 0:
                print(f"[warn] skip {program_id}: filled region could not be formed at threshold {threshold:.6g}.")
                continue
        contour = ax.tricontour(
            triangulation,
            values,
            levels=[threshold],
            colors=[color],
            linewidths=float(args.line_width),
            zorder=3,
        )
        if len(contour.allsegs[0]) == 0:
            print(f"[warn] skip {program_id}: contour could not be formed at threshold {threshold:.6g}.")
            continue

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=float(args.line_width),
                alpha=1.0,
                label=f"{program_id} ({threshold_source}, thr={threshold:.3g})",
            )
        )
        summary_rows.append(
            {
                "program_id": program_id,
                "threshold_source": threshold_source,
                "threshold": float(threshold),
                "positive_spot_count": int(np.count_nonzero(values > 0)),
                "boundary_active_spot_count": int(np.count_nonzero(values > threshold)),
            }
        )

    if not summary_rows:
        raise ValueError("No program boundary was drawn. Check program IDs, thresholds, and activation source.")

    ax.set_title(f"{sample_id} | Program activation overlap ({activation_label})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    out_path = _resolve_output_path(program_bundle=program_bundle, sample_id=sample_id, out_arg=args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), facecolor="white")
    plt.close(fig)

    print(f"[ok] sample_id={sample_id}")
    print(f"[ok] program_bundle={program_bundle}")
    print(f"[ok] activation_source={activation_label}")
    print(f"[ok] render_mode={args.render_mode}")
    print(f"[ok] output={out_path}")
    for row in summary_rows:
        print(
            "[ok] "
            f"program_id={row['program_id']} "
            f"threshold_source={row['threshold_source']} "
            f"threshold={row['threshold']:.6g} "
            f"positive_spots={row['positive_spot_count']} "
            f"above_threshold_spots={row['boundary_active_spot_count']}"
        )


if __name__ == "__main__":
    main()
