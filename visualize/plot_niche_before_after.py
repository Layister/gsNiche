from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, Delaunay, QhullError


DEFAULT_WORK_DIR = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "COAD"
DEFAULT_SAMPLE_ID = "TENX89"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "niche_before_after"


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw a focused before/after page for Niche filtering of Domain relation graphs.",
    )
    parser.add_argument("--work_dir", default=str(DEFAULT_WORK_DIR), type=str, help="Root directory of ST data.")
    parser.add_argument("--cancer", default=DEFAULT_CANCER, type=str, help="Cancer type.")
    parser.add_argument("--sample_id", default=DEFAULT_SAMPLE_ID, type=str, help="Sample ID.")
    parser.add_argument("--domain_bundle", default=None, type=str, help="Optional explicit domain_bundle path.")
    parser.add_argument("--niche_bundle", default=None, type=str, help="Optional explicit niche_bundle path.")
    parser.add_argument("--h5ad", default=None, type=str, help="Optional explicit h5ad path for spot coordinates.")
    parser.add_argument("--niche_id", default=None, type=str, help="Optional niche_id. Default: auto select.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), type=str, help="Output directory.")
    parser.add_argument("--background_spot_size", default=3.8, type=float, help="Spot size for full-sample background.")
    parser.add_argument("--candidate_alpha", default=0.095, type=float, help="Light fill alpha for before candidate footprints.")
    parser.add_argument("--selected_alpha", default=0.145, type=float, help="Light fill alpha for after selected niche footprints.")
    parser.add_argument("--outline_width", default=1.35, type=float, help="Domain footprint outline width.")
    parser.add_argument("--backbone_outline_width", default=2.0, type=float, help="Backbone domain outline width in after panel.")
    parser.add_argument("--relation_hint_alpha", default=0.055, type=float, help="Very faint relation hints in before panel.")
    parser.add_argument("--relation_hint_width", default=0.45, type=float, help="Very thin relation hints in before panel.")
    parser.add_argument("--dpi", default=240, type=int, help="Output figure DPI.")
    parser.add_argument("--fig_width", default=11.5, type=float, help="Figure width.")
    parser.add_argument("--fig_height", default=5.4, type=float, help="Figure height.")
    return parser


def _resolve_h5ad_path(args: argparse.Namespace) -> Path:
    if args.h5ad:
        h5ad_path = Path(args.h5ad).expanduser()
    else:
        h5ad_path = Path(args.work_dir).expanduser() / str(args.cancer) / "ST" / f"{args.sample_id}.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"找不到 h5ad 文件，无法读取真实 spot 坐标：{h5ad_path}")
    return h5ad_path


def _resolve_bundle_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    work_dir = Path(args.work_dir).expanduser()
    sample_dir = work_dir / str(args.cancer) / "ST" / str(args.sample_id)
    domain_bundle = Path(args.domain_bundle).expanduser() if args.domain_bundle else sample_dir / "domain_bundle"
    niche_bundle = Path(args.niche_bundle).expanduser() if args.niche_bundle else sample_dir / "niche_bundle"
    if not domain_bundle.exists():
        raise FileNotFoundError(f"找不到 domain_bundle：{domain_bundle}")
    if not niche_bundle.exists():
        raise FileNotFoundError(f"找不到 niche_bundle：{niche_bundle}")
    return domain_bundle, niche_bundle


def _require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} 缺少必要列：{missing}")


def _load_tables(domain_bundle: Path, niche_bundle: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "domains": domain_bundle / "domains.parquet",
        "domain_spot_membership": domain_bundle / "domain_spot_membership.parquet",
        "membership": niche_bundle / "niche_membership.parquet",
        "edges": niche_bundle / "domain_adjacency_edges.parquet",
        "structures": niche_bundle / "niche_structures.parquet",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到 {name} 表：{path}")

    domains = pd.read_parquet(paths["domains"])
    domain_spot_membership = pd.read_parquet(paths["domain_spot_membership"])
    membership = pd.read_parquet(paths["membership"])
    edges = pd.read_parquet(paths["edges"])
    structures = pd.read_parquet(paths["structures"])

    _require_columns(domains, ["domain_key", "domain_id", "program_seed_id", "geo_centroid_x", "geo_centroid_y"], "domains.parquet")
    _require_columns(domain_spot_membership, ["domain_key", "spot_id"], "domain_spot_membership.parquet")
    _require_columns(membership, ["niche_id", "domain_key", "program_id"], "niche_membership.parquet")
    _require_columns(edges, ["domain_key_i", "domain_key_j", "is_strong_edge"], "domain_adjacency_edges.parquet")
    _require_columns(structures, ["niche_id", "member_count", "interaction_confidence"], "niche_structures.parquet")

    domains["domain_key"] = domains["domain_key"].astype(str)
    domains["domain_id"] = domains["domain_id"].astype(str)
    domains["program_seed_id"] = domains["program_seed_id"].astype(str)
    domains["geo_centroid_x"] = pd.to_numeric(domains["geo_centroid_x"], errors="coerce")
    domains["geo_centroid_y"] = pd.to_numeric(domains["geo_centroid_y"], errors="coerce")
    domain_spot_membership["domain_key"] = domain_spot_membership["domain_key"].astype(str)
    domain_spot_membership["spot_id"] = domain_spot_membership["spot_id"].astype(str)

    membership["niche_id"] = membership["niche_id"].astype(str)
    membership["domain_key"] = membership["domain_key"].astype(str)
    membership["program_id"] = membership["program_id"].astype(str)
    if "is_backbone_member" in membership.columns:
        membership["is_backbone_member"] = membership["is_backbone_member"].fillna(False).astype(bool)
    else:
        membership["is_backbone_member"] = False

    edges["domain_key_i"] = edges["domain_key_i"].astype(str)
    edges["domain_key_j"] = edges["domain_key_j"].astype(str)
    edges["is_strong_edge"] = edges["is_strong_edge"].fillna(False).astype(bool)
    if "relation_types" not in edges.columns:
        edges["relation_types"] = ""
    if "edge_strength" not in edges.columns:
        edges["edge_strength"] = 1.0
    edges["edge_strength"] = pd.to_numeric(edges["edge_strength"], errors="coerce").fillna(0.0)

    structures["niche_id"] = structures["niche_id"].astype(str)
    structures["interaction_confidence"] = pd.to_numeric(structures["interaction_confidence"], errors="coerce").fillna(0.0)
    return {
        "domains": domains,
        "domain_spot_membership": domain_spot_membership,
        "membership": membership,
        "edges": edges,
        "structures": structures,
    }


def _load_spot_table(h5ad_path: Path) -> pd.DataFrame:
    try:
        import anndata as ad
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("缺少 anndata，无法读取 h5ad 中的 spot 坐标。") from exc
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if "spatial" not in adata.obsm:
        raise ValueError(f"h5ad 中缺少 adata.obsm['spatial']：{h5ad_path}")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"spatial 坐标格式错误：{coords.shape}")
    out = pd.DataFrame(
        {
            "spot_id": np.asarray(adata.obs_names).astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    return out[np.isfinite(out["x"]) & np.isfinite(out["y"])].reset_index(drop=True)


def _split_key_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x)]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _internal_edges(edges: pd.DataFrame, domain_keys: set[str]) -> pd.DataFrame:
    sub = edges.loc[edges["domain_key_i"].isin(domain_keys) & edges["domain_key_j"].isin(domain_keys)].copy()
    if sub.empty:
        return sub
    sub["edge_pair"] = [
        "||".join(sorted((str(a), str(b)))) for a, b in zip(sub["domain_key_i"], sub["domain_key_j"], strict=True)
    ]
    return sub.drop_duplicates("edge_pair").reset_index(drop=True)


def _candidate_edges(edges: pd.DataFrame, seed_keys: set[str]) -> pd.DataFrame:
    sub = edges.loc[edges["domain_key_i"].isin(seed_keys) | edges["domain_key_j"].isin(seed_keys)].copy()
    if sub.empty:
        return sub
    sub["edge_pair"] = [
        "||".join(sorted((str(a), str(b)))) for a, b in zip(sub["domain_key_i"], sub["domain_key_j"], strict=True)
    ]
    return sub.drop_duplicates("edge_pair").reset_index(drop=True)


def _select_niche(structures: pd.DataFrame, membership: pd.DataFrame, edges: pd.DataFrame, explicit_id: str | None) -> str:
    if explicit_id:
        niche_id = str(explicit_id)
        if niche_id not in set(structures["niche_id"].astype(str)):
            raise ValueError(f"指定的 niche_id 不存在：{niche_id}")
        return niche_id

    rows: list[dict[str, Any]] = []
    for row in structures.to_dict(orient="records"):
        niche_id = str(row["niche_id"])
        keys = set(membership.loc[membership["niche_id"] == niche_id, "domain_key"].astype(str))
        if not keys:
            continue
        edge_sub = _internal_edges(edges, keys)
        before_count = int(edge_sub.shape[0])
        after_count = int(edge_sub["is_strong_edge"].sum()) if not edge_sub.empty else 0
        member_count = len(keys)
        if member_count < 3 or before_count == 0 or after_count == 0:
            continue
        target_size_bonus = 1.0 if 5 <= member_count <= 12 else 0.35
        clutter_bonus = min(before_count / max(after_count, 1), 4.0)
        after_bonus = min(after_count, 10) / 10.0
        confidence = float(row.get("interaction_confidence", 0.0))
        score = confidence * 4.0 + target_size_bonus + clutter_bonus * 0.55 + after_bonus * 0.6
        rows.append(
            {
                "niche_id": niche_id,
                "score": score,
                "member_count": member_count,
                "before_count": before_count,
                "after_count": after_count,
                "interaction_confidence": confidence,
            }
        )
    if not rows:
        raise ValueError("没有找到可用于绘图的 niche：需要同时包含 member domain、before 边和 after strong 边。")
    selected = sorted(rows, key=lambda x: (-float(x["score"]), str(x["niche_id"])))[0]
    return str(selected["niche_id"])


def _prepare_niche_payload(
    niche_id: str,
    domains: pd.DataFrame,
    domain_spot_membership: pd.DataFrame,
    membership: pd.DataFrame,
    edges: pd.DataFrame,
    structures: pd.DataFrame,
    spot_table: pd.DataFrame,
) -> dict[str, Any]:
    mem = membership.loc[membership["niche_id"] == str(niche_id)].copy()
    if mem.empty:
        raise ValueError(f"niche_id={niche_id} 没有 membership 记录。")
    domain_keys = set(mem["domain_key"].astype(str))
    selected_nodes = mem.merge(
        domains.loc[:, ["domain_key", "domain_id", "program_seed_id", "geo_centroid_x", "geo_centroid_y"]],
        on="domain_key",
        how="left",
        validate="many_to_one",
    )
    selected_nodes["program_id"] = selected_nodes["program_id"].replace("", np.nan).fillna(selected_nodes["program_seed_id"]).astype(str)
    selected_nodes["geo_centroid_x"] = pd.to_numeric(selected_nodes["geo_centroid_x"], errors="coerce")
    selected_nodes["geo_centroid_y"] = pd.to_numeric(selected_nodes["geo_centroid_y"], errors="coerce")
    selected_nodes = selected_nodes[np.isfinite(selected_nodes["geo_centroid_x"]) & np.isfinite(selected_nodes["geo_centroid_y"])].copy()
    if selected_nodes.empty:
        raise ValueError(f"niche_id={niche_id} 的 Domain 节点没有可用空间中心坐标。")

    struct_rows = structures.loc[structures["niche_id"] == str(niche_id)]
    struct = struct_rows.iloc[0].to_dict() if not struct_rows.empty else {}
    backbone_keys = set(_split_key_list(struct.get("backbone_node_keys", None)))
    if not backbone_keys:
        backbone_keys = set(selected_nodes.loc[selected_nodes["is_backbone_member"].fillna(False).astype(bool), "domain_key"].astype(str))
    selected_nodes["is_backbone"] = selected_nodes["domain_key"].astype(str).isin(backbone_keys)

    before_edges = _candidate_edges(edges, domain_keys)
    after_edges = _internal_edges(edges, domain_keys).loc[lambda df: df["is_strong_edge"]].copy()
    if backbone_keys:
        after_edges = after_edges.loc[
            after_edges["domain_key_i"].isin(backbone_keys) & after_edges["domain_key_j"].isin(backbone_keys)
        ].copy()
        if after_edges.empty:
            after_edges = _internal_edges(edges, domain_keys).loc[lambda df: df["is_strong_edge"]].copy()

    if before_edges.empty:
        raise ValueError(f"niche_id={niche_id} 没有 before relation edges。")
    if after_edges.empty:
        raise ValueError(f"niche_id={niche_id} 没有 after strong/backbone edges。")

    candidate_keys = set(before_edges["domain_key_i"].astype(str)) | set(before_edges["domain_key_j"].astype(str)) | domain_keys
    candidate_nodes = domains.loc[domains["domain_key"].isin(candidate_keys), [
        "domain_key",
        "domain_id",
        "program_seed_id",
        "geo_centroid_x",
        "geo_centroid_y",
    ]].copy()
    candidate_nodes["program_id"] = candidate_nodes["program_seed_id"].astype(str)
    candidate_nodes["is_selected"] = candidate_nodes["domain_key"].isin(domain_keys)
    candidate_nodes["is_backbone"] = candidate_nodes["domain_key"].isin(backbone_keys)
    selected_nodes["is_selected"] = True

    candidate_footprint = (
        domain_spot_membership.loc[domain_spot_membership["domain_key"].isin(candidate_keys)]
        .merge(spot_table, on="spot_id", how="inner")
        .merge(candidate_nodes.loc[:, ["domain_key", "program_id", "is_selected", "is_backbone"]], on="domain_key", how="left")
    )
    selected_footprint = candidate_footprint.loc[candidate_footprint["domain_key"].isin(domain_keys)].copy()
    if candidate_footprint.empty or selected_footprint.empty:
        raise ValueError(f"niche_id={niche_id} 没有可用 footprint spot 坐标。")

    return {
        "candidate_nodes": candidate_nodes.drop_duplicates("domain_key").reset_index(drop=True),
        "selected_nodes": selected_nodes.drop_duplicates("domain_key").reset_index(drop=True),
        "candidate_footprint": candidate_footprint.reset_index(drop=True),
        "selected_footprint": selected_footprint.reset_index(drop=True),
        "spot_table": spot_table,
        "before_edges": before_edges.reset_index(drop=True),
        "after_edges": after_edges.reset_index(drop=True),
        "structure": struct,
        "backbone_keys": backbone_keys,
    }


def _program_colors(program_ids: list[str]) -> dict[str, Any]:
    uniq = sorted({str(x) for x in program_ids if str(x)})
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(1, len(uniq)))
    return {pid: cmap(i) for i, pid in enumerate(uniq)}


def _padded_limits(points: pd.DataFrame, x_col: str = "x", y_col: str = "y", pad_ratio: float = 0.06) -> tuple[float, float, float, float]:
    x = points[x_col].to_numpy(dtype=float)
    y = points[y_col].to_numpy(dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    pad_x = max(1.0, span_x * float(pad_ratio))
    pad_y = max(1.0, span_y * float(pad_ratio))
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def _edge_coords(edge: pd.Series, node_lookup: dict[str, tuple[float, float]]) -> tuple[float, float, float, float] | None:
    ki = str(edge["domain_key_i"])
    kj = str(edge["domain_key_j"])
    if ki not in node_lookup or kj not in node_lookup:
        return None
    xi, yi = node_lookup[ki]
    xj, yj = node_lookup[kj]
    return xi, yi, xj, yj


def _estimate_spot_spacing(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 1.0
    sample = points
    if sample.shape[0] > 600:
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(sample.shape[0], size=600, replace=False)]
    diffs = sample[:, None, :] - sample[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    dist[dist == 0] = np.nan
    nearest = np.nanmin(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 1.0
    return float(np.median(nearest))


def _alpha_boundary_segments(points: np.ndarray, spacing: float) -> list[tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 4:
        return []
    try:
        tri = Delaunay(points)
    except QhullError:
        return []

    max_edge = max(float(spacing) * 1.85, 1.0)
    edge_counts: dict[tuple[int, int], int] = {}
    for simplex in tri.simplices:
        p = points[simplex]
        d01 = float(np.linalg.norm(p[0] - p[1]))
        d12 = float(np.linalg.norm(p[1] - p[2]))
        d20 = float(np.linalg.norm(p[2] - p[0]))
        if max(d01, d12, d20) > max_edge:
            continue
        for a, b in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])):
            key = tuple(sorted((int(a), int(b))))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for (a, b), count in edge_counts.items():
        if count == 1:
            segments.append((points[a], points[b]))
    return segments


def _draw_sharp_outline(
    ax: plt.Axes,
    points: np.ndarray,
    color: Any,
    spacing: float,
    linewidth: float,
    alpha: float,
    fill_alpha: float,
    zorder: float,
) -> None:
    if points.shape[0] < 3:
        ax.scatter(points[:, 0], points[:, 1], s=7.0, c=[color], alpha=alpha, linewidths=0.0, zorder=zorder)
        return
    segments = _alpha_boundary_segments(points, spacing=spacing)
    if segments:
        try:
            hull = ConvexHull(points)
            outline = points[np.r_[hull.vertices, hull.vertices[0]]]
            if fill_alpha > 0:
                ax.fill(outline[:, 0], outline[:, 1], color=color, alpha=fill_alpha, zorder=zorder)
        except QhullError:
            pass
        for p0, p1 in segments:
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=zorder + 0.4,
            )
        return
    try:
        hull = ConvexHull(points)
        outline = points[np.r_[hull.vertices, hull.vertices[0]]]
        if fill_alpha > 0:
            ax.fill(outline[:, 0], outline[:, 1], color=color, alpha=fill_alpha, zorder=zorder)
        ax.plot(outline[:, 0], outline[:, 1], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder + 0.4)
    except QhullError:
        ax.scatter(points[:, 0], points[:, 1], s=7.0, c=[color], alpha=alpha, linewidths=0.0, zorder=zorder)


def _draw_edges(
    ax: plt.Axes,
    edge_df: pd.DataFrame,
    node_lookup: dict[str, tuple[float, float]],
    color: str,
    alpha: float,
    linewidth: float,
    zorder: int,
    scale_by_strength: bool = False,
) -> None:
    if edge_df.empty:
        return
    strengths = edge_df["edge_strength"].to_numpy(dtype=float) if "edge_strength" in edge_df.columns else np.ones(edge_df.shape[0])
    max_strength = max(float(np.nanmax(strengths)), 1e-8)
    for idx, edge in enumerate(edge_df.to_dict(orient="records")):
        coords = _edge_coords(pd.Series(edge), node_lookup)
        if coords is None:
            continue
        xi, yi, xj, yj = coords
        width = linewidth
        if scale_by_strength:
            width = linewidth * (0.75 + 0.65 * min(float(strengths[idx]) / max_strength, 1.0))
        ax.plot([xi, xj], [yi, yj], color=color, alpha=alpha, linewidth=width, solid_capstyle="round", zorder=zorder)


def _draw_footprints(
    ax: plt.Axes,
    spot_table: pd.DataFrame,
    footprint: pd.DataFrame,
    colors: dict[str, Any],
    limits: tuple[float, float, float, float],
    after_mode: bool,
    background_spot_size: float,
    fill_alpha: float,
    outline_width: float,
    backbone_outline_width: float,
) -> None:
    x_min, x_max, y_min, y_max = limits
    bg = spot_table.loc[
        (spot_table["x"] >= x_min)
        & (spot_table["x"] <= x_max)
        & (spot_table["y"] >= y_min)
        & (spot_table["y"] <= y_max)
    ]
    if not bg.empty:
        ax.scatter(
            bg["x"],
            bg["y"],
            s=float(background_spot_size),
            c="#BDBDBD",
            alpha=0.38,
            linewidths=0.0,
            zorder=0,
        )

    all_points = footprint.loc[:, ["x", "y"]].drop_duplicates().to_numpy(dtype=float)
    spacing = _estimate_spot_spacing(all_points)
    for domain_key, sub in footprint.groupby("domain_key", sort=False):
        program_id = str(sub["program_id"].iloc[0])
        is_backbone = bool(sub["is_backbone"].fillna(False).iloc[0])
        is_selected = bool(sub.get("is_selected", pd.Series([True])).fillna(False).iloc[0])
        color = colors.get(program_id, "#777777")
        alpha = float(fill_alpha)
        if after_mode and not is_backbone:
            alpha *= 0.78
        if not is_selected:
            alpha *= 0.35
        pts = sub.loc[:, ["x", "y"]].drop_duplicates().to_numpy(dtype=float)
        line_alpha = 0.38 if not is_selected else (0.92 if after_mode and is_backbone else 0.76)
        line_width = float(outline_width)
        if after_mode and is_backbone:
            line_width = float(backbone_outline_width)
        if not is_selected:
            line_width *= 0.72
        _draw_sharp_outline(
            ax=ax,
            points=pts,
            color=color,
            spacing=spacing,
            linewidth=line_width,
            alpha=line_alpha,
            fill_alpha=alpha,
            zorder=1.5 if not after_mode or not is_backbone else 2.2,
        )


def _format_ax(ax: plt.Axes, limits: tuple[float, float, float, float], title: str) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_before_after(
    sample_id: str,
    niche_id: str,
    candidate_nodes: pd.DataFrame,
    selected_nodes: pd.DataFrame,
    candidate_footprint: pd.DataFrame,
    selected_footprint: pd.DataFrame,
    spot_table: pd.DataFrame,
    before_edges: pd.DataFrame,
    after_edges: pd.DataFrame,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    colors = _program_colors(candidate_nodes["program_id"].astype(str).tolist())
    node_lookup = {
        str(row["domain_key"]): (float(row["geo_centroid_x"]), float(row["geo_centroid_y"]))
        for row in candidate_nodes.to_dict(orient="records")
    }
    limits = _padded_limits(spot_table, x_col="x", y_col="y", pad_ratio=0.035)
    fig, axes = plt.subplots(1, 2, figsize=(float(args.fig_width), float(args.fig_height)), facecolor="white")

    _draw_footprints(
        axes[0],
        spot_table=spot_table,
        footprint=candidate_footprint,
        colors=colors,
        limits=limits,
        after_mode=False,
        background_spot_size=float(args.background_spot_size),
        fill_alpha=float(args.candidate_alpha),
        outline_width=float(args.outline_width),
        backbone_outline_width=float(args.backbone_outline_width),
    )
    _draw_edges(
        axes[0],
        before_edges,
        node_lookup=node_lookup,
        color="#6F6F6F",
        alpha=float(args.relation_hint_alpha),
        linewidth=float(args.relation_hint_width),
        zorder=4,
        scale_by_strength=False,
    )
    _format_ax(axes[0], limits=limits, title="Before: candidate local relation field")

    _draw_footprints(
        axes[1],
        spot_table=spot_table,
        footprint=selected_footprint,
        colors=colors,
        limits=limits,
        after_mode=True,
        background_spot_size=float(args.background_spot_size),
        fill_alpha=float(args.selected_alpha),
        outline_width=float(args.outline_width),
        backbone_outline_width=float(args.backbone_outline_width),
    )
    _format_ax(axes[1], limits=limits, title="After: selected niche structure")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888", alpha=0.34, markersize=7, label=f"Candidate domains: {candidate_nodes.shape[0]}"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", alpha=0.48, markersize=7, label=f"Selected niche domains: {selected_nodes.shape[0]}"),
        Line2D([0], [0], color="#6F6F6F", linewidth=0.8, alpha=0.18, label=f"Relation hints: {len(before_edges)}"),
        Line2D([0], [0], color="#222222", linewidth=1.6, alpha=0.72, label=f"Backbone members: {int(selected_nodes['is_backbone'].sum())}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{sample_id} / {niche_id}", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_summary(
    out_path: Path,
    sample_id: str,
    domain_bundle: Path,
    niche_bundle: Path,
    niche_id: str,
    candidate_nodes: pd.DataFrame,
    selected_nodes: pd.DataFrame,
    before_edges: pd.DataFrame,
    after_edges: pd.DataFrame,
    candidate_footprint: pd.DataFrame,
    selected_footprint: pd.DataFrame,
    structure: dict[str, Any],
    image_path: Path,
) -> None:
    before_count = int(before_edges.shape[0])
    after_count = int(after_edges.shape[0])
    backbone_node_count = int(selected_nodes["is_backbone"].sum())
    backbone_edge_count = int(structure.get("backbone_edge_count", after_count) or after_count)
    summary = {
        "sample_id": str(sample_id),
        "domain_bundle_path": str(domain_bundle),
        "niche_bundle_path": str(niche_bundle),
        "selected_niche_id": str(niche_id),
        "candidate_domain_count": int(candidate_nodes.shape[0]),
        "selected_domain_count": int(selected_nodes.shape[0]),
        "member_domain_count": int(selected_nodes.shape[0]),
        "before_edge_count": before_count,
        "after_edge_count": after_count,
        "backbone_node_count": backbone_node_count,
        "backbone_edge_count": backbone_edge_count,
        "candidate_footprint_spot_count": int(candidate_footprint["spot_id"].nunique()),
        "selected_footprint_spot_count": int(selected_footprint["spot_id"].nunique()),
        "edge_filtering_ratio": float(after_count / before_count) if before_count else None,
        "program_ids": sorted(selected_nodes["program_id"].astype(str).unique().tolist()),
        "candidate_program_ids": sorted(candidate_nodes["program_id"].astype(str).unique().tolist()),
        "output_files": {"figure": str(image_path), "summary": str(out_path)},
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = _build_cli().parse_args()
    domain_bundle, niche_bundle = _resolve_bundle_paths(args)
    h5ad_path = _resolve_h5ad_path(args)
    tables = _load_tables(domain_bundle=domain_bundle, niche_bundle=niche_bundle)
    spot_table = _load_spot_table(h5ad_path)
    niche_id = _select_niche(
        structures=tables["structures"],
        membership=tables["membership"],
        edges=tables["edges"],
        explicit_id=args.niche_id,
    )
    payload = _prepare_niche_payload(
        niche_id=niche_id,
        domains=tables["domains"],
        domain_spot_membership=tables["domain_spot_membership"],
        membership=tables["membership"],
        edges=tables["edges"],
        structures=tables["structures"],
        spot_table=spot_table,
    )

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    figure_path = outdir / f"niche_before_after.{args.sample_id}.{niche_id}.png"
    summary_path = outdir / "niche_before_after_summary.json"
    _plot_before_after(
        sample_id=str(args.sample_id),
        niche_id=niche_id,
        candidate_nodes=payload["candidate_nodes"],
        selected_nodes=payload["selected_nodes"],
        candidate_footprint=payload["candidate_footprint"],
        selected_footprint=payload["selected_footprint"],
        spot_table=payload["spot_table"],
        before_edges=payload["before_edges"],
        after_edges=payload["after_edges"],
        out_path=figure_path,
        args=args,
    )
    _write_summary(
        out_path=summary_path,
        sample_id=str(args.sample_id),
        domain_bundle=domain_bundle,
        niche_bundle=niche_bundle,
        niche_id=niche_id,
        candidate_nodes=payload["candidate_nodes"],
        selected_nodes=payload["selected_nodes"],
        before_edges=payload["before_edges"],
        after_edges=payload["after_edges"],
        candidate_footprint=payload["candidate_footprint"],
        selected_footprint=payload["selected_footprint"],
        structure=payload["structure"],
        image_path=figure_path,
    )
    print(f"[ok] sample_id={args.sample_id}")
    print(f"[ok] selected_niche_id={niche_id}")
    print(f"[ok] output={figure_path}")
    print(f"[ok] summary={summary_path}")


if __name__ == "__main__":
    main()
