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
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


DEFAULT_SAMPLE_ID = "TENX89"
DEFAULT_H5AD = Path("/Users/wuyang/Documents/SC-ST data/COAD/ST/TENX89.h5ad")
DEFAULT_GSS_BUNDLE = Path("/Users/wuyang/Documents/SC-ST data/COAD/ST/TENX89/gss_bundle")
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "raw_vs_gss_local_evidence"
DEFAULT_KS = [2, 3, 4]
DEFAULT_N_HVG = 2000
DEFAULT_N_PCS = 20
DEFAULT_SPOT_SIZE = 18.0
DEFAULT_RANDOM_STATE = 0


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="绘制 raw vs GSS 的空间聚类对照图，用于展示 GSS 局部证据层的空间连续性。",
    )
    parser.add_argument("--project_root", default=str(PROJECT_ROOT), type=str, help="项目根目录。")
    parser.add_argument("--h5ad", default=str(DEFAULT_H5AD), type=str, help="空间转录组 h5ad 文件。")
    parser.add_argument("--gss_bundle", default=str(DEFAULT_GSS_BUNDLE), type=str, help="GSS bundle 目录。")
    parser.add_argument("--sample_id", default=DEFAULT_SAMPLE_ID, type=str, help="样本 ID。")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), type=str, help="输出目录。")
    parser.add_argument("--ks", default=DEFAULT_KS, nargs="+", type=int, help="需要尝试的 KMeans k 值。")
    parser.add_argument("--n_hvg", default=DEFAULT_N_HVG, type=int, help="raw 表示使用的高变基因数量。")
    parser.add_argument("--n_pcs", default=DEFAULT_N_PCS, type=int, help="轻量表示维度；GSS 取前 n_pcs 维，raw 取约 n_pcs*10 个 HVG。")
    parser.add_argument("--spot_size", default=DEFAULT_SPOT_SIZE, type=float, help="散点大小。")
    parser.add_argument("--random_state", default=DEFAULT_RANDOM_STATE, type=int, help="随机种子。")
    return parser


def _load_h5ad_reader() -> Any:
    try:
        import anndata as ad
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("缺少 anndata，无法读取 h5ad。请在包含 anndata 的环境中运行。") from exc
    return ad


def _validate_coords(adata: Any, h5ad_path: Path) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise ValueError(f"空间坐标缺失：{h5ad_path} 中没有 adata.obsm['spatial']。")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != adata.n_obs:
        raise ValueError(f"空间坐标格式错误：期望 shape 为 [spot数, >=2]，实际为 {coords.shape}。")
    if not np.isfinite(coords[:, :2]).all():
        raise ValueError("空间坐标包含 NaN 或 Inf，无法绘图。")
    return coords[:, :2]


def _as_float_matrix(x: Any) -> np.ndarray | sparse.csr_matrix:
    if sparse.issparse(x):
        return x.astype(np.float32).tocsr(copy=True)
    return np.asarray(x, dtype=np.float32)


def _normalize_total_log1p(x: np.ndarray | sparse.csr_matrix, target_sum: float = 1e4) -> np.ndarray | sparse.csr_matrix:
    # raw expression 标准预处理：按 spot 总量归一化后 log1p。
    if sparse.issparse(x):
        row_sum = np.asarray(x.sum(axis=1)).reshape(-1).astype(np.float32)
        scale = np.divide(target_sum, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
        normalized = sparse.diags(scale).dot(x).tocsr()
        normalized.data = np.log1p(normalized.data)
        return normalized
    row_sum = x.sum(axis=1).astype(np.float32)
    scale = np.divide(target_sum, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    return np.log1p(x * scale[:, None]).astype(np.float32)


def _gene_variance(x: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).reshape(-1)
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).reshape(-1)
        return np.maximum(mean_sq - mean * mean, 0.0)
    return np.var(x, axis=0)


def _build_raw_repr(adata: Any, n_hvg: int, n_pcs: int, random_state: int) -> np.ndarray:
    del random_state
    # 轻量 raw 表示：normalize total、log1p、HVG 后直接使用少量 HVG 特征。
    # 这里不再做 PCA/SVD，避免为了一个对照图引入重计算。
    x = _as_float_matrix(adata.X)
    x = _normalize_total_log1p(x)
    feature_cap = max(10, min(int(n_hvg), int(n_pcs) * 10, 300, int(adata.n_vars)))
    n_top_genes = min(feature_cap, int(adata.n_vars))
    if n_top_genes <= 0:
        raise ValueError("raw 表达矩阵没有可用基因，无法构建 raw_repr。")
    variance = _gene_variance(x)
    hvg_idx = np.argsort(variance)[-n_top_genes:]
    if sparse.issparse(x):
        x_hvg = x[:, hvg_idx].toarray()
    else:
        x_hvg = x[:, hvg_idx]
    return np.asarray(x_hvg, dtype=np.float32)


def _trim_dense(matrix: np.ndarray, n_pcs: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"GSS latent 维度错误：期望二维矩阵，实际 shape={matrix.shape}。")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("GSS latent 为空，无法构建 gss_repr。")
    n_dims = max(1, min(int(n_pcs), matrix.shape[1]))
    return matrix[:, :n_dims].astype(np.float32, copy=False)


def _load_gss_from_parquet(parquet_path: Path, spot_ids: list[str], n_pcs: int, random_state: int) -> np.ndarray:
    import pandas as pd

    del random_state
    df = pd.read_parquet(parquet_path, columns=["spot_id", "gene", "gss", "rank_in_spot"])
    required = {"spot_id", "gene", "gss"}
    if not required.issubset(df.columns):
        raise ValueError(f"GSS parquet 缺少必要列：需要 {sorted(required)}，实际为 {list(df.columns)}。")
    if df.empty:
        raise ValueError(f"GSS parquet 为空：{parquet_path}")

    spot_index = {str(spot_id): idx for idx, spot_id in enumerate(spot_ids)}
    spot_codes = df["spot_id"].astype(str).map(spot_index)
    keep = spot_codes.notna()
    if int(keep.sum()) == 0:
        raise ValueError("GSS parquet 的 spot_id 与 h5ad.obs_names 没有交集，无法对齐。")

    rank = pd.to_numeric(df.loc[keep, "rank_in_spot"], errors="coerce")
    rank_keep = rank <= max(1, int(n_pcs))
    filtered = df.loc[keep, ["gene", "gss"]].loc[rank_keep].copy()
    rows = spot_codes.loc[keep].loc[rank_keep].astype(np.int64).to_numpy()
    if filtered.empty:
        raise ValueError("按 rank_in_spot 过滤后 GSS parquet 为空，无法构建 gss_repr。")
    gene_codes, _ = pd.factorize(filtered["gene"].astype(str), sort=True)
    values = pd.to_numeric(filtered["gss"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    matrix = sparse.coo_matrix(
        (values, (rows, gene_codes.astype(np.int64))),
        shape=(len(spot_ids), int(gene_codes.max()) + 1),
        dtype=np.float32,
    ).tocsr()
    return matrix.toarray().astype(np.float32)


def _build_gss_repr(gss_bundle: Path, spot_ids: list[str], n_pcs: int, random_state: int) -> tuple[np.ndarray, str]:
    latent_path = gss_bundle / "latent" / "latent.gat_ae.npy"
    parquet_path = gss_bundle / "gss" / "gss_sparse.parquet"
    if latent_path.exists():
        latent = np.load(latent_path)
        if latent.shape[0] != len(spot_ids):
            raise ValueError(
                f"GSS latent spot 数与 h5ad 不一致：latent={latent.shape[0]}，h5ad={len(spot_ids)}。"
            )
        return _trim_dense(latent, n_pcs=n_pcs), str(latent_path)
    if parquet_path.exists():
        return _load_gss_from_parquet(parquet_path, spot_ids, n_pcs=n_pcs, random_state=random_state), str(parquet_path)
    raise FileNotFoundError(
        "没有找到可用的 GSS 表示：既不存在 "
        f"{latent_path}，也不存在 {parquet_path}。"
    )


def _cluster_labels(repr_matrix: np.ndarray, k: int, random_state: int) -> np.ndarray:
    if k <= 1:
        raise ValueError(f"k 必须大于 1，实际为 {k}。")
    if k > repr_matrix.shape[0]:
        raise ValueError(f"k={k} 大于 spot 数 {repr_matrix.shape[0]}，无法聚类。")
    model = KMeans(n_clusters=int(k), random_state=int(random_state), n_init=20)
    return model.fit_predict(repr_matrix).astype(int)


def _spatial_neighbor_indices(coords: np.ndarray) -> np.ndarray:
    n_spots = coords.shape[0]
    n_neighbors = min(8, n_spots - 1)
    if n_neighbors <= 0:
        raise ValueError("spot 数量不足，无法计算空间连续性。")
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)
    return indices[:, 1:]


def _continuity_score(labels: np.ndarray, neighbor_idx: np.ndarray) -> float:
    same = labels[neighbor_idx] == labels[:, None]
    return float(np.mean(same))


def _cluster_sizes(labels: np.ndarray, k: int) -> dict[str, int]:
    counts = np.bincount(labels.astype(int), minlength=int(k))
    return {str(idx): int(value) for idx, value in enumerate(counts.tolist())}


def _collapse_penalty(labels: np.ndarray, k: int) -> float:
    counts = np.bincount(labels.astype(int), minlength=int(k)).astype(np.float64)
    proportions = counts / max(1.0, counts.sum())
    max_prop = float(proportions.max()) if proportions.size else 1.0
    empty_fraction = float(np.mean(counts == 0)) if counts.size else 1.0
    # 允许自然不均衡，但对明显塌缩和空类做扣分。
    return max(0.0, max_prop - 0.65) + empty_fraction


def _recommend_k(score_rows: dict[str, dict[str, Any]]) -> int:
    best_k = None
    best_score = -np.inf
    for k_text, row in score_rows.items():
        k = int(k_text)
        improvement = float(row["continuity_improvement"])
        penalty = float(row["gss_collapse_penalty"])
        middle_bonus = 0.02 if k == 3 else 0.0
        recommendation_score = improvement - penalty + middle_bonus
        row["recommendation_score"] = float(recommendation_score)
        if recommendation_score > best_score:
            best_score = recommendation_score
            best_k = k
    if best_k is None:
        raise ValueError("没有可用于推荐的 k。")
    return int(best_k)


def _remap_by_overlap(raw_labels: np.ndarray, gss_labels: np.ndarray, k: int) -> np.ndarray:
    overlap = np.zeros((int(k), int(k)), dtype=np.int64)
    for raw_label, gss_label in zip(raw_labels.astype(int), gss_labels.astype(int), strict=True):
        if 0 <= raw_label < k and 0 <= gss_label < k:
            overlap[raw_label, gss_label] += 1
    raw_ind, gss_ind = linear_sum_assignment(-overlap)
    mapping = {int(gss): int(raw) for raw, gss in zip(raw_ind, gss_ind, strict=True)}
    unused = [idx for idx in range(int(k)) if idx not in mapping.values()]
    remapped = np.empty_like(gss_labels, dtype=int)
    for old_label in range(int(k)):
        if old_label not in mapping:
            mapping[old_label] = unused.pop(0) if unused else old_label
        remapped[gss_labels == old_label] = mapping[old_label]
    return remapped


def _plot_pair(
    coords: np.ndarray,
    raw_labels: np.ndarray,
    gss_labels: np.ndarray,
    k: int,
    spot_size: float,
    out_path: Path,
) -> None:
    colors = plt.colormaps.get_cmap("tab10").resampled(max(10, int(k)))
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    pad_x = max((x_max - x_min) * 0.03, 1e-6)
    pad_y = max((y_max - y_min) * 0.03, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), facecolor="white")
    panels = [
        (axes[0], raw_labels, f"Raw spatial observations (k={k})"),
        (axes[1], gss_labels, f"GSS-derived local evidence (k={k})"),
    ]
    for ax, labels, title in panels:
        ax.set_facecolor("white")
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            s=float(spot_size),
            cmap=colors,
            vmin=-0.5,
            vmax=float(k) - 0.5,
            linewidths=0.0,
            alpha=0.96,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_grid(
    coords: np.ndarray,
    label_by_k: dict[int, tuple[np.ndarray, np.ndarray]],
    ks: list[int],
    spot_size: float,
    out_path: Path,
) -> None:
    nrows = len(ks)
    fig, axes = plt.subplots(nrows, 2, figsize=(9.6, 4.2 * nrows), squeeze=False, facecolor="white")
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    pad_x = max((x_max - x_min) * 0.03, 1e-6)
    pad_y = max((y_max - y_min) * 0.03, 1e-6)
    for row, k in enumerate(ks):
        colors = plt.colormaps.get_cmap("tab10").resampled(max(10, int(k)))
        raw_labels, gss_labels = label_by_k[int(k)]
        panels = [
            (axes[row][0], raw_labels, f"Raw spatial observations (k={k})"),
            (axes[row][1], gss_labels, f"GSS-derived local evidence (k={k})"),
        ]
        for ax, labels, title in panels:
            ax.set_facecolor("white")
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=labels,
                s=float(spot_size),
                cmap=colors,
                vmin=-0.5,
                vmax=float(k) - 0.5,
                linewidths=0.0,
                alpha=0.96,
            )
            ax.set_title(title, fontsize=12)
            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = _build_cli().parse_args()
    h5ad_path = Path(args.h5ad).expanduser()
    gss_bundle = Path(args.gss_bundle).expanduser()
    outdir = Path(args.outdir).expanduser()
    ks = sorted({int(k) for k in args.ks})
    if not ks:
        raise ValueError("--ks 至少需要提供一个 k。")
    if not h5ad_path.exists():
        raise FileNotFoundError(f"找不到 h5ad 文件：{h5ad_path}")
    if not gss_bundle.exists():
        raise FileNotFoundError(f"找不到 GSS bundle 目录：{gss_bundle}")
    outdir.mkdir(parents=True, exist_ok=True)

    ad = _load_h5ad_reader()
    adata = ad.read_h5ad(h5ad_path)
    coords = _validate_coords(adata, h5ad_path=h5ad_path)
    spot_ids = [str(x) for x in adata.obs_names.tolist()]

    raw_repr = _build_raw_repr(
        adata=adata,
        n_hvg=int(args.n_hvg),
        n_pcs=int(args.n_pcs),
        random_state=int(args.random_state),
    )
    gss_repr, gss_source = _build_gss_repr(
        gss_bundle=gss_bundle,
        spot_ids=spot_ids,
        n_pcs=int(args.n_pcs),
        random_state=int(args.random_state),
    )
    if raw_repr.shape[0] != coords.shape[0] or gss_repr.shape[0] != coords.shape[0]:
        raise ValueError(
            f"表示矩阵与空间坐标 spot 数不一致：coords={coords.shape[0]}，"
            f"raw={raw_repr.shape[0]}，gss={gss_repr.shape[0]}。"
        )

    neighbor_idx = _spatial_neighbor_indices(coords)
    summary_by_k: dict[str, dict[str, Any]] = {}
    label_by_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    output_files: dict[str, str] = {}

    for k in ks:
        raw_labels = _cluster_labels(raw_repr, k=k, random_state=int(args.random_state))
        gss_labels = _cluster_labels(gss_repr, k=k, random_state=int(args.random_state))
        gss_labels = _remap_by_overlap(raw_labels=raw_labels, gss_labels=gss_labels, k=k)
        raw_score = _continuity_score(raw_labels, neighbor_idx)
        gss_score = _continuity_score(gss_labels, neighbor_idx)
        penalty = _collapse_penalty(gss_labels, k=k)

        out_path = outdir / f"raw_vs_gss_k{k}.png"
        _plot_pair(
            coords=coords,
            raw_labels=raw_labels,
            gss_labels=gss_labels,
            k=k,
            spot_size=float(args.spot_size),
            out_path=out_path,
        )

        label_by_k[int(k)] = (raw_labels, gss_labels)
        output_files[f"raw_vs_gss_k{k}"] = str(out_path)
        summary_by_k[str(k)] = {
            "raw_cluster_size": _cluster_sizes(raw_labels, k=k),
            "gss_cluster_size": _cluster_sizes(gss_labels, k=k),
            "raw_continuity_score": raw_score,
            "gss_continuity_score": gss_score,
            "continuity_improvement": float(gss_score - raw_score),
            "gss_collapse_penalty": penalty,
        }

    recommended_k = _recommend_k(summary_by_k)
    grid_path = outdir / "raw_vs_gss_grid.png"
    _plot_grid(coords=coords, label_by_k=label_by_k, ks=ks, spot_size=float(args.spot_size), out_path=grid_path)
    output_files["raw_vs_gss_grid"] = str(grid_path)

    best_path = outdir / "raw_vs_gss_best.png"
    best_raw, best_gss = label_by_k[recommended_k]
    _plot_pair(
        coords=coords,
        raw_labels=best_raw,
        gss_labels=best_gss,
        k=recommended_k,
        spot_size=float(args.spot_size),
        out_path=best_path,
    )
    output_files["raw_vs_gss_best"] = str(best_path)

    summary = {
        "sample_id": str(args.sample_id),
        "h5ad_path": str(h5ad_path),
        "gss_bundle_path": str(gss_bundle),
        "gss_source": str(gss_source),
        "parameters": {
            "project_root": str(Path(args.project_root).expanduser()),
            "ks": ks,
            "n_hvg": int(args.n_hvg),
            "n_pcs": int(args.n_pcs),
            "spot_size": float(args.spot_size),
            "random_state": int(args.random_state),
        },
        "scores_by_k": summary_by_k,
        "recommended_k": int(recommended_k),
        "output_files": output_files,
    }
    summary_path = outdir / "raw_vs_gss_summary.json"
    _write_json(summary_path, summary)
    print(f"完成：推荐 k={recommended_k}")
    print(f"输出目录：{outdir}")
    print(f"summary：{summary_path}")


if __name__ == "__main__":
    main()
