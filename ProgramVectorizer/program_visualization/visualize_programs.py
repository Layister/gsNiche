from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gsniche-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gsniche-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/gsniche-numba-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value 1 overridden to 1 by setting random_state\. Use no seed for parallelism\.",
    category=UserWarning,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform


DEFAULT_WORK_DIR = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "PRAD"
DEFAULT_SAMPLE_ID = "TENX46"

GO_OBO_DIR = "/Users/wuyang/Documents/MyPaper/3/gsNiche/resources/go-basic.obo"
GO_BP_JSON_PATH = "/Users/wuyang/Documents/MyPaper/3/gsNiche/resources/c5.go.bp.v2026.1.Hs.json"
ANNOTATION_METADATA_COLUMNS: tuple[str, ...] = (
    "routing_status",
    "annotation_confidence",
    "annotation_confidence_level",
    "program_confidence",
    "n_significant_terms",
    "program_size",
    "top_genes",
    "summary_text",
)
REQUIRED_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "program_id",
    "routing_status",
    "annotation_source",
    "term_scores",
    "annotation_confidence",
    "annotation_confidence_level",
    "n_significant_terms",
    "program_confidence",
    "top_genes",
)

def _normalize_term_name(term: str, source: str) -> str:
    txt = str(term).strip()
    upper = txt.upper()
    prefixes = {
        "hallmark": ("HALLMARK_",),
        "go_bp": ("GO_BIOLOGICAL_PROCESS_", "GOBP_", "GOBP"),
        "reactome": ("REACTOME_",),
        "kegg": ("KEGG_", "KEGG_MEDICUS_"),
    }.get(str(source).strip().lower(), ())
    for prefix in prefixes:
        if upper.startswith(prefix):
            txt = txt[len(prefix):]
            break
    txt = re.sub(r"[_/]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt or str(term).strip()


def _has_annotation(row: dict) -> bool:
    n_significant_terms = row.get("n_significant_terms", None)
    try:
        if n_significant_terms is not None and int(n_significant_terms) > 0:
            return True
    except Exception:
        pass

    term_scores = row.get("term_scores", {})
    return isinstance(term_scores, dict) and any(float(score) > 0.0 for score in term_scores.values())


def _parse_selected_program_ids(program_ids_arg: str | None, program_ids_file: str | None) -> list[str]:
    selected: list[str] = []
    if program_ids_arg:
        selected.extend([x.strip() for x in str(program_ids_arg).split(",") if x.strip()])
    if program_ids_file:
        path = Path(program_ids_file)
        if not path.exists():
            raise FileNotFoundError(f"Program id file not found: {path}")
        selected.extend([x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()])
    seen: set[str] = set()
    out: list[str] = []
    for pid in selected:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _validate_current_annotation_payload(payload: list[dict[str, Any]], path: Path) -> None:
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid annotation row at index {idx} in {path}: expected dict.")
        missing = [col for col in REQUIRED_ANNOTATION_COLUMNS if col not in row]
        if missing:
            raise ValueError(
                f"Annotation summary is missing current-format fields {missing} at row {idx} in {path}."
            )


def _load_annotation_payload(
    path: Path,
    selected_program_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int, int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid or empty program annotation summary JSON: {path}")
    _validate_current_annotation_payload(payload, path)

    total_programs = len(payload)
    if selected_program_ids:
        selected_set = {str(pid) for pid in selected_program_ids}
        payload = [row for row in payload if str(row.get("program_id", "")) in selected_set]
        found = {str(row.get("program_id", "")) for row in payload}
        missing = [pid for pid in selected_program_ids if pid not in found]
        if missing:
            raise ValueError(f"Selected program_id not found in annotation summary: {missing[:10]}")
    else:
        payload = [row for row in payload if _has_annotation(row)]
    annotated_programs = len(payload)
    if not payload:
        if selected_program_ids:
            raise ValueError(f"No selected programs available for visualization in: {path}")
        raise ValueError(f"No annotated programs found in: {path}")
    annotation_source = ""
    for row in payload:
        annotation_source = annotation_source or str(row.get("annotation_source", "")).strip().lower()
    return payload, total_programs, annotated_programs, annotation_source


def _build_raw_program_vectors(
    payload: list[dict[str, Any]],
    annotation_source: str,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    term_ids = sorted(
        {
            str(term_id)
            for row in payload
            if isinstance(row, dict)
            for term_id in row.get("term_scores", {}).keys()
        }
    )
    if not term_ids:
        raise ValueError("No term_scores found in annotation payload.")

    term_labels: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        for term_id in row.get("term_scores", {}).keys():
            term_id = str(term_id)
            term_labels[term_id] = _normalize_term_name(term_id, annotation_source)

    rows: list[dict] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        compact = row.get("term_scores", {})
        if not isinstance(compact, dict):
            compact = {}
        record = {"program_id": str(row.get("program_id", ""))}
        for col in ANNOTATION_METADATA_COLUMNS:
            record[col] = row.get(col, "")
        for term_id in term_ids:
            record[term_id] = float(compact.get(term_id, 0.0))
        rows.append(record)

    matrix_df = pd.DataFrame(rows)
    matrix_df = matrix_df.loc[matrix_df["program_id"].astype(str).str.len() > 0].copy()
    if matrix_df.empty:
        raise ValueError("No valid program rows found in annotation payload.")

    return matrix_df, term_ids, term_labels


def _resolve_go_obo_path(go_obo: str | None) -> Path:
    candidates: list[Path] = []
    if go_obo:
        candidates.append(Path(go_obo))
    env_obo = os.environ.get("GO_BASIC_OBO", "").strip()
    if env_obo:
        candidates.append(Path(env_obo))
    candidates.extend(
        [
            Path.cwd() / "go-basic.obo",
            Path.cwd() / "ProgramVectorizer" / "program_visualization" / "go-basic.obo",
            Path.home() / "go-basic.obo",
            Path.home() / "Documents" / "go-basic.obo",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "GO BP grouped visualization requires a GO OBO file. "
        "Provide it via --go-obo or set GO_BASIC_OBO."
    )


def _load_go_name_to_id_map(go_obo_path: Path) -> tuple[Any, dict[str, str]]:
    try:
        from goatools.obo_parser import GODag  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "GO BP grouped visualization requires goatools. Install it in the runtime environment."
        ) from exc

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dag = GODag(str(go_obo_path), optional_attrs={"relationship"})
    name_to_ids: dict[str, list[str]] = {}
    for go_id, rec in dag.items():
        if str(getattr(rec, "namespace", "")).strip().lower() != "biological_process":
            continue
        key = _normalize_term_name(str(getattr(rec, "name", go_id)), "go_bp").lower()
        name_to_ids.setdefault(key, []).append(str(go_id))

    unique_name_to_id = {
        key: ids[0]
        for key, ids in name_to_ids.items()
        if len(ids) == 1
    }
    return dag, unique_name_to_id


def _load_go_bp_accession_map(go_bp_json: Path | None) -> dict[str, str]:
    if go_bp_json is None or not go_bp_json.exists():
        return {}
    obj = json.loads(go_bp_json.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    mapping: dict[str, str] = {}
    for term_id, payload in obj.items():
        if not isinstance(payload, dict):
            continue
        exact = str(payload.get("exactSource", "")).strip().upper()
        if re.fullmatch(r"GO:\d{7}", exact):
            mapping[str(term_id)] = exact
    return mapping


def _go_semantic_similarity(go_id_a: str, go_id_b: str, dag: Any) -> float:
    if go_id_a == go_id_b:
        return 1.0
    rec_a = dag.get(go_id_a)
    rec_b = dag.get(go_id_b)
    if rec_a is None or rec_b is None:
        return 0.0
    if str(getattr(rec_a, "namespace", "")).strip().lower() != "biological_process":
        return 0.0
    if str(getattr(rec_b, "namespace", "")).strip().lower() != "biological_process":
        return 0.0
    ancestors_a = set(rec_a.get_all_parents()) | {go_id_a}
    ancestors_b = set(rec_b.get_all_parents()) | {go_id_b}
    union = ancestors_a | ancestors_b
    if not union:
        return 0.0
    return float(len(ancestors_a & ancestors_b) / len(union))


def _build_go_bp_grouped_vectors(
    payload: list[dict[str, Any]],
    go_obo_path: Path,
    go_bp_json_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str], pd.DataFrame]:
    raw_df, raw_term_ids, raw_term_labels = _build_raw_program_vectors(payload, annotation_source="go_bp")
    dag, name_to_go_id = _load_go_name_to_id_map(go_obo_path)
    accession_map = _load_go_bp_accession_map(go_bp_json_path)

    term_to_go_id: dict[str, str] = {}
    for term_id in raw_term_ids:
        go_id = accession_map.get(str(term_id), "")
        if not go_id:
            term_label = raw_term_labels.get(term_id, _normalize_term_name(term_id, "go_bp")).lower()
            go_id = name_to_go_id.get(term_label, "")
        if go_id:
            term_to_go_id[str(term_id)] = str(go_id)

    n_terms = len(raw_term_ids)
    if n_terms == 0:
        raise ValueError("No GO BP terms available for grouped visualization.")

    sim = np.eye(n_terms, dtype=np.float64)
    for i in range(n_terms):
        term_i = str(raw_term_ids[i])
        go_i = term_to_go_id.get(term_i, "")
        for j in range(i + 1, n_terms):
            term_j = str(raw_term_ids[j])
            go_j = term_to_go_id.get(term_j, "")
            if go_i and go_j:
                s = _go_semantic_similarity(go_i, go_j, dag)
            else:
                s = 1.0 if raw_term_labels.get(term_i, term_i) == raw_term_labels.get(term_j, term_j) else 0.0
            sim[i, j] = s
            sim[j, i] = s

    if n_terms == 1:
        cluster_ids = np.asarray([1], dtype=np.int32)
    else:
        dist_vec = squareform(1.0 - np.clip(sim, 0.0, 1.0), checks=False)
        linkage_mat = linkage(dist_vec, method="average")
        cluster_ids = fcluster(linkage_mat, t=0.60, criterion="distance").astype(np.int32)

    grouped_members: dict[int, list[str]] = {}
    for term_id, cluster_id in zip(raw_term_ids, cluster_ids.tolist()):
        grouped_members.setdefault(int(cluster_id), []).append(str(term_id))

    term_weight_sum = raw_df.loc[:, raw_term_ids].sum(axis=0).to_dict()
    group_term_ids: list[str] = []
    group_labels: dict[str, str] = {}
    group_map_rows: list[dict[str, Any]] = []
    group_columns: dict[str, pd.Series] = {}

    for idx, cluster_id in enumerate(sorted(grouped_members), start=1):
        members = grouped_members[int(cluster_id)]
        members_sorted = sorted(
            members,
            key=lambda term_id: (-float(term_weight_sum.get(term_id, 0.0)), raw_term_labels.get(term_id, term_id)),
        )
        representative = str(members_sorted[0])
        group_term_id = f"GOBP_GROUP_{idx:02d}"
        group_label = raw_term_labels.get(representative, representative)
        if len(members_sorted) > 1:
            group_label = f"{group_label} (+{len(members_sorted) - 1})"
        group_columns[group_term_id] = raw_df.loc[:, members].sum(axis=1).astype(float)
        group_term_ids.append(group_term_id)
        group_labels[group_term_id] = str(group_label)
        for member in members_sorted:
            group_map_rows.append(
                {
                    "group_id": group_term_id,
                    "group_label": str(group_label),
                    "term_id": str(member),
                    "term_label": str(raw_term_labels.get(member, member)),
                    "go_id": str(term_to_go_id.get(member, "")),
                    "member_weight_sum": float(term_weight_sum.get(member, 0.0)),
                }
            )

    metadata_cols = [c for c in ["program_id", *ANNOTATION_METADATA_COLUMNS] if c in raw_df.columns]
    grouped_matrix = pd.concat(
        [
            raw_df.loc[:, metadata_cols].copy(),
            pd.DataFrame(group_columns, index=raw_df.index),
        ],
        axis=1,
    )
    group_map_df = pd.DataFrame(group_map_rows)
    return grouped_matrix, group_term_ids, group_labels, group_map_df


def _load_program_vectors(
    path: Path,
    term_mode: str,
    go_obo: str | None,
    go_bp_json: str | None,
    selected_program_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str], int, int, str, str, pd.DataFrame | None]:
    payload, total_programs, annotated_programs, annotation_source = _load_annotation_payload(
        path,
        selected_program_ids=selected_program_ids,
    )
    mode = str(term_mode).strip().lower()
    if mode not in {"auto", "raw", "grouped"}:
        raise ValueError(f"Unsupported term_mode: {term_mode}")

    resolved_mode = mode
    if mode == "auto":
        resolved_mode = "grouped" if annotation_source == "go_bp" else "raw"

    if annotation_source == "go_bp" and resolved_mode == "grouped":
        go_obo_path = _resolve_go_obo_path(go_obo)
        matrix_df, term_ids, term_labels, group_map_df = _build_go_bp_grouped_vectors(
            payload=payload,
            go_obo_path=go_obo_path,
            go_bp_json_path=Path(go_bp_json) if go_bp_json else None,
        )
    else:
        matrix_df, term_ids, term_labels = _build_raw_program_vectors(
            payload=payload,
            annotation_source=annotation_source,
        )
        group_map_df = None

    return matrix_df, term_ids, term_labels, total_programs, annotated_programs, annotation_source, resolved_mode, group_map_df


def _run_umap(
    matrix_df: pd.DataFrame,
    term_ids: list[str],
    seed: int,
    n_neighbors: int | None = None,
    min_dist: float = 0.0,
) -> pd.DataFrame:
    x = matrix_df.loc[:, term_ids].to_numpy(dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("UMAP requires at least two Programs.")

    # Small program sets are easy to over-connect in UMAP.
    # Use a smaller neighborhood and tighter packing so local structure is not
    # flattened into an almost uniform scatter.
    resolved_n_neighbors = (
        min(max(3, int(np.ceil(np.sqrt(x.shape[0])))), max(2, x.shape[0] - 1))
        if n_neighbors is None
        else min(max(2, int(n_neighbors)), max(2, x.shape[0] - 1))
    )
    reducer = umap.UMAP(
        n_neighbors=resolved_n_neighbors,
        min_dist=float(min_dist),
        spread=0.9,
        metric="cosine",
        init="spectral",
        random_state=int(seed),
    )
    embedding = reducer.fit_transform(x)

    metadata_cols = [c for c in ["program_id", *ANNOTATION_METADATA_COLUMNS] if c in matrix_df.columns]
    out = matrix_df.loc[:, metadata_cols].copy()
    out["umap_1"] = embedding[:, 0]
    out["umap_2"] = embedding[:, 1]
    return out


def _plot_umap(embedding_df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    cluster_ids = embedding_df.get("cluster_id", pd.Series(np.ones(embedding_df.shape[0], dtype=int))).astype(int)
    unique_clusters = sorted(pd.unique(cluster_ids).tolist())
    cmap = plt.get_cmap("tab10", max(len(unique_clusters), 1))
    color_by_cluster = {cluster_id: cmap(idx) for idx, cluster_id in enumerate(unique_clusters)}

    for cluster_id in unique_clusters:
        sub = embedding_df.loc[cluster_ids == int(cluster_id)].copy()
        ax.scatter(
            sub["umap_1"].to_numpy(dtype=float),
            sub["umap_2"].to_numpy(dtype=float),
            s=58,
            c=[color_by_cluster[int(cluster_id)]],
            alpha=0.95,
            edgecolors="white",
            linewidths=0.7,
            label=f"Cluster {int(cluster_id)}",
        )

    for row in embedding_df.itertuples(index=False):
        ax.text(
            float(row.umap_1),
            float(row.umap_2),
            str(row.program_id),
            fontsize=7,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2, linewidth=0.5)
    if len(unique_clusters) > 1:
        ax.legend(loc="best", fontsize=7, frameon=False, title="HClust")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _run_hierarchical_clustering(
    matrix_df: pd.DataFrame,
    term_ids: list[str],
    linkage_method: str,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    x = matrix_df.loc[:, term_ids].to_numpy(dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("Hierarchical clustering requires at least two Programs.")

    def _safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
        u_norm = float(np.linalg.norm(u))
        v_norm = float(np.linalg.norm(v))
        if u_norm <= 1e-12 and v_norm <= 1e-12:
            return 0.0
        if u_norm <= 1e-12 or v_norm <= 1e-12:
            return 1.0
        return float(1.0 - np.dot(u, v) / (u_norm * v_norm))

    distance_vec = pdist(x, metric=_safe_cosine)
    linkage_mat = linkage(distance_vec, method=str(linkage_method))
    dendro = dendrogram(linkage_mat, no_plot=True)
    row_order = [int(i) for i in dendro["leaves"]]

    n_clusters = min(max(2, int(np.ceil(np.sqrt(max(matrix_df.shape[0], 1) / 2.0)))), max(2, matrix_df.shape[0]))
    cluster_labels = fcluster(linkage_mat, t=n_clusters, criterion="maxclust")
    cluster_map = {
        str(matrix_df.iloc[idx]["program_id"]): int(cluster_labels[idx])
        for idx in range(matrix_df.shape[0])
    }
    ordered_df = matrix_df.iloc[row_order].reset_index(drop=True)
    ordered_df["cluster_id"] = ordered_df["program_id"].astype(str).map(cluster_map).fillna(0).astype(int)
    ordered_df.insert(1, "cluster_order", np.arange(1, ordered_df.shape[0] + 1, dtype=int))
    return linkage_mat, row_order, ordered_df


def _compute_cosine_distance_matrix(matrix_df: pd.DataFrame, term_ids: list[str]) -> pd.DataFrame:
    x = matrix_df.loc[:, term_ids].to_numpy(dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("Cosine distance heatmap requires at least two Programs.")

    def _safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
        u_norm = float(np.linalg.norm(u))
        v_norm = float(np.linalg.norm(v))
        if u_norm <= 1e-12 and v_norm <= 1e-12:
            return 0.0
        if u_norm <= 1e-12 or v_norm <= 1e-12:
            return 1.0
        return float(1.0 - np.dot(u, v) / (u_norm * v_norm))

    distance_vec = pdist(x, metric=_safe_cosine)
    distance_mat = squareform(distance_vec)
    np.fill_diagonal(distance_mat, 0.0)
    return pd.DataFrame(
        distance_mat,
        index=matrix_df["program_id"].astype(str).tolist(),
        columns=matrix_df["program_id"].astype(str).tolist(),
    )


def _plot_hclust_heatmap(
    matrix_df: pd.DataFrame,
    term_ids: list[str],
    term_labels: dict[str, str],
    linkage_mat: np.ndarray,
    row_order: list[int],
    out_png: Path,
    title: str,
) -> None:
    ordered_df = matrix_df.iloc[row_order].reset_index(drop=True)
    heatmap = ordered_df.loc[:, term_ids].to_numpy(dtype=np.float64)

    fig_width = max(12.0, min(28.0, 6.0 + 0.22 * len(term_ids)))
    fig_height = max(8.0, min(18.0, 4.0 + 0.28 * ordered_df.shape[0]))
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=180)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.6, max(4.5, 0.12 * len(term_ids))],
        left=0.06,
        right=0.985,
        bottom=0.18,
        top=0.90,
        wspace=0.02,
    )

    ax_d = fig.add_subplot(gs[0, 0])
    dendrogram(
        linkage_mat,
        orientation="left",
        ax=ax_d,
        color_threshold=0.0,
        above_threshold_color="#444444",
    )
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.tick_params(axis="y", left=False, labelleft=False)
    for spine in ax_d.spines.values():
        spine.set_visible(False)

    ax_h = fig.add_subplot(gs[0, 1])
    im = ax_h.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis")
    ax_h.set_yticks(np.arange(ordered_df.shape[0]))
    ax_h.set_yticklabels(ordered_df["program_id"].astype(str).tolist(), fontsize=8)
    ax_h.set_xticks(np.arange(len(term_ids)))
    ax_h.set_xticklabels(
        [term_labels.get(term_id, term_id) for term_id in term_ids],
        rotation=55,
        ha="right",
        fontsize=7,
    )
    ax_h.set_xlabel("Annotation Term")
    ax_h.set_ylabel("")

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02)
    cbar.set_label("Annotation score", rotation=90)

    fig.suptitle(title)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_program_distance_heatmap(
    distance_df: pd.DataFrame,
    row_order: list[int],
    out_png: Path,
    title: str,
) -> None:
    ordered_ids = [str(distance_df.index[i]) for i in row_order]
    ordered_distance = distance_df.loc[ordered_ids, ordered_ids]

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=180)
    im = ax.imshow(ordered_distance.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="magma_r")
    ax.set_xticks(np.arange(len(ordered_ids)))
    ax.set_xticklabels(ordered_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ordered_ids)))
    ax.set_yticklabels(ordered_ids, fontsize=8)
    ax.set_xlabel("Program")
    ax.set_ylabel("Program")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Cosine distance", rotation=90)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _run_classical_mds(distance_df: pd.DataFrame) -> pd.DataFrame:
    d = distance_df.to_numpy(dtype=np.float64)
    n = d.shape[0]
    if n < 2:
        raise ValueError("MDS requires at least two Programs.")

    d2 = d ** 2
    j = np.eye(n) - np.ones((n, n), dtype=np.float64) / float(n)
    b = -0.5 * j @ d2 @ j

    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = np.clip(eigvals[:2], a_min=0.0, a_max=None)
    coords = eigvecs[:, :2] * np.sqrt(positive)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant", constant_values=0.0)

    return pd.DataFrame(
        {
            "program_id": distance_df.index.astype(str).tolist(),
            "mds_1": coords[:, 0],
            "mds_2": coords[:, 1],
        }
    )


def _plot_mds(mds_df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    cluster_ids = mds_df.get("cluster_id", pd.Series(np.ones(mds_df.shape[0], dtype=int))).astype(int)
    unique_clusters = sorted(pd.unique(cluster_ids).tolist())
    cmap = plt.get_cmap("tab10", max(len(unique_clusters), 1))
    color_by_cluster = {cluster_id: cmap(idx) for idx, cluster_id in enumerate(unique_clusters)}

    for cluster_id in unique_clusters:
        sub = mds_df.loc[cluster_ids == int(cluster_id)].copy()
        ax.scatter(
            sub["mds_1"].to_numpy(dtype=float),
            sub["mds_2"].to_numpy(dtype=float),
            s=58,
            c=[color_by_cluster[int(cluster_id)]],
            alpha=0.95,
            edgecolors="white",
            linewidths=0.7,
            label=f"Cluster {int(cluster_id)}",
        )

    for row in mds_df.itertuples(index=False):
        ax.text(
            float(row.mds_1),
            float(row.mds_2),
            str(row.program_id),
            fontsize=7,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )

    ax.set_title(title)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.grid(alpha=0.2, linewidth=0.5)
    if len(unique_clusters) > 1:
        ax.legend(loc="best", fontsize=7, frameon=False, title="HClust")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a UMAP from Program term_scores values."
    )
    parser.add_argument("--work-dir", type=str, default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--cancer", type=str, default=DEFAULT_CANCER)
    parser.add_argument("--sample-id", type=str, default=DEFAULT_SAMPLE_ID)
    parser.add_argument(
        "--annotation-json",
        type=str,
        default=None,
        help="Optional override for program_annotation_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory (default: <program_bundle>/plot).",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=None,
        help="Override UMAP n_neighbors (default: adaptive sqrt(n_programs)).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist (default: 0.0 for tighter local clustering).",
    )
    parser.add_argument("--linkage-method", type=str, default="average")
    parser.add_argument(
        "--term-mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "grouped"],
        help="Annotation term mode. Default: auto (Hallmark raw, GO BP grouped).",
    )
    parser.add_argument(
        "--go-obo",
        type=str,
        default=GO_OBO_DIR,
        help="Path to go-basic.obo for GO BP grouped visualization.",
    )
    parser.add_argument(
        "--go-bp-json",
        type=str,
        default=GO_BP_JSON_PATH,
        help="Path to GO BP JSON metadata containing exactSource GO accession mapping.",
    )
    parser.add_argument(
        "--program-ids",
        type=str,
        default=None,
        help="Comma-separated program_id list to visualize. If omitted, uses all annotated programs.",
    )
    parser.add_argument(
        "--program-ids-file",
        type=str,
        default=None,
        help="Optional newline-delimited file of program_id values to visualize.",
    )
    args = parser.parse_args()

    if args.annotation_json:
        annotation_json = Path(args.annotation_json)
        default_out_dir = annotation_json.parent.parent / "plot"
    else:
        program_bundle = Path(args.work_dir) / str(args.cancer) / "ST" / str(args.sample_id) / "program_bundle"
        annotation_json = program_bundle / "program_annotation" / "program_annotation_summary.json"
        default_out_dir = program_bundle / "plot"

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not annotation_json.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {annotation_json}")

    selected_program_ids = _parse_selected_program_ids(args.program_ids, args.program_ids_file)
    matrix_df, term_ids, term_labels, total_programs, annotated_programs, annotation_source, resolved_term_mode, group_map_df = _load_program_vectors(
        annotation_json,
        term_mode=str(args.term_mode),
        go_obo=args.go_obo,
        go_bp_json=args.go_bp_json,
        selected_program_ids=selected_program_ids or None,
    )
    embedding_df = _run_umap(
        matrix_df=matrix_df,
        term_ids=term_ids,
        seed=int(args.seed),
        n_neighbors=args.umap_n_neighbors,
        min_dist=float(args.umap_min_dist),
    )
    linkage_mat, row_order, ordered_df = _run_hierarchical_clustering(
        matrix_df=matrix_df,
        term_ids=term_ids,
        linkage_method=str(args.linkage_method),
    )
    distance_df = _compute_cosine_distance_matrix(matrix_df=matrix_df, term_ids=term_ids)
    metadata_cols = [c for c in ["program_id", *ANNOTATION_METADATA_COLUMNS] if c in ordered_df.columns]
    embedding_df = embedding_df.merge(
        ordered_df.loc[:, ["program_id", "cluster_id"]],
        on="program_id",
        how="left",
    )
    embedding_df["cluster_id"] = embedding_df["cluster_id"].fillna(0).astype(int)
    mds_df = _run_classical_mds(distance_df=distance_df).merge(
        ordered_df.loc[:, ["program_id", "cluster_id", *[c for c in ANNOTATION_METADATA_COLUMNS if c in ordered_df.columns]]],
        on="program_id",
        how="left",
    )
    mds_df["cluster_id"] = mds_df["cluster_id"].fillna(0).astype(int)

    coords_path = out_dir / "program_annotation_umap.tsv"
    fig_path = out_dir / "program_annotation_umap.png"
    hclust_order_path = out_dir / "program_annotation_hclust_order.tsv"
    hclust_fig_path = out_dir / "program_annotation_hclust_heatmap.png"
    distance_tsv_path = out_dir / "program_program_cosine_distance.tsv"
    distance_fig_path = out_dir / "program_program_cosine_distance_heatmap.png"
    mds_tsv_path = out_dir / "program_annotation_mds.tsv"
    mds_fig_path = out_dir / "program_annotation_mds.png"
    grouped_map_path = out_dir / "program_annotation_group_map.tsv"

    embedding_df.to_csv(coords_path, sep="\t", index=False)
    ordered_df.to_csv(hclust_order_path, sep="\t", index=False)
    distance_df.to_csv(distance_tsv_path, sep="\t", index=True)
    mds_df.to_csv(mds_tsv_path, sep="\t", index=False)
    if group_map_df is not None:
        group_map_df.to_csv(grouped_map_path, sep="\t", index=False)
    _plot_umap(
        embedding_df=embedding_df,
        out_png=fig_path,
        title=f"{args.cancer} {args.sample_id} Program UMAP ({annotation_source}, {resolved_term_mode})",
    )
    _plot_hclust_heatmap(
        matrix_df=matrix_df,
        term_ids=term_ids,
        term_labels=term_labels,
        linkage_mat=linkage_mat,
        row_order=row_order,
        out_png=hclust_fig_path,
        title=f"{args.cancer} {args.sample_id} Program Hierarchical Clustering ({resolved_term_mode})",
    )
    _plot_program_distance_heatmap(
        distance_df=distance_df,
        row_order=row_order,
        out_png=distance_fig_path,
        title=f"{args.cancer} {args.sample_id} Program Cosine Distance ({resolved_term_mode})",
    )
    _plot_mds(
        mds_df=mds_df,
        out_png=mds_fig_path,
        title=f"{args.cancer} {args.sample_id} Program MDS ({annotation_source}, {resolved_term_mode})",
    )

    print(f"[ok] input: {annotation_json}")
    print(f"[ok] coordinates: {coords_path}")
    print(f"[ok] figure: {fig_path}")
    print(f"[ok] hclust order: {hclust_order_path}")
    print(f"[ok] hclust heatmap: {hclust_fig_path}")
    print(f"[ok] cosine distance table: {distance_tsv_path}")
    print(f"[ok] cosine distance heatmap: {distance_fig_path}")
    print(f"[ok] mds coordinates: {mds_tsv_path}")
    print(f"[ok] mds figure: {mds_fig_path}")
    if group_map_df is not None:
        print(f"[ok] grouped term map: {grouped_map_path}")
    print(f"[ok] programs shown: {embedding_df.shape[0]}")
    print(f"[ok] programs filtered out (unannotated): {max(0, total_programs - annotated_programs)}")
    print(f"[ok] hclust clusters: {embedding_df['cluster_id'].nunique()}")
    print(f"[ok] annotation terms: {len(term_ids)}")
    print(f"[ok] annotation source: {annotation_source}")
    print(f"[ok] term mode: {resolved_term_mode}")


if __name__ == "__main__":
    main()
