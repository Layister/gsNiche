from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

from .bundle_io import read_json, write_json, write_tsv
from .schema import AtlasConfig


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_json_dict(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        source = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return {}
        try:
            source = json.loads(text)
        except Exception:
            return {}
    out: dict[str, float] = {}
    for key, raw_val in source.items():
        score = _safe_float(raw_val, default=float("nan"))
        if not math.isfinite(score) or score <= 0.0:
            continue
        out[str(key)] = float(score)
    return out


def _canonical_pair(left: object, right: object) -> str:
    a = str(left).strip()
    b = str(right).strip()
    if not a:
        a = "unresolved"
    if not b:
        b = "unresolved"
    if b < a:
        a, b = b, a
    return f"{a}<->{b}"


def _cohort_sample_id(niche_bundle_path: Path) -> str:
    manifest_path = niche_bundle_path / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        sample_id = str(manifest.get("sample_id", "")).strip()
        if sample_id:
            return sample_id
    return niche_bundle_path.parent.name


def _interface_signature(program_i: object, program_j: object, edge_type: object) -> str:
    pair = _canonical_pair(program_i, program_j)
    relation = str(edge_type).strip().lower() or "unknown"
    return f"{pair}|{relation}"


def _merge_score_dicts(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for source in (left, right):
        for term, score in source.items():
            out[term] = float(out.get(term, 0.0)) + float(score)
    return out


def _normalize_by_group_sum(values: pd.Series, groups: pd.Series) -> np.ndarray:
    group_sum = values.groupby(groups).transform("sum").to_numpy(dtype=np.float64)
    denom = np.where(group_sum > 0.0, group_sum, 1.0)
    return values.to_numpy(dtype=np.float64) / denom


def _load_sample_tables(
    niche_bundle_path: Path,
    cfg: AtlasConfig,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    sample_id = _cohort_sample_id(niche_bundle_path)
    interpretation_path = niche_bundle_path / cfg.input.interpretation_relpath
    top_interfaces_path = niche_bundle_path / cfg.input.top_interfaces_relpath

    status = {
        "sample_id": sample_id,
        "niche_bundle_path": str(niche_bundle_path),
        "interpretation_path": str(interpretation_path),
        "top_interfaces_path": str(top_interfaces_path),
        "loaded": False,
        "reason": "",
    }

    if not niche_bundle_path.exists():
        status["reason"] = "missing_niche_bundle"
        if bool(cfg.input.allow_missing_annotation):
            return None, None, status
        raise FileNotFoundError(f"Missing niche_bundle for sample={sample_id}: {niche_bundle_path}")

    if not interpretation_path.exists() or not top_interfaces_path.exists():
        status["reason"] = "missing_niche_annotation"
        if bool(cfg.input.allow_missing_annotation):
            return None, None, status
        missing = [str(p) for p in [interpretation_path, top_interfaces_path] if not p.exists()]
        raise FileNotFoundError(f"Missing niche annotation inputs for sample={sample_id}: {missing}")

    interpretation = pd.read_csv(interpretation_path, sep="\t")
    top_interfaces = pd.read_csv(top_interfaces_path, sep="\t")
    if interpretation.empty or top_interfaces.empty:
        status["reason"] = "empty_niche_annotation"
        if bool(cfg.input.allow_missing_annotation):
            return None, None, status
        raise ValueError(f"Empty niche annotation inputs for sample={sample_id}")

    interpretation["sample_id"] = sample_id
    interpretation["niche_id"] = interpretation["niche_id"].astype(str)
    interpretation["niche_key"] = interpretation["sample_id"].astype(str) + "::" + interpretation["niche_id"].astype(str)

    for col in (
        "member_count",
        "backbone_node_count",
        "backbone_edge_count",
        "interaction_confidence",
        "contact_fraction",
        "overlap_fraction",
        "soft_fraction",
        "core_profiled_fraction",
        "context_unresolved_fraction",
    ):
        if col not in interpretation.columns:
            interpretation[col] = 0.0
        interpretation[col] = pd.to_numeric(interpretation[col], errors="coerce").fillna(0.0)

    interpretation["program_count"] = interpretation["program_ids"].fillna("").astype(str).map(
        lambda x: len([part for part in x.split("|") if part.strip()])
    )
    interpretation["backbone_program_pair_count"] = interpretation["backbone_program_pairs"].fillna("").astype(str).map(
        lambda x: len([part for part in x.split(";") if part.strip()])
    )

    top_interfaces["sample_id"] = sample_id
    top_interfaces["niche_id"] = top_interfaces["niche_id"].astype(str)
    top_interfaces["niche_key"] = top_interfaces["sample_id"].astype(str) + "::" + top_interfaces["niche_id"].astype(str)
    top_interfaces["edge_type"] = top_interfaces["edge_type"].fillna("unknown").astype(str)
    top_interfaces["program_i"] = top_interfaces["program_i"].fillna("unresolved").astype(str)
    top_interfaces["program_j"] = top_interfaces["program_j"].fillna("unresolved").astype(str)
    top_interfaces["edge_strength"] = pd.to_numeric(top_interfaces["edge_strength"], errors="coerce").fillna(0.0)
    top_interfaces["edge_reliability"] = pd.to_numeric(top_interfaces["edge_reliability"], errors="coerce").fillna(0.0)
    top_interfaces["interface_signature"] = [
        _interface_signature(a, b, edge_type)
        for a, b, edge_type in top_interfaces.loc[:, ["program_i", "program_j", "edge_type"]].itertuples(index=False)
    ]
    top_interfaces["interface_weight"] = (
        np.clip(top_interfaces["edge_strength"].to_numpy(dtype=np.float64), 0.0, None)
        * np.clip(top_interfaces["edge_reliability"].to_numpy(dtype=np.float64), 0.0, 1.0)
    )
    top_interfaces["left_program_term_scores"] = top_interfaces["left_program_term_scores_json"].map(_safe_json_dict)
    top_interfaces["right_program_term_scores"] = top_interfaces["right_program_term_scores_json"].map(_safe_json_dict)
    top_interfaces["combined_term_scores"] = [
        _merge_score_dicts(left, right)
        for left, right in top_interfaces.loc[:, ["left_program_term_scores", "right_program_term_scores"]].itertuples(index=False)
    ]

    top_interfaces = top_interfaces.sort_values(
        ["niche_key", "interface_weight", "edge_reliability", "interface_signature"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    top_interfaces["local_rank"] = top_interfaces.groupby("niche_key").cumcount().astype(np.int64) + 1
    top_interfaces = top_interfaces.loc[
        top_interfaces["local_rank"].to_numpy(dtype=np.int64) <= int(max(1, cfg.feature.local_top_interface_count))
    ].copy()

    keep_keys = set(interpretation["niche_key"].tolist())
    top_interfaces = top_interfaces.loc[top_interfaces["niche_key"].isin(keep_keys)].reset_index(drop=True)

    status["loaded"] = True
    status["niche_count"] = int(interpretation.shape[0])
    status["top_interface_rows"] = int(top_interfaces.shape[0])
    return interpretation.reset_index(drop=True), top_interfaces.reset_index(drop=True), status


def _build_interface_catalog(top_interfaces: pd.DataFrame, cfg: AtlasConfig) -> pd.DataFrame:
    local = top_interfaces.copy()
    local["normalized_weight"] = _normalize_by_group_sum(local["interface_weight"], local["niche_key"])
    catalog = (
        local.groupby("interface_signature", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            niche_count=("niche_key", "nunique"),
            total_weight=("normalized_weight", "sum"),
            mean_weight=("normalized_weight", "mean"),
            total_interface_weight=("interface_weight", "sum"),
            mean_edge_reliability=("edge_reliability", "mean"),
        )
        .sort_values(
            ["sample_count", "niche_count", "total_weight", "total_interface_weight", "interface_signature"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    catalog["global_rank"] = np.arange(1, catalog.shape[0] + 1, dtype=np.int64)
    keep_n = int(max(1, cfg.feature.global_top_interface_count))
    catalog["selected_for_features"] = catalog["global_rank"].to_numpy(dtype=np.int64) <= keep_n
    return catalog


def _build_term_catalog(top_interfaces: pd.DataFrame, cfg: AtlasConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    local = top_interfaces.copy()
    local["normalized_interface_weight"] = _normalize_by_group_sum(local["interface_weight"], local["niche_key"])
    for niche_key, sample_id, score_dict, row_weight in local.loc[
        :, ["niche_key", "sample_id", "combined_term_scores", "normalized_interface_weight"]
    ].itertuples(index=False):
        for term, score in score_dict.items():
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "niche_key": str(niche_key),
                    "term": str(term),
                    "weighted_score": float(score) * float(row_weight),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "term",
                "sample_count",
                "niche_count",
                "total_weighted_score",
                "mean_weighted_score",
                "global_rank",
                "selected_for_features",
            ]
        )
    exploded = pd.DataFrame(rows)
    catalog = (
        exploded.groupby("term", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            niche_count=("niche_key", "nunique"),
            total_weighted_score=("weighted_score", "sum"),
            mean_weighted_score=("weighted_score", "mean"),
        )
        .sort_values(
            ["sample_count", "niche_count", "total_weighted_score", "term"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    catalog["global_rank"] = np.arange(1, catalog.shape[0] + 1, dtype=np.int64)
    keep_n = int(max(1, cfg.feature.global_top_term_count))
    catalog["selected_for_features"] = catalog["global_rank"].to_numpy(dtype=np.int64) <= keep_n
    return catalog


def _normalized_log_series(series: pd.Series) -> np.ndarray:
    values = np.log1p(pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
    max_val = float(np.max(values)) if values.size else 0.0
    if max_val <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / max_val


def _build_feature_table(
    interpretation: pd.DataFrame,
    top_interfaces: pd.DataFrame,
    interface_catalog: pd.DataFrame,
    term_catalog: pd.DataFrame,
    cfg: AtlasConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = interpretation.copy()
    base["struct__interaction_confidence"] = np.clip(
        base["interaction_confidence"].to_numpy(dtype=np.float64), 0.0, 1.0
    )
    base["struct__backbone_edge_count"] = _normalized_log_series(base["backbone_edge_count"])
    base["struct__backbone_node_count"] = _normalized_log_series(base["backbone_node_count"])
    base["struct__member_count"] = _normalized_log_series(base["member_count"])
    base["struct__program_count"] = _normalized_log_series(base["program_count"])
    base["struct__backbone_program_pair_count"] = _normalized_log_series(base["backbone_program_pair_count"])
    base["struct__contact_fraction"] = np.clip(base["contact_fraction"].to_numpy(dtype=np.float64), 0.0, 1.0)
    base["struct__overlap_fraction"] = np.clip(base["overlap_fraction"].to_numpy(dtype=np.float64), 0.0, 1.0)
    base["struct__soft_fraction"] = np.clip(base["soft_fraction"].to_numpy(dtype=np.float64), 0.0, 1.0)
    base["struct__core_profiled_fraction"] = np.clip(base["core_profiled_fraction"].to_numpy(dtype=np.float64), 0.0, 1.0)
    base["struct__context_unresolved_fraction"] = np.clip(
        base["context_unresolved_fraction"].to_numpy(dtype=np.float64), 0.0, 1.0
    )

    local = top_interfaces.copy()
    local["normalized_weight"] = _normalize_by_group_sum(local["interface_weight"], local["niche_key"])

    selected_interfaces = interface_catalog.loc[
        interface_catalog["selected_for_features"].to_numpy(dtype=bool),
        "interface_signature",
    ].astype(str).tolist()
    interface_matrix = (
        local.loc[
            local["interface_signature"].astype(str).isin(selected_interfaces),
            ["niche_key", "interface_signature", "normalized_weight"],
        ]
        .pivot_table(index="niche_key", columns="interface_signature", values="normalized_weight", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    if interface_matrix.empty:
        interface_matrix = pd.DataFrame({"niche_key": base["niche_key"].tolist()})
    for signature in selected_interfaces:
        if signature not in interface_matrix.columns:
            interface_matrix[signature] = 0.0

    selected_terms = term_catalog.loc[
        term_catalog["selected_for_features"].to_numpy(dtype=bool),
        "term",
    ].astype(str).tolist()
    term_rows: list[dict[str, object]] = []
    for niche_key, score_dict, row_weight in local.loc[
        :, ["niche_key", "combined_term_scores", "normalized_weight"]
    ].itertuples(index=False):
        for term, score in score_dict.items():
            if selected_terms and term not in selected_terms:
                continue
            term_rows.append(
                {
                    "niche_key": str(niche_key),
                    "term": str(term),
                    "weighted_score": float(score) * float(row_weight),
                }
            )
    if term_rows:
        term_long = pd.DataFrame(term_rows)
        total_term_mass = term_long.groupby("niche_key")["weighted_score"].transform("sum").to_numpy(dtype=np.float64)
        denom = np.where(total_term_mass > 0.0, total_term_mass, 1.0)
        term_long["normalized_score"] = term_long["weighted_score"].to_numpy(dtype=np.float64) / denom
        term_matrix = (
            term_long.pivot_table(index="niche_key", columns="term", values="normalized_score", aggfunc="sum", fill_value=0.0)
            .reset_index()
        )
    else:
        term_matrix = pd.DataFrame({"niche_key": base["niche_key"].tolist()})
    for term in selected_terms:
        if term not in term_matrix.columns:
            term_matrix[term] = 0.0

    feature_df = base.merge(interface_matrix, on="niche_key", how="left", validate="one_to_one")
    feature_df = feature_df.merge(term_matrix, on="niche_key", how="left", validate="one_to_one")

    interface_cols = [f"if__{signature}" for signature in selected_interfaces]
    for signature in selected_interfaces:
        raw_col = signature
        out_col = f"if__{signature}"
        feature_df[out_col] = pd.to_numeric(feature_df[raw_col], errors="coerce").fillna(0.0).astype(np.float64)

    if interface_cols:
        selected_mass = np.sum(feature_df.loc[:, interface_cols].to_numpy(dtype=np.float64), axis=1)
    else:
        selected_mass = np.zeros(feature_df.shape[0], dtype=np.float64)
    feature_df["if__other_interfaces"] = np.clip(1.0 - selected_mass, 0.0, 1.0)

    term_cols = [f"term__{term}" for term in selected_terms]
    for term in selected_terms:
        raw_col = term
        out_col = f"term__{term}"
        feature_df[out_col] = pd.to_numeric(feature_df[raw_col], errors="coerce").fillna(0.0).astype(np.float64)

    cluster_df = feature_df.loc[:, ["sample_id", "niche_id", "niche_key"]].copy()
    interface_feature_cols = interface_cols + ["if__other_interfaces"]
    structural_cols = [
        "struct__interaction_confidence",
        "struct__backbone_edge_count",
        "struct__backbone_node_count",
        "struct__member_count",
        "struct__program_count",
        "struct__backbone_program_pair_count",
        "struct__contact_fraction",
        "struct__overlap_fraction",
        "struct__soft_fraction",
        "struct__core_profiled_fraction",
        "struct__context_unresolved_fraction",
    ]
    structural_weight_map = {
        "struct__interaction_confidence": float(cfg.feature.interaction_confidence_weight),
        "struct__backbone_edge_count": float(cfg.feature.backbone_size_weight),
        "struct__backbone_node_count": float(cfg.feature.backbone_size_weight),
        "struct__member_count": float(cfg.feature.member_size_weight),
        "struct__program_count": float(cfg.feature.member_size_weight),
        "struct__backbone_program_pair_count": float(cfg.feature.edge_mix_weight),
        "struct__contact_fraction": float(cfg.feature.edge_mix_weight),
        "struct__overlap_fraction": float(cfg.feature.edge_mix_weight),
        "struct__soft_fraction": float(cfg.feature.edge_mix_weight),
        "struct__core_profiled_fraction": float(cfg.feature.core_profile_weight),
        "struct__context_unresolved_fraction": float(cfg.feature.context_unresolved_weight),
    }

    for col in interface_feature_cols:
        cluster_df[col] = (
            pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0).astype(np.float64)
            * float(cfg.feature.interface_group_weight)
        )
    for col in term_cols:
        cluster_df[col] = (
            pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0).astype(np.float64)
            * float(cfg.feature.term_group_weight)
        )
    for col in structural_cols:
        cluster_df[col] = (
            pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0).astype(np.float64)
            * float(cfg.feature.structural_group_weight)
            * float(structural_weight_map.get(col, 1.0))
        )

    feature_cols = interface_feature_cols + term_cols + structural_cols
    cluster_df["feature_l2_norm"] = np.sqrt(np.sum(cluster_df.loc[:, feature_cols].to_numpy(dtype=np.float64) ** 2, axis=1))
    zero_mask = cluster_df["feature_l2_norm"].to_numpy(dtype=np.float64) <= 0.0
    if bool(np.any(zero_mask)):
        cluster_df.loc[zero_mask, "struct__interaction_confidence"] = float(cfg.feature.structural_group_weight) * 1e-6
    cluster_df = cluster_df.drop(columns=["feature_l2_norm"])
    return feature_df, cluster_df


def _relabel_by_leaf_order(labels: np.ndarray, leaves: list[int]) -> np.ndarray:
    positions = {int(idx): i for i, idx in enumerate(leaves)}
    order = []
    for cluster_id in sorted(set(int(x) for x in labels.tolist())):
        members = np.where(labels == cluster_id)[0]
        first_pos = min(positions[int(idx)] for idx in members.tolist())
        order.append((first_pos, cluster_id))
    mapping = {cluster_id: rank for rank, (_, cluster_id) in enumerate(sorted(order), start=1)}
    return np.asarray([mapping[int(x)] for x in labels.tolist()], dtype=np.int64)


def _cluster_overlap_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    total = int(labels_a.shape[0])
    if total == 0:
        return 0.0
    score = 0.0
    for cluster_a in sorted(set(int(x) for x in labels_a.tolist())):
        members_a = labels_a == cluster_a
        size_a = int(np.count_nonzero(members_a))
        if size_a == 0:
            continue
        best = 0.0
        for cluster_b in sorted(set(int(x) for x in labels_b.tolist())):
            members_b = labels_b == cluster_b
            inter = int(np.count_nonzero(members_a & members_b))
            union = int(np.count_nonzero(members_a | members_b))
            jac = float(inter / union) if union > 0 else 0.0
            if jac > best:
                best = jac
        score += float(size_a) * best
    return float(score / total)


def _recommended_k(cut_eval: pd.DataFrame) -> int:
    ranked = cut_eval.sort_values(
        ["singleton_fraction", "neighbor_stability", "silhouette_score", "k"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    return int(ranked.iloc[0]["k"])


def _cluster_summary_for_cut(
    feature_df: pd.DataFrame,
    assignments: pd.DataFrame,
    interface_catalog: pd.DataFrame,
    term_catalog: pd.DataFrame,
    cluster_col: str,
) -> pd.DataFrame:
    selected_interfaces = interface_catalog.loc[
        interface_catalog["selected_for_features"].to_numpy(dtype=bool),
        "interface_signature",
    ].astype(str).tolist()
    selected_terms = term_catalog.loc[
        term_catalog["selected_for_features"].to_numpy(dtype=bool),
        "term",
    ].astype(str).tolist()
    interface_cols = [f"if__{signature}" for signature in selected_interfaces]
    term_cols = [f"term__{term}" for term in selected_terms]
    rows: list[dict[str, object]] = []
    merged = feature_df.merge(assignments.loc[:, ["niche_key", cluster_col]], on="niche_key", how="left", validate="one_to_one")

    for cluster_id, grp in merged.groupby(cluster_col):
        cluster_size = int(grp.shape[0])
        if cluster_size == 0:
            continue
        mean_interface = (
            grp.loc[:, interface_cols].mean(axis=0).sort_values(ascending=False)
            if interface_cols else pd.Series(dtype=np.float64)
        )
        mean_terms = (
            grp.loc[:, term_cols].mean(axis=0).sort_values(ascending=False)
            if term_cols else pd.Series(dtype=np.float64)
        )
        top_interfaces = [str(col).replace("if__", "", 1) for col in mean_interface.index[:3].tolist()]
        top_interface_weights = [float(mean_interface.iloc[i]) for i in range(min(3, mean_interface.shape[0]))]
        top_terms = [str(col).replace("term__", "", 1) for col in mean_terms.index[:3].tolist()]
        top_term_weights = [float(mean_terms.iloc[i]) for i in range(min(3, mean_terms.shape[0]))]
        top1_interface = top_interfaces[0] if top_interfaces else ""
        row_top1 = []
        if top1_interface and interface_cols:
            row_top1 = grp.loc[:, interface_cols].idxmax(axis=1).astype(str).str.replace("if__", "", regex=False).tolist()
        top1_purity = float(np.mean([label == top1_interface for label in row_top1])) if row_top1 and top1_interface else 0.0
        rows.append(
            {
                cluster_col: int(cluster_id),
                "niche_count": cluster_size,
                "is_singleton_cluster": bool(cluster_size == 1),
                "archetype_status": "provisional_singleton" if cluster_size == 1 else "cluster_family",
                "sample_count": int(grp["sample_id"].nunique()),
                "sample_ids": "|".join(sorted(grp["sample_id"].astype(str).unique().tolist())),
                "top_interface_1": top_interfaces[0] if len(top_interfaces) > 0 else "",
                "top_interface_2": top_interfaces[1] if len(top_interfaces) > 1 else "",
                "top_interface_3": top_interfaces[2] if len(top_interfaces) > 2 else "",
                "top_interface_weight_1": top_interface_weights[0] if len(top_interface_weights) > 0 else 0.0,
                "top_interface_weight_2": top_interface_weights[1] if len(top_interface_weights) > 1 else 0.0,
                "top_interface_weight_3": top_interface_weights[2] if len(top_interface_weights) > 2 else 0.0,
                "top_term_1": top_terms[0] if len(top_terms) > 0 else "",
                "top_term_2": top_terms[1] if len(top_terms) > 1 else "",
                "top_term_3": top_terms[2] if len(top_terms) > 2 else "",
                "top_term_weight_1": top_term_weights[0] if len(top_term_weights) > 0 else 0.0,
                "top_term_weight_2": top_term_weights[1] if len(top_term_weights) > 1 else 0.0,
                "top_term_weight_3": top_term_weights[2] if len(top_term_weights) > 2 else 0.0,
                "top_interface_purity": float(top1_purity),
                "mean_interaction_confidence": float(grp["interaction_confidence"].mean()),
                "mean_backbone_edge_count": float(grp["backbone_edge_count"].mean()),
                "mean_member_count": float(grp["member_count"].mean()),
                "mean_contact_fraction": float(grp["contact_fraction"].mean()),
                "mean_overlap_fraction": float(grp["overlap_fraction"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values([cluster_col]).reset_index(drop=True)
    return out


def run_atlas_pipeline(
    niche_bundle_paths: list[str | Path],
    out_root: str | Path,
    cohort_id: str,
    config: AtlasConfig | None = None,
) -> Path:
    cfg = config or AtlasConfig()
    out_dir = Path(out_root) / "atlas"
    out_dir.mkdir(parents=True, exist_ok=True)

    interpretation_frames: list[pd.DataFrame] = []
    top_interface_frames: list[pd.DataFrame] = []
    sample_status_rows: list[dict] = []
    for raw_path in niche_bundle_paths:
        niche_bundle_path = Path(raw_path)
        interpretation, top_interfaces, status = _load_sample_tables(niche_bundle_path=niche_bundle_path, cfg=cfg)
        sample_status_rows.append(status)
        if interpretation is None or top_interfaces is None:
            continue
        interpretation_frames.append(interpretation)
        top_interface_frames.append(top_interfaces)

    sample_status = pd.DataFrame(sample_status_rows)
    write_tsv(sample_status, out_dir / "sample_inventory.tsv")

    loaded_mask = sample_status["loaded"].to_numpy(dtype=bool) if not sample_status.empty else np.asarray([], dtype=bool)
    if int(np.count_nonzero(loaded_mask)) == 0:
        raise ValueError("No samples with usable niche_annotation outputs were found for Atlas.")

    interpretation = pd.concat(interpretation_frames, ignore_index=True).reset_index(drop=True)
    top_interfaces = pd.concat(top_interface_frames, ignore_index=True).reset_index(drop=True)
    interpretation = interpretation.sort_values(["sample_id", "niche_id"]).reset_index(drop=True)
    top_interfaces = top_interfaces.sort_values(["sample_id", "niche_id", "local_rank"]).reset_index(drop=True)

    interface_catalog = _build_interface_catalog(top_interfaces=top_interfaces, cfg=cfg)
    term_catalog = _build_term_catalog(top_interfaces=top_interfaces, cfg=cfg)
    feature_df, cluster_df = _build_feature_table(
        interpretation=interpretation,
        top_interfaces=top_interfaces,
        interface_catalog=interface_catalog,
        term_catalog=term_catalog,
        cfg=cfg,
    )

    feature_cols = [col for col in cluster_df.columns if col.startswith(("if__", "term__", "struct__"))]
    x = cluster_df.loc[:, feature_cols].to_numpy(dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("Need at least two niches to perform archetype clustering.")

    distance_vec = pdist(x, metric=str(cfg.clustering.distance_metric))
    if distance_vec.size == 0:
        raise ValueError("Distance vector is empty; cannot cluster a single niche.")
    linkage_mat = linkage(distance_vec, method=str(cfg.clustering.linkage_method))
    leaves = [int(x) for x in dendrogram(linkage_mat, no_plot=True)["leaves"]]

    order_df = cluster_df.loc[leaves, ["sample_id", "niche_id", "niche_key"]].copy().reset_index(drop=True)
    order_df["leaf_order"] = np.arange(1, order_df.shape[0] + 1, dtype=np.int64)

    n_niches = int(cluster_df.shape[0])
    square_dist = squareform(distance_vec)
    assignment_frames: list[pd.DataFrame] = []
    cut_eval_rows: list[dict] = []
    cluster_summary_frames: list[pd.DataFrame] = []
    label_map: dict[int, np.ndarray] = {}

    for k in sorted(set(int(v) for v in cfg.clustering.k_values if int(v) >= 2)):
        if k > n_niches:
            continue
        labels = fcluster(linkage_mat, t=k, criterion="maxclust").astype(np.int64)
        labels = _relabel_by_leaf_order(labels=labels, leaves=leaves)
        label_map[k] = labels

        cluster_col = f"k{k}_cluster"
        assign = cluster_df.loc[:, ["sample_id", "niche_id", "niche_key"]].copy()
        assign["k"] = int(k)
        assign["cluster_id"] = labels.astype(np.int64)
        assign["archetype_id"] = [f"K{k}_A{cid:02d}" for cid in assign["cluster_id"].tolist()]
        cluster_sizes = assign["cluster_id"].value_counts().to_dict()
        assign["cluster_size"] = assign["cluster_id"].map(cluster_sizes).astype(np.int64)
        assign["archetype_status"] = np.where(
            assign["cluster_size"].to_numpy(dtype=np.int64) == 1,
            "provisional_singleton",
            "cluster_family",
        )
        assignment_frames.append(assign)

        sil = float("nan")
        if 1 < len(set(labels.tolist())) < n_niches:
            sil = float(silhouette_score(square_dist, labels, metric="precomputed"))
        _, counts = np.unique(labels, return_counts=True)
        singleton_clusters = int(np.count_nonzero(counts == 1))
        singleton_niches = int(np.sum(counts[counts == 1]))
        cut_eval_rows.append(
            {
                "k": int(k),
                "cluster_count": int(len(set(labels.tolist()))),
                "silhouette_score": sil,
                "singleton_cluster_count": singleton_clusters,
                "singleton_fraction": float(singleton_niches / n_niches),
            }
        )

        summary = _cluster_summary_for_cut(
            feature_df=feature_df,
            assignments=assign.rename(columns={"cluster_id": cluster_col}),
            interface_catalog=interface_catalog,
            term_catalog=term_catalog,
            cluster_col=cluster_col,
        )
        if not summary.empty:
            summary.insert(0, "k", int(k))
            summary = summary.rename(columns={cluster_col: "cluster_id"})
            summary["archetype_id"] = [f"K{k}_A{cid:02d}" for cid in summary["cluster_id"].tolist()]
            cluster_summary_frames.append(summary)

    cut_eval = pd.DataFrame(cut_eval_rows)
    if cut_eval.empty:
        raise ValueError("No valid k values remain after filtering against niche count.")

    neighbor_scores: list[float] = []
    for k in cut_eval["k"].astype(int).tolist():
        neighbors = []
        for other_k in (k - 1, k + 1):
            if other_k in label_map:
                neighbors.append(_cluster_overlap_score(label_map[k], label_map[other_k]))
        neighbor_scores.append(float(np.mean(neighbors)) if neighbors else 0.0)
    cut_eval["neighbor_stability"] = np.asarray(neighbor_scores, dtype=np.float64)
    cut_eval = cut_eval.sort_values(["k"]).reset_index(drop=True)

    recommended_k = _recommended_k(cut_eval=cut_eval)
    assignments = pd.concat(assignment_frames, ignore_index=True).reset_index(drop=True)
    assignments["is_recommended_k"] = assignments["k"].to_numpy(dtype=np.int64) == int(recommended_k)
    cluster_summary = pd.concat(cluster_summary_frames, ignore_index=True).reset_index(drop=True)
    cluster_summary["is_recommended_k"] = cluster_summary["k"].to_numpy(dtype=np.int64) == int(recommended_k)

    linkage_df = pd.DataFrame(
        linkage_mat,
        columns=["left", "right", "distance", "leaf_count"],
    )
    linkage_df.insert(0, "merge_step", np.arange(1, linkage_df.shape[0] + 1, dtype=np.int64))

    manifest = {
        "schema_version": cfg.schema_version,
        "cohort_id": str(cohort_id),
        "sample_count_loaded": int(np.count_nonzero(loaded_mask)),
        "sample_count_requested": int(sample_status.shape[0]),
        "niche_count": int(feature_df.shape[0]),
        "recommended_k": int(recommended_k),
        "selected_interface_features": interface_catalog.loc[
            interface_catalog["selected_for_features"].to_numpy(dtype=bool),
            "interface_signature",
        ].astype(str).tolist(),
        "selected_term_features": term_catalog.loc[
            term_catalog["selected_for_features"].to_numpy(dtype=bool),
            "term",
        ].astype(str).tolist(),
        "distance_metric": str(cfg.clustering.distance_metric),
        "linkage_method": str(cfg.clustering.linkage_method),
        "feature_groups": {
            "interface": {
                "count": int(np.sum([col.startswith("if__") for col in feature_cols])),
                "weight": float(cfg.feature.interface_group_weight),
            },
            "term": {
                "count": int(np.sum([col.startswith("term__") for col in feature_cols])),
                "weight": float(cfg.feature.term_group_weight),
            },
            "structural": {
                "count": int(np.sum([col.startswith("struct__") for col in feature_cols])),
                "weight": float(cfg.feature.structural_group_weight),
            },
        },
        "config": asdict(cfg),
        "files": {
            "sample_inventory": "sample_inventory.tsv",
            "interface_catalog": "interface_catalog.tsv",
            "term_catalog": "term_catalog.tsv",
            "niche_features": "atlas_features.tsv",
            "cluster_matrix": "atlas_cluster_matrix.tsv",
            "dendrogram_order": "dendrogram_order.tsv",
            "linkage_matrix": "linkage_matrix.tsv",
            "cut_evaluation": "cut_evaluation.tsv",
            "archetype_assignments": "archetype_assignments.tsv",
            "archetype_cluster_summary": "archetype_cluster_summary.tsv",
        },
    }

    write_tsv(interface_catalog, out_dir / "interface_catalog.tsv")
    write_tsv(term_catalog, out_dir / "term_catalog.tsv")
    write_tsv(feature_df, out_dir / "atlas_features.tsv")
    write_tsv(cluster_df, out_dir / "atlas_cluster_matrix.tsv")
    write_tsv(order_df, out_dir / "dendrogram_order.tsv")
    write_tsv(linkage_df, out_dir / "linkage_matrix.tsv")
    write_tsv(cut_eval, out_dir / "cut_evaluation.tsv")
    write_tsv(assignments, out_dir / "archetype_assignments.tsv")
    write_tsv(cluster_summary, out_dir / "archetype_cluster_summary.tsv")
    write_json(out_dir / "manifest.json", manifest)
    return out_dir
