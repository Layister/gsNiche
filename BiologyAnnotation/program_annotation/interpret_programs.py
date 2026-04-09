from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gsniche-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gsniche-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import yaml

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from BiologyAnnotation.bundle_io import write_json, write_tsv
    from NicheGraph.common import benjamini_hochberg
else:
    from ..bundle_io import write_json, write_tsv
    from NicheGraph.common import benjamini_hochberg


DEFAULT_WORK_DIR = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "PRAD"
DEFAULT_SAMPLE_ID = "INT27"
DEFAULT_SOURCE_PROFILE_YAML = Path("/Users/wuyang/Documents/MyPaper/3/gsNiche/resources/program_annotation_sources.yaml")

PROGRAM_QUALITY_COLUMNS: tuple[str, ...] = (
    "program_size_genes",
    "program_gene_frac",
    "template_run_support_frac",
    "template_spot_support_frac",
    "template_focus_score",
    "program_template_evidence_score",
    "program_confidence",
    "validity_status",
    "routing_status",
    "default_use_support_score",
    "default_use_reason_count",
)


@dataclass(frozen=True)
class AnnotationSourceProfile:
    source: str
    gmt_path: str
    term_category: str
    min_gene_set_size: int
    max_gene_set_size: int
    trim_redundant_terms: bool
    redundancy_jaccard_threshold: float

@dataclass
class ProgramAnnotationConfig:
    top_gene_scales: tuple[int, ...] = (10, 20, 30)
    top_genes_for_summary: int = 10
    fdr_threshold: float = 0.20
    min_overlap: int = 5
    min_gene_set_size: int = 10
    max_gene_set_size: int = 1000
    confidence_score_scale: float = 4.0
    confidence_magnitude_weight: float = 0.45
    confidence_share_weight: float = 0.35
    confidence_significance_weight: float = 0.20
    go_redundancy_jaccard_threshold: float = 0.80
    summary_term_score_share_threshold: float = 0.80
    emit_raw_expression_ora: bool = True
    raw_expression_data_layer: str | None = None
    raw_expression_min_active_spots: int = 8
    raw_expression_gene_set_size_factor: float = 1.0
    raw_expression_min_gene_set_size: int = 15
    raw_expression_max_gene_set_size: int = 300
    random_seed: int = 2024


def _load_source_profiles(path: Path) -> tuple[str, dict[str, AnnotationSourceProfile]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid source profile YAML: {path}")

    default_source = str(payload.get("default_source", "")).strip().lower()
    sources = payload.get("sources", {})
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"Source profile YAML missing sources mapping: {path}")

    profiles: dict[str, AnnotationSourceProfile] = {}
    for source_name, raw in sources.items():
        if not isinstance(raw, dict):
            continue
        name = str(source_name).strip().lower()
        gmt_relpath = str(raw.get("gmt_relpath", "")).strip()
        if not name or not gmt_relpath:
            continue
        gmt_path = (path.parent / gmt_relpath).resolve()
        profiles[name] = AnnotationSourceProfile(
            source=name,
            gmt_path=str(gmt_path),
            term_category=str(raw.get("term_category", name)),
            min_gene_set_size=int(raw.get("min_gene_set_size", 10)),
            max_gene_set_size=int(raw.get("max_gene_set_size", 2000)),
            trim_redundant_terms=bool(raw.get("trim_redundant_terms", False)),
            redundancy_jaccard_threshold=float(raw.get("redundancy_jaccard_threshold", 1.0)),
        )

    if not profiles:
        raise ValueError(f"Source profile YAML contains no valid profiles: {path}")
    if not default_source:
        default_source = next(iter(sorted(profiles.keys())))
    if default_source not in profiles:
        raise ValueError(f"default_source '{default_source}' not found in source profile YAML: {path}")
    return default_source, profiles


def _norm_gene(gene: str) -> str:
    return str(gene).strip().upper()


def _join_pipe(values: list[str]) -> str:
    return "|".join([str(v).strip() for v in values if str(v).strip()])


def _first_non_na(series: pd.Series):
    for value in series.tolist():
        if pd.notna(value):
            return value
    return np.nan


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _saturating_score(value: float, scale: float) -> float:
    scale_value = max(float(scale), 1e-8)
    return _clamp01(1.0 - math.exp(-max(float(value), 0.0) / scale_value))


def _normalize_term_name(term: str, source: str) -> str:
    txt = str(term).strip()
    upper = txt.upper()
    prefixes = {
        "hallmark": ("HALLMARK_",),
        "go_bp": ("GO_BIOLOGICAL_PROCESS_", "GOBP_", "GOBP"),
        "reactome": ("REACTOME_",),
        "kegg": ("KEGG_", "KEGG_MEDICUS_",),
    }.get(str(source), ())
    for prefix in prefixes:
        if upper.startswith(prefix):
            txt = txt[len(prefix):]
            break
    txt = re.sub(r"[_/]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return "Unresolved Annotation"
    return " ".join(word.capitalize() for word in txt.split())


def _clean_term_description(term: str, description: str) -> str:
    desc = str(description).strip()
    if not desc:
        return ""
    if desc.lower() in {"na", "n/a", "none", "null"}:
        return ""
    if desc == str(term).strip():
        return ""
    return desc


def _read_gmt(path: Path, source: str, profile: AnnotationSourceProfile) -> dict[str, dict]:
    term_records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            term_id = str(parts[0]).strip()
            if not term_id:
                continue
            genes = {_norm_gene(g) for g in parts[2:] if str(g).strip()}
            if not genes:
                continue
            term_records[term_id] = {
                "term_id": term_id,
                "term_name": _normalize_term_name(term_id, source=source),
                "term_description": _clean_term_description(term_id, str(parts[1]).strip()),
                "term_category": str(profile.term_category),
                "genes": genes,
            }
    if not term_records:
        raise ValueError(f"No valid terms parsed from GMT: {path}")
    return term_records


def _resolve_dataset_gene_universe(program_bundle: Path, programs_df: pd.DataFrame) -> tuple[set[str], str]:
    manifest_path = program_bundle / "manifest.json"
    if not manifest_path.exists():
        return {_norm_gene(x) for x in programs_df["gene"].astype(str).tolist()}, "program_genes_fallback"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    inputs = manifest.get("inputs", {})
    params = manifest.get("params", {})
    gss_bundle_path = str(inputs.get("gss_bundle_path", "")).strip()
    gss_sparse_relpath = str(params.get("input", {}).get("gss_sparse_relpath", "gss/gss_sparse.parquet")).strip()
    if not gss_bundle_path:
        return {_norm_gene(x) for x in programs_df["gene"].astype(str).tolist()}, "program_genes_fallback"

    gss_sparse_path = Path(gss_bundle_path) / gss_sparse_relpath
    if not gss_sparse_path.exists():
        return {_norm_gene(x) for x in programs_df["gene"].astype(str).tolist()}, "program_genes_fallback"

    try:
        gss_df = pd.read_parquet(gss_sparse_path, columns=["gene"])
    except Exception:
        gss_df = pd.read_parquet(gss_sparse_path)
        if "gene" not in gss_df.columns:
            return {_norm_gene(x) for x in programs_df["gene"].astype(str).tolist()}, "program_genes_fallback"
    return {_norm_gene(x) for x in gss_df["gene"].astype(str).tolist()}, "gss_sparse_gene_universe"


def _resolve_raw_h5ad_path(program_bundle: Path) -> Path:
    manifest_path = program_bundle / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing program manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gss_bundle_path = Path(str(manifest.get("inputs", {}).get("gss_bundle_path", "")).strip())
    if not str(gss_bundle_path):
        raise ValueError(f"program manifest missing inputs.gss_bundle_path: {manifest_path}")
    gss_manifest_path = gss_bundle_path / "manifest.json"
    if not gss_manifest_path.exists():
        raise FileNotFoundError(f"Missing gss manifest: {gss_manifest_path}")
    gss_manifest = json.loads(gss_manifest_path.read_text(encoding="utf-8"))
    raw_h5ad = Path(str(gss_manifest.get("inputs", {}).get("h5ad_path", "")).strip())
    if not str(raw_h5ad):
        raise ValueError(f"gss manifest missing inputs.h5ad_path: {gss_manifest_path}")
    if not raw_h5ad.exists():
        raise FileNotFoundError(f"Raw h5ad not found: {raw_h5ad}")
    return raw_h5ad


def _build_program_quality_table(program_bundle: Path, programs_df_raw: pd.DataFrame) -> pd.DataFrame:
    if programs_df_raw.empty:
        return pd.DataFrame(columns=["program_id", "n_genes", *PROGRAM_QUALITY_COLUMNS])

    base = (
        programs_df_raw.loc[:, ["program_id", "gene"]]
        .copy()
        .assign(program_id=lambda d: d["program_id"].astype(str), gene=lambda d: d["gene"].astype(str))
        .groupby("program_id", as_index=False)["gene"]
        .nunique()
        .rename(columns={"gene": "n_genes"})
    )

    quality_cols = [c for c in PROGRAM_QUALITY_COLUMNS if c in programs_df_raw.columns]
    if quality_cols:
        q = programs_df_raw.loc[:, ["program_id", *quality_cols]].copy()
        q["program_id"] = q["program_id"].astype(str)
        q = q.groupby("program_id", as_index=False).agg({c: _first_non_na for c in quality_cols})
        base = base.merge(q, on="program_id", how="left")

    program_qc_path = program_bundle / "qc_tables" / "program_qc.parquet"
    if program_qc_path.exists():
        try:
            qc_df = pd.read_parquet(program_qc_path)
        except Exception:
            qc_df = pd.DataFrame()
        if not qc_df.empty and "program_id" in qc_df.columns:
            keep_cols = [
                "program_id",
                "template_evidence_score",
                "stability_score",
                "activation_score",
                "activation_coverage",
                "activation_threshold",
                "top20_jaccard_p50",
                "rank_corr_p50",
                "stable_high_contribution_gene_set_size",
                "assignment_ari_p50",
            ]
            keep_cols = [c for c in keep_cols if c in qc_df.columns]
            if keep_cols:
                qc_df = qc_df.loc[:, keep_cols].copy()
                qc_df["program_id"] = qc_df["program_id"].astype(str)
                qc_df = qc_df.drop_duplicates(subset=["program_id"])
                base = base.merge(qc_df, on="program_id", how="left")

    return base.sort_values("program_id").reset_index(drop=True)


def _load_program_inputs(program_bundle: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    programs_path = program_bundle / "programs.parquet"
    if not programs_path.exists():
        raise FileNotFoundError(f"Missing programs.parquet: {programs_path}")

    raw = pd.read_parquet(programs_path)
    required = {"program_id", "gene"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"programs.parquet missing required columns: {sorted(missing)}")

    if "weight" not in raw.columns:
        raise ValueError("programs.parquet must include 'weight'")

    programs_df = raw.loc[:, ["program_id", "gene", "weight"]].copy()
    programs_df = programs_df.rename(columns={"weight": "gene_weight"})
    programs_df["program_id"] = programs_df["program_id"].astype(str)
    programs_df["gene"] = programs_df["gene"].astype(str)
    programs_df["gene_norm"] = programs_df["gene"].map(_norm_gene)
    programs_df["gene_weight"] = pd.to_numeric(programs_df["gene_weight"], errors="coerce").fillna(0.0)
    programs_df["gene_weight_abs"] = programs_df["gene_weight"].abs()

    quality_table = _build_program_quality_table(program_bundle=program_bundle, programs_df_raw=raw)
    return programs_df, quality_table


def _ranked_gene_set_strings(sub: pd.DataFrame, scales: tuple[int, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    genes = sub["gene"].astype(str).tolist()
    for n in scales:
        out[f"top{int(n)}_genes"] = _join_pipe(genes[: max(1, int(n))])
    return out


def _build_program_gene_table(
    programs_df: pd.DataFrame,
    program_quality_table: pd.DataFrame,
    cfg: ProgramAnnotationConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    ranked_tables: dict[str, pd.DataFrame] = {}

    for pid, sub in programs_df.groupby("program_id"):
        pid_str = str(pid)
        sub = sub.copy().sort_values(["gene_weight_abs", "gene_norm"], ascending=[False, True]).reset_index(drop=True)
        n = int(sub.shape[0])
        abs_weights = sub["gene_weight_abs"].to_numpy(dtype=np.float64)
        total = float(np.sum(abs_weights))
        if total <= 0.0:
            abs_weights = np.ones_like(abs_weights, dtype=np.float64)
            total = float(np.sum(abs_weights))
        contribution = abs_weights / total

        sub["rank"] = np.arange(1, n + 1, dtype=np.int64)
        sub["gene_contribution"] = contribution.astype(np.float64)
        ranked_tables[pid_str] = sub.copy()
        top_gene_strings = _ranked_gene_set_strings(sub, cfg.top_gene_scales)

        for row in sub.itertuples(index=False):
            record = {
                "program_id": pid_str,
                "gene": str(row.gene),
                "gene_norm": str(row.gene_norm),
                "gene_weight": float(row.gene_weight),
                "gene_weight_abs": float(row.gene_weight_abs),
                "rank": int(row.rank),
                "gene_contribution": float(row.gene_contribution),
            }
            record.update(top_gene_strings)
            rows.append(record)

    out = pd.DataFrame(rows)
    if out.empty:
        cols = ["program_id", "gene", "gene_norm", "gene_weight", "gene_weight_abs", "rank", "gene_contribution"]
        cols.extend([f"top{int(n)}_genes" for n in cfg.top_gene_scales])
        out = pd.DataFrame(columns=cols)

    if not program_quality_table.empty:
        out = out.merge(program_quality_table, on="program_id", how="left")

    return out.sort_values(["program_id", "rank"]).reset_index(drop=True), ranked_tables


def _score_vector_to_contribution(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        if arr.size == 0:
            return arr
        arr = np.ones(arr.shape[0], dtype=np.float64)
        total = float(arr.sum())
    return arr / total


def _select_raw_expression_program_genes(
    gene_names: np.ndarray,
    score: np.ndarray,
    active_mean: np.ndarray,
    target_size: int,
) -> pd.DataFrame:
    genes = np.asarray(gene_names).astype(str, copy=False)
    score_arr = np.asarray(score, dtype=np.float64).reshape(-1)
    active_arr = np.asarray(active_mean, dtype=np.float64).reshape(-1)
    positive = np.flatnonzero(np.isfinite(score_arr) & np.isfinite(active_arr) & (score_arr > 0.0))
    if positive.size == 0:
        positive = np.flatnonzero(np.isfinite(active_arr) & (active_arr > 0.0))
    if positive.size == 0:
        return pd.DataFrame(columns=["gene", "gene_norm", "gene_weight", "gene_weight_abs", "rank", "gene_contribution"])

    order = positive[np.lexsort((genes[positive], -active_arr[positive], -score_arr[positive]))]
    keep_n = int(max(1, min(target_size, order.size)))
    keep = order[:keep_n]
    kept_scores = np.clip(score_arr[keep], 0.0, None)
    if float(np.sum(kept_scores)) <= 0.0:
        kept_scores = np.clip(active_arr[keep], 0.0, None)
    contribution = _score_vector_to_contribution(kept_scores)
    out = pd.DataFrame(
        {
            "gene": genes[keep].astype(str),
            "gene_norm": [ _norm_gene(g) for g in genes[keep].astype(str).tolist() ],
            "gene_weight": score_arr[keep].astype(np.float64),
            "gene_weight_abs": np.abs(score_arr[keep]).astype(np.float64),
            "rank": np.arange(1, keep_n + 1, dtype=np.int64),
            "gene_contribution": contribution.astype(np.float64),
        }
    )
    return out


def _build_raw_expression_ranked_tables(
    program_bundle: Path,
    program_quality_table: pd.DataFrame,
    cfg: ProgramAnnotationConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, object], set[str]]:
    activation_path = program_bundle / "program_activation.parquet"
    if not activation_path.exists():
        raise FileNotFoundError(f"Missing program_activation.parquet: {activation_path}")

    import scanpy as sc

    h5ad_path = _resolve_raw_h5ad_path(program_bundle=program_bundle)
    adata = sc.read_h5ad(h5ad_path)
    expr = adata.layers[str(cfg.raw_expression_data_layer)] if cfg.raw_expression_data_layer else adata.X
    gene_names = adata.var_names.astype(str).to_numpy()
    spot_ids = adata.obs_names.astype(str).to_numpy()

    activation_df = pd.read_parquet(
        activation_path,
        columns=["program_id", "spot_id", "activation_identity_view", "activation_full"],
    )
    if activation_df.empty:
        empty = pd.DataFrame(columns=["program_id", "gene", "gene_norm", "gene_weight", "gene_weight_abs", "rank", "gene_contribution"])
        return empty, {}, {
            "enabled": True,
            "raw_h5ad_path": str(h5ad_path.resolve()),
            "raw_expression_data_layer": cfg.raw_expression_data_layer,
            "gene_universe_size": int(gene_names.shape[0]),
            "n_programs_with_raw_expression_annotation": 0,
            "min_active_spots": int(cfg.raw_expression_min_active_spots),
        }, {_norm_gene(x) for x in gene_names.tolist()}

    activation_df["program_id"] = activation_df["program_id"].astype(str)
    activation_df["spot_id"] = activation_df["spot_id"].astype(str)
    activation_df["activation_identity_view"] = pd.to_numeric(activation_df["activation_identity_view"], errors="coerce").fillna(0.0)
    activation_df["activation_full"] = pd.to_numeric(activation_df["activation_full"], errors="coerce").fillna(0.0)

    spot_index = pd.Index(spot_ids.astype(str))
    quality_by_pid = (
        program_quality_table.set_index(program_quality_table["program_id"].astype(str)).to_dict(orient="index")
        if (not program_quality_table.empty and "program_id" in program_quality_table.columns)
        else {}
    )

    rows: list[dict] = []
    ranked_tables: dict[str, pd.DataFrame] = {}
    used_programs = 0
    for pid, sub in activation_df.groupby("program_id"):
        active_spot_ids = sub.loc[sub["activation_identity_view"].to_numpy(dtype=np.float64) > 0.0, "spot_id"].astype(str).unique().tolist()
        if len(active_spot_ids) < int(cfg.raw_expression_min_active_spots):
            active_spot_ids = sub.sort_values(
                ["activation_identity_view", "activation_full", "spot_id"],
                ascending=[False, False, True],
            )["spot_id"].astype(str).unique().tolist()
        active_spot_ids = active_spot_ids[:]
        if len(active_spot_ids) < int(cfg.raw_expression_min_active_spots):
            continue

        active_idx = spot_index.get_indexer(active_spot_ids)
        active_idx = active_idx[active_idx >= 0]
        if active_idx.size < int(cfg.raw_expression_min_active_spots):
            continue
        inactive_mask = np.ones(spot_ids.shape[0], dtype=bool)
        inactive_mask[active_idx] = False
        inactive_idx = np.flatnonzero(inactive_mask)

        active_mat = expr[active_idx]
        active_mean = np.asarray(active_mat.mean(axis=0)).reshape(-1).astype(np.float64)
        if inactive_idx.size > 0:
            inactive_mat = expr[inactive_idx]
            inactive_mean = np.asarray(inactive_mat.mean(axis=0)).reshape(-1).astype(np.float64)
        else:
            inactive_mean = np.zeros_like(active_mean)
        contrast = np.log1p(np.clip(active_mean, 0.0, None)) - np.log1p(np.clip(inactive_mean, 0.0, None))

        base_program_size = int(quality_by_pid.get(str(pid), {}).get("n_genes", 0))
        target_size = int(round(max(1, base_program_size) * float(cfg.raw_expression_gene_set_size_factor)))
        target_size = int(max(int(cfg.raw_expression_min_gene_set_size), min(int(cfg.raw_expression_max_gene_set_size), target_size)))
        ranked_df = _select_raw_expression_program_genes(
            gene_names=gene_names,
            score=contrast,
            active_mean=active_mean,
            target_size=target_size,
        )
        if ranked_df.empty:
            continue
        ranked_tables[str(pid)] = ranked_df.copy()
        used_programs += 1
        top_gene_strings = _ranked_gene_set_strings(ranked_df, cfg.top_gene_scales)
        for row in ranked_df.itertuples(index=False):
            record = {
                "program_id": str(pid),
                "gene": str(row.gene),
                "gene_norm": str(row.gene_norm),
                "gene_weight": float(row.gene_weight),
                "gene_weight_abs": float(row.gene_weight_abs),
                "rank": int(row.rank),
                "gene_contribution": float(row.gene_contribution),
            }
            record.update(top_gene_strings)
            rows.append(record)

    gene_table = pd.DataFrame(rows)
    if not gene_table.empty and not program_quality_table.empty:
        gene_table = gene_table.merge(program_quality_table, on="program_id", how="left")
    return gene_table.sort_values(["program_id", "rank"]).reset_index(drop=True), ranked_tables, {
        "enabled": True,
        "raw_h5ad_path": str(h5ad_path.resolve()),
        "raw_expression_data_layer": cfg.raw_expression_data_layer,
        "gene_universe_size": int(gene_names.shape[0]),
        "n_programs_with_raw_expression_annotation": int(used_programs),
        "min_active_spots": int(cfg.raw_expression_min_active_spots),
        "gene_set_size_factor": float(cfg.raw_expression_gene_set_size_factor),
        "min_gene_set_size": int(cfg.raw_expression_min_gene_set_size),
        "max_gene_set_size": int(cfg.raw_expression_max_gene_set_size),
    }, {_norm_gene(x) for x in gene_names.tolist()}


def _empty_hits_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "program_id",
            "routing_status",
            "annotation_source",
            "term_id",
            "term_name",
            "term_description",
            "term_category",
            "score",
            "rank",
            "overlap_size",
            "term_size",
            "program_size",
            "background_size",
            "overlap_ratio_program",
            "overlap_ratio_term",
            "enrichment_ratio",
            "effect_size",
            "expected_overlap",
            "leading_genes",
            "is_primary",
            "p_value",
            "fdr",
            "is_significant",
        ]
    )


def _prepare_term_records_for_ora(
    dataset_gene_universe: set[str],
    term_records: dict[str, dict],
    profile: AnnotationSourceProfile,
    cfg: ProgramAnnotationConfig,
) -> dict[str, dict]:
    prepared: dict[str, dict] = {}

    min_size = max(int(cfg.min_gene_set_size), int(profile.min_gene_set_size))
    max_size = min(int(cfg.max_gene_set_size), int(profile.max_gene_set_size))
    max_size = max(min_size, max_size)

    for term_id, record in term_records.items():
        genes_in_universe = set(record.get("genes", set())) & set(dataset_gene_universe)
        term_size = int(len(genes_in_universe))
        if term_size < min_size or term_size > max_size:
            continue
        prepared[str(term_id)] = {
            **record,
            "genes_in_universe": genes_in_universe,
            "term_size": term_size,
        }

    return prepared


def _run_ora_for_program(
    program_id: str,
    ranked_df: pd.DataFrame,
    prepared_term_records: dict[str, dict],
    dataset_gene_universe: set[str],
    source: str,
    cfg: ProgramAnnotationConfig,
) -> pd.DataFrame:
    if ranked_df.empty or not prepared_term_records or not dataset_gene_universe:
        return _empty_hits_table()

    ranked_df = (
        ranked_df.loc[:, ["gene", "gene_norm", "rank", "gene_contribution"]]
        .copy()
        .sort_values(["rank", "gene_norm"], ascending=[True, True])
        .drop_duplicates(subset=["gene_norm"], keep="first")
        .reset_index(drop=True)
    )
    program_genes = set(ranked_df["gene_norm"].astype(str).tolist()) & set(dataset_gene_universe)
    program_size = int(len(program_genes))
    background_size = int(len(dataset_gene_universe))
    if program_size <= 0 or background_size <= 0:
        return _empty_hits_table()

    gene_rank_map = ranked_df.set_index("gene_norm")["rank"].to_dict()
    gene_name_map = ranked_df.set_index("gene_norm")["gene"].astype(str).to_dict()

    rows: list[dict] = []
    for term_id, record in prepared_term_records.items():
        genes_in_universe = set(record.get("genes_in_universe", set()))
        overlap_genes = program_genes & genes_in_universe
        overlap_size = int(len(overlap_genes))
        if overlap_size < int(cfg.min_overlap):
            continue

        term_size = int(record.get("term_size", len(genes_in_universe)))
        expected_overlap = float(program_size * term_size / max(1, background_size))
        enrichment_ratio = float((overlap_size * background_size) / max(1, program_size * term_size))
        effect_size = float(max(0.0, math.log2(max(enrichment_ratio, 1e-12))))
        p_value = float(
            hypergeom.sf(overlap_size - 1, background_size, term_size, program_size)
        )
        ordered_overlap = sorted(overlap_genes, key=lambda g: (gene_rank_map.get(g, 10**9), g))
        leading_genes = [str(gene_name_map.get(g, g)) for g in ordered_overlap]

        rows.append(
            {
                "program_id": str(program_id),
                "annotation_source": str(source),
                "term_id": str(term_id),
                "term_name": str(record["term_name"]),
                "term_description": str(record["term_description"]),
                "term_category": str(record["term_category"]),
                "score": 0.0,
                "rank": 0,
                "overlap_size": int(overlap_size),
                "term_size": int(term_size),
                "program_size": int(program_size),
                "background_size": int(background_size),
                "overlap_ratio_program": float(overlap_size / max(1, program_size)),
                "overlap_ratio_term": float(overlap_size / max(1, term_size)),
                "enrichment_ratio": float(enrichment_ratio),
                "effect_size": float(effect_size),
                "expected_overlap": float(expected_overlap),
                "leading_genes": _join_pipe(leading_genes),
                "is_primary": False,
                "p_value": float(p_value),
                "fdr": 1.0,
                "is_significant": False,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_hits_table()
    out["fdr"] = benjamini_hochberg(out["p_value"].to_numpy(dtype=np.float64))
    significance_component = np.maximum(0.0, -np.log10(np.clip(out["fdr"].to_numpy(dtype=np.float64), 1e-12, 1.0)))
    effect_component = np.maximum(0.0, out["effect_size"].to_numpy(dtype=np.float64))
    overlap_component = np.sqrt(np.maximum(out["overlap_size"].to_numpy(dtype=np.float64), 1.0))
    out["score"] = effect_component * (1.0 + significance_component) * overlap_component
    out["is_significant"] = (
        (out["fdr"].to_numpy(dtype=np.float64) <= float(cfg.fdr_threshold))
        & (out["effect_size"].to_numpy(dtype=np.float64) > 0.0)
    )
    return out.sort_values(
        ["score", "fdr", "overlap_size", "term_id"], ascending=[False, True, False, True]
    ).reset_index(drop=True)


def _term_jaccard(term_a: str, term_b: str, term_records: dict[str, dict]) -> float:
    genes_a = set(term_records.get(str(term_a), {}).get("genes", set()))
    genes_b = set(term_records.get(str(term_b), {}).get("genes", set()))
    if not genes_a or not genes_b:
        return 0.0
    return float(len(genes_a & genes_b) / max(1, len(genes_a | genes_b)))


def _select_profile_term_ids(
    candidate_hits: pd.DataFrame,
    term_records: dict[str, dict],
    profile: AnnotationSourceProfile,
    cfg: ProgramAnnotationConfig,
) -> list[str]:
    if candidate_hits.empty:
        return []
    if "go_bp" == str(profile.source) and profile.trim_redundant_terms:
        threshold = float(cfg.go_redundancy_jaccard_threshold)
    else:
        threshold = float(profile.redundancy_jaccard_threshold)

    if not profile.trim_redundant_terms:
        return candidate_hits["term_id"].astype(str).tolist()

    selected_ids: list[str] = []
    for row in candidate_hits.itertuples(index=False):
        term_id = str(row.term_id)
        redundant = any(_term_jaccard(term_id, prev, term_records=term_records) >= threshold for prev in selected_ids)
        if not redundant:
            selected_ids.append(term_id)
    return selected_ids


def _postprocess_program_hits(
    hits_df: pd.DataFrame,
    term_records: dict[str, dict],
    profile: AnnotationSourceProfile,
    cfg: ProgramAnnotationConfig,
) -> pd.DataFrame:
    if hits_df.empty:
        return _empty_hits_table()

    out = hits_df.copy().sort_values(
        ["score", "fdr", "overlap_size", "term_id"], ascending=[False, True, False, True]
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, out.shape[0] + 1, dtype=np.int64)

    candidate = out.loc[out["is_significant"].astype(bool)].copy()
    selected_ids = _select_profile_term_ids(candidate, term_records=term_records, profile=profile, cfg=cfg)
    primary_id = selected_ids[0] if selected_ids else ""
    out["is_primary"] = out["term_id"].astype(str).eq(primary_id)
    return out


def _score_programs_against_gene_sets(
    ranked_tables: dict[str, pd.DataFrame],
    term_records: dict[str, dict],
    dataset_gene_universe: set[str],
    source: str,
    profile: AnnotationSourceProfile,
    cfg: ProgramAnnotationConfig,
) -> pd.DataFrame:
    prepared_term_records = _prepare_term_records_for_ora(
        dataset_gene_universe=dataset_gene_universe,
        term_records=term_records,
        profile=profile,
        cfg=cfg,
    )
    parts: list[pd.DataFrame] = []
    for pid in sorted(ranked_tables.keys()):
        raw_hits = _run_ora_for_program(
            program_id=pid,
            ranked_df=ranked_tables[pid],
            prepared_term_records=prepared_term_records,
            dataset_gene_universe=dataset_gene_universe,
            source=source,
            cfg=cfg,
        )
        parts.append(
            _postprocess_program_hits(
                hits_df=raw_hits,
                term_records=term_records,
                profile=profile,
                cfg=cfg,
            )
        )
    parts = [part for part in parts if not part.empty]
    if not parts:
        return _empty_hits_table()
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        return _empty_hits_table()
    return out.sort_values(["program_id", "rank", "score", "term_id"], ascending=[True, True, False, True]).reset_index(drop=True)


def _confidence_level(score: float) -> str:
    value = _clamp01(score)
    if value >= 0.70:
        return "strong"
    if value >= 0.40:
        return "moderate"
    return "weak"


def _summary_confidence(all_hits: pd.DataFrame, primary_score: float, cfg: ProgramAnnotationConfig) -> float:
    if all_hits.empty or primary_score <= 0.0:
        return 0.0
    positive = all_hits.loc[all_hits["score"].astype(float) > 0.0].copy()
    total_score = float(positive["score"].sum()) if not positive.empty else float(primary_score)
    primary_share = float(primary_score / total_score) if total_score > 0.0 else 0.0
    magnitude = _saturating_score(primary_score, cfg.confidence_score_scale)
    significant_count = int(all_hits["is_significant"].sum()) if "is_significant" in all_hits.columns else 0
    significance = float(min(1.0, significant_count / 2.0))
    return _clamp01(
        float(cfg.confidence_magnitude_weight) * magnitude
        + float(cfg.confidence_share_weight) * primary_share
        + float(cfg.confidence_significance_weight) * significance
    )


def _phrase_join(values: list[str]) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _source_display_name(source: str) -> str:
    return str(source).replace("_", " ").strip()


def _truncate_term_scores(
    significant_hits: pd.DataFrame,
    share_threshold: float,
) -> tuple[list[str], dict[str, float], float]:
    if significant_hits.empty:
        return [], {}, 0.0

    term_ids = significant_hits["term_id"].astype(str).tolist()
    scores = [max(_coerce_float(x, default=0.0), 0.0) for x in significant_hits["score"].tolist()]
    total_score = float(sum(scores))
    if total_score <= 0.0:
        return [], {}, 0.0

    target_share = _clamp01(share_threshold)
    kept_ids: list[str] = []
    kept_scores: dict[str, float] = {}
    cumulative = 0.0
    for term_id, score in zip(term_ids, scores, strict=False):
        if not term_id or score <= 0.0:
            continue
        kept_ids.append(term_id)
        kept_scores[term_id] = float(score)
        cumulative += float(score)
        if cumulative / total_score >= target_share:
            break

    retained_share = float(cumulative / total_score) if total_score > 0.0 else 0.0
    return kept_ids, kept_scores, retained_share


def _build_program_annotation_summary(
    program_ids: list[str],
    hits_table: pd.DataFrame,
    program_gene_table: pd.DataFrame,
    program_quality_table: pd.DataFrame,
    term_records: dict[str, dict],
    source: str,
    profile: AnnotationSourceProfile,
    cfg: ProgramAnnotationConfig,
) -> pd.DataFrame:
    quality_map = (
        {
            str(row["program_id"]): dict(row)
            for row in program_quality_table.to_dict(orient="records")
        }
        if not program_quality_table.empty and "program_id" in program_quality_table.columns
        else {}
    )

    rows: list[dict] = []
    for program_id in program_ids:
        pid = str(program_id)
        sub = hits_table.loc[hits_table["program_id"].astype(str) == pid].copy()
        sub = sub.sort_values(["rank", "score", "term_id"], ascending=[True, False, True]).reset_index(drop=True)

        significant_hits = sub.loc[sub["is_significant"].astype(bool)].copy()
        significant_hits = significant_hits.sort_values(
            ["score", "fdr", "overlap_size", "term_id"], ascending=[False, True, False, True]
        ).reset_index(drop=True)
        selected_term_ids = _select_profile_term_ids(
            significant_hits,
            term_records=term_records,
            profile=profile,
            cfg=cfg,
        )
        if selected_term_ids:
            significant_hits = (
                significant_hits.assign(
                    _term_order=lambda d: pd.Categorical(
                        d["term_id"].astype(str),
                        categories=selected_term_ids,
                        ordered=True,
                    )
                )
                .loc[lambda d: d["term_id"].astype(str).isin(selected_term_ids)]
                .sort_values(["_term_order", "score", "fdr"], ascending=[True, False, True])
                .drop(columns="_term_order")
                .reset_index(drop=True)
            )
        else:
            significant_hits = significant_hits.iloc[0:0].copy()

        significant_term_ids = significant_hits["term_id"].astype(str).tolist() if not significant_hits.empty else []
        displayed_term_ids, term_scores, displayed_term_score_share = _truncate_term_scores(
            significant_hits=significant_hits,
            share_threshold=cfg.summary_term_score_share_threshold,
        )

        gene_sub = (
            program_gene_table.loc[program_gene_table["program_id"].astype(str) == pid]
            .copy()
            .sort_values("rank", ascending=True)
        )
        top_genes = _join_pipe(
            gene_sub["gene"].astype(str).tolist()[: max(1, int(cfg.top_genes_for_summary))]
        )
        quality = quality_map.get(pid, {})

        primary_score = float(significant_hits.iloc[0]["score"]) if not significant_hits.empty else 0.0
        confidence = _summary_confidence(all_hits=sub, primary_score=primary_score, cfg=cfg)
        annotation_confidence_level = _confidence_level(confidence)
        source_display = _source_display_name(source)
        summary_terms = (
            significant_hits.loc[significant_hits["term_id"].astype(str).isin(displayed_term_ids), "term_name"]
            .astype(str)
            .tolist()
            if displayed_term_ids
            else []
        )
        if summary_terms:
            summary_text = (
                f"A {annotation_confidence_level} {source_display} annotation enriched for "
                f"{_phrase_join(summary_terms)}."
            )
        else:
            summary_text = f"No significant {source_display} enrichment detected."

        rows.append(
            {
                "program_id": pid,
                "routing_status": str(quality.get("routing_status", "")),
                "annotation_source": str(source),
                "significant_term_ids": list(significant_term_ids),
                "displayed_term_ids": list(displayed_term_ids),
                "term_scores": dict(term_scores),
                "displayed_term_score_share": float(displayed_term_score_share),
                "annotation_confidence": float(confidence),
                "annotation_confidence_level": str(annotation_confidence_level),
                "program_size": int(gene_sub["gene_norm"].nunique()) if not gene_sub.empty else int(quality.get("n_genes", 0)),
                "top_genes": str(top_genes),
                "n_significant_terms": int(len(significant_term_ids)),
                "n_display_terms": int(len(displayed_term_ids)),
                "program_confidence": _coerce_float(quality.get("program_confidence", 0.0), default=0.0),
                "summary_text": str(summary_text),
                "_primary_score_sort": float(primary_score),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "program_id",
                "routing_status",
                "annotation_source",
                "significant_term_ids",
                "displayed_term_ids",
                "term_scores",
                "displayed_term_score_share",
                "annotation_confidence",
                "annotation_confidence_level",
                "program_size",
                "top_genes",
                "n_significant_terms",
                "n_display_terms",
                "program_confidence",
                "summary_text",
                "_primary_score_sort",
            ]
        )

    level_order = {"strong": 0, "moderate": 1, "weak": 2}
    out["_level_order"] = out["annotation_confidence_level"].astype(str).map(level_order).fillna(9).astype(int)
    out = out.sort_values(
        ["_level_order", "annotation_confidence", "_primary_score_sort", "program_id"],
        ascending=[True, False, False, True],
    ).drop(columns=["_level_order", "_primary_score_sort"]).reset_index(drop=True)
    return out


def run_program_annotation(
    program_bundle_path: str | Path,
    gmt_file: str | Path | None = None,
    annotation_source: str | None = None,
    source_profile_yaml: str | Path = DEFAULT_SOURCE_PROFILE_YAML,
    out_dir: str | Path | None = None,
    config: ProgramAnnotationConfig | None = None,
) -> Path:
    cfg = config or ProgramAnnotationConfig()
    program_bundle = Path(program_bundle_path)
    source_profile_yaml_path = Path(source_profile_yaml)
    default_source, source_profiles = _load_source_profiles(source_profile_yaml_path)
    source = str(annotation_source or default_source).strip().lower()

    if source not in source_profiles:
        valid = ", ".join(sorted(source_profiles))
        raise ValueError(f"Unsupported annotation_source '{source}'. Expected one of: {valid}")
    profile = source_profiles[source]
    gmt_path = Path(gmt_file) if gmt_file else Path(profile.gmt_path)

    if not program_bundle.exists():
        raise FileNotFoundError(f"program_bundle not found: {program_bundle}")
    if not gmt_path.exists():
        raise FileNotFoundError(f"GMT file not found: {gmt_path}")

    programs_df, program_quality_table = _load_program_inputs(program_bundle=program_bundle)
    dataset_gene_universe, dataset_gene_universe_source = _resolve_dataset_gene_universe(
        program_bundle=program_bundle,
        programs_df=programs_df,
    )
    program_gene_table, ranked_tables = _build_program_gene_table(
        programs_df=programs_df,
        program_quality_table=program_quality_table,
        cfg=cfg,
    )
    program_ids = sorted(ranked_tables.keys())
    term_records = _read_gmt(path=gmt_path, source=source, profile=profile)

    hits_table = _score_programs_against_gene_sets(
        ranked_tables=ranked_tables,
        term_records=term_records,
        dataset_gene_universe=dataset_gene_universe,
        source=source,
        profile=profile,
        cfg=cfg,
    )
    summary_table = _build_program_annotation_summary(
        program_ids=program_ids,
        hits_table=hits_table,
        program_gene_table=program_gene_table,
        program_quality_table=program_quality_table,
        term_records=term_records,
        source=source,
        profile=profile,
        cfg=cfg,
    )

    if not program_quality_table.empty and "program_id" in program_quality_table.columns:
        status_cols = [c for c in ("program_id", "routing_status") if c in program_quality_table.columns]
        if len(status_cols) > 1:
            status_df = (
                program_quality_table.loc[:, status_cols]
                .copy()
                .drop_duplicates(subset=["program_id"])
            )
            hits_table = hits_table.merge(status_df, on="program_id", how="left")
            desired_cols = [
                "program_id",
                "routing_status",
                "annotation_source",
                "term_id",
                "term_name",
                "term_description",
                "term_category",
                "score",
                "rank",
                "overlap_size",
                "term_size",
                "program_size",
                "background_size",
                "overlap_ratio_program",
                "overlap_ratio_term",
                "enrichment_ratio",
                "effect_size",
                "expected_overlap",
                "leading_genes",
                "is_primary",
                "p_value",
                "fdr",
                "is_significant",
            ]
            for col in desired_cols:
                if col not in hits_table.columns:
                    hits_table[col] = ""
            hits_table = hits_table.loc[:, desired_cols]

    out_root = Path(out_dir) if out_dir else (program_bundle / "program_annotation")
    out_root.mkdir(parents=True, exist_ok=True)

    write_tsv(hits_table, out_root / "program_annotation_hits.tsv")
    write_json(out_root / "program_annotation_summary.json", summary_table.to_dict(orient="records"))
    raw_expression_meta: dict[str, object] | None = None
    if bool(cfg.emit_raw_expression_ora):
        try:
            raw_gene_table, raw_ranked_tables, raw_expression_meta, raw_gene_universe = _build_raw_expression_ranked_tables(
                program_bundle=program_bundle,
                program_quality_table=program_quality_table,
                cfg=cfg,
            )
            raw_hits_table = _score_programs_against_gene_sets(
                ranked_tables=raw_ranked_tables,
                term_records=term_records,
                dataset_gene_universe=raw_gene_universe,
                source=source,
                profile=profile,
                cfg=cfg,
            )
            raw_summary_table = _build_program_annotation_summary(
                program_ids=sorted(raw_ranked_tables.keys()),
                hits_table=raw_hits_table,
                program_gene_table=raw_gene_table,
                program_quality_table=program_quality_table,
                term_records=term_records,
                source=source,
                profile=profile,
                cfg=cfg,
            )
            if not program_quality_table.empty and "program_id" in program_quality_table.columns:
                status_cols = [c for c in ("program_id", "routing_status") if c in program_quality_table.columns]
                if len(status_cols) > 1 and not raw_hits_table.empty:
                    status_df = (
                        program_quality_table.loc[:, status_cols]
                        .copy()
                        .drop_duplicates(subset=["program_id"])
                    )
                    raw_hits_table = raw_hits_table.merge(status_df, on="program_id", how="left")
                    desired_cols = list(_empty_hits_table().columns)
                    for col in desired_cols:
                        if col not in raw_hits_table.columns:
                            raw_hits_table[col] = ""
                    raw_hits_table = raw_hits_table.loc[:, desired_cols]
            write_tsv(raw_hits_table, out_root / "program_annotation_raw_expression_hits.tsv")
            write_json(
                out_root / "program_annotation_raw_expression_summary.json",
                raw_summary_table.to_dict(orient="records"),
            )
            raw_expression_meta = {
                **(raw_expression_meta or {}),
                "status": "ok",
                "gene_universe_source": "raw_h5ad_var_names",
                "outputs": {
                    "program_annotation_raw_expression_hits": "program_annotation_raw_expression_hits.tsv",
                    "program_annotation_raw_expression_summary": "program_annotation_raw_expression_summary.json",
                },
                "n_hit_rows": int(raw_hits_table.shape[0]),
                "n_annotated_programs": int(raw_summary_table.shape[0]),
                "n_programs_with_significant_terms": int(
                    np.count_nonzero(raw_summary_table["n_significant_terms"].to_numpy(dtype=np.int64) > 0)
                ) if not raw_summary_table.empty and "n_significant_terms" in raw_summary_table.columns else 0,
            }
        except Exception as exc:  # noqa: BLE001
            raw_expression_meta = {
                "enabled": True,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    legacy_summary_tsv = out_root / "program_annotation_summary.tsv"
    if legacy_summary_tsv.exists():
        legacy_summary_tsv.unlink()

    meta = {
        "program_bundle_path": str(program_bundle.resolve()),
        "design_reference": "single-source GMT-first program annotation engine",
        "annotation_source": str(source),
        "source_profile_yaml": str(source_profile_yaml_path.resolve()),
        "gmt_file": str(gmt_path.resolve()),
        "dataset_context": {
            "gene_universe_size": int(len(dataset_gene_universe)),
            "gene_universe_source": str(dataset_gene_universe_source),
        },
        "methodology": {
            "annotation_positioning": (
                "Each run uses one authoritative GMT source and performs scoring only within that source."
            ),
            "scoring_logic": (
                "Programs are annotated with over-representation analysis against one authoritative GMT source; "
                "the exported score combines effect size, FDR-derived significance and overlap magnitude."
            ),
            "postprocess_logic": (
                "Source-specific postprocessing is applied after a common ORA pass; GO BP trims redundant parent-child-like terms by Jaccard."
            ),
            "output_contract": (
                "program_annotation_hits.tsv stores program-by-term ORA hits, and program_annotation_summary.json stores one stable per-program summary record per program."
            ),
        },
        "n_programs": int(len(program_ids)),
        "n_hit_rows": int(hits_table.shape[0]),
        "n_annotated_programs": int(summary_table.shape[0]),
        "n_programs_with_significant_terms": int(
            np.count_nonzero(summary_table["n_significant_terms"].to_numpy(dtype=np.int64) > 0)
        ) if not summary_table.empty and "n_significant_terms" in summary_table.columns else 0,
        "significant_hit_rows": int(hits_table["is_significant"].sum()) if "is_significant" in hits_table.columns else 0,
        "outputs": {
            "program_annotation_hits": "program_annotation_hits.tsv",
            "program_annotation_summary": "program_annotation_summary.json",
        },
        "raw_expression_ora": raw_expression_meta,
        "profile": asdict(profile),
        "config": asdict(cfg),
    }
    write_json(out_root / "annotation_meta.json", meta)
    return out_root


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-source GMT-first program annotation engine.")
    parser.add_argument("--program-bundle", type=str, default=None, help="Path to program_bundle directory.")
    parser.add_argument("--work-dir", type=str, default=str(DEFAULT_WORK_DIR), help=f"Root directory (default: {DEFAULT_WORK_DIR}).")
    parser.add_argument("--cancer", type=str, default=str(DEFAULT_CANCER), help=f"Cancer cohort (default: {DEFAULT_CANCER}).")
    parser.add_argument("--sample-id", type=str, default=str(DEFAULT_SAMPLE_ID), help=f"Sample id (default: {DEFAULT_SAMPLE_ID}).")
    parser.add_argument("--annotation-source", type=str, default=None, help="Single knowledge source profile. Defaults to the YAML default_source.")
    parser.add_argument("--source-profile-yaml", type=str, default=str(DEFAULT_SOURCE_PROFILE_YAML), help="YAML file that defines annotation sources and their GMT/profile settings.")
    parser.add_argument("--gmt-file", type=str, default=None, help="GMT file for the chosen annotation source.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <program_bundle>/program_annotation).")
    parser.add_argument("--top-genes-for-summary", type=int, default=10)
    parser.add_argument("--fdr-threshold", type=float, default=0.25)
    parser.add_argument("--min-overlap", type=int, default=3)
    parser.add_argument("--min-gene-set-size", type=int, default=10)
    parser.add_argument("--max-gene-set-size", type=int, default=2000)
    parser.add_argument("--go-redundancy-jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--random-seed", type=int, default=2024)
    return parser


def _resolve_program_bundle_path(args: argparse.Namespace) -> Path:
    if args.program_bundle:
        return Path(args.program_bundle)

    missing = []
    if not args.work_dir:
        missing.append("--work-dir")
    if not args.cancer:
        missing.append("--cancer")
    if not args.sample_id:
        missing.append("--sample-id")
    if missing:
        raise ValueError("Missing path arguments. Provide either --program-bundle or all of --work-dir/--cancer/--sample-id")

    return Path(args.work_dir) / str(args.cancer) / "ST" / str(args.sample_id) / "program_bundle"


def main() -> None:
    args = _build_arg_parser().parse_args()
    program_bundle = _resolve_program_bundle_path(args)
    out_dir = Path(args.out_dir) if args.out_dir else (program_bundle / "program_annotation")

    cfg = ProgramAnnotationConfig(
        top_genes_for_summary=int(args.top_genes_for_summary),
        fdr_threshold=float(args.fdr_threshold),
        min_overlap=int(args.min_overlap),
        min_gene_set_size=int(args.min_gene_set_size),
        max_gene_set_size=int(args.max_gene_set_size),
        go_redundancy_jaccard_threshold=float(args.go_redundancy_jaccard_threshold),
        random_seed=int(args.random_seed),
    )

    out = run_program_annotation(
        program_bundle_path=program_bundle,
        gmt_file=Path(args.gmt_file) if args.gmt_file else None,
        annotation_source=args.annotation_source,
        source_profile_yaml=args.source_profile_yaml,
        out_dir=out_dir,
        config=cfg,
    )
    print(f"[ok] program annotation output: {out}")


if __name__ == "__main__":
    main()
