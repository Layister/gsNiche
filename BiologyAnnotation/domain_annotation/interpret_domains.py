from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from BiologyAnnotation.bundle_io import read_json, write_json, write_tsv
else:
    from ..bundle_io import read_json, write_json, write_tsv


DEFAULT_WORK_DIR = Path("/Users/wuyang/Documents/SC-ST data")
DEFAULT_CANCER = "PRAD"
DEFAULT_SAMPLE_ID = "INT25"


@dataclass
class DomainAnnotationConfig:
    domains_relpath: str = "domains.parquet"
    program_annotation_relpath: str = "program_annotation/program_annotation_summary.json"
    output_dirname: str = "domain_annotation"
    eligible_require_qc_pass: bool = True
    eligible_exclude_background: bool = False
    align_niche_participation_rule: bool = True
    save_only_niche_participating_domains: bool = True
    low_confidence_reliability_threshold: float = 0.05
    moderate_confidence_reliability_threshold: float = 0.20
    high_confidence_reliability_threshold: float = 0.40
    min_dominant_annotation_term_score: float = 0.0
    dominant_annotation_term_gap_threshold: float = 0.10
    fragmented_components_min: int = 2
    branching_leaf_ratio_min: float = 0.05
    branching_articulation_ratio_min: float = 0.15
    elongated_elongation_min: float = 2.75
    diffuse_boundary_ratio_min: float = 0.80
    diffuse_internal_density_max: float = 0.15
    diffuse_compactness_max: float = 0.0025
    salience_peak_weight: float = 0.40
    salience_prominence_weight: float = 0.40
    salience_mean_weight: float = 0.20
    salience_strong_threshold: float = 0.67
    salience_weak_threshold: float = 0.33
    annotation_profile_summary_top_k: int = 3


REQUIRED_DOMAIN_COLUMNS: tuple[str, ...] = (
    "domain_id",
    "domain_key",
    "sample_id",
    "program_seed_id",
    "spot_count",
    "geo_compactness",
    "geo_boundary_ratio",
    "geo_elongation",
    "geo_leaf_ratio",
    "geo_articulation_ratio",
    "components_count",
    "internal_density",
    "prog_peak_value",
    "prog_prominence",
    "prog_seed_mean",
    "qc_pass",
    "is_background",
    "domain_reliability",
)


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    return bool(value)


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
    txt = txt.replace("_", " ").replace("/", " ").strip()
    txt = " ".join(txt.split())
    if not txt:
        return "Unresolved Annotation"
    return " ".join(word.capitalize() for word in txt.split())


def _ordered_annotation_terms(record: dict, compact: dict[str, float]) -> list[str]:
    ordered_ids = [
        str(x).strip()
        for x in (record.get("significant_term_ids") or [])
        if str(x).strip()
    ]
    if compact:
        if ordered_ids:
            ordered = [term_id for term_id in ordered_ids if term_id in compact]
            remaining = [
                term_id
                for term_id, _ in sorted(
                    compact.items(),
                    key=lambda kv: (-_coerce_float(kv[1], default=0.0), kv[0]),
                )
                if term_id not in ordered
            ]
            return ordered + remaining
        return [
            term_id
            for term_id, _ in sorted(
                compact.items(),
                key=lambda kv: (-_coerce_float(kv[1], default=0.0), kv[0]),
            )
        ]
    return []


def _summarize_annotation_profile(
    record: dict,
    compact: dict[str, float],
    top_k: int,
) -> tuple[list[str], list[str], float]:
    ordered_ids = _ordered_annotation_terms(record=record, compact=compact)
    ordered_names = [
        _normalize_term_name(term_id, str(record.get("annotation_source", "")))
        for term_id in ordered_ids
    ]
    peak_score = _coerce_float(compact.get(ordered_ids[0], 0.0), default=0.0) if ordered_ids else 0.0
    return ordered_ids, ordered_names, float(peak_score)


def _safe_percentile_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=np.float64)
    return series.rank(method="average", pct=True).astype(np.float64)


def _normalize_record_map(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for record in records:
        key = str(record.get("program_id", "")).strip()
        if key:
            out[key] = record
    return out


def _infer_program_annotation_summary(domain_bundle: Path, cfg: DomainAnnotationConfig) -> Path:
    manifest_path = domain_bundle / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing domain manifest: {manifest_path}")

    manifest = read_json(manifest_path)
    program_bundle_path = str(manifest.get("inputs", {}).get("program_bundle_path", "")).strip()
    if not program_bundle_path:
        raise ValueError(
            f"domain manifest missing inputs.program_bundle_path: {manifest_path}"
        )
    return Path(program_bundle_path) / cfg.program_annotation_relpath


def _load_domain_inputs(
    domain_bundle: Path,
    cfg: DomainAnnotationConfig,
    program_annotation_summary: Path | None,
) -> tuple[pd.DataFrame, list[dict], dict, Path]:
    domains_path = domain_bundle / cfg.domains_relpath
    if not domains_path.exists():
        raise FileNotFoundError(f"Missing domains parquet: {domains_path}")

    domains_df = pd.read_parquet(domains_path)
    missing = sorted(set(REQUIRED_DOMAIN_COLUMNS) - set(domains_df.columns))
    if missing:
        raise ValueError(f"domains.parquet missing required columns: {missing}")

    annotation_summary = program_annotation_summary or _infer_program_annotation_summary(
        domain_bundle=domain_bundle,
        cfg=cfg,
    )
    if not annotation_summary.exists():
        raise FileNotFoundError(f"Missing program annotation summary JSON: {annotation_summary}")

    program_records = read_json(annotation_summary)
    if not isinstance(program_records, list):
        raise ValueError(f"program annotation summary must be a list of records: {annotation_summary}")
    if any("program_id" not in record for record in program_records if isinstance(record, dict)):
        raise ValueError(f"program annotation summary missing program_id: {annotation_summary}")

    manifest_path = domain_bundle / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return domains_df.copy(), program_records, manifest, annotation_summary


def _prepare_domains(domains_df: pd.DataFrame) -> pd.DataFrame:
    out = domains_df.copy()
    text_cols = [
        "domain_id",
        "domain_key",
        "sample_id",
        "program_seed_id",
        "qc_reject_reasons",
        "screening_reject_tags",
        "screening_decision",
    ]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].astype(str)

    bool_cols = ["qc_pass", "is_background", "merged_child"]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].map(_coerce_bool)

    numeric_cols = [
        "spot_count",
        "geo_compactness",
        "geo_boundary_ratio",
        "geo_elongation",
        "geo_leaf_ratio",
        "geo_articulation_ratio",
        "components_count",
        "internal_density",
        "prog_peak_value",
        "prog_prominence",
        "prog_seed_mean",
        "prog_seed_sum",
        "domain_reliability",
        "geo_centroid_x",
        "geo_centroid_y",
        "geo_area_est",
        "internal_edge_count",
        "boundary_edge_count",
        "program_confidence_raw",
        "program_confidence_used",
        "program_confidence_weight",
        "domain_confidence_component",
        "domain_prominence_component",
        "domain_density_component",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _compute_eligibility(domains_df: pd.DataFrame, cfg: DomainAnnotationConfig) -> pd.DataFrame:
    out = domains_df.copy()
    eligible = np.ones(out.shape[0], dtype=bool)
    reasons: list[list[str]] = [[] for _ in range(out.shape[0])]

    # Domain annotation must cover every domain that can actually enter Niche.
    # NicheGraph currently keeps domains_df.loc[domains_df["qc_pass"], ...].
    require_qc_pass = bool(cfg.eligible_require_qc_pass) or bool(cfg.align_niche_participation_rule)
    if require_qc_pass:
        bad = ~out["qc_pass"].to_numpy(dtype=bool)
        eligible = eligible & ~bad
        for idx in np.flatnonzero(bad):
            reasons[int(idx)].append("qc_fail")

    if bool(cfg.eligible_exclude_background):
        bg = out["is_background"].to_numpy(dtype=bool)
        eligible = eligible & ~bg
        for idx in np.flatnonzero(bg):
            reasons[int(idx)].append("background")

    out["annotation_eligible"] = eligible
    out["annotation_skip_reasons"] = [";".join(vals) for vals in reasons]
    return out


def _annotate_annotation_term(
    domains_df: pd.DataFrame,
    program_record_map: dict[str, dict],
    cfg: DomainAnnotationConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    for row in domains_df.itertuples(index=False):
        pid = str(row.program_seed_id)
        record = program_record_map.get(pid, {})
        raw_vector = record.get("term_scores", {})
        if isinstance(raw_vector, str):
            try:
                compact = json.loads(raw_vector)
            except Exception:
                compact = {}
        elif isinstance(raw_vector, dict):
            compact = dict(raw_vector)
        else:
            compact = {}
        compact = {
            str(name): _coerce_float(score, default=0.0)
            for name, score in compact.items()
            if str(name).strip() and _coerce_float(score, default=0.0) > 0.0
        }
        ordered_ids, ordered_names, peak_score = _summarize_annotation_profile(
            record=record,
            compact=compact,
            top_k=cfg.annotation_profile_summary_top_k,
        )

        rows.append(
            {
                "domain_key": str(row.domain_key),
                "annotation_term_id_list": "|".join(ordered_ids),
                "annotation_term_name_list": "|".join(ordered_names),
                "annotation_term_profile_peak_score": float(peak_score),
                "annotation_term_count": int(len(ordered_ids)),
                "program_annotation_support_score": _coerce_float(
                    record.get("annotation_confidence", 0.0),
                    default=0.0,
                ),
                "program_annotation_support_level": str(record.get("annotation_confidence_level", "unknown")),
                "annotation_term_score_vector_compact": compact if isinstance(compact, dict) else {},
                "annotation_term_score_vector": compact if isinstance(compact, dict) else {},
                "program_annotation_source": str(record.get("annotation_source", "")),
                "program_routing_status": str(record.get("routing_status", "")),
                "program_annotation_program_confidence": _coerce_float(
                    record.get("program_confidence", 0.0),
                    default=0.0,
                ),
            }
        )
    return pd.DataFrame(rows)


def _annotate_morphology(domains_df: pd.DataFrame, cfg: DomainAnnotationConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for row in domains_df.itertuples(index=False):
        components_count = int(_coerce_float(getattr(row, "components_count", 1), default=1))
        elongation = _coerce_float(getattr(row, "geo_elongation", 0.0), default=0.0)
        leaf_ratio = _coerce_float(getattr(row, "geo_leaf_ratio", 0.0), default=0.0)
        articulation_ratio = _coerce_float(getattr(row, "geo_articulation_ratio", 0.0), default=0.0)
        boundary_ratio = _coerce_float(getattr(row, "geo_boundary_ratio", 0.0), default=0.0)
        internal_density = _coerce_float(getattr(row, "internal_density", 0.0), default=0.0)
        compactness = _coerce_float(getattr(row, "geo_compactness", 0.0), default=0.0)

        fragmented = components_count >= int(cfg.fragmented_components_min)
        branching = (
            leaf_ratio >= float(cfg.branching_leaf_ratio_min)
            or articulation_ratio >= float(cfg.branching_articulation_ratio_min)
        )
        elongated = elongation >= float(cfg.elongated_elongation_min)
        diffuse = (
            boundary_ratio >= float(cfg.diffuse_boundary_ratio_min)
            and internal_density <= float(cfg.diffuse_internal_density_max)
        ) or (
            compactness <= float(cfg.diffuse_compactness_max)
            and internal_density <= float(cfg.diffuse_internal_density_max)
        )

        if fragmented:
            morphology_label = "fragmented"
            primary_rule = "components_count"
        elif branching:
            morphology_label = "branching"
            primary_rule = "leaf_or_articulation_ratio"
        elif elongated:
            morphology_label = "elongated"
            primary_rule = "geo_elongation"
        elif diffuse:
            morphology_label = "diffuse"
            primary_rule = "boundary_density_compactness"
        else:
            morphology_label = "compact"
            primary_rule = "fallback_compact"

        rule_hits = []
        if fragmented:
            rule_hits.append("fragmented")
        if branching:
            rule_hits.append("branching")
        if elongated:
            rule_hits.append("elongated")
        if diffuse:
            rule_hits.append("diffuse")

        rows.append(
            {
                "domain_key": str(row.domain_key),
                "morphology_label": str(morphology_label),
                "morphology_primary_rule": str(primary_rule),
                "morphology_rule_hits": "|".join(rule_hits),
            }
        )
    return pd.DataFrame(rows)


def _annotate_salience(domains_df: pd.DataFrame, cfg: DomainAnnotationConfig) -> pd.DataFrame:
    eligible = domains_df.loc[domains_df["annotation_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "domain_key",
                "salience_label",
                "salience_score",
                "salience_peak_percentile",
                "salience_prominence_percentile",
                "salience_mean_percentile",
                "program_domain_count_for_salience",
            ]
        )

    parts: list[pd.DataFrame] = []
    for pid, sub in eligible.groupby("program_seed_id", sort=True):
        tmp = sub.loc[:, ["domain_key", "program_seed_id", "prog_peak_value", "prog_prominence", "prog_seed_mean"]].copy()
        tmp["salience_peak_percentile"] = _safe_percentile_rank(tmp["prog_peak_value"])
        tmp["salience_prominence_percentile"] = _safe_percentile_rank(tmp["prog_prominence"])
        tmp["salience_mean_percentile"] = _safe_percentile_rank(tmp["prog_seed_mean"])
        tmp["salience_score"] = (
            float(cfg.salience_peak_weight) * tmp["salience_peak_percentile"].to_numpy(dtype=np.float64)
            + float(cfg.salience_prominence_weight) * tmp["salience_prominence_percentile"].to_numpy(dtype=np.float64)
            + float(cfg.salience_mean_weight) * tmp["salience_mean_percentile"].to_numpy(dtype=np.float64)
        )

        score = tmp["salience_score"].to_numpy(dtype=np.float64)
        labels = np.full(tmp.shape[0], "moderate", dtype=object)
        labels[score >= float(cfg.salience_strong_threshold)] = "strong"
        labels[score < float(cfg.salience_weak_threshold)] = "weak"
        tmp["salience_label"] = labels
        tmp["program_domain_count_for_salience"] = int(tmp.shape[0])
        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True)
    return out.loc[
        :,
        [
            "domain_key",
            "salience_label",
            "salience_score",
            "salience_peak_percentile",
            "salience_prominence_percentile",
            "salience_mean_percentile",
            "program_domain_count_for_salience",
        ],
    ].copy()


def _annotate_confidence(domains_df: pd.DataFrame, cfg: DomainAnnotationConfig) -> pd.DataFrame:
    out = domains_df.loc[:, ["domain_key", "domain_reliability"]].copy()
    reliability = pd.to_numeric(out["domain_reliability"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    labels = np.full(out.shape[0], "moderate", dtype=object)
    labels[reliability <= float(cfg.low_confidence_reliability_threshold)] = "low"
    labels[reliability >= float(cfg.high_confidence_reliability_threshold)] = "high"

    moderate_hi = float(cfg.high_confidence_reliability_threshold)
    moderate_lo = float(cfg.moderate_confidence_reliability_threshold)
    if moderate_lo > moderate_hi:
        moderate_lo = moderate_hi
    labels[(reliability > moderate_lo) & (reliability < moderate_hi)] = "moderate"

    out["annotation_confidence_label"] = labels
    out["low_confidence"] = reliability <= float(cfg.low_confidence_reliability_threshold)
    return out.loc[:, ["domain_key", "annotation_confidence_label", "low_confidence"]].copy()


def _build_annotation_table(
    domains_df: pd.DataFrame,
    annotation_term_df: pd.DataFrame,
    morphology_df: pd.DataFrame,
    salience_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
) -> pd.DataFrame:
    out = domains_df.copy()
    for extra in (annotation_term_df, morphology_df, salience_df, confidence_df):
        if not extra.empty:
            out = out.merge(extra, on="domain_key", how="left", validate="one_to_one")

    out["salience_label"] = out["salience_label"].fillna("not_assessed")
    out["salience_score"] = pd.to_numeric(out["salience_score"], errors="coerce")
    out["annotation_confidence_label"] = out["annotation_confidence_label"].fillna("unknown")
    out["low_confidence"] = out["low_confidence"].fillna(False).map(_coerce_bool)
    out["morphology_label"] = out["morphology_label"].fillna("unresolved")
    out["annotation_term_count"] = pd.to_numeric(out.get("annotation_term_count", 0), errors="coerce").fillna(0).astype(int)

    out["domain_annotation_label"] = None
    eligible = out["annotation_eligible"].to_numpy(dtype=bool)
    profiled = (
        "profiled_"
        + out["morphology_label"].astype(str)
        + "_"
        + out["salience_label"].astype(str)
        + "_domain"
    )
    unresolved = (
        "unresolved_"
        + out["morphology_label"].astype(str)
        + "_"
        + out["salience_label"].astype(str)
        + "_domain"
    )
    has_terms = out["annotation_term_count"].to_numpy(dtype=np.int64) > 0
    out.loc[eligible & has_terms, "domain_annotation_label"] = profiled.loc[eligible & has_terms]
    out.loc[eligible & ~has_terms, "domain_annotation_label"] = unresolved.loc[eligible & ~has_terms]

    status = np.full(out.shape[0], "skipped", dtype=object)
    status[eligible & has_terms] = "annotated"
    status[eligible & ~has_terms] = "unresolved_program_annotation"
    out["annotation_status"] = status
    return out


def _json_ready(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        if not np.isfinite(value):
            return None
        return value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _build_domain_objective_data_records(annotation_df: pd.DataFrame, cfg: DomainAnnotationConfig) -> list[dict]:
    records: list[dict] = []
    for row in annotation_df.to_dict(orient="records"):
        records.append(
            {
                "domain_id": str(row.get("domain_id", "")),
                "domain_key": str(row.get("domain_key", "")),
                "domain_annotation_label": str(row.get("domain_annotation_label", "")),
                "objective_data": {
                    "domain_identity": {
                        "sample_id": str(row.get("sample_id", "")),
                        "program_seed_id": str(row.get("program_seed_id", "")),
                    },
                    "annotation_term": {
                        "annotation_term_id_list": [
                            x for x in str(row.get("annotation_term_id_list", "")).split("|") if x
                        ],
                        "annotation_term_name_list": [
                            x for x in str(row.get("annotation_term_name_list", "")).split("|") if x
                        ],
                        "annotation_source": str(row.get("program_annotation_source", "")),
                        "profile_peak_score": _json_ready(row.get("annotation_term_profile_peak_score")),
                        "annotation_term_count": _json_ready(row.get("annotation_term_count")),
                        "program_annotation_support_score": _json_ready(
                            row.get("program_annotation_support_score")
                        ),
                        "program_annotation_support_level": str(
                            row.get("program_annotation_support_level", "unknown")
                        ),
                        "annotation_term_score_vector_compact": row.get(
                            "annotation_term_score_vector_compact",
                            {},
                        ),
                        "annotation_term_score_vector": row.get("annotation_term_score_vector", {}),
                    },
                    "structure": {
                        "spot_count": _json_ready(row.get("spot_count")),
                        "centroid": {
                            "x": _json_ready(row.get("geo_centroid_x")),
                            "y": _json_ready(row.get("geo_centroid_y")),
                        },
                        "geometry": {
                            "geo_area_est": _json_ready(row.get("geo_area_est")),
                            "geo_compactness": _json_ready(row.get("geo_compactness")),
                            "geo_boundary_ratio": _json_ready(row.get("geo_boundary_ratio")),
                            "geo_elongation": _json_ready(row.get("geo_elongation")),
                            "geo_leaf_ratio": _json_ready(row.get("geo_leaf_ratio")),
                            "geo_articulation_ratio": _json_ready(
                                row.get("geo_articulation_ratio")
                            ),
                            "components_count": _json_ready(row.get("components_count")),
                        },
                        "graph": {
                            "internal_edge_count": _json_ready(row.get("internal_edge_count")),
                            "boundary_edge_count": _json_ready(row.get("boundary_edge_count")),
                            "internal_density": _json_ready(row.get("internal_density")),
                        },
                    },
                    "morphology": {
                        "morphology_label": str(row.get("morphology_label", "unresolved")),
                        "primary_rule": str(row.get("morphology_primary_rule", "unknown")),
                        "rule_hits": [
                            x for x in str(row.get("morphology_rule_hits", "")).split("|") if x
                        ],
                        "observed_metrics": {
                            "components_count": _json_ready(row.get("components_count")),
                            "geo_elongation": _json_ready(row.get("geo_elongation")),
                            "geo_leaf_ratio": _json_ready(row.get("geo_leaf_ratio")),
                            "geo_articulation_ratio": _json_ready(
                                row.get("geo_articulation_ratio")
                            ),
                            "geo_boundary_ratio": _json_ready(row.get("geo_boundary_ratio")),
                            "geo_compactness": _json_ready(row.get("geo_compactness")),
                            "internal_density": _json_ready(row.get("internal_density")),
                        },
                        "thresholds": {
                            "fragmented_components_min": int(cfg.fragmented_components_min),
                            "branching_leaf_ratio_min": float(cfg.branching_leaf_ratio_min),
                            "branching_articulation_ratio_min": float(
                                cfg.branching_articulation_ratio_min
                            ),
                            "elongated_elongation_min": float(cfg.elongated_elongation_min),
                            "diffuse_boundary_ratio_min": float(cfg.diffuse_boundary_ratio_min),
                            "diffuse_internal_density_max": float(
                                cfg.diffuse_internal_density_max
                            ),
                            "diffuse_compactness_max": float(cfg.diffuse_compactness_max),
                        },
                    },
                    "salience": {
                        "salience_label": str(row.get("salience_label", "not_assessed")),
                        "salience_score": _json_ready(row.get("salience_score")),
                        "program_domain_count_for_salience": _json_ready(
                            row.get("program_domain_count_for_salience")
                        ),
                        "within_program_percentiles": {
                            "peak": _json_ready(row.get("salience_peak_percentile")),
                            "prominence": _json_ready(row.get("salience_prominence_percentile")),
                            "mean_activation": _json_ready(row.get("salience_mean_percentile")),
                        },
                        "thresholds": {
                            "strong_threshold": float(cfg.salience_strong_threshold),
                            "weak_threshold": float(cfg.salience_weak_threshold),
                            "peak_weight": float(cfg.salience_peak_weight),
                            "prominence_weight": float(cfg.salience_prominence_weight),
                            "mean_weight": float(cfg.salience_mean_weight),
                        },
                    },
                    "confidence": {
                        "confidence_label": str(row.get("annotation_confidence_label", "unknown")),
                        "low_confidence": bool(row.get("low_confidence", False)),
                        "domain_reliability": _json_ready(row.get("domain_reliability")),
                        "reliability_components": {
                            "confidence": _json_ready(row.get("domain_confidence_component")),
                            "prominence": _json_ready(row.get("domain_prominence_component")),
                            "density": _json_ready(row.get("domain_density_component")),
                        },
                        "reliability_thresholds": {
                            "low": float(cfg.low_confidence_reliability_threshold),
                            "moderate": float(cfg.moderate_confidence_reliability_threshold),
                            "high": float(cfg.high_confidence_reliability_threshold),
                        },
                    },
                },
            }
        )
    return records


def _build_domain_biological_annotation_records(annotation_df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    keep_cols = [
        "domain_id",
        "domain_key",
        "sample_id",
        "program_seed_id",
        "domain_annotation_label",
        "annotation_term_id_list",
        "annotation_term_name_list",
        "annotation_term_score_vector_compact",
        "annotation_term_profile_peak_score",
        "annotation_term_count",
        "program_annotation_source",
        "morphology_label",
        "morphology_primary_rule",
        "salience_label",
        "salience_score",
        "annotation_confidence_label",
        "low_confidence",
        "domain_reliability",
        "spot_count",
    ]
    for row in annotation_df.to_dict(orient="records"):
        record = {}
        for col in keep_cols:
            value = row.get(col)
            record[col] = _json_ready(value)
        records.append(record)
    return records


def run_domain_annotation(
    domain_bundle_path: str | Path,
    program_annotation_summary: str | Path | None = None,
    out_dir: str | Path | None = None,
    config: DomainAnnotationConfig | None = None,
) -> Path:
    cfg = config or DomainAnnotationConfig()
    domain_bundle = Path(domain_bundle_path)
    if not domain_bundle.exists():
        raise FileNotFoundError(f"domain_bundle not found: {domain_bundle}")

    program_annotation_path = Path(program_annotation_summary) if program_annotation_summary else None
    domains_df, program_records, manifest, resolved_program_annotation_summary = _load_domain_inputs(
        domain_bundle=domain_bundle,
        cfg=cfg,
        program_annotation_summary=program_annotation_path,
    )
    domains_df = _prepare_domains(domains_df=domains_df)
    domains_df = _compute_eligibility(domains_df=domains_df, cfg=cfg)

    program_record_map = _normalize_record_map(program_records)
    annotation_term_df = _annotate_annotation_term(
        domains_df=domains_df,
        program_record_map=program_record_map,
        cfg=cfg,
    )
    morphology_df = _annotate_morphology(domains_df=domains_df, cfg=cfg)
    salience_df = _annotate_salience(domains_df=domains_df, cfg=cfg)
    confidence_df = _annotate_confidence(domains_df=domains_df, cfg=cfg)
    annotation_df = _build_annotation_table(
        domains_df=domains_df,
        annotation_term_df=annotation_term_df,
        morphology_df=morphology_df,
        salience_df=salience_df,
        confidence_df=confidence_df,
    )

    niche_participating_mask = annotation_df["qc_pass"].to_numpy(dtype=bool)
    niche_participating_count = int(np.sum(niche_participating_mask))
    annotated_count = int(np.sum(annotation_df["annotation_eligible"].to_numpy(dtype=bool)))
    if bool(cfg.align_niche_participation_rule) and annotated_count != niche_participating_count:
        raise RuntimeError(
            "Domain annotation invariant failed: annotated domain count does not match "
            f"Niche-participating domain count (annotated={annotated_count}, niche={niche_participating_count})."
        )

    objective_records = _build_domain_objective_data_records(annotation_df=annotation_df, cfg=cfg)
    biological_records = _build_domain_biological_annotation_records(annotation_df=annotation_df)

    saved_annotation_df = (
        annotation_df.loc[annotation_df["annotation_eligible"]].copy()
        if bool(cfg.save_only_niche_participating_domains)
        else annotation_df.copy()
    )
    saved_objective_records = (
        [r for r in objective_records if str(r.get("domain_key", "")) in set(saved_annotation_df["domain_key"].astype(str))]
        if bool(cfg.save_only_niche_participating_domains)
        else objective_records
    )
    saved_biological_records = (
        [r for r in biological_records if str(r.get("domain_key", "")) in set(saved_annotation_df["domain_key"].astype(str))]
        if bool(cfg.save_only_niche_participating_domains)
        else biological_records
    )

    out_root = Path(out_dir) if out_dir else (domain_bundle / cfg.output_dirname)
    out_root.mkdir(parents=True, exist_ok=True)

    table_cols = [
        "domain_id",
        "domain_key",
        "sample_id",
        "program_seed_id",
        "domain_annotation_label",
        "annotation_term_profile_peak_score",
        "annotation_term_count",
        "program_annotation_source",
        "morphology_label",
        "morphology_primary_rule",
        "salience_label",
        "salience_score",
        "annotation_confidence_label",
        "low_confidence",
        "domain_reliability",
        "spot_count",
    ]
    table_cols = [c for c in table_cols if c in saved_annotation_df.columns]
    write_tsv(saved_annotation_df.loc[:, table_cols].copy(), out_root / "domain_annotation_table.tsv")
    write_json(out_root / "domain_objective_data.json", saved_objective_records)
    write_json(out_root / "domain_biological_annotation.json", saved_biological_records)

    meta = {
        "domain_bundle_path": str(domain_bundle.resolve()),
        "program_annotation_summary": str(resolved_program_annotation_summary.resolve()),
        "design_reference": "rule-based domain annotation design driven by single-source program annotation",
        "methodology": {
            "semantic_scope": (
                "Domain annotation describes Program-driven basin structure rather than tissue-level ecology."
            ),
            "annotation_term_source": (
                "Program-level annotation term profiles are propagated from program_annotation_summary.json using program_seed_id."
            ),
            "morphology_logic": (
                "Morphology is classified by fixed rules over connected components, elongation, "
                "leaf/articulation ratios, boundary ratio, compactness and internal density."
            ),
            "salience_logic": (
                "Salience is computed within each program from percentile-ranked peak value, prominence "
                "and mean activation."
            ),
            "confidence_logic": (
                "Low-confidence marking follows domain_reliability thresholds aligned with downstream "
                "NicheGraph usage."
            ),
        },
        "n_input_domains": int(annotation_df.shape[0]),
        "n_eligible_domains": int(annotation_df["annotation_eligible"].sum()),
        "n_annotated_domains": int((annotation_df["annotation_status"] == "annotated").sum()),
        "n_niche_participating_domains": int(niche_participating_count),
        "annotation_covers_all_niche_domains": bool(annotated_count == niche_participating_count),
        "save_only_niche_participating_domains": bool(cfg.save_only_niche_participating_domains),
        "n_saved_domains": int(saved_annotation_df.shape[0]),
        "label_counts": saved_annotation_df["domain_annotation_label"].value_counts(dropna=True).to_dict(),
        "morphology_counts": saved_annotation_df["morphology_label"].value_counts(dropna=False).to_dict(),
        "salience_counts": saved_annotation_df["salience_label"].value_counts(dropna=False).to_dict(),
        "confidence_counts": saved_annotation_df["annotation_confidence_label"].value_counts(dropna=False).to_dict(),
        "program_annotation_record_count": int(len(program_records)),
        "outputs": {
            "domain_annotation_table": "domain_annotation_table.tsv",
            "domain_objective_data": "domain_objective_data.json",
            "domain_biological_annotation": "domain_biological_annotation.json",
        },
        "domain_bundle_manifest_summary": {
            "sample_id": manifest.get("sample_id"),
            "schema_version": manifest.get("schema_version"),
            "final_domain_stage": manifest.get("inputs", {}).get("final_domain_stage"),
        },
        "config": asdict(cfg),
    }
    write_json(out_root / "annotation_meta.json", meta)
    return out_root


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based domain biological annotation.")
    parser.add_argument("--domain-bundle", type=str, default=None, help="Path to domain_bundle directory.")
    parser.add_argument("--program-annotation-summary", type=str, default=None, help="Optional explicit path to program_annotation_summary.json.")
    parser.add_argument("--work-dir", type=str, default=str(DEFAULT_WORK_DIR), help=f"Root directory (default: {DEFAULT_WORK_DIR}).")
    parser.add_argument("--cancer", type=str, default=str(DEFAULT_CANCER), help=f"Cancer cohort (default: {DEFAULT_CANCER}).")
    parser.add_argument("--sample-id", type=str, default=str(DEFAULT_SAMPLE_ID), help=f"Sample id (default: {DEFAULT_SAMPLE_ID}).")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <domain_bundle>/domain_annotation).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    domain_bundle = (
        Path(args.domain_bundle)
        if args.domain_bundle
        else Path(args.work_dir) / args.cancer / "ST" / args.sample_id / "domain_bundle"
    )
    out_dir = Path(args.out_dir) if args.out_dir else None

    out = run_domain_annotation(
        domain_bundle_path=domain_bundle,
        program_annotation_summary=args.program_annotation_summary,
        out_dir=out_dir,
        config=DomainAnnotationConfig(),
    )
    print(f"[ok] domain annotation output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
