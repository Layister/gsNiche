from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle_io import (
    ensure_bundle_dirs,
    get_code_version,
    hash_array,
    hash_file,
    iso_now,
    promote_bundle,
    set_random_seed,
    write_json,
    write_parquet,
)
from .data_prep import load_domain_inputs
from .domain_ops import (
    _select_program_active_mask,
    _smooth_on_graph,
    build_domain_graph_table,
    build_domain_membership_table,
    compute_domain_geometry_metrics,
    extract_candidate_domains,
    propose_program_merge_groups,
)
from .qc_ops import build_program_domain_summary_table, build_qc_report
from .schema import (
    DomainAdjacencyConfig,
    DomainFilterConfig,
    DomainInputConfig,
    DomainMergeConfig,
    DomainPipelineConfig,
    DomainQCConfig,
    DomainReliabilityConfig,
    PotentialConfig,
    ProgramConfidenceConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DomainInputConfig",
    "PotentialConfig",
    "DomainFilterConfig",
    "DomainQCConfig",
    "DomainAdjacencyConfig",
    "DomainMergeConfig",
    "ProgramConfidenceConfig",
    "DomainReliabilityConfig",
    "DomainPipelineConfig",
    "run_domain_pipeline",
]


def _stable_domain_key(sample_id: str, program_id: str, spot_ids_subset: np.ndarray) -> str:
    ordered = sorted(str(x) for x in spot_ids_subset.tolist())
    payload = f"{sample_id}|{program_id}|{'|'.join(ordered)}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"{sample_id}:{program_id}:{digest}"


def _uniq_reasons(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for r in values:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _build_merged_domains(
    sample_id: str,
    spot_ids: np.ndarray,
    domains_all: list[dict],
    merge_groups: list[list[int]],
    dense_activation: np.ndarray,
    program_ids: np.ndarray,
    adjacency: list[set[int]],
    coords: np.ndarray | None,
    potential_cfg: PotentialConfig,
    filter_cfg: DomainFilterConfig,
    program_weight_info: dict[str, dict[str, float]] | None = None,
) -> tuple[list[dict], list[dict]]:
    if not merge_groups:
        return [], []

    pid_to_col = {str(pid): i for i, pid in enumerate(program_ids)}
    n_spots = int(dense_activation.shape[0])
    base_mask = np.ones(n_spots, dtype=bool)
    program_field_cache: dict[str, dict] = {}
    merged_domains: list[dict] = []
    merge_log_rows: list[dict] = []
    flow_graph_mode = str(potential_cfg.flow_graph_mode)
    bridge_gap_effective = int(max(0, potential_cfg.bridge_gap_spots)) if flow_graph_mode == "spatial" else 0
    program_weight_info = program_weight_info or {}

    for gid, group in enumerate(merge_groups, start=1):
        members = [domains_all[i] for i in group]
        pid = str(members[0]["program_seed_id"])
        if any(str(m["program_seed_id"]) != pid for m in members):
            continue
        col = pid_to_col.get(pid)
        if col is None:
            continue

        if pid not in program_field_cache:
            raw = np.asarray(dense_activation[:, col], dtype=np.float32)
            prog_active_mask, active_floor, program_scale = _select_program_active_mask(
                raw=raw,
                base_mask=base_mask,
                potential_cfg=potential_cfg,
            )
            smooth = _smooth_on_graph(
                values=raw,
                adjacency=adjacency,
                active_mask=prog_active_mask,
                cfg=potential_cfg,
            )
            active_idx = np.flatnonzero(prog_active_mask)
            global_baseline = float(np.quantile(smooth[active_idx], 0.50)) if active_idx.size > 0 else 0.0
            global_baseline_raw = float(np.quantile(raw[active_idx], 0.50)) if active_idx.size > 0 else 0.0
            program_field_cache[pid] = {
                "raw": raw,
                "smooth": smooth,
                "prog_active_mask": prog_active_mask,
                "global_baseline": global_baseline,
                "global_baseline_raw": global_baseline_raw,
                "active_floor": float(active_floor),
                "program_scale": float(program_scale),
            }

        field = program_field_cache[pid]
        raw = np.asarray(field["raw"], dtype=np.float32)
        smooth = np.asarray(field["smooth"], dtype=np.float32)
        prog_active_mask = np.asarray(field["prog_active_mask"], dtype=bool)
        global_baseline = float(field["global_baseline"])
        global_baseline_raw = float(field["global_baseline_raw"])
        active_floor = float(field["active_floor"])
        program_scale = float(field["program_scale"])

        merged_spots = np.unique(
            np.concatenate([np.asarray(m["spot_indices"], dtype=np.int32) for m in members])
        ).astype(np.int32, copy=False)
        if merged_spots.size == 0:
            continue

        domain_key = _stable_domain_key(sample_id=sample_id, program_id=pid, spot_ids_subset=spot_ids[merged_spots])
        raw_vals = raw[merged_spots]
        smooth_vals = smooth[merged_spots]
        peak_value = float(np.max(smooth_vals)) if smooth_vals.size > 0 else 0.0
        mean_activation = float(np.mean(raw_vals)) if raw_vals.size > 0 else 0.0
        sum_activation = float(np.sum(raw_vals)) if raw_vals.size > 0 else 0.0

        spot_set = set(int(x) for x in merged_spots.tolist())
        boundary_out: list[float] = []
        boundary_out_raw: list[float] = []
        for u in spot_set:
            for v in adjacency[u]:
                if (not prog_active_mask[v]) or (v in spot_set):
                    continue
                boundary_out.append(float(smooth[v]))
                boundary_out_raw.append(float(raw[v]))
        outside_q = (
            float(np.quantile(np.asarray(boundary_out, dtype=np.float32), float(filter_cfg.prominence_outside_quantile)))
            if boundary_out
            else global_baseline
        )
        prominence = float(max(0.0, peak_value - outside_q))
        outside_raw_mean = (
            float(np.mean(np.asarray(boundary_out_raw, dtype=np.float32))) if boundary_out_raw else global_baseline_raw
        )
        mean_enrichment_ratio = float(mean_activation / (outside_raw_mean + 1e-8))
        mean_enrichment_delta = float(mean_activation - outside_raw_mean)
        p_weight = program_weight_info.get(pid, {})
        p_conf_raw = float(p_weight.get("program_confidence_raw", 1.0))
        p_conf_used = float(p_weight.get("program_confidence_used", 1.0))
        p_conf_weight = float(p_weight.get("program_confidence_weight", 1.0))

        geom = compute_domain_geometry_metrics(merged_spots, adjacency=adjacency, coords=coords)
        merged_domain = {
            "domain_id": "",
            "domain_key": domain_key,
            "sample_id": sample_id,
            "program_seed_id": pid,
            "spot_indices": merged_spots,
            "spot_count": int(merged_spots.size),
            "geo_centroid_x": float(geom["geo_centroid_x"]),
            "geo_centroid_y": float(geom["geo_centroid_y"]),
            "geo_area_est": float(geom["geo_area_est"]),
            "geo_compactness": float(geom["geo_compactness"]),
            "geo_boundary_ratio": float(geom["geo_boundary_ratio"]),
            "geo_elongation": float(geom["geo_elongation"]),
            "geo_leaf_ratio": float(geom["geo_leaf_ratio"]),
            "geo_articulation_ratio": float(geom["geo_articulation_ratio"]),
            "components_count": int(geom.get("components_count", 1)),
            "internal_edge_count": int(geom["internal_edge_count"]),
            "boundary_edge_count": int(geom["boundary_edge_count"]),
            "internal_density": float(geom["internal_density"]),
            "prog_seed_mean": mean_activation,
            "prog_seed_sum": sum_activation,
            "prog_mean_enrichment_ratio": mean_enrichment_ratio,
            "prog_mean_enrichment_delta": mean_enrichment_delta,
            "prog_peak_value": peak_value,
            "prog_prominence": prominence,
            "prog_scale_median": float(program_scale),
            "program_confidence_raw": float(p_conf_raw),
            "program_confidence_used": float(p_conf_used),
            "program_confidence_weight": float(p_conf_weight),
            "flow_graph_mode": flow_graph_mode,
            "seg_active_floor": float(active_floor),
            "bridge_gap_spots_effective": int(bridge_gap_effective),
            "thr_min_domain_spots_effective": int(
                np.max([int(m.get("thr_min_domain_spots_effective", 0)) for m in members])
            ),
            "thr_min_domain_internal_density_effective": float(
                np.max([float(m.get("thr_min_domain_internal_density_effective", 0.0)) for m in members])
            ),
            "qc_pass": True,
            "qc_reject_reasons": [],
            "screening_reject_tags": [],
            "screening_decision": "kept",
            "is_background": False,
            "screening_pass": True,
            "merged_from_domain_keys": [str(m["domain_key"]) for m in members],
            "merged_group_size": int(len(members)),
        }
        merged_domains.append(merged_domain)

        for idx in group:
            child = domains_all[idx]
            child["qc_pass"] = False
            rr = list(child.get("qc_reject_reasons", []))
            rr.append(f"merged_into:{domain_key}")
            child["qc_reject_reasons"] = _uniq_reasons(rr)
            tt = list(child.get("screening_reject_tags", []))
            tt.append("rejected_by_merge")
            child["screening_reject_tags"] = _uniq_reasons(tt)
            child["screening_decision"] = "rejected"
            child["is_background"] = True
            child["merged_child"] = True
            merge_log_rows.append(
                {
                    "merge_group_id": int(gid),
                    "program_seed_id": pid,
                    "merged_domain_key": domain_key,
                    "child_domain_key": str(child["domain_key"]),
                    "child_spot_count": int(child.get("spot_count", 0)),
                }
            )

    return merged_domains, merge_log_rows


def _assign_domain_ids(domains: list[dict]) -> None:
    ordered = sorted(
        domains,
        key=lambda d: (
            str(d.get("program_seed_id", "")),
            -int(d.get("spot_count", 0)),
            str(d.get("domain_key", "")),
        ),
    )
    for i, d in enumerate(ordered, start=1):
        d["domain_id"] = f"D{i:06d}"


def _attach_domain_reliability(
    domains_df: pd.DataFrame,
    cfg: DomainPipelineConfig,
) -> tuple[pd.DataFrame, dict]:
    out = domains_df.copy()
    rel_cfg = cfg.domain_reliability

    if out.empty:
        out["domain_reliability"] = np.array([], dtype=np.float32)
        out["domain_confidence_component"] = np.array([], dtype=np.float32)
        out["domain_prominence_component"] = np.array([], dtype=np.float32)
        out["domain_density_component"] = np.array([], dtype=np.float32)
        return out, {
            "enabled": bool(rel_cfg.enabled),
            "prominence_scale": float("nan"),
            "density_scale": float(max(1e-8, rel_cfg.density_scale)),
            "reliability_min": float("nan"),
            "reliability_max": float("nan"),
            "reliability_mean": float("nan"),
            "low_reliability_frac": 0.0,
        }

    if not bool(rel_cfg.enabled):
        out["domain_reliability"] = np.ones(out.shape[0], dtype=np.float32)
        out["domain_confidence_component"] = np.ones(out.shape[0], dtype=np.float32)
        out["domain_prominence_component"] = np.ones(out.shape[0], dtype=np.float32)
        out["domain_density_component"] = np.ones(out.shape[0], dtype=np.float32)
        return out, {
            "enabled": False,
            "prominence_scale": float("nan"),
            "density_scale": float(max(1e-8, rel_cfg.density_scale)),
            "reliability_min": 1.0,
            "reliability_max": 1.0,
            "reliability_mean": 1.0,
            "low_reliability_frac": 0.0,
        }

    required = ["program_confidence_weight", "prog_prominence", "internal_density"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            "Domain reliability requires columns missing from domains table: "
            f"{missing}. This is an internal DomainBuilder error."
        )

    conf = pd.to_numeric(out["program_confidence_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    conf = np.clip(conf, 0.0, 1.0)

    prom = pd.to_numeric(out["prog_prominence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    prom = np.clip(prom, 0.0, None)
    prom_pos = prom[prom > 0]
    q = float(np.clip(rel_cfg.prominence_scale_quantile, 0.0, 1.0))
    prom_scale = float(np.quantile(prom_pos, q)) if prom_pos.size > 0 else 1.0
    prom_scale = max(1e-8, prom_scale)
    prom_norm = np.clip(prom / prom_scale, 0.0, 1.0)

    density = pd.to_numeric(out["internal_density"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    density = np.clip(density, 0.0, None)
    density_scale = float(max(1e-8, rel_cfg.density_scale))
    density_norm = np.clip(density / density_scale, 0.0, 1.0)

    a = float(max(0.0, rel_cfg.confidence_exponent))
    b = float(max(0.0, rel_cfg.prominence_exponent))
    c = float(max(0.0, rel_cfg.density_exponent))
    raw = (np.power(conf, a) if a > 0 else np.ones_like(conf)) * (
        np.power(prom_norm, b) if b > 0 else np.ones_like(prom_norm)
    ) * (
        np.power(density_norm, c) if c > 0 else np.ones_like(density_norm)
    )

    lo = float(np.clip(rel_cfg.min_node_reliability, 0.0, 1.0))
    hi = float(np.clip(rel_cfg.max_node_reliability, lo, 1.0))
    rel = np.clip(np.asarray(raw, dtype=np.float64), lo, hi)

    out["domain_reliability"] = rel.astype(np.float32)
    out["domain_confidence_component"] = conf.astype(np.float32)
    out["domain_prominence_component"] = prom_norm.astype(np.float32)
    out["domain_density_component"] = density_norm.astype(np.float32)

    return out, {
        "enabled": True,
        "prominence_scale": float(prom_scale),
        "density_scale": float(density_scale),
        "reliability_min": float(np.min(rel)),
        "reliability_max": float(np.max(rel)),
        "reliability_mean": float(np.mean(rel)),
        "low_reliability_frac": float(np.mean(rel <= lo + 1e-8)),
    }


def refine_domains(
    candidate_domains: list[dict],
    sample_id: str,
    spot_ids: np.ndarray,
    cfg: DomainPipelineConfig,
    dense_activation: np.ndarray,
    program_ids: np.ndarray,
    program_weight_info: dict[str, dict[str, float]] | None,
    adjacency: list[set[int]],
    coords: np.ndarray | None,
) -> dict:
    domains_all, _ = _finalize_domains(
        candidate_domains=candidate_domains,
        sample_id=sample_id,
        spot_ids=spot_ids,
        cfg=cfg,
    )

    merge_summary = {
        "enabled": bool(cfg.merge.enabled),
        "merge_group_count": 0,
        "merged_domain_count": 0,
        "edge_scan_count": 0,
        "edge_pass_count": 0,
    }
    merge_edge_log = pd.DataFrame(
        columns=[
            "program_seed_id",
            "domain_key_i",
            "domain_key_j",
            "shared_boundary_edges",
            "centroid_distance",
            "peak_ratio",
            "prominence_ratio",
            "edge_pass",
        ]
    )
    merge_child_log = pd.DataFrame(
        columns=[
            "merge_group_id",
            "program_seed_id",
            "merged_domain_key",
            "child_domain_key",
            "child_spot_count",
        ]
    )
    if bool(cfg.merge.enabled):
        merge_groups, merge_edge_log = propose_program_merge_groups(
            domains=domains_all,
            adjacency=adjacency,
            cfg=cfg.merge,
        )
        merge_summary["merge_group_count"] = int(len(merge_groups))
        merge_summary["edge_scan_count"] = int(merge_edge_log.shape[0])
        merge_summary["edge_pass_count"] = (
            int(np.sum(merge_edge_log["edge_pass"].to_numpy(dtype=bool))) if not merge_edge_log.empty else 0
        )
        merged_domains, merge_rows = _build_merged_domains(
            sample_id=sample_id,
            spot_ids=spot_ids,
            domains_all=domains_all,
            merge_groups=merge_groups,
            dense_activation=dense_activation,
            program_ids=program_ids,
            adjacency=adjacency,
            coords=coords,
            potential_cfg=cfg.potential,
            filter_cfg=cfg.filter,
            program_weight_info=program_weight_info,
        )
        if merge_rows:
            merge_child_log = pd.DataFrame(merge_rows)
        if merged_domains:
            domains_all.extend(merged_domains)
            merge_summary["merged_domain_count"] = int(len(merged_domains))

    _assign_domain_ids(domains_all)
    accepted_domains = [d for d in domains_all if bool(d.get("qc_pass", False))]

    return {
        "final_stage": "refine_domains.v1",
        "domains_all": domains_all,
        "accepted_domains": accepted_domains,
        "merge_summary": merge_summary,
        "merge_edge_log": merge_edge_log,
        "merge_child_log": merge_child_log,
    }


def _finalize_domains(
    candidate_domains: list[dict],
    sample_id: str,
    spot_ids: np.ndarray,
    cfg: DomainPipelineConfig,
) -> tuple[list[dict], list[dict]]:
    domains_out: list[dict] = []
    accepted_domains: list[dict] = []

    for idx, d in enumerate(candidate_domains, start=1):
        reasons = list(d.get("screening_reasons", []))
        screening_tags = list(d.get("screening_reject_tags", []))
        reasons = _uniq_reasons(reasons)
        screening_tags = _uniq_reasons(screening_tags)
        qc_pass = len(reasons) == 0

        spot_idx = np.asarray(d["spot_indices"], dtype=np.int32)
        domain_key = _stable_domain_key(
            sample_id=sample_id,
            program_id=str(d["program_id"]),
            spot_ids_subset=spot_ids[spot_idx],
        )
        domain = {
            "domain_id": f"D{idx:06d}",
            "domain_key": domain_key,
            "sample_id": sample_id,
            "program_seed_id": str(d["program_id"]),
            "spot_indices": spot_idx,
            "spot_count": int(d["spot_count"]),
            "geo_centroid_x": float(d["geo_centroid_x"]),
            "geo_centroid_y": float(d["geo_centroid_y"]),
            "geo_area_est": float(d["geo_area_est"]),
            "geo_compactness": float(d["geo_compactness"]),
            "geo_boundary_ratio": float(d["geo_boundary_ratio"]),
            "geo_elongation": float(d.get("geo_elongation", float("nan"))),
            "geo_leaf_ratio": float(d.get("geo_leaf_ratio", float("nan"))),
            "geo_articulation_ratio": float(d.get("geo_articulation_ratio", float("nan"))),
            "components_count": int(d.get("components_count", 1)),
            "internal_edge_count": int(d["internal_edge_count"]),
            "boundary_edge_count": int(d["boundary_edge_count"]),
            "internal_density": float(d["internal_density"]),
            "prog_seed_mean": float(d["prog_seed_mean"]),
            "prog_seed_sum": float(d["prog_seed_sum"]),
            "prog_mean_enrichment_ratio": float(d.get("prog_mean_enrichment_ratio", 0.0)),
            "prog_mean_enrichment_delta": float(d.get("prog_mean_enrichment_delta", 0.0)),
            "prog_peak_value": float(d["prog_peak_value"]),
            "prog_prominence": float(d["prog_prominence"]),
            "prog_scale_median": float(d.get("program_scale", 0.0)),
            "program_confidence_raw": float(d.get("program_confidence_raw", 1.0)),
            "program_confidence_used": float(d.get("program_confidence_used", 1.0)),
            "program_confidence_weight": float(d.get("program_confidence_weight", 1.0)),
            "flow_graph_mode": str(d.get("flow_graph_mode", str(cfg.potential.flow_graph_mode))),
            "seg_active_floor": float(d.get("active_floor", 0.0)),
            "bridge_gap_spots_effective": int(d.get("bridge_gap_spots_effective", 0)),
            "thr_min_domain_spots_effective": int(d.get("min_domain_spots_effective", 0)),
            "thr_min_domain_internal_density_effective": float(
                d.get("min_domain_internal_density_effective", 0.0)
            ),
            "qc_pass": bool(qc_pass),
            "qc_reject_reasons": reasons,
            "screening_reject_tags": screening_tags,
            "screening_decision": "kept" if qc_pass else "rejected",
            "is_background": bool(not qc_pass),
            "screening_pass": bool(d["screening_pass"]),
        }
        domains_out.append(domain)
        if qc_pass:
            accepted_domains.append(domain)

    return domains_out, accepted_domains


def _domain_rows_for_parquet(domains: list[dict]) -> pd.DataFrame:
    rows = []
    for d in domains:
        rows.append(
            {
                "domain_id": d["domain_id"],
                "domain_key": d["domain_key"],
                "sample_id": d["sample_id"],
                "program_seed_id": d["program_seed_id"],
                "spot_count": d["spot_count"],
                "geo_centroid_x": d["geo_centroid_x"],
                "geo_centroid_y": d["geo_centroid_y"],
                "geo_area_est": d["geo_area_est"],
                "geo_compactness": d["geo_compactness"],
                "geo_boundary_ratio": d["geo_boundary_ratio"],
                "geo_elongation": d.get("geo_elongation", float("nan")),
                "geo_leaf_ratio": d.get("geo_leaf_ratio", float("nan")),
                "geo_articulation_ratio": d.get("geo_articulation_ratio", float("nan")),
                "components_count": int(d.get("components_count", 1)),
                "internal_edge_count": d["internal_edge_count"],
                "boundary_edge_count": d["boundary_edge_count"],
                "internal_density": d["internal_density"],
                "prog_seed_mean": d["prog_seed_mean"],
                "prog_seed_sum": d["prog_seed_sum"],
                "prog_mean_enrichment_ratio": d.get("prog_mean_enrichment_ratio", 0.0),
                "prog_mean_enrichment_delta": d.get("prog_mean_enrichment_delta", 0.0),
                "prog_peak_value": d["prog_peak_value"],
                "prog_prominence": d["prog_prominence"],
                "prog_scale_median": d["prog_scale_median"],
                "program_confidence_raw": d.get("program_confidence_raw", 1.0),
                "program_confidence_used": d.get("program_confidence_used", 1.0),
                "program_confidence_weight": d.get("program_confidence_weight", 1.0),
                "flow_graph_mode": d.get("flow_graph_mode", str("unknown")),
                "seg_active_floor": d["seg_active_floor"],
                "bridge_gap_spots_effective": int(d.get("bridge_gap_spots_effective", 0)),
                "thr_min_domain_spots_effective": d["thr_min_domain_spots_effective"],
                "thr_min_domain_internal_density_effective": d["thr_min_domain_internal_density_effective"],
                "qc_pass": d["qc_pass"],
                "qc_reject_reasons": ";".join(d["qc_reject_reasons"]),
                "screening_reject_tags": ";".join(d.get("screening_reject_tags", [])),
                "screening_decision": str(d.get("screening_decision", "unknown")),
                "is_background": d["is_background"],
                "merged_group_size": int(d.get("merged_group_size", 0)),
                "merged_from_domain_keys": ";".join(d.get("merged_from_domain_keys", [])),
                "merged_child": bool(d.get("merged_child", False)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "domain_id",
                "domain_key",
                "sample_id",
                "program_seed_id",
                "spot_count",
                "geo_centroid_x",
                "geo_centroid_y",
                "geo_area_est",
                "geo_compactness",
                "geo_boundary_ratio",
                "geo_elongation",
                "geo_leaf_ratio",
                "geo_articulation_ratio",
                "components_count",
                "internal_edge_count",
                "boundary_edge_count",
                "internal_density",
                "prog_seed_mean",
                "prog_seed_sum",
                "prog_mean_enrichment_ratio",
                "prog_mean_enrichment_delta",
                "prog_peak_value",
                "prog_prominence",
                "prog_scale_median",
                "program_confidence_raw",
                "program_confidence_used",
                "program_confidence_weight",
                "flow_graph_mode",
                "seg_active_floor",
                "bridge_gap_spots_effective",
                "thr_min_domain_spots_effective",
                "thr_min_domain_internal_density_effective",
                "qc_pass",
                "qc_reject_reasons",
                "screening_reject_tags",
                "screening_decision",
                "is_background",
                "merged_group_size",
                "merged_from_domain_keys",
                "merged_child",
            ]
        )
    return df


def run_domain_pipeline(
    program_bundle_path: str | Path,
    out_root: str | Path,
    sample_id: str,
    config: DomainPipelineConfig | None = None,
) -> Path:
    cfg = config or DomainPipelineConfig()
    set_random_seed(cfg.random_seed)

    program_bundle_path = Path(program_bundle_path)
    out_root = Path(out_root)

    sample_dir = out_root / sample_id
    final_bundle = sample_dir / "domain_bundle"
    tmp_bundle = sample_dir / f"domain_bundle.__tmp__{int(time.time())}_{os.getpid()}"

    if tmp_bundle.exists():
        shutil.rmtree(tmp_bundle)
    ensure_bundle_dirs(tmp_bundle)

    try:
        payload = load_domain_inputs(program_bundle_path=program_bundle_path, cfg=cfg)
        dense_activation = payload["dense_activation"]
        program_ids = payload["program_ids"]
        program_weight_info = payload.get("program_weight_info", {})
        program_confidence_summary = payload.get("program_confidence_summary", {})
        program_qc_selection_summary = payload.get("program_qc_selection_summary", {})
        spot_ids = payload["spot_ids"]
        n_spots = int(spot_ids.shape[0])
        n_programs = int(program_ids.shape[0])

        candidate_domains, program_summary_rows, segmentation_summary = extract_candidate_domains(
            dense_activation=dense_activation,
            program_ids=program_ids,
            adjacency=payload["adjacency"],
            coords=payload["coords"],
            potential_cfg=cfg.potential,
            filter_cfg=cfg.filter,
            program_weight_info=program_weight_info,
            active_mask=None,
            min_domain_spots_override=None,
        )

        refinement = refine_domains(
            candidate_domains=candidate_domains,
            sample_id=sample_id,
            spot_ids=spot_ids,
            cfg=cfg,
            dense_activation=dense_activation,
            program_ids=program_ids,
            program_weight_info=program_weight_info,
            adjacency=payload["adjacency"],
            coords=payload["coords"],
        )
        domains_all = refinement["domains_all"]
        accepted_domains = refinement["accepted_domains"]
        merge_summary = refinement["merge_summary"]
        merge_edge_log = refinement["merge_edge_log"]
        merge_child_log = refinement["merge_child_log"]
        final_stage = str(refinement["final_stage"])

        domains_df = _domain_rows_for_parquet(domains_all)
        domains_df, domain_reliability_summary = _attach_domain_reliability(domains_df=domains_df, cfg=cfg)
        if (not cfg.qc.keep_rejected_domains_in_table) and (not domains_df.empty):
            domains_df = domains_df.loc[domains_df["qc_pass"]].reset_index(drop=True)

        membership_df = build_domain_membership_table(accepted_domains, spot_ids=spot_ids)
        domain_graph_df = build_domain_graph_table(
            domains=accepted_domains,
            spot_edges=payload["spot_edges"],
            cfg=cfg.adjacency,
        )

        write_parquet(domains_df, tmp_bundle / "domains.parquet")
        write_parquet(membership_df, tmp_bundle / "domain_spot_membership.parquet")
        write_parquet(domain_graph_df, tmp_bundle / "domain_graph.parquet")

        domain_program_map = domains_df.loc[:, ["domain_id", "domain_key", "program_seed_id", "qc_pass"]].copy()
        write_parquet(domain_program_map, tmp_bundle / "domain_program_map.parquet")

        domain_qc_table = domains_df.copy()
        program_summary_table = build_program_domain_summary_table(
            program_summary_rows=program_summary_rows,
            domains=domains_all,
        )
        write_parquet(domain_qc_table, tmp_bundle / "qc_tables" / "domain_qc.parquet")
        write_parquet(program_summary_table, tmp_bundle / "qc_tables" / "program_domain_summary.parquet")
        write_parquet(payload["program_confidence_table"], tmp_bundle / "qc_tables" / "program_confidence_weighting.parquet")
        write_parquet(merge_edge_log, tmp_bundle / "qc_tables" / "domain_merge_edge_scan.parquet")
        write_parquet(merge_child_log, tmp_bundle / "qc_tables" / "domain_merge_log.parquet")

        acceptance_gates = {
            "flow_graph_mode": str(cfg.potential.flow_graph_mode),
            "spatial_graph_k": int(cfg.potential.spatial_graph_k),
            "bridge_gap_spots": int(cfg.potential.bridge_gap_spots),
            "bridge_gap_spots_effective": int(cfg.potential.bridge_gap_spots)
            if str(cfg.potential.flow_graph_mode) == "spatial"
            else 0,
            "min_domain_spots": int(cfg.filter.min_domain_spots)
            if cfg.filter.min_domain_spots is not None
            else None,
            "min_domain_spots_frac": float(cfg.filter.min_domain_spots_frac),
            "min_domain_internal_density": float(cfg.filter.min_domain_internal_density),
            "smoothing_alpha": float(cfg.potential.smoothing_alpha),
            "smoothing_steps": int(cfg.potential.smoothing_steps),
            "active_floor_abs": float(cfg.potential.active_floor_abs),
            "active_floor_scale_factor": float(cfg.potential.active_floor_scale_factor),
            "enforce_spatial_entity": bool(cfg.potential.enforce_spatial_entity),
            "merge_small_basins_enabled": bool(cfg.potential.merge_small_basins_enabled),
            "merge_small_basin_max_spots": int(cfg.potential.merge_small_basin_max_spots),
            "merge_enabled": bool(cfg.merge.enabled),
            "merge_min_shared_boundary_edges": int(cfg.merge.min_shared_boundary_edges),
            "merge_max_centroid_distance": float(cfg.merge.max_centroid_distance),
            "merge_max_peak_ratio": float(cfg.merge.max_peak_ratio),
            "merge_max_prominence_ratio": float(cfg.merge.max_prominence_ratio),
            "program_confidence_enabled": bool(cfg.program_confidence.enabled),
            "program_confidence_col": str(cfg.program_confidence.confidence_col),
            "program_confidence_min": float(cfg.program_confidence.min_confidence),
            "program_confidence_gamma": float(cfg.program_confidence.gamma),
            "program_confidence_strict": bool(cfg.program_confidence.strict),
            "domain_reliability_enabled": bool(cfg.domain_reliability.enabled),
            "domain_reliability_confidence_exponent": float(cfg.domain_reliability.confidence_exponent),
            "domain_reliability_prominence_exponent": float(cfg.domain_reliability.prominence_exponent),
            "domain_reliability_density_exponent": float(cfg.domain_reliability.density_exponent),
            "domain_reliability_prominence_scale_quantile": float(cfg.domain_reliability.prominence_scale_quantile),
            "domain_reliability_density_scale": float(cfg.domain_reliability.density_scale),
            "domain_reliability_min_node": float(cfg.domain_reliability.min_node_reliability),
            "domain_reliability_max_node": float(cfg.domain_reliability.max_node_reliability),
        }
        qc_report = build_qc_report(
            sample_id=sample_id,
            n_spots=n_spots,
            n_programs=n_programs,
            segmentation_summary=segmentation_summary,
            domains=domains_all,
            program_summary_table=program_summary_table,
            merge_summary=merge_summary,
            acceptance_gates=acceptance_gates,
        )
        qc_report["final_domain_stage"] = final_stage
        qc_report["program_confidence_weighting"] = program_confidence_summary
        qc_report["program_qc_selection"] = program_qc_selection_summary
        qc_report["domain_reliability"] = domain_reliability_summary
        write_json(tmp_bundle / "qc_report.json", qc_report)

        domain_meta = {
            "sample_id": sample_id,
            "n_spots": n_spots,
            "n_programs": n_programs,
            "spot_order_source": payload["spot_order_source"],
            "spot_ids_hash": hash_array(spot_ids.astype(str)),
            "candidate_domain_count": int(len(domains_all)),
            "accepted_domain_count": int(len(accepted_domains)),
            "flow_graph_mode": str(payload.get("flow_graph_mode", str(cfg.potential.flow_graph_mode))),
            "graph_source": str(payload.get("graph_source", "unknown")),
            "spatial_graph_k": int(cfg.potential.spatial_graph_k),
            "accepted_program_count": int(
                np.sum(program_summary_table["domain_count_pass"].to_numpy(dtype=np.int32) > 0)
            )
            if ("domain_count_pass" in program_summary_table.columns and not program_summary_table.empty)
            else 0,
            "domain_graph_edge_count": int(domain_graph_df.shape[0]),
            "domain_adjacency_mode": cfg.adjacency.mode,
            "program_confidence_weighting": program_confidence_summary,
            "program_qc_selection": program_qc_selection_summary,
            "domain_reliability": domain_reliability_summary,
            "merge_group_count": int(merge_summary.get("merge_group_count", 0)),
            "merged_domain_count": int(merge_summary.get("merged_domain_count", 0)),
            "final_domain_stage": final_stage,
        }
        write_json(tmp_bundle / "domain_meta.json", domain_meta)

        program_manifest_path = program_bundle_path / cfg.input.program_manifest_relpath
        programs_path = program_bundle_path / cfg.input.programs_relpath
        activation_path = program_bundle_path / cfg.input.program_activation_relpath
        program_qc_path = payload["program_qc_path"]
        gss_manifest_path = payload["gss_bundle_path"] / cfg.input.gss_manifest_relpath

        manifest = {
            "schema_version": cfg.schema_version,
            "created_at": iso_now(),
            "sample_id": sample_id,
            "code_version": get_code_version(
                repo_root=Path(__file__).resolve().parents[1],
                override=cfg.code_version_override,
            ),
            "random_seed": cfg.random_seed,
            "inputs": {
                "program_bundle_path": str(program_bundle_path.resolve()),
                "gss_bundle_path": str(payload["gss_bundle_path"].resolve()),
                "program_manifest_path": str(program_manifest_path.resolve()),
                "programs_path": str(programs_path.resolve()),
                "program_activation_path": str(activation_path.resolve()),
                "program_qc_path": str(program_qc_path.resolve()) if program_qc_path.exists() else None,
                "gss_manifest_path": str(gss_manifest_path.resolve()),
                "neighbors_idx_path": str(payload["neighbors_idx_path"].resolve()),
                "neighbors_meta_path": str(payload["neighbors_meta_path"].resolve()),
                "program_manifest_hash": hash_file(program_manifest_path),
                "programs_hash": hash_file(programs_path),
                "program_activation_hash": hash_file(activation_path),
                "program_qc_hash": hash_file(program_qc_path) if program_qc_path.exists() else None,
                "gss_manifest_hash": hash_file(gss_manifest_path),
                "neighbors_idx_hash": hash_file(payload["neighbors_idx_path"]),
                "n_spots": n_spots,
                "n_programs": n_programs,
                "spot_order_hash": hash_array(spot_ids.astype(str)),
                "program_confidence_weighting": program_confidence_summary,
                "program_qc_selection": program_qc_selection_summary,
                "domain_reliability": domain_reliability_summary,
                "final_domain_stage": final_stage,
            },
            "params": asdict(cfg),
            "outputs": {
                "domains": "domains.parquet",
                "domain_spot_membership": "domain_spot_membership.parquet",
                "domain_graph": "domain_graph.parquet",
                "domain_program_map": "domain_program_map.parquet",
                "domain_meta": "domain_meta.json",
                "qc_report": "qc_report.json",
                "qc_domain_table": "qc_tables/domain_qc.parquet",
                "qc_program_domain_summary": "qc_tables/program_domain_summary.parquet",
                "qc_program_confidence_weighting": "qc_tables/program_confidence_weighting.parquet",
                "qc_domain_merge_edge_scan": "qc_tables/domain_merge_edge_scan.parquet",
                "qc_domain_merge_log": "qc_tables/domain_merge_log.parquet",
            },
            "timestamps": {"finished_at": iso_now()},
        }
        write_json(tmp_bundle / "manifest.json", manifest)

        promote_bundle(tmp_bundle, final_bundle)
        logger.info("Domain bundle generated at %s", final_bundle)
        return final_bundle
    except Exception:
        if tmp_bundle.exists():
            shutil.rmtree(tmp_bundle, ignore_errors=True)
        raise
