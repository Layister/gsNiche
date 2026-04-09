from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .schema import AxisDefinition, ProgramEvidence, RepresentationInputBundle, RepresentationPipelineConfig
from .scoring.evidence_extractors import clamp01, saturating_score


def _safe_quantiles(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "mean": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "mean": float(np.mean(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _axis_definition_record(axis: AxisDefinition) -> dict:
    raw = asdict(axis)
    return {
        "axis_id": raw["axis_id"],
        "axis_name": raw["axis_name"] or raw["axis_id"],
        "axis_type": raw["axis_type"],
        "axis_description": raw["axis_description"] or raw["description"],
        "score_range": list(raw["score_range"]),
        "score_semantics": raw["score_semantics"],
        "primary_evidence_sources": list(raw["primary_evidence_sources"] or raw["evidence_sources"]),
        "positive_gene_markers": list(raw["positive_gene_markers"] or raw["gene_markers"]),
        "positive_annotation_terms": list(raw["positive_annotation_terms"] or raw["annotation_keywords"]),
        "topology_hints": list(raw["topology_hints"]),
        "negative_hints": list(raw["negative_hints"]),
        "gating": dict(raw.get("gating", {}) or {}),
        "weights": dict(raw.get("weights", {}) or {}),
        "notes": raw["notes"],
    }


def build_axis_definition_payload(
    cancer_type: str,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> dict:
    return {
        "cancer_type": cancer_type,
        "component_axis_definition_source": {
            "mode": "yaml_overlay",
            "common": "Representation/resources/component_axes/common.yaml",
            "cancer_specific": f"Representation/resources/component_axes/{str(cancer_type).upper()}.yaml",
        },
        "component_axes": [_axis_definition_record(axis) for axis in component_axes],
        "role_axes": [_axis_definition_record(axis) for axis in role_axes],
        "scoring_contract": {
            "v1_scope": "program_level_macro_representation_only",
            "soft_scoring": "Programs receive independent soft [0,1] support scores for each axis.",
            "axis_independence": {
                "no_across_axis_softmax": True,
                "multiple_axes_can_be_high": True,
                "all_axes_can_be_low": True,
                "score_meaning": "Axis scores represent support strength for the axis, not probability and not label assignment.",
            },
            "component_axis_logic": "Gene/scaffold evidence is primary; annotation summary is auxiliary; topology is not a primary component-axis definition source.",
            "role_axis_logic": "Coarse activation topology is primary; gene and annotation are weak auxiliary hints; role axes are not pure keyword categories.",
            "normalization": {
                "axis_score_range": [0.0, 1.0],
                "component_score_formula": {
                    "gene_weight": float(cfg.scoring.component_gene_weight),
                    "annotation_weight": float(cfg.scoring.component_annotation_weight),
                },
                "role_score_formula": {
                    "topology_weight": float(cfg.scoring.role_topology_weight),
                    "annotation_weight": float(cfg.scoring.role_annotation_weight),
                    "gene_hint_weight": float(cfg.scoring.role_gene_weight),
                },
                "sample_burden": {
                    "raw_burden": "eligible_program_activation_mass normalized within sample",
                    "confidence_weighted_burden": "eligible_program_activation_mass * overall_profile_confidence normalized within sample",
                    "summary_default": "confidence_weighted_burden",
                },
            },
            "parameters": asdict(cfg.scoring),
            "eligibility": asdict(cfg.eligibility),
        },
    }


def _information_metrics(
    evidence: ProgramEvidence,
    cfg: RepresentationPipelineConfig,
) -> dict[str, float | int | bool]:
    gene_depth = clamp01(
        0.5 * min(len(evidence.top_genes) / max(1, int(cfg.scoring.top_gene_limit)), 1.0)
        + 0.5 * min(len(evidence.scaffold_genes) / max(1, int(cfg.scoring.scaffold_gene_limit)), 1.0)
    )
    annotation_depth = evidence.annotation_confidence if evidence.annotation_available else 0.0
    topology_depth = 0.0
    if evidence.topology_available:
        topology_depth = clamp01(
            0.45 * saturating_score(evidence.activation_coverage, 0.10)
            + 0.30 * saturating_score(evidence.active_spot_count, 18.0)
            + 0.25 * evidence.topology_component_density
        )

    informative_sources = int(
        (gene_depth >= float(cfg.scoring.low_information_gene_depth_threshold))
        + (annotation_depth >= float(cfg.scoring.low_information_annotation_threshold))
        + (topology_depth >= float(cfg.scoring.low_information_topology_depth_threshold))
    )
    return {
        "gene_depth": float(gene_depth),
        "annotation_depth": float(annotation_depth),
        "topology_depth": float(topology_depth),
        "informative_source_count": informative_sources,
    }


def _overall_profile_confidence(
    evidence: ProgramEvidence,
    max_component: float,
    max_role: float,
    cfg: RepresentationPipelineConfig,
) -> tuple[float, bool, dict[str, float | int | bool]]:
    info = _information_metrics(evidence=evidence, cfg=cfg)
    axis_strength = clamp01(0.6 * max_component + 0.4 * max_role)
    source_penalty = {0: 0.35, 1: 0.70}.get(int(info["informative_source_count"]), 1.0)
    confidence = clamp01(
        (
            0.30 * evidence.program_confidence
            + 0.20 * evidence.template_evidence_score
            + 0.15 * evidence.default_use_support_score
            + 0.15 * float(info["gene_depth"])
            + 0.10 * float(info["topology_depth"])
            + 0.05 * float(info["annotation_depth"])
            + 0.05 * axis_strength
        )
        * source_penalty
    )
    low_information_flag = bool(
        int(info["informative_source_count"]) == 0
        or (
            int(info["informative_source_count"]) == 1
            and axis_strength < 0.45
        )
    )
    info["information_deficit_gene"] = bool(float(info["gene_depth"]) < float(cfg.scoring.low_information_gene_depth_threshold))
    info["information_deficit_annotation"] = bool(
        float(info["annotation_depth"]) < float(cfg.scoring.low_information_annotation_threshold)
    )
    info["information_deficit_topology"] = bool(
        float(info["topology_depth"]) < float(cfg.scoring.low_information_topology_depth_threshold)
    )
    info["low_information_flag"] = low_information_flag
    info["axis_strength"] = float(axis_strength)
    info["source_penalty"] = float(source_penalty)
    return confidence, low_information_flag, info


def build_program_profile_table(
    evidences: list[ProgramEvidence],
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    component_score_map: dict[str, dict[str, float]],
    component_detail_map: dict[str, dict[str, dict]],
    role_score_map: dict[str, dict[str, float]],
    role_detail_map: dict[str, dict[str, dict]],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    rows: list[dict] = []
    tracked_component_overlap_axes = [
        axis_id
        for axis_id in ("epithelial_like", "stromal_reactive", "inflammatory_stress")
        if axis_id in component_ids
    ]

    for evidence in evidences:
        comp_scores = component_score_map.get(evidence.program_id, {})
        role_scores = role_score_map.get(evidence.program_id, {})
        max_component = max([float(comp_scores.get(axis_id, 0.0)) for axis_id in component_ids], default=0.0)
        max_role = max([float(role_scores.get(axis_id, 0.0)) for axis_id in role_ids], default=0.0)
        overall_conf, low_information_flag, info = _overall_profile_confidence(
            evidence=evidence,
            max_component=max_component,
            max_role=max_role,
            cfg=cfg,
        )

        row = {
            "sample_id": evidence.sample_id,
            "cancer_type": evidence.cancer_type,
            "program_id": evidence.program_id,
            "validity_status": evidence.validity_status,
            "routing_status": evidence.routing_status,
            "eligibility_status": evidence.eligibility_status,
            "eligible_for_burden": bool(evidence.eligible_for_burden),
            "program_confidence": float(evidence.program_confidence),
            "template_evidence_score": float(evidence.template_evidence_score),
            "default_use_support_score": float(evidence.default_use_support_score),
            "redundancy_status": evidence.redundancy_status,
            "activation_mass": float(evidence.activation_mass),
            "activation_coverage": float(evidence.activation_coverage),
            "active_spot_count": int(evidence.active_spot_count),
            "activation_mean_active": float(evidence.activation_mean_active),
            "activation_hotspot_share": float(evidence.activation_hotspot_share),
            "hotspot_share": float(evidence.activation_hotspot_share),
            "activation_peakiness": float(evidence.activation_peakiness),
            "activation_entropy": float(evidence.activation_entropy),
            "activation_sparsity": float(evidence.activation_sparsity),
            "main_component_frac": float(evidence.main_component_frac),
            "high_activation_spot_count": int(evidence.high_activation_spot_count),
            "topology_available": bool(evidence.topology_available),
            "topology_available_flag": bool(evidence.topology_available),
            "topology_boundary_fraction": float(evidence.topology_boundary_fraction),
            "boundary_fraction": float(evidence.topology_boundary_fraction),
            "topology_local_purity": float(evidence.topology_local_purity),
            "local_purity": float(evidence.topology_local_purity),
            "topology_component_count": int(evidence.topology_component_count),
            "component_count": int(evidence.topology_component_count),
            "topology_component_density": float(evidence.topology_component_density),
            "annotation_available": bool(evidence.annotation_available),
            "annotation_available_flag": bool(evidence.annotation_available),
            "annotation_confidence": float(evidence.annotation_confidence),
            "overall_profile_confidence": float(overall_conf),
            "low_information_flag": bool(low_information_flag),
            "information_gene_depth": float(info["gene_depth"]),
            "information_annotation_depth": float(info["annotation_depth"]),
            "information_topology_depth": float(info["topology_depth"]),
            "informative_source_count": int(info["informative_source_count"]),
            "information_deficit_gene": bool(info["information_deficit_gene"]),
            "information_deficit_annotation": bool(info["information_deficit_annotation"]),
            "information_deficit_topology": bool(info["information_deficit_topology"]),
            "top_gene_count": int(len(evidence.top_genes)),
            "scaffold_gene_count": int(len(evidence.scaffold_genes)),
            "annotation_term_count": int(len(evidence.annotation_term_ids)),
        }
        for axis_id in component_ids:
            row[axis_id] = float(comp_scores.get(axis_id, 0.0))
            row[f"final_score_{axis_id}"] = float(comp_scores.get(axis_id, 0.0))
            row[f"gene_match_{axis_id}"] = float(
                component_detail_map.get(evidence.program_id, {}).get(axis_id, {}).get("gene_score", 0.0)
            )
            row[f"annotation_match_{axis_id}"] = float(
                component_detail_map.get(evidence.program_id, {}).get(axis_id, {}).get("effective_annotation_score", 0.0)
            )
            row[f"raw_annotation_match_{axis_id}"] = float(
                component_detail_map.get(evidence.program_id, {}).get(axis_id, {}).get("annotation_score", 0.0)
            )
            row[f"annotation_gate_{axis_id}"] = float(
                component_detail_map.get(evidence.program_id, {}).get(axis_id, {}).get("annotation_gate", 0.0)
            )
            row[f"component_overlap_penalty_{axis_id}"] = float(
                component_detail_map.get(evidence.program_id, {}).get(axis_id, {}).get("overlap_penalty", 0.0)
            )
        for axis_id in role_ids:
            row[axis_id] = float(role_scores.get(axis_id, 0.0))
            row[f"topology_match_{axis_id}"] = float(
                role_detail_map.get(evidence.program_id, {}).get(axis_id, {}).get("topology_score", 0.0)
            )
            row[f"annotation_match_{axis_id}"] = float(
                role_detail_map.get(evidence.program_id, {}).get(axis_id, {}).get("annotation_score", 0.0)
            )
        role_debug = role_detail_map.get(evidence.program_id, {}).get("interface_like", {})
        row["interface_like_boundary_component"] = float(role_debug.get("boundary_fraction", 0.0))
        row["interface_like_mixed_neighbors_component"] = float(role_debug.get("mixed_neighbors", 0.0))
        row["interface_like_context_component"] = float(role_debug.get("interface_context", 0.0))
        row["interface_like_penalty_component"] = float(role_debug.get("interface_penalty", 0.0))
        node_debug = role_detail_map.get(evidence.program_id, {}).get("node_like", {})
        row["node_like_hotspot_component"] = float(node_debug.get("node_like_hotspot_component", 0.0))
        row["node_like_peakiness_component"] = float(node_debug.get("node_like_peakiness_component", 0.0))
        row["node_like_low_coverage_component"] = float(node_debug.get("node_like_low_coverage_component", 0.0))
        row["node_like_single_component_component"] = float(node_debug.get("node_like_single_component_component", 0.0))
        row["node_like_penalty_component"] = float(node_debug.get("node_like_penalty_component", 0.0))
        ordered_components = sorted(
            [(axis_id, float(comp_scores.get(axis_id, 0.0))) for axis_id in component_ids],
            key=lambda x: (-x[1], x[0]),
        )
        row["component_primary_axis"] = str(ordered_components[0][0]) if ordered_components else ""
        row["component_primary_margin"] = float(
            ordered_components[0][1] - ordered_components[1][1]
        ) if len(ordered_components) > 1 else float(ordered_components[0][1]) if ordered_components else 0.0
        row["component_multi_high_count"] = int(
            sum(score >= float(cfg.scoring.supported_axis_score_threshold) for _, score in ordered_components)
        )
        row["component_multi_high_flag"] = bool(row["component_multi_high_count"] > 1)
        triad_scores = [float(comp_scores.get(axis_id, 0.0)) for axis_id in tracked_component_overlap_axes]
        row["component_triad_multi_high_count"] = int(
            sum(score >= float(cfg.scoring.supported_axis_score_threshold) for score in triad_scores)
        )
        row["component_triad_multi_high_flag"] = bool(row["component_triad_multi_high_count"] > 1)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        base_cols = [
            "sample_id",
            "cancer_type",
            "program_id",
            "eligibility_status",
            "overall_profile_confidence",
            "low_information_flag",
        ]
        return pd.DataFrame(columns=base_cols + component_ids + role_ids)
    return out.sort_values(["sample_id", "program_id"]).reset_index(drop=True)


def _pipe_join(values: list[str]) -> str:
    return "|".join([str(x) for x in values if str(x)])


def _pipe_split(value: object) -> list[str]:
    return [x for x in str(value or "").split("|") if x]


def build_sample_burden_table(
    bundle: RepresentationInputBundle,
    program_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    eligible = program_profile_df.loc[program_profile_df["eligible_for_burden"].astype(bool)].copy()
    if eligible.empty:
        row = {"sample_id": bundle.sample_id, "cancer_type": bundle.cancer_type, "eligible_program_count": 0}
        for axis_id in [*component_ids, *role_ids]:
            row[f"{axis_id}_raw_burden"] = 0.0
            row[f"{axis_id}_confidence_weighted_burden"] = 0.0
        row["dominant_component_axes"] = ""
        row["dominant_role_axes"] = ""
        return pd.DataFrame([row])

    eligible["raw_weight_base"] = pd.to_numeric(eligible["activation_mass"], errors="coerce").fillna(0.0).clip(lower=0.0)
    eligible["confidence_weight_base"] = (
        eligible["raw_weight_base"]
        * pd.to_numeric(eligible["overall_profile_confidence"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    if float(eligible["raw_weight_base"].sum()) <= 0.0:
        eligible["raw_weight_base"] = 1.0
    if float(eligible["confidence_weight_base"].sum()) <= 0.0:
        eligible["confidence_weight_base"] = 1.0
    eligible["raw_weight"] = eligible["raw_weight_base"] / float(eligible["raw_weight_base"].sum())
    eligible["confidence_weight"] = eligible["confidence_weight_base"] / float(eligible["confidence_weight_base"].sum())

    row = {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "eligible_program_count": int(eligible.shape[0]),
        "total_raw_activation_mass": float(eligible["raw_weight_base"].sum()),
        "total_confidence_weighted_activation_mass": float(eligible["confidence_weight_base"].sum()),
    }

    component_conf_burdens: dict[str, float] = {}
    role_conf_burdens: dict[str, float] = {}
    role_raw_burdens: dict[str, float] = {}
    for axis_id in component_ids:
        row[f"{axis_id}_raw_burden"] = float((eligible["raw_weight"] * eligible[axis_id]).sum())
        row[f"{axis_id}_confidence_weighted_burden"] = float((eligible["confidence_weight"] * eligible[axis_id]).sum())
        component_conf_burdens[axis_id] = float(row[f"{axis_id}_confidence_weighted_burden"])
    for axis_id in role_ids:
        raw_burden = float((eligible["raw_weight"] * eligible[axis_id]).sum())
        conf_burden = float((eligible["confidence_weight"] * eligible[axis_id]).sum())
        if axis_id == "node_like":
            conf_burden *= float(cfg.scoring.node_like_sample_burden_attenuation)
        row[f"{axis_id}_raw_burden"] = raw_burden
        row[f"{axis_id}_confidence_weighted_burden"] = conf_burden
        role_conf_burdens[axis_id] = float(row[f"{axis_id}_confidence_weighted_burden"])
        role_raw_burdens[axis_id] = float(raw_burden)

    node_like_representative_program_count = 0
    node_like_candidate = eligible.loc[
        (
            pd.to_numeric(eligible.get("node_like", 0.0), errors="coerce").fillna(0.0)
            >= float(cfg.scoring.supported_axis_score_threshold)
        )
        & (
            pd.to_numeric(eligible.get("overall_profile_confidence", 0.0), errors="coerce").fillna(0.0)
            >= float(cfg.scoring.representative_program_min_confidence)
        )
    ].copy()
    if not node_like_candidate.empty:
        node_like_representative_program_count = int(node_like_candidate.shape[0])
    row["node_like_representative_program_count"] = int(node_like_representative_program_count)
    row["node_like_raw_burden_unattenuated"] = float(role_raw_burdens.get("node_like", 0.0))
    row["node_like_confidence_weighted_burden_unattenuated"] = float(
        (eligible["confidence_weight"] * eligible["node_like"]).sum()
    ) if "node_like" in eligible.columns else 0.0

    top_k = max(1, int(cfg.scoring.dominant_axis_top_k))
    row["dominant_component_axes"] = _pipe_join(
        [axis_id for axis_id, _ in sorted(component_conf_burdens.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
    )
    row["dominant_role_axes"] = _pipe_join(
        [axis_id for axis_id, _ in sorted(role_conf_burdens.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
    )
    competing_node_roles = [role_conf_burdens.get(axis_id, 0.0) for axis_id in role_ids if axis_id != "node_like"]
    node_competing_max = max(competing_node_roles) if competing_node_roles else 0.0
    row["node_like_dominance_margin"] = float(role_conf_burdens.get("node_like", 0.0) - node_competing_max)
    row["node_like_competing_role_burden"] = float(node_competing_max)

    component_representation_scores: dict[str, float] = {}
    burden_rank_map = _rank_map(component_conf_burdens)
    component_primary_threshold = float(cfg.scoring.component_representation_axis_score_threshold)
    representative_threshold = float(cfg.scoring.representative_program_min_confidence)
    high_conf_threshold = float(cfg.scoring.high_confidence_profile_threshold)
    eligible_count = max(int(eligible.shape[0]), 1)
    max_primary_count = max(
        [int((eligible.get("component_primary_axis", pd.Series(dtype=str)).astype(str) == axis_id).sum()) for axis_id in component_ids],
        default=1,
    )
    max_rep_program_count = 1
    for axis_id in component_ids:
        primary_mask = eligible.get("component_primary_axis", pd.Series(dtype=str)).astype(str) == axis_id
        primary = eligible.loc[primary_mask].copy()
        rep_mask = (
            (pd.to_numeric(eligible.get(axis_id, 0.0), errors="coerce").fillna(0.0) >= component_primary_threshold)
            & (pd.to_numeric(eligible.get("overall_profile_confidence", 0.0), errors="coerce").fillna(0.0) >= representative_threshold)
            & (~eligible.get("low_information_flag", pd.Series(dtype=bool)).astype(bool))
        )
        representative = eligible.loc[rep_mask].copy()
        max_rep_program_count = max(max_rep_program_count, int(representative.shape[0]))
    total_conf_mass = max(float(eligible["confidence_weight_base"].sum()), 1e-8)
    for axis_id in component_ids:
        primary_mask = eligible.get("component_primary_axis", pd.Series(dtype=str)).astype(str) == axis_id
        primary = eligible.loc[primary_mask].copy()
        representative_mask = (
            (pd.to_numeric(eligible.get(axis_id, 0.0), errors="coerce").fillna(0.0) >= component_primary_threshold)
            & (pd.to_numeric(eligible.get("overall_profile_confidence", 0.0), errors="coerce").fillna(0.0) >= representative_threshold)
            & (~eligible.get("low_information_flag", pd.Series(dtype=bool)).astype(bool))
        )
        representative = eligible.loc[representative_mask].copy()
        high_conf_primary_mask = (
            primary.get("overall_profile_confidence", pd.Series(dtype=float)).astype(float) >= high_conf_threshold
        ) & (~primary.get("low_information_flag", pd.Series(dtype=bool)).astype(bool))
        primary_axis_program_count = int(primary.shape[0])
        high_conf_primary_axis_program_count = int(high_conf_primary_mask.sum()) if not primary.empty else 0
        total_activation_mass_for_primary_axis_programs = float(primary["activation_mass"].sum()) if not primary.empty else 0.0
        total_weighted_mass_for_primary_axis_programs = float(primary["confidence_weight_base"].sum()) if not primary.empty else 0.0
        representative_program_count = int(representative.shape[0])
        top_program_contribution_share = 0.0
        if not primary.empty and total_weighted_mass_for_primary_axis_programs > 0.0:
            top_program_contribution_share = float(primary["confidence_weight_base"].max() / total_weighted_mass_for_primary_axis_programs)
        diversity_support = float(1.0 - top_program_contribution_share) if primary_axis_program_count > 1 else 0.0
        representation_support = clamp01(
            float(cfg.scoring.component_representation_mass_weight)
            * clamp01(total_weighted_mass_for_primary_axis_programs / total_conf_mass)
            + float(cfg.scoring.component_representation_primary_count_weight)
            * clamp01(primary_axis_program_count / max(max_primary_count, 1))
            + float(cfg.scoring.component_representation_high_conf_count_weight)
            * clamp01(high_conf_primary_axis_program_count / max(max_primary_count, 1))
            + float(cfg.scoring.component_representation_representative_count_weight)
            * clamp01(representative_program_count / max(max_rep_program_count, 1))
            + float(cfg.scoring.component_representation_diversity_weight)
            * diversity_support
        )
        row[f"{axis_id}_primary_axis_program_count"] = primary_axis_program_count
        row[f"{axis_id}_high_confidence_primary_axis_program_count"] = high_conf_primary_axis_program_count
        row[f"{axis_id}_total_activation_mass_for_primary_axis_programs"] = total_activation_mass_for_primary_axis_programs
        row[f"{axis_id}_total_weighted_mass_for_primary_axis_programs"] = total_weighted_mass_for_primary_axis_programs
        row[f"{axis_id}_representative_program_count"] = representative_program_count
        row[f"{axis_id}_top_program_contribution_share"] = top_program_contribution_share
        row[f"{axis_id}_representation_support"] = representation_support
        component_representation_scores[axis_id] = representation_support

    representation_rank_map = _rank_map(component_representation_scores)
    for axis_id in component_ids:
        row[f"{axis_id}_axis_burden_rank"] = int(burden_rank_map.get(axis_id, 0))
        row[f"{axis_id}_axis_representation_rank"] = int(representation_rank_map.get(axis_id, 0))

    return pd.DataFrame([row])


def _signature_record(
    burden_row: dict,
    axis_id: str | None,
) -> dict | None:
    if not axis_id:
        return None
    return {
        "axis": axis_id,
        "confidence_weighted_burden": float(burden_row.get(f"{axis_id}_confidence_weighted_burden", 0.0)),
        "raw_burden": float(burden_row.get(f"{axis_id}_raw_burden", 0.0)),
    }


def _component_summary_signature(
    burden_row: dict,
    axis_id: str | None,
) -> dict | None:
    if not axis_id:
        return None
    return {
        "axis": axis_id,
        "confidence_weighted_burden": float(burden_row.get(f"{axis_id}_confidence_weighted_burden", 0.0)),
        "representation_support": float(burden_row.get(f"{axis_id}_representation_support", 0.0)),
        "primary_axis_program_count": int(burden_row.get(f"{axis_id}_primary_axis_program_count", 0)),
        "high_confidence_primary_axis_program_count": int(
            burden_row.get(f"{axis_id}_high_confidence_primary_axis_program_count", 0)
        ),
        "representative_program_count": int(burden_row.get(f"{axis_id}_representative_program_count", 0)),
        "top_program_contribution_share": float(burden_row.get(f"{axis_id}_top_program_contribution_share", 0.0)),
        "burden_rank": int(burden_row.get(f"{axis_id}_axis_burden_rank", 0)),
        "representation_rank": int(burden_row.get(f"{axis_id}_axis_representation_rank", 0)),
    }


def _rank_map(score_map: dict[str, float]) -> dict[str, int]:
    ordered = [axis_id for axis_id, _ in sorted(score_map.items(), key=lambda x: (-x[1], x[0]))]
    return {axis_id: idx + 1 for idx, axis_id in enumerate(ordered)}


def _resolve_component_order_for_summary(
    burden_row: dict,
    component_ids: list[str],
    cfg: RepresentationPipelineConfig,
) -> tuple[list[str], dict]:
    burden_scores = {
        axis_id: float(burden_row.get(f"{axis_id}_confidence_weighted_burden", 0.0))
        for axis_id in component_ids
    }
    representation_scores = {
        axis_id: float(burden_row.get(f"{axis_id}_representation_support", 0.0))
        for axis_id in component_ids
    }
    burden_order = [axis_id for axis_id, _ in sorted(burden_scores.items(), key=lambda x: (-x[1], x[0]))]
    representation_order = [
        axis_id for axis_id, _ in sorted(representation_scores.items(), key=lambda x: (-x[1], x[0]))
    ]
    final_order = list(representation_order)
    diagnostics = {
        "burden_scores": burden_scores,
        "representation_scores": representation_scores,
        "dominant_component_by_burden": burden_order[0] if burden_order else "",
        "dominant_component_by_representation": representation_order[0] if representation_order else "",
        "used_burden_fallback": False,
        "representation_margin": 0.0,
        "burden_margin": 0.0,
    }
    if representation_order:
        rep_top = representation_scores.get(representation_order[0], 0.0)
        rep_next = representation_scores.get(representation_order[1], 0.0) if len(representation_order) > 1 else 0.0
        diagnostics["representation_margin"] = float(rep_top - rep_next)
    if burden_order:
        burden_top = burden_scores.get(burden_order[0], 0.0)
        burden_next = burden_scores.get(burden_order[1], 0.0) if len(burden_order) > 1 else 0.0
        diagnostics["burden_margin"] = float(burden_top - burden_next)
    if burden_order and representation_order and burden_order[0] != representation_order[0]:
        if diagnostics["representation_margin"] < float(cfg.scoring.component_dominant_representation_margin):
            if diagnostics["burden_margin"] >= float(cfg.scoring.component_dominant_burden_fallback_margin):
                final_order = list(burden_order)
                diagnostics["used_burden_fallback"] = True
    return final_order, diagnostics


def _resolve_role_order_for_summary(
    burden_row: dict,
    role_ids: list[str],
    cfg: RepresentationPipelineConfig,
) -> tuple[list[str], dict]:
    role_burdens = {
        axis_id: float(burden_row.get(f"{axis_id}_confidence_weighted_burden", 0.0))
        for axis_id in role_ids
    }
    ordered = [axis_id for axis_id, _ in sorted(role_burdens.items(), key=lambda x: (-x[1], x[0]))]
    diagnostics = {
        "role_burdens": role_burdens,
        "node_like_demoted": False,
        "node_like_margin_vs_next": 0.0,
        "node_like_competing_max": 0.0,
    }
    if ordered and ordered[0] == "node_like":
        next_best = max([role_burdens.get(axis_id, 0.0) for axis_id in role_ids if axis_id != "node_like"], default=0.0)
        margin = float(role_burdens["node_like"] - next_best)
        diagnostics["node_like_margin_vs_next"] = margin
        diagnostics["node_like_competing_max"] = float(next_best)
        qualifies = (
            role_burdens["node_like"] >= float(cfg.scoring.node_like_dominant_min_confidence_weighted_burden)
            and margin >= float(cfg.scoring.node_like_dominant_min_margin)
            and next_best <= float(cfg.scoring.node_like_dominant_max_competing_role_burden)
        )
        if not qualifies:
            ordered = [axis_id for axis_id in ordered if axis_id != "node_like"]
            insert_at = min(1, len(ordered))
            ordered.insert(insert_at, "node_like")
            diagnostics["node_like_demoted"] = True
    return ordered, diagnostics


def _summary_candidates(program_profile_df: pd.DataFrame, cfg: RepresentationPipelineConfig) -> pd.DataFrame:
    eligible = program_profile_df.loc[program_profile_df["eligible_for_burden"].astype(bool)].copy()
    if eligible.empty:
        return eligible
    preferred = eligible.loc[
        (~eligible["low_information_flag"].astype(bool))
        & (pd.to_numeric(eligible["overall_profile_confidence"], errors="coerce").fillna(0.0) >= float(cfg.scoring.representative_program_min_confidence))
    ].copy()
    if not preferred.empty:
        return preferred
    fallback = eligible.loc[~eligible["low_information_flag"].astype(bool)].copy()
    if not fallback.empty:
        return fallback
    return eligible


def _summary_reliability_hint(
    eligible_program_count: int,
    low_information_ratio: float,
    annotation_deficit_ratio: float,
    dominant_margin: float,
) -> str:
    if (
        eligible_program_count >= 12
        and low_information_ratio <= 0.25
        and annotation_deficit_ratio <= 0.35
        and dominant_margin >= 0.03
    ):
        return "high"
    if (
        eligible_program_count >= 8
        and low_information_ratio <= 0.60
        and annotation_deficit_ratio <= 0.70
        and dominant_margin >= 0.01
    ):
        return "medium"
    return "low"


def _component_summary_reliability(
    burden_row: dict,
    component_ids: list[str],
) -> str:
    if not component_ids:
        return "low"
    dominant_axis = max(
        component_ids,
        key=lambda axis_id: (
            float(burden_row.get(f"{axis_id}_representation_support", 0.0)),
            -int(burden_row.get(f"{axis_id}_axis_representation_rank", 999)),
        ),
    )
    support = float(burden_row.get(f"{dominant_axis}_representation_support", 0.0))
    rep_count = int(burden_row.get(f"{dominant_axis}_representative_program_count", 0))
    top_share = float(burden_row.get(f"{dominant_axis}_top_program_contribution_share", 1.0))
    if support >= 0.45 and rep_count >= 3 and top_share <= 0.60:
        return "high"
    if support >= 0.25 and rep_count >= 1 and top_share <= 0.80:
        return "medium"
    return "low"


def _top_programs_for_axis(
    df: pd.DataFrame,
    axis_id: str,
    weight_col: str,
    top_n: int,
) -> list[dict]:
    if df.empty:
        return []
    local = df.copy()
    local["axis_contribution"] = (
        pd.to_numeric(local[axis_id], errors="coerce").fillna(0.0)
        * pd.to_numeric(local[weight_col], errors="coerce").fillna(0.0)
    )
    keep = local.sort_values(
        ["axis_contribution", axis_id, "overall_profile_confidence", "program_id"],
        ascending=[False, False, False, True],
    ).head(max(1, int(top_n)))
    return [
        {
            "program_id": str(row["program_id"]),
            "axis_score": float(row[axis_id]),
            "axis_contribution": float(row["axis_contribution"]),
            "overall_profile_confidence": float(row["overall_profile_confidence"]),
            "low_information_flag": bool(row["low_information_flag"]),
        }
        for _, row in keep.iterrows()
    ]


def build_sample_summary_payload(
    bundle: RepresentationInputBundle,
    program_profile_df: pd.DataFrame,
    sample_burden_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> dict:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    burden_row = sample_burden_df.iloc[0].to_dict()
    dominant_component_axes = _pipe_split(burden_row.get("dominant_component_axes", ""))
    dominant_role_axes = _pipe_split(burden_row.get("dominant_role_axes", ""))
    eligible = program_profile_df.loc[program_profile_df["eligible_for_burden"].astype(bool)].copy()
    summary_candidates = _summary_candidates(program_profile_df, cfg)

    if eligible.empty:
        return {
            "sample_id": bundle.sample_id,
            "cancer_type": bundle.cancer_type,
            "dominant_component_axes": [],
            "dominant_role_axes": [],
            "dominant_component_signature": None,
            "secondary_component_signature": None,
            "dominant_role_signature": None,
            "secondary_role_signature": None,
            "top_programs": [],
            "top_contributing_programs_by_component": [],
            "top_contributing_programs_by_role": [],
            "program_level_overview": [],
        }

    eligible["raw_weight_base"] = pd.to_numeric(eligible["activation_mass"], errors="coerce").fillna(0.0).clip(lower=0.0)
    eligible["confidence_weight_base"] = (
        eligible["raw_weight_base"]
        * pd.to_numeric(eligible["overall_profile_confidence"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    if float(eligible["raw_weight_base"].sum()) <= 0.0:
        eligible["raw_weight_base"] = 1.0
    if float(eligible["confidence_weight_base"].sum()) <= 0.0:
        eligible["confidence_weight_base"] = 1.0
    eligible["confidence_weight"] = eligible["confidence_weight_base"] / float(eligible["confidence_weight_base"].sum())
    if not summary_candidates.empty:
        summary_candidates = summary_candidates.merge(
            eligible.loc[:, ["program_id", "raw_weight_base", "confidence_weight_base", "confidence_weight"]],
            on="program_id",
            how="left",
        )

    top_programs: list[dict] = []
    for _, row in summary_candidates.sort_values(
        ["confidence_weight_base", "overall_profile_confidence", "program_id"],
        ascending=[False, False, True],
    ).head(max(3, int(cfg.scoring.top_programs_per_axis))).iterrows():
        top_comp = sorted([(axis_id, float(row[axis_id])) for axis_id in component_ids], key=lambda x: (-x[1], x[0]))[:2]
        top_role = sorted([(axis_id, float(row[axis_id])) for axis_id in role_ids], key=lambda x: (-x[1], x[0]))[:2]
        top_programs.append(
            {
                "program_id": str(row["program_id"]),
                "confidence_weighted_mass": float(row["confidence_weight_base"]),
                "overall_profile_confidence": float(row["overall_profile_confidence"]),
                "low_information_flag": bool(row["low_information_flag"]),
                "top_component_axes": [{"axis": axis_id, "score": score} for axis_id, score in top_comp],
                "top_role_axes": [{"axis": axis_id, "score": score} for axis_id, score in top_role],
            }
        )

    overview: list[dict] = []
    for _, row in eligible.sort_values(["overall_profile_confidence", "program_id"], ascending=[False, True]).iterrows():
        top_comp = sorted([(axis_id, float(row[axis_id])) for axis_id in component_ids], key=lambda x: (-x[1], x[0]))[:2]
        top_role = sorted([(axis_id, float(row[axis_id])) for axis_id in role_ids], key=lambda x: (-x[1], x[0]))[:1]
        overview.append(
            {
                "program_id": str(row["program_id"]),
                "eligibility_status": str(row["eligibility_status"]),
                "low_information_flag": bool(row["low_information_flag"]),
                "overall_profile_confidence": float(row["overall_profile_confidence"]),
                "macro_profile_text": (
                    f"component={top_comp[0][0]}:{top_comp[0][1]:.2f}"
                    + (f", {top_comp[1][0]}:{top_comp[1][1]:.2f}" if len(top_comp) > 1 else "")
                    + (f"; role={top_role[0][0]}:{top_role[0][1]:.2f}" if top_role else "")
                ),
            }
        )

    top_component_axes_for_summary = dominant_component_axes[:2]
    component_order_for_summary, component_summary_diag = _resolve_component_order_for_summary(
        burden_row=burden_row,
        component_ids=component_ids,
        cfg=cfg,
    )
    final_component_axes_for_summary = component_order_for_summary[:2]
    role_order_for_summary, node_like_role_diag = _resolve_role_order_for_summary(burden_row, role_ids, cfg)
    top_role_axes_for_summary = role_order_for_summary[:2]
    component_contributors = [
        {
            "axis": axis_id,
            "programs": _top_programs_for_axis(summary_candidates, axis_id, "confidence_weight_base", cfg.scoring.top_programs_per_axis),
        }
        for axis_id in final_component_axes_for_summary
    ]
    role_contributors = [
        {
            "axis": axis_id,
            "programs": _top_programs_for_axis(summary_candidates, axis_id, "confidence_weight_base", cfg.scoring.top_programs_per_axis),
        }
        for axis_id in top_role_axes_for_summary
    ]
    low_information_ratio = float(eligible["low_information_flag"].astype(bool).mean()) if len(eligible) else 1.0
    annotation_deficit_ratio = float(eligible["information_deficit_annotation"].astype(bool).mean()) if len(eligible) else 1.0
    dominant_component_margin = 0.0
    dominant_role_margin = 0.0
    if len(dominant_component_axes) >= 2:
        dominant_component_margin = float(
            burden_row.get(f"{dominant_component_axes[0]}_confidence_weighted_burden", 0.0)
            - burden_row.get(f"{dominant_component_axes[1]}_confidence_weighted_burden", 0.0)
        )
    elif len(dominant_component_axes) == 1:
        dominant_component_margin = float(burden_row.get(f"{dominant_component_axes[0]}_confidence_weighted_burden", 0.0))
    if len(dominant_role_axes) >= 2:
        dominant_role_margin = float(
            burden_row.get(f"{top_role_axes_for_summary[0]}_confidence_weighted_burden", 0.0)
            - burden_row.get(f"{top_role_axes_for_summary[1]}_confidence_weighted_burden", 0.0)
        )
    elif len(top_role_axes_for_summary) == 1:
        dominant_role_margin = float(burden_row.get(f"{top_role_axes_for_summary[0]}_confidence_weighted_burden", 0.0))

    return {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "dominant_component_axes": [
            _component_summary_signature(burden_row, axis_id) for axis_id in final_component_axes_for_summary if axis_id
        ],
        "dominant_role_axes": [
            _signature_record(burden_row, axis_id) for axis_id in top_role_axes_for_summary if axis_id
        ],
        "dominant_component_signature": _component_summary_signature(
            burden_row, final_component_axes_for_summary[0] if final_component_axes_for_summary else None
        ),
        "secondary_component_signature": _component_summary_signature(
            burden_row, final_component_axes_for_summary[1] if len(final_component_axes_for_summary) > 1 else None
        ),
        "dominant_component_by_burden": _component_summary_signature(
            burden_row, dominant_component_axes[0] if dominant_component_axes else None
        ),
        "dominant_component_by_representation": _component_summary_signature(
            burden_row, component_summary_diag.get("dominant_component_by_representation") or None
        ),
        "dominant_component_final": _component_summary_signature(
            burden_row, final_component_axes_for_summary[0] if final_component_axes_for_summary else None
        ),
        "secondary_component_final": _component_summary_signature(
            burden_row, final_component_axes_for_summary[1] if len(final_component_axes_for_summary) > 1 else None
        ),
        "dominant_role_signature": _signature_record(burden_row, top_role_axes_for_summary[0] if top_role_axes_for_summary else None),
        "secondary_role_signature": _signature_record(burden_row, top_role_axes_for_summary[1] if len(top_role_axes_for_summary) > 1 else None),
        "component_representation_support_summary": [
            _component_summary_signature(burden_row, axis_id)
            for axis_id in sorted(
                component_ids,
                key=lambda axis_id: (
                    -float(burden_row.get(f"{axis_id}_representation_support", 0.0)),
                    axis_id,
                ),
            )
        ],
        "component_summary_reliability": _component_summary_reliability(burden_row, component_ids),
        "summary_reliability_hint": _summary_reliability_hint(
            eligible_program_count=int(eligible.shape[0]),
            low_information_ratio=low_information_ratio,
            annotation_deficit_ratio=annotation_deficit_ratio,
            dominant_margin=max(dominant_component_margin, dominant_role_margin),
        ),
        "summary_reliability_metrics": {
            "eligible_program_count": int(eligible.shape[0]),
            "low_information_ratio": float(low_information_ratio),
            "annotation_deficit_ratio": float(annotation_deficit_ratio),
            "dominant_component_margin": float(dominant_component_margin),
            "dominant_role_margin": float(dominant_role_margin),
            "dominant_component_by_burden": str(component_summary_diag.get("dominant_component_by_burden", "")),
            "dominant_component_by_representation": str(component_summary_diag.get("dominant_component_by_representation", "")),
            "used_component_burden_fallback": bool(component_summary_diag.get("used_burden_fallback", False)),
            "component_representation_margin": float(component_summary_diag.get("representation_margin", 0.0)),
            "component_burden_margin": float(component_summary_diag.get("burden_margin", 0.0)),
            "node_like_demoted": bool(node_like_role_diag.get("node_like_demoted", False)),
            "node_like_margin_vs_next": float(node_like_role_diag.get("node_like_margin_vs_next", 0.0)),
            "node_like_competing_max": float(node_like_role_diag.get("node_like_competing_max", 0.0)),
        },
        "top_programs": top_programs,
        "top_contributing_programs_by_component": component_contributors,
        "top_contributing_programs_by_role": role_contributors,
        "program_level_overview": overview,
    }


def build_program_summary_markdown(summary: dict) -> str:
    dominant_component_by_burden = summary.get("dominant_component_by_burden") or {}
    dominant_component_by_representation = summary.get("dominant_component_by_representation") or {}
    dominant_component_final = summary.get("dominant_component_final") or {}
    secondary_component_final = summary.get("secondary_component_final") or {}
    dominant_role = summary.get("dominant_role_signature") or {}
    secondary_role = summary.get("secondary_role_signature") or {}
    representative_programs = [item.get("program_id", "") for item in (summary.get("top_programs") or []) if item.get("program_id")]
    overview = (
        f"Program-level macro portrait is centered on `{dominant_component_final.get('axis', 'NA')}` "
        f"with secondary emphasis on `{secondary_component_final.get('axis', 'NA')}`, while the role layer "
        f"is led by `{dominant_role.get('axis', 'NA')}` and `{secondary_role.get('axis', 'NA')}`."
    )
    lines = [
        "# Program Macro Summary",
        "",
        f"- sample_id: `{summary.get('sample_id', '')}`",
        f"- cancer_type: `{summary.get('cancer_type', '')}`",
        f"- dominant_component_by_burden: `{dominant_component_by_burden.get('axis', 'NA')}`",
        f"- dominant_component_by_representation: `{dominant_component_by_representation.get('axis', 'NA')}`",
        f"- dominant_component_final: `{dominant_component_final.get('axis', 'NA')}`",
        f"- secondary_component_final: `{secondary_component_final.get('axis', 'NA')}`",
        f"- dominant_role_signature: `{dominant_role.get('axis', 'NA')}`",
        f"- secondary_role_signature: `{secondary_role.get('axis', 'NA')}`",
        f"- summary_reliability_hint: `{summary.get('summary_reliability_hint', 'NA')}`",
        f"- representative_programs: `{', '.join(representative_programs) if representative_programs else 'NA'}`",
        "",
        "## Structured Overview",
        "",
        overview,
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def _cross_sample_empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id_a",
            "program_id_a",
            "sample_id_b",
            "program_id_b",
            "similarity_score",
            "component_similarity",
            "role_similarity",
            "optional_structure_similarity",
            "confidence_pair_score",
            "comparable_rank_within_program_a",
            "comparable_rank_within_program_b",
            "short_comparability_note",
        ]
    )


def _safe_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _similarity_from_vectors(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(clamp01(1.0 - np.mean(np.abs(a - b))))


def build_program_cross_sample_comparability_table(
    current_program_profile_df: pd.DataFrame,
    reference_program_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    if current_program_profile_df.empty or reference_program_profile_df.empty:
        return _cross_sample_empty_pairs()

    component_ids = [axis.axis_id for axis in component_axes if axis.axis_id in current_program_profile_df.columns]
    role_ids = [axis.axis_id for axis in role_axes if axis.axis_id in current_program_profile_df.columns]
    stat_cols = [col for col in ("activation_mass", "activation_coverage", "hotspot_share") if col in current_program_profile_df.columns]
    current = current_program_profile_df.loc[current_program_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    reference = reference_program_profile_df.loc[reference_program_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    if current.empty or reference.empty:
        return _cross_sample_empty_pairs()

    for col in ("sample_id", "program_id"):
        current[col] = current[col].astype(str)
        reference[col] = reference[col].astype(str)
    reference = reference.loc[reference["sample_id"] != current["sample_id"].iloc[0]].copy()
    if reference.empty:
        return _cross_sample_empty_pairs()

    rows: list[dict] = []
    for _, row_a in current.iterrows():
        vec_a_comp = pd.to_numeric(row_a.reindex(component_ids), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        vec_a_role = pd.to_numeric(row_a.reindex(role_ids), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if stat_cols:
            stat_values_a = []
            for col in stat_cols:
                raw = float(pd.to_numeric(pd.Series([row_a.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                if col == "activation_mass":
                    stat_values_a.append(clamp01(np.log1p(raw) / 3.5))
                else:
                    stat_values_a.append(clamp01(raw))
            vec_a_stats = np.asarray(stat_values_a, dtype=float)
        else:
            vec_a_stats = np.asarray([], dtype=float)

        for _, row_b in reference.iterrows():
            vec_b_comp = pd.to_numeric(row_b.reindex(component_ids), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            vec_b_role = pd.to_numeric(row_b.reindex(role_ids), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if stat_cols:
                stat_values_b = []
                for col in stat_cols:
                    raw = float(pd.to_numeric(pd.Series([row_b.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                    if col == "activation_mass":
                        stat_values_b.append(clamp01(np.log1p(raw) / 3.5))
                    else:
                        stat_values_b.append(clamp01(raw))
                vec_b_stats = np.asarray(stat_values_b, dtype=float)
            else:
                vec_b_stats = np.asarray([], dtype=float)

            component_similarity = _similarity_from_vectors(vec_a_comp, vec_b_comp)
            role_similarity = _similarity_from_vectors(vec_a_role, vec_b_role)
            optional_structure_similarity = _similarity_from_vectors(vec_a_stats, vec_b_stats) if stat_cols else 0.0
            confidence_pair_score = np.sqrt(
                clamp01(float(row_a.get("overall_profile_confidence", 0.0)))
                * clamp01(float(row_b.get("overall_profile_confidence", 0.0)))
            )
            if _safe_bool(row_a.get("low_information_flag")):
                confidence_pair_score *= float(cfg.scoring.cross_sample_low_information_penalty)
            if _safe_bool(row_b.get("low_information_flag")):
                confidence_pair_score *= float(cfg.scoring.cross_sample_low_information_penalty)
            if not _safe_bool(row_a.get("annotation_available_flag", True)):
                confidence_pair_score *= float(cfg.scoring.cross_sample_missing_annotation_penalty)
            if not _safe_bool(row_b.get("annotation_available_flag", True)):
                confidence_pair_score *= float(cfg.scoring.cross_sample_missing_annotation_penalty)
            if not _safe_bool(row_a.get("topology_available_flag", True)):
                confidence_pair_score *= float(cfg.scoring.cross_sample_missing_topology_penalty)
            if not _safe_bool(row_b.get("topology_available_flag", True)):
                confidence_pair_score *= float(cfg.scoring.cross_sample_missing_topology_penalty)
            confidence_pair_score = clamp01(confidence_pair_score)
            similarity_score = clamp01(
                (
                    float(cfg.scoring.cross_sample_component_similarity_weight) * component_similarity
                    + float(cfg.scoring.cross_sample_role_similarity_weight) * role_similarity
                    + float(cfg.scoring.cross_sample_optional_stats_weight) * optional_structure_similarity
                )
                * confidence_pair_score
            )
            rows.append(
                {
                    "sample_id_a": str(row_a["sample_id"]),
                    "program_id_a": str(row_a["program_id"]),
                    "sample_id_b": str(row_b["sample_id"]),
                    "program_id_b": str(row_b["program_id"]),
                    "similarity_score": float(similarity_score),
                    "component_similarity": float(component_similarity),
                    "role_similarity": float(role_similarity),
                    "optional_structure_similarity": float(optional_structure_similarity),
                    "confidence_pair_score": float(confidence_pair_score),
                }
            )

    if not rows:
        return _cross_sample_empty_pairs()

    pairs = pd.DataFrame(rows)
    pairs = pairs.sort_values(
        ["program_id_a", "similarity_score", "component_similarity", "role_similarity", "program_id_b"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    pairs["comparable_rank_within_program_a"] = pairs.groupby("program_id_a").cumcount() + 1
    pairs["comparable_rank_within_program_b"] = (
        pairs.sort_values(
            ["program_id_b", "similarity_score", "component_similarity", "role_similarity", "program_id_a"],
            ascending=[True, False, False, False, True],
        )
        .groupby("program_id_b")
        .cumcount()
        .reindex(pairs.index)
        .fillna(0)
        .astype(int)
        + 1
    )
    top_k = max(1, int(cfg.scoring.cross_sample_comparable_top_k))
    pairs = pairs.loc[pairs["comparable_rank_within_program_a"] <= top_k].copy()
    pairs["short_comparability_note"] = pairs.apply(
        lambda row: (
            f"Shared tendency is strongest in component ({row['component_similarity']:.2f}) "
            f"with supportive role similarity ({row['role_similarity']:.2f})."
        ),
        axis=1,
    )
    return pairs.sort_values(
        ["similarity_score", "component_similarity", "role_similarity", "program_id_a", "program_id_b"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def _comparability_reliability_hint(reference_sample_count: int, pair_count: int, mean_similarity: float) -> str:
    if reference_sample_count >= 3 and pair_count >= 6 and mean_similarity >= 0.60:
        return "high"
    if reference_sample_count >= 1 and pair_count >= 2 and mean_similarity >= 0.35:
        return "medium"
    return "low"


def _axis_weighted_mean(df: pd.DataFrame, axis_ids: list[str], weight_col: str) -> dict[str, float]:
    if df.empty:
        return {axis_id: 0.0 for axis_id in axis_ids}
    weights = pd.to_numeric(df.get(weight_col, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) <= 0.0:
        weights = pd.Series(np.ones(len(df), dtype=float), index=df.index)
    out: dict[str, float] = {}
    for axis_id in axis_ids:
        values = pd.to_numeric(df.get(axis_id, 0.0), errors="coerce").fillna(0.0)
        out[axis_id] = float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))
    return out


def build_program_cross_sample_summary_payload(
    bundle: RepresentationInputBundle,
    current_program_profile_df: pd.DataFrame,
    cross_sample_pairs_df: pd.DataFrame,
    reference_program_profile_df: pd.DataFrame,
    reference_program_summary_map: dict[str, dict],
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> dict:
    component_ids = [axis.axis_id for axis in component_axes if axis.axis_id in current_program_profile_df.columns]
    role_ids = [axis.axis_id for axis in role_axes if axis.axis_id in current_program_profile_df.columns]
    current = current_program_profile_df.loc[current_program_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    current["summary_weight"] = (
        pd.to_numeric(current.get("activation_mass", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(current.get("overall_profile_confidence", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    current_component_mean = _axis_weighted_mean(current, component_ids, "summary_weight")
    current_role_mean = _axis_weighted_mean(current, role_ids, "summary_weight")

    if cross_sample_pairs_df.empty:
        return {
            "sample_id": bundle.sample_id,
            "cancer_type": bundle.cancer_type,
            "nearest_comparable_samples": [],
            "shared_dominant_component_tendencies": [],
            "shared_dominant_role_tendencies": [],
            "sample_specific_component_emphasis": [],
            "sample_specific_role_emphasis": [],
            "top_comparable_program_pairs": [],
            "comparability_reliability_hint": "low",
            "reference_sample_count": 0,
            "comparable_pair_count": 0,
        }

    sample_rows: list[dict] = []
    for sample_id_b, group in cross_sample_pairs_df.groupby("sample_id_b", sort=False):
        ordered = group.sort_values(["similarity_score", "component_similarity", "role_similarity"], ascending=[False, False, False])
        top_mean = float(ordered["similarity_score"].head(max(1, int(cfg.scoring.cross_sample_comparable_top_k))).mean())
        top_max = float(ordered["similarity_score"].max())
        pair_count = int(ordered.shape[0])
        score = clamp01(0.6 * top_mean + 0.4 * top_max)
        sample_rows.append(
            {
                "sample_id": str(sample_id_b),
                "comparability_score": float(score),
                "pair_count": pair_count,
                "top_similarity": top_max,
                "mean_similarity": top_mean,
                "dominant_component_final": (
                    (reference_program_summary_map.get(str(sample_id_b), {}).get("dominant_component_final") or {}).get("axis", "")
                ),
                "dominant_role_signature": (
                    (reference_program_summary_map.get(str(sample_id_b), {}).get("dominant_role_signature") or {}).get("axis", "")
                ),
            }
        )
    nearest_samples = sorted(
        sample_rows,
        key=lambda item: (-float(item["comparability_score"]), -int(item["pair_count"]), item["sample_id"]),
    )[: max(1, int(cfg.scoring.cross_sample_nearest_samples_top_k))]

    top_pairs_df = cross_sample_pairs_df.sort_values(
        ["similarity_score", "component_similarity", "role_similarity", "program_id_a", "program_id_b"],
        ascending=[False, False, False, True, True],
    ).head(max(1, int(cfg.scoring.cross_sample_summary_top_pairs)))
    top_pairs = [
        {
            "sample_id_a": str(row["sample_id_a"]),
            "program_id_a": str(row["program_id_a"]),
            "sample_id_b": str(row["sample_id_b"]),
            "program_id_b": str(row["program_id_b"]),
            "similarity_score": float(row["similarity_score"]),
            "component_similarity": float(row["component_similarity"]),
            "role_similarity": float(row["role_similarity"]),
            "confidence_pair_score": float(row["confidence_pair_score"]),
            "comparable_rank_within_program_a": int(row["comparable_rank_within_program_a"]),
            "comparable_rank_within_program_b": int(row["comparable_rank_within_program_b"]),
            "short_comparability_note": str(row["short_comparability_note"]),
        }
        for _, row in top_pairs_df.iterrows()
    ]
    matched_reference_ids = sorted({str(x) for x in cross_sample_pairs_df["sample_id_b"].astype(str).unique().tolist()})
    matched_reference = reference_program_profile_df.loc[
        reference_program_profile_df.get("sample_id", pd.Series(dtype=object)).astype(str).isin(matched_reference_ids)
        & reference_program_profile_df.get("eligible_for_burden", False).astype(bool)
    ].copy()
    matched_reference["summary_weight"] = (
        pd.to_numeric(matched_reference.get("activation_mass", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(matched_reference.get("overall_profile_confidence", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    matched_component_mean = _axis_weighted_mean(matched_reference, component_ids, "summary_weight")
    matched_role_mean = _axis_weighted_mean(matched_reference, role_ids, "summary_weight")

    shared_component_scores = []
    for axis_id in component_ids:
        shared_score = 0.0
        for _, row in cross_sample_pairs_df.iterrows():
            current_row = current.loc[current["program_id"] == str(row["program_id_a"])].head(1)
            reference_row = matched_reference.loc[
                (matched_reference["sample_id"].astype(str) == str(row["sample_id_b"]))
                & (matched_reference["program_id"].astype(str) == str(row["program_id_b"]))
            ].head(1)
            if current_row.empty or reference_row.empty:
                continue
            shared_score += float(row["similarity_score"]) * min(
                float(current_row.iloc[0].get(axis_id, 0.0)),
                float(reference_row.iloc[0].get(axis_id, 0.0)),
            )
        shared_component_scores.append({"axis": axis_id, "score": float(shared_score)})
    shared_role_scores = []
    for axis_id in role_ids:
        shared_score = 0.0
        for _, row in cross_sample_pairs_df.iterrows():
            current_row = current.loc[current["program_id"] == str(row["program_id_a"])].head(1)
            reference_row = matched_reference.loc[
                (matched_reference["sample_id"].astype(str) == str(row["sample_id_b"]))
                & (matched_reference["program_id"].astype(str) == str(row["program_id_b"]))
            ].head(1)
            if current_row.empty or reference_row.empty:
                continue
            shared_score += float(row["similarity_score"]) * min(
                float(current_row.iloc[0].get(axis_id, 0.0)),
                float(reference_row.iloc[0].get(axis_id, 0.0)),
            )
        shared_role_scores.append({"axis": axis_id, "score": float(shared_score)})

    sample_specific_components = sorted(
        [{"axis": axis_id, "delta": float(current_component_mean.get(axis_id, 0.0) - matched_component_mean.get(axis_id, 0.0))}
         for axis_id in component_ids],
        key=lambda item: (-item["delta"], item["axis"]),
    )
    sample_specific_roles = sorted(
        [{"axis": axis_id, "delta": float(current_role_mean.get(axis_id, 0.0) - matched_role_mean.get(axis_id, 0.0))}
         for axis_id in role_ids],
        key=lambda item: (-item["delta"], item["axis"]),
    )

    pair_count = int(cross_sample_pairs_df.shape[0])
    mean_similarity = float(cross_sample_pairs_df["similarity_score"].mean()) if pair_count else 0.0
    return {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "nearest_comparable_samples": nearest_samples,
        "shared_dominant_component_tendencies": sorted(shared_component_scores, key=lambda item: (-item["score"], item["axis"]))[:3],
        "shared_dominant_role_tendencies": sorted(shared_role_scores, key=lambda item: (-item["score"], item["axis"]))[:3],
        "sample_specific_component_emphasis": [item for item in sample_specific_components if item["delta"] > 0.01][:3],
        "sample_specific_role_emphasis": [item for item in sample_specific_roles if item["delta"] > 0.01][:3],
        "top_comparable_program_pairs": top_pairs,
        "comparability_reliability_hint": _comparability_reliability_hint(
            reference_sample_count=len(matched_reference_ids),
            pair_count=pair_count,
            mean_similarity=mean_similarity,
        ),
        "reference_sample_count": len(matched_reference_ids),
        "comparable_pair_count": pair_count,
    }


def build_program_cross_sample_summary_markdown(summary: dict) -> str:
    nearest_samples = [item.get("sample_id", "") for item in (summary.get("nearest_comparable_samples") or []) if item.get("sample_id")]
    shared_components = [item.get("axis", "") for item in (summary.get("shared_dominant_component_tendencies") or []) if item.get("axis")]
    shared_roles = [item.get("axis", "") for item in (summary.get("shared_dominant_role_tendencies") or []) if item.get("axis")]
    specific_components = [item.get("axis", "") for item in (summary.get("sample_specific_component_emphasis") or []) if item.get("axis")]
    specific_roles = [item.get("axis", "") for item in (summary.get("sample_specific_role_emphasis") or []) if item.get("axis")]
    top_pairs = summary.get("top_comparable_program_pairs") or []
    top_pair_labels = [
        f"{item.get('program_id_a', '')}->{item.get('sample_id_b', '')}:{item.get('program_id_b', '')}"
        for item in top_pairs[:3]
        if item.get("program_id_a") and item.get("program_id_b")
    ]
    lines = [
        "# Program Cross-Sample Comparability Summary",
        "",
        f"- sample_id: `{summary.get('sample_id', '')}`",
        f"- cancer_type: `{summary.get('cancer_type', '')}`",
        f"- nearest comparable samples: `{', '.join(nearest_samples) if nearest_samples else 'NA'}`",
        f"- shared dominant component tendencies: `{', '.join(shared_components) if shared_components else 'NA'}`",
        f"- shared dominant role tendencies: `{', '.join(shared_roles) if shared_roles else 'NA'}`",
        f"- sample-specific component emphasis: `{', '.join(specific_components) if specific_components else 'NA'}`",
        f"- sample-specific role emphasis: `{', '.join(specific_roles) if specific_roles else 'NA'}`",
        f"- top comparable program pairs: `{', '.join(top_pair_labels) if top_pair_labels else 'NA'}`",
        f"- comparability_reliability_hint: `{summary.get('comparability_reliability_hint', 'NA')}`",
        "",
        "## Structured Overview",
        "",
        (
            f"Program comparability places `{summary.get('sample_id', '')}` nearest to "
            f"`{nearest_samples[0] if nearest_samples else 'NA'}` with shared tendencies in "
            f"`{shared_components[0] if shared_components else 'NA'}` and `{shared_roles[0] if shared_roles else 'NA'}`."
        ),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def _cross_sample_empty_domain_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id_a",
            "domain_id_a",
            "sample_id_b",
            "domain_id_b",
            "similarity_score",
            "component_similarity",
            "role_similarity",
            "structure_similarity",
            "confidence_pair_score",
            "comparable_rank_within_domain_a",
            "comparable_rank_within_domain_b",
            "source_program_id_a",
            "source_program_id_b",
            "short_comparability_note",
        ]
    )


def build_domain_cross_sample_comparability_table(
    current_domain_profile_df: pd.DataFrame,
    reference_domain_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    if current_domain_profile_df.empty or reference_domain_profile_df.empty:
        return _cross_sample_empty_domain_pairs()

    component_ids = [axis.axis_id for axis in component_axes if axis.axis_id in current_domain_profile_df.columns]
    role_ids = [axis.axis_id for axis in role_axes if axis.axis_id in current_domain_profile_df.columns]
    structure_cols = [
        col
        for col in (
            "spot_count",
            "internal_density",
            "geo_boundary_ratio",
            "geo_elongation",
            "components_count",
            "mixed_neighbor_fraction",
            "boundary_contact_score",
        )
        if col in current_domain_profile_df.columns
    ]
    current = current_domain_profile_df.loc[current_domain_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    reference = reference_domain_profile_df.loc[reference_domain_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    if current.empty or reference.empty:
        return _cross_sample_empty_domain_pairs()

    for col in ("sample_id", "domain_id", "source_program_id"):
        current[col] = current[col].astype(str)
        reference[col] = reference[col].astype(str)
    reference = reference.loc[reference["sample_id"] != current["sample_id"].iloc[0]].copy()
    if reference.empty:
        return _cross_sample_empty_domain_pairs()

    def _prepare_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=float)
        return (
            df.loc[:, cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float, copy=False)
        )

    def _prepare_structure_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=float)
        mat = _prepare_matrix(df, cols)
        if mat.size == 0:
            return mat
        out = mat.copy()
        for idx, col in enumerate(cols):
            if col in {"spot_count", "components_count"}:
                out[:, idx] = np.clip(np.log1p(np.clip(out[:, idx], a_min=0.0, a_max=None)) / 3.0, 0.0, 1.0)
            elif col == "geo_elongation":
                out[:, idx] = np.clip((out[:, idx] - 1.0) / 2.0, 0.0, 1.0)
            else:
                out[:, idx] = np.clip(out[:, idx], 0.0, 1.0)
        return out

    current_comp = _prepare_matrix(current, component_ids)
    current_role = _prepare_matrix(current, role_ids)
    current_struct = _prepare_structure_matrix(current, structure_cols)
    current_domain_conf = np.clip(pd.to_numeric(current.get("domain_level_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    current_program_conf = np.clip(pd.to_numeric(current.get("source_program_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    top_k = max(1, int(cfg.scoring.domain_cross_sample_comparable_top_k))

    rows: list[dict] = []
    for sample_id_b, ref_block in reference.groupby("sample_id", sort=False):
        ref_block = ref_block.reset_index(drop=True)
        if ref_block.empty:
            continue
        ref_comp = _prepare_matrix(ref_block, component_ids)
        ref_role = _prepare_matrix(ref_block, role_ids)
        ref_struct = _prepare_structure_matrix(ref_block, structure_cols)
        ref_domain_conf = np.clip(pd.to_numeric(ref_block.get("domain_level_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
        ref_program_conf = np.clip(pd.to_numeric(ref_block.get("source_program_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)

        component_similarity = np.clip(1.0 - np.mean(np.abs(current_comp[:, None, :] - ref_comp[None, :, :]), axis=2), 0.0, 1.0)
        role_similarity = np.clip(1.0 - np.mean(np.abs(current_role[:, None, :] - ref_role[None, :, :]), axis=2), 0.0, 1.0)
        if structure_cols:
            structure_similarity = np.clip(1.0 - np.mean(np.abs(current_struct[:, None, :] - ref_struct[None, :, :]), axis=2), 0.0, 1.0)
        else:
            structure_similarity = np.zeros_like(component_similarity)
        confidence_pair_score = np.sqrt(current_domain_conf[:, None] * ref_domain_conf[None, :])
        confidence_pair_score *= np.sqrt(current_program_conf[:, None] * ref_program_conf[None, :])
        confidence_pair_score = np.clip(confidence_pair_score, 0.0, 1.0)
        similarity_score = np.clip(
            (
                float(cfg.scoring.domain_cross_sample_component_similarity_weight) * component_similarity
                + float(cfg.scoring.domain_cross_sample_role_similarity_weight) * role_similarity
                + float(cfg.scoring.domain_cross_sample_structure_similarity_weight) * structure_similarity
            )
            * confidence_pair_score,
            0.0,
            1.0,
        )

        k_block = min(top_k, ref_block.shape[0])
        if k_block <= 0:
            continue
        top_idx_unsorted = np.argpartition(-similarity_score, kth=k_block - 1, axis=1)[:, :k_block]
        top_scores = np.take_along_axis(similarity_score, top_idx_unsorted, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)

        for i, ref_indices in enumerate(top_idx):
            for j in ref_indices.tolist():
                rows.append(
                    {
                        "sample_id_a": str(current.iloc[i]["sample_id"]),
                        "domain_id_a": str(current.iloc[i]["domain_id"]),
                        "sample_id_b": str(sample_id_b),
                        "domain_id_b": str(ref_block.iloc[j]["domain_id"]),
                        "similarity_score": float(similarity_score[i, j]),
                        "component_similarity": float(component_similarity[i, j]),
                        "role_similarity": float(role_similarity[i, j]),
                        "structure_similarity": float(structure_similarity[i, j]),
                        "confidence_pair_score": float(confidence_pair_score[i, j]),
                        "source_program_id_a": str(current.iloc[i].get("source_program_id", "")),
                        "source_program_id_b": str(ref_block.iloc[j].get("source_program_id", "")),
                    }
                )

    if not rows:
        return _cross_sample_empty_domain_pairs()

    pairs = pd.DataFrame(rows)
    pairs = pairs.sort_values(
        ["domain_id_a", "similarity_score", "component_similarity", "role_similarity", "domain_id_b"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    pairs["comparable_rank_within_domain_a"] = pairs.groupby("domain_id_a").cumcount() + 1
    pairs["comparable_rank_within_domain_b"] = (
        pairs.sort_values(
            ["domain_id_b", "similarity_score", "component_similarity", "role_similarity", "domain_id_a"],
            ascending=[True, False, False, False, True],
        )
        .groupby("domain_id_b")
        .cumcount()
        .reindex(pairs.index)
        .fillna(0)
        .astype(int)
        + 1
    )
    pairs = pairs.loc[pairs["comparable_rank_within_domain_a"] <= top_k].copy()
    pairs["short_comparability_note"] = pairs.apply(
        lambda row: (
            f"Domain similarity is strongest in component ({row['component_similarity']:.2f}) "
            f"with supporting role ({row['role_similarity']:.2f}) and structure ({row['structure_similarity']:.2f})."
        ),
        axis=1,
    )
    return pairs.sort_values(
        ["similarity_score", "component_similarity", "role_similarity", "domain_id_a", "domain_id_b"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def build_domain_cross_sample_summary_payload(
    bundle: RepresentationInputBundle,
    current_domain_profile_df: pd.DataFrame,
    cross_sample_pairs_df: pd.DataFrame,
    reference_domain_profile_df: pd.DataFrame,
    reference_domain_summary_map: dict[str, dict],
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> dict:
    component_ids = [axis.axis_id for axis in component_axes if axis.axis_id in current_domain_profile_df.columns]
    role_ids = [axis.axis_id for axis in role_axes if axis.axis_id in current_domain_profile_df.columns]
    current = current_domain_profile_df.loc[current_domain_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    current["summary_weight"] = (
        pd.to_numeric(current.get("spot_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(current.get("domain_level_confidence", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    current_component_mean = _axis_weighted_mean(current, component_ids, "summary_weight")
    current_role_mean = _axis_weighted_mean(current, role_ids, "summary_weight")

    if cross_sample_pairs_df.empty:
        return {
            "sample_id": bundle.sample_id,
            "cancer_type": bundle.cancer_type,
            "nearest_comparable_samples": [],
            "shared_dominant_component_tendencies": [],
            "shared_dominant_role_tendencies": [],
            "sample_specific_component_emphasis": [],
            "sample_specific_role_emphasis": [],
            "top_comparable_domain_pairs": [],
            "comparability_reliability_hint": "low",
            "reference_sample_count": 0,
            "comparable_pair_count": 0,
        }

    sample_rows: list[dict] = []
    for sample_id_b, group in cross_sample_pairs_df.groupby("sample_id_b", sort=False):
        ordered = group.sort_values(["similarity_score", "component_similarity", "role_similarity"], ascending=[False, False, False])
        top_mean = float(ordered["similarity_score"].head(max(1, int(cfg.scoring.domain_cross_sample_comparable_top_k))).mean())
        top_max = float(ordered["similarity_score"].max())
        pair_count = int(ordered.shape[0])
        score = clamp01(0.6 * top_mean + 0.4 * top_max)
        sample_rows.append(
            {
                "sample_id": str(sample_id_b),
                "comparability_score": float(score),
                "pair_count": pair_count,
                "top_similarity": top_max,
                "mean_similarity": top_mean,
                "dominant_component": (
                    (reference_domain_summary_map.get(str(sample_id_b), {}).get("dominant_component") or {}).get("axis", "")
                ),
                "dominant_role": (
                    (reference_domain_summary_map.get(str(sample_id_b), {}).get("dominant_role") or {}).get("axis", "")
                ),
            }
        )
    nearest_samples = sorted(
        sample_rows,
        key=lambda item: (-float(item["comparability_score"]), -int(item["pair_count"]), item["sample_id"]),
    )[: max(1, int(cfg.scoring.domain_cross_sample_nearest_samples_top_k))]

    top_pairs_df = cross_sample_pairs_df.sort_values(
        ["similarity_score", "component_similarity", "role_similarity", "domain_id_a", "domain_id_b"],
        ascending=[False, False, False, True, True],
    ).head(max(1, int(cfg.scoring.domain_cross_sample_summary_top_pairs)))
    top_pairs = [
        {
            "sample_id_a": str(row["sample_id_a"]),
            "domain_id_a": str(row["domain_id_a"]),
            "sample_id_b": str(row["sample_id_b"]),
            "domain_id_b": str(row["domain_id_b"]),
            "similarity_score": float(row["similarity_score"]),
            "component_similarity": float(row["component_similarity"]),
            "role_similarity": float(row["role_similarity"]),
            "structure_similarity": float(row["structure_similarity"]),
            "confidence_pair_score": float(row["confidence_pair_score"]),
            "comparable_rank_within_domain_a": int(row["comparable_rank_within_domain_a"]),
            "comparable_rank_within_domain_b": int(row["comparable_rank_within_domain_b"]),
            "source_program_id_a": str(row["source_program_id_a"]),
            "source_program_id_b": str(row["source_program_id_b"]),
            "short_comparability_note": str(row["short_comparability_note"]),
        }
        for _, row in top_pairs_df.iterrows()
    ]
    matched_reference_ids = sorted({str(x) for x in cross_sample_pairs_df["sample_id_b"].astype(str).unique().tolist()})
    matched_reference = reference_domain_profile_df.loc[
        reference_domain_profile_df.get("sample_id", pd.Series(dtype=object)).astype(str).isin(matched_reference_ids)
        & reference_domain_profile_df.get("eligible_for_burden", False).astype(bool)
    ].copy()
    matched_reference["summary_weight"] = (
        pd.to_numeric(matched_reference.get("spot_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(matched_reference.get("domain_level_confidence", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    matched_component_mean = _axis_weighted_mean(matched_reference, component_ids, "summary_weight")
    matched_role_mean = _axis_weighted_mean(matched_reference, role_ids, "summary_weight")

    shared_component_scores = []
    for axis_id in component_ids:
        shared_score = 0.0
        for _, row in cross_sample_pairs_df.iterrows():
            current_row = current.loc[current["domain_id"] == str(row["domain_id_a"])].head(1)
            reference_row = matched_reference.loc[
                (matched_reference["sample_id"].astype(str) == str(row["sample_id_b"]))
                & (matched_reference["domain_id"].astype(str) == str(row["domain_id_b"]))
            ].head(1)
            if current_row.empty or reference_row.empty:
                continue
            shared_score += float(row["similarity_score"]) * min(
                float(current_row.iloc[0].get(axis_id, 0.0)),
                float(reference_row.iloc[0].get(axis_id, 0.0)),
            )
        shared_component_scores.append({"axis": axis_id, "score": float(shared_score)})
    shared_role_scores = []
    for axis_id in role_ids:
        shared_score = 0.0
        for _, row in cross_sample_pairs_df.iterrows():
            current_row = current.loc[current["domain_id"] == str(row["domain_id_a"])].head(1)
            reference_row = matched_reference.loc[
                (matched_reference["sample_id"].astype(str) == str(row["sample_id_b"]))
                & (matched_reference["domain_id"].astype(str) == str(row["domain_id_b"]))
            ].head(1)
            if current_row.empty or reference_row.empty:
                continue
            shared_score += float(row["similarity_score"]) * min(
                float(current_row.iloc[0].get(axis_id, 0.0)),
                float(reference_row.iloc[0].get(axis_id, 0.0)),
            )
        shared_role_scores.append({"axis": axis_id, "score": float(shared_score)})

    sample_specific_components = sorted(
        [{"axis": axis_id, "delta": float(current_component_mean.get(axis_id, 0.0) - matched_component_mean.get(axis_id, 0.0))}
         for axis_id in component_ids],
        key=lambda item: (-item["delta"], item["axis"]),
    )
    sample_specific_roles = sorted(
        [{"axis": axis_id, "delta": float(current_role_mean.get(axis_id, 0.0) - matched_role_mean.get(axis_id, 0.0))}
         for axis_id in role_ids],
        key=lambda item: (-item["delta"], item["axis"]),
    )

    pair_count = int(cross_sample_pairs_df.shape[0])
    mean_similarity = float(cross_sample_pairs_df["similarity_score"].mean()) if pair_count else 0.0
    return {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "nearest_comparable_samples": nearest_samples,
        "shared_dominant_component_tendencies": sorted(shared_component_scores, key=lambda item: (-item["score"], item["axis"]))[:3],
        "shared_dominant_role_tendencies": sorted(shared_role_scores, key=lambda item: (-item["score"], item["axis"]))[:3],
        "sample_specific_component_emphasis": [item for item in sample_specific_components if item["delta"] > 0.01][:3],
        "sample_specific_role_emphasis": [item for item in sample_specific_roles if item["delta"] > 0.01][:3],
        "top_comparable_domain_pairs": top_pairs,
        "comparability_reliability_hint": _comparability_reliability_hint(
            reference_sample_count=len(matched_reference_ids),
            pair_count=pair_count,
            mean_similarity=mean_similarity,
        ),
        "reference_sample_count": len(matched_reference_ids),
        "comparable_pair_count": pair_count,
    }


def build_domain_cross_sample_summary_markdown(summary: dict) -> str:
    nearest_samples = [item.get("sample_id", "") for item in (summary.get("nearest_comparable_samples") or []) if item.get("sample_id")]
    shared_components = [item.get("axis", "") for item in (summary.get("shared_dominant_component_tendencies") or []) if item.get("axis")]
    shared_roles = [item.get("axis", "") for item in (summary.get("shared_dominant_role_tendencies") or []) if item.get("axis")]
    specific_components = [item.get("axis", "") for item in (summary.get("sample_specific_component_emphasis") or []) if item.get("axis")]
    specific_roles = [item.get("axis", "") for item in (summary.get("sample_specific_role_emphasis") or []) if item.get("axis")]
    top_pairs = summary.get("top_comparable_domain_pairs") or []
    top_pair_labels = [
        f"{item.get('domain_id_a', '')}->{item.get('sample_id_b', '')}:{item.get('domain_id_b', '')}"
        for item in top_pairs[:3]
        if item.get("domain_id_a") and item.get("domain_id_b")
    ]
    lines = [
        "# Domain Cross-Sample Comparability Summary",
        "",
        f"- sample_id: `{summary.get('sample_id', '')}`",
        f"- cancer_type: `{summary.get('cancer_type', '')}`",
        f"- nearest comparable samples: `{', '.join(nearest_samples) if nearest_samples else 'NA'}`",
        f"- shared dominant component tendencies at Domain level: `{', '.join(shared_components) if shared_components else 'NA'}`",
        f"- shared dominant role tendencies at Domain level: `{', '.join(shared_roles) if shared_roles else 'NA'}`",
        f"- sample-specific component emphasis at Domain level: `{', '.join(specific_components) if specific_components else 'NA'}`",
        f"- sample-specific role emphasis at Domain level: `{', '.join(specific_roles) if specific_roles else 'NA'}`",
        f"- top comparable domain pairs: `{', '.join(top_pair_labels) if top_pair_labels else 'NA'}`",
        f"- comparability_reliability_hint: `{summary.get('comparability_reliability_hint', 'NA')}`",
        "",
        "## Structured Overview",
        "",
        (
            f"Domain comparability places `{summary.get('sample_id', '')}` nearest to "
            f"`{nearest_samples[0] if nearest_samples else 'NA'}` with shared Domain tendencies in "
            f"`{shared_components[0] if shared_components else 'NA'}` and `{shared_roles[0] if shared_roles else 'NA'}`."
        ),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def _cross_sample_empty_niche_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id_a",
            "niche_id_a",
            "sample_id_b",
            "niche_id_b",
            "similarity_score",
            "component_similarity",
            "role_similarity",
            "structure_similarity",
            "confidence_pair_score",
            "comparable_rank_within_niche_a",
            "comparable_rank_within_niche_b",
            "short_comparability_note",
        ]
    )


def build_niche_cross_sample_comparability_table(
    current_niche_profile_df: pd.DataFrame,
    reference_niche_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    if current_niche_profile_df.empty or reference_niche_profile_df.empty:
        return _cross_sample_empty_niche_pairs()

    component_ids = [axis.axis_id for axis in component_axes if axis.axis_id in current_niche_profile_df.columns]
    role_ids = [axis.axis_id for axis in role_axes if axis.axis_id in current_niche_profile_df.columns]
    structure_cols = [
        col
        for col in (
            "niche_member_count",
            "backbone_node_count",
            "program_count",
            "mean_edge_strength",
            "mean_edge_reliability",
            "interaction_confidence",
        )
        if col in current_niche_profile_df.columns
    ]
    current = current_niche_profile_df.loc[current_niche_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    reference = reference_niche_profile_df.loc[reference_niche_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    if current.empty or reference.empty:
        return _cross_sample_empty_niche_pairs()

    for col in ("sample_id", "niche_id"):
        current[col] = current[col].astype(str)
        reference[col] = reference[col].astype(str)
    reference = reference.loc[reference["sample_id"] != current["sample_id"].iloc[0]].copy()
    if reference.empty:
        return _cross_sample_empty_niche_pairs()

    def _prepare_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=float)
        return (
            df.loc[:, cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float, copy=False)
        )

    def _prepare_structure_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=float)
        mat = _prepare_matrix(df, cols)
        if mat.size == 0:
            return mat
        out = mat.copy()
        for idx, col in enumerate(cols):
            if col in {"niche_member_count", "backbone_node_count", "program_count"}:
                out[:, idx] = np.clip(np.log1p(np.clip(out[:, idx], a_min=0.0, a_max=None)) / 3.0, 0.0, 1.0)
            else:
                out[:, idx] = np.clip(out[:, idx], 0.0, 1.0)
        return out

    current_comp = _prepare_matrix(current, component_ids)
    current_role = _prepare_matrix(current, role_ids)
    current_struct = _prepare_structure_matrix(current, structure_cols)
    current_conf = np.clip(pd.to_numeric(current.get("niche_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    top_k = max(1, int(cfg.scoring.niche_cross_sample_comparable_top_k))

    rows: list[dict] = []
    for sample_id_b, ref_block in reference.groupby("sample_id", sort=False):
        ref_block = ref_block.reset_index(drop=True)
        if ref_block.empty:
            continue
        ref_comp = _prepare_matrix(ref_block, component_ids)
        ref_role = _prepare_matrix(ref_block, role_ids)
        ref_struct = _prepare_structure_matrix(ref_block, structure_cols)
        ref_conf = np.clip(pd.to_numeric(ref_block.get("niche_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)

        component_similarity = np.clip(1.0 - np.mean(np.abs(current_comp[:, None, :] - ref_comp[None, :, :]), axis=2), 0.0, 1.0)
        role_similarity = np.clip(1.0 - np.mean(np.abs(current_role[:, None, :] - ref_role[None, :, :]), axis=2), 0.0, 1.0)
        if structure_cols:
            structure_similarity = np.clip(1.0 - np.mean(np.abs(current_struct[:, None, :] - ref_struct[None, :, :]), axis=2), 0.0, 1.0)
        else:
            structure_similarity = np.zeros_like(component_similarity)
        confidence_pair_score = np.sqrt(current_conf[:, None] * ref_conf[None, :])
        confidence_pair_score = np.clip(confidence_pair_score, 0.0, 1.0)
        similarity_score = np.clip(
            (
                float(cfg.scoring.niche_cross_sample_component_similarity_weight) * component_similarity
                + float(cfg.scoring.niche_cross_sample_role_similarity_weight) * role_similarity
                + float(cfg.scoring.niche_cross_sample_structure_similarity_weight) * structure_similarity
            )
            * confidence_pair_score,
            0.0,
            1.0,
        )

        k_block = min(top_k, ref_block.shape[0])
        if k_block <= 0:
            continue
        top_idx_unsorted = np.argpartition(-similarity_score, kth=k_block - 1, axis=1)[:, :k_block]
        top_scores = np.take_along_axis(similarity_score, top_idx_unsorted, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)

        for i, ref_indices in enumerate(top_idx):
            for j in ref_indices.tolist():
                rows.append(
                    {
                        "sample_id_a": str(current.iloc[i]["sample_id"]),
                        "niche_id_a": str(current.iloc[i]["niche_id"]),
                        "sample_id_b": str(sample_id_b),
                        "niche_id_b": str(ref_block.iloc[j]["niche_id"]),
                        "similarity_score": float(similarity_score[i, j]),
                        "component_similarity": float(component_similarity[i, j]),
                        "role_similarity": float(role_similarity[i, j]),
                        "structure_similarity": float(structure_similarity[i, j]),
                        "confidence_pair_score": float(confidence_pair_score[i, j]),
                    }
                )

    if not rows:
        return _cross_sample_empty_niche_pairs()

    pairs = pd.DataFrame(rows)
    pairs = pairs.sort_values(
        ["niche_id_a", "similarity_score", "component_similarity", "role_similarity", "niche_id_b"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    pairs["comparable_rank_within_niche_a"] = pairs.groupby("niche_id_a").cumcount() + 1
    pairs["comparable_rank_within_niche_b"] = (
        pairs.sort_values(
            ["niche_id_b", "similarity_score", "component_similarity", "role_similarity", "niche_id_a"],
            ascending=[True, False, False, False, True],
        )
        .groupby("niche_id_b")
        .cumcount()
        .reindex(pairs.index)
        .fillna(0)
        .astype(int)
        + 1
    )
    pairs = pairs.loc[pairs["comparable_rank_within_niche_a"] <= top_k].copy()
    pairs["short_comparability_note"] = pairs.apply(
        lambda row: (
            f"Niche similarity is strongest in component ({row['component_similarity']:.2f}) "
            f"with supporting role ({row['role_similarity']:.2f}) and structure ({row['structure_similarity']:.2f})."
        ),
        axis=1,
    )
    return pairs.sort_values(
        ["similarity_score", "component_similarity", "role_similarity", "niche_id_a", "niche_id_b"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def build_niche_cross_sample_summary_payload(
    bundle: RepresentationInputBundle,
    current_niche_profile_df: pd.DataFrame,
    cross_sample_pairs_df: pd.DataFrame,
    reference_niche_profile_df: pd.DataFrame,
    reference_niche_summary_map: dict[str, dict],
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> dict:
    component_ids = [axis.axis_id for axis in component_axes if axis.axis_id in current_niche_profile_df.columns]
    role_ids = [axis.axis_id for axis in role_axes if axis.axis_id in current_niche_profile_df.columns]
    current = current_niche_profile_df.loc[current_niche_profile_df.get("eligible_for_burden", False).astype(bool)].copy()
    current["summary_weight"] = (
        pd.to_numeric(current.get("niche_member_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(current.get("niche_confidence", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    current_component_mean = _axis_weighted_mean(current, component_ids, "summary_weight")
    current_role_mean = _axis_weighted_mean(current, role_ids, "summary_weight")

    if cross_sample_pairs_df.empty:
        return {
            "sample_id": bundle.sample_id,
            "cancer_type": bundle.cancer_type,
            "nearest_comparable_samples": [],
            "shared_dominant_component_tendencies": [],
            "shared_dominant_role_tendencies": [],
            "sample_specific_component_emphasis": [],
            "sample_specific_role_emphasis": [],
            "top_comparable_niche_pairs": [],
            "comparability_reliability_hint": "low",
            "reference_sample_count": 0,
            "comparable_pair_count": 0,
        }

    sample_rows: list[dict] = []
    for sample_id_b, group in cross_sample_pairs_df.groupby("sample_id_b", sort=False):
        ordered = group.sort_values(["similarity_score", "component_similarity", "role_similarity"], ascending=[False, False, False])
        top_mean = float(ordered["similarity_score"].head(max(1, int(cfg.scoring.niche_cross_sample_comparable_top_k))).mean())
        top_max = float(ordered["similarity_score"].max())
        pair_count = int(ordered.shape[0])
        score = clamp01(0.6 * top_mean + 0.4 * top_max)
        sample_rows.append(
            {
                "sample_id": str(sample_id_b),
                "comparability_score": float(score),
                "pair_count": pair_count,
                "top_similarity": top_max,
                "mean_similarity": top_mean,
                "dominant_component_mix": (
                    (reference_niche_summary_map.get(str(sample_id_b), {}).get("dominant_component_mix") or {}).get("axis", "")
                ),
                "dominant_role_mix": (
                    (reference_niche_summary_map.get(str(sample_id_b), {}).get("dominant_role_mix") or {}).get("axis", "")
                ),
            }
        )
    nearest_samples = sorted(
        sample_rows,
        key=lambda item: (-float(item["comparability_score"]), -int(item["pair_count"]), item["sample_id"]),
    )[: max(1, int(cfg.scoring.niche_cross_sample_nearest_samples_top_k))]

    top_pairs_df = cross_sample_pairs_df.sort_values(
        ["similarity_score", "component_similarity", "role_similarity", "niche_id_a", "niche_id_b"],
        ascending=[False, False, False, True, True],
    ).head(max(1, int(cfg.scoring.niche_cross_sample_summary_top_pairs)))
    top_pairs = [
        {
            "sample_id_a": str(row["sample_id_a"]),
            "niche_id_a": str(row["niche_id_a"]),
            "sample_id_b": str(row["sample_id_b"]),
            "niche_id_b": str(row["niche_id_b"]),
            "similarity_score": float(row["similarity_score"]),
            "component_similarity": float(row["component_similarity"]),
            "role_similarity": float(row["role_similarity"]),
            "structure_similarity": float(row["structure_similarity"]),
            "confidence_pair_score": float(row["confidence_pair_score"]),
            "comparable_rank_within_niche_a": int(row["comparable_rank_within_niche_a"]),
            "comparable_rank_within_niche_b": int(row["comparable_rank_within_niche_b"]),
            "short_comparability_note": str(row["short_comparability_note"]),
        }
        for _, row in top_pairs_df.iterrows()
    ]
    matched_reference_ids = sorted({str(x) for x in cross_sample_pairs_df["sample_id_b"].astype(str).unique().tolist()})
    matched_reference = reference_niche_profile_df.loc[
        reference_niche_profile_df.get("sample_id", pd.Series(dtype=object)).astype(str).isin(matched_reference_ids)
        & reference_niche_profile_df.get("eligible_for_burden", False).astype(bool)
    ].copy()
    matched_reference["summary_weight"] = (
        pd.to_numeric(matched_reference.get("niche_member_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(matched_reference.get("niche_confidence", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    matched_component_mean = _axis_weighted_mean(matched_reference, component_ids, "summary_weight")
    matched_role_mean = _axis_weighted_mean(matched_reference, role_ids, "summary_weight")

    shared_component_scores = []
    for axis_id in component_ids:
        shared_score = 0.0
        for _, row in cross_sample_pairs_df.iterrows():
            current_row = current.loc[current["niche_id"] == str(row["niche_id_a"])].head(1)
            reference_row = matched_reference.loc[
                (matched_reference["sample_id"].astype(str) == str(row["sample_id_b"]))
                & (matched_reference["niche_id"].astype(str) == str(row["niche_id_b"]))
            ].head(1)
            if current_row.empty or reference_row.empty:
                continue
            shared_score += float(row["similarity_score"]) * min(
                float(current_row.iloc[0].get(axis_id, 0.0)),
                float(reference_row.iloc[0].get(axis_id, 0.0)),
            )
        shared_component_scores.append({"axis": axis_id, "score": float(shared_score)})
    shared_role_scores = []
    for axis_id in role_ids:
        shared_score = 0.0
        for _, row in cross_sample_pairs_df.iterrows():
            current_row = current.loc[current["niche_id"] == str(row["niche_id_a"])].head(1)
            reference_row = matched_reference.loc[
                (matched_reference["sample_id"].astype(str) == str(row["sample_id_b"]))
                & (matched_reference["niche_id"].astype(str) == str(row["niche_id_b"]))
            ].head(1)
            if current_row.empty or reference_row.empty:
                continue
            shared_score += float(row["similarity_score"]) * min(
                float(current_row.iloc[0].get(axis_id, 0.0)),
                float(reference_row.iloc[0].get(axis_id, 0.0)),
            )
        shared_role_scores.append({"axis": axis_id, "score": float(shared_score)})

    sample_specific_components = sorted(
        [{"axis": axis_id, "delta": float(current_component_mean.get(axis_id, 0.0) - matched_component_mean.get(axis_id, 0.0))}
         for axis_id in component_ids],
        key=lambda item: (-item["delta"], item["axis"]),
    )
    sample_specific_roles = sorted(
        [{"axis": axis_id, "delta": float(current_role_mean.get(axis_id, 0.0) - matched_role_mean.get(axis_id, 0.0))}
         for axis_id in role_ids],
        key=lambda item: (-item["delta"], item["axis"]),
    )

    pair_count = int(cross_sample_pairs_df.shape[0])
    mean_similarity = float(cross_sample_pairs_df["similarity_score"].mean()) if pair_count else 0.0
    return {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "nearest_comparable_samples": nearest_samples,
        "shared_dominant_component_tendencies": sorted(shared_component_scores, key=lambda item: (-item["score"], item["axis"]))[:3],
        "shared_dominant_role_tendencies": sorted(shared_role_scores, key=lambda item: (-item["score"], item["axis"]))[:3],
        "sample_specific_component_emphasis": [item for item in sample_specific_components if item["delta"] > 0.01][:3],
        "sample_specific_role_emphasis": [item for item in sample_specific_roles if item["delta"] > 0.01][:3],
        "top_comparable_niche_pairs": top_pairs,
        "comparability_reliability_hint": _comparability_reliability_hint(
            reference_sample_count=len(matched_reference_ids),
            pair_count=pair_count,
            mean_similarity=mean_similarity,
        ),
        "reference_sample_count": len(matched_reference_ids),
        "comparable_pair_count": pair_count,
    }


def build_niche_cross_sample_summary_markdown(summary: dict) -> str:
    nearest_samples = [item.get("sample_id", "") for item in (summary.get("nearest_comparable_samples") or []) if item.get("sample_id")]
    shared_components = [item.get("axis", "") for item in (summary.get("shared_dominant_component_tendencies") or []) if item.get("axis")]
    shared_roles = [item.get("axis", "") for item in (summary.get("shared_dominant_role_tendencies") or []) if item.get("axis")]
    specific_components = [item.get("axis", "") for item in (summary.get("sample_specific_component_emphasis") or []) if item.get("axis")]
    specific_roles = [item.get("axis", "") for item in (summary.get("sample_specific_role_emphasis") or []) if item.get("axis")]
    top_pairs = summary.get("top_comparable_niche_pairs") or []
    top_pair_labels = [
        f"{item.get('niche_id_a', '')}->{item.get('sample_id_b', '')}:{item.get('niche_id_b', '')}"
        for item in top_pairs[:3]
        if item.get("niche_id_a") and item.get("niche_id_b")
    ]
    lines = [
        "# Niche Cross-Sample Comparability Summary",
        "",
        f"- sample_id: `{summary.get('sample_id', '')}`",
        f"- cancer_type: `{summary.get('cancer_type', '')}`",
        f"- nearest comparable samples: `{', '.join(nearest_samples) if nearest_samples else 'NA'}`",
        f"- shared dominant component composition tendencies at Niche level: `{', '.join(shared_components) if shared_components else 'NA'}`",
        f"- shared dominant role composition tendencies at Niche level: `{', '.join(shared_roles) if shared_roles else 'NA'}`",
        f"- sample-specific component emphasis at Niche level: `{', '.join(specific_components) if specific_components else 'NA'}`",
        f"- sample-specific role emphasis at Niche level: `{', '.join(specific_roles) if specific_roles else 'NA'}`",
        f"- top comparable niche pairs: `{', '.join(top_pair_labels) if top_pair_labels else 'NA'}`",
        f"- comparability_reliability_hint: `{summary.get('comparability_reliability_hint', 'NA')}`",
        "",
        "## Structured Overview",
        "",
        (
            f"Niche comparability places `{summary.get('sample_id', '')}` nearest to "
            f"`{nearest_samples[0] if nearest_samples else 'NA'}` with shared Niche tendencies in "
            f"`{shared_components[0] if shared_components else 'NA'}` and `{shared_roles[0] if shared_roles else 'NA'}`."
        ),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def build_sample_summary_view(
    summary: dict,
    domain_summary: dict | None = None,
    niche_summary: dict | None = None,
    program_comparability_summary: dict | None = None,
    domain_comparability_summary: dict | None = None,
    niche_comparability_summary: dict | None = None,
) -> dict:
    dominant_component = summary.get("dominant_component_final") or summary.get("dominant_component_signature")
    secondary_component = summary.get("secondary_component_final") or summary.get("secondary_component_signature")
    dominant_role = summary.get("dominant_role_signature")
    secondary_role = summary.get("secondary_role_signature")
    view = {
        "sample_id": summary.get("sample_id", ""),
        "cancer_type": summary.get("cancer_type", ""),
        "dominant_backbone": dominant_component,
        "secondary_emphasis": secondary_component,
        "dominant_role_signature": dominant_role,
        "secondary_role_signature": secondary_role,
        "reliability": {
            "summary_reliability_hint": summary.get("summary_reliability_hint", "NA"),
            "component_summary_reliability": summary.get("component_summary_reliability", "NA"),
        },
        "cross_layer_summary": (
            f"Current cross-layer summary is driven by Program-level Representation: dominant backbone "
            f"`{(dominant_component or {}).get('axis', 'NA')}`, secondary emphasis "
            f"`{(secondary_component or {}).get('axis', 'NA')}`."
        ),
        "layer_status": {
            "program": "implemented",
            "domain": "implemented" if domain_summary else "not_implemented_yet",
            "niche": "not_implemented_yet",
        },
        "program_summary_ref": "program/macro_summary.json",
    }
    if program_comparability_summary:
        view["program_comparability_summary"] = {
            "nearest_comparable_samples": program_comparability_summary.get("nearest_comparable_samples", []),
            "shared_dominant_component_tendencies": program_comparability_summary.get("shared_dominant_component_tendencies", []),
            "shared_dominant_role_tendencies": program_comparability_summary.get("shared_dominant_role_tendencies", []),
            "sample_specific_component_emphasis": program_comparability_summary.get("sample_specific_component_emphasis", []),
            "sample_specific_role_emphasis": program_comparability_summary.get("sample_specific_role_emphasis", []),
            "comparability_reliability_hint": program_comparability_summary.get("comparability_reliability_hint", "NA"),
            "top_comparable_program_pairs": program_comparability_summary.get("top_comparable_program_pairs", []),
        }
        view["program_comparability_ref"] = "program/cross_sample_summary.json"
    if domain_summary:
        view["domain_level_summary"] = {
            "dominant_component": domain_summary.get("dominant_component"),
            "secondary_component": domain_summary.get("secondary_component"),
            "dominant_role": domain_summary.get("dominant_role"),
            "secondary_role": domain_summary.get("secondary_role"),
            "summary_reliability_hint": domain_summary.get("summary_reliability_hint", "NA"),
            "representative_domains": domain_summary.get("representative_domains", []),
        }
        view["domain_summary_ref"] = "domain/macro_summary.json"
    if domain_comparability_summary:
        view["domain_comparability_summary"] = {
            "nearest_comparable_samples": domain_comparability_summary.get("nearest_comparable_samples", []),
            "shared_dominant_component_tendencies": domain_comparability_summary.get("shared_dominant_component_tendencies", []),
            "shared_dominant_role_tendencies": domain_comparability_summary.get("shared_dominant_role_tendencies", []),
            "sample_specific_component_emphasis": domain_comparability_summary.get("sample_specific_component_emphasis", []),
            "sample_specific_role_emphasis": domain_comparability_summary.get("sample_specific_role_emphasis", []),
            "comparability_reliability_hint": domain_comparability_summary.get("comparability_reliability_hint", "NA"),
            "top_comparable_domain_pairs": domain_comparability_summary.get("top_comparable_domain_pairs", []),
        }
        view["domain_comparability_ref"] = "domain/cross_sample_summary.json"
    if niche_summary:
        view["layer_status"]["niche"] = "implemented"
        view["niche_level_summary"] = {
            "dominant_component_mix": niche_summary.get("dominant_component_mix"),
            "secondary_component_mix": niche_summary.get("secondary_component_mix"),
            "dominant_role_mix": niche_summary.get("dominant_role_mix"),
            "secondary_role_mix": niche_summary.get("secondary_role_mix"),
            "summary_reliability_hint": niche_summary.get("summary_reliability_hint", "NA"),
            "representative_niches": niche_summary.get("representative_niches", []),
        }
        view["niche_summary_ref"] = "niche/macro_summary.json"
    if niche_comparability_summary:
        view["niche_comparability_summary"] = {
            "nearest_comparable_samples": niche_comparability_summary.get("nearest_comparable_samples", []),
            "shared_dominant_component_tendencies": niche_comparability_summary.get("shared_dominant_component_tendencies", []),
            "shared_dominant_role_tendencies": niche_comparability_summary.get("shared_dominant_role_tendencies", []),
            "sample_specific_component_emphasis": niche_comparability_summary.get("sample_specific_component_emphasis", []),
            "sample_specific_role_emphasis": niche_comparability_summary.get("sample_specific_role_emphasis", []),
            "comparability_reliability_hint": niche_comparability_summary.get("comparability_reliability_hint", "NA"),
            "top_comparable_niche_pairs": niche_comparability_summary.get("top_comparable_niche_pairs", []),
        }
        view["niche_comparability_ref"] = "niche/cross_sample_summary.json"
    return view


def build_sample_summary_markdown(sample_summary: dict) -> str:
    dominant_backbone = sample_summary.get("dominant_backbone") or {}
    secondary_emphasis = sample_summary.get("secondary_emphasis") or {}
    dominant_role = sample_summary.get("dominant_role_signature") or {}
    secondary_role = sample_summary.get("secondary_role_signature") or {}
    reliability = sample_summary.get("reliability") or {}
    lines = [
        "# Sample Macro Summary",
        "",
        f"- sample_id: `{sample_summary.get('sample_id', '')}`",
        f"- cancer_type: `{sample_summary.get('cancer_type', '')}`",
        f"- dominant backbone: `{dominant_backbone.get('axis', 'NA')}`",
        f"- secondary emphasis: `{secondary_emphasis.get('axis', 'NA')}`",
        f"- dominant role: `{dominant_role.get('axis', 'NA')}`",
        f"- secondary role: `{secondary_role.get('axis', 'NA')}`",
        f"- summary_reliability_hint: `{reliability.get('summary_reliability_hint', 'NA')}`",
        f"- component_summary_reliability: `{reliability.get('component_summary_reliability', 'NA')}`",
        "",
        "## Cross-Layer Summary",
        "",
        sample_summary.get("cross_layer_summary", ""),
        "",
    ]
    if sample_summary.get("program_comparability_summary"):
        program_comp = sample_summary.get("program_comparability_summary") or {}
        nearest_samples = [item.get("sample_id", "") for item in (program_comp.get("nearest_comparable_samples") or []) if item.get("sample_id")]
        shared_components = [item.get("axis", "") for item in (program_comp.get("shared_dominant_component_tendencies") or []) if item.get("axis")]
        shared_roles = [item.get("axis", "") for item in (program_comp.get("shared_dominant_role_tendencies") or []) if item.get("axis")]
        lines.extend(
            [
                "## Program Comparability",
                "",
                f"- nearest comparable samples: `{', '.join(nearest_samples) if nearest_samples else 'NA'}`",
                f"- shared component tendencies: `{', '.join(shared_components) if shared_components else 'NA'}`",
                f"- shared role tendencies: `{', '.join(shared_roles) if shared_roles else 'NA'}`",
                f"- comparability_reliability_hint: `{program_comp.get('comparability_reliability_hint', 'NA')}`",
                "",
            ]
        )
    if sample_summary.get("domain_comparability_summary"):
        domain_comp = sample_summary.get("domain_comparability_summary") or {}
        nearest_samples = [item.get("sample_id", "") for item in (domain_comp.get("nearest_comparable_samples") or []) if item.get("sample_id")]
        shared_components = [item.get("axis", "") for item in (domain_comp.get("shared_dominant_component_tendencies") or []) if item.get("axis")]
        shared_roles = [item.get("axis", "") for item in (domain_comp.get("shared_dominant_role_tendencies") or []) if item.get("axis")]
        lines.extend(
            [
                "## Domain Comparability",
                "",
                f"- nearest comparable samples: `{', '.join(nearest_samples) if nearest_samples else 'NA'}`",
                f"- shared component tendencies: `{', '.join(shared_components) if shared_components else 'NA'}`",
                f"- shared role tendencies: `{', '.join(shared_roles) if shared_roles else 'NA'}`",
                f"- comparability_reliability_hint: `{domain_comp.get('comparability_reliability_hint', 'NA')}`",
                "",
            ]
        )
    if sample_summary.get("domain_level_summary"):
        domain_section = sample_summary.get("domain_level_summary") or {}
        lines.extend(
            [
                "## Domain-Level Summary",
                "",
                f"- dominant component: `{((domain_section.get('dominant_component') or {}).get('axis', 'NA'))}`",
                f"- secondary component: `{((domain_section.get('secondary_component') or {}).get('axis', 'NA'))}`",
                f"- dominant role: `{((domain_section.get('dominant_role') or {}).get('axis', 'NA'))}`",
                f"- secondary role: `{((domain_section.get('secondary_role') or {}).get('axis', 'NA'))}`",
                f"- summary_reliability_hint: `{domain_section.get('summary_reliability_hint', 'NA')}`",
                "",
            ]
        )
    else:
        lines.append("- Domain-level: not implemented yet")
    if sample_summary.get("niche_level_summary"):
        niche_section = sample_summary.get("niche_level_summary") or {}
        lines.extend(
            [
                "## Niche-Level Summary",
                "",
                f"- dominant component composition: `{((niche_section.get('dominant_component_mix') or {}).get('axis', 'NA'))}`",
                f"- secondary component composition: `{((niche_section.get('secondary_component_mix') or {}).get('axis', 'NA'))}`",
                f"- dominant role composition: `{((niche_section.get('dominant_role_mix') or {}).get('axis', 'NA'))}`",
                f"- secondary role composition: `{((niche_section.get('secondary_role_mix') or {}).get('axis', 'NA'))}`",
                f"- summary_reliability_hint: `{niche_section.get('summary_reliability_hint', 'NA')}`",
                "",
            ]
        )
    else:
        lines.append("- Niche-level: not implemented yet")
    if sample_summary.get("niche_comparability_summary"):
        niche_comp = sample_summary.get("niche_comparability_summary") or {}
        nearest_samples = [item.get("sample_id", "") for item in (niche_comp.get("nearest_comparable_samples") or []) if item.get("sample_id")]
        shared_components = [item.get("axis", "") for item in (niche_comp.get("shared_dominant_component_tendencies") or []) if item.get("axis")]
        shared_roles = [item.get("axis", "") for item in (niche_comp.get("shared_dominant_role_tendencies") or []) if item.get("axis")]
        lines.extend(
            [
                "## Niche Comparability",
                "",
                f"- nearest comparable samples: `{', '.join(nearest_samples) if nearest_samples else 'NA'}`",
                f"- shared component tendencies: `{', '.join(shared_components) if shared_components else 'NA'}`",
                f"- shared role tendencies: `{', '.join(shared_roles) if shared_roles else 'NA'}`",
                f"- comparability_reliability_hint: `{niche_comp.get('comparability_reliability_hint', 'NA')}`",
                "",
            ]
        )
    lines.extend(
        [
        "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_future_layer_meta(layer: str) -> dict:
    return {
        "layer": str(layer),
        "status": "not_implemented",
        "purpose": f"future {layer}-level macro representation",
        "planned_outputs": [
            "macro_profile.parquet",
            "sample_burden.parquet",
            "macro_summary.json",
            "macro_summary.md",
        ],
    }


def _domain_neighbor_features(domains_df: pd.DataFrame, domain_graph_df: pd.DataFrame) -> pd.DataFrame:
    if domains_df.empty:
        return pd.DataFrame(columns=["domain_key", "neighbor_count", "mixed_neighbor_fraction", "boundary_contact_score"])
    source_col = "source_program_id" if "source_program_id" in domains_df.columns else "program_seed_id"
    base = domains_df.loc[:, [c for c in ("domain_key", source_col) if c in domains_df.columns]].copy()
    if domain_graph_df.empty or "domain_key_i" not in domain_graph_df.columns or "domain_key_j" not in domain_graph_df.columns:
        out = base.loc[:, ["domain_key"]].drop_duplicates().copy()
        out["neighbor_count"] = 0
        out["mixed_neighbor_fraction"] = 0.0
        out["boundary_contact_score"] = 0.0
        return out
    seed_map = dict(
        base.drop_duplicates("domain_key")
        .rename(columns={source_col: "source_program_id"})
        .loc[:, ["domain_key", "source_program_id"]]
        .itertuples(index=False, name=None)
    )
    rows: list[dict] = []
    for row in domain_graph_df.itertuples(index=False):
        key_i = str(getattr(row, "domain_key_i", ""))
        key_j = str(getattr(row, "domain_key_j", ""))
        shared = float(getattr(row, "shared_boundary_edges", 0.0) or 0.0)
        for src, nbr in ((key_i, key_j), (key_j, key_i)):
            if not src or not nbr:
                continue
            rows.append(
                {
                    "domain_key": src,
                    "neighbor_key": nbr,
                    "shared_boundary_edges": shared,
                    "mixed_seed": float(seed_map.get(src, "") != seed_map.get(nbr, "")),
                }
            )
    if not rows:
        out = base.loc[:, ["domain_key"]].drop_duplicates().copy()
        out["neighbor_count"] = 0
        out["mixed_neighbor_fraction"] = 0.0
        out["boundary_contact_score"] = 0.0
        return out
    edges = pd.DataFrame(rows)
    grouped = edges.groupby("domain_key", dropna=False)
    out = grouped.agg(
        neighbor_count=("neighbor_key", "nunique"),
        mixed_neighbor_fraction=("mixed_seed", "mean"),
        boundary_contact_score=("shared_boundary_edges", "sum"),
    ).reset_index()
    max_boundary = float(out["boundary_contact_score"].max()) if not out.empty else 0.0
    if max_boundary > 0.0:
        out["boundary_contact_score"] = out["boundary_contact_score"] / max_boundary
    else:
        out["boundary_contact_score"] = 0.0
    return out


def build_domain_profile_table(
    bundle: RepresentationInputBundle,
    domains_df: pd.DataFrame,
    domain_program_map_df: pd.DataFrame,
    domain_graph_df: pd.DataFrame,
    program_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    if domains_df.empty:
        cols = ["domain_id", "sample_id", "source_program_id", "domain_level_confidence", "eligible_for_burden"]
        return pd.DataFrame(columns=cols + component_ids + role_ids)
    program_keep = ["program_id", "overall_profile_confidence", "low_information_flag", *component_ids, *role_ids]
    program_ref = program_profile_df.loc[:, [c for c in program_keep if c in program_profile_df.columns]].copy()
    program_ref = program_ref.rename(columns={"program_id": "source_program_id", "overall_profile_confidence": "source_program_confidence"})
    domains = domains_df.copy()
    if "program_seed_id" in domains.columns:
        domains = domains.rename(columns={"program_seed_id": "source_program_id"})
    if not domain_program_map_df.empty:
        dmap = domain_program_map_df.copy()
        if "program_seed_id" in dmap.columns:
            dmap = dmap.rename(columns={"program_seed_id": "source_program_id"})
        domains = domains.merge(
            dmap.loc[:, [c for c in ("domain_id", "source_program_id", "qc_pass") if c in dmap.columns]].drop_duplicates("domain_id"),
            on="domain_id",
            how="left",
            suffixes=("", "_map"),
        )
        if "source_program_id_map" in domains.columns:
            domains["source_program_id"] = domains["source_program_id"].fillna(domains["source_program_id_map"]).astype(str)
            domains = domains.drop(columns=["source_program_id_map"])
        if "qc_pass_map" in domains.columns:
            domains["qc_pass"] = domains.get("qc_pass", pd.Series(dtype=bool)).fillna(domains["qc_pass_map"])
            domains = domains.drop(columns=["qc_pass_map"])
    neighbor_features = _domain_neighbor_features(domains, domain_graph_df)
    domains = domains.merge(neighbor_features, on="domain_key", how="left")
    domains = domains.merge(program_ref, on="source_program_id", how="left")
    domains["sample_id"] = domains.get("sample_id", pd.Series(dtype=object)).fillna(bundle.sample_id).astype(str)
    domains["qc_pass"] = domains.get("qc_pass", pd.Series(dtype=bool)).fillna(False).astype(bool)
    domains["source_program_confidence"] = pd.to_numeric(domains.get("source_program_confidence", 0.0), errors="coerce").fillna(0.0)
    domains["domain_reliability"] = pd.to_numeric(domains.get("domain_reliability", 0.0), errors="coerce").fillna(0.0)
    domains["spot_count"] = pd.to_numeric(domains.get("spot_count", 0.0), errors="coerce").fillna(0.0)
    domains["geo_boundary_ratio"] = pd.to_numeric(domains.get("geo_boundary_ratio", 0.0), errors="coerce").fillna(0.0)
    domains["geo_elongation"] = pd.to_numeric(domains.get("geo_elongation", 0.0), errors="coerce").fillna(0.0)
    domains["internal_density"] = pd.to_numeric(domains.get("internal_density", 0.0), errors="coerce").fillna(0.0)
    domains["components_count"] = pd.to_numeric(domains.get("components_count", 1.0), errors="coerce").fillna(1.0)
    domains["neighbor_count"] = pd.to_numeric(domains.get("neighbor_count", 0.0), errors="coerce").fillna(0.0)
    domains["mixed_neighbor_fraction"] = pd.to_numeric(domains.get("mixed_neighbor_fraction", 0.0), errors="coerce").fillna(0.0)
    domains["boundary_contact_score"] = pd.to_numeric(domains.get("boundary_contact_score", 0.0), errors="coerce").fillna(0.0)

    spot_scale = max(float(domains["spot_count"].quantile(0.75)), 1.0)
    density_scale = max(float(domains["internal_density"].quantile(0.75)), 1e-6)
    boundary_scale = max(float(domains["geo_boundary_ratio"].quantile(0.75)), 1e-6)
    elong_scale = max(float(domains["geo_elongation"].quantile(0.75)), 1.5)

    rows: list[dict] = []
    for row in domains.itertuples(index=False):
        size_support = saturating_score(getattr(row, "spot_count", 0.0), spot_scale)
        compact_support = saturating_score(getattr(row, "internal_density", 0.0), density_scale)
        boundary_support = saturating_score(getattr(row, "geo_boundary_ratio", 0.0), boundary_scale)
        elongated_support = clamp01(max(float(getattr(row, "geo_elongation", 0.0)) - 1.0, 0.0) / max(elong_scale - 1.0, 1e-6))
        small_domain = clamp01(1.0 - size_support)
        single_component = clamp01(1.0 / max(float(getattr(row, "components_count", 1.0)), 1.0))
        mixed_neighbor = clamp01(getattr(row, "mixed_neighbor_fraction", 0.0))
        domain_conf = clamp01(
            0.45 * float(getattr(row, "source_program_confidence", 0.0))
            + 0.35 * float(getattr(row, "domain_reliability", 0.0))
            + 0.10 * size_support
            + 0.10 * compact_support
        )
        record = {
            "domain_id": str(getattr(row, "domain_id", "")),
            "domain_key": str(getattr(row, "domain_key", "")),
            "sample_id": str(getattr(row, "sample_id", bundle.sample_id)),
            "source_program_id": str(getattr(row, "source_program_id", "")),
            "source_program_confidence": float(getattr(row, "source_program_confidence", 0.0)),
            "qc_pass": bool(getattr(row, "qc_pass", False)),
            "eligible_for_burden": bool(getattr(row, "qc_pass", False) and str(getattr(row, "source_program_id", "")).strip()),
            "spot_count": float(getattr(row, "spot_count", 0.0)),
            "geo_area_est": float(getattr(row, "geo_area_est", 0.0)),
            "geo_compactness": float(getattr(row, "geo_compactness", 0.0)),
            "geo_boundary_ratio": float(getattr(row, "geo_boundary_ratio", 0.0)),
            "geo_elongation": float(getattr(row, "geo_elongation", 0.0)),
            "components_count": int(round(float(getattr(row, "components_count", 1.0)))),
            "internal_density": float(getattr(row, "internal_density", 0.0)),
            "domain_reliability": float(getattr(row, "domain_reliability", 0.0)),
            "neighbor_count": int(round(float(getattr(row, "neighbor_count", 0.0)))),
            "mixed_neighbor_fraction": float(mixed_neighbor),
            "boundary_contact_score": float(getattr(row, "boundary_contact_score", 0.0)),
            "domain_size_support": float(size_support),
            "domain_boundary_support": float(boundary_support),
            "domain_compact_support": float(compact_support),
            "domain_elongated_support": float(elongated_support),
            "domain_level_confidence": float(domain_conf),
        }
        for axis_id in component_ids:
            src = float(getattr(row, axis_id, 0.0))
            factor = 0.97 + 0.03 * float(getattr(row, "domain_reliability", 0.0))
            record[axis_id] = clamp01(src * factor)
            record[f"source_program_{axis_id}"] = src
            record[f"domain_component_adjustment_{axis_id}"] = float(factor)
        for axis_id in role_ids:
            src = float(getattr(row, axis_id, 0.0))
            if axis_id == "scaffold_like":
                adjusted = clamp01(0.70 * src + 0.15 * size_support + 0.10 * compact_support + 0.05 * float(getattr(row, "domain_reliability", 0.0)))
            elif axis_id == "interface_like":
                adjusted = clamp01(0.65 * src + 0.15 * boundary_support + 0.15 * mixed_neighbor + 0.05 * elongated_support)
            elif axis_id == "node_like":
                adjusted = clamp01(0.70 * src + 0.15 * small_domain + 0.10 * compact_support + 0.05 * single_component)
            else:
                non_dominant = clamp01(1.0 - 0.7 * size_support)
                adjusted = clamp01(0.70 * src + 0.15 * non_dominant + 0.10 * (1.0 - mixed_neighbor) + 0.05 * (1.0 - size_support))
            record[axis_id] = adjusted
            record[f"source_program_{axis_id}"] = src
            record[f"domain_role_adjustment_{axis_id}"] = float(adjusted - src)
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["sample_id", "domain_id"]).reset_index(drop=True)


def build_domain_burden_table(
    bundle: RepresentationInputBundle,
    domain_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    eligible = domain_profile_df.loc[domain_profile_df.get("eligible_for_burden", pd.Series(dtype=bool)).astype(bool)].copy()
    row = {"sample_id": bundle.sample_id, "cancer_type": bundle.cancer_type, "eligible_domain_count": int(eligible.shape[0])}
    if eligible.empty:
        for axis_id in [*component_ids, *role_ids]:
            row[f"{axis_id}_raw_burden"] = 0.0
            row[f"{axis_id}_confidence_weighted_burden"] = 0.0
        row["dominant_component_by_burden"] = ""
        row["dominant_role_by_burden"] = ""
        return pd.DataFrame([row])
    eligible["raw_weight_base"] = pd.to_numeric(eligible.get("spot_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    eligible["confidence_weight_base"] = eligible["raw_weight_base"] * pd.to_numeric(
        eligible.get("domain_level_confidence", 0.0), errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    if float(eligible["raw_weight_base"].sum()) <= 0.0:
        eligible["raw_weight_base"] = 1.0
    if float(eligible["confidence_weight_base"].sum()) <= 0.0:
        eligible["confidence_weight_base"] = 1.0
    eligible["raw_weight"] = eligible["raw_weight_base"] / float(eligible["raw_weight_base"].sum())
    eligible["confidence_weight"] = eligible["confidence_weight_base"] / float(eligible["confidence_weight_base"].sum())
    comp_burdens: dict[str, float] = {}
    role_burdens: dict[str, float] = {}
    for axis_id in component_ids:
        row[f"{axis_id}_raw_burden"] = float((eligible["raw_weight"] * eligible[axis_id]).sum())
        row[f"{axis_id}_confidence_weighted_burden"] = float((eligible["confidence_weight"] * eligible[axis_id]).sum())
        rep_count = int(((eligible[axis_id] >= float(cfg.scoring.supported_axis_score_threshold)) & (eligible["domain_level_confidence"] >= float(cfg.scoring.representative_program_min_confidence))).sum())
        row[f"{axis_id}_representative_domain_count"] = rep_count
        comp_burdens[axis_id] = float(row[f"{axis_id}_confidence_weighted_burden"])
    for axis_id in role_ids:
        row[f"{axis_id}_raw_burden"] = float((eligible["raw_weight"] * eligible[axis_id]).sum())
        row[f"{axis_id}_confidence_weighted_burden"] = float((eligible["confidence_weight"] * eligible[axis_id]).sum())
        rep_count = int(((eligible[axis_id] >= float(cfg.scoring.supported_axis_score_threshold)) & (eligible["domain_level_confidence"] >= float(cfg.scoring.representative_program_min_confidence))).sum())
        row[f"{axis_id}_representative_domain_count"] = rep_count
        role_burdens[axis_id] = float(row[f"{axis_id}_confidence_weighted_burden"])
    comp_order = [axis_id for axis_id, _ in sorted(comp_burdens.items(), key=lambda x: (-x[1], x[0]))]
    role_order = [axis_id for axis_id, _ in sorted(role_burdens.items(), key=lambda x: (-x[1], x[0]))]
    row["dominant_component_by_burden"] = comp_order[0] if comp_order else ""
    row["dominant_role_by_burden"] = role_order[0] if role_order else ""
    row["secondary_component_by_burden"] = comp_order[1] if len(comp_order) > 1 else ""
    row["secondary_role_by_burden"] = role_order[1] if len(role_order) > 1 else ""
    return pd.DataFrame([row])


def build_domain_summary_payload(
    bundle: RepresentationInputBundle,
    domain_profile_df: pd.DataFrame,
    domain_burden_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> dict:
    burden_row = domain_burden_df.iloc[0].to_dict() if not domain_burden_df.empty else {}
    eligible = domain_profile_df.loc[domain_profile_df.get("eligible_for_burden", pd.Series(dtype=bool)).astype(bool)].copy()
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    if eligible.empty:
        return {
            "sample_id": bundle.sample_id,
            "cancer_type": bundle.cancer_type,
            "dominant_component": None,
            "secondary_component": None,
            "dominant_role": None,
            "secondary_role": None,
            "representative_domains": [],
            "summary_reliability_hint": "low",
        }
    comp_order = [axis_id for axis_id, _ in sorted(
        {axis_id: float(burden_row.get(f"{axis_id}_confidence_weighted_burden", 0.0)) for axis_id in component_ids}.items(),
        key=lambda x: (-x[1], x[0]),
    )]
    role_order = [axis_id for axis_id, _ in sorted(
        {axis_id: float(burden_row.get(f"{axis_id}_confidence_weighted_burden", 0.0)) for axis_id in role_ids}.items(),
        key=lambda x: (-x[1], x[0]),
    )]
    reps = eligible.sort_values(["spot_count", "domain_level_confidence", "domain_id"], ascending=[False, False, True]).head(max(3, int(cfg.scoring.top_programs_per_axis)))
    representative_domains = [
        {
            "domain_id": str(r["domain_id"]),
            "source_program_id": str(r["source_program_id"]),
            "spot_count": int(round(float(r["spot_count"]))),
            "domain_level_confidence": float(r["domain_level_confidence"]),
            "top_component": str(max(component_ids, key=lambda axis_id: float(r.get(axis_id, 0.0)))),
            "top_role": str(max(role_ids, key=lambda axis_id: float(r.get(axis_id, 0.0)))),
        }
        for _, r in reps.iterrows()
    ]
    reliability = "high" if int(eligible.shape[0]) >= 6 else "medium" if int(eligible.shape[0]) >= 3 else "low"
    return {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "dominant_component": _signature_record(burden_row, comp_order[0] if comp_order else None),
        "secondary_component": _signature_record(burden_row, comp_order[1] if len(comp_order) > 1 else None),
        "dominant_role": _signature_record(burden_row, role_order[0] if role_order else None),
        "secondary_role": _signature_record(burden_row, role_order[1] if len(role_order) > 1 else None),
        "representative_domains": representative_domains,
        "summary_reliability_hint": reliability,
        "structured_overview": (
            f"Domain-level macro portrait is centered on `{comp_order[0] if comp_order else 'NA'}` domains with "
            f"`{role_order[0] if role_order else 'NA'}` as the dominant spatial role."
        ),
    }


def build_domain_summary_markdown(summary: dict) -> str:
    reps = ", ".join([str(x.get("domain_id", "")) for x in summary.get("representative_domains", []) if x.get("domain_id")]) or "NA"
    return (
        "# Domain Macro Summary\n\n"
        f"- sample_id: `{summary.get('sample_id', '')}`\n"
        f"- cancer_type: `{summary.get('cancer_type', '')}`\n"
        f"- Domain-level dominant component: `{((summary.get('dominant_component') or {}).get('axis', 'NA'))}`\n"
        f"- Domain-level secondary component: `{((summary.get('secondary_component') or {}).get('axis', 'NA'))}`\n"
        f"- Domain-level dominant role: `{((summary.get('dominant_role') or {}).get('axis', 'NA'))}`\n"
        f"- Domain-level secondary role: `{((summary.get('secondary_role') or {}).get('axis', 'NA'))}`\n"
        f"- representative domains: `{reps}`\n"
        f"- summary_reliability_hint: `{summary.get('summary_reliability_hint', 'NA')}`\n\n"
        "## Structured Overview\n\n"
        f"{summary.get('structured_overview', '')}\n"
    )


def build_niche_profile_table(
    bundle: RepresentationInputBundle,
    niche_structures_df: pd.DataFrame,
    niche_membership_df: pd.DataFrame,
    niche_edges_df: pd.DataFrame,
    domain_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    if niche_structures_df.empty or niche_membership_df.empty or domain_profile_df.empty:
        cols = ["niche_id", "sample_id", "niche_member_count", "niche_confidence", "eligible_for_burden"]
        return pd.DataFrame(columns=cols + component_ids + role_ids)
    domain_keep = [
        "domain_id",
        "domain_key",
        "sample_id",
        "source_program_id",
        "spot_count",
        "domain_level_confidence",
        *component_ids,
        *role_ids,
    ]
    domain_ref = domain_profile_df.loc[:, [c for c in domain_keep if c in domain_profile_df.columns]].copy()
    membership = niche_membership_df.copy()
    membership = membership.merge(domain_ref, on="domain_key", how="left")
    membership["domain_level_confidence"] = pd.to_numeric(membership.get("domain_level_confidence", 0.0), errors="coerce").fillna(0.0)
    membership["spot_count"] = pd.to_numeric(membership.get("spot_count", 0.0), errors="coerce").fillna(0.0)
    membership["is_backbone_member"] = membership.get("is_backbone_member", pd.Series(dtype=bool)).fillna(False).astype(bool)
    membership["member_weight"] = (
        membership["spot_count"].clip(lower=0.0)
        * membership["domain_level_confidence"].clip(lower=0.0)
        * np.where(membership["is_backbone_member"], 1.15, 1.0)
    )
    if niche_edges_df.empty:
        niche_edge_stats = pd.DataFrame(columns=["niche_id", "member_edge_count", "mean_member_edge_strength"])
    else:
        member_pairs = membership.loc[:, ["niche_id", "domain_id"]].dropna().drop_duplicates()
        left = niche_edges_df.merge(member_pairs, left_on="domain_id_i", right_on="domain_id", how="inner").rename(columns={"niche_id": "niche_id_i"})
        both = left.merge(member_pairs, left_on="domain_id_j", right_on="domain_id", how="inner").rename(columns={"niche_id": "niche_id_j"})
        both = both.loc[both["niche_id_i"] == both["niche_id_j"]].copy()
        both["edge_strength"] = pd.to_numeric(both.get("edge_strength", 0.0), errors="coerce").fillna(0.0)
        niche_edge_stats = both.groupby("niche_id_i", dropna=False).agg(
            member_edge_count=("domain_id_i", "size"),
            mean_member_edge_strength=("edge_strength", "mean"),
        ).reset_index().rename(columns={"niche_id_i": "niche_id"})
    rows: list[dict] = []
    grouped_members = membership.groupby("niche_id", dropna=False)
    edge_map = niche_edge_stats.set_index("niche_id").to_dict(orient="index") if not niche_edge_stats.empty else {}
    for niche_row in niche_structures_df.itertuples(index=False):
        niche_id = str(getattr(niche_row, "niche_id", ""))
        members = grouped_members.get_group(niche_id).copy() if niche_id in grouped_members.groups else pd.DataFrame()
        if members.empty:
            continue
        weights = pd.to_numeric(members.get("member_weight", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights, dtype=np.float64)
        weights = weights / max(float(weights.sum()), 1e-8)
        member_count = int(getattr(niche_row, "member_count", members.shape[0]))
        backbone_count = int(getattr(niche_row, "backbone_node_count", 0))
        program_count = int(getattr(niche_row, "program_count", 0))
        edge_count = int(edge_map.get(niche_id, {}).get("member_edge_count", 0))
        possible_edge_count = max(member_count * max(member_count - 1, 0) / 2.0, 1.0)
        member_edge_density = clamp01(float(edge_count) / possible_edge_count) if member_count > 1 else 0.0
        backbone_fraction = clamp01(float(backbone_count) / max(float(member_count), 1.0))
        dominant_member_share = float(weights.max()) if weights.size else 1.0
        member_balance = clamp01(1.0 - dominant_member_share)
        multi_program_fraction = clamp01(float(program_count) / max(float(member_count), 1.0)) if member_count > 0 else 0.0
        contact_fraction = clamp01(float(getattr(niche_row, "contact_fraction", 0.0)))
        overlap_fraction = clamp01(float(getattr(niche_row, "overlap_fraction", 0.0)))
        proximity_fraction = clamp01(float(getattr(niche_row, "proximity_fraction", 0.0)))
        mean_member_edge_strength = float(edge_map.get(niche_id, {}).get("mean_member_edge_strength", 0.0))
        cohesion_strength = clamp01(0.6 * clamp01(mean_member_edge_strength) + 0.4 * member_edge_density)
        organizational_cohesion = clamp01(
            0.40 * cohesion_strength
            + 0.25 * backbone_fraction
            + 0.20 * clamp01(float(getattr(niche_row, "mean_edge_reliability", 0.0)))
            + 0.15 * saturating_score(float(member_count), 5.0)
        )
        interface_organization = clamp01(
            0.40 * contact_fraction
            + 0.20 * proximity_fraction
            + 0.20 * member_balance
            + 0.20 * multi_program_fraction
        )
        node_organization = clamp01(
            0.45 * dominant_member_share
            + 0.30 * clamp01(1.0 - saturating_score(float(member_count), 4.0))
            + 0.25 * clamp01(1.0 - member_edge_density)
        )
        companion_organization = clamp01(
            0.35 * member_balance
            + 0.25 * clamp01(1.0 - backbone_fraction)
            + 0.20 * clamp01(1.0 - cohesion_strength)
            + 0.20 * overlap_fraction
        )
        niche_conf = clamp01(
            0.45 * float(getattr(niche_row, "interaction_confidence", 0.0))
            + 0.30 * float(members["domain_level_confidence"].mean())
            + 0.15 * saturating_score(float(member_count), 6.0)
            + 0.10 * clamp01(float(getattr(niche_row, "mean_edge_reliability", 0.0)))
        )
        record = {
            "niche_id": niche_id,
            "sample_id": str(bundle.sample_id),
            "canonical_pattern_id": str(getattr(niche_row, "canonical_pattern_id", "")),
            "niche_member_count": member_count,
            "backbone_node_count": backbone_count,
            "program_count": program_count,
            "cross_program_edge_count": int(getattr(niche_row, "cross_program_edge_count", 0)),
            "strong_edge_count": int(getattr(niche_row, "strong_edge_count", 0)),
            "backbone_edge_count": int(getattr(niche_row, "backbone_edge_count", 0)),
            "contact_fraction": contact_fraction,
            "overlap_fraction": overlap_fraction,
            "proximity_fraction": proximity_fraction,
            "mean_edge_strength": float(getattr(niche_row, "mean_edge_strength", 0.0)),
            "mean_edge_reliability": float(getattr(niche_row, "mean_edge_reliability", 0.0)),
            "interaction_confidence": float(getattr(niche_row, "interaction_confidence", 0.0)),
            "member_edge_count": edge_count,
            "mean_member_edge_strength": mean_member_edge_strength,
            "member_edge_density": float(member_edge_density),
            "backbone_fraction": float(backbone_fraction),
            "dominant_member_share": float(dominant_member_share),
            "member_balance": float(member_balance),
            "multi_program_fraction": float(multi_program_fraction),
            "organizational_cohesion": float(organizational_cohesion),
            "interface_organization": float(interface_organization),
            "node_organization": float(node_organization),
            "companion_organization": float(companion_organization),
            "niche_confidence": float(niche_conf),
            "eligible_for_burden": bool(bool(getattr(niche_row, "basic_qc_pass", True)) and niche_conf > 0.0),
        }
        for axis_id in component_ids:
            values = pd.to_numeric(members.get(axis_id, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            base_comp = float(np.average(values, weights=weights))
            supporting_member_fraction = float(weights[values >= float(cfg.scoring.supported_axis_score_threshold)].sum()) if values.size else 0.0
            component_support_diversity = clamp01(supporting_member_fraction)
            organization_boost = clamp01(
                0.55 * component_support_diversity
                + 0.25 * member_balance
                + 0.20 * multi_program_fraction
            )
            comp = clamp01(0.82 * base_comp + 0.18 * base_comp * organization_boost)
            record[axis_id] = comp
            record[f"{axis_id}_composition_score"] = record[axis_id]
            record[f"{axis_id}_supporting_member_fraction"] = float(supporting_member_fraction)
        for axis_id in role_ids:
            values = pd.to_numeric(members.get(axis_id, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            base_comp = float(np.average(values, weights=weights))
            if axis_id == "scaffold_like":
                organization_signal = organizational_cohesion
            elif axis_id == "interface_like":
                organization_signal = interface_organization
            elif axis_id == "node_like":
                organization_signal = node_organization
            elif axis_id == "companion_like":
                organization_signal = companion_organization
            else:
                organization_signal = 0.0
            comp = clamp01(0.78 * base_comp + 0.22 * organization_signal)
            record[axis_id] = comp
            record[f"{axis_id}_composition_score"] = record[axis_id]
            record[f"{axis_id}_organization_signal"] = float(organization_signal)
        component_strengths = np.asarray([float(record[axis_id]) for axis_id in component_ids], dtype=np.float64)
        role_strengths = np.asarray([float(record[axis_id]) for axis_id in role_ids], dtype=np.float64)
        component_norm = component_strengths / max(float(component_strengths.sum()), 1e-8) if component_strengths.size else np.asarray([], dtype=np.float64)
        role_norm = role_strengths / max(float(role_strengths.sum()), 1e-8) if role_strengths.size else np.asarray([], dtype=np.float64)
        component_entropy = float(-np.sum(component_norm[component_norm > 0] * np.log(component_norm[component_norm > 0])) / np.log(max(len(component_ids), 2))) if component_norm.size else 0.0
        role_entropy = float(-np.sum(role_norm[role_norm > 0] * np.log(role_norm[role_norm > 0])) / np.log(max(len(role_ids), 2))) if role_norm.size else 0.0
        record["component_mix_diversity"] = clamp01(component_entropy)
        record["role_mix_diversity"] = clamp01(role_entropy)
        component_order = sorted([(axis_id, float(record[axis_id])) for axis_id in component_ids], key=lambda x: (-x[1], x[0]))
        role_order = sorted([(axis_id, float(record[axis_id])) for axis_id in role_ids], key=lambda x: (-x[1], x[0]))
        record["dominant_component_mix"] = component_order[0][0] if component_order else ""
        record["secondary_component_mix"] = component_order[1][0] if len(component_order) > 1 else ""
        record["dominant_role_mix"] = role_order[0][0] if role_order else ""
        record["secondary_role_mix"] = role_order[1][0] if len(role_order) > 1 else ""
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["sample_id", "niche_id"]).reset_index(drop=True)


def build_niche_burden_table(
    bundle: RepresentationInputBundle,
    niche_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> pd.DataFrame:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    eligible = niche_profile_df.loc[niche_profile_df.get("eligible_for_burden", pd.Series(dtype=bool)).astype(bool)].copy()
    row = {"sample_id": bundle.sample_id, "cancer_type": bundle.cancer_type, "eligible_niche_count": int(eligible.shape[0])}
    if eligible.empty:
        for axis_id in [*component_ids, *role_ids]:
            row[f"{axis_id}_raw_burden"] = 0.0
            row[f"{axis_id}_confidence_weighted_burden"] = 0.0
        row["dominant_component_by_burden"] = ""
        row["dominant_role_by_burden"] = ""
        return pd.DataFrame([row])
    eligible["raw_weight_base"] = pd.to_numeric(eligible.get("niche_member_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    eligible["confidence_weight_base"] = eligible["raw_weight_base"] * pd.to_numeric(eligible.get("niche_confidence", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(eligible["raw_weight_base"].sum()) <= 0.0:
        eligible["raw_weight_base"] = 1.0
    if float(eligible["confidence_weight_base"].sum()) <= 0.0:
        eligible["confidence_weight_base"] = 1.0
    eligible["raw_weight"] = eligible["raw_weight_base"] / float(eligible["raw_weight_base"].sum())
    eligible["confidence_weight"] = eligible["confidence_weight_base"] / float(eligible["confidence_weight_base"].sum())
    comp_burdens: dict[str, float] = {}
    role_burdens: dict[str, float] = {}
    for axis_id in component_ids:
        row[f"{axis_id}_raw_burden"] = float((eligible["raw_weight"] * eligible[axis_id]).sum())
        row[f"{axis_id}_confidence_weighted_burden"] = float((eligible["confidence_weight"] * eligible[axis_id]).sum())
        row[f"{axis_id}_representative_niche_count"] = int(((eligible[axis_id] >= float(cfg.scoring.supported_axis_score_threshold)) & (eligible["niche_confidence"] >= float(cfg.scoring.representative_program_min_confidence))).sum())
        comp_burdens[axis_id] = float(row[f"{axis_id}_confidence_weighted_burden"])
    for axis_id in role_ids:
        row[f"{axis_id}_raw_burden"] = float((eligible["raw_weight"] * eligible[axis_id]).sum())
        row[f"{axis_id}_confidence_weighted_burden"] = float((eligible["confidence_weight"] * eligible[axis_id]).sum())
        row[f"{axis_id}_representative_niche_count"] = int(((eligible[axis_id] >= float(cfg.scoring.supported_axis_score_threshold)) & (eligible["niche_confidence"] >= float(cfg.scoring.representative_program_min_confidence))).sum())
        role_burdens[axis_id] = float(row[f"{axis_id}_confidence_weighted_burden"])
    comp_order = [axis_id for axis_id, _ in sorted(comp_burdens.items(), key=lambda x: (-x[1], x[0]))]
    role_order = [axis_id for axis_id, _ in sorted(role_burdens.items(), key=lambda x: (-x[1], x[0]))]
    row["dominant_component_by_burden"] = comp_order[0] if comp_order else ""
    row["secondary_component_by_burden"] = comp_order[1] if len(comp_order) > 1 else ""
    row["dominant_role_by_burden"] = role_order[0] if role_order else ""
    row["secondary_role_by_burden"] = role_order[1] if len(role_order) > 1 else ""
    return pd.DataFrame([row])


def build_niche_summary_payload(
    bundle: RepresentationInputBundle,
    niche_profile_df: pd.DataFrame,
    niche_burden_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> dict:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    burden_row = niche_burden_df.iloc[0].to_dict() if not niche_burden_df.empty else {}
    eligible = niche_profile_df.loc[niche_profile_df.get("eligible_for_burden", pd.Series(dtype=bool)).astype(bool)].copy()
    if eligible.empty:
        return {
            "sample_id": bundle.sample_id,
            "cancer_type": bundle.cancer_type,
            "dominant_component_mix": None,
            "secondary_component_mix": None,
            "dominant_role_mix": None,
            "secondary_role_mix": None,
            "representative_niches": [],
            "summary_reliability_hint": "low",
        }
    comp_order = [axis_id for axis_id, _ in sorted(
        {axis_id: float(burden_row.get(f"{axis_id}_confidence_weighted_burden", 0.0)) for axis_id in component_ids}.items(),
        key=lambda x: (-x[1], x[0]),
    )]
    role_order = [axis_id for axis_id, _ in sorted(
        {axis_id: float(burden_row.get(f"{axis_id}_confidence_weighted_burden", 0.0)) for axis_id in role_ids}.items(),
        key=lambda x: (-x[1], x[0]),
    )]
    reps = eligible.sort_values(["niche_member_count", "niche_confidence", "niche_id"], ascending=[False, False, True]).head(max(3, int(cfg.scoring.top_programs_per_axis)))
    representative_niches = [
        {
            "niche_id": str(r["niche_id"]),
            "niche_member_count": int(r["niche_member_count"]),
            "niche_confidence": float(r["niche_confidence"]),
            "dominant_component_mix": str(r.get("dominant_component_mix", "")),
            "dominant_role_mix": str(r.get("dominant_role_mix", "")),
        }
        for _, r in reps.iterrows()
    ]
    reliability = "high" if int(eligible.shape[0]) >= 8 else "medium" if int(eligible.shape[0]) >= 4 else "low"
    return {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "dominant_component_mix": _signature_record(burden_row, comp_order[0] if comp_order else None),
        "secondary_component_mix": _signature_record(burden_row, comp_order[1] if len(comp_order) > 1 else None),
        "dominant_role_mix": _signature_record(burden_row, role_order[0] if role_order else None),
        "secondary_role_mix": _signature_record(burden_row, role_order[1] if len(role_order) > 1 else None),
        "representative_niches": representative_niches,
        "summary_reliability_hint": reliability,
        "structured_overview": (
            f"Niche-level macro portrait is centered on `{comp_order[0] if comp_order else 'NA'}` composition with "
            f"`{role_order[0] if role_order else 'NA'}` as the dominant role mix."
        ),
    }


def build_niche_summary_markdown(summary: dict) -> str:
    reps = ", ".join([str(x.get("niche_id", "")) for x in summary.get("representative_niches", []) if x.get("niche_id")]) or "NA"
    return (
        "# Niche Macro Summary\n\n"
        f"- sample_id: `{summary.get('sample_id', '')}`\n"
        f"- cancer_type: `{summary.get('cancer_type', '')}`\n"
        f"- dominant niche component composition: `{((summary.get('dominant_component_mix') or {}).get('axis', 'NA'))}`\n"
        f"- secondary niche component composition: `{((summary.get('secondary_component_mix') or {}).get('axis', 'NA'))}`\n"
        f"- dominant niche role composition: `{((summary.get('dominant_role_mix') or {}).get('axis', 'NA'))}`\n"
        f"- secondary niche role composition: `{((summary.get('secondary_role_mix') or {}).get('axis', 'NA'))}`\n"
        f"- representative niches: `{reps}`\n"
        f"- summary_reliability_hint: `{summary.get('summary_reliability_hint', 'NA')}`\n\n"
        "## Structured Overview\n\n"
        f"{summary.get('structured_overview', '')}\n"
    )


def build_qc_report(
    bundle: RepresentationInputBundle,
    eligibility_df: pd.DataFrame,
    program_profile_df: pd.DataFrame,
    component_axes: list[AxisDefinition],
    role_axes: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> dict:
    component_ids = [axis.axis_id for axis in component_axes]
    role_ids = [axis.axis_id for axis in role_axes]
    component_distribution = {
        axis_id: _safe_quantiles(program_profile_df[axis_id]) if axis_id in program_profile_df.columns else _safe_quantiles(pd.Series(dtype=float))
        for axis_id in component_ids
    }
    role_distribution = {
        axis_id: _safe_quantiles(program_profile_df[axis_id]) if axis_id in program_profile_df.columns else _safe_quantiles(pd.Series(dtype=float))
        for axis_id in role_ids
    }
    role_dynamic_range = {
        axis_id: float(
            pd.to_numeric(program_profile_df.get(axis_id, pd.Series(dtype=float)), errors="coerce").fillna(0.0).max()
            - pd.to_numeric(program_profile_df.get(axis_id, pd.Series(dtype=float)), errors="coerce").fillna(0.0).min()
        )
        for axis_id in role_ids
    }
    axis_supported_program_count = {
        axis_id: int((pd.to_numeric(program_profile_df.get(axis_id, 0.0), errors="coerce").fillna(0.0) >= float(cfg.scoring.supported_axis_score_threshold)).sum())
        for axis_id in [*component_ids, *role_ids]
    }
    axis_high_confidence_program_count = {
        axis_id: int(
            (
                (pd.to_numeric(program_profile_df.get(axis_id, 0.0), errors="coerce").fillna(0.0) >= float(cfg.scoring.supported_axis_score_threshold))
                & (pd.to_numeric(program_profile_df.get("overall_profile_confidence", 0.0), errors="coerce").fillna(0.0) >= float(cfg.scoring.high_confidence_profile_threshold))
                & (~program_profile_df.get("low_information_flag", pd.Series(dtype=bool)).astype(bool))
            ).sum()
        )
        for axis_id in [*component_ids, *role_ids]
    }
    routing_counts = eligibility_df["routing_status"].astype(str).value_counts().to_dict() if not eligibility_df.empty else {}
    validity_counts = eligibility_df["validity_status"].astype(str).value_counts().to_dict() if not eligibility_df.empty else {}
    eligibility_counts = eligibility_df["eligibility_status"].astype(str).value_counts().to_dict() if not eligibility_df.empty else {}
    low_info_count = int(program_profile_df.get("low_information_flag", pd.Series(dtype=bool)).astype(bool).sum()) if not program_profile_df.empty else 0
    annotation_missing_count = int((~program_profile_df.get("annotation_available_flag", pd.Series(dtype=bool)).astype(bool)).sum()) if not program_profile_df.empty else 0
    topology_missing_count = int((~program_profile_df.get("topology_available_flag", pd.Series(dtype=bool)).astype(bool)).sum()) if not program_profile_df.empty else 0
    deficit_gene_count = int(program_profile_df.get("information_deficit_gene", pd.Series(dtype=bool)).astype(bool).sum()) if not program_profile_df.empty else 0
    deficit_annotation_count = int(program_profile_df.get("information_deficit_annotation", pd.Series(dtype=bool)).astype(bool).sum()) if not program_profile_df.empty else 0
    deficit_topology_count = int(program_profile_df.get("information_deficit_topology", pd.Series(dtype=bool)).astype(bool).sum()) if not program_profile_df.empty else 0
    dominant_role_margin = 0.0
    dominant_role_name = ""
    if not program_profile_df.empty:
        role_means = {
            axis_id: float(pd.to_numeric(program_profile_df.get(axis_id, 0.0), errors="coerce").fillna(0.0).mean())
            for axis_id in role_ids
        }
        ordered_role = sorted(role_means.items(), key=lambda x: (-x[1], x[0]))
        if ordered_role:
            dominant_role_name = str(ordered_role[0][0])
            dominant_role_margin = float(ordered_role[0][1] - ordered_role[1][1]) if len(ordered_role) > 1 else float(ordered_role[0][1])
    inflammatory_margin_summary = {}
    if "inflammatory_stress" in program_profile_df.columns:
        infl = pd.to_numeric(program_profile_df["inflammatory_stress"], errors="coerce").fillna(0.0)
        inflammatory_margin_summary = {
            "vs_stromal_reactive_mean_gap": float(
                infl.mean() - pd.to_numeric(program_profile_df.get("stromal_reactive", 0.0), errors="coerce").fillna(0.0).mean()
            ),
            "vs_epithelial_like_mean_gap": float(
                infl.mean() - pd.to_numeric(program_profile_df.get("epithelial_like", 0.0), errors="coerce").fillna(0.0).mean()
            ),
            "supported_program_count": int((infl >= float(cfg.scoring.supported_axis_score_threshold)).sum()),
            "high_confidence_program_count": int(
                (
                    (infl >= float(cfg.scoring.supported_axis_score_threshold))
                    & (pd.to_numeric(program_profile_df.get("overall_profile_confidence", 0.0), errors="coerce").fillna(0.0) >= float(cfg.scoring.high_confidence_profile_threshold))
                    & (~program_profile_df.get("low_information_flag", pd.Series(dtype=bool)).astype(bool))
                ).sum()
            ),
        }
    component_overlap_diagnostics = {
        "component_multi_high_program_count": int(program_profile_df.get("component_multi_high_flag", pd.Series(dtype=bool)).astype(bool).sum()) if not program_profile_df.empty else 0,
        "component_triad_multi_high_program_count": int(program_profile_df.get("component_triad_multi_high_flag", pd.Series(dtype=bool)).astype(bool).sum()) if not program_profile_df.empty else 0,
        "component_primary_axis_counts": (
            program_profile_df.get("component_primary_axis", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
            if not program_profile_df.empty
            else {}
        ),
        "component_primary_margin_summary": _safe_quantiles(program_profile_df.get("component_primary_margin", pd.Series(dtype=float))),
    }
    component_mass_mean_map = {
        axis_id: float(pd.to_numeric(program_profile_df.get(axis_id, 0.0), errors="coerce").fillna(0.0).mean())
        for axis_id in component_ids
    }
    component_primary_conf_mean_map = {
        axis_id: float(
            pd.to_numeric(
                program_profile_df.loc[
                    program_profile_df.get("component_primary_axis", pd.Series(dtype=str)).astype(str) == axis_id,
                    "overall_profile_confidence",
                ],
                errors="coerce",
            ).fillna(0.0).mean()
        ) if not program_profile_df.empty else 0.0
        for axis_id in component_ids
    }
    component_burden_rank_map = _rank_map(component_mass_mean_map)
    component_representation_rank_map = _rank_map(component_primary_conf_mean_map)
    component_burden_vs_representation_disagreement = [
        axis_id for axis_id in component_ids
        if component_burden_rank_map.get(axis_id) != component_representation_rank_map.get(axis_id)
    ]
    axes_with_mass_dominance_but_low_representation_support = [
        axis_id for axis_id in component_ids
        if component_burden_rank_map.get(axis_id, 999) == 1 and component_representation_rank_map.get(axis_id, 999) > 1
    ]
    axes_with_high_representation_support_but_lower_mass = [
        axis_id for axis_id in component_ids
        if component_representation_rank_map.get(axis_id, 999) == 1 and component_burden_rank_map.get(axis_id, 999) > 1
    ]

    return {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "total_program_count": int(eligibility_df.shape[0]),
        "eligible_program_count": int(eligibility_df["eligible_for_burden"].astype(bool).sum()) if not eligibility_df.empty else 0,
        "default_use_count": int(routing_counts.get("default_use", 0)),
        "review_only_included_count": int(eligibility_counts.get("eligible_review_only_high_confidence", 0)),
        "invalid_count": int(validity_counts.get("invalid", 0)),
        "rejected_count": int(routing_counts.get("rejected", 0)),
        "low_information_profile_count": low_info_count,
        "annotation_missing_count": annotation_missing_count,
        "topology_missing_count": topology_missing_count,
        "routing_status_counts": {str(k): int(v) for k, v in routing_counts.items()},
        "validity_status_counts": {str(k): int(v) for k, v in validity_counts.items()},
        "eligibility_status_counts": {str(k): int(v) for k, v in eligibility_counts.items()},
        "axis_score_distribution_summary": component_distribution,
        "role_score_distribution_summary": role_distribution,
        "role_axis_dynamic_range": role_dynamic_range,
        "dominant_role_margin": float(dominant_role_margin),
        "dominant_role_name_by_mean": dominant_role_name,
        "component_axis_dynamic_range": {
            axis_id: float(
                pd.to_numeric(program_profile_df.get(axis_id, pd.Series(dtype=float)), errors="coerce").fillna(0.0).max()
                - pd.to_numeric(program_profile_df.get(axis_id, pd.Series(dtype=float)), errors="coerce").fillna(0.0).min()
            )
            for axis_id in component_ids
        },
        "component_overlap_diagnostics": component_overlap_diagnostics,
        "inflammatory_stress_margin_summary": inflammatory_margin_summary,
        "component_burden_vs_representation_disagreement": component_burden_vs_representation_disagreement,
        "axes_with_mass_dominance_but_low_representation_support": axes_with_mass_dominance_but_low_representation_support,
        "axes_with_high_representation_support_but_lower_mass": axes_with_high_representation_support_but_lower_mass,
        "sample_component_summary_diagnostics": {
            "burden_rank_by_component_mean": component_burden_rank_map,
            "representation_rank_by_primary_axis_confidence_mean": component_representation_rank_map,
        },
        "axis_supported_program_count": axis_supported_program_count,
        "axis_high_confidence_program_count": axis_high_confidence_program_count,
        "annotation_status": bundle.annotation_status,
        "topology_status": bundle.topology_status,
        "annotation_missing_handling": bundle.annotation_status.get(
            "missing_policy",
            "annotation evidence contributes zero score and gene/topology evidence remain active",
        ),
        "topology_missing_handling": bundle.topology_status.get(
            "missing_policy",
            "role axes retain weak annotation/gene hints but topology-led evidence contributes zero support",
        ),
        "profile_confidence_low_conditions": (
            "overall_profile_confidence is reduced when few evidence sources are informative, "
            "when annotation/topology are missing, or when axis support remains weak even for eligible programs."
        ),
        "low_confidence_profile_threshold": float(cfg.scoring.low_confidence_profile_threshold),
        "low_confidence_profile_count": int(
            (
                pd.to_numeric(program_profile_df.get("overall_profile_confidence", 0.0), errors="coerce")
                .fillna(0.0)
                < float(cfg.scoring.low_confidence_profile_threshold)
            ).sum()
        )
        if not program_profile_df.empty
        else 0,
        "activation_column_used": bundle.activation_col,
        "sample_burden_semantics": {
            "raw_burden": "normalized activation mass only",
            "confidence_weighted_burden": "normalized activation mass multiplied by overall_profile_confidence",
            "summary_default": "confidence_weighted_burden",
        },
        "low_information_source_summary": {
            "deficit_gene_count": deficit_gene_count,
            "deficit_annotation_count": deficit_annotation_count,
            "deficit_topology_count": deficit_topology_count,
        },
        "node_like_summary_diagnostics": {
            "node_like_burden_attenuation": float(cfg.scoring.node_like_sample_burden_attenuation),
            "node_like_dominant_min_margin": float(cfg.scoring.node_like_dominant_min_margin),
            "node_like_dominant_min_confidence_weighted_burden": float(cfg.scoring.node_like_dominant_min_confidence_weighted_burden),
            "node_like_dominant_max_competing_role_burden": float(cfg.scoring.node_like_dominant_max_competing_role_burden),
            "node_like_dominant_by_mean": bool(dominant_role_name == "node_like"),
            "node_like_dynamic_range": float(role_dynamic_range.get("node_like", 0.0)),
            "node_like_supported_program_count": int(axis_supported_program_count.get("node_like", 0)),
            "node_like_high_confidence_program_count": int(axis_high_confidence_program_count.get("node_like", 0)),
        },
    }
