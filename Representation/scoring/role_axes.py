from __future__ import annotations

from ..schema import AxisDefinition, ProgramEvidence, RepresentationPipelineConfig
from .evidence_extractors import clamp01, keyword_score, saturating_score, triangular_membership


def _topology_memberships(evidence: ProgramEvidence) -> dict[str, float]:
    coverage = clamp01(evidence.activation_coverage)
    hotspot = clamp01(evidence.activation_hotspot_share)
    peakiness = clamp01(evidence.activation_peakiness)
    entropy = clamp01(evidence.activation_entropy)
    main_component_frac = clamp01(evidence.main_component_frac)
    boundary_fraction = clamp01(evidence.topology_boundary_fraction)
    mixed_neighbors = clamp01(1.0 - evidence.topology_local_purity) if evidence.topology_available else 0.0
    connectedness = clamp01(evidence.topology_component_density if evidence.topology_available else main_component_frac)

    return {
        "broad_coverage": saturating_score(coverage, 0.18),
        "moderate_coverage": triangular_membership(coverage, 0.03, 0.15, 0.42),
        "low_coverage": clamp01(1.0 - saturating_score(coverage, 0.12)),
        "high_hotspot": clamp01(0.6 * hotspot + 0.4 * peakiness),
        "mid_hotspot": triangular_membership(hotspot, 0.18, 0.42, 0.72),
        "moderate_hotspot": triangular_membership(hotspot, 0.10, 0.32, 0.58),
        "low_hotspot": clamp01(1.0 - hotspot),
        "high_peakiness": peakiness,
        "high_entropy": entropy,
        "single_component": main_component_frac,
        "boundary_fraction": boundary_fraction if evidence.topology_available else triangular_membership(coverage, 0.04, 0.18, 0.35),
        "mixed_neighbors": mixed_neighbors if evidence.topology_available else triangular_membership(coverage, 0.05, 0.18, 0.36),
        "connectedness": connectedness,
        "non_dominant": clamp01(1.0 - max(peakiness, hotspot)),
    }


def score_role_axes(
    evidence: ProgramEvidence,
    axis_definitions: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> tuple[dict[str, float], dict[str, dict]]:
    # Role axes are topology-led. These memberships are coarse, interpretable hints for
    # interface/scaffold/node/companion behavior, not a generic pattern classifier.
    topo = _topology_memberships(evidence)
    interface_core = min(topo["boundary_fraction"], topo["mixed_neighbors"])
    interface_context = 0.65 * topo["moderate_coverage"] + 0.35 * topo["mid_hotspot"]
    interface_penalty = max(topo["broad_coverage"], topo["high_hotspot"])
    interface_like = clamp01(
        0.70 * interface_core
        + 0.30 * interface_context
        - 0.20 * interface_penalty
    )
    scaffold_like = clamp01(
        0.35 * topo["broad_coverage"]
        + 0.25 * topo["high_entropy"]
        + 0.20 * topo["low_hotspot"]
        + 0.20 * topo["connectedness"]
    )
    node_like = clamp01(
        0.35 * topo["high_hotspot"]
        + 0.25 * topo["high_peakiness"]
        + 0.20 * topo["low_coverage"]
        + 0.20 * topo["single_component"]
    )
    companion_like = clamp01(
        0.35 * topo["moderate_coverage"]
        + 0.20 * topo["moderate_hotspot"]
        + 0.25 * topo["non_dominant"]
        + 0.20 * clamp01(1.0 - max(interface_like, node_like))
    )
    topology_score_map = {
        "interface_like": interface_like,
        "scaffold_like": scaffold_like,
        "node_like": node_like,
        "companion_like": companion_like,
    }

    scores: dict[str, float] = {}
    details: dict[str, dict] = {}
    for axis in axis_definitions:
        ann_score = keyword_score(
            [*evidence.annotation_term_ids, evidence.annotation_summary_text],
            axis.annotation_keywords,
            saturation=cfg.scoring.annotation_keyword_saturation,
        )
        gene_hint = 0.0
        if axis.axis_id == "scaffold_like":
            gene_hint = keyword_score(evidence.top_genes, ("COL", "FN1", "POSTN", "VWF", "PECAM1"), saturation=2)
        elif axis.axis_id == "node_like":
            gene_hint = keyword_score(evidence.top_genes, ("MKI67", "TOP2A", "BIRC5", "CXCL8"), saturation=2)
        elif axis.axis_id == "interface_like":
            gene_hint = keyword_score(evidence.top_genes, ("CXCL9", "CXCL10", "VCAM1", "SELE", "KRT19"), saturation=2)
        else:
            gene_hint = keyword_score(evidence.top_genes, ("TFF3", "REG4", "CCL2", "LCN2"), saturation=2)
        topology_score = topology_score_map.get(axis.axis_id, 0.0)
        if not evidence.topology_available:
            topology_score *= 0.25
        final_score = clamp01(
            cfg.scoring.role_topology_weight * topology_score
            + cfg.scoring.role_annotation_weight * ann_score
            + cfg.scoring.role_gene_weight * gene_hint
        )
        scores[axis.axis_id] = final_score
        details[axis.axis_id] = {
            "topology_score": float(topology_score),
            "annotation_score": float(ann_score),
            "gene_hint_score": float(gene_hint),
            "boundary_fraction": float(topo["boundary_fraction"]),
            "mixed_neighbors": float(topo["mixed_neighbors"]),
            "moderate_coverage": float(topo["moderate_coverage"]),
            "mid_hotspot": float(topo["mid_hotspot"]),
            "broad_coverage": float(topo["broad_coverage"]),
            "high_hotspot": float(topo["high_hotspot"]),
            "interface_core": float(interface_core),
            "interface_context": float(interface_context),
            "interface_penalty": float(interface_penalty),
            "node_like_hotspot_component": float(topo["high_hotspot"]),
            "node_like_peakiness_component": float(topo["high_peakiness"]),
            "node_like_low_coverage_component": float(topo["low_coverage"]),
            "node_like_single_component_component": float(topo["single_component"]),
            "node_like_penalty_component": float(max(interface_like, scaffold_like)),
        }
    return scores, details
