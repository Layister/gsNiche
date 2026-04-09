from __future__ import annotations

from ..schema import AxisDefinition, ProgramEvidence, RepresentationPipelineConfig
from .evidence_extractors import clamp01, keyword_score, marker_rank_score


def _component_weights(axis: AxisDefinition, cfg: RepresentationPipelineConfig) -> tuple[float, float]:
    gene_weight = float(axis.weights.get("gene", cfg.scoring.component_gene_weight))
    annotation_weight = float(axis.weights.get("annotation", cfg.scoring.component_annotation_weight))
    total = gene_weight + annotation_weight
    if total <= 0.0:
        return float(cfg.scoring.component_gene_weight), float(cfg.scoring.component_annotation_weight)
    return gene_weight / total, annotation_weight / total


def _annotation_gate(axis: AxisDefinition, gene_score: float, annotation_score: float) -> tuple[float, float]:
    gating = dict(axis.gating or {})
    effective_annotation = float(annotation_score)
    annotation_gate = 1.0
    if gating.get("annotation_requires_gene_support", False):
        floor = float(gating.get("annotation_gene_support_floor", gating.get("require_min_gene_support", 0.0)) or 0.0)
        full_at = float(gating.get("annotation_gene_support_full_at", max(floor, 0.20)) or max(floor, 0.20))
        if full_at <= floor:
            full_at = floor + 1e-6
        annotation_gate = clamp01((float(gene_score) - floor) / (full_at - floor))
        effective_annotation *= annotation_gate
    else:
        soft_floor = float(gating.get("annotation_soft_gene_floor", 0.0) or 0.0)
        if float(gene_score) < soft_floor:
            annotation_gate = float(gating.get("max_annotation_multiplier_without_gene_support", 1.0) or 1.0)
            effective_annotation *= annotation_gate
    return clamp01(effective_annotation), clamp01(annotation_gate)


def _gene_support_gate(axis: AxisDefinition, gene_score: float) -> float:
    min_gene = float(dict(axis.gating or {}).get("require_min_gene_support", 0.0) or 0.0)
    if min_gene <= 0.0:
        return 1.0
    return clamp01(float(gene_score) / min_gene)


def _overlap_penalty(
    axis: AxisDefinition,
    axis_id: str,
    provisional_scores: dict[str, float],
    provisional_details: dict[str, dict],
) -> float:
    gating = dict(axis.gating or {})
    peers = [str(x) for x in gating.get("overlap_penalty_axes", []) if str(x).strip()]
    strength = float(gating.get("overlap_penalty_strength", 0.0) or 0.0)
    start = float(gating.get("overlap_penalty_start", 0.0) or 0.0)
    gene_protect_at = float(gating.get("overlap_penalty_gene_protect_at", 1.0) or 1.0)
    if not peers or strength <= 0.0:
        return 0.0
    own_score = float(provisional_scores.get(axis_id, 0.0))
    peer_score = max([float(provisional_scores.get(peer, 0.0)) for peer in peers], default=0.0)
    if own_score <= start or peer_score <= start:
        return 0.0
    own_term = clamp01((own_score - start) / max(1e-6, 1.0 - start))
    peer_term = clamp01((peer_score - start) / max(1e-6, 1.0 - start))
    gene_score = float(provisional_details.get(axis_id, {}).get("gene_score", 0.0))
    gene_protection = 1.0 - 0.6 * clamp01(gene_score / max(gene_protect_at, 1e-6))
    return clamp01(strength * own_term * peer_term * gene_protection)


def score_component_axes(
    evidence: ProgramEvidence,
    axis_definitions: list[AxisDefinition],
    cfg: RepresentationPipelineConfig,
) -> tuple[dict[str, float], dict[str, dict]]:
    # Component axes are intentionally content-led: scaffold/ranked genes define the axis,
    # and annotation only nudges the score instead of dominating it.
    provisional_scores: dict[str, float] = {}
    provisional_details: dict[str, dict] = {}
    for axis in axis_definitions:
        scaffold_score, scaffold_hits = marker_rank_score(
            evidence.scaffold_genes,
            axis.positive_gene_markers or axis.gene_markers,
            cfg.scoring.scaffold_gene_limit,
        )
        top_score, top_hits = marker_rank_score(
            evidence.top_genes,
            axis.positive_gene_markers or axis.gene_markers,
            cfg.scoring.top_gene_limit,
        )
        gene_score = clamp01(0.65 * scaffold_score + 0.35 * top_score)
        ann_score = keyword_score(
            [*evidence.annotation_term_ids, evidence.annotation_summary_text],
            axis.positive_annotation_terms or axis.annotation_keywords,
            saturation=cfg.scoring.annotation_keyword_saturation,
        )
        effective_ann_score, annotation_gate = _annotation_gate(axis, gene_score, ann_score)
        gene_weight, annotation_weight = _component_weights(axis, cfg)
        gene_support_gate = _gene_support_gate(axis, gene_score)
        base_score = clamp01(
            (
                gene_weight * gene_score
                + annotation_weight * effective_ann_score
            )
            * gene_support_gate
        )
        provisional_scores[axis.axis_id] = base_score
        provisional_details[axis.axis_id] = {
            "gene_score": float(gene_score),
            "annotation_score": float(ann_score),
            "effective_annotation_score": float(effective_ann_score),
            "annotation_gate": float(annotation_gate),
            "gene_support_gate": float(gene_support_gate),
            "gene_weight": float(gene_weight),
            "annotation_weight": float(annotation_weight),
            "base_score": float(base_score),
            "scaffold_marker_hits": int(scaffold_hits),
            "top_marker_hits": int(top_hits),
        }

    scores: dict[str, float] = {}
    details: dict[str, dict] = {}
    for axis in axis_definitions:
        overlap_penalty = _overlap_penalty(
            axis=axis,
            axis_id=axis.axis_id,
            provisional_scores=provisional_scores,
            provisional_details=provisional_details,
        )
        final_score = clamp01(float(provisional_scores.get(axis.axis_id, 0.0)) - overlap_penalty)
        scores[axis.axis_id] = final_score
        details[axis.axis_id] = {
            **provisional_details.get(axis.axis_id, {}),
            "overlap_penalty": float(overlap_penalty),
            "final_score": float(final_score),
        }
    return scores, details
