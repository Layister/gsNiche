from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from ..schema import ProgramEvidence, RepresentationInputBundle, RepresentationPipelineConfig


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def saturating_score(value: float, scale: float) -> float:
    scale_value = max(float(scale), 1e-8)
    return clamp01(1.0 - math.exp(-max(float(value), 0.0) / scale_value))


def triangular_membership(value: float, left: float, center: float, right: float) -> float:
    v = float(value)
    if v <= left or v >= right:
        return 0.0
    if v == center:
        return 1.0
    if v < center:
        return clamp01((v - left) / max(center - left, 1e-8))
    return clamp01((right - v) / max(right - center, 1e-8))


def normalize_text_token(token: object) -> str:
    txt = str(token).strip().lower()
    txt = txt.replace("_", " ")
    txt = re.sub(r"[^a-z0-9\s]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def keyword_score(text_parts: Iterable[object], keywords: Iterable[str], saturation: int = 3) -> float:
    normalized_text = " || ".join(normalize_text_token(x) for x in text_parts if str(x).strip())
    if not normalized_text:
        return 0.0
    hits = 0
    for keyword in keywords:
        key = normalize_text_token(keyword)
        if key and key in normalized_text:
            hits += 1
    denom = max(int(saturation), 1)
    return clamp01(hits / denom)


def marker_rank_score(genes: tuple[str, ...], markers: tuple[str, ...], limit: int) -> tuple[float, int]:
    markers_upper = {str(x).strip().upper() for x in markers if str(x).strip()}
    if not genes or not markers_upper:
        return 0.0, 0
    score = 0.0
    hits = 0
    for rank, gene in enumerate(genes[: max(1, int(limit))], start=1):
        if str(gene).strip().upper() in markers_upper:
            hits += 1
            score += (max(1, int(limit)) - rank + 1) / max(1, int(limit))
    return clamp01(score / max(1.0, min(len(markers_upper), int(limit)))), hits


def _compute_component_metrics(active_indices: np.ndarray, neighbor_idx: np.ndarray) -> tuple[int, float, float]:
    active_set = set(int(x) for x in active_indices.tolist())
    if not active_set:
        return 0, 0.0, 0.0

    visited: set[int] = set()
    component_sizes: list[int] = []
    boundary_hits = 0
    purity_values: list[float] = []

    for node in active_set:
        neighbors_raw = neighbor_idx[node]
        valid_neighbors = [int(n) for n in neighbors_raw.tolist() if int(n) >= 0 and int(n) != node]
        if valid_neighbors:
            active_neighbor_count = sum(1 for n in valid_neighbors if n in active_set)
            purity_values.append(active_neighbor_count / len(valid_neighbors))
            if 0 < active_neighbor_count < len(valid_neighbors):
                boundary_hits += 1
        else:
            purity_values.append(0.0)

        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        comp_size = 0
        while stack:
            current = stack.pop()
            comp_size += 1
            for nbr in neighbor_idx[current].tolist():
                nbr_int = int(nbr)
                if nbr_int < 0 or nbr_int == current or nbr_int not in active_set or nbr_int in visited:
                    continue
                visited.add(nbr_int)
                stack.append(nbr_int)
        component_sizes.append(comp_size)

    component_count = len(component_sizes)
    largest_component_frac = (max(component_sizes) / max(1, len(active_set))) if component_sizes else 0.0
    boundary_fraction = boundary_hits / max(1, len(active_set))
    purity = float(np.mean(purity_values)) if purity_values else 0.0
    return component_count, clamp01(boundary_fraction), clamp01(purity * largest_component_frac)


def extract_program_evidence(
    bundle: RepresentationInputBundle,
    eligibility_df: pd.DataFrame,
    cfg: RepresentationPipelineConfig,
) -> list[ProgramEvidence]:
    programs_df = bundle.programs_df.copy()
    activation_df = bundle.activation_df.copy()

    weight_col = "gene_weight_identity_view" if "gene_weight_identity_view" in programs_df.columns else (
        "weight" if "weight" in programs_df.columns else None
    )
    if weight_col is None:
        programs_df["__gene_weight__"] = 0.0
        weight_col = "__gene_weight__"
    else:
        programs_df[weight_col] = pd.to_numeric(programs_df[weight_col], errors="coerce").fillna(0.0)

    by_program = {pid: sub.copy() for pid, sub in programs_df.groupby("program_id", sort=True)}
    act_by_program = {pid: sub.copy() for pid, sub in activation_df.groupby("program_id", sort=True)}
    eligibility_map = {
        str(row["program_id"]): dict(row)
        for row in eligibility_df.to_dict(orient="records")
    }
    spot_to_idx = (
        {str(spot_id): i for i, spot_id in enumerate(bundle.spot_ids.tolist())}
        if bundle.spot_ids is not None
        else {}
    )

    evidence_rows: list[ProgramEvidence] = []
    for program_id in sorted(by_program):
        gene_sub = by_program[program_id].sort_values(
            [weight_col, "gene"],
            ascending=[False, True],
        ).reset_index(drop=True)
        scaffold_sub = (
            gene_sub.loc[gene_sub.get("is_core_scaffold_gene", False).astype(bool)]
            if "is_core_scaffold_gene" in gene_sub.columns
            else gene_sub.iloc[0:0].copy()
        )
        top_genes = tuple(gene_sub["gene"].astype(str).tolist()[: max(1, int(cfg.scoring.top_gene_limit))])
        scaffold_genes = tuple(scaffold_sub["gene"].astype(str).tolist()[: max(1, int(cfg.scoring.scaffold_gene_limit))])

        ann = dict(bundle.annotation_map.get(program_id, {}))
        annotation_term_ids = tuple(str(x) for x in ann.get("displayed_term_ids", ann.get("significant_term_ids", [])) if str(x).strip())
        annotation_summary_text = str(ann.get("summary_text", "")).strip()
        annotation_confidence = _safe_float(ann.get("annotation_confidence", 0.0), default=0.0)
        annotation_available = bool(bundle.annotation_status.get("available")) and bool(ann)

        act_sub = act_by_program.get(program_id, pd.DataFrame(columns=activation_df.columns))
        if not act_sub.empty:
            weights = pd.to_numeric(act_sub[bundle.activation_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            weights = np.clip(weights, 0.0, None)
            act_sub = act_sub.assign(__activation_weight__=weights)
            active_spot_count = int(act_sub["spot_id"].astype(str).nunique())
            activation_mass = float(weights.sum())
            activation_mean_active = float(activation_mass / max(1, active_spot_count))
            top_n = max(1, int(math.ceil(weights.size * float(cfg.scoring.hotspot_top_fraction))))
            top_weights = np.sort(weights)[::-1][:top_n]
            activation_hotspot_share = float(top_weights.sum() / activation_mass) if activation_mass > 0 else 0.0
        else:
            active_spot_count = 0
            activation_mass = 0.0
            activation_mean_active = 0.0
            activation_hotspot_share = 0.0

        coverage = float(active_spot_count / max(1, bundle.total_spots))
        eligibility = eligibility_map.get(program_id, {})

        program_row = gene_sub.iloc[0].to_dict() if not gene_sub.empty else {}
        peakiness = _safe_float(
            eligibility.get("activation_peakiness", program_row.get("activation_peakiness", 0.0)),
            default=0.0,
        )
        entropy = _safe_float(
            eligibility.get("activation_entropy", program_row.get("activation_entropy", 0.0)),
            default=0.0,
        )
        sparsity = _safe_float(
            eligibility.get("activation_sparsity", program_row.get("activation_sparsity", 0.0)),
            default=0.0,
        )
        main_component_frac = _safe_float(
            eligibility.get("main_component_frac", program_row.get("main_component_frac", 0.0)),
            default=0.0,
        )
        high_activation_spot_count = int(
            round(
                _safe_float(
                    eligibility.get("high_activation_spot_count", program_row.get("high_activation_spot_count", 0.0)),
                    default=0.0,
                )
            )
        )

        topology_available = False
        boundary_fraction = 0.0
        local_purity = 0.0
        component_count = 0
        component_density = 0.0
        if (
            bundle.neighbor_idx is not None
            and bundle.spot_ids is not None
            and not act_sub.empty
            and "spot_id" in act_sub.columns
        ):
            active_indices = np.asarray(
                [spot_to_idx[sid] for sid in act_sub["spot_id"].astype(str).tolist() if sid in spot_to_idx],
                dtype=np.int64,
            )
            if active_indices.size > 0:
                topology_available = True
                component_count, boundary_fraction, local_purity = _compute_component_metrics(
                    np.unique(active_indices),
                    bundle.neighbor_idx,
                )
                component_density = clamp01(1.0 / max(1, component_count))

        evidence_rows.append(
            ProgramEvidence(
                sample_id=bundle.sample_id,
                cancer_type=bundle.cancer_type,
                program_id=str(program_id),
                validity_status=str(eligibility.get("validity_status", program_row.get("validity_status", "unknown"))),
                routing_status=str(eligibility.get("routing_status", program_row.get("routing_status", "unknown"))),
                eligibility_status=str(eligibility.get("eligibility_status", "excluded_other")),
                eligible_for_burden=bool(eligibility.get("eligible_for_burden", False)),
                program_confidence=_safe_float(eligibility.get("program_confidence", program_row.get("program_confidence", 0.0))),
                template_evidence_score=_safe_float(
                    eligibility.get("template_evidence_score", program_row.get("program_template_evidence_score", 0.0))
                ),
                default_use_support_score=_safe_float(
                    eligibility.get("default_use_support_score", program_row.get("default_use_support_score", 0.0))
                ),
                redundancy_status=str(eligibility.get("redundancy_status", program_row.get("redundancy_status", "unknown"))),
                top_genes=top_genes,
                scaffold_genes=scaffold_genes,
                annotation_term_ids=annotation_term_ids,
                annotation_summary_text=annotation_summary_text,
                annotation_confidence=annotation_confidence,
                annotation_available=annotation_available,
                activation_mass=activation_mass,
                activation_coverage=coverage,
                active_spot_count=active_spot_count,
                activation_mean_active=activation_mean_active,
                activation_hotspot_share=activation_hotspot_share,
                activation_peakiness=peakiness,
                activation_entropy=entropy,
                activation_sparsity=sparsity,
                main_component_frac=main_component_frac,
                high_activation_spot_count=high_activation_spot_count,
                topology_available=topology_available,
                topology_boundary_fraction=boundary_fraction,
                topology_local_purity=local_purity,
                topology_component_count=component_count,
                topology_component_density=component_density,
            )
        )

    return evidence_rows
