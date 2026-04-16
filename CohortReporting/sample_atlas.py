from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .loaders import CohortReportingInputs, SampleReportingBundle
from .schema import CohortReportingConfig
from .tables import derive_support_spread, encode_full_axis_strip, encode_supported_axes


SECTION_QUESTIONS = {
    "program": "Which Program mechanisms are present in this sample, and what are their main component compositions?",
    "domain": "How do these Programs land in space, and what broad Domain shapes do they form?",
    "niche": "Which semantically distinct Domains come together locally, and how do they contact into local structures?",
}

SECTION_LABELS = {
    "program": "Program",
    "domain": "Domain",
    "niche": "Niche",
}

OBJECT_ID_COLUMNS = {
    "program": "program_id",
    "domain": "domain_id",
    "niche": "niche_id",
}

CONFIDENCE_COLUMNS = {
    "program": "overall_profile_confidence",
    "domain": "domain_level_confidence",
    "niche": "niche_confidence",
}

PROGRAM_STATIC_FIGURE_NAME = "program_composition_overview"
PROGRAM_STATIC_FIGURE_ROLE = "lightweight_program_composition_overview"
PROGRAM_STATIC_OBSERVATION_UNIT = "program_rows"

DOMAIN_PRIMARY_QUESTION = "program_spatial_deployment"
DOMAIN_VISUALIZATION_PRIMARY_MODE = "one_static_matrix_plus_one_interactive_viewer"
DOMAIN_PRIMARY_VISUALIZATION_MODE = "single_deployment_matrix_plus_interactive_viewer"
DOMAIN_STATIC_FIGURE_NAME = "program_domain_deployment_matrix"
DOMAIN_STATIC_FIGURE_ROLE = "high_density_overview_matrix"
DOMAIN_STATIC_OBSERVATION_UNIT = "program_rows"
DOMAIN_STATIC_FIGURE_QUESTION = "how_programs_are_partitioned_and_deployed_in_space"
DOMAIN_BRIDGE_MATRIX_PROGRAM_SCOPE = "all_programs_in_sample"
DOMAIN_SPATIAL_VIEWER_ROLE = "interactive_program_domain_overlap_viewer"
DOMAIN_SPATIAL_VIEWER_MODE = "single_fixed_canvas"
DOMAIN_SPATIAL_VIEWER_DATA_MODE = "precomputed_only"
DOMAIN_SPATIAL_GEOMETRY_SOURCE = "spot_membership_contours_from_precomputed_spatial_coords"
DOMAIN_SPATIAL_VIEWER_COLOR_MODE = "stable_program_color_mapping"
DOMAIN_SPATIAL_VIEWER_BOUNDARY_HIGHLIGHT = "representative_or_high_coverage"
DOMAIN_SPATIAL_VIEWER_LABEL_MODE = "hover_or_representative_only"
DOMAIN_FOOTPRINT_NOTE = (
    "Viewer contours are generated from precomputed Domain spot memberships and spatial coordinates to show relative position, clustering, separation, and overlap trends. "
    "They are overlap-view contours for reporting, not exact boundary-survey reconstructions."
)
BRIDGE_MATRIX_SPLIT_ENCODING = "three_fixed_horizontal_microbars"
BRIDGE_MATRIX_DEPLOYMENT_ENCODING = "four_fixed_order_heat_cells"
BRIDGE_MATRIX_MORPHOLOGY_ENCODING = "three_fixed_order_heat_cells"
NICHE_VISUALIZATION_PRIMARY_MODE = "single_assembly_matrix_only"
NICHE_PRIMARY_QUESTION = "how_semantically_distinct_domains_assemble_into_local_structures"
NICHE_STATIC_FIGURE_NAME = "sample_level_niche_assembly_matrix"
NICHE_STATIC_FIGURE_ROLE = "full_sample_local_structure_assembly_overview"
NICHE_STATIC_OBSERVATION_UNIT = "niche_rows"
NICHE_SPATIAL_VIEWER_ROLE = "interactive_niche_member_spatial_viewer"
NICHE_SPATIAL_VIEWER_MODE = "single_fixed_canvas"
NICHE_SPATIAL_VIEWER_SELECTION_MODE = "single_niche_only"
NICHE_SPATIAL_VIEWER_GEOMETRY_MODE = "member_domain_spot_membership_contours"
NICHE_SPATIAL_VIEWER_NICHE_ORDER = "representative_then_confidence_then_member_count_then_niche_id"
NICHE_SPATIAL_VIEWER_PROGRAM_COLOR_MODE = "sample_stable_and_domain_viewer_aligned_when_available"
NICHE_FOOTPRINT_NOTE = (
    "Viewer contours are generated from precomputed member-Domain spot memberships and spatial coordinates to show relative position, separation, adjacency, and local overlap trends. "
    "They are local-assembly footprints for reporting, not exact boundary reconstructions."
)
NICHE_MEMBER_PROGRAM_COMPOSITION_ENCODING = "fixed_order_recipe_strip"
NICHE_MEMBER_ROLE_COMPOSITION_ENCODING = "fixed_order_recipe_strip"
NICHE_MEMBER_PROGRAM_COMPOSITION_ROLE = "primary proxy for member-domain semantic composition"
NICHE_ROLE_AXIS_ORDER = ["scaffold_like", "interface_like", "node_like", "companion_like"]
NICHE_ROLE_COMPOSITION_MISSING_AXIS_POLICY = "keep_fixed_slot_with_zero_or_empty_value"
CONTACT_STRUCTURE_HINT_VOCAB = [
    "edge_contact",
    "node_focal",
    "mixed_mesh",
    "scaffold_attached_interface",
    "parallel_contact",
    "diffuse_contact",
    "compact_contact",
]
ASSEMBLY_PATTERN_LABEL_VOCAB = [
    "scaffold_edge",
    "mixed_interface",
    "node_focal",
    "multi_component_mesh",
    "compact_single_source",
    "distributed_multi_source",
    "interface_bridged",
]

SPLIT_THRESHOLD_HIGH_SHARE = 0.65
SPLIT_THRESHOLD_LOW_SHARE = 0.35
SPLIT_THRESHOLD_HIGH_DOMAIN_COUNT = 5
SPLIT_THRESHOLD_HIGH_TOTAL_SPOTS = 30.0
TRANSPORT_THRESHOLD_HIGH_CHAIN_COVERAGE = 0.20
TRANSPORT_THRESHOLD_HIGH_CONCENTRATION = 0.55
TRANSPORT_THRESHOLD_LOW_DOMAIN_COUNT = 2


@dataclass
class LayerAtlasArtifacts:
    overview_df: pd.DataFrame
    payload: dict[str, Any]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def _layer_profile_df(bundle: SampleReportingBundle, layer_name: str) -> pd.DataFrame:
    if layer_name == "program":
        return bundle.program_profile_df.copy()
    if layer_name == "domain":
        return bundle.domain_profile_df.copy()
    return bundle.niche_profile_df.copy()


def _object_id_column(layer_name: str) -> str:
    return OBJECT_ID_COLUMNS[layer_name]


def _confidence_column(layer_name: str) -> str:
    return CONFIDENCE_COLUMNS[layer_name]


def _leading_anchor(values: np.ndarray, axis_ids: list[str]) -> str:
    if values.size == 0 or not axis_ids:
        return ""
    idx = int(np.argmax(values))
    if idx >= len(axis_ids):
        return ""
    return str(axis_ids[idx])


def _representative_ids(bundle: SampleReportingBundle, layer_name: str) -> set[str]:
    if layer_name == "program":
        summary = bundle.program_summary or {}
        reps = {str(item.get("program_id", "")) for item in summary.get("top_programs", []) if item.get("program_id")}
        for key in ("representative_objects_by_component_axis", "representative_objects_by_role_axis"):
            for item in summary.get(key, []) or []:
                obj_id = str(item.get("program_id", "") or item.get("object_id", ""))
                if obj_id:
                    reps.add(obj_id)
        return reps
    if layer_name == "domain":
        summary = bundle.domain_summary or {}
        return {str(item.get("domain_id", "")) for item in summary.get("representative_domains", []) if item.get("domain_id")}
    summary = bundle.niche_summary or {}
    return {str(item.get("niche_id", "")) for item in summary.get("representative_niches", []) if item.get("niche_id")}


def _group_order_map(component_axes: list[str], role_axes: list[str]) -> dict[str, int]:
    order: dict[str, int] = {}
    for idx, axis_id in enumerate(component_axes + role_axes):
        order[str(axis_id)] = idx
    return order


def _spread_values(layer_name: str, row: pd.Series, component_values: np.ndarray, role_values: np.ndarray) -> tuple[float, float]:
    if layer_name == "niche":
        return (
            _safe_float(row.get("component_mix_diversity", 0.0), derive_support_spread(component_values)),
            _safe_float(row.get("role_mix_diversity", 0.0), derive_support_spread(role_values)),
        )
    return derive_support_spread(component_values), derive_support_spread(role_values)


def _domain_morphology_hint(boundary_ratio: float, mixed_neighbor_fraction: float, elongation: float) -> str:
    if boundary_ratio >= 0.50 and mixed_neighbor_fraction < 0.30:
        return "boundary_skewed"
    if mixed_neighbor_fraction >= 0.40:
        return "mixed_neighbor"
    if elongation >= 2.0:
        return "elongated"
    return "compact"


def _program_table_records(overview_df: pd.DataFrame) -> list[dict[str, Any]]:
    if overview_df.empty:
        return []
    display = overview_df.loc[
        :,
        ["program_id", "leading_component_anchor", "leading_role_anchor", "confidence"],
    ].copy()
    display = display.rename(
        columns={
            "program_id": "program_id",
            "leading_component_anchor": "main_component",
            "leading_role_anchor": "aux_role",
            "confidence": "confidence",
        }
    )
    display["confidence"] = pd.to_numeric(display["confidence"], errors="coerce").fillna(0.0).round(3)
    return display.to_dict(orient="records")


def _domain_table_records(overview_df: pd.DataFrame) -> list[dict[str, Any]]:
    if overview_df.empty:
        return []
    display_cols = [
        "domain_key",
        "domain_id",
        "source_program_id",
        "morphology_hint",
        "spot_count",
        "geo_boundary_ratio",
        "mixed_neighbor_fraction",
        "representative_status",
    ]
    display = overview_df.loc[:, [col for col in display_cols if col in overview_df.columns]].copy()
    for col in ("geo_boundary_ratio", "mixed_neighbor_fraction"):
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").fillna(0.0).round(3)
    return display.to_dict(orient="records")


def _niche_table_records(niche_matrix_df: pd.DataFrame) -> list[dict[str, Any]]:
    if niche_matrix_df.empty:
        return []
    display = niche_matrix_df.loc[
        :,
        ["niche_id", "member_program_composition", "dominant_contact_pair", "member_count", "niche_confidence"],
    ].copy()
    display = display.rename(
        columns={
            "member_program_composition": "member_recipe",
            "dominant_contact_pair": "main_contact",
            "member_count": "member_count",
            "niche_confidence": "confidence",
        }
    )
    display["confidence"] = pd.to_numeric(display["confidence"], errors="coerce").fillna(0.0).round(3)
    return display.to_dict(orient="records")


def _support_counts(df: pd.DataFrame, axis_ids: list[str], threshold: float) -> list[dict[str, Any]]:
    counts: list[dict[str, Any]] = []
    if df.empty:
        return counts
    for axis_id in axis_ids:
        values = pd.to_numeric(df.get(axis_id, 0.0), errors="coerce").fillna(0.0)
        counts.append(
            {
                "axis": axis_id,
                "supported_object_count": int((values >= float(threshold)).sum()),
                "mean_score": float(values.mean()) if not values.empty else 0.0,
                "max_score": float(values.max()) if not values.empty else 0.0,
            }
        )
    return counts


def _quantile_snapshot(values: pd.Series | np.ndarray) -> dict[str, float]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def _encode_quantile_snapshot(values: pd.Series | np.ndarray) -> str:
    snap = _quantile_snapshot(values)
    return ";".join(f"{key}={value:.3f}" for key, value in snap.items())


def _herfindahl_concentration(counts: pd.Series) -> float:
    series = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(series.sum())
    if total <= 0.0:
        return 0.0
    probs = series / total
    return float(np.clip(np.square(probs).sum(), 0.0, 1.0))


def _encode_composition_strip(value_map: dict[str, float], axis_order: list[str]) -> str:
    return ";".join(f"{axis}={float(value_map.get(axis, 0.0)):.3f}" for axis in axis_order)


def _normalize_count_map(values: pd.Series) -> dict[str, float]:
    counts = values.astype(str).value_counts()
    total = float(counts.sum())
    if total <= 0:
        return {}
    return {str(k): float(v / total) for k, v in counts.items()}


def _pair_sort_key(pair: str) -> tuple[str, str]:
    if "↔" not in str(pair):
        return (str(pair), "")
    left, right = str(pair).split("↔", 1)
    return (left, right)


def _role_pair_label(left: str, right: str) -> str:
    a, b = sorted([str(left), str(right)])
    return f"{a}↔{b}"


def _contact_hint_from_pair(
    dominant_pair: str,
    secondary_pair: str,
    program_count: int,
    backbone_pair_count: int,
    cross_program_edge_count: int,
) -> str:
    pair = str(dominant_pair)
    if "scaffold_like" in pair and "interface_like" in pair:
        return "scaffold_attached_interface"
    if "node_like" in pair:
        return "node_focal"
    if backbone_pair_count >= 4 or cross_program_edge_count >= 6:
        return "mixed_mesh"
    if program_count <= 2 and backbone_pair_count <= 1:
        return "compact_contact"
    if secondary_pair and secondary_pair == dominant_pair:
        return "parallel_contact"
    if program_count >= 4:
        return "diffuse_contact"
    return "edge_contact"


def _assembly_label_from_hint(
    contact_hint: str,
    dominant_pair: str,
    program_count: int,
    member_count: int,
) -> str:
    if program_count <= 1:
        return "compact_single_source"
    if contact_hint == "scaffold_attached_interface":
        return "scaffold_edge"
    if contact_hint == "node_focal":
        return "node_focal"
    if contact_hint == "mixed_mesh":
        return "multi_component_mesh"
    if "interface_like" in str(dominant_pair):
        return "interface_bridged" if member_count >= 6 else "mixed_interface"
    return "distributed_multi_source"


def build_niche_assembly_matrix_df(
    bundle: SampleReportingBundle,
    inputs: CohortReportingInputs,
) -> pd.DataFrame:
    niche_df = bundle.niche_profile_df.copy()
    membership_df = bundle.niche_membership_df.copy()
    structures_df = bundle.niche_structures_df.copy()
    if niche_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "niche_id",
                "member_count",
                "niche_confidence",
                "organizational_cohesion",
                "member_program_composition",
                "member_role_composition",
                "dominant_contact_pair",
                "secondary_contact_pair",
                "contact_structure_hint",
                "niche_size",
                "dominant_contact_pair_sort_key",
                "secondary_contact_pair_sort_key",
                "member_program_composition_sort_key",
            ]
        )
    representative_niche_ids = _representative_ids(bundle, "niche")
    domain_df = bundle.domain_profile_df.copy()
    domain_df["domain_key"] = domain_df.get("domain_key", pd.Series(dtype=str)).astype(str)
    domain_df["source_program_id"] = domain_df.get("source_program_id", pd.Series(dtype=str)).astype(str)
    membership_df["domain_key"] = membership_df.get("domain_key", pd.Series(dtype=str)).astype(str)
    membership_df["niche_id"] = membership_df.get("niche_id", pd.Series(dtype=str)).astype(str)
    membership_df["program_id"] = membership_df.get("program_id", pd.Series(dtype=str)).astype(str)
    niche_df["niche_id"] = niche_df.get("niche_id", pd.Series(dtype=str)).astype(str)
    role_axis_order = [axis for axis in NICHE_ROLE_AXIS_ORDER if axis in inputs.role_axes] + [axis for axis in NICHE_ROLE_AXIS_ORDER if axis not in inputs.role_axes]
    program_order = domain_df["source_program_id"].astype(str).drop_duplicates().tolist()
    if not program_order and "program_id" in bundle.program_profile_df.columns:
        program_order = bundle.program_profile_df["program_id"].astype(str).drop_duplicates().tolist()

    domain_role_lookup = domain_df.set_index("domain_key", drop=False) if not domain_df.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, niche_row in niche_df.iterrows():
        niche_id = str(niche_row.get("niche_id", ""))
        members = membership_df.loc[membership_df["niche_id"].astype(str) == niche_id].copy()
        member_count = int(_safe_float(niche_row.get("niche_member_count", niche_row.get("member_count", members["domain_key"].nunique() if not members.empty else 0)), 0.0))
        program_comp_map = _normalize_count_map(members["program_id"]) if not members.empty and "program_id" in members.columns else {}
        role_vectors: list[np.ndarray] = []
        per_program_role_anchor: dict[str, str] = {}
        if not members.empty and not domain_role_lookup.empty:
            members = members.merge(domain_df.loc[:, ["domain_key", "source_program_id", *[axis for axis in role_axis_order if axis in domain_df.columns]]], on="domain_key", how="left")
            for _, member in members.iterrows():
                vec = np.asarray([_safe_float(member.get(axis, 0.0)) for axis in role_axis_order], dtype=float)
                role_vectors.append(vec)
            if role_vectors:
                role_comp_values = np.mean(np.vstack(role_vectors), axis=0)
            else:
                role_comp_values = np.zeros(len(role_axis_order), dtype=float)
            for program_id, sub in members.groupby("program_id", dropna=False):
                vec = np.asarray(
                    [[_safe_float(r.get(axis, 0.0)) for axis in role_axis_order] for _, r in sub.iterrows()],
                    dtype=float,
                )
                vec_mean = vec.mean(axis=0) if vec.size else np.zeros(len(role_axis_order), dtype=float)
                per_program_role_anchor[str(program_id)] = _leading_anchor(vec_mean, role_axis_order)
        else:
            role_comp_values = np.zeros(len(role_axis_order), dtype=float)
        role_comp_map = {axis: float(role_comp_values[idx]) for idx, axis in enumerate(role_axis_order)}

        dominant_pair = ""
        secondary_pair = ""
        if not structures_df.empty and "niche_id" in structures_df.columns:
            struct_row = structures_df.loc[structures_df["niche_id"].astype(str) == niche_id]
            if not struct_row.empty:
                backbone_pairs_raw = str(struct_row.iloc[0].get("backbone_program_pairs", "") or "")
                role_pair_counts: dict[str, int] = {}
                for pair in [x for x in backbone_pairs_raw.split(";") if x]:
                    if "|" not in pair:
                        continue
                    left_program, right_program = pair.split("|", 1)
                    left_role = per_program_role_anchor.get(str(left_program), "")
                    right_role = per_program_role_anchor.get(str(right_program), "")
                    if left_role and right_role:
                        label = _role_pair_label(left_role, right_role)
                        role_pair_counts[label] = role_pair_counts.get(label, 0) + 1
                ranked_pairs = sorted(role_pair_counts.items(), key=lambda item: (-item[1], _pair_sort_key(item[0])))
                if ranked_pairs:
                    dominant_pair = ranked_pairs[0][0]
                if len(ranked_pairs) >= 2:
                    secondary_pair = ranked_pairs[1][0]
                backbone_pair_count = int(_safe_float(struct_row.iloc[0].get("backbone_program_pair_count", 0.0), 0.0))
                cross_program_edge_count = int(_safe_float(struct_row.iloc[0].get("cross_program_edge_count", 0.0), 0.0))
            else:
                backbone_pair_count = 0
                cross_program_edge_count = 0
        else:
            backbone_pair_count = 0
            cross_program_edge_count = 0
        contact_hint = _contact_hint_from_pair(dominant_pair, secondary_pair, len(program_comp_map), backbone_pair_count, cross_program_edge_count)
        if contact_hint not in CONTACT_STRUCTURE_HINT_VOCAB:
            contact_hint = "edge_contact"
        program_comp_strip = _encode_composition_strip(program_comp_map, program_order)
        role_comp_strip = _encode_composition_strip(role_comp_map, role_axis_order)
        rows.append(
            {
                "sample_id": bundle.sample_id,
                "niche_id": niche_id,
                "member_count": member_count,
                "niche_confidence": _safe_float(niche_row.get("niche_confidence", 0.0)),
                "organizational_cohesion": _safe_float(niche_row.get("organizational_cohesion", 0.0)),
                "representative_status": bool(niche_id in representative_niche_ids),
                "member_program_composition": program_comp_strip,
                "member_role_composition": role_comp_strip,
                "dominant_contact_pair": dominant_pair,
                "secondary_contact_pair": secondary_pair,
                "contact_structure_hint": contact_hint,
                "niche_size": member_count,
                "dominant_contact_pair_sort_key": dominant_pair,
                "secondary_contact_pair_sort_key": secondary_pair,
                "member_program_composition_sort_key": program_comp_strip,
                "program_axis_order": ",".join(program_order),
                "role_axis_order": ",".join(role_axis_order),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        [
            "dominant_contact_pair_sort_key",
            "secondary_contact_pair_sort_key",
            "member_program_composition_sort_key",
            "member_count",
            "niche_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    return out


def _program_split_pattern(domain_count: float, total_spot_count: float, largest_domain_share_by_spots: float) -> str:
    if float(largest_domain_share_by_spots) >= SPLIT_THRESHOLD_HIGH_SHARE:
        return "single_block_deployment"
    if (
        float(total_spot_count) >= SPLIT_THRESHOLD_HIGH_TOTAL_SPOTS
        and float(domain_count) >= SPLIT_THRESHOLD_HIGH_DOMAIN_COUNT
        and float(largest_domain_share_by_spots) < SPLIT_THRESHOLD_LOW_SHARE
    ):
        return "large_but_dispersed"
    if float(domain_count) >= SPLIT_THRESHOLD_HIGH_DOMAIN_COUNT and float(largest_domain_share_by_spots) < SPLIT_THRESHOLD_LOW_SHARE:
        return "fragmented_expansion"
    return "mixed_partitioning"


def _transport_pattern(domain_count: float, linked_niche_count: float, top_chain_coverage_share: float, niche_participation_concentration: float) -> str:
    if float(domain_count) >= SPLIT_THRESHOLD_HIGH_DOMAIN_COUNT and float(linked_niche_count) <= 1 and float(top_chain_coverage_share) < TRANSPORT_THRESHOLD_HIGH_CHAIN_COVERAGE:
        return "low_linkout_fragmented"
    if float(linked_niche_count) >= 4 or float(top_chain_coverage_share) >= 0.35:
        return "bridge_prone_broad_linkout"
    if float(linked_niche_count) >= 2 or float(niche_participation_concentration) >= TRANSPORT_THRESHOLD_HIGH_CONCENTRATION:
        return "bridge_capable_selective_linkout"
    return "low_linkout_fragmented"


def _group_metadata(ordered_df: pd.DataFrame) -> list[dict[str, Any]]:
    if ordered_df.empty:
        return []
    groups: list[dict[str, Any]] = []
    current_group = None
    start_idx = 0
    for idx, value in enumerate(ordered_df["leading_anchor_group"].astype(str).tolist()):
        if current_group is None:
            current_group = value
            start_idx = idx
            continue
        if value != current_group:
            groups.append(
                {
                    "group_key": current_group,
                    "start_row": start_idx,
                    "end_row": idx - 1,
                    "row_count": idx - start_idx,
                }
            )
            current_group = value
            start_idx = idx
    if current_group is not None:
        groups.append(
            {
                "group_key": current_group,
                "start_row": start_idx,
                "end_row": int(ordered_df.shape[0]) - 1,
                "row_count": int(ordered_df.shape[0]) - start_idx,
            }
        )
    return groups


def _build_section_summary(
    layer_name: str,
    overview_df: pd.DataFrame,
    confidence_col: str,
) -> tuple[str, str]:
    label = SECTION_LABELS[layer_name]
    if overview_df.empty:
        return (
            f"No {label.lower()} objects are available for this sample.",
            f"No {label.lower()}-level takeaway is available.",
        )
    avg_conf = float(pd.to_numeric(overview_df[confidence_col], errors="coerce").fillna(0.0).mean())
    if layer_name == "program":
        top_anchor_counts = overview_df["leading_component_anchor"].astype(str).replace("", np.nan).dropna().value_counts()
        top_anchor = str(top_anchor_counts.index[0]) if not top_anchor_counts.empty else "unresolved"
        summary = (
            f"This sample contains {int(overview_df.shape[0])} Program objects. "
            f"The Program section is only for reading which mechanism programs are present and what their main component ingredients are."
        )
        takeaway = (
            f"The clearest Program-level takeaway is repeated `{top_anchor}` component support, "
            f"with role kept auxiliary and mean confidence {avg_conf:.2f}."
        )
        return summary, takeaway
    if layer_name == "domain":
        morph_counts = overview_df["morphology_hint"].astype(str).replace("", np.nan).dropna().value_counts()
        top_morph = str(morph_counts.index[0]) if not morph_counts.empty else "compact"
        summary = (
            f"This sample contains {int(overview_df.shape[0])} Domain objects. "
            f"The Domain section is only for reading how Programs land in space and what broad shapes those spatial landings take."
        )
        takeaway = (
            f"The clearest Domain-level takeaway is a `{top_morph}`-leaning landing pattern, "
            f"with mean Domain confidence {avg_conf:.2f}."
        )
        return summary, takeaway
    pair_counts = (
        overview_df["dominant_contact_pair"].astype(str).replace("", np.nan).dropna().value_counts()
        if "dominant_contact_pair" in overview_df.columns
        else pd.Series(dtype=int)
    )
    top_pair = str(pair_counts.index[0]) if not pair_counts.empty else "unresolved_contact"
    summary = (
        f"This sample contains {int(overview_df.shape[0])} Niche objects. "
        f"The Niche section is a local-structure checklist: which Domains co-assemble, how roles divide, and what contact pattern organizes the local structure."
    )
    takeaway = (
        f"The clearest Niche-level takeaway is repeated `{top_pair}` contact organization, "
        f"with mean Niche confidence {avg_conf:.2f}."
    )
    return summary, takeaway


def _build_domain_bridge_takeaway(
    split_df: pd.DataFrame,
    morphology_df: pd.DataFrame,
    transport_df: pd.DataFrame,
) -> tuple[str, str]:
    if split_df.empty:
        return (
            "No domain bridge summary is available for this sample.",
            "No Domain-level bridge takeaway is available.",
        )
    single_block = split_df.loc[split_df["program_split_pattern"].astype(str) == "single_block_deployment", "source_program_id"].astype(str).tolist()
    fragmented = split_df.loc[split_df["program_split_pattern"].astype(str) == "fragmented_expansion", "source_program_id"].astype(str).tolist()
    dispersed = split_df.loc[split_df["program_split_pattern"].astype(str) == "large_but_dispersed", "source_program_id"].astype(str).tolist()
    mixed_neighbor = (
        morphology_df.groupby("source_program_id", dropna=False)["mixed_neighbor_fraction"].mean().sort_values(ascending=False)
        if not morphology_df.empty
        else pd.Series(dtype=float)
    )
    mixed_neighbor_program = str(mixed_neighbor.index[0]) if not mixed_neighbor.empty else ""
    stable_transport = transport_df.loc[
        transport_df["program_transport_pattern"].astype(str) == "bridge_capable_selective_linkout",
        "source_program_id",
    ].astype(str).tolist()
    reconverged = transport_df.loc[
        transport_df["program_transport_pattern"].astype(str) == "bridge_prone_broad_linkout",
        "source_program_id",
    ].astype(str).tolist()
    summary = (
        "Domain objects are treated as a bridge layer: they show how Program states are partitioned into spatial blocks, "
        "how those blocks take on different morphology, and how they continue into Niche assembly."
    )
    takeaway_parts: list[str] = []
    if single_block:
        takeaway_parts.append(f"Single-block deployment is most visible for `{single_block[0]}`")
    if fragmented:
        takeaway_parts.append(f"fragmented expansion is clearest for `{fragmented[0]}`")
    if dispersed:
        takeaway_parts.append(f"large but internally dispersed deployment appears in `{dispersed[0]}`")
    if mixed_neighbor_program:
        takeaway_parts.append(f"`{mixed_neighbor_program}` shows the strongest boundary/mixed-neighbor tendency")
    if stable_transport:
        takeaway_parts.append(f"`{stable_transport[0]}` keeps a selective but still active downstream bridge into Niches")
    elif reconverged:
        takeaway_parts.append(f"`{reconverged[0]}` shows broad downstream link-out after fragmented Domain deployment")
    takeaway = ". ".join(takeaway_parts[:2]) + "." if takeaway_parts else "Domain-level bridge behavior remains mixed across Programs."
    return summary, takeaway


def _build_domain_split_agg(bundle: SampleReportingBundle) -> pd.DataFrame:
    domains = bundle.domain_profile_df.copy()
    if domains.empty or "source_program_id" not in domains.columns:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "source_program_id",
                "domain_count",
                "total_spot_count",
                "total_area_est",
                "largest_domain_share_by_spots",
                "spot_count_distribution",
                "components_count_distribution",
                "program_split_pattern",
            ]
        )
    domains["source_program_id"] = domains["source_program_id"].astype(str)
    domains["spot_count"] = pd.to_numeric(domains.get("spot_count", 0.0), errors="coerce").fillna(0.0)
    domains["geo_area_est"] = pd.to_numeric(domains.get("geo_area_est", 0.0), errors="coerce").fillna(0.0)
    domains["components_count"] = pd.to_numeric(domains.get("components_count", 0.0), errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for program_id, sub in domains.groupby("source_program_id", sort=True):
        total_spots = float(sub["spot_count"].sum())
        total_area_est = float(sub["geo_area_est"].sum()) if "geo_area_est" in sub.columns else 0.0
        largest_share = float(sub["spot_count"].max() / total_spots) if total_spots > 0 else 0.0
        domain_count = int(sub.shape[0])
        rows.append(
            {
                "sample_id": bundle.sample_id,
                "source_program_id": str(program_id),
                "domain_count": domain_count,
                "total_spot_count": total_spots,
                "total_area_est": total_area_est,
                "largest_domain_share_by_spots": largest_share,
                "spot_count_distribution": _encode_quantile_snapshot(sub["spot_count"]),
                "components_count_distribution": _encode_quantile_snapshot(sub["components_count"]),
                "program_split_pattern": _program_split_pattern(domain_count, total_spots, largest_share),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain_count", "total_spot_count", "source_program_id"], ascending=[False, False, True]).reset_index(drop=True)


def _build_domain_morphology_agg(bundle: SampleReportingBundle) -> pd.DataFrame:
    domains = bundle.domain_profile_df.copy()
    columns = [
        "sample_id",
        "source_program_id",
        "domain_id",
        "spot_count",
        "geo_boundary_ratio",
        "geo_elongation",
        "mixed_neighbor_fraction",
        "internal_density",
        "components_count",
        "boundary_contact_score",
    ]
    if domains.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(
        {
            "sample_id": bundle.sample_id,
            "source_program_id": domains.get("source_program_id", pd.Series(dtype=str)).astype(str),
            "domain_id": domains.get("domain_id", pd.Series(dtype=str)).astype(str),
            "spot_count": pd.to_numeric(domains.get("spot_count", 0.0), errors="coerce").fillna(0.0),
            "geo_boundary_ratio": pd.to_numeric(domains.get("geo_boundary_ratio", 0.0), errors="coerce").fillna(0.0),
            "geo_elongation": pd.to_numeric(domains.get("geo_elongation", 0.0), errors="coerce").fillna(0.0),
            "mixed_neighbor_fraction": pd.to_numeric(domains.get("mixed_neighbor_fraction", 0.0), errors="coerce").fillna(0.0),
            "internal_density": pd.to_numeric(domains.get("internal_density", 0.0), errors="coerce").fillna(0.0),
            "components_count": pd.to_numeric(domains.get("components_count", 0.0), errors="coerce").fillna(0.0),
            "boundary_contact_score": pd.to_numeric(domains.get("boundary_contact_score", 0.0), errors="coerce").fillna(0.0),
        }
    )
    return out.sort_values(["source_program_id", "domain_id"]).reset_index(drop=True)


def _build_domain_transport_agg(bundle: SampleReportingBundle, sample_chains_df: pd.DataFrame) -> pd.DataFrame:
    domains = bundle.domain_profile_df.copy()
    memberships = bundle.niche_membership_df.copy()
    columns = [
        "sample_id",
        "source_program_id",
        "domain_count_per_program",
        "linked_domain_count",
        "linked_niche_count",
        "top_niche_participation",
        "top_chain_coverage_share_for_program",
        "niche_participation_concentration",
        "program_transport_pattern",
    ]
    if domains.empty or "source_program_id" not in domains.columns:
        return pd.DataFrame(columns=columns)
    domains["source_program_id"] = domains["source_program_id"].astype(str)
    domains["domain_key"] = domains.get("domain_key", pd.Series(dtype=str)).astype(str)
    memberships["domain_key"] = memberships.get("domain_key", pd.Series(dtype=str)).astype(str)
    memberships["niche_id"] = memberships.get("niche_id", pd.Series(dtype=str)).astype(str)
    domain_program_lookup = domains.loc[:, ["domain_key", "source_program_id"]].drop_duplicates()
    transport = memberships.merge(domain_program_lookup, on="domain_key", how="left")
    rows: list[dict[str, Any]] = []
    for program_id, sub in domains.groupby("source_program_id", sort=True):
        linked = transport.loc[transport["source_program_id"].astype(str) == str(program_id)].copy() if not transport.empty else pd.DataFrame()
        domain_count = int(sub.shape[0])
        linked_domain_count = int(linked["domain_key"].astype(str).nunique()) if not linked.empty else 0
        linked_niche_count = int(linked["niche_id"].astype(str).nunique()) if not linked.empty else 0
        niche_counts = linked["niche_id"].astype(str).value_counts() if not linked.empty else pd.Series(dtype=int)
        top_niche_participation = ",".join([f"{niche_id}:{int(count)}" for niche_id, count in niche_counts.head(3).items()])
        concentration = _herfindahl_concentration(niche_counts)
        program_chains = sample_chains_df.loc[sample_chains_df["program_id"].astype(str) == str(program_id)].copy()
        top_chain_coverage_share = (
            float(np.clip(program_chains["coverage_burden_share"].astype(float).dropna().head(3).sum(), 0.0, 1.0))
            if not program_chains.empty and "coverage_burden_share" in program_chains.columns
            else 0.0
        )
        rows.append(
            {
                "sample_id": bundle.sample_id,
                "source_program_id": str(program_id),
                "domain_count_per_program": domain_count,
                "linked_domain_count": linked_domain_count,
                "linked_niche_count": linked_niche_count,
                "top_niche_participation": top_niche_participation,
                "top_chain_coverage_share_for_program": top_chain_coverage_share,
                "niche_participation_concentration": concentration,
                "niche_participation_concentration_formula": "sum_i(p_i^2) over normalized program-linked niche membership counts",
                "program_transport_pattern": _transport_pattern(
                    domain_count,
                    linked_niche_count,
                    top_chain_coverage_share,
                    concentration,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["domain_count_per_program", "top_chain_coverage_share_for_program", "source_program_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_domain_bridge_matrix_df(
    bundle: SampleReportingBundle,
    inputs: CohortReportingInputs,
    sample_chains_df: pd.DataFrame,
) -> pd.DataFrame:
    split_df = _build_domain_split_agg(bundle)
    morph_df = _build_domain_morphology_agg(bundle)
    transport_df = _build_domain_transport_agg(bundle, sample_chains_df)
    if split_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "source_program_id",
                "leading_component_anchor",
                "leading_role_anchor",
                "domain_count",
                "total_spot_count",
                "largest_domain_share_by_spots",
                "spot_count_mean",
                "geo_boundary_ratio_mean",
                "geo_elongation_mean",
                "mixed_neighbor_fraction_mean",
                "program_split_pattern",
                "morphology_hint",
                "linked_domain_fraction",
                "top_chain_coverage_share_for_program",
                "downstream_linkage_hint",
                "niche_participation_concentration",
            ]
        )
    morph_agg = (
        morph_df.groupby("source_program_id", dropna=False)
        .agg(
            spot_count_mean=("spot_count", "mean"),
            geo_boundary_ratio_mean=("geo_boundary_ratio", "mean"),
            geo_elongation_mean=("geo_elongation", "mean"),
            mixed_neighbor_fraction_mean=("mixed_neighbor_fraction", "mean"),
        )
        .reset_index()
        if not morph_df.empty
        else pd.DataFrame(columns=["source_program_id", "spot_count_mean", "geo_boundary_ratio_mean", "geo_elongation_mean", "mixed_neighbor_fraction_mean"])
    )
    bridge_df = split_df.merge(morph_agg, on="source_program_id", how="left").merge(
        transport_df.loc[
            :,
            [
                "source_program_id",
                "linked_domain_count",
                "linked_niche_count",
                "top_chain_coverage_share_for_program",
                "niche_participation_concentration",
                "program_transport_pattern",
            ],
        ],
        on="source_program_id",
        how="left",
    )
    bridge_df["linked_domain_fraction"] = bridge_df.apply(
        lambda row: float(_safe_float(row.get("linked_domain_count", 0.0)) / max(1.0, _safe_float(row.get("domain_count", 0.0)))),
        axis=1,
    )
    bridge_df["morphology_hint"] = bridge_df.apply(
        lambda row: _domain_morphology_hint(
            _safe_float(row.get("geo_boundary_ratio_mean", 0.0)),
            _safe_float(row.get("mixed_neighbor_fraction_mean", 0.0)),
            _safe_float(row.get("geo_elongation_mean", 0.0)),
        ),
        axis=1,
    )
    bridge_df["downstream_linkage_hint"] = bridge_df["morphology_hint"].astype(str)
    program_anchor_df = pd.DataFrame(columns=["source_program_id", "leading_component_anchor", "leading_role_anchor"])
    if not bundle.program_profile_df.empty and "program_id" in bundle.program_profile_df.columns:
        anchor_rows: list[dict[str, str]] = []
        for _, row in bundle.program_profile_df.iterrows():
            program_id = str(row.get("program_id", ""))
            component_values = np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in inputs.component_axes], dtype=float)
            role_values = np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in inputs.role_axes], dtype=float)
            anchor_rows.append(
                {
                    "source_program_id": program_id,
                    "leading_component_anchor": _leading_anchor(component_values, inputs.component_axes),
                    "leading_role_anchor": _leading_anchor(role_values, inputs.role_axes),
                }
            )
        program_anchor_df = pd.DataFrame(anchor_rows).drop_duplicates("source_program_id")
    bridge_df = bridge_df.merge(program_anchor_df, on="source_program_id", how="left")
    bridge_df["leading_component_anchor"] = bridge_df["leading_component_anchor"].fillna("").astype(str)
    bridge_df["leading_role_anchor"] = bridge_df["leading_role_anchor"].fillna("").astype(str)
    bridge_df["sample_id"] = bundle.sample_id
    bridge_df = bridge_df.drop(columns=["sample_id_x", "sample_id_y"], errors="ignore")
    bridge_df = bridge_df.sort_values(["total_spot_count", "domain_count", "source_program_id"], ascending=[False, False, True]).reset_index(drop=True)
    return bridge_df


def build_domain_spatial_viewer_df(
    bundle: SampleReportingBundle,
    sample_chains_df: pd.DataFrame,
) -> pd.DataFrame:
    domains = bundle.domains_df.copy() if not bundle.domains_df.empty else bundle.domain_profile_df.copy()
    if domains.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "domain_key",
                "domain_id",
                "source_program_id",
                "leading_anchor",
                "spot_count",
                "representative_status",
                "geo_centroid_x",
                "geo_centroid_y",
                "geo_area_est",
                "geo_elongation",
                "geo_boundary_ratio",
                "coverage_burden_share",
                "linkage_support_flag",
                "adjacent_domain_count",
            ]
        )
    if "source_program_id" not in domains.columns:
        if "program_seed_id" in domains.columns:
            domains["source_program_id"] = domains["program_seed_id"].astype(str)
        else:
            domains["source_program_id"] = ""
    representative_ids = _representative_ids(bundle, "domain")
    coverage_map = (
        sample_chains_df.groupby("domain_key", dropna=False)["coverage_burden_share"].max().to_dict()
        if "domain_key" in sample_chains_df.columns and not sample_chains_df.empty
        else {}
    )
    adjacency_counts: dict[str, int] = {}
    if not bundle.domain_graph_df.empty:
        for col_a, col_b in (("domain_key_i", "domain_key_j"), ("domain_key_j", "domain_key_i")):
            if col_a in bundle.domain_graph_df.columns and col_b in bundle.domain_graph_df.columns:
                counts = bundle.domain_graph_df.groupby(col_a, dropna=False)[col_b].nunique().to_dict()
                for key, value in counts.items():
                    adjacency_counts[str(key)] = adjacency_counts.get(str(key), 0) + int(value)
    rows: list[dict[str, Any]] = []
    for _, row in domains.iterrows():
        domain_key = str(row.get("domain_key", ""))
        domain_id = str(row.get("domain_id", domain_key))
        source_program_id = str(row.get("source_program_id", ""))
        leading_anchor = source_program_id
        rows.append(
            {
                "sample_id": bundle.sample_id,
                "domain_key": domain_key,
                "domain_id": domain_id,
                "source_program_id": source_program_id,
                "leading_anchor": leading_anchor,
                "spot_count": int(_safe_float(row.get("spot_count", 0.0), 0.0)),
                "representative_status": bool(domain_id in representative_ids or domain_key in representative_ids),
                "geo_centroid_x": _safe_float(row.get("geo_centroid_x", 0.0)),
                "geo_centroid_y": _safe_float(row.get("geo_centroid_y", 0.0)),
                "geo_area_est": _safe_float(row.get("geo_area_est", row.get("spot_count", 0.0))),
                "geo_elongation": _safe_float(row.get("geo_elongation", 1.0), 1.0),
                "geo_boundary_ratio": _safe_float(row.get("geo_boundary_ratio", 0.0)),
                "coverage_burden_share": _safe_float(coverage_map.get(domain_key, 0.0)),
                "linkage_support_flag": bool(domain_key in set(bundle.niche_membership_df.get("domain_key", pd.Series(dtype=str)).astype(str).tolist())),
                "adjacent_domain_count": int(adjacency_counts.get(domain_key, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_program_id", "domain_key"]).reset_index(drop=True)


def build_domain_spatial_viewer_payload(
    bundle: SampleReportingBundle,
    bridge_df: pd.DataFrame,
    viewer_data_ref: str,
    cfg: CohortReportingConfig,
) -> dict[str, Any]:
    available_program_ids = [str(x) for x in bridge_df.get("source_program_id", pd.Series(dtype=str)).astype(str).tolist() if str(x)]
    available_program_ids = list(dict.fromkeys(available_program_ids))
    default_target = max(1, int(cfg.display.domain_spatial_viewer_default_program_count))
    default_count = min(default_target, len(available_program_ids))
    if 0 < len(available_program_ids) < default_target:
        default_count = len(available_program_ids)
    default_selected_program_ids = available_program_ids[:default_count]
    if len(default_selected_program_ids) < 2 and len(available_program_ids) >= 2:
        default_selected_program_ids = available_program_ids[:2]
    return {
        "available_program_ids": available_program_ids,
        "default_selected_program_ids": default_selected_program_ids,
        "min_program_count": int(cfg.display.domain_spatial_viewer_min_programs),
        "max_program_count": int(cfg.display.domain_spatial_viewer_max_programs),
        "default_selection": cfg.display.domain_spatial_viewer_default_selection,
        "default_program_count": int(cfg.display.domain_spatial_viewer_default_program_count),
        "default_count_fallback_rule": "fallback_to_2_then_available_count_when_recommended_set_is_smaller_than_3",
        "default_view_mode": cfg.display.domain_spatial_viewer_default_view_mode,
        "view_modes": list(cfg.display.domain_spatial_viewer_view_modes),
        "dense_selection_threshold": int(cfg.display.domain_spatial_viewer_dense_selection_threshold),
        "dense_selection_note": cfg.display.domain_spatial_viewer_dense_selection_note,
        "representative_only_toggle_enabled": True,
        "representative_only_uses_precomputed_flags": True,
        "data_mode": DOMAIN_SPATIAL_VIEWER_DATA_MODE,
        "geometry_source": DOMAIN_SPATIAL_GEOMETRY_SOURCE,
        "data_ref": viewer_data_ref,
        "role": DOMAIN_SPATIAL_VIEWER_ROLE,
        "mode": DOMAIN_SPATIAL_VIEWER_MODE,
        "color_mode": DOMAIN_SPATIAL_VIEWER_COLOR_MODE,
        "boundary_highlight": DOMAIN_SPATIAL_VIEWER_BOUNDARY_HIGHLIGHT,
        "label_mode": DOMAIN_SPATIAL_VIEWER_LABEL_MODE,
        "footprint_note": DOMAIN_FOOTPRINT_NOTE,
    }


def build_niche_spatial_viewer_df(
    bundle: SampleReportingBundle,
    inputs: CohortReportingInputs,
) -> pd.DataFrame:
    memberships = bundle.niche_membership_df.copy()
    if memberships.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "niche_id",
                "domain_key",
                "domain_id",
                "source_program_id",
                "spot_id",
                "x",
                "y",
                "member_count",
                "niche_confidence",
                "is_backbone_member",
                "is_structure_member",
                "representative_status",
                "leading_component_anchor",
                "leading_role_anchor",
                "component_strip",
                "role_strip",
                "domain_level_confidence",
                "spot_count",
                "geo_boundary_ratio",
                "mixed_neighbor_fraction",
                "dominant_contact_pair",
                "secondary_contact_pair",
                "contact_structure_hint",
                "member_program_composition",
                "member_role_composition",
                "is_sample_background_spot",
            ]
        )
    memberships["niche_id"] = memberships.get("niche_id", pd.Series(dtype=str)).astype(str)
    memberships["domain_key"] = memberships.get("domain_key", pd.Series(dtype=str)).astype(str)
    memberships["program_id"] = memberships.get("program_id", pd.Series(dtype=str)).astype(str)
    niche_lookup = build_niche_assembly_matrix_df(bundle, inputs)
    representative_ids = _representative_ids(bundle, "domain")
    domain_meta_df = bundle.domains_df.copy() if not bundle.domains_df.empty else bundle.domain_profile_df.copy()
    domain_profile_df = bundle.domain_profile_df.copy()
    domain_meta_df["domain_key"] = domain_meta_df.get("domain_key", pd.Series(dtype=str)).astype(str)
    domain_meta_df["domain_id"] = domain_meta_df.get("domain_id", pd.Series(dtype=str)).astype(str)
    if "source_program_id" not in domain_meta_df.columns:
        domain_meta_df["source_program_id"] = domain_meta_df.get("program_seed_id", pd.Series(dtype=str)).astype(str)
    else:
        domain_meta_df["source_program_id"] = domain_meta_df["source_program_id"].astype(str)
    domain_profile_df["domain_key"] = domain_profile_df.get("domain_key", pd.Series(dtype=str)).astype(str)
    domain_profile_df["domain_id"] = domain_profile_df.get("domain_id", pd.Series(dtype=str)).astype(str)
    domain_profile_df["source_program_id"] = domain_profile_df.get("source_program_id", pd.Series(dtype=str)).astype(str)
    domain_rows: list[dict[str, Any]] = []
    merged_domain = domain_meta_df.merge(
        domain_profile_df,
        on=["domain_key", "domain_id", "source_program_id"],
        how="left",
        suffixes=("", "_profile"),
    ) if not domain_meta_df.empty else domain_profile_df.copy()
    for _, row in merged_domain.iterrows():
        component_values = np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in inputs.component_axes], dtype=float)
        role_values = np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in inputs.role_axes], dtype=float)
        domain_rows.append(
            {
                "domain_key": str(row.get("domain_key", "")),
                "domain_id": str(row.get("domain_id", "")),
                "source_program_id": str(row.get("source_program_id", "")),
                "leading_component_anchor": _leading_anchor(component_values, inputs.component_axes),
                "leading_role_anchor": _leading_anchor(role_values, inputs.role_axes),
                "component_strip": encode_full_axis_strip(component_values, inputs.component_axes),
                "role_strip": encode_full_axis_strip(role_values, inputs.role_axes),
                "domain_level_confidence": _safe_float(row.get("domain_level_confidence", 0.0)),
                "spot_count": int(_safe_float(row.get("spot_count", 0.0), 0.0)),
                "geo_boundary_ratio": _safe_float(row.get("geo_boundary_ratio", 0.0)),
                "mixed_neighbor_fraction": _safe_float(row.get("mixed_neighbor_fraction", 0.0)),
                "geo_centroid_x": _safe_float(row.get("geo_centroid_x", 0.0)),
                "geo_centroid_y": _safe_float(row.get("geo_centroid_y", 0.0)),
                "representative_status": bool(
                    str(row.get("domain_id", "")) in representative_ids or str(row.get("domain_key", "")) in representative_ids
                ),
            }
        )
    domain_lookup = pd.DataFrame(domain_rows).drop_duplicates("domain_key") if domain_rows else pd.DataFrame()
    has_spot_geometry = not bundle.domain_spot_membership_df.empty and not bundle.spot_coords_df.empty
    if has_spot_geometry:
        domain_spots = bundle.domain_spot_membership_df.copy()
        domain_spots["domain_key"] = domain_spots.get("domain_key", pd.Series(dtype=str)).astype(str)
        domain_spots["spot_id"] = domain_spots.get("spot_id", pd.Series(dtype=str)).astype(str)
        coords = bundle.spot_coords_df.copy()
        coords["spot_id"] = coords.get("spot_id", pd.Series(dtype=str)).astype(str)
        merged = memberships.merge(domain_spots, on="domain_key", how="left").merge(coords, on="spot_id", how="left")
    else:
        merged = memberships.copy()
        merged["spot_id"] = merged["domain_key"].astype(str) + "::centroid"
    if not domain_lookup.empty:
        merged = merged.merge(domain_lookup, on="domain_key", how="left")
    if not has_spot_geometry:
        merged["x"] = pd.to_numeric(merged.get("geo_centroid_x", 0.0), errors="coerce").fillna(0.0)
        merged["y"] = pd.to_numeric(merged.get("geo_centroid_y", 0.0), errors="coerce").fillna(0.0)
    if not niche_lookup.empty:
        merged = merged.merge(
            niche_lookup.loc[
                :,
                [
                    "niche_id",
                    "member_count",
                    "niche_confidence",
                    "dominant_contact_pair",
                    "secondary_contact_pair",
                    "contact_structure_hint",
                    "member_program_composition",
                    "member_role_composition",
                ],
            ],
            on="niche_id",
            how="left",
        )
    merged["sample_id"] = bundle.sample_id
    for col in ("is_backbone_member", "is_structure_member", "representative_status"):
        if col not in merged.columns:
            merged[col] = False
        merged[col] = merged[col].fillna(False).astype(bool)
    for col in ("dominant_contact_pair", "secondary_contact_pair", "contact_structure_hint"):
        merged[col] = merged.get(col, pd.Series(dtype=str)).fillna("").astype(str)
    merged["is_sample_background_spot"] = False
    if has_spot_geometry:
        background = coords.loc[:, ["spot_id", "x", "y"]].dropna().drop_duplicates("spot_id").copy()
        if not background.empty:
            background["sample_id"] = bundle.sample_id
            background["niche_id"] = ""
            background["domain_key"] = ""
            background["domain_id"] = ""
            background["source_program_id"] = ""
            background["member_count"] = 0
            background["niche_confidence"] = 0.0
            background["is_backbone_member"] = False
            background["is_structure_member"] = False
            background["representative_status"] = False
            background["leading_component_anchor"] = ""
            background["leading_role_anchor"] = ""
            background["component_strip"] = ""
            background["role_strip"] = ""
            background["domain_level_confidence"] = 0.0
            background["spot_count"] = 0
            background["geo_boundary_ratio"] = 0.0
            background["mixed_neighbor_fraction"] = 0.0
            background["dominant_contact_pair"] = ""
            background["secondary_contact_pair"] = ""
            background["contact_structure_hint"] = ""
            background["member_program_composition"] = ""
            background["member_role_composition"] = ""
            background["is_sample_background_spot"] = True
            merged = pd.concat([merged, background], ignore_index=True, sort=False)
    return merged.sort_values(["niche_id", "source_program_id", "domain_key", "spot_id"]).reset_index(drop=True)


def build_niche_spatial_viewer_payload(
    niche_matrix_df: pd.DataFrame,
    viewer_data_ref: str,
    cfg: CohortReportingConfig,
) -> dict[str, Any]:
    if niche_matrix_df.empty:
        available_niche_ids: list[str] = []
    else:
        order_df = niche_matrix_df.copy()
        order_df["representative_status"] = order_df.get("representative_status", False).fillna(False).astype(bool)
        order_df["niche_confidence"] = pd.to_numeric(order_df.get("niche_confidence", 0.0), errors="coerce").fillna(0.0)
        order_df["member_count"] = pd.to_numeric(order_df.get("member_count", 0.0), errors="coerce").fillna(0.0)
        order_df = order_df.sort_values(
            ["representative_status", "niche_confidence", "member_count", "niche_id"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        available_niche_ids = order_df["niche_id"].astype(str).tolist()
    default_selected_niche_id = available_niche_ids[0] if available_niche_ids else ""
    return {
        "available_niche_ids": available_niche_ids,
        "default_selected_niche_id": default_selected_niche_id,
        "default_view_mode": cfg.display.niche_spatial_viewer_default_view_mode,
        "view_modes": list(cfg.display.niche_spatial_viewer_view_modes),
        "data_ref": viewer_data_ref,
        "role": NICHE_SPATIAL_VIEWER_ROLE,
        "mode": NICHE_SPATIAL_VIEWER_MODE,
        "selection_mode": NICHE_SPATIAL_VIEWER_SELECTION_MODE,
        "geometry_mode": NICHE_SPATIAL_VIEWER_GEOMETRY_MODE,
        "niche_order": NICHE_SPATIAL_VIEWER_NICHE_ORDER,
        "program_color_mode": NICHE_SPATIAL_VIEWER_PROGRAM_COLOR_MODE,
        "footprint_note": NICHE_FOOTPRINT_NOTE,
    }


def build_layer_atlas_artifacts(
    bundle: SampleReportingBundle,
    layer_name: str,
    inputs: CohortReportingInputs,
    cfg: CohortReportingConfig,
) -> LayerAtlasArtifacts:
    profile_df = _layer_profile_df(bundle, layer_name)
    object_id_col = _object_id_column(layer_name)
    confidence_col = _confidence_column(layer_name)
    representative_ids = _representative_ids(bundle, layer_name)
    component_threshold = float(cfg.display.supported_axis_threshold_component)
    role_threshold = float(cfg.display.supported_axis_threshold_role)
    rows: list[dict[str, Any]] = []
    for _, row in profile_df.iterrows():
        object_id = str(row.get(object_id_col, ""))
        component_values = np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in inputs.component_axes], dtype=float)
        role_values = np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in inputs.role_axes], dtype=float)
        leading_component_anchor = _leading_anchor(component_values, inputs.component_axes)
        leading_role_anchor = _leading_anchor(role_values, inputs.role_axes)
        component_spread, role_spread = _spread_values(layer_name, row, component_values, role_values)
        leading_anchor_group = leading_component_anchor or leading_role_anchor or "unanchored"
        record = {
            "sample_id": bundle.sample_id,
            "cancer_type": bundle.cancer_type,
            "layer_name": layer_name,
            "object_id": object_id,
            object_id_col: object_id,
            "leading_component_anchor": leading_component_anchor,
            "leading_role_anchor": leading_role_anchor,
            "leading_anchor_group": leading_anchor_group,
            "component_strip": encode_full_axis_strip(component_values, inputs.component_axes),
            "role_strip": encode_full_axis_strip(role_values, inputs.role_axes),
            "supported_component_axes_above_threshold": encode_supported_axes(component_values, inputs.component_axes, component_threshold),
            "supported_role_axes_above_threshold": encode_supported_axes(role_values, inputs.role_axes, role_threshold),
            "component_support_spread": float(component_spread),
            "role_support_spread": float(role_spread),
            "confidence": _safe_float(row.get(confidence_col, 0.0)),
            confidence_col: _safe_float(row.get(confidence_col, 0.0)),
            "representative_status": bool(object_id in representative_ids),
        }
        if layer_name == "program":
            record["eligible_for_burden"] = bool(row.get("eligible_for_burden", False))
        elif layer_name == "domain":
            record["source_program_id"] = str(row.get("source_program_id", ""))
            record["domain_key"] = str(row.get("domain_key", ""))
            record["spot_count"] = int(_safe_float(row.get("spot_count", 0.0), 0.0))
            record["geo_boundary_ratio"] = _safe_float(row.get("geo_boundary_ratio", 0.0))
            record["mixed_neighbor_fraction"] = _safe_float(row.get("mixed_neighbor_fraction", 0.0))
            record["geo_elongation"] = _safe_float(row.get("geo_elongation", 0.0))
            record["morphology_hint"] = _domain_morphology_hint(
                record["geo_boundary_ratio"],
                record["mixed_neighbor_fraction"],
                record["geo_elongation"],
            )
        else:
            record["niche_member_count"] = int(_safe_float(row.get("niche_member_count", row.get("member_count", 0.0)), 0.0))
            record["organizational_cohesion"] = _safe_float(row.get("organizational_cohesion", 0.0))
        rows.append(record)
    overview_df = pd.DataFrame(rows)
    if overview_df.empty:
        payload = {
            "layer_name": layer_name,
            "section_question": SECTION_QUESTIONS[layer_name],
            "section_summary_text": f"No {SECTION_LABELS[layer_name].lower()} objects are available for this sample.",
            "section_takeaway": f"No {SECTION_LABELS[layer_name].lower()}-level takeaway is available.",
        }
        return LayerAtlasArtifacts(overview_df=overview_df, payload=payload)

    overview_df = overview_df.sort_values(
        ["leading_component_anchor", "confidence", "object_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    overview_df["object_sort_index"] = np.arange(int(overview_df.shape[0]), dtype=int)

    section_summary_text, section_takeaway = _build_section_summary(layer_name, overview_df, confidence_col)
    payload = {
        "layer_name": layer_name,
        "section_question": SECTION_QUESTIONS[layer_name],
        "section_summary_text": section_summary_text,
        "section_takeaway": section_takeaway,
    }
    return LayerAtlasArtifacts(overview_df=overview_df, payload=payload)


def build_sample_atlas_payload(
    bundle: SampleReportingBundle,
    inputs: CohortReportingInputs,
    cfg: CohortReportingConfig,
    sample_chains_df: pd.DataFrame,
    *,
    figures_root: str,
    tables_root: str,
) -> tuple[dict[str, LayerAtlasArtifacts], dict[str, Any]]:
    sections: dict[str, LayerAtlasArtifacts] = {
        layer_name: build_layer_atlas_artifacts(bundle, layer_name, inputs, cfg)
        for layer_name in ("program", "domain", "niche")
    }
    domain_bridge_matrix_df = build_domain_bridge_matrix_df(bundle, inputs, sample_chains_df)
    domain_spatial_viewer_df = build_domain_spatial_viewer_df(bundle, sample_chains_df)
    niche_matrix_df = build_niche_assembly_matrix_df(bundle, inputs)
    niche_spatial_viewer_df = build_niche_spatial_viewer_df(bundle, inputs)
    payload = {
        "sample_id": bundle.sample_id,
        "cancer_type": bundle.cancer_type,
        "sections": {},
    }
    for layer_name, artifacts in sections.items():
        if layer_name == "program":
            object_overview_records = _program_table_records(artifacts.overview_df)
        elif layer_name == "domain":
            object_overview_records = _domain_table_records(artifacts.overview_df)
        else:
            object_overview_records = _niche_table_records(niche_matrix_df)
        payload["sections"][layer_name] = {
            **artifacts.payload,
            "object_overview_records": object_overview_records,
        }
    program_section = payload["sections"]["program"]
    program_section["program_composition_overview"] = {
        "figure_png": f"{figures_root}/{bundle.sample_id}/{PROGRAM_STATIC_FIGURE_NAME}.png",
        "figure_pdf": f"{figures_root}/{bundle.sample_id}/{PROGRAM_STATIC_FIGURE_NAME}.pdf",
        "observation_unit": PROGRAM_STATIC_OBSERVATION_UNIT,
        "records": sections["program"].overview_df.loc[
            :,
            [
                "program_id",
                "leading_component_anchor",
                "leading_role_anchor",
                "component_strip",
                "role_strip",
                "confidence",
            ],
        ].to_dict(orient="records"),
    }
    domain_section = payload["sections"]["domain"]
    domain_section["program_domain_deployment_matrix"] = {
        "figure_png": f"{figures_root}/{bundle.sample_id}/{DOMAIN_STATIC_FIGURE_NAME}.png",
        "figure_pdf": f"{figures_root}/{bundle.sample_id}/{DOMAIN_STATIC_FIGURE_NAME}.pdf",
        "observation_unit": DOMAIN_STATIC_OBSERVATION_UNIT,
        "program_scope": DOMAIN_BRIDGE_MATRIX_PROGRAM_SCOPE,
        "records": domain_bridge_matrix_df.to_dict(orient="records"),
    }
    domain_section["domain_spatial_viewer"] = build_domain_spatial_viewer_payload(
        bundle,
        domain_bridge_matrix_df,
        f"sample_atlas/{bundle.sample_id}/domain_spatial_viewer.parquet",
        cfg,
    )
    domain_section["domain_spatial_viewer"]["footprint_note"] = DOMAIN_FOOTPRINT_NOTE

    niche_section = payload["sections"]["niche"]
    niche_section["sample_level_niche_assembly_matrix"] = {
        "figure_png": f"{figures_root}/{bundle.sample_id}/{NICHE_STATIC_FIGURE_NAME}.png",
        "figure_pdf": f"{figures_root}/{bundle.sample_id}/{NICHE_STATIC_FIGURE_NAME}.pdf",
        "data_ref": f"sample_atlas/{bundle.sample_id}/niche_assembly_matrix.parquet",
        "observation_unit": NICHE_STATIC_OBSERVATION_UNIT,
        "records_preview": niche_matrix_df.head(20).to_dict(orient="records"),
    }
    niche_section["niche_spatial_viewer"] = build_niche_spatial_viewer_payload(
        niche_matrix_df,
        f"sample_atlas/{bundle.sample_id}/niche_spatial_viewer.parquet",
        cfg,
    )

    payload["niche_assembly_matrix_records"] = niche_matrix_df.to_dict(orient="records")
    payload["niche_spatial_viewer_records"] = niche_spatial_viewer_df.to_dict(orient="records")
    return sections, payload
