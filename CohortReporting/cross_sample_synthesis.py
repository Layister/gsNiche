from __future__ import annotations

import math
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .loaders import CohortReportingInputs, SampleReportingBundle
from .tables import derive_support_spread


TRIAGE_CLASSES = ["stable_recurrent", "conditional_variant", "sample_specific"]
TRIAGE_LEVELS = ["program", "domain", "niche", "chain"]
PROGRAM_PATTERN_CONTRACT = "leading_component_anchor_plus_leading_role_anchor_only"
DOMAIN_PATTERN_CONTRACT = "three_core_deployment_axes_only"
NICHE_PATTERN_CONTRACT = "member_program_recipe_signature_plus_dominant_contact_pair_plus_cross_sample_contact_class"
NICHE_RECIPE_SIGNATURE_CONTRACT = "reduced_fixed_order_major_component_signature_not_full_composition_string"
CHAIN_OBJECT_COUNT_MIN = 5
CHAIN_SAMPLE_OCCURRENCE_MIN = 2
CHAIN_OBJECT_COUNT_NOTE = "v1 default gate for current cohort scale"
MAX_DISPLAYED_CROSS_SAMPLE_CHAINS = 12
MAX_OPTIONAL_CONDITIONAL_CHAINS = 4
ROLE_ORDER = ["scaffold_like", "interface_like", "node_like", "companion_like"]
CROSS_SAMPLE_CONTACT_CLASS_VOCAB = [
    "interface_mesh",
    "attached_interface",
    "node_focal_contact",
    "diffuse_edge_contact",
]
PROGRAM_UMAP_MIN_OBJECTS = 15
PROGRAM_UMAP_MIN_PAIR_COVERAGE = 0.15
PROGRAM_UMAP_MAX_IMPUTED_SHARE = 0.40
PROGRAM_UMAP_RANDOM_STATE = 42
PROGRAM_UMAP_MIN_DIST = 0.2
PROGRAM_UMAP_LAYOUT_METHOD = "umap_precomputed_distance"
PROGRAM_UMAP_DISTANCE_SOURCE = "program_cross_sample_comparability"
PROGRAM_UMAP_DISTANCE_IMPUTATION = "missing_pairs_set_to_1.0"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _leading_anchor(row: pd.Series | dict[str, Any], axis_ids: list[str]) -> str:
    if not axis_ids:
        return ""
    values = np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in axis_ids], dtype=float)
    if values.size == 0:
        return ""
    return str(axis_ids[int(np.argmax(values))])


def _parse_strip(encoded: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in str(encoded or "").split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[str(key)] = _safe_float(value, 0.0)
    return values


def _reduced_signature(value_map: dict[str, float], fixed_order: list[str], keep_threshold: float, high_threshold: float) -> str:
    if not fixed_order:
        return ""
    ordered = [(key, max(0.0, _safe_float(value_map.get(key, 0.0), 0.0))) for key in fixed_order]
    total = float(sum(value for _, value in ordered))
    if total <= 0.0:
        return "empty"
    normalized = [(key, value / total) for key, value in ordered]
    kept = [(key, share) for key, share in normalized if share >= keep_threshold]
    if not kept:
        top_key, top_share = max(normalized, key=lambda item: (item[1], item[0]))
        kept = [(top_key, top_share)]
    parts: list[str] = []
    for key, share in kept:
        level = "major" if share >= high_threshold else "support"
        parts.append(f"{key}:{level}")
    return "+".join(parts)


def _reduced_niche_program_recipe_signature(
    value_map: dict[str, float],
    fixed_order: list[str],
    *,
    primary_threshold: float = 0.55,
    secondary_threshold: float = 0.20,
    max_terms: int = 2,
) -> str:
    if not fixed_order:
        return ""
    ordered = [(key, max(0.0, _safe_float(value_map.get(key, 0.0), 0.0))) for key in fixed_order]
    total = float(sum(value for _, value in ordered))
    if total <= 0.0:
        return "empty"
    normalized = [(key, value / total) for key, value in ordered]
    ranked = sorted(normalized, key=lambda item: (-item[1], fixed_order.index(item[0]) if item[0] in fixed_order else len(fixed_order)))
    top_key, top_share = ranked[0]
    kept: set[str] = {str(top_key)}
    if top_share < primary_threshold:
        for key, share in ranked[1:]:
            if share >= secondary_threshold:
                kept.add(str(key))
            if len(kept) >= max_terms:
                break
    else:
        for key, share in ranked[1:]:
            if share >= max(secondary_threshold + 0.05, 0.25):
                kept.add(str(key))
                break
    ordered_kept = [key for key in fixed_order if key in kept]
    return "+".join(ordered_kept[:max_terms]) if ordered_kept else str(top_key)


def _cross_sample_contact_class(contact_structure_hint: str, dominant_contact_pair: str) -> str:
    hint = str(contact_structure_hint or "")
    pair = str(dominant_contact_pair or "")
    if hint in {"mixed_mesh", "parallel_contact"} or pair == "interface_like↔interface_like":
        return "interface_mesh"
    if hint == "scaffold_attached_interface" or "scaffold_like" in pair:
        return "attached_interface"
    if hint == "node_focal" or "node_like" in pair:
        return "node_focal_contact"
    return "diffuse_edge_contact"


def _representative_ids(values: list[str], max_items: int = 5) -> str:
    cleaned = [str(x) for x in values if str(x)]
    if not cleaned:
        return ""
    unique = list(dict.fromkeys(cleaned))
    return ",".join(unique[:max_items])


def _stability_class(sample_occurrence: int, total_samples: int) -> str:
    if total_samples <= 0:
        return "sample_specific"
    stable_threshold = max(2, int(math.ceil(total_samples * 0.6)))
    if sample_occurrence >= stable_threshold:
        return "stable_recurrent"
    if sample_occurrence >= 2:
        return "conditional_variant"
    return "sample_specific"


def _program_pattern_id(component_anchor: str, role_anchor: str) -> str:
    return f"{component_anchor}__{role_anchor}"


def _block_vs_fragmented(domain_count: float, largest_share: float) -> str:
    if largest_share >= 0.65:
        return "block"
    if domain_count >= 5 and largest_share < 0.35:
        return "fragmented"
    return "mixed"


def _boundary_vs_mixed_neighbor(boundary_ratio: float, mixed_neighbor_fraction: float) -> str:
    if boundary_ratio >= 0.50 and mixed_neighbor_fraction < 0.35:
        return "boundary"
    if mixed_neighbor_fraction >= 0.35:
        return "mixed_neighbor"
    return "neutral"


def _bridge_tendency_state(
    linked_domain_fraction: float,
    linked_niche_count: float,
    niche_participation_concentration: float,
) -> str:
    if linked_domain_fraction >= 0.65 or linked_niche_count >= 4:
        return "bridge_prone"
    if linked_domain_fraction >= 0.30 or (linked_niche_count >= 2 and niche_participation_concentration >= 0.35):
        return "bridge_capable"
    return "non_bridge"


def _domain_deployment_pattern_id(block_state: str, boundary_state: str, bridge_state: str) -> str:
    return f"{block_state}__{boundary_state}__{bridge_state}"


def _program_pattern_map(inputs: CohortReportingInputs) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    mapping: dict[tuple[str, str], str] = {}
    for bundle in inputs.sample_bundles:
        if bundle.program_profile_df.empty:
            continue
        for _, row in bundle.program_profile_df.iterrows():
            program_id = str(row.get("program_id", ""))
            component_anchor = _leading_anchor(row, inputs.component_axes)
            role_anchor = _leading_anchor(row, inputs.role_axes)
            pattern_id = _program_pattern_id(component_anchor, role_anchor)
            mapping[(bundle.sample_id, program_id)] = pattern_id
            rows.append(
                {
                    "sample_id": bundle.sample_id,
                    "program_id": program_id,
                    "program_pattern_id": pattern_id,
                    "leading_component_anchor": component_anchor,
                    "leading_role_anchor": role_anchor,
                    "component_state_type": str((bundle.program_summary or {}).get("component_state_type", "")),
                    "role_state_type": str((bundle.program_summary or {}).get("role_state_type", "")),
                    "component_support_spread": derive_support_spread(
                        np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in inputs.component_axes], dtype=float)
                    ),
                    "role_support_spread": derive_support_spread(
                        np.asarray([_safe_float(row.get(axis_id, 0.0)) for axis_id in inputs.role_axes], dtype=float)
                    ),
                    "overall_profile_confidence": _safe_float(row.get("overall_profile_confidence", 0.0)),
                }
            )
    return mapping, pd.DataFrame(rows)


def build_program_cross_sample_pattern_catalog(inputs: CohortReportingInputs) -> pd.DataFrame:
    _, program_df = _program_pattern_map(inputs)
    columns = [
        "program_pattern_id",
        "leading_component_anchor",
        "leading_role_anchor",
        "sample_occurrence",
        "object_count",
        "representative_program_ids",
        "support_profile_notes",
        "stability_class",
    ]
    if program_df.empty:
        return pd.DataFrame(columns=columns)
    total_samples = len(inputs.sample_bundles)
    rows: list[dict[str, Any]] = []
    for pattern_id, sub in program_df.groupby("program_pattern_id", sort=True):
        component_anchor = str(sub["leading_component_anchor"].iloc[0])
        role_anchor = str(sub["leading_role_anchor"].iloc[0])
        sample_occurrence = int(sub["sample_id"].astype(str).nunique())
        object_count = int(sub.shape[0])
        representative_program_ids = _representative_ids([f"{sid}:{pid}" for sid, pid in zip(sub["sample_id"], sub["program_id"])])
        component_spread_mean = float(pd.to_numeric(sub["component_support_spread"], errors="coerce").fillna(0.0).mean())
        role_spread_mean = float(pd.to_numeric(sub["role_support_spread"], errors="coerce").fillna(0.0).mean())
        support_profile_notes = (
            f"Anchored by component `{component_anchor}` and role `{role_anchor}`. "
            f"Mean component spread={component_spread_mean:.2f}, mean role spread={role_spread_mean:.2f}."
        )
        rows.append(
            {
                "program_pattern_id": pattern_id,
                "leading_component_anchor": component_anchor,
                "leading_role_anchor": role_anchor,
                "sample_occurrence": sample_occurrence,
                "object_count": object_count,
                "representative_program_ids": representative_program_ids,
                "support_profile_notes": support_profile_notes,
                "stability_class": _stability_class(sample_occurrence, total_samples),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["sample_occurrence", "object_count", "program_pattern_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_domain_cross_sample_deployment_catalog(
    inputs: CohortReportingInputs,
    sample_atlas_payloads: dict[str, dict],
) -> pd.DataFrame:
    columns = [
        "program_pattern_id",
        "domain_deployment_pattern_id",
        "block_vs_fragmented",
        "boundary_vs_mixed_neighbor",
        "bridge_tendency",
        "sample_occurrence",
        "object_count",
        "representative_program_ids",
        "representative_domain_keys",
        "stability_class",
    ]
    program_pattern_map, _ = _program_pattern_map(inputs)
    domain_key_lookup: dict[tuple[str, str], list[str]] = {}
    for bundle in inputs.sample_bundles:
        if bundle.domain_profile_df.empty:
            continue
        for program_id, sub in bundle.domain_profile_df.groupby("source_program_id", sort=False):
            domain_key_lookup[(bundle.sample_id, str(program_id))] = sub.get("domain_key", pd.Series(dtype=str)).astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for sample_id, atlas_payload in sample_atlas_payloads.items():
        domain_records = pd.DataFrame((((atlas_payload.get("sections", {}) or {}).get("domain", {}) or {}).get("program_domain_deployment_matrix", {}) or {}).get("records", []) or [])
        if domain_records.empty:
            continue
        for _, row in domain_records.iterrows():
            program_id = str(row.get("source_program_id", ""))
            program_pattern_id = program_pattern_map.get((sample_id, program_id), "")
            block_state = _block_vs_fragmented(_safe_float(row.get("domain_count", 0.0)), _safe_float(row.get("largest_domain_share_by_spots", 0.0)))
            boundary_state = _boundary_vs_mixed_neighbor(_safe_float(row.get("geo_boundary_ratio_mean", 0.0)), _safe_float(row.get("mixed_neighbor_fraction_mean", 0.0)))
            bridge_state = _bridge_tendency_state(
                _safe_float(row.get("linked_domain_fraction", 0.0)),
                _safe_float(row.get("linked_niche_count", 0.0)),
                _safe_float(row.get("niche_participation_concentration", 0.0)),
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_program_id": program_id,
                    "program_pattern_id": program_pattern_id,
                    "domain_deployment_pattern_id": _domain_deployment_pattern_id(block_state, boundary_state, bridge_state),
                    "block_vs_fragmented": block_state,
                    "boundary_vs_mixed_neighbor": boundary_state,
                    "bridge_tendency": bridge_state,
                    "representative_program_object": f"{sample_id}:{program_id}",
                    "representative_domain_keys_raw": _representative_ids(domain_key_lookup.get((sample_id, program_id), []), max_items=8),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    total_samples = len(inputs.sample_bundles)
    out_rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(["program_pattern_id", "domain_deployment_pattern_id", "block_vs_fragmented", "boundary_vs_mixed_neighbor", "bridge_tendency"], sort=True):
        program_pattern_id, deployment_pattern_id, block_state, boundary_state, bridge_state = keys
        sample_occurrence = int(sub["sample_id"].astype(str).nunique())
        object_count = int(sub.shape[0])
        out_rows.append(
            {
                "program_pattern_id": str(program_pattern_id),
                "domain_deployment_pattern_id": str(deployment_pattern_id),
                "block_vs_fragmented": str(block_state),
                "boundary_vs_mixed_neighbor": str(boundary_state),
                "bridge_tendency": str(bridge_state),
                "sample_occurrence": sample_occurrence,
                "object_count": object_count,
                "representative_program_ids": _representative_ids(sub["representative_program_object"].astype(str).tolist()),
                "representative_domain_keys": _representative_ids(
                    [item for raw in sub["representative_domain_keys_raw"].astype(str).tolist() for item in raw.split(",") if item],
                    max_items=8,
                ),
                "stability_class": _stability_class(sample_occurrence, total_samples),
            }
        )
    return pd.DataFrame(out_rows).sort_values(
        ["sample_occurrence", "object_count", "program_pattern_id", "domain_deployment_pattern_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def build_niche_cross_sample_structure_catalog(
    inputs: CohortReportingInputs,
    sample_atlas_payloads: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "niche_structure_pattern_id",
        "member_program_recipe_signature",
        "member_role_recipe_signature",
        "dominant_contact_pair",
        "secondary_contact_pair",
        "contact_structure_hint",
        "cross_sample_contact_class",
        "sample_occurrence",
        "object_count",
        "representative_niche_ids",
        "stability_class",
    ]
    program_pattern_map, program_df = _program_pattern_map(inputs)
    ordered_program_patterns = sorted(program_df["program_pattern_id"].astype(str).unique().tolist()) if not program_df.empty else []
    role_order = list(ROLE_ORDER)
    row_records: list[dict[str, Any]] = []
    for sample_id, atlas_payload in sample_atlas_payloads.items():
        niche_rows = pd.DataFrame(atlas_payload.get("niche_assembly_matrix_records", []) or [])
        if niche_rows.empty:
            continue
        for _, row in niche_rows.iterrows():
            program_comp = _parse_strip(str(row.get("member_program_composition", "")))
            lifted_program_comp: dict[str, float] = {}
            for raw_program_id, value in program_comp.items():
                pattern_id = program_pattern_map.get((sample_id, str(raw_program_id)), "")
                if pattern_id:
                    lifted_program_comp[pattern_id] = lifted_program_comp.get(pattern_id, 0.0) + _safe_float(value, 0.0)
            member_program_recipe_signature = _reduced_niche_program_recipe_signature(
                lifted_program_comp,
                ordered_program_patterns,
                primary_threshold=0.55,
                secondary_threshold=0.20,
                max_terms=2,
            )
            role_comp = _parse_strip(str(row.get("member_role_composition", "")))
            member_role_recipe_signature = _reduced_signature(role_comp, role_order, keep_threshold=0.20, high_threshold=0.45)
            dominant_contact_pair = str(row.get("dominant_contact_pair", "") or "")
            secondary_contact_pair = str(row.get("secondary_contact_pair", "") or "")
            contact_structure_hint = str(row.get("contact_structure_hint", "") or "")
            cross_sample_contact_class = _cross_sample_contact_class(contact_structure_hint, dominant_contact_pair)
            niche_structure_pattern_id = f"{member_program_recipe_signature}|{dominant_contact_pair}|{cross_sample_contact_class}"
            row_records.append(
                {
                    "sample_id": sample_id,
                    "niche_id": str(row.get("niche_id", "")),
                    "niche_structure_pattern_id": niche_structure_pattern_id,
                    "member_program_recipe_signature": member_program_recipe_signature,
                    "member_role_recipe_signature": member_role_recipe_signature,
                    "dominant_contact_pair": dominant_contact_pair,
                    "secondary_contact_pair": secondary_contact_pair,
                    "contact_structure_hint": contact_structure_hint,
                    "cross_sample_contact_class": cross_sample_contact_class,
                    "member_count": int(_safe_float(row.get("member_count", 0.0), 0.0)),
                    "niche_confidence": _safe_float(row.get("niche_confidence", 0.0)),
                    "organizational_cohesion": _safe_float(row.get("organizational_cohesion", 0.0)),
                    "representative_niche_object": f"{sample_id}:{row.get('niche_id', '')}",
                }
            )
    if not row_records:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=["sample_id", "niche_id", "niche_structure_pattern_id"])
    rows_df = pd.DataFrame(row_records)
    total_samples = len(inputs.sample_bundles)
    out_rows: list[dict[str, Any]] = []
    for keys, sub in rows_df.groupby(
        [
            "niche_structure_pattern_id",
            "member_program_recipe_signature",
            "member_role_recipe_signature",
            "dominant_contact_pair",
            "secondary_contact_pair",
            "contact_structure_hint",
            "cross_sample_contact_class",
        ],
        sort=True,
    ):
        niche_structure_pattern_id, member_program_recipe_signature, member_role_recipe_signature, dominant_contact_pair, secondary_contact_pair, contact_structure_hint, cross_sample_contact_class = keys
        sample_occurrence = int(sub["sample_id"].astype(str).nunique())
        object_count = int(sub.shape[0])
        out_rows.append(
            {
                "niche_structure_pattern_id": str(niche_structure_pattern_id),
                "member_program_recipe_signature": str(member_program_recipe_signature),
                "member_role_recipe_signature": str(member_role_recipe_signature),
                "dominant_contact_pair": str(dominant_contact_pair),
                "secondary_contact_pair": str(secondary_contact_pair),
                "contact_structure_hint": str(contact_structure_hint),
                "cross_sample_contact_class": str(cross_sample_contact_class),
                "sample_occurrence": sample_occurrence,
                "object_count": object_count,
                "representative_niche_ids": _representative_ids(sub["representative_niche_object"].astype(str).tolist()),
                "stability_class": _stability_class(sample_occurrence, total_samples),
            }
        )
    return (
        pd.DataFrame(out_rows).sort_values(
            ["sample_occurrence", "object_count", "niche_structure_pattern_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True),
        rows_df.loc[:, ["sample_id", "niche_id", "niche_structure_pattern_id"]].drop_duplicates().reset_index(drop=True),
    )


def _build_cross_layer_chain_rows(
    inputs: CohortReportingInputs,
    sample_atlas_payloads: dict[str, dict],
    niche_rows_df: pd.DataFrame,
) -> pd.DataFrame:
    if niche_rows_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "program_id",
                "domain_key",
                "niche_id",
                "cross_layer_chain_id",
                "program_pattern_id",
                "domain_deployment_pattern_id",
                "niche_structure_pattern_id",
                "coverage_burden_share",
                "representative_object",
            ]
        )
    program_pattern_map, _ = _program_pattern_map(inputs)
    domain_pattern_lookup: dict[tuple[str, str], str] = {}
    niche_pattern_lookup = {
        (str(row["sample_id"]), str(row["niche_id"])): str(row["niche_structure_pattern_id"])
        for _, row in niche_rows_df.iterrows()
    }
    bundle_lookup = {bundle.sample_id: bundle for bundle in inputs.sample_bundles}
    for sample_id, atlas_payload in sample_atlas_payloads.items():
        domain_records = pd.DataFrame((((atlas_payload.get("sections", {}) or {}).get("domain", {}) or {}).get("program_domain_deployment_matrix", {}) or {}).get("records", []) or [])
        if domain_records.empty:
            continue
        for _, row in domain_records.iterrows():
            program_id = str(row.get("source_program_id", ""))
            program_pattern_id = program_pattern_map.get((sample_id, program_id), "")
            block_state = _block_vs_fragmented(_safe_float(row.get("domain_count", 0.0)), _safe_float(row.get("largest_domain_share_by_spots", 0.0)))
            boundary_state = _boundary_vs_mixed_neighbor(_safe_float(row.get("geo_boundary_ratio_mean", 0.0)), _safe_float(row.get("mixed_neighbor_fraction_mean", 0.0)))
            bridge_state = _bridge_tendency_state(
                _safe_float(row.get("linked_domain_fraction", 0.0)),
                _safe_float(row.get("linked_niche_count", 0.0)),
                _safe_float(row.get("niche_participation_concentration", 0.0)),
            )
            domain_pattern_lookup[(sample_id, program_id)] = f"{block_state}__{boundary_state}__{bridge_state}"
    chain_rows: list[dict[str, Any]] = []
    for sample_id, bundle in bundle_lookup.items():
        if bundle.niche_membership_df.empty or bundle.niche_profile_df.empty:
            continue
        niche_mass_df = bundle.niche_profile_df.copy()
        niche_mass_df["coverage_mass"] = (
            pd.to_numeric(niche_mass_df.get("niche_member_count", niche_mass_df.get("member_count", 0.0)), errors="coerce").fillna(0.0).clip(lower=0.0)
            * pd.to_numeric(niche_mass_df.get("niche_confidence", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        total_mass = float(niche_mass_df["coverage_mass"].sum())
        niche_mass_map = {str(row["niche_id"]): _safe_float(row["coverage_mass"], 0.0) for _, row in niche_mass_df.iterrows()} if "niche_id" in niche_mass_df.columns else {}
        domain_key_to_program = {
            str(row["domain_key"]): str(row["source_program_id"])
            for _, row in bundle.domain_profile_df.loc[:, ["domain_key", "source_program_id"]].drop_duplicates().iterrows()
        } if {"domain_key", "source_program_id"}.issubset(bundle.domain_profile_df.columns) else {}
        for _, membership in bundle.niche_membership_df.iterrows():
            niche_id = str(membership.get("niche_id", ""))
            domain_key = str(membership.get("domain_key", ""))
            program_id = str(membership.get("program_id", "") or domain_key_to_program.get(domain_key, ""))
            program_pattern_id = program_pattern_map.get((sample_id, program_id), "")
            domain_deployment_pattern_id = domain_pattern_lookup.get((sample_id, program_id), "")
            niche_structure_pattern_id = niche_pattern_lookup.get((sample_id, niche_id), "")
            if not (program_pattern_id and domain_deployment_pattern_id and niche_structure_pattern_id):
                continue
            coverage_share = float(_safe_float(niche_mass_map.get(niche_id, 0.0), 0.0) / total_mass) if total_mass > 0 else 0.0
            chain_rows.append(
                {
                    "sample_id": sample_id,
                    "program_id": program_id,
                    "domain_key": domain_key,
                    "niche_id": niche_id,
                    "cross_layer_chain_id": f"{program_pattern_id}|{domain_deployment_pattern_id}|{niche_structure_pattern_id}",
                    "program_pattern_id": program_pattern_id,
                    "domain_deployment_pattern_id": domain_deployment_pattern_id,
                    "niche_structure_pattern_id": niche_structure_pattern_id,
                    "coverage_burden_share": coverage_share,
                    "representative_object": f"{sample_id}:{program_id}:{domain_key}:{niche_id}",
                }
            )
    return pd.DataFrame(chain_rows)


def build_cross_layer_chain_catalog(
    inputs: CohortReportingInputs,
    sample_atlas_payloads: dict[str, dict],
    domain_catalog_df: pd.DataFrame,
    niche_rows_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "cross_layer_chain_id",
        "program_pattern_id",
        "domain_deployment_pattern_id",
        "niche_structure_pattern_id",
        "sample_occurrence",
        "object_count",
        "coverage_burden_share",
        "representative_sample_ids",
        "representative_objects",
        "stability_class",
    ]
    if domain_catalog_df.empty or niche_rows_df.empty:
        return pd.DataFrame(columns=columns)
    df = _build_cross_layer_chain_rows(inputs, sample_atlas_payloads, niche_rows_df)
    if df.empty:
        return pd.DataFrame(columns=columns)
    total_samples = len(inputs.sample_bundles)
    out_rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(["cross_layer_chain_id", "program_pattern_id", "domain_deployment_pattern_id", "niche_structure_pattern_id"], sort=True):
        cross_layer_chain_id, program_pattern_id, domain_deployment_pattern_id, niche_structure_pattern_id = keys
        sample_occurrence = int(sub["sample_id"].astype(str).nunique())
        object_count = int(sub.shape[0])
        if sample_occurrence < CHAIN_SAMPLE_OCCURRENCE_MIN or object_count < CHAIN_OBJECT_COUNT_MIN:
            continue
        out_rows.append(
            {
                "cross_layer_chain_id": str(cross_layer_chain_id),
                "program_pattern_id": str(program_pattern_id),
                "domain_deployment_pattern_id": str(domain_deployment_pattern_id),
                "niche_structure_pattern_id": str(niche_structure_pattern_id),
                "sample_occurrence": sample_occurrence,
                "object_count": object_count,
                "coverage_burden_share": float(pd.to_numeric(sub["coverage_burden_share"], errors="coerce").fillna(0.0).mean()),
                "representative_sample_ids": _representative_ids(sub["sample_id"].astype(str).tolist()),
                "representative_objects": _representative_ids(sub["representative_object"].astype(str).tolist()),
                "stability_class": _stability_class(sample_occurrence, total_samples),
            }
        )
    return pd.DataFrame(out_rows).sort_values(
        ["sample_occurrence", "object_count", "cross_layer_chain_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True) if out_rows else pd.DataFrame(columns=columns)


def build_sample_chain_support_table(
    inputs: CohortReportingInputs,
    sample_atlas_payloads: dict[str, dict],
    domain_catalog_df: pd.DataFrame,
    niche_rows_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "program_id",
        "domain_key",
        "niche_id",
        "cross_layer_chain_id",
        "program_pattern_id",
        "domain_deployment_pattern_id",
        "niche_structure_pattern_id",
        "coverage_burden_share",
    ]
    if domain_catalog_df.empty or niche_rows_df.empty:
        return pd.DataFrame(columns=columns)
    df = _build_cross_layer_chain_rows(inputs, sample_atlas_payloads, niche_rows_df)
    if df.empty:
        return pd.DataFrame(columns=columns)
    group_sizes = (
        df.groupby("cross_layer_chain_id", dropna=False)
        .agg(
            sample_occurrence=("sample_id", lambda x: x.astype(str).nunique()),
            object_count=("cross_layer_chain_id", "size"),
        )
        .reset_index()
    )
    valid_ids = set(
        group_sizes.loc[
            (group_sizes["sample_occurrence"] >= CHAIN_SAMPLE_OCCURRENCE_MIN)
            & (group_sizes["object_count"] >= CHAIN_OBJECT_COUNT_MIN),
            "cross_layer_chain_id",
        ].astype(str)
    )
    if not valid_ids:
        return pd.DataFrame(columns=columns)
    filtered = df.loc[df["cross_layer_chain_id"].astype(str).isin(valid_ids), columns].copy()
    return filtered.sort_values(
        ["sample_id", "coverage_burden_share", "cross_layer_chain_id", "program_id", "domain_key", "niche_id"],
        ascending=[True, False, True, True, True, True],
    ).reset_index(drop=True)


def build_cross_sample_result_triage(
    total_samples: int,
    program_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    niche_df: pd.DataFrame,
    chain_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = [
        ("program", "program_pattern_id", program_df),
        ("domain", "domain_deployment_pattern_id", domain_df),
        ("niche", "niche_structure_pattern_id", niche_df),
        ("chain", "cross_layer_chain_id", chain_df),
    ]
    for result_level, id_col, df in specs:
        if df.empty or id_col not in df.columns:
            continue
        for _, row in df.iterrows():
            sample_occurrence = int(_safe_float(row.get("sample_occurrence", 0.0), 0.0))
            object_count = int(_safe_float(row.get("object_count", 0.0), 0.0))
            stability_class = str(row.get("stability_class", _stability_class(sample_occurrence, total_samples)))
            triage_note = {
                "stable_recurrent": "Observed across a broad sample fraction; suitable for core cross-sample narrative.",
                "conditional_variant": "Observed in multiple samples but not broadly recurrent; useful for heterogeneity analysis.",
                "sample_specific": "Observed in a limited sample set; keep as sample-specific structural note.",
            }.get(stability_class, "")
            rows.append(
                {
                    "result_level": result_level,
                    "result_id": str(row.get(id_col, "")),
                    "stability_class": stability_class,
                    "sample_occurrence": sample_occurrence,
                    "object_count": object_count,
                    "triage_note": triage_note,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["result_level", "sample_occurrence", "object_count", "result_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_cross_sample_triage_overview_df(triage_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level in TRIAGE_LEVELS:
        level_df = triage_df.loc[triage_df["result_level"].astype(str) == level].copy() if not triage_df.empty else pd.DataFrame()
        total_count = int(level_df.shape[0])
        for idx, stability_class in enumerate(TRIAGE_CLASSES):
            count = int((level_df["stability_class"].astype(str) == stability_class).sum()) if not level_df.empty else 0
            rows.append(
                {
                    "result_level": level,
                    "stability_class": stability_class,
                    "count": count,
                    "level_order": TRIAGE_LEVELS.index(level),
                    "class_order": idx,
                    "total_count": total_count,
                }
            )
    return pd.DataFrame(rows)


def build_program_cross_sample_umap_payload(inputs: CohortReportingInputs) -> dict[str, Any]:
    object_rows: list[dict[str, Any]] = []
    for bundle in inputs.sample_bundles:
        if bundle.program_profile_df.empty:
            continue
        eligible_df = bundle.program_profile_df.loc[
            bundle.program_profile_df.get("eligible_for_burden", False).astype(bool)
        ].copy()
        for _, row in eligible_df.iterrows():
            component_anchor = _leading_anchor(row, inputs.component_axes)
            role_anchor = _leading_anchor(row, inputs.role_axes)
            object_rows.append(
                {
                    "sample_id": str(bundle.sample_id),
                    "program_id": str(row.get("program_id", "")),
                    "program_pattern_id": _program_pattern_id(component_anchor, role_anchor),
                    "leading_component_anchor": component_anchor,
                    "leading_role_anchor": role_anchor,
                    "overall_profile_confidence": _safe_float(row.get("overall_profile_confidence", 0.0)),
                }
            )
    objects_df = pd.DataFrame(object_rows)
    base_payload: dict[str, Any] = {
        "status": "skipped",
        "layout_method": PROGRAM_UMAP_LAYOUT_METHOD,
        "distance_source": PROGRAM_UMAP_DISTANCE_SOURCE,
        "distance_imputation": PROGRAM_UMAP_DISTANCE_IMPUTATION,
        "umap_random_state": PROGRAM_UMAP_RANDOM_STATE,
        "umap_n_neighbors": 0,
        "umap_min_dist": PROGRAM_UMAP_MIN_DIST,
        "min_object_count": PROGRAM_UMAP_MIN_OBJECTS,
        "min_pair_coverage": PROGRAM_UMAP_MIN_PAIR_COVERAGE,
        "max_imputed_share": PROGRAM_UMAP_MAX_IMPUTED_SHARE,
        "object_count": int(objects_df.shape[0]),
        "effective_pair_coverage": 0.0,
        "default_distance_share": 1.0,
        "points": [],
    }
    if objects_df.shape[0] < PROGRAM_UMAP_MIN_OBJECTS:
        base_payload["skip_reason"] = f"Program object count {int(objects_df.shape[0])} is below the minimum {PROGRAM_UMAP_MIN_OBJECTS}."
        return base_payload
    try:
        umap_available = importlib.util.find_spec("umap") is not None
    except Exception:
        umap_available = False
    if not umap_available:
        base_payload["skip_reason"] = "umap-learn is not available in the current environment."
        return base_payload

    objects_df = objects_df.sort_values(["sample_id", "program_id"], ascending=[True, True]).reset_index(drop=True)
    object_keys = [(str(r["sample_id"]), str(r["program_id"])) for _, r in objects_df.iterrows()]
    key_to_index = {key: idx for idx, key in enumerate(object_keys)}
    n_objects = len(object_keys)
    distance = np.ones((n_objects, n_objects), dtype=float)
    np.fill_diagonal(distance, 0.0)
    known_pair_keys: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    possible_pairs = 0
    for i in range(n_objects):
        for j in range(i + 1, n_objects):
            if object_keys[i][0] != object_keys[j][0]:
                possible_pairs += 1
    for bundle in inputs.sample_bundles:
        pairs_df = bundle.program_pairs_df.copy()
        if pairs_df.empty:
            continue
        for _, row in pairs_df.iterrows():
            left = (str(row.get("sample_id_a", "")), str(row.get("program_id_a", "")))
            right = (str(row.get("sample_id_b", "")), str(row.get("program_id_b", "")))
            left_idx = key_to_index.get(left)
            right_idx = key_to_index.get(right)
            if left_idx is None or right_idx is None or left_idx == right_idx:
                continue
            sim = _safe_float(row.get("similarity_score", 0.0), 0.0)
            dist = max(0.0, min(1.0, 1.0 - sim))
            distance[left_idx, right_idx] = dist
            distance[right_idx, left_idx] = dist
            pair_key = tuple(sorted((left, right)))
            known_pair_keys.add(pair_key)
    known_pairs = len(known_pair_keys)
    effective_pair_coverage = float(known_pairs / possible_pairs) if possible_pairs > 0 else 0.0
    default_distance_count = 0
    for i in range(n_objects):
        for j in range(i + 1, n_objects):
            if object_keys[i][0] != object_keys[j][0] and distance[i, j] >= 1.0 - 1e-12:
                default_distance_count += 1
    default_distance_share = float(default_distance_count / possible_pairs) if possible_pairs > 0 else 1.0
    base_payload["effective_pair_coverage"] = effective_pair_coverage
    base_payload["default_distance_share"] = default_distance_share
    if effective_pair_coverage < PROGRAM_UMAP_MIN_PAIR_COVERAGE:
        base_payload["skip_reason"] = f"Effective comparability pair coverage {effective_pair_coverage:.3f} is below the minimum {PROGRAM_UMAP_MIN_PAIR_COVERAGE:.2f}."
        return base_payload
    if default_distance_share > PROGRAM_UMAP_MAX_IMPUTED_SHARE:
        base_payload["skip_reason"] = f"Default-distance share {default_distance_share:.3f} exceeds the maximum {PROGRAM_UMAP_MAX_IMPUTED_SHARE:.2f}."
        return base_payload

    import umap.umap_ as umap  # type: ignore

    n_neighbors = max(2, min(15, n_objects - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=PROGRAM_UMAP_MIN_DIST,
        metric="precomputed",
        random_state=PROGRAM_UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(distance)
    points: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(objects_df.iterrows()):
        points.append(
            {
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "sample_id": str(row["sample_id"]),
                "program_id": str(row["program_id"]),
                "program_pattern_id": str(row["program_pattern_id"]),
                "leading_component_anchor": str(row["leading_component_anchor"]),
                "leading_role_anchor": str(row["leading_role_anchor"]),
                "overall_profile_confidence": float(row["overall_profile_confidence"]),
                "layout_method": PROGRAM_UMAP_LAYOUT_METHOD,
                "distance_source": PROGRAM_UMAP_DISTANCE_SOURCE,
                "distance_imputation": PROGRAM_UMAP_DISTANCE_IMPUTATION,
                "umap_random_state": PROGRAM_UMAP_RANDOM_STATE,
                "umap_n_neighbors": n_neighbors,
                "umap_min_dist": PROGRAM_UMAP_MIN_DIST,
            }
        )
    base_payload.update(
        {
            "status": "ok",
            "skip_reason": "",
            "umap_n_neighbors": n_neighbors,
            "points": points,
        }
    )
    return base_payload


def build_cross_sample_synthesis_payload(
    program_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    niche_df: pd.DataFrame,
    chain_df: pd.DataFrame,
    triage_df: pd.DataFrame,
) -> dict[str, Any]:
    def _summary(df: pd.DataFrame, id_col: str) -> dict[str, Any]:
        if df.empty:
            return {
                "pattern_count": 0,
                "stable_recurrent_count": 0,
                "conditional_variant_count": 0,
                "sample_specific_count": 0,
                "top_ids": [],
            }
        return {
            "pattern_count": int(df[id_col].astype(str).nunique()),
            "stable_recurrent_count": int((df["stability_class"].astype(str) == "stable_recurrent").sum()),
            "conditional_variant_count": int((df["stability_class"].astype(str) == "conditional_variant").sum()),
            "sample_specific_count": int((df["stability_class"].astype(str) == "sample_specific").sum()),
            "top_ids": df[id_col].astype(str).head(5).tolist(),
        }

    return {
        "cross_sample_research_mode": "structure_synthesis",
        "pattern_unit": "rule_based_archetypes",
        "result_classes": TRIAGE_CLASSES,
        "sections": {
            "program_patterns": {
                "question": "Which Program mechanism patterns recur across samples?",
                "summary": _summary(program_df, "program_pattern_id"),
                "table_ref": "tables/program_cross_sample_pattern_catalog.csv",
            },
            "domain_deployment": {
                "question": "How do recurrent Program patterns get spatially deployed across samples?",
                "summary": _summary(domain_df, "domain_deployment_pattern_id"),
                "table_ref": "tables/domain_cross_sample_deployment_catalog.csv",
            },
            "niche_structures": {
                "question": "Which local structure recipes recur across samples?",
                "summary": _summary(niche_df, "niche_structure_pattern_id"),
                "table_ref": "tables/niche_cross_sample_structure_catalog.csv",
            },
            "cross_layer_chains": {
                "question": "Which cross-layer object chains recur strongly enough to count as structural rules?",
                "summary": _summary(chain_df, "cross_layer_chain_id"),
                "table_ref": "tables/cross_layer_chain_catalog.csv",
            },
        },
        "triage_table_ref": "tables/cross_sample_result_triage.csv",
        "triage_counts": {
            label: int((triage_df["stability_class"].astype(str) == label).sum()) if not triage_df.empty else 0
            for label in TRIAGE_CLASSES
        },
    }
