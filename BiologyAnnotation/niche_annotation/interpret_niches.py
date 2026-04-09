from __future__ import annotations

import argparse
from dataclasses import dataclass
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
class NicheAnnotationConfig:
    niche_structures_relpath: str = "niche_structures.parquet"
    niche_membership_relpath: str = "niche_membership.parquet"
    niche_edges_relpath: str = "domain_adjacency_edges.parquet"
    domain_annotation_table_relpath: str = "domain_annotation/domain_annotation_table.tsv"
    program_annotation_relpath: str = "program_annotation/program_annotation_summary.json"
    output_dirname: str = "niche_annotation"


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clean_label(text: object) -> str:
    label = str(text or "").strip()
    return label if label else ""


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(x) for x in value]
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


def _json_dumps(value: object) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)


def _load_program_annotation_map(program_annotation_path: Path) -> dict[str, dict]:
    if not program_annotation_path.exists():
        return {}
    payload = read_json(program_annotation_path)
    if isinstance(payload, list):
        return {
            str(record.get("program_id", "")): dict(record)
            for record in payload
            if str(record.get("program_id", "")).strip()
        }
    if isinstance(payload, dict):
        return {str(k): dict(v) for k, v in payload.items() if str(k).strip()}
    return {}


def _infer_bundle_paths(niche_bundle: Path, cfg: NicheAnnotationConfig) -> tuple[dict, Path, Path, Path]:
    manifest_path = niche_bundle / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing niche manifest: {manifest_path}")
    manifest = read_json(manifest_path)

    domain_bundle_path = Path(str(manifest.get("inputs", {}).get("domain_bundle_path", "")).strip())
    program_bundle_path = Path(str(manifest.get("inputs", {}).get("program_bundle_path", "")).strip())
    if not str(domain_bundle_path):
        raise ValueError(f"niche manifest missing inputs.domain_bundle_path: {manifest_path}")
    if not str(program_bundle_path):
        raise ValueError(f"niche manifest missing inputs.program_bundle_path: {manifest_path}")

    domain_annotation_table = domain_bundle_path / cfg.domain_annotation_table_relpath
    program_annotation_summary = program_bundle_path / cfg.program_annotation_relpath
    return manifest, domain_bundle_path, domain_annotation_table, program_annotation_summary


def _load_inputs(niche_bundle: Path, cfg: NicheAnnotationConfig) -> dict:
    manifest, _, domain_annotation_table, program_annotation_summary = _infer_bundle_paths(
        niche_bundle=niche_bundle,
        cfg=cfg,
    )

    structures = pd.read_parquet(niche_bundle / cfg.niche_structures_relpath)
    membership = pd.read_parquet(niche_bundle / cfg.niche_membership_relpath)
    edges = pd.read_parquet(niche_bundle / cfg.niche_edges_relpath)
    domain_ann = pd.read_csv(domain_annotation_table, sep="\t") if domain_annotation_table.exists() else pd.DataFrame()
    program_ann = _load_program_annotation_map(program_annotation_summary)

    for df, col in [(structures, "niche_id"), (membership, "niche_id"), (membership, "domain_key")]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if not domain_ann.empty and "domain_key" in domain_ann.columns:
        domain_ann["domain_key"] = domain_ann["domain_key"].astype(str)
    for col in ["domain_key_i", "domain_key_j", "program_id_i", "program_id_j"]:
        if col in edges.columns:
            edges[col] = edges[col].astype(str)

    return {
        "manifest": manifest,
        "structures": structures,
        "membership": membership,
        "edges": edges,
        "domain_annotation": domain_ann,
        "program_annotation": program_ann,
    }


def _domain_record_map(domain_annotation: pd.DataFrame) -> dict[str, dict]:
    if domain_annotation.empty or "domain_key" not in domain_annotation.columns:
        return {}
    return {
        str(row["domain_key"]): dict(row)
        for row in domain_annotation.to_dict(orient="records")
        if str(row.get("domain_key", "")).strip()
    }


def _program_evidence(program_id: str, program_annotations: dict[str, dict]) -> dict:
    record = dict(program_annotations.get(str(program_id), {}) or {})
    term_scores = record.get("term_scores", {})
    displayed_ids = record.get("displayed_term_ids", [])
    return {
        "program_term_scores": term_scores if isinstance(term_scores, dict) else {},
        "program_displayed_term_ids": displayed_ids if isinstance(displayed_ids, list) else [],
        "program_annotation_confidence": _coerce_float(
            record.get("annotation_confidence", record.get("program_confidence", 0.0)),
            default=0.0,
        ),
    }


def _member_record(
    member_row: dict,
    domain_records: dict[str, dict],
    program_annotations: dict[str, dict],
) -> dict:
    domain_key = str(member_row.get("domain_key", ""))
    program_id = str(member_row.get("program_id", ""))
    domain_rec = dict(domain_records.get(domain_key, {}) or {})
    program_ev = _program_evidence(program_id, program_annotations)
    return {
        "domain_key": domain_key,
        "program_id": program_id,
        "is_backbone_member": bool(member_row.get("is_backbone_member", False)),
        "is_structure_member": bool(member_row.get("is_structure_member", False)),
        "domain_annotation_label": _clean_label(domain_rec.get("domain_annotation_label", "")),
        "morphology_label": _clean_label(domain_rec.get("morphology_label", "")),
        "salience_label": _clean_label(domain_rec.get("salience_label", "")),
        "annotation_confidence_label": _clean_label(domain_rec.get("annotation_confidence_label", "")),
        "annotation_term_count": int(_coerce_float(domain_rec.get("annotation_term_count", 0), default=0.0)),
        "domain_reliability": _coerce_float(
            domain_rec.get("domain_reliability", member_row.get("domain_reliability", 0.0)),
            default=0.0,
        ),
        "salience_score": _coerce_float(domain_rec.get("salience_score", 0.0), default=0.0),
        "spot_count": int(_coerce_float(domain_rec.get("spot_count", 0), default=0.0)),
        "program_term_scores": program_ev["program_term_scores"],
        "program_displayed_term_ids": program_ev["program_displayed_term_ids"],
        "program_annotation_confidence": program_ev["program_annotation_confidence"],
    }


def _edge_type(edge: dict) -> str:
    has_contact = bool(edge.get("is_strong_contact", False))
    has_overlap = bool(edge.get("is_strong_overlap", False))
    if has_contact and has_overlap:
        return "mixed"
    if has_overlap:
        return "overlap"
    return "contact"


def _build_member_table(
    membership: pd.DataFrame,
    domain_records: dict[str, dict],
    program_annotations: dict[str, dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for row in membership.to_dict(orient="records"):
        member = _member_record(row, domain_records=domain_records, program_annotations=program_annotations)
        rows.append(
            {
                "niche_id": str(row.get("niche_id", "")),
                **member,
                "program_term_scores_json": _json_dumps(member["program_term_scores"]),
                "program_displayed_term_ids_json": _json_dumps(member["program_displayed_term_ids"]),
            }
        )
    return pd.DataFrame(rows)


def _build_interface_table(
    structures: pd.DataFrame,
    member_table: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    if member_table.empty or edges.empty:
        return pd.DataFrame()

    structure_lookup = {
        str(row["niche_id"]): dict(row)
        for row in structures.to_dict(orient="records")
    }

    rows: list[dict] = []
    for niche_id, member_sub in member_table.groupby("niche_id"):
        member_lookup = {
            str(row["domain_key"]): dict(row)
            for row in member_sub.to_dict(orient="records")
        }
        member_keys = set(member_lookup.keys())
        edge_sub = edges.loc[
            edges["domain_key_i"].astype(str).isin(member_keys)
            & edges["domain_key_j"].astype(str).isin(member_keys)
            & edges["is_strong_edge"].fillna(False).astype(bool)
        ].copy()
        if edge_sub.empty:
            continue

        struct = structure_lookup.get(str(niche_id), {})
        for edge in edge_sub.to_dict(orient="records"):
            left_key = str(edge.get("domain_key_i", ""))
            right_key = str(edge.get("domain_key_j", ""))
            left = member_lookup.get(left_key, {})
            right = member_lookup.get(right_key, {})
            rows.append(
                {
                    "niche_id": str(niche_id),
                    "canonical_pattern_id": str(struct.get("canonical_pattern_id", "")),
                    "component_id": int(_coerce_float(struct.get("component_id", 0), default=0.0)),
                    "edge_type": _edge_type(edge),
                    "edge_strength": _coerce_float(edge.get("edge_strength", 0.0), default=0.0),
                    "edge_reliability": _coerce_float(edge.get("edge_reliability", 0.0), default=0.0),
                    "domain_i": left_key,
                    "domain_j": right_key,
                    "program_i": str(edge.get("program_id_i", left.get("program_id", ""))),
                    "program_j": str(edge.get("program_id_j", right.get("program_id", ""))),
                    "left_domain_label": _clean_label(left.get("domain_annotation_label", "")),
                    "right_domain_label": _clean_label(right.get("domain_annotation_label", "")),
                    "left_morphology": _clean_label(left.get("morphology_label", "")),
                    "right_morphology": _clean_label(right.get("morphology_label", "")),
                    "left_salience": _clean_label(left.get("salience_label", "")),
                    "right_salience": _clean_label(right.get("salience_label", "")),
                    "left_annotation_term_count": int(_coerce_float(left.get("annotation_term_count", 0), default=0.0)),
                    "right_annotation_term_count": int(_coerce_float(right.get("annotation_term_count", 0), default=0.0)),
                    "left_domain_reliability": _coerce_float(left.get("domain_reliability", 0.0), default=0.0),
                    "right_domain_reliability": _coerce_float(right.get("domain_reliability", 0.0), default=0.0),
                    "left_program_term_scores_json": _json_dumps(left.get("program_term_scores", {})),
                    "right_program_term_scores_json": _json_dumps(right.get("program_term_scores", {})),
                    "left_program_displayed_term_ids_json": _json_dumps(left.get("program_displayed_term_ids", [])),
                    "right_program_displayed_term_ids_json": _json_dumps(right.get("program_displayed_term_ids", [])),
                    "left_program_annotation_confidence": _coerce_float(left.get("program_annotation_confidence", 0.0), default=0.0),
                    "right_program_annotation_confidence": _coerce_float(right.get("program_annotation_confidence", 0.0), default=0.0),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["niche_id", "edge_reliability", "edge_strength", "domain_i", "domain_j"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def _label_composition_summary(values: list[str], empty_default: str, max_items: int = 2) -> str:
    clean = [_clean_label(x) for x in values if _clean_label(x)]
    if not clean:
        return empty_default
    counts = pd.Series(clean).value_counts()
    labels = counts.index.astype(str).tolist()
    if len(labels) == 1:
        return labels[0]
    kept = labels[: max(1, int(max_items))]
    return f"mixed ({' + '.join(kept)})"


def _resolve_core_annotation_term_summary(
    core_annotation_term_summary: object,
    periphery_annotation_term_summary: object,
    top_interfaces: pd.DataFrame,
) -> str:
    core_summary = _clean_label(core_annotation_term_summary)
    if core_summary:
        return core_summary

    periphery_summary = _clean_label(periphery_annotation_term_summary)
    if periphery_summary:
        return periphery_summary

    if top_interfaces.empty:
        return ""

    values: list[str] = []
    for col in ("annotation_term_left", "annotation_term_right"):
        if col not in top_interfaces.columns:
            continue
        for value in top_interfaces[col].astype(str).tolist():
            label = _clean_label(value)
            if label and label not in values:
                values.append(label)
    return "|".join(values)


def _domains_from_interface_side(interface_sub: pd.DataFrame, side_prefix: str) -> pd.DataFrame:
    if interface_sub.empty:
        return pd.DataFrame(
            columns=[
                "domain_key",
                "domain_label",
                "morphology",
                "salience",
                "domain_reliability",
                "annotation_term_count",
            ]
        )
    key_col = "domain_i" if side_prefix == "left" else "domain_j"
    label_col = f"{side_prefix}_domain_label"
    morph_col = f"{side_prefix}_morphology"
    salience_col = f"{side_prefix}_salience"
    term_count_col = f"{side_prefix}_annotation_term_count"
    rel_col = f"{side_prefix}_domain_reliability"
    out = interface_sub[
        [c for c in [key_col, label_col, morph_col, salience_col, term_count_col, rel_col] if c in interface_sub.columns]
    ].copy()
    rename_map = {
        key_col: "domain_key",
        label_col: "domain_label",
        morph_col: "morphology",
        salience_col: "salience",
        term_count_col: "annotation_term_count",
        rel_col: "domain_reliability",
    }
    out = out.rename(columns=rename_map)
    return out.drop_duplicates(subset=["domain_key"]).reset_index(drop=True)


def _core_interface_domain_table(interface_sub: pd.DataFrame) -> pd.DataFrame:
    left = _domains_from_interface_side(interface_sub, "left")
    right = _domains_from_interface_side(interface_sub, "right")
    if left.empty and right.empty:
        return pd.DataFrame(
            columns=[
                "domain_key",
                "domain_label",
                "morphology",
                "salience",
                "domain_reliability",
                "annotation_term_count",
            ]
        )
    return (
        pd.concat([left, right], ignore_index=True)
        .drop_duplicates(subset=["domain_key"])
        .reset_index(drop=True)
    )


def _context_domain_table(member_sub: pd.DataFrame, core_domains: set[str]) -> pd.DataFrame:
    if member_sub.empty:
        return pd.DataFrame()
    sub = member_sub.loc[
        member_sub["is_structure_member"].fillna(False).astype(bool)
        & ~member_sub["domain_key"].astype(str).isin(core_domains)
    ].copy()
    return sub.drop_duplicates(subset=["domain_key"]).reset_index(drop=True)


def _structure_slice_summary(domain_sub: pd.DataFrame, empty_default: str, context: str) -> str:
    if domain_sub.empty:
        return empty_default
    morph = _label_composition_summary(domain_sub.get("morphology", domain_sub.get("morphology_label", pd.Series(dtype=str))).astype(str).tolist(), "mixed")
    salience = _label_composition_summary(domain_sub.get("salience", domain_sub.get("salience_label", pd.Series(dtype=str))).astype(str).tolist(), "mixed")
    return f"{morph} / {salience} {context}"


def _profiled_fraction(member_sub: pd.DataFrame) -> float:
    if member_sub.empty:
        return 0.0
    if "annotation_term_count" in member_sub.columns:
        counts = pd.to_numeric(member_sub["annotation_term_count"], errors="coerce").fillna(0.0)
        return float(np.mean(counts.to_numpy(dtype=float) > 0.0))
    label_col = "domain_label" if "domain_label" in member_sub.columns else "domain_annotation_label"
    if label_col not in member_sub.columns:
        return 0.0
    labels = member_sub[label_col].astype(str)
    return float(np.mean(~labels.str.startswith("unresolved")))


def _unresolved_fraction(member_sub: pd.DataFrame) -> float:
    label_col = "domain_label" if "domain_label" in member_sub.columns else "domain_annotation_label"
    if member_sub.empty or label_col not in member_sub.columns:
        return 0.0
    labels = member_sub[label_col].astype(str)
    return float(np.mean(labels.str.startswith("unresolved")))


def _build_interpretation_table(
    structures: pd.DataFrame,
    member_table: pd.DataFrame,
    interface_table: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    summary_records: list[dict] = []

    interface_lookup = {
        str(niche_id): sub.copy()
        for niche_id, sub in interface_table.groupby("niche_id")
    } if not interface_table.empty else {}

    for struct in structures.to_dict(orient="records"):
        niche_id = str(struct.get("niche_id", ""))
        member_sub = member_table.loc[member_table["niche_id"].astype(str) == niche_id].copy()
        interface_sub = interface_lookup.get(niche_id, pd.DataFrame())
        core_domains_df = _core_interface_domain_table(interface_sub)
        core_domain_keys = set(core_domains_df["domain_key"].astype(str).tolist()) if not core_domains_df.empty else set()
        context_sub = _context_domain_table(member_sub, core_domain_keys)

        edge_type_counts = (
            interface_sub["edge_type"].astype(str).value_counts(normalize=True).to_dict()
            if not interface_sub.empty and "edge_type" in interface_sub.columns
            else {}
        )
        contact_fraction = float(edge_type_counts.get("contact", 0.0))
        overlap_fraction = float(edge_type_counts.get("overlap", 0.0))
        soft_fraction = float(edge_type_counts.get("soft", 0.0))
        core_summary = _structure_slice_summary(core_domains_df, "minimal interface", "core interface")
        context_summary = _structure_slice_summary(context_sub, "minimal surrounding context", "surrounding context")
        core_profiled_fraction = _profiled_fraction(core_domains_df)
        context_unresolved_fraction = _unresolved_fraction(context_sub)
        if context_sub.empty:
            summary_text = (
                f"This niche forms a local interaction structure (contact={contact_fraction:.2f}, overlap={overlap_fraction:.2f}), centered on a {core_summary}."
            )
        else:
            summary_text = (
                f"This niche forms a local interaction structure (contact={contact_fraction:.2f}, overlap={overlap_fraction:.2f}), with a {core_summary} and {context_summary}."
            )

        rows.append(
            {
                "niche_id": niche_id,
                "canonical_pattern_id": str(struct.get("canonical_pattern_id", "")),
                "component_id": int(_coerce_float(struct.get("component_id", 0), default=0.0)),
                "program_ids": str(struct.get("program_ids", "")),
                "backbone_program_pairs": str(struct.get("backbone_program_pairs", "")),
                "member_count": int(_coerce_float(struct.get("member_count", 0), default=0.0)),
                "backbone_node_count": int(_coerce_float(struct.get("backbone_node_count", 0), default=0.0)),
                "backbone_edge_count": int(_coerce_float(struct.get("backbone_edge_count", 0), default=0.0)),
                "interaction_confidence": _coerce_float(struct.get("interaction_confidence", 0.0), default=0.0),
                "contact_fraction": contact_fraction,
                "overlap_fraction": overlap_fraction,
                "soft_fraction": soft_fraction,
                "core_structure_summary": core_summary,
                "context_structure_summary": context_summary,
                "core_profiled_fraction": core_profiled_fraction,
                "context_unresolved_fraction": context_unresolved_fraction,
                "summary_text": summary_text,
            }
        )

        summary_records.append(
            {
                "niche_id": niche_id,
                "canonical_pattern_id": str(struct.get("canonical_pattern_id", "")),
                "component_id": int(_coerce_float(struct.get("component_id", 0), default=0.0)),
                "contact_fraction": contact_fraction,
                "overlap_fraction": overlap_fraction,
                "soft_fraction": soft_fraction,
                "core_structure_summary": core_summary,
                "context_structure_summary": context_summary,
                "summary_text": summary_text,
                "core_profiled_fraction": core_profiled_fraction,
                "context_unresolved_fraction": context_unresolved_fraction,
                "members": [
                    {
                        "domain_key": str(row.get("domain_key", "")),
                        "program_id": str(row.get("program_id", "")),
                        "is_backbone_member": bool(row.get("is_backbone_member", False)),
                        "is_structure_member": bool(row.get("is_structure_member", False)),
                        "domain_annotation_label": _clean_label(row.get("domain_annotation_label", "")),
                        "morphology_label": _clean_label(row.get("morphology_label", "")),
                        "salience_label": _clean_label(row.get("salience_label", "")),
                        "annotation_confidence_label": _clean_label(row.get("annotation_confidence_label", "")),
                        "domain_reliability": _coerce_float(row.get("domain_reliability", 0.0), default=0.0),
                        "salience_score": _coerce_float(row.get("salience_score", 0.0), default=0.0),
                        "spot_count": int(_coerce_float(row.get("spot_count", 0), default=0.0)),
                        "program_term_scores": _json_ready(row.get("program_term_scores", {})),
                        "program_displayed_term_ids": _json_ready(row.get("program_displayed_term_ids", [])),
                        "program_annotation_confidence": _coerce_float(row.get("program_annotation_confidence", 0.0), default=0.0),
                    }
                    for row in member_sub.to_dict(orient="records")
                ],
                "interfaces": [
                    {
                        "edge_type": str(row.get("edge_type", "")),
                        "edge_strength": _coerce_float(row.get("edge_strength", 0.0), default=0.0),
                        "edge_reliability": _coerce_float(row.get("edge_reliability", 0.0), default=0.0),
                        "domain_i": str(row.get("domain_i", "")),
                        "domain_j": str(row.get("domain_j", "")),
                        "program_i": str(row.get("program_i", "")),
                        "program_j": str(row.get("program_j", "")),
                        "left_domain_label": _clean_label(row.get("left_domain_label", "")),
                        "right_domain_label": _clean_label(row.get("right_domain_label", "")),
                        "left_morphology": _clean_label(row.get("left_morphology", "")),
                        "right_morphology": _clean_label(row.get("right_morphology", "")),
                        "left_salience": _clean_label(row.get("left_salience", "")),
                        "right_salience": _clean_label(row.get("right_salience", "")),
                        "left_domain_reliability": _coerce_float(row.get("left_domain_reliability", 0.0), default=0.0),
                        "right_domain_reliability": _coerce_float(row.get("right_domain_reliability", 0.0), default=0.0),
                        "left_program_term_scores": _json_ready(json.loads(str(row.get("left_program_term_scores_json", "{}")))),
                        "right_program_term_scores": _json_ready(json.loads(str(row.get("right_program_term_scores_json", "{}")))),
                        "left_program_displayed_term_ids": _json_ready(json.loads(str(row.get("left_program_displayed_term_ids_json", "[]")))),
                        "right_program_displayed_term_ids": _json_ready(json.loads(str(row.get("right_program_displayed_term_ids_json", "[]")))),
                        "left_program_annotation_confidence": _coerce_float(row.get("left_program_annotation_confidence", 0.0), default=0.0),
                        "right_program_annotation_confidence": _coerce_float(row.get("right_program_annotation_confidence", 0.0), default=0.0),
                    }
                    for row in interface_sub.to_dict(orient="records")
                ],
            }
        )

    interpretation = pd.DataFrame(rows)
    if interpretation.empty:
        return interpretation, summary_records
    return interpretation.sort_values(
        ["interaction_confidence", "member_count", "niche_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True), summary_records


def run_niche_annotation(
    niche_bundle_path: str | Path,
    out_dir: str | Path | None = None,
    config: NicheAnnotationConfig | None = None,
) -> Path:
    cfg = config or NicheAnnotationConfig()
    niche_bundle = Path(niche_bundle_path)
    payload = _load_inputs(niche_bundle=niche_bundle, cfg=cfg)
    structures = payload["structures"]
    membership = payload["membership"]
    edges = payload["edges"]
    domain_records = _domain_record_map(payload["domain_annotation"])
    program_annotations = payload["program_annotation"]

    out_root = Path(out_dir) if out_dir else (niche_bundle / cfg.output_dirname)
    out_root.mkdir(parents=True, exist_ok=True)

    member_table = _build_member_table(
        membership=membership,
        domain_records=domain_records,
        program_annotations=program_annotations,
    )
    interface_table = _build_interface_table(
        structures=structures,
        member_table=member_table,
        edges=edges,
    )
    interpretation, summary_records = _build_interpretation_table(
        structures=structures,
        member_table=member_table,
        interface_table=interface_table,
    )

    write_tsv(interpretation, out_root / "niche_interpretation.tsv")
    write_tsv(interface_table, out_root / "niche_top_interfaces.tsv")
    write_json(
        out_root / "niche_summary.json",
        {
            "sample_id": str(payload["manifest"].get("sample_id", niche_bundle.parent.name)),
            "niches": summary_records,
        },
    )
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpret interaction-first NicheGraph outputs.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--cancer", type=str, default=DEFAULT_CANCER)
    parser.add_argument("--sample-id", type=str, default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--niche-bundle", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    niche_bundle = (
        Path(args.niche_bundle).resolve()
        if args.niche_bundle is not None
        else Path(args.work_dir).resolve() / str(args.cancer) / "ST" / str(args.sample_id) / "niche_bundle"
    )
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else None
    out = run_niche_annotation(
        niche_bundle_path=niche_bundle,
        out_dir=out_dir,
        config=NicheAnnotationConfig(),
    )
    print(out)


if __name__ == "__main__":
    main()
