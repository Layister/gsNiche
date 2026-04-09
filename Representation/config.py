from __future__ import annotations

from pathlib import Path

import yaml

from .schema import AxisDefinition, RepresentationPipelineConfig


_COMPONENT_AXIS_RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources" / "component_axes"


def _read_component_axis_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("component_axes", [])
    if not isinstance(rows, list):
        raise ValueError(f"component_axes must be a list in {path}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _merge_axis_record(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in {"gating", "weights"}:
            merged[key] = {**dict(base.get(key, {}) or {}), **dict(value or {})}
        elif value is not None:
            merged[key] = value
    return merged


def _component_axis_from_record(record: dict) -> AxisDefinition:
    positive_gene_markers = tuple(str(x) for x in record.get("positive_gene_markers", []) if str(x).strip())
    positive_annotation_terms = tuple(str(x) for x in record.get("positive_annotation_terms", []) if str(x).strip())
    evidence_sources = tuple(
        str(x) for x in record.get("evidence_sources", ["scaffold_genes", "ranked_genes", "annotation_summary"])
        if str(x).strip()
    )
    primary_evidence_sources = tuple(
        str(x) for x in record.get("primary_evidence_sources", ["gene_scaffold", "annotation_summary"])
        if str(x).strip()
    )
    return AxisDefinition(
        axis_id=str(record["axis_id"]),
        axis_name=str(record.get("axis_name", record["axis_id"])),
        axis_type="component",
        description=str(record.get("description", record.get("axis_description", ""))),
        axis_description=str(record.get("axis_description", record.get("description", ""))),
        evidence_sources=evidence_sources,
        primary_evidence_sources=primary_evidence_sources,
        positive_gene_markers=positive_gene_markers,
        positive_annotation_terms=positive_annotation_terms,
        gene_markers=positive_gene_markers,
        annotation_keywords=positive_annotation_terms,
        negative_hints=tuple(str(x) for x in record.get("negative_hints", []) if str(x).strip()),
        gating=dict(record.get("gating", {}) or {}),
        weights={str(k): float(v) for k, v in dict(record.get("weights", {}) or {}).items()},
        notes=str(record.get("notes", "")),
    )


def _load_component_axes(cancer_type: str) -> list[AxisDefinition]:
    cancer = str(cancer_type or "COAD").strip().upper() or "COAD"
    common_records = _read_component_axis_records(_COMPONENT_AXIS_RESOURCE_DIR / "common.yaml")
    merged_records = {str(row["axis_id"]): dict(row) for row in common_records if row.get("axis_id")}
    cancer_records = _read_component_axis_records(_COMPONENT_AXIS_RESOURCE_DIR / f"{cancer}.yaml")
    for record in cancer_records:
        axis_id = str(record.get("axis_id", "")).strip()
        if not axis_id:
            continue
        merged_records[axis_id] = _merge_axis_record(merged_records.get(axis_id, {"axis_id": axis_id}), record)
    return [_component_axis_from_record(record) for _, record in sorted(merged_records.items(), key=lambda x: x[0])]


def _generic_role_axes() -> list[AxisDefinition]:
    return [
        AxisDefinition(
            axis_id="interface_like",
            axis_name="interface_like",
            axis_type="role",
            description="Activation looks more boundary-facing or interfacial than purely focal or diffuse.",
            axis_description="Program behaves more like a local boundary, transition band, or heterotypic adjacency zone than a pure hotspot or diffuse background.",
            evidence_sources=("activation_topology", "annotation_summary"),
            primary_evidence_sources=("activation_topology",),
            annotation_keywords=("adhesion", "junction", "epithelial mesenchymal transition", "inflammatory response"),
            positive_annotation_terms=("adhesion", "junction", "epithelial mesenchymal transition", "inflammatory response"),
            topology_preferences={"boundary_fraction": 1.0, "moderate_coverage": 0.8, "mixed_neighbors": 0.8, "mid_hotspot": 0.5},
            topology_hints=("boundary_fraction", "mixed_neighbors", "moderate_coverage", "mid_hotspot"),
            negative_hints=("single compact focal center", "fully diffuse scaffold background"),
            notes="Topology-led role axis. No hard interface detector is used in v1; support should come mainly from coarse boundary-like activation structure.",
        ),
        AxisDefinition(
            axis_id="scaffold_like",
            axis_name="scaffold_like",
            axis_type="role",
            description="Activation behaves like a broad structural backdrop or supportive scaffold.",
            axis_description="Program behaves more like a broad supporting background, continuous carrier band, or structural backbone than a compact hotspot.",
            evidence_sources=("activation_topology", "annotation_summary"),
            primary_evidence_sources=("activation_topology",),
            annotation_keywords=("extracellular matrix", "collagen", "angiogenesis", "myogenesis"),
            positive_annotation_terms=("extracellular matrix", "collagen", "angiogenesis", "myogenesis"),
            topology_preferences={"broad_coverage": 1.0, "high_entropy": 0.8, "low_hotspot": 0.8, "connectedness": 0.6},
            topology_hints=("broad_coverage", "high_entropy", "low_hotspot", "connectedness"),
            negative_hints=("discrete high-peak focal node", "patchy multi-point boundary fragments"),
            notes="Intended for stable, broad, supportive roles rather than discrete hotspots. Gene/annotation cues are only weak secondary hints.",
        ),
        AxisDefinition(
            axis_id="node_like",
            axis_name="node_like",
            axis_type="role",
            description="Activation behaves like a focal node or local hotspot.",
            axis_description="Program behaves more like a compact local hotspot or discrete focal center than a broad supporting field.",
            evidence_sources=("activation_topology", "annotation_summary"),
            primary_evidence_sources=("activation_topology",),
            annotation_keywords=("cell cycle", "mitotic", "proliferation", "hypoxia"),
            positive_annotation_terms=("cell cycle", "mitotic", "proliferation", "hypoxia"),
            topology_preferences={"high_hotspot": 1.0, "high_peakiness": 0.9, "low_coverage": 0.8, "single_component": 0.6},
            topology_hints=("high_hotspot", "high_peakiness", "low_coverage", "single_component"),
            negative_hints=("broad continuous scaffold background", "boundary-like mixed interface field"),
            notes="Represents focal nodes without turning them into a new object class. Topology carries the main meaning here.",
        ),
        AxisDefinition(
            axis_id="companion_like",
            axis_name="companion_like",
            axis_type="role",
            description="Activation behaves like a supportive or accompanying context rather than a dominant structural or focal role.",
            axis_description="Program behaves more like an attached or accompanying component than the main structural backbone or the dominant focal center.",
            evidence_sources=("activation_topology", "annotation_summary"),
            primary_evidence_sources=("activation_topology",),
            annotation_keywords=("complement", "cytokine", "response", "secretory"),
            positive_annotation_terms=("complement", "cytokine", "response", "secretory"),
            topology_preferences={"moderate_coverage": 0.8, "moderate_hotspot": 0.6, "non_dominant": 1.0, "mixed_topology": 0.5},
            topology_hints=("moderate_coverage", "moderate_hotspot", "non_dominant"),
            negative_hints=("dominant broad scaffold", "dominant compact focal node"),
            notes="A conservative role axis for secondary/supportive programs. It should stay weakly defined unless topology supports a non-dominant accompanying role.",
        ),
    ]


def resolve_axis_catalog(cancer_type: str, cfg: RepresentationPipelineConfig | None = None) -> dict[str, list[AxisDefinition]]:
    cancer = str(cancer_type or (cfg.default_cancer_type if cfg else "COAD")).strip().upper() or "COAD"
    return {
        "component_axes": _load_component_axes(cancer),
        "role_axes": _generic_role_axes(),
    }
