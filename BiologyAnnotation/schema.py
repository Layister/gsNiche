from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProgramAnnotationResourceConfig:
    annotation_source: str | None = None
    source_profile_yaml_relpath: str = "resources/program_annotation_sources.yaml"
