from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BiologyAnnotation.program_annotation.interpret_programs import ProgramAnnotationConfig, run_program_annotation

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


work_dir = Path("/Users/wuyang/Documents/SC-ST data")

cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]

source_profile_yaml = Path("/Users/wuyang/Documents/MyPaper/3/gsNiche/resources/program_annotation_sources.yaml")

annotation_config = ProgramAnnotationConfig()


def main() -> None:
    for sample_id in sample_ids:
        program_bundle = work_dir / cancer / "ST" / sample_id / "program_bundle"
        out_dir = program_bundle / "program_annotation"

        print(f"Annotating {cancer} / {sample_id} ...")
        out = run_program_annotation(
            program_bundle_path=program_bundle,
            source_profile_yaml=source_profile_yaml,
            out_dir=out_dir,
            config=annotation_config,
        )
        print(f"Done: {out}")


if __name__ == "__main__":
    main()
