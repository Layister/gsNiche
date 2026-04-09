from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Representation.pipeline import run_representation_pipeline
from Representation.schema import RepresentationPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


work_dir = Path("/Users/wuyang/Documents/SC-ST data")
cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]

pipeline_config = RepresentationPipelineConfig()


def main() -> None:
    for sample_id in sample_ids:
        program_bundle = work_dir / cancer / "ST" / sample_id / "program_bundle"
        out_root = work_dir / cancer / "ST"
        print(f"Processing representation for {cancer} / {sample_id} ...")
        representation_bundle = run_representation_pipeline(
            program_bundle_path=program_bundle,
            out_root=out_root,
            sample_id=sample_id,
            cancer_type=cancer,
            config=pipeline_config,
        )
        print(f"Done: {representation_bundle}")


if __name__ == "__main__":
    main()
