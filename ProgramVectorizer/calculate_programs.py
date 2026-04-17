from __future__ import annotations

import logging
import sys
from pathlib import Path

# Keep package imports working when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ProgramVectorizer.pipeline import ProgramPipelineConfig, run_program_pipeline
from utils.dataset_registry import get_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


dataset = get_dataset("DLPFC")
sample_ids = dataset.sample_ids

pipeline_config = ProgramPipelineConfig()


def main() -> None:
    for sample_id in sample_ids:
        gss_bundle = dataset.sample_dir(sample_id) / "gss_bundle"
        out_root = dataset.out_root()
        print(f"Processing {dataset.dataset_id} / {sample_id} ...")
        program_bundle = run_program_pipeline(
            gss_bundle_path=gss_bundle,
            out_root=out_root,
            sample_id=sample_id,
            config=pipeline_config,
        )
        print(f"Done: {program_bundle}")


if __name__ == "__main__":
    main()
