from __future__ import annotations

import logging
import sys
from pathlib import Path

# Keep package imports working when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ProgramVectorizer.pipeline import ProgramPipelineConfig, run_program_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


# Example paths.
work_dir = Path("/Users/wuyang/Documents/SC-ST data")
cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]

"""
cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]

cancer = "PAAD"
sample_ids = ["NCBI569", "NCBI570", "NCBI571", "NCBI572"]

cancer = "IDC"
sample_ids = ["NCBI681", "NCBI682", "NCBI683", "TENX13", "TENX14"]

cancer = "PRAD"
sample_ids = ["INT25", "INT26", "INT27", "INT28", "TENX40", "TENX46"]

cancer = "EPM"
sample_ids = ["NCBI629", "NCBI630", "NCBI631", "NCBI632", "NCBI633"]
"""

pipeline_config = ProgramPipelineConfig()


def main() -> None:
    for sample_id in sample_ids:
        gss_bundle = work_dir / cancer / "ST" / sample_id / "gss_bundle"
        out_root = work_dir / cancer / "ST"
        print(f"Processing {cancer} / {sample_id} ...")
        program_bundle = run_program_pipeline(
            gss_bundle_path=gss_bundle,
            out_root=out_root,
            sample_id=sample_id,
            config=pipeline_config,
        )
        print(f"Done: {program_bundle}")


if __name__ == "__main__":
    main()
