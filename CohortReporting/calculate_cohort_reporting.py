from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CohortReporting.pipeline import run_cohort_reporting_pipeline
from CohortReporting.schema import CohortReportingConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


work_dir = Path("/Users/wuyang/Documents/SC-ST data")
cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]
pipeline_config = CohortReportingConfig()


def main() -> None:
    out_root = work_dir / cancer / "ST"
    reporting_dir = run_cohort_reporting_pipeline(
        out_root=out_root,
        sample_ids=sample_ids,
        cancer_type=cancer,
        config=pipeline_config,
    )
    print(f"Done: {reporting_dir}")


if __name__ == "__main__":
    main()
