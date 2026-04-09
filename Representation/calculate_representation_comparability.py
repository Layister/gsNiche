from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Representation.comparability import run_representation_comparability
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
    out_root = work_dir / cancer / "ST"
    updated = run_representation_comparability(
        out_root=out_root,
        sample_ids=sample_ids,
        cancer_type=cancer,
        config=pipeline_config,
    )
    for bundle_dir in updated:
        print(f"Updated comparability: {bundle_dir}")


if __name__ == "__main__":
    main()
