from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Atlas.pipeline import run_atlas_pipeline
from Atlas.schema import AtlasConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


work_dir = Path("/Users/wuyang/Documents/SC-ST data")

cancer = "IDC"
sample_ids = ["NCBI681", "NCBI682", "NCBI683", "TENX13", "TENX14"]

atlas_config = AtlasConfig()


def main() -> None:
    niche_bundle_paths = [work_dir / cancer / "ST" / sample_id / "niche_bundle" for sample_id in sample_ids]
    out_root = work_dir / cancer / "ST"

    print(f"Building Atlas for {cancer} across {len(sample_ids)} samples ...")
    out_dir = run_atlas_pipeline(
        niche_bundle_paths=niche_bundle_paths,
        out_root=out_root,
        cohort_id=cancer,
        config=atlas_config,
    )
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
