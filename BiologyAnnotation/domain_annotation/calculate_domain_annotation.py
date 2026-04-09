from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BiologyAnnotation.domain_annotation.interpret_domains import (
    DomainAnnotationConfig,
    run_domain_annotation,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


work_dir = Path("/Users/wuyang/Documents/SC-ST data")
cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]

annotation_config = DomainAnnotationConfig()


def main() -> None:
    for sample_id in sample_ids:
        domain_bundle = work_dir / cancer / "ST" / sample_id / "domain_bundle"
        out_dir = domain_bundle / "domain_annotation"

        print(f"Annotating domains for {cancer} / {sample_id} ...")
        out = run_domain_annotation(
            domain_bundle_path=domain_bundle,
            out_dir=out_dir,
            config=annotation_config,
        )
        print(f"Done: {out}")


if __name__ == "__main__":
    main()
