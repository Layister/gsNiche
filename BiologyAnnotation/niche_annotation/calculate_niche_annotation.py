from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BiologyAnnotation.niche_annotation.interpret_niches import (
    NicheAnnotationConfig,
    run_niche_annotation,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
)


work_dir = Path("/Users/wuyang/Documents/SC-ST data")
cancer = "COAD"
sample_ids = ["TENX89", "TENX90", "TENX91", "TENX92"]

annotation_config = NicheAnnotationConfig()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run niche annotation for one or more samples.")
    parser.add_argument("--niche-bundle", type=str, default=None, help="Path to a niche_bundle directory.")
    parser.add_argument("--work-dir", type=str, default=str(work_dir), help=f"Root directory (default: {work_dir}).")
    parser.add_argument("--cancer", type=str, default=str(cancer), help=f"Cancer cohort (default: {cancer}).")
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=",".join(sample_ids),
        help=f"Comma-separated sample ids (default: {','.join(sample_ids)}). Ignored when --niche-bundle is provided.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Valid only for a single bundle/sample run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    out_dir_arg = Path(args.out_dir).resolve() if args.out_dir else None

    if args.niche_bundle:
        niche_bundles = [Path(args.niche_bundle).resolve()]
    else:
        sample_id_list = [x.strip() for x in str(args.sample_ids).split(",") if x.strip()]
        niche_bundles = [
            Path(args.work_dir).resolve() / args.cancer / "ST" / sample_id / "niche_bundle"
            for sample_id in sample_id_list
        ]

    if out_dir_arg is not None and len(niche_bundles) != 1:
        raise ValueError("--out-dir can only be used when annotating a single niche bundle.")

    for niche_bundle in niche_bundles:
        sample_id = niche_bundle.parent.name
        out_dir = out_dir_arg if out_dir_arg is not None else (niche_bundle / "niche_annotation")

        print(f"Annotating niches for {niche_bundle.parent.parent.parent.name} / {sample_id} ...")
        out = run_niche_annotation(
            niche_bundle_path=niche_bundle,
            out_dir=out_dir,
            config=annotation_config,
        )
        print(f"Done: {out}")


if __name__ == "__main__":
    main()
