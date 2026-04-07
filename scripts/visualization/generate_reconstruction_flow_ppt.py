#!/usr/bin/env python3
"""Generate a PPT flow report for the coronary 3D-2D reconstruction plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.reconstruction_3d2d.presentation import (  # noqa: E402
    ReconstructionFlowPptConfig,
    build_reconstruction_flow_ppt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PPT report for the coronary 3D-2D reconstruction flow.")
    parser.add_argument("--case-id", type=str, default="Normal_2", help="Case id shown in the PPT.")
    parser.add_argument("--output", type=Path, required=True, help="Output pptx path.")
    parser.add_argument("--overview-png", type=Path, default=None, help="Optional stage5 overview image.")
    parser.add_argument("--projection-png", type=Path, default=None, help="Optional synthetic projection preview image.")
    parser.add_argument("--title", type=str, default="冠脉 3D-2D 重建流程图", help="PPT title.")
    parser.add_argument("--subtitle", type=str, default="从 clean tree 到 synthetic X-ray 的最小闭环", help="PPT subtitle.")
    parser.add_argument("--note", action="append", default=None, help="Optional note bullets appended to the last slide.")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else (REPO_ROOT / args.output).resolve()
    overview_png = None if args.overview_png is None else (args.overview_png if args.overview_png.is_absolute() else (REPO_ROOT / args.overview_png).resolve())
    projection_png = None if args.projection_png is None else (args.projection_png if args.projection_png.is_absolute() else (REPO_ROOT / args.projection_png).resolve())
    config = ReconstructionFlowPptConfig(
        case_id=args.case_id,
        output_path=output,
        title=args.title,
        subtitle=args.subtitle,
        overview_png=overview_png,
        projection_png=projection_png,
        notes=tuple(args.note or ()),
    )
    path = build_reconstruction_flow_ppt(config)
    print(f"pptx: {path}")


if __name__ == "__main__":
    main()
