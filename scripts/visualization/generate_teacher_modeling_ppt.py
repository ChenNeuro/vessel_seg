#!/usr/bin/env python3
"""Generate the teacher-facing modeling-flow PPT from repo assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.reconstruction_3d2d import (  # noqa: E402
    TeacherModelingPptConfig,
    build_teacher_modeling_notes,
    build_teacher_modeling_ppt,
)


def _default_asset(path_str: str) -> Path:
    return (REPO_ROOT / path_str).resolve()


def _resolved_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a teacher-facing PPT for coronary modeling flow and data requirements.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated/teacher_modeling_flow_20260408.pptx"),
        help="Output PPTX path.",
    )
    parser.add_argument(
        "--notes-output",
        type=Path,
        default=Path("docs/generated/teacher_modeling_flow_20260408_notes.md"),
        help="Optional markdown speaker notes output.",
    )
    parser.add_argument("--title", type=str, default="冠脉建模流程与所需数据", help="PPT title.")
    parser.add_argument(
        "--subtitle",
        type=str,
        default="基于仓库现状，面向“现有分割 -> 世界系放置 -> X 光重建”的汇报版本",
        help="PPT subtitle.",
    )
    parser.add_argument(
        "--pipeline-overview-png",
        type=Path,
        default=_default_asset("docs/archive/assets/teacher_report_20260209/pipeline_overview.png"),
        help="Pipeline overview image.",
    )
    parser.add_argument(
        "--cross-case-summary-png",
        type=Path,
        default=_default_asset("docs/archive/assets/teacher_report_20260209/cross_case_summary.png"),
        help="Cross-case summary image.",
    )
    parser.add_argument(
        "--normal1-centerline-png",
        type=Path,
        default=_default_asset("docs/archive/assets/teacher_report_20260209/normal_1_step3_centerline.png"),
        help="Optional Normal_1 centerline image.",
    )
    parser.add_argument(
        "--normal2-centerline-png",
        type=Path,
        default=_default_asset("docs/archive/assets/teacher_report_20260209/normal_2_step3_centerline.png"),
        help="Optional Normal_2 centerline image.",
    )
    args = parser.parse_args()

    config = TeacherModelingPptConfig(
        output_path=_resolved_path(args.output),
        notes_path=_resolved_path(args.notes_output),
        title=args.title,
        subtitle=args.subtitle,
        pipeline_overview_png=_resolved_path(args.pipeline_overview_png),
        cross_case_summary_png=_resolved_path(args.cross_case_summary_png),
        normal1_centerline_png=_resolved_path(args.normal1_centerline_png),
        normal2_centerline_png=_resolved_path(args.normal2_centerline_png),
    )
    ppt_path = build_teacher_modeling_ppt(config)
    notes_path = build_teacher_modeling_notes(config)
    print(f"pptx: {ppt_path}")
    if notes_path is not None:
        print(f"notes: {notes_path}")


if __name__ == "__main__":
    main()
