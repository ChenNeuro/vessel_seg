#!/usr/bin/env python3
"""Project an existing repaired coronary tree into a synthetic X-ray detector view."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.reconstruction_3d2d import (  # noqa: E402
    BranchLatentState,
    CArmConfig,
    DetectorConfig,
    HeartState,
    Pose3D,
    project_case_centerlines,
    render_projection_preview,
    render_projection_wall_preview,
    save_projection_result_json,
)


def _resolve_inputs(case_dir: Path | None, tree_json: Path | None, centerline_vtp: Path | None) -> tuple[Path, Path]:
    if case_dir is not None:
        repair_dir = case_dir / "stages" / "03_centerline_repair"
        return (repair_dir / "tree.json", repair_dir / "centerline_repaired.vtp")
    if tree_json is None or centerline_vtp is None:
        raise ValueError("Either --case-dir or both --tree-json and --centerline-vtp must be provided.")
    return (tree_json, centerline_vtp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project a repaired centerline tree into a synthetic X-ray view.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Case directory under outputs_reorganized_runs/cases/<case>.")
    parser.add_argument("--tree-json", type=Path, default=None, help="Path to stage3 tree.json.")
    parser.add_argument("--centerline-vtp", type=Path, default=None, help="Path to stage3 centerline_repaired.vtp.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for synthetic projection artifacts.")
    parser.add_argument("--lao-rao-deg", type=float, default=25.0, help="LAO/RAO angle in degrees.")
    parser.add_argument("--cra-cau-deg", type=float, default=10.0, help="CRA/CAU angle in degrees.")
    parser.add_argument("--sid-mm", type=float, default=1200.0, help="Source to detector distance in mm.")
    parser.add_argument("--sod-mm", type=float, default=750.0, help="Source to isocenter distance in mm.")
    parser.add_argument("--detector-width-px", type=int, default=1024, help="Detector width in pixels.")
    parser.add_argument("--detector-height-px", type=int, default=1024, help="Detector height in pixels.")
    parser.add_argument("--pixel-spacing-mm", type=float, default=0.30, help="Detector pixel spacing in mm.")
    parser.add_argument("--heart-translation-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Heart translation in world coordinates.")
    parser.add_argument("--heart-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Heart roll/pitch/yaw in degrees.")
    parser.add_argument("--ecg-phase", type=float, default=0.0, help="Normalized ECG phase in [0, 1].")
    args = parser.parse_args()

    case_dir = None if args.case_dir is None else (args.case_dir if args.case_dir.is_absolute() else (REPO_ROOT / args.case_dir).resolve())
    tree_json, centerline_vtp = _resolve_inputs(
        case_dir,
        None if args.tree_json is None else (args.tree_json if args.tree_json.is_absolute() else (REPO_ROOT / args.tree_json).resolve()),
        None if args.centerline_vtp is None else (args.centerline_vtp if args.centerline_vtp.is_absolute() else (REPO_ROOT / args.centerline_vtp).resolve()),
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir).resolve()

    c_arm = CArmConfig(
        lao_rao_deg=float(args.lao_rao_deg),
        cra_cau_deg=float(args.cra_cau_deg),
        sid_mm=float(args.sid_mm),
        sod_mm=float(args.sod_mm),
        detector=DetectorConfig(
            width_px=int(args.detector_width_px),
            height_px=int(args.detector_height_px),
            pixel_spacing_mm=float(args.pixel_spacing_mm),
        ),
    )
    heart_state = HeartState(
        world_pose=Pose3D(
            translation_mm=tuple(float(value) for value in args.heart_translation_mm),
            rpy_deg=tuple(float(value) for value in args.heart_rpy_deg),
        ),
        ecg_phase=float(args.ecg_phase),
        left_state=BranchLatentState(side_group="LCA"),
        right_state=BranchLatentState(side_group="RCA"),
    )
    result = project_case_centerlines(tree_json, centerline_vtp, c_arm, heart_state=heart_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    result_json = out_dir / "synthetic_projection.json"
    preview_png = out_dir / "projection_preview.png"
    wall_preview_png = out_dir / "projection_wall_preview.png"
    save_projection_result_json(result, result_json)
    render_projection_preview(result, preview_png)
    render_projection_wall_preview(result, wall_preview_png)

    print(f"result_json: {result_json}")
    print(f"preview_png: {preview_png}")
    print(f"wall_preview_png: {wall_preview_png}")
    print(f"case_id: {result.case_id}")
    print(f"visible_branches: {result.metadata.get('visible_branch_count')}/{result.metadata.get('branch_count')}")


if __name__ == "__main__":
    main()
