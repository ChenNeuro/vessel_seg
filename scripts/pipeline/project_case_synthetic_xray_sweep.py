#!/usr/bin/env python3
"""Generate a fixed set of synthetic C-arm views and a gallery for one case."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.reconstruction_3d2d.sweep import (  # noqa: E402
    DEFAULT_VIEW_SPECS,
    ProjectionViewSpec,
    run_projection_sweep,
    save_projection_sweep_gallery,
)


def _resolve_inputs(case_dir: Path | None, tree_json: Path | None, centerline_vtp: Path | None) -> tuple[Path, Path]:
    if case_dir is not None:
        repair_dir = case_dir / "stages" / "03_centerline_repair"
        return (repair_dir / "tree.json", repair_dir / "centerline_repaired.vtp")
    if tree_json is None or centerline_vtp is None:
        raise ValueError("Either --case-dir or both --tree-json and --centerline-vtp must be provided.")
    return (tree_json, centerline_vtp)


def _parse_view_spec(raw: str) -> ProjectionViewSpec:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Invalid --view '{raw}'. Expected format name:alpha_lao_rao:beta_cra_cau")
    return ProjectionViewSpec(parts[0], float(parts[1]), float(parts[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple synthetic X-ray views and a summary gallery.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Case directory under outputs_reorganized_runs/cases/<case>.")
    parser.add_argument("--tree-json", type=Path, default=None, help="Path to stage3 tree.json.")
    parser.add_argument("--centerline-vtp", type=Path, default=None, help="Path to stage3 centerline_repaired.vtp.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for sweep artifacts.")
    parser.add_argument(
        "--view",
        action="append",
        default=None,
        help="Optional custom view spec name:alpha_lao_rao:beta_cra_cau. Can be repeated.",
    )
    parser.add_argument("--bed-longitudinal-mm", type=float, default=0.0, help="Bed longitudinal translation in W0.")
    parser.add_argument("--bed-lateral-mm", type=float, default=0.0, help="Bed lateral translation in W0.")
    parser.add_argument("--bed-height-mm", type=float, default=0.0, help="Bed height translation in W0.")
    parser.add_argument("--bed-tilt-deg", type=float, default=0.0, help="Simplified single-axis bed tilt.")
    parser.add_argument(
        "--bed-from-heart-translation-mm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Heart translation expressed in bed frame B.",
    )
    parser.add_argument(
        "--bed-from-heart-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Heart roll/pitch/yaw expressed in bed frame B.",
    )
    parser.add_argument(
        "--coronary-frame-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Residual local coronary-frame roll/pitch/yaw inside H.",
    )
    parser.add_argument("--ecg-phase", type=float, default=0.0, help="Normalized ECG phase in [0, 1].")
    parser.add_argument("--source-to-detector-mm", type=float, default=1000.0, help="Source to detector distance in mm.")
    parser.add_argument("--source-to-isocenter-mm", type=float, default=765.0, help="Source to isocenter distance in mm.")
    parser.add_argument("--detector-width-px", type=int, default=1024, help="Detector width in pixels.")
    parser.add_argument("--detector-height-px", type=int, default=1024, help="Detector height in pixels.")
    parser.add_argument("--pixel-spacing-mm", type=float, default=0.30, help="Detector pixel spacing in mm.")
    args = parser.parse_args()

    case_dir = None if args.case_dir is None else (args.case_dir if args.case_dir.is_absolute() else (REPO_ROOT / args.case_dir).resolve())
    tree_json, centerline_vtp = _resolve_inputs(
        case_dir,
        None if args.tree_json is None else (args.tree_json if args.tree_json.is_absolute() else (REPO_ROOT / args.tree_json).resolve()),
        None if args.centerline_vtp is None else (args.centerline_vtp if args.centerline_vtp.is_absolute() else (REPO_ROOT / args.centerline_vtp).resolve()),
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir).resolve()
    view_specs = tuple(_parse_view_spec(item) for item in args.view) if args.view else DEFAULT_VIEW_SPECS

    image_paths = run_projection_sweep(
        tree_json_path=tree_json,
        centerline_vtp_path=centerline_vtp,
        output_dir=out_dir,
        view_specs=view_specs,
        bed_longitudinal_mm=float(args.bed_longitudinal_mm),
        bed_lateral_mm=float(args.bed_lateral_mm),
        bed_height_mm=float(args.bed_height_mm),
        bed_tilt_deg=float(args.bed_tilt_deg),
        bed_from_heart_translation_mm=tuple(float(value) for value in args.bed_from_heart_translation_mm),
        bed_from_heart_rpy_deg=tuple(float(value) for value in args.bed_from_heart_rpy_deg),
        coronary_frame_rpy_deg=tuple(float(value) for value in args.coronary_frame_rpy_deg),
        ecg_phase=float(args.ecg_phase),
        source_to_detector_mm=float(args.source_to_detector_mm),
        source_to_isocenter_mm=float(args.source_to_isocenter_mm),
        detector_width_px=int(args.detector_width_px),
        detector_height_px=int(args.detector_height_px),
        pixel_spacing_mm=float(args.pixel_spacing_mm),
    )
    gallery_path = save_projection_sweep_gallery(
        output_dir=out_dir,
        view_specs=view_specs,
        gallery_name="projection_sweep_gallery.png",
        image_name="projection_preview.png",
    )
    wall_gallery_path = save_projection_sweep_gallery(
        output_dir=out_dir,
        view_specs=view_specs,
        gallery_name="projection_wall_sweep_gallery.png",
        image_name="projection_wall_preview.png",
    )

    print(f"output_dir: {out_dir}")
    print(f"gallery_png: {gallery_path}")
    print(f"wall_gallery_png: {wall_gallery_path}")
    print(f"image_count: {len(image_paths)}")


if __name__ == "__main__":
    main()
