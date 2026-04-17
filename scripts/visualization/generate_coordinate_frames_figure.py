#!/usr/bin/env python3
"""Generate a pure coordinate-frame transform figure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.reconstruction_3d2d.coordinate_frames_figure import (  # noqa: E402
    CoordinateFramesFigureConfig,
    build_coordinate_frames_figure,
)
from vessel_seg.reconstruction_3d2d.contracts import (  # noqa: E402
    BedConfig,
    CArmConfig,
    DetectorConfig,
    HeartState,
    Pose3D,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a coordinate-frames transform figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated/coordinate_frames_transform_demo_20260408.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--point-heart-mm", nargs=3, type=float, default=(20.0, 20.0, 10.0), help="Sample point in the heart frame.")
    parser.add_argument("--bed-longitudinal-mm", type=float, default=0.0, help="Bed longitudinal translation in W0.")
    parser.add_argument("--bed-lateral-mm", type=float, default=0.0, help="Bed lateral translation in W0.")
    parser.add_argument("--bed-height-mm", type=float, default=0.0, help="Bed height translation in W0.")
    parser.add_argument("--bed-tilt-deg", type=float, default=0.0, help="Single-axis bed tilt.")
    parser.add_argument("--bed-from-heart-translation-mm", nargs=3, type=float, default=(20.0, -10.0, 120.0), help="Translation of H in bed frame B.")
    parser.add_argument(
        "--bed-from-heart-rpy-deg",
        nargs=3,
        type=float,
        default=(12.092074922655693, 5.445368236416436, 39.543906052732424),
        help="Roll/pitch/yaw of H in bed frame B.",
    )
    parser.add_argument("--coronary-frame-rpy-deg", nargs=3, type=float, default=(15.0, -10.0, 25.0), help="Residual local coronary rotation in H.")
    parser.add_argument("--alpha-lao-rao-deg", type=float, default=40.0, help="C-arm alpha angle in LAO/RAO convention.")
    parser.add_argument("--beta-cra-cau-deg", type=float, default=15.0, help="C-arm beta angle in CRA/CAU convention.")
    parser.add_argument("--source-to-detector-mm", type=float, default=1000.0, help="Source to detector distance.")
    parser.add_argument("--source-to-isocenter-mm", type=float, default=765.0, help="Source to isocenter distance.")
    parser.add_argument("--detector-width-px", type=int, default=1400, help="Detector width in pixels.")
    parser.add_argument("--detector-height-px", type=int, default=1200, help="Detector height in pixels.")
    parser.add_argument("--pixel-spacing-mm", type=float, default=0.55, help="Detector pixel spacing in mm.")
    args = parser.parse_args()

    config = CoordinateFramesFigureConfig(
        output_path=_resolve(args.output),
        point_in_heart_mm=tuple(float(v) for v in args.point_heart_mm),
        heart_state=HeartState(
            bed_from_heart=Pose3D(
                translation_mm=tuple(float(v) for v in args.bed_from_heart_translation_mm),
                rpy_deg=tuple(float(v) for v in args.bed_from_heart_rpy_deg),
            ),
            bed_config=BedConfig(
                longitudinal_mm=float(args.bed_longitudinal_mm),
                lateral_mm=float(args.bed_lateral_mm),
                height_mm=float(args.bed_height_mm),
                tilt_deg=float(args.bed_tilt_deg),
            ),
            coronary_frame_rpy_deg=tuple(float(v) for v in args.coronary_frame_rpy_deg),
            ecg_phase=0.35,
        ),
        c_arm=CArmConfig(
            alpha_lao_rao_deg=float(args.alpha_lao_rao_deg),
            beta_cra_cau_deg=float(args.beta_cra_cau_deg),
            source_to_detector_mm=float(args.source_to_detector_mm),
            source_to_isocenter_mm=float(args.source_to_isocenter_mm),
            detector=DetectorConfig(
                width_px=int(args.detector_width_px),
                height_px=int(args.detector_height_px),
                pixel_spacing_mm=float(args.pixel_spacing_mm),
            ),
        ),
    )
    path = build_coordinate_frames_figure(config)
    print(f"png: {path}")


if __name__ == "__main__":
    main()
