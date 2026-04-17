"""Visualization helpers for coordinate-frame transforms in 3D-2D reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .carm_geometry import (
    bed_pose_matrix_mm,
    inverse_transform,
    observer_from_world_transform,
    observer_points_to_detector_pixels,
    project_points_world_to_observer,
    pose_matrix_mm,
    transform_points_mm,
    world0_from_heart_transform,
    world_from_carm_transform,
)
from .contracts import BedConfig, CArmConfig, DetectorConfig, HeartAnatomicalLandmarks, HeartState, Pose3D
from .heart_anatomy import build_bed_from_heart_pose


DEFAULT_LANDMARKS = HeartAnatomicalLandmarks(
    apex_mm=(20.0, -20.0, 80.0),
    base_center_mm=(0.0, -10.0, -15.0),
    lm_ostium_mm=(0.0, 0.0, 0.0),
    lm_bifurcation_mm=(35.0, 25.0, 10.0),
    rca_ostium_mm=(-22.0, -18.0, 4.0),
)


def _default_heart_state() -> HeartState:
    anatomical_pose = build_bed_from_heart_pose(DEFAULT_LANDMARKS)
    return HeartState(
        bed_from_heart=Pose3D(
            translation_mm=(20.0, -10.0, 120.0),
            rpy_deg=anatomical_pose.rpy_deg,
        ),
        bed_config=BedConfig(
            longitudinal_mm=0.0,
            lateral_mm=0.0,
            height_mm=0.0,
            tilt_deg=0.0,
        ),
        ecg_phase=0.35,
        coronary_frame_rpy_deg=(15.0, -10.0, 25.0),
        anatomical_landmarks=DEFAULT_LANDMARKS,
    )


def _default_carm_config() -> CArmConfig:
    return CArmConfig(
        alpha_lao_rao_deg=40.0,
        beta_cra_cau_deg=15.0,
        source_to_detector_mm=1000.0,
        source_to_isocenter_mm=765.0,
        detector=DetectorConfig(
            width_px=1400,
            height_px=1200,
            pixel_spacing_mm=0.55,
        ),
    )


@dataclass(frozen=True)
class CoordinateFramesFigureConfig:
    """Configuration for a coordinate-frame transform figure."""

    output_path: Path
    title: str = "Coordinate Frames: W0 / B / H / C / O"
    point_in_heart_mm: tuple[float, float, float] = (20.0, 20.0, 10.0)
    heart_state: HeartState = field(default_factory=_default_heart_state)
    c_arm: CArmConfig = field(default_factory=_default_carm_config)


def _frame_axes_world(transform: np.ndarray, axis_length_mm: float = 45.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    x_axis = origin + axis_length_mm * rotation[:, 0]
    y_axis = origin + axis_length_mm * rotation[:, 1]
    z_axis = origin + axis_length_mm * rotation[:, 2]
    return origin, x_axis, y_axis, z_axis


def _draw_triad(
    ax,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    *,
    label: str,
    label_offset: tuple[float, float, float] = (8.0, 8.0, 8.0),
) -> None:
    ax.quiver(*origin, *(x_axis - origin), color="#d64b4b", linewidth=2.0, arrow_length_ratio=0.12)
    ax.quiver(*origin, *(y_axis - origin), color="#52a35f", linewidth=2.0, arrow_length_ratio=0.12)
    ax.quiver(*origin, *(z_axis - origin), color="#4d84d6", linewidth=2.0, arrow_length_ratio=0.12)
    ax.text(*(origin + np.asarray(label_offset, dtype=np.float64)), label, fontsize=11, weight="bold", color="#222222")
    ax.text(*x_axis, "x", fontsize=9, color="#d64b4b")
    ax.text(*y_axis, "y", fontsize=9, color="#52a35f")
    ax.text(*z_axis, "z", fontsize=9, color="#4d84d6")


def _set_axes_equal(ax) -> None:
    limits = np.asarray([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=np.float64)
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

def _draw_link(ax, start: np.ndarray, end: np.ndarray, label: str, color: str, text_offset: tuple[float, float, float]) -> None:
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        linewidth=1.7,
        linestyle="--",
    )
    midpoint = 0.5 * (start + end) + np.asarray(text_offset, dtype=np.float64)
    ax.text(*midpoint, label, fontsize=9, color=color)


def _landmark_points_in_heart(heart_state: HeartState) -> dict[str, np.ndarray]:
    landmarks = heart_state.anatomical_landmarks
    if landmarks is None:
        return {}
    bed_from_heart = pose_matrix_mm(heart_state.bed_from_heart)
    heart_from_bed = inverse_transform(bed_from_heart)
    landmark_points_bed = {
        "LM ostium": np.asarray(landmarks.lm_ostium_mm, dtype=np.float64),
        "LM bifurcation": np.asarray(landmarks.lm_bifurcation_mm, dtype=np.float64),
        "Base": np.asarray(landmarks.base_center_mm, dtype=np.float64),
        "Apex": np.asarray(landmarks.apex_mm, dtype=np.float64),
    }
    return {
        name: transform_points_mm(point.reshape(1, 3), heart_from_bed)[0]
        for name, point in landmark_points_bed.items()
    }


def build_coordinate_frames_figure(config: CoordinateFramesFigureConfig) -> Path:
    """Generate a figure showing 3D coordinate frames and 2D detector projection."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    point_h = np.asarray(config.point_in_heart_mm, dtype=np.float64).reshape(1, 3)
    world0_from_bed = bed_pose_matrix_mm(config.heart_state.bed_config)
    world0_from_heart = world0_from_heart_transform(config.heart_state)
    world0_from_carm = world_from_carm_transform(config.c_arm)
    world0_from_observer = inverse_transform(observer_from_world_transform(config.c_arm))
    detector_center_o = np.asarray([[0.0, 0.0, float(config.c_arm.source_to_detector_mm)]], dtype=np.float64)
    detector_center_w0 = transform_points_mm(detector_center_o, world0_from_observer)[0]
    source_w0 = world0_from_observer[:3, 3]

    point_w0 = transform_points_mm(point_h, world0_from_heart)[0]
    heart_axes_points_w0 = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [55.0, 0.0, 0.0],
            [0.0, 55.0, 0.0],
            [0.0, 0.0, 55.0],
            config.point_in_heart_mm,
        ],
        dtype=np.float64,
    )
    heart_axes_world0 = transform_points_mm(heart_axes_points_w0, world0_from_heart)
    observer_points, valid_observer = project_points_world_to_observer(heart_axes_world0, config.c_arm)
    pixels, valid_pixels = observer_points_to_detector_pixels(observer_points, config.c_arm)
    valid = valid_observer & valid_pixels
    principal = np.asarray(config.c_arm.detector.resolved_principal_point_px(), dtype=np.float64)
    point_display = point_w0
    landmark_points_h = _landmark_points_in_heart(config.heart_state)
    display_landmarks = {
        name: transform_points_mm(point.reshape(1, 3), world0_from_heart)[0]
        for name, point in landmark_points_h.items()
    }

    world0_origin, world0_x, world0_y, world0_z = _frame_axes_world(np.eye(4, dtype=np.float64), axis_length_mm=90.0)
    bed_origin, bed_x, bed_y, bed_z = _frame_axes_world(world0_from_bed, axis_length_mm=78.0)
    heart_origin, heart_x, heart_y, heart_z = _frame_axes_world(world0_from_heart, axis_length_mm=78.0)
    carm_origin, carm_x, carm_y, carm_z = _frame_axes_world(world0_from_carm, axis_length_mm=62.0)
    observer_origin, observer_x, observer_y, observer_z = _frame_axes_world(world0_from_observer, axis_length_mm=62.0)

    fig = plt.figure(figsize=(14.2, 7.2), dpi=180, constrained_layout=False)
    fig.patch.set_facecolor("#f7f7f3")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.12)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.set_facecolor("#fcfcfa")
    ax3d.set_proj_type("ortho")
    _draw_triad(ax3d, world0_origin, world0_x, world0_y, world0_z, label="W0", label_offset=(-38.0, -28.0, -24.0))
    _draw_triad(ax3d, bed_origin, bed_x, bed_y, bed_z, label="B", label_offset=(16.0, -14.0, -28.0))
    _draw_triad(ax3d, heart_origin, heart_x, heart_y, heart_z, label="H", label_offset=(16.0, 12.0, 18.0))
    _draw_triad(ax3d, carm_origin, carm_x, carm_y, carm_z, label="C", label_offset=(-38.0, 18.0, 18.0))
    _draw_triad(ax3d, observer_origin, observer_x, observer_y, observer_z, label="O", label_offset=(16.0, 16.0, 18.0))

    ax3d.scatter(*point_display, s=60, color="#7c3aed")
    ax3d.text(*(point_display + np.asarray([10.0, 10.0, 10.0], dtype=np.float64)), "p_H -> p_W0", fontsize=10, color="#7c3aed")
    for name, point in display_landmarks.items():
        ax3d.scatter(*point, s=24, color="#c48b2c")
        ax3d.text(*(point + np.asarray([5.0, 5.0, 5.0], dtype=np.float64)), name, fontsize=8, color="#8a631c")
    ax3d.scatter(*source_w0, s=36, color="#111111")
    ax3d.text(*(source_w0 + np.asarray([8.0, 8.0, 8.0], dtype=np.float64)), "X-ray source", fontsize=8.5, color="#111111")
    ax3d.scatter(*detector_center_w0, s=36, color="#6b7280")
    ax3d.text(
        *(detector_center_w0 + np.asarray([8.0, 8.0, 8.0], dtype=np.float64)),
        "detector center",
        fontsize=8.5,
        color="#4b5563",
    )

    _draw_link(ax3d, world0_origin, bed_origin, "^W0 T_B", "#aa3c3c", (10.0, -14.0, 10.0))
    _draw_link(ax3d, bed_origin, heart_origin, "^B T_H", "#4b7f3c", (10.0, -8.0, 12.0))
    if float(np.linalg.norm(world0_origin - carm_origin)) > 1e-6:
        _draw_link(ax3d, world0_origin, carm_origin, "^C T_W0", "#2f6ea6", (8.0, 10.0, 10.0))
    else:
        ax3d.text(*(carm_origin + np.asarray([-78.0, 20.0, 34.0], dtype=np.float64)), "^C T_W0: origin at isocenter", fontsize=8.8, color="#2f6ea6")
    _draw_link(ax3d, carm_origin, observer_origin, "^O T_C", "#555555", (10.0, 10.0, 8.0))
    ax3d.plot(
        [observer_origin[0], carm_origin[0]],
        [observer_origin[1], carm_origin[1]],
        [observer_origin[2], carm_origin[2]],
        color="#555555",
        linewidth=1.2,
    )
    ax3d.plot(
        [source_w0[0], detector_center_w0[0]],
        [source_w0[1], detector_center_w0[1]],
        [source_w0[2], detector_center_w0[2]],
        color="#6b7280",
        linewidth=1.4,
        linestyle=":",
    )

    all_points = np.vstack(
        [
            world0_origin,
            world0_x,
            world0_y,
            world0_z,
            bed_origin,
            bed_x,
            bed_y,
            bed_z,
            heart_origin,
            heart_x,
            heart_y,
            heart_z,
            carm_origin,
            carm_x,
            carm_y,
            carm_z,
            observer_origin,
            observer_x,
            observer_y,
            observer_z,
            source_w0,
            detector_center_w0,
            point_display,
            *display_landmarks.values(),
        ]
    )
    mins = all_points.min(axis=0) - 60.0
    maxs = all_points.max(axis=0) + 60.0
    ax3d.set_xlim(mins[0], maxs[0])
    ax3d.set_ylim(mins[1], maxs[1])
    ax3d.set_zlim(mins[2], maxs[2])
    _set_axes_equal(ax3d)
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.view_init(elev=20, azim=-38)
    ax3d.set_title("Physical 3D View: W0 / B / H / C / O", fontsize=13, pad=10)

    ax2d = fig.add_subplot(gs[0, 1])
    ax2d.set_facecolor("#fcfcfa")
    width_px = config.c_arm.detector.width_px
    height_px = config.c_arm.detector.height_px
    rect = Rectangle((0.0, 0.0), width_px, height_px, linewidth=1.5, edgecolor="#777777", facecolor="#ffffff")
    ax2d.add_patch(rect)
    ax2d.scatter(principal[0], principal[1], s=35, color="#222222")
    ax2d.text(principal[0] + 16.0, principal[1] + 10.0, "principal point", fontsize=9, color="#222222")

    p_origin_px = pixels[0]
    p_x_px = pixels[1]
    p_y_px = pixels[2]
    p_z_px = pixels[3]
    p_point_px = pixels[4]
    if bool(valid[4]):
        ax2d.scatter(p_point_px[0], p_point_px[1], s=50, color="#7c3aed")
        ax2d.text(p_point_px[0] + 14.0, p_point_px[1] - 16.0, "pi(p_W0)", fontsize=10, color="#7c3aed")
    if bool(valid[0]) and bool(valid[1]):
        ax2d.annotate("", xy=p_x_px, xytext=p_origin_px, arrowprops=dict(arrowstyle="->", color="#d64b4b", lw=2.0))
        ax2d.text(p_x_px[0] + 10.0, p_x_px[1], "x_H", color="#d64b4b", fontsize=9)
    if bool(valid[0]) and bool(valid[2]):
        ax2d.annotate("", xy=p_y_px, xytext=p_origin_px, arrowprops=dict(arrowstyle="->", color="#52a35f", lw=2.0))
        ax2d.text(p_y_px[0] + 10.0, p_y_px[1], "y_H", color="#52a35f", fontsize=9)
    if bool(valid[0]) and bool(valid[3]):
        ax2d.annotate("", xy=p_z_px, xytext=p_origin_px, arrowprops=dict(arrowstyle="->", color="#4d84d6", lw=2.0))
        ax2d.text(p_z_px[0] + 10.0, p_z_px[1], "z_H", color="#4d84d6", fontsize=9)

    ax2d.text(
        24.0,
        52.0,
        "\n".join(
            [
                "Transform chain",
                "p_H -> ^B T_H -> ^W0 T_B -> ^C T_W0 -> ^O T_C -> p_O",
                "",
                "H-frame landmarks",
                "O_H = LM ostium",
                "z_H || base -> apex",
                "x_H || proj(LM bif - LM ostium, z_H^perp)",
                "y_H = z_H x x_H",
                "",
                (
                    f"bed xyz/tilt = ({config.heart_state.bed_config.longitudinal_mm:.1f}, "
                    f"{config.heart_state.bed_config.lateral_mm:.1f}, "
                    f"{config.heart_state.bed_config.height_mm:.1f}, "
                    f"{config.heart_state.bed_config.tilt_deg:.1f})"
                ),
                f"^B T_H translation = {config.heart_state.bed_from_heart.translation_mm}",
                f"coronary frame rpy = {config.heart_state.coronary_frame_rpy_deg} deg",
                f"alpha = {config.c_arm.alpha_lao_rao_deg:.1f} deg",
                f"beta = {config.c_arm.beta_cra_cau_deg:.1f} deg",
                f"d_SD = {config.c_arm.source_to_detector_mm:.1f} mm",
                f"d_SI = {config.c_arm.source_to_isocenter_mm:.1f} mm",
                f"point_H = {tuple(float(v) for v in point_h[0])} mm",
                f"point_W0 = ({point_w0[0]:.1f}, {point_w0[1]:.1f}, {point_w0[2]:.1f}) mm",
                f"source_W0 = ({source_w0[0]:.1f}, {source_w0[1]:.1f}, {source_w0[2]:.1f}) mm",
                f"point_O = ({observer_points[4, 0]:.1f}, {observer_points[4, 1]:.1f}, {observer_points[4, 2]:.1f}) mm"
                if bool(valid[4])
                else "point_O = invalid",
            ]
        ),
        fontsize=8.8,
        color="#2a2a2a",
        va="top",
    )
    ax2d.set_xlim(-20.0, width_px + 20.0)
    ax2d.set_ylim(height_px + 20.0, -20.0)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_title("Observer / Detector Projection", fontsize=13, pad=10)
    ax2d.set_xlabel("u (px)")
    ax2d.set_ylabel("v (px)")
    ax2d.grid(True, color="#e4e7eb", linewidth=0.8, alpha=0.8)

    fig.suptitle(config.title, fontsize=16, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Left: physical-consistent world layout with C at isocenter and O at the X-ray source. Right: detector effect after projecting H-frame axes and a sample point through the observer model.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.87, bottom=0.10, wspace=0.12)
    fig.savefig(config.output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return config.output_path
