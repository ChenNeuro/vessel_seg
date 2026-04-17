"""Minimal C-arm geometry and projection helpers."""

from __future__ import annotations

import math

import numpy as np

from .contracts import BedConfig, CArmConfig, HeartState, Pose3D


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix_from_rpy_deg(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    """Build a rotation matrix with Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (math.radians(float(value)) for value in rpy_deg)
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def pose_matrix_mm(pose: Pose3D) -> np.ndarray:
    """Convert pose dataclass into a homogeneous transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix_from_rpy_deg(pose.rpy_deg)
    matrix[:3, 3] = np.asarray(pose.translation_mm, dtype=np.float64)
    return matrix


def bed_pose_matrix_mm(bed_config: BedConfig) -> np.ndarray:
    """Build ^W0 T_B from the simplified bed parameters."""
    return pose_matrix_mm(
        Pose3D(
            translation_mm=(
                float(bed_config.longitudinal_mm),
                float(bed_config.lateral_mm),
                float(bed_config.height_mm),
            ),
            rpy_deg=(0.0, float(bed_config.tilt_deg), 0.0),
        )
    )


def transform_points_mm(points_mm: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to a point cloud."""
    if points_mm.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    homogeneous = np.concatenate(
        [points_mm.astype(np.float64, copy=False), np.ones((points_mm.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3]


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def carm_rotation_in_world(config: CArmConfig) -> np.ndarray:
    """Approximate the C-arm mechanical frame rotation expressed in W0."""
    alpha = math.radians(float(config.alpha_lao_rao_deg))
    beta = math.radians(float(config.beta_cra_cau_deg))
    return _rotation_x(beta) @ _rotation_z(alpha)


def observer_from_carm_transform(config: CArmConfig) -> np.ndarray:
    """Build ^O T_C with O at the X-ray source and z_O pointing toward the detector."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray([0.0, 0.0, float(config.source_to_isocenter_mm)], dtype=np.float64)
    return transform


def world_from_carm_transform(config: CArmConfig) -> np.ndarray:
    """Build ^W0 T_C for the current C-arm orientation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = carm_rotation_in_world(config)
    return transform


def observer_from_world_transform(config: CArmConfig) -> np.ndarray:
    """Build ^O T_W0 by chaining ^O T_C and ^C T_W0."""
    return observer_from_carm_transform(config) @ inverse_transform(world_from_carm_transform(config))


def world0_from_heart_transform(heart_state: HeartState) -> np.ndarray:
    """Build ^W0 T_H from bed motion, heart pose in bed, and local coronary residual rotation."""
    world0_from_bed = bed_pose_matrix_mm(heart_state.bed_config)
    bed_from_heart = pose_matrix_mm(heart_state.bed_from_heart)
    coronary_residual = np.eye(4, dtype=np.float64)
    coronary_residual[:3, :3] = rotation_matrix_from_rpy_deg(heart_state.coronary_frame_rpy_deg)
    return world0_from_bed @ bed_from_heart @ coronary_residual


def project_points_world_to_observer(points_world_mm: np.ndarray, config: CArmConfig) -> tuple[np.ndarray, np.ndarray]:
    """Map W0 points into the observer frame O."""
    if points_world_mm.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)
    observer_points = transform_points_mm(points_world_mm, observer_from_world_transform(config))
    valid = observer_points[:, 2] > 1e-6
    return observer_points, valid


def observer_points_to_detector_pixels(points_observer_mm: np.ndarray, config: CArmConfig) -> tuple[np.ndarray, np.ndarray]:
    """Project observer-frame points to detector pixels."""
    if points_observer_mm.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

    points_observer_mm = points_observer_mm.astype(np.float64, copy=False)
    x_obs = points_observer_mm[:, 0]
    y_obs = points_observer_mm[:, 1]
    z_obs = points_observer_mm[:, 2]
    valid = z_obs > 1e-6

    xn = np.zeros_like(x_obs)
    yn = np.zeros_like(y_obs)
    xn[valid] = x_obs[valid] / z_obs[valid]
    yn[valid] = y_obs[valid] / z_obs[valid]
    if config.detector.distortion_coeffs:
        r2 = xn * xn + yn * yn
        scale = np.ones_like(r2)
        for power, coeff in enumerate(config.detector.distortion_coeffs, start=1):
            scale += float(coeff) * np.power(r2, power)
        xn = xn * scale
        yn = yn * scale

    plane_distance_mm = float(config.source_to_detector_mm)
    plane_x_mm = plane_distance_mm * xn
    plane_y_mm = plane_distance_mm * yn
    pixel_spacing_mm = max(float(config.detector.pixel_spacing_mm), 1e-6)
    cx_px, cy_px = config.detector.resolved_principal_point_px()
    u_px = plane_x_mm / pixel_spacing_mm + cx_px
    v_px = cy_px - plane_y_mm / pixel_spacing_mm
    pixels = np.stack([u_px, v_px], axis=1)
    return pixels, valid
