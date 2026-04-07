"""Minimal C-arm geometry and projection helpers."""

from __future__ import annotations

import math

import numpy as np

from .contracts import CArmConfig, Pose3D


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
    """Build rotation matrix with Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (math.radians(float(value)) for value in rpy_deg)
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def pose_matrix_mm(pose: Pose3D) -> np.ndarray:
    """Convert pose dataclass into homogeneous transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix_from_rpy_deg(pose.rpy_deg)
    matrix[:3, 3] = np.asarray(pose.translation_mm, dtype=np.float64)
    return matrix


def transform_points_mm(points_mm: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply homogeneous transform to a point cloud."""
    if points_mm.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    homogeneous = np.concatenate(
        [points_mm.astype(np.float64, copy=False), np.ones((points_mm.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3]


def carm_rotation_world(config: CArmConfig) -> np.ndarray:
    """Approximate C-arm rotation from LAO/RAO and CRA/CAU angles."""
    lao_rao = math.radians(float(config.lao_rao_deg))
    cra_cau = math.radians(float(config.cra_cau_deg))
    return _rotation_x(cra_cau) @ _rotation_z(lao_rao)


def project_points_world_to_detector(points_world_mm: np.ndarray, config: CArmConfig) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to detector pixels using a pinhole C-arm model."""
    if points_world_mm.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

    rotation_world = carm_rotation_world(config)
    x_axis_world = rotation_world @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    y_axis_world = rotation_world @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis_world = rotation_world @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    source_world = rotation_world @ np.asarray([0.0, 0.0, -float(config.sod_mm)], dtype=np.float64)

    rel = points_world_mm.astype(np.float64, copy=False) - source_world.reshape(1, 3)
    x_cam = rel @ x_axis_world
    y_cam = rel @ y_axis_world
    z_cam = rel @ z_axis_world
    valid = z_cam > 1e-6

    xn = np.zeros_like(x_cam)
    yn = np.zeros_like(y_cam)
    xn[valid] = x_cam[valid] / z_cam[valid]
    yn[valid] = y_cam[valid] / z_cam[valid]
    if config.detector.distortion_coeffs:
        r2 = xn * xn + yn * yn
        scale = np.ones_like(r2)
        for power, coeff in enumerate(config.detector.distortion_coeffs, start=1):
            scale += float(coeff) * np.power(r2, power)
        xn = xn * scale
        yn = yn * scale

    plane_x_mm = float(config.sid_mm) * xn
    plane_y_mm = float(config.sid_mm) * yn
    pixel_spacing_mm = max(float(config.detector.pixel_spacing_mm), 1e-6)
    cx_px, cy_px = config.detector.resolved_principal_point_px()
    u_px = plane_x_mm / pixel_spacing_mm + cx_px
    v_px = cy_px - plane_y_mm / pixel_spacing_mm
    pixels = np.stack([u_px, v_px], axis=1)
    return pixels, valid
