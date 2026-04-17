"""Anatomical heart-frame construction helpers."""

from __future__ import annotations

import math

import numpy as np

from .contracts import HeartAnatomicalLandmarks, Pose3D


def _as_array(point_mm: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(point_mm, dtype=np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Landmark-derived direction has near-zero length.")
    return vector / norm


def _origin_from_mode(landmarks: HeartAnatomicalLandmarks, origin_mode: str) -> np.ndarray:
    if origin_mode == "lm_ostium":
        return _as_array(landmarks.lm_ostium_mm)
    if origin_mode == "base_center":
        return _as_array(landmarks.base_center_mm)
    raise ValueError(f"Unsupported origin_mode: {origin_mode}")


def build_heart_frame_from_landmarks(
    landmarks: HeartAnatomicalLandmarks,
    origin_mode: str = "lm_ostium",
) -> np.ndarray:
    """Build a homogeneous transform whose columns are the anatomical H-frame axes."""
    origin = _origin_from_mode(landmarks, origin_mode)
    apex = _as_array(landmarks.apex_mm)
    base_center = _as_array(landmarks.base_center_mm)
    lm_ostium = _as_array(landmarks.lm_ostium_mm)
    lm_bifurcation = _as_array(landmarks.lm_bifurcation_mm)

    z_axis = _normalize(apex - base_center)
    coronary_direction = lm_bifurcation - lm_ostium
    x_unprojected = coronary_direction - float(coronary_direction @ z_axis) * z_axis
    x_axis = _normalize(x_unprojected)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    return transform


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    """Convert a rotation matrix into roll / pitch / yaw degrees for ZYX order."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {rotation.shape}.")

    sy = -float(rotation[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def build_bed_from_heart_pose(
    landmarks: HeartAnatomicalLandmarks,
    origin_mode: str = "lm_ostium",
) -> Pose3D:
    """Convert the landmark-derived heart frame into a Pose3D contract."""
    transform = build_heart_frame_from_landmarks(landmarks, origin_mode=origin_mode)
    return Pose3D(
        translation_mm=tuple(float(value) for value in transform[:3, 3]),
        rpy_deg=rotation_matrix_to_rpy_deg(transform[:3, :3]),
    )
