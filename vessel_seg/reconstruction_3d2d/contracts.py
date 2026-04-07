"""Dataclass contracts for 3D-2D reconstruction research modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Pose3D:
    """Rigid pose in 3D space."""

    translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectorConfig:
    """Flat detector configuration."""

    width_px: int = 1024
    height_px: int = 1024
    pixel_spacing_mm: float = 0.30
    principal_point_px: tuple[float, float] | None = None
    distortion_coeffs: tuple[float, ...] = ()

    def resolved_principal_point_px(self) -> tuple[float, float]:
        if self.principal_point_px is not None:
            return self.principal_point_px
        return (0.5 * float(self.width_px - 1), 0.5 * float(self.height_px - 1))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["principal_point_px"] = list(self.resolved_principal_point_px())
        return payload


@dataclass(frozen=True)
class CArmConfig:
    """Minimal C-arm geometry used by the synthetic projector."""

    lao_rao_deg: float = 0.0
    cra_cau_deg: float = 0.0
    sid_mm: float = 1200.0
    sod_mm: float = 750.0
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["detector"] = self.detector.to_dict()
        return payload


@dataclass(frozen=True)
class BranchLatentState:
    """Per-side latent state placeholder for future rigid/soft updates."""

    side_group: str
    joint_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    soft_coeffs: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeartState:
    """State variables in heart space."""

    world_pose: Pose3D = field(default_factory=Pose3D)
    ecg_phase: float = 0.0
    cav_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_state: BranchLatentState = field(default_factory=lambda: BranchLatentState(side_group="LCA"))
    right_state: BranchLatentState = field(default_factory=lambda: BranchLatentState(side_group="RCA"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_pose": self.world_pose.to_dict(),
            "ecg_phase": float(self.ecg_phase),
            "cav_rpy_deg": list(self.cav_rpy_deg),
            "left_state": self.left_state.to_dict(),
            "right_state": self.right_state.to_dict(),
        }


@dataclass(frozen=True)
class ProjectedBranch2D:
    """Projected 2D centerline for one branch."""

    branch_id: int
    canonical_name: str
    semantic_name: str | None
    side_group: str
    depth: int
    visible: bool
    points_px: tuple[tuple[float, float], ...]
    radii_px: tuple[float, ...] = ()
    mean_radius_mm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": int(self.branch_id),
            "canonical_name": self.canonical_name,
            "semantic_name": self.semantic_name,
            "side_group": self.side_group,
            "depth": int(self.depth),
            "visible": bool(self.visible),
            "points_px": [list(point) for point in self.points_px],
            "radii_px": [float(value) for value in self.radii_px],
            "mean_radius_mm": None if self.mean_radius_mm is None else float(self.mean_radius_mm),
        }


@dataclass(frozen=True)
class SyntheticProjectionResult:
    """Synthetic X-ray style observation bundle."""

    case_id: str
    tree_json_path: Path
    centerline_vtp_path: Path
    c_arm: CArmConfig
    heart_state: HeartState
    projected_branches: tuple[ProjectedBranch2D, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tree_json_path": str(self.tree_json_path),
            "centerline_vtp_path": str(self.centerline_vtp_path),
            "c_arm": self.c_arm.to_dict(),
            "heart_state": self.heart_state.to_dict(),
            "projected_branches": [branch.to_dict() for branch in self.projected_branches],
            "metadata": self.metadata,
        }
