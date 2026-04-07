"""Research modules for 3D-2D coronary reconstruction and projection."""

from .contracts import (
    BranchLatentState,
    CArmConfig,
    DetectorConfig,
    HeartState,
    Pose3D,
    ProjectedBranch2D,
    SyntheticProjectionResult,
)
from .synthetic_projection import (
    project_case_centerlines,
    render_projection_preview,
    render_projection_wall_preview,
    save_projection_result_json,
)

__all__ = [
    "BranchLatentState",
    "CArmConfig",
    "DetectorConfig",
    "HeartState",
    "Pose3D",
    "ProjectedBranch2D",
    "SyntheticProjectionResult",
    "project_case_centerlines",
    "render_projection_preview",
    "render_projection_wall_preview",
    "save_projection_result_json",
]
