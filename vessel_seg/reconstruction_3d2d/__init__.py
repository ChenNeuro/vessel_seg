"""Research modules for 3D-2D coronary reconstruction and projection."""

from .contracts import (
    BedConfig,
    BranchLatentState,
    CArmConfig,
    DetectorConfig,
    HeartAnatomicalLandmarks,
    HeartState,
    Pose3D,
    ProjectedBranch2D,
    SyntheticProjectionResult,
)
from .heart_anatomy import (
    build_bed_from_heart_pose,
    build_heart_frame_from_landmarks,
    rotation_matrix_to_rpy_deg,
)
from .synthetic_projection import (
    project_case_centerlines,
    render_projection_preview,
    render_projection_wall_preview,
    save_projection_result_json,
)
from .coordinate_frames_figure import (
    CoordinateFramesFigureConfig,
    build_coordinate_frames_figure,
)
from .teacher_presentation import (
    TeacherModelingPptConfig,
    build_teacher_modeling_notes,
    build_teacher_modeling_ppt,
)

__all__ = [
    "BedConfig",
    "BranchLatentState",
    "CArmConfig",
    "DetectorConfig",
    "HeartAnatomicalLandmarks",
    "HeartState",
    "Pose3D",
    "ProjectedBranch2D",
    "SyntheticProjectionResult",
    "build_bed_from_heart_pose",
    "build_heart_frame_from_landmarks",
    "rotation_matrix_to_rpy_deg",
    "project_case_centerlines",
    "render_projection_preview",
    "render_projection_wall_preview",
    "save_projection_result_json",
    "CoordinateFramesFigureConfig",
    "build_coordinate_frames_figure",
    "TeacherModelingPptConfig",
    "build_teacher_modeling_notes",
    "build_teacher_modeling_ppt",
]
