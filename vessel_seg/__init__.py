"""Core package for vessel segmentation preprocessing utilities."""

from .conversion import convert_nrrd_to_nii
from .coronary_tree import CoronaryTree, CoronaryTreeNode, SegmentAssignment
from .metadata import SegmentationMetadata, harmonise_label_names, load_metadata, save_metadata
from .shape import (
    Branch,
    BranchProfile,
    compute_polar_profiles,
    export_branch_features,
    extract_branches,
    reconstruct_from_features,
)

__all__ = [
    "SegmentationMetadata",
    "CoronaryTree",
    "CoronaryTreeNode",
    "SegmentAssignment",
    "convert_nrrd_to_nii",
    "harmonise_label_names",
    "load_metadata",
    "save_metadata",
    "Branch",
    "BranchProfile",
    "compute_polar_profiles",
    "export_branch_features",
    "extract_branches",
    "reconstruct_from_features",
]
