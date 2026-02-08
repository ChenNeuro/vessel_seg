"""Core package for vessel segmentation preprocessing utilities."""

from .conversion import convert_nrrd_to_nii
from .coronary_tree import CoronaryTree, CoronaryTreeNode, SegmentAssignment
from .branch_model import BranchShapeModel, SimpleLengthRadiusModel
from .config import ProjectPaths
from .graph_structure import Branch as TreeBranch
from .graph_structure import CoronaryTree as CenterlineTree
from .io import VolumeData, load_mask, load_volume, save_nifti
from .metadata import SegmentationMetadata, harmonise_label_names, load_metadata, save_metadata
from .tree_prior import CoronaryTreePrior

__all__ = [
    "SegmentationMetadata",
    "CoronaryTree",
    "CoronaryTreeNode",
    "SegmentAssignment",
    "convert_nrrd_to_nii",
    "FGPMBranchModel",
    "fourier_coefficients",
    "radii_from_coefficients",
    "gather_training_set",
    "harmonise_label_names",
    "load_metadata",
    "save_metadata",
    "Branch",
    "BranchProfile",
    "CentrelineParams",
    "ProfileParams",
    "ReconstructionParams",
    "compute_polar_profiles",
    "export_branch_features",
    "extract_branches",
    "reconstruct_from_features",
    "BranchShapeModel",
    "SimpleLengthRadiusModel",
    "TreeBranch",
    "CenterlineTree",
    "ProjectPaths",
    "VolumeData",
    "load_volume",
    "load_mask",
    "save_nifti",
    "CoronaryTreePrior",
]

# Optional heavy modules (scipy/networkx/pyvista-dependent) are loaded lazily.
try:  # pragma: no cover - optional runtime dependency compatibility
    from .fgpm import (
        FGPMBranchModel,
        fourier_coefficients,
        gather_training_set,
        radii_from_coefficients,
    )

    __all__.extend(
        [
            "FGPMBranchModel",
            "fourier_coefficients",
            "radii_from_coefficients",
            "gather_training_set",
        ]
    )
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional runtime dependency compatibility
    from .shape import (
        Branch,
        BranchProfile,
        CentrelineParams,
        ProfileParams,
        ReconstructionParams,
        compute_polar_profiles,
        export_branch_features,
        extract_branches,
        reconstruct_from_features,
    )

    __all__.extend(
        [
            "Branch",
            "BranchProfile",
            "CentrelineParams",
            "ProfileParams",
            "ReconstructionParams",
            "compute_polar_profiles",
            "export_branch_features",
            "extract_branches",
            "reconstruct_from_features",
        ]
    )
except Exception:  # pragma: no cover
    pass
