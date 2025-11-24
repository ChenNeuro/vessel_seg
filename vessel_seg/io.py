"""I/O utilities for CT volumes and masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np


@dataclass
class VolumeData:
    """Container for volumetric data and metadata."""

    data: np.ndarray
    affine: np.ndarray
    spacing: Tuple[float, float, float]
    path: Optional[Path] = None


def load_nifti(path: str | Path) -> Tuple[np.ndarray, dict]:
    """Load a NIfTI file and return array + meta dict."""
    path = Path(path)
    image = nib.load(str(path))
    data = image.get_fdata()
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
    meta = {"affine": image.affine, "spacing": spacing, "path": path}
    return data, meta


def save_nifti(array: np.ndarray, meta: dict, path: str | Path) -> None:
    """Save an array to NIfTI format using provided affine."""
    affine = meta.get("affine", np.eye(4))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine), str(path))


def load_pair(volume_path: str | Path, mask_path: str | Path) -> Tuple[VolumeData, VolumeData]:
    """Load intensity volume and mask into VolumeData structures."""
    vol_arr, vol_meta = load_nifti(volume_path)
    mask_arr, mask_meta = load_nifti(mask_path)
    vol = VolumeData(
        data=vol_arr.astype(np.float32),
        affine=vol_meta["affine"],
        spacing=vol_meta["spacing"],
        path=Path(volume_path),
    )
    mask = VolumeData(
        data=(mask_arr > 0.5).astype(np.uint8),
        affine=mask_meta["affine"],
        spacing=mask_meta["spacing"],
        path=Path(mask_path),
    )
    return vol, mask


def load_volume(path: str | Path) -> VolumeData:
    arr, meta = load_nifti(path)
    return VolumeData(data=arr.astype(np.float32), affine=meta["affine"], spacing=meta["spacing"], path=Path(path))


def load_mask(path: str | Path, threshold: float = 0.5) -> VolumeData:
    arr, meta = load_nifti(path)
    mask = (arr > threshold).astype(np.uint8)
    return VolumeData(data=mask, affine=meta["affine"], spacing=meta["spacing"], path=Path(path))
