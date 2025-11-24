"""Preprocessing utilities for volumes and masks."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def resample_isotropic(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    new_spacing: float,
    order: int = 1,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Resample a volume to isotropic spacing using scipy.ndimage.zoom."""
    spacing = tuple(float(s) for s in spacing)
    factors = tuple(s / float(new_spacing) for s in spacing)
    resampled = zoom(volume, factors, order=order)
    return resampled, (new_spacing, new_spacing, new_spacing)


def crop_to_mask(volume: np.ndarray, mask: np.ndarray, margin: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Crop volume and mask around mask bounding box with optional voxel margin."""
    if volume.shape != mask.shape:
        raise ValueError("Volume and mask must have the same shape for cropping.")
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return volume, mask
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, volume.shape)
    slices = tuple(slice(lo, hi) for lo, hi in zip(mins, maxs))
    return volume[slices], mask[slices]
