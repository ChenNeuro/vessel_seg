"""Lightweight visualization helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .graph_structure import Branch, CoronaryTree


def show_slice_with_mask(volume: np.ndarray, mask: np.ndarray, axis: int = 2, index: Optional[int] = None) -> plt.Axes:
    """Overlay mask on a slice for quick inspection."""
    if index is None:
        index = volume.shape[axis] // 2
    vol_slice = np.take(volume, index, axis=axis)
    mask_slice = np.take(mask, index, axis=axis)
    _, ax = plt.subplots()
    ax.imshow(vol_slice.T, cmap="gray", origin="lower")
    ax.imshow(mask_slice.T, cmap="jet", alpha=0.4, origin="lower")
    ax.set_title(f"Slice {index} (axis {axis})")
    ax.axis("off")
    return ax


def plot_centerlines_2d(branches: Iterable[Branch], plane: str = "axial", ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Project centerlines to 2D plane."""
    if ax is None:
        _, ax = plt.subplots()
    axis_map = {"axial": (0, 1), "sagittal": (1, 2), "coronal": (0, 2)}
    axes_idx = axis_map.get(plane.lower())
    if axes_idx is None:
        raise ValueError(f"Unknown plane {plane}")
    for br in branches:
        pts = br.centerline
        ax.plot(pts[:, axes_idx[0]], pts[:, axes_idx[1]], linewidth=1.0, alpha=0.8)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title(f"Centerlines ({plane})")
    ax.axis("equal")
    return ax


def plot_centerlines_3d(tree: CoronaryTree, show_radii: bool = False) -> plt.Axes:
    """Basic 3D line plot of centerlines."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for br in tree.iter_branches():
        pts = br.centerline
        if show_radii and br.radii is not None:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.0, alpha=0.8, label=f"{br.id}")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=br.radii, cmap="viridis", s=4)
        else:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.0, alpha=0.8)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Centerlines (3D)")
    return ax
