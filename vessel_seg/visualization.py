"""Lightweight visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

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


def select_mask_slices(
    mask: np.ndarray,
    axis: int = 2,
    num_slices: int = 6,
) -> list[int]:
    """Select representative slices that contain foreground."""
    if axis not in (0, 1, 2):
        raise ValueError(f"Unsupported axis: {axis}")

    axes = tuple(idx for idx in range(mask.ndim) if idx != axis)
    slice_area = np.asarray(mask.sum(axis=axes), dtype=np.int64)
    foreground = np.flatnonzero(slice_area > 0)
    if foreground.size == 0:
        return [mask.shape[axis] // 2]

    if foreground.size <= num_slices:
        return foreground.astype(int).tolist()

    positions = np.linspace(0, foreground.size - 1, num_slices)
    picked = sorted({int(foreground[int(round(pos))]) for pos in positions})
    return picked


def _window_ct_slice(
    ct_slice: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    low, high = window
    clipped = np.clip(np.asarray(ct_slice, dtype=np.float32), low, high)
    denom = max(high - low, 1e-6)
    return (clipped - low) / denom


def plot_segmentation_slices(
    ct_volume: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
    *,
    axis: int = 2,
    num_slices: int = 6,
    slice_indices: Sequence[int] | None = None,
    window: tuple[float, float] = (-200.0, 800.0),
    metrics: dict[str, object] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Render CT slices with predicted/GT mask contours."""
    if ct_volume.shape != pred_mask.shape:
        raise ValueError(f"CT/pred shape mismatch: {ct_volume.shape} vs {pred_mask.shape}")
    if gt_mask is not None and gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Pred/gt shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")

    if slice_indices is None:
        union_mask = pred_mask if gt_mask is None else np.logical_or(pred_mask, gt_mask)
        slice_indices = select_mask_slices(union_mask, axis=axis, num_slices=num_slices)
    else:
        slice_indices = [int(idx) for idx in slice_indices]

    ncols = min(3, max(1, len(slice_indices)))
    nrows = int(np.ceil(len(slice_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.6 * nrows), dpi=180)
    axes_array = np.atleast_1d(axes).ravel()

    for ax, slice_idx in zip(axes_array, slice_indices):
        ct_slice = np.take(ct_volume, slice_idx, axis=axis)
        pred_slice = np.take(pred_mask, slice_idx, axis=axis).astype(bool)
        gt_slice = None if gt_mask is None else np.take(gt_mask, slice_idx, axis=axis).astype(bool)

        ax.imshow(_window_ct_slice(ct_slice, window).T, cmap="gray", origin="lower")
        ax.contour(pred_slice.T.astype(float), levels=[0.5], colors=["crimson"], linewidths=1.4)
        if gt_slice is not None and np.any(gt_slice):
            ax.contour(gt_slice.T.astype(float), levels=[0.5], colors=["lime"], linewidths=1.2)
            error_slice = np.logical_xor(pred_slice, gt_slice)
            if np.any(error_slice):
                overlay = np.zeros(error_slice.T.shape + (4,), dtype=np.float32)
                overlay[..., 0] = 1.0
                overlay[..., 1] = 0.75
                overlay[..., 3] = error_slice.T.astype(np.float32) * 0.22
                ax.imshow(overlay, origin="lower")
        elif np.any(pred_slice):
            overlay = np.zeros(pred_slice.T.shape + (4,), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 3] = pred_slice.T.astype(np.float32) * 0.16
            ax.imshow(overlay, origin="lower")

        ax.set_title(f"slice={slice_idx}")
        ax.axis("off")

    for ax in axes_array[len(slice_indices):]:
        ax.axis("off")

    title_lines = [title] if title else ["Segmentation Overview"]
    if metrics:
        summary_parts = []
        for key in ("dice", "hd95_mm", "cldice", "pred_mask_components_26", "pred_skeleton_components_26"):
            value = metrics.get(key)
            if value is None:
                continue
            if isinstance(value, float):
                summary_parts.append(f"{key}={value:.4f}")
            else:
                summary_parts.append(f"{key}={value}")
        if summary_parts:
            title_lines.append(" | ".join(summary_parts))
    fig.suptitle("\n".join(title_lines), fontsize=12)
    fig.tight_layout()
    return fig


def save_segmentation_slices(
    ct_volume: np.ndarray,
    pred_mask: np.ndarray,
    out_path: str | Path,
    gt_mask: np.ndarray | None = None,
    *,
    axis: int = 2,
    num_slices: int = 6,
    slice_indices: Sequence[int] | None = None,
    window: tuple[float, float] = (-200.0, 800.0),
    metrics: dict[str, object] | None = None,
    title: str | None = None,
) -> Path:
    """Save segmentation slice visualization to disk."""
    figure = plot_segmentation_slices(
        ct_volume,
        pred_mask,
        gt_mask,
        axis=axis,
        num_slices=num_slices,
        slice_indices=slice_indices,
        window=window,
        metrics=metrics,
        title=title,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return out_path
