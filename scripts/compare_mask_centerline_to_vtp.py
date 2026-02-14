#!/usr/bin/env python3
"""Compare skeletonized mask centerline (mask->skeleton) to GT VTP centerlines.

This avoids writing VTP for predicted centerlines; useful when VTK export is unstable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize  # 2D fallback

try:
    import vtk  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("vtk is required to read VTP centerlines.") from exc


def read_vtp_points(vtp_path: Path) -> np.ndarray:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    return np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)


def skeleton_points_from_mask(mask_path: Path) -> np.ndarray:
    img = nib.load(str(mask_path))
    mask = img.get_fdata() > 0.5
    affine = img.affine
    # 2D skeleton per slice (z)
    skel_slices = [skeletonize(mask[:, :, z]) for z in range(mask.shape[2])]
    skel = np.stack(skel_slices, axis=2)
    idx = np.argwhere(skel)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=float)
    # voxel -> world
    idx_h = np.c_[idx, np.ones((idx.shape[0], 1))]
    world = (affine @ idx_h.T).T[:, :3]
    return world


def symmetric_distances(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.size == 0 or b.size == 0:
        return np.array([]), np.array([])
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    dist_a, _ = tree_b.query(a, k=1)
    dist_b, _ = tree_a.query(b, k=1)
    return dist_a, dist_b


def summarize(dist: np.ndarray) -> dict:
    if dist.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mask-skeleton centerline to GT VTP.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask (.nii/.nii.gz).")
    parser.add_argument("--gt", type=Path, required=True, help="GT VTP centerlines.")
    parser.add_argument("--thr", type=float, default=1.0, help="Coverage threshold (mm).")
    args = parser.parse_args()

    pred_pts = skeleton_points_from_mask(args.mask)
    gt_pts = read_vtp_points(args.gt)
    dist_pred, dist_gt = symmetric_distances(pred_pts, gt_pts)

    coverage_pred = float((dist_pred <= args.thr).mean()) if dist_pred.size else None
    coverage_gt = float((dist_gt <= args.thr).mean()) if dist_gt.size else None

    print("pred(mask-skel)->gt:", summarize(dist_pred))
    print("gt->pred(mask-skel):", summarize(dist_gt))
    print(f"coverage_pred@{args.thr}mm:", coverage_pred)
    print(f"coverage_gt@{args.thr}mm:", coverage_gt)


if __name__ == "__main__":
    main()
