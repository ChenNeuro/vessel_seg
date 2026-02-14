#!/usr/bin/env python3
"""Compare two VTP centerline files via symmetric point distances.

Outputs mean/median/95%/max distances and coverage within a threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

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
    arr = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)
    return arr


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
    parser = argparse.ArgumentParser(description="Compare two VTP centerlines by point distances.")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted/processed VTP.")
    parser.add_argument("--gt", type=Path, required=True, help="GT VTP.")
    parser.add_argument("--thr", type=float, default=1.0, help="Coverage threshold (mm).")
    args = parser.parse_args()

    pred_pts = read_vtp_points(args.pred)
    gt_pts = read_vtp_points(args.gt)
    dist_pred, dist_gt = symmetric_distances(pred_pts, gt_pts)

    coverage_pred = float((dist_pred <= args.thr).mean()) if dist_pred.size else None
    coverage_gt = float((dist_gt <= args.thr).mean()) if dist_gt.size else None

    print("pred->gt:", summarize(dist_pred))
    print("gt->pred:", summarize(dist_gt))
    print(f"coverage_pred@{args.thr}mm:", coverage_pred)
    print(f"coverage_gt@{args.thr}mm:", coverage_gt)


if __name__ == "__main__":
    main()
