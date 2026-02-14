#!/usr/bin/env python3
"""Batch-compare VMTK centerlines vs GT centerlines and write CSV summary."""

from __future__ import annotations

import csv
from pathlib import Path
import argparse

import numpy as np
from scipy.spatial import cKDTree

try:
    import vtk  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("vtk is required to read VTP centerlines.") from exc


def read_vtp_points(path: Path) -> np.ndarray:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    return np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)


def read_vtp_lines(path: Path) -> int:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    return int(poly.GetNumberOfLines())


def symmetric_distances(a: np.ndarray, b: np.ndarray):
    if a.size == 0 or b.size == 0:
        return np.array([]), np.array([])
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    dist_a, _ = tree_b.query(a, k=1)
    dist_b, _ = tree_a.query(b, k=1)
    return dist_a, dist_b


def summarize(dist: np.ndarray):
    if dist.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-compare VTP centerlines.")
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/vmtk_centerlines"))
    parser.add_argument("--gt-dir", type=Path, default=Path("ASOCA2020/Normal/Centerlines"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/vmtk_centerlines/compare_summary.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    out_csv = args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, 21):
        pred = args.pred_dir / f"Normal_{i}_vmtk.vtp"
        gt = args.gt_dir / f"Normal_{i}.vtp"
        pred_pts = read_vtp_points(pred)
        gt_pts = read_vtp_points(gt)
        pred_lines = read_vtp_lines(pred)
        gt_lines = read_vtp_lines(gt)
        dist_pred, dist_gt = symmetric_distances(pred_pts, gt_pts)
        sp = summarize(dist_pred)
        sg = summarize(dist_gt)
        cov_pred = float((dist_pred <= 1.0).mean()) if dist_pred.size else None
        cov_gt = float((dist_gt <= 1.0).mean()) if dist_gt.size else None
        rows.append(
            {
                "case": f"Normal_{i}",
                "pred_pts": int(pred_pts.shape[0]),
                "gt_pts": int(gt_pts.shape[0]),
                "pred_lines": pred_lines,
                "gt_lines": gt_lines,
                "pred2gt_mean": sp["mean"],
                "pred2gt_median": sp["median"],
                "pred2gt_p95": sp["p95"],
                "pred2gt_max": sp["max"],
                "gt2pred_mean": sg["mean"],
                "gt2pred_median": sg["median"],
                "gt2pred_p95": sg["p95"],
                "gt2pred_max": sg["max"],
                "coverage_pred@1mm": cov_pred,
                "coverage_gt@1mm": cov_gt,
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
