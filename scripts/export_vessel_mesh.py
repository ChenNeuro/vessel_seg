"""
Export a sampled vessel (centerline + radii grid) to a VTP mesh for 3D Slicer.

Input npz is produced by scripts/sample_vessel_from_prior.py and contains:
  centerline (K,3), tk (K,), angles (M,), radii (K,M)

The exporter builds a triangular tube by sweeping each circumferential ring along the centerline
using a parallel-transported frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import vtk


def parallel_transport_frames(centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute tangent and normal frames along centerline using simple parallel transport."""
    pts = np.asarray(centerline, dtype=float)
    K = len(pts)
    tangents = np.zeros((K, 3), dtype=float)
    normals = np.zeros((K, 3), dtype=float)

    # tangents
    diffs = np.diff(pts, axis=0, prepend=pts[:1])
    for i in range(K):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == K - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(t)
        tangents[i] = t / (norm + 1e-8)

    # initial normal
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, tangents[0])) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    n0 = np.cross(tangents[0], ref)
    n0 /= np.linalg.norm(n0) + 1e-8
    normals[0] = n0
    binorm = np.cross(tangents[0], normals[0])

    for i in range(1, K):
        v = tangents[i - 1]
        w = tangents[i]
        axis = np.cross(v, w)
        if np.linalg.norm(axis) < 1e-8:
            normals[i] = normals[i - 1]
            binorm = np.cross(tangents[i], normals[i])
            continue
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(v, w), -1.0, 1.0))
        # Rodrigues rotation
        n_prev = normals[i - 1]
        n_rot = (
            n_prev * np.cos(angle)
            + np.cross(axis, n_prev) * np.sin(angle)
            + axis * np.dot(axis, n_prev) * (1 - np.cos(angle))
        )
        normals[i] = n_rot / (np.linalg.norm(n_rot) + 1e-8)
        binorm = np.cross(tangents[i], normals[i])
    return tangents, normals


def build_mesh(centerline: np.ndarray, radii: np.ndarray, angles: np.ndarray) -> vtk.vtkPolyData:
    K, M = radii.shape
    tangents, normals = parallel_transport_frames(centerline)
    points = vtk.vtkPoints()
    # store mapping (k, m) -> point id
    ids = np.zeros((K, M), dtype=int)
    for k in range(K):
        t = tangents[k]
        n = normals[k]
        b = np.cross(t, n)
        b /= np.linalg.norm(b) + 1e-8
        for j, ang in enumerate(angles):
            u = np.cos(ang) * n + np.sin(ang) * b
            p = centerline[k] + radii[k, j] * u
            pid = points.InsertNextPoint(*p.tolist())
            ids[k, j] = pid

    polys = vtk.vtkCellArray()
    for k in range(K - 1):
        for j in range(M):
            jp = (j + 1) % M
            # quad (k,j)-(k,jp)-(k+1,jp)-(k+1,j)
            v0 = ids[k, j]
            v1 = ids[k, jp]
            v2 = ids[k + 1, jp]
            v3 = ids[k + 1, j]
            # two triangles
            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0, tri[0])
                cell.GetPointIds().SetId(1, tri[1])
                cell.GetPointIds().SetId(2, tri[2])
                polys.InsertNextCell(cell)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    return polydata


def save_vtp(poly: vtk.vtkPolyData, out_path: Path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(poly)
    writer.Write()


def main():
    parser = argparse.ArgumentParser(description="Export sampled vessel npz to VTP mesh.")
    parser.add_argument("--npz", type=Path, required=True, help="npz from sample_vessel_from_prior.py")
    parser.add_argument("--out", type=Path, required=True, help="Output VTP path")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    centerline = data["centerline"]
    radii = data["radii"]
    angles = data["angles"]

    poly = build_mesh(centerline, radii, angles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_vtp(poly, args.out)
    print(f"Saved VTP mesh to {args.out} | points={poly.GetNumberOfPoints()} polys={poly.GetNumberOfPolys()}")


if __name__ == "__main__":
    main()
