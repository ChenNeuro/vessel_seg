"""I/O helpers for quantitative pipeline stages."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    import vtk  # type: ignore
except Exception:  # pragma: no cover
    vtk = None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_binary_mask(path: Path, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    image = nib.load(str(path))
    data = image.get_fdata()
    mask = data > threshold
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return mask.astype(bool), image.affine, spacing


def save_binary_mask(path: Path, mask: np.ndarray, affine: np.ndarray) -> None:
    ensure_parent(path)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def read_vtp_points(vtp_path: Path) -> np.ndarray:
    if vtk is None:
        raise ImportError("vtk is required to read VTP centerlines.")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    return np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)


def read_mesh_points(mesh_path: Path) -> np.ndarray:
    if vtk is None:
        raise ImportError("vtk is required to read mesh files.")

    suffix = mesh_path.suffix.lower()
    if suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif suffix == ".stl":
        reader = vtk.vtkSTLReader()
    elif suffix == ".obj":
        reader = vtk.vtkOBJReader()
    elif suffix == ".ply":
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")

    reader.SetFileName(str(mesh_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    return np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)


def skeleton_points_from_mask(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize_3d

        skel = skeletonize_3d(mask)
    except Exception:
        from skimage.morphology import skeletonize

        slices = [skeletonize(mask[:, :, i]) for i in range(mask.shape[2])]
        skel = np.stack(slices, axis=2)

    idx = np.argwhere(skel)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=float)

    idx_h = np.c_[idx, np.ones((idx.shape[0], 1), dtype=float)]
    world = (affine @ idx_h.T).T[:, :3]
    return world.astype(float)


def boundary_points_from_mask(mask: np.ndarray, affine: np.ndarray, max_points: int = 200000) -> np.ndarray:
    if mask.size == 0:
        return np.zeros((0, 3), dtype=float)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    boundary = mask & (~eroded)

    idx = np.argwhere(boundary)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=float)

    if idx.shape[0] > max_points:
        step = max(1, idx.shape[0] // max_points)
        idx = idx[::step]

    idx_h = np.c_[idx, np.ones((idx.shape[0], 1), dtype=float)]
    world = (affine @ idx_h.T).T[:, :3]
    return world.astype(float)


def load_feature_summary(features_dir: Path) -> dict[str, Any]:
    summary_path = features_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_global_descriptor(features_dir: Path, summary: dict[str, Any]) -> np.ndarray:
    rel = summary.get("global_descriptor", "global_descriptor.npy")
    path = features_dir / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing global descriptor: {path}")
    return np.asarray(np.load(path), dtype=float)


def to_float_dict(metrics: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in fields:
        value = metrics.get(key)
        if value is None:
            row[key] = None
        elif isinstance(value, (int, float, np.floating)):
            row[key] = float(value)
        else:
            row[key] = value
    return row
