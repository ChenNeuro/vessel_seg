"""Centerline extraction and repair interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import ndimage

from .graph_structure import Branch, CoronaryTree


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Simple skeletonization wrapper with fallbacks."""
    try:
        from skimage.morphology import skeletonize_3d

        return skeletonize_3d(mask > 0)
    except Exception:
        try:
            from skimage.morphology import skeletonize
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "scikit-image with skeletonize or skeletonize_3d is required. Install/upgrade `scikit-image`."
            ) from exc
        # Fallback: skeletonize each slice (2D) along the last axis
        vol = mask > 0
        slices = [skeletonize(vol[:, :, i]) for i in range(vol.shape[2])]
        return np.stack(slices, axis=2)


def _iter_neighbors(coord: Tuple[int, int, int], shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    x, y, z = coord
    nbrs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                    nbrs.append((nx, ny, nz))
    return nbrs


def _degree_map(skel: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    coords = np.argwhere(skel > 0)
    shape = skel.shape
    deg: Dict[Tuple[int, int, int], int] = {}
    for c in coords:
        coord = tuple(int(v) for v in c)
        nbrs = _iter_neighbors(coord, shape)
        deg[coord] = sum(1 for n in nbrs if skel[n] > 0)
    return deg


def _trace_branches(skel: np.ndarray, spacing: Tuple[float, float, float]) -> List[Branch]:
    """Trace skeleton into polyline branches split at junctions/endpoints."""
    deg = _degree_map(skel)
    shape = skel.shape
    visited = set()
    branches: List[Branch] = []
    branch_id = 0

    critical = {c for c, d in deg.items() if d != 2}
    for start in critical:
        for nbr in _iter_neighbors(start, shape):
            if skel[nbr] == 0:
                continue
            edge_key = tuple(sorted([start, nbr]))
            if edge_key in visited:
                continue
            coords_voxel = [start]
            current = nbr
            prev = start
            visited.add(edge_key)
            while True:
                coords_voxel.append(current)
                nbrs = [n for n in _iter_neighbors(current, shape) if skel[n] > 0 and n != prev]
                degree = deg.get(current, 0)
                if degree != 2 or not nbrs:
                    break
                next_node = nbrs[0]
                prev, current = current, next_node
                visited.add(tuple(sorted([prev, current])))

            coords_world = np.asarray(coords_voxel, dtype=np.float32) * np.asarray(spacing, dtype=np.float32)
            branch = Branch(
                id=branch_id,
                centerline=coords_world,
                parent_id=None,
                child_ids=[],
                start_coord=coords_voxel[0],
                end_coord=coords_voxel[-1],
            )
            branches.append(branch)
            branch_id += 1
    return branches


def _estimate_radii(mask: np.ndarray, spacing: Tuple[float, float, float], branches: List[Branch]) -> None:
    """Attach rough radius estimates using distance transform."""
    dt = ndimage.distance_transform_edt(mask > 0, sampling=spacing)
    for br in branches:
        radii = []
        for pt_world in br.centerline:
            vox = np.round(pt_world / np.asarray(spacing)).astype(int)
            if np.any(vox < 0) or np.any(vox >= dt.shape):
                radii.append(0.0)
                continue
            radii.append(float(dt[tuple(vox.tolist())]))
        br.radii = np.asarray(radii, dtype=np.float32)


def _link_branches(branches: List[Branch]) -> None:
    """Populate child relationships based on shared endpoints (undirected adjacency for now)."""
    endpoints: Dict[Tuple[int, int, int], List[int]] = {}
    for br in branches:
        endpoints.setdefault(br.start_coord, []).append(br.id)
        endpoints.setdefault(br.end_coord, []).append(br.id)
    for br in branches:
        neighbors = set()
        for coord in (br.start_coord, br.end_coord):
            for other in endpoints.get(coord, []):
                if other != br.id:
                    neighbors.add(other)
        br.child_ids = sorted(neighbors)


def extract_centerlines_from_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
) -> CoronaryTree:
    """Extract a CoronaryTree from a binary mask using 3D skeletonization."""
    skeleton = _skeletonize(mask.astype(np.uint8))
    branches = _trace_branches(skeleton, spacing)
    _estimate_radii(mask, spacing, branches)
    _link_branches(branches)
    tree = CoronaryTree()
    for br in branches:
        tree.add_branch(br)
    return tree


def extract_centerlines_from_vtp(vtp_path: str | Path, tolerance: float = 1e-3) -> CoronaryTree:
    """Load VMTK-generated centerlines (.vtp) and convert to CoronaryTree."""
    try:
        import vtk
    except Exception as exc:  # pragma: no cover
        raise ImportError("vtk is required to read VTP centerlines. Install vtk (e.g., via conda-forge).") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    lines = poly.GetLines()
    points = poly.GetPoints()

    branches: List[Branch] = []
    id_list = vtk.vtkIdList()
    branch_id = 0
    lines.InitTraversal()
    while lines.GetNextCell(id_list):
        pts = []
        for i in range(id_list.GetNumberOfIds()):
            pid = id_list.GetId(i)
            pts.append(points.GetPoint(pid))
        if len(pts) < 2:
            continue
        centerline = np.asarray(pts, dtype=np.float32)
        br = Branch(id=branch_id, centerline=centerline)
        branches.append(br)
        branch_id += 1

    _link_branches_vtp(branches, tolerance=tolerance)
    tree = CoronaryTree()
    for br in branches:
        tree.add_branch(br)
    return tree


def _link_branches_vtp(branches: List[Branch], tolerance: float = 1e-3) -> None:
    """Populate child_ids by matching endpoints within tolerance."""
    def key(pt: np.ndarray) -> Tuple[int, int, int]:
        return tuple(int(round(x / tolerance)) for x in pt)

    endpoints: Dict[Tuple[int, int, int], List[int]] = {}
    for br in branches:
        start = key(br.centerline[0])
        end = key(br.centerline[-1])
        endpoints.setdefault(start, []).append(br.id)
        endpoints.setdefault(end, []).append(br.id)
    for br in branches:
        neighbors = set()
        for pt in (br.centerline[0], br.centerline[-1]):
            for other in endpoints.get(key(pt), []):
                if other != br.id:
                    neighbors.add(other)
        br.child_ids = sorted(neighbors)
