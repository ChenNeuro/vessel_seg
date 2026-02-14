#!/usr/bin/env python3
"""Extract centerlines from a binary mask using VMTK with non-interactive seeds.

Pipeline:
1) vmtkmarchingcubes -> surface.vtp
2) find open boundary loops (profiles) on the surface
3) choose largest loop as source; all others as targets
4) vmtkcenterlines -seedselector pointlist
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import vtk  # type: ignore
    from vtk.util import numpy_support  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("vtk is required to read VTP surfaces.") from exc


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def marching_cubes(mask_path: Path, surface_path: Path, level: float) -> None:
    run(
        [
            "conda",
            "run",
            "-n",
            "vessel_seg",
            "vmtkmarchingcubes",
            "-ifile",
            str(mask_path),
            "-l",
            str(level),
            "-ofile",
            str(surface_path),
        ]
    )


def decimate_surface(surface_path: Path, output_path: Path, reduction: float) -> None:
    run(
        [
            "conda",
            "run",
            "-n",
            "vessel_seg",
            "vmtksurfacedecimation",
            "-ifile",
            str(surface_path),
            "-reduction",
            str(reduction),
            "-ofile",
            str(output_path),
        ]
    )


def mask_spacing(mask_path: Path) -> Tuple[float, float, float]:
    img = nib.load(str(mask_path))
    affine = img.affine
    spacing = np.abs(np.diag(affine)[:3])
    return float(spacing[0]), float(spacing[1]), float(spacing[2])


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def extract_open_profiles(surface: vtk.vtkPolyData) -> List[np.ndarray]:
    feature = vtk.vtkFeatureEdges()
    feature.SetInputData(surface)
    feature.BoundaryEdgesOn()
    feature.FeatureEdgesOff()
    feature.ManifoldEdgesOff()
    feature.NonManifoldEdgesOff()
    feature.Update()

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(feature.GetOutput())
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()

    n_regions = conn.GetNumberOfExtractedRegions()
    if n_regions == 0:
        return []

    profiles: List[np.ndarray] = []
    for region_id in range(n_regions):
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(conn.GetOutput())
        thresh.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId"
        )
        thresh.SetLowerThreshold(region_id)
        thresh.SetUpperThreshold(region_id)
        thresh.Update()

        geom = vtk.vtkGeometryFilter()
        geom.SetInputData(thresh.GetOutput())
        geom.Update()

        poly = geom.GetOutput()
        pts = poly.GetPoints()
        if pts is None or pts.GetNumberOfPoints() == 0:
            continue
        coords = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)
        profiles.append(coords)
    return profiles


def profile_center_and_size(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    center = coords.mean(axis=0)
    size = float(np.linalg.norm(coords - center, axis=1).mean())
    return center, size


def build_seed_points(profiles: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    centers_sizes = [profile_center_and_size(p) for p in profiles]
    if not centers_sizes:
        raise ValueError("No open profiles found on surface.")
    centers = [c for c, _ in centers_sizes]
    sizes = [s for _, s in centers_sizes]
    source_idx = int(np.argmax(sizes))
    source = centers[source_idx]
    targets = [c for i, c in enumerate(centers) if i != source_idx]
    if not targets:
        raise ValueError("Only one open profile found; cannot define targets.")
    return source, targets


def centerline_endpoints_vtp(centerline_path: Path) -> np.ndarray:
    poly = read_polydata(centerline_path)
    points = poly.GetPoints()
    if points is None:
        return np.zeros((0, 3), dtype=float)
    n_pts = points.GetNumberOfPoints()
    if n_pts == 0:
        return np.zeros((0, 3), dtype=float)

    degrees = np.zeros(n_pts, dtype=int)
    lines = poly.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    while lines.GetNextCell(id_list):
        m = id_list.GetNumberOfIds()
        for idx in range(m - 1):
            u = id_list.GetId(idx)
            v = id_list.GetId(idx + 1)
            degrees[u] += 1
            degrees[v] += 1

    endpoint_ids = np.where(degrees == 1)[0]
    if endpoint_ids.size == 0:
        return np.zeros((0, 3), dtype=float)
    coords = np.array([points.GetPoint(int(i)) for i in endpoint_ids], dtype=float)
    return coords


def project_points_to_surface(surface: vtk.vtkPolyData, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    unique_ids = []
    seen = set()
    for p in points:
        pid = int(locator.FindClosestPoint(p))
        if pid in seen:
            continue
        seen.add(pid)
        unique_ids.append(pid)
    if not unique_ids:
        return np.zeros((0, 3), dtype=float)
    coords = np.array([surface.GetPoint(pid) for pid in unique_ids], dtype=float)
    return coords


def select_source_target(points: np.ndarray, mask_path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    if points.size == 0:
        raise ValueError("No seed points available.")

    img = nib.load(str(mask_path))
    mask = np.asarray(img.dataobj) > 0.5
    if mask.ndim != 3:
        raise ValueError("Mask must be 3D.")

    affine = img.affine
    spacing = np.abs(np.diag(affine)[:3])
    inv_affine = np.linalg.inv(affine)

    dist = ndi.distance_transform_edt(mask, sampling=spacing)
    coords_h = np.c_[points, np.ones(points.shape[0])]
    vox = (inv_affine @ coords_h.T).T[:, :3]
    idx = np.rint(vox).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, dist.shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, dist.shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, dist.shape[2] - 1)
    radii = dist[idx[:, 0], idx[:, 1], idx[:, 2]]
    source_idx = int(np.argmax(radii))
    source = points[source_idx]
    targets = [points[i] for i in range(points.shape[0]) if i != source_idx]
    if not targets:
        raise ValueError("Only one seed point found.")
    return source, targets


def centerlineimage_to_vtp(image_path: Path, output_path: Path) -> int:
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(image_path))
    reader.Update()
    img = reader.GetOutput()
    dims = img.GetDimensions()
    if dims[0] * dims[1] * dims[2] == 0:
        raise ValueError("Empty centerline image.")
    origin = np.array(img.GetOrigin(), dtype=float)
    spacing = np.array(img.GetSpacing(), dtype=float)
    scalars = img.GetPointData().GetScalars()
    if scalars is None:
        raise ValueError("Centerline image has no scalars.")
    arr = numpy_support.vtk_to_numpy(scalars)
    idx = np.nonzero(arr > 0)[0]
    if idx.size == 0:
        raise ValueError("Centerline image is empty after thresholding.")

    nx, ny, nz = dims
    i = idx % nx
    j = (idx // nx) % ny
    k = idx // (nx * ny)
    voxels_raw = np.vstack([i, j, k]).T.astype(np.int32)
    volume = np.zeros((nx, ny, nz), dtype=bool)
    volume[voxels_raw[:, 0], voxels_raw[:, 1], voxels_raw[:, 2]] = True

    try:
        from skimage.morphology import skeletonize_3d

        volume = skeletonize_3d(volume > 0)
    except Exception:
        try:
            from skimage.morphology import skeletonize

            volume = skeletonize(volume > 0)
        except Exception:
            pass

    labeled, num_labels = ndi.label(volume)
    if num_labels > 1:
        counts = ndi.sum(volume, labeled, index=range(1, num_labels + 1))
        largest_label = int(np.argmax(counts) + 1)
        volume = labeled == largest_label

    voxels = np.argwhere(volume > 0).astype(np.int32)
    if voxels.size == 0:
        voxels = voxels_raw

    coords = origin + spacing * voxels

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(coords.shape[0])
    for n, (x, y, z) in enumerate(coords):
        points.SetPoint(n, float(x), float(y), float(z))

    index_map: Dict[Tuple[int, int, int], int] = {
        (int(vx), int(vy), int(vz)): int(pid)
        for pid, (vx, vy, vz) in enumerate(voxels.tolist())
    }
    neighbors: Dict[int, Set[int]] = {pid: set() for pid in range(coords.shape[0])}

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    for pid, (vx, vy, vz) in enumerate(voxels.tolist()):
        for dx, dy, dz in offsets:
            nid = index_map.get((vx + dx, vy + dy, vz + dz))
            if nid is None or nid == pid:
                continue
            neighbors[pid].add(nid)

    def edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    lines = vtk.vtkCellArray()
    visited_edges: Set[Tuple[int, int]] = set()
    special_nodes = [nid for nid, nbrs in neighbors.items() if len(nbrs) != 2]
    if not special_nodes and neighbors:
        special_nodes = [next(iter(neighbors.keys()))]

    def add_path(path_nodes: List[int]) -> None:
        if len(path_nodes) < 2:
            return
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(path_nodes))
        for idx_node, pid in enumerate(path_nodes):
            polyline.GetPointIds().SetId(idx_node, pid)
        lines.InsertNextCell(polyline)

    for start in special_nodes:
        for nxt in neighbors[start]:
            key = edge_key(start, nxt)
            if key in visited_edges:
                continue
            path_nodes = [start]
            prev = start
            cur = nxt
            visited_edges.add(key)
            while True:
                path_nodes.append(cur)
                cur_neighbors = [nid for nid in neighbors[cur] if nid != prev]
                if len(neighbors[cur]) != 2 or not cur_neighbors:
                    break
                next_node = cur_neighbors[0]
                next_key = edge_key(cur, next_node)
                if next_key in visited_edges:
                    break
                visited_edges.add(next_key)
                prev, cur = cur, next_node
            add_path(path_nodes)

    # Capture any residual cycle edges that were not visited above.
    for a, nbrs in neighbors.items():
        for b in nbrs:
            key = edge_key(a, b)
            if key in visited_edges:
                continue
            path_nodes = [a, b]
            visited_edges.add(key)
            prev = a
            cur = b
            while True:
                cur_neighbors = [nid for nid in neighbors[cur] if nid != prev]
                if not cur_neighbors:
                    break
                next_node = cur_neighbors[0]
                next_key = edge_key(cur, next_node)
                if next_key in visited_edges:
                    break
                visited_edges.add(next_key)
                path_nodes.append(next_node)
                prev, cur = cur, next_node
            add_path(path_nodes)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(poly)
    writer.Write()
    return int(coords.shape[0])


def write_vtp_polylines(polylines: List[np.ndarray], output_path: Path) -> dict:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    total_points = 0
    total_lines = 0

    for poly in polylines:
        if poly.shape[0] < 2:
            continue
        start_id = points.GetNumberOfPoints()
        for p in poly:
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(poly.shape[0])
        for i in range(poly.shape[0]):
            line.GetPointIds().SetId(i, start_id + i)
        lines.InsertNextCell(line)
        total_lines += 1
        total_points += int(poly.shape[0])

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(poly)
    writer.Write()
    return {"points": total_points, "lines": total_lines}


def centerlines_from_mask_skeleton(mask_path: Path, output_path: Path) -> dict:
    from vessel_seg.shape import CentrelineParams, extract_branches

    params = CentrelineParams(
        min_length_mm=5.0,
        short_bridge_max_mm=6.0,
        closing_iterations=1,
        smooth_sigma_mm=0.8,
        adaptive_min_step_mm=0.6,
        adaptive_max_step_mm=2.5,
        curvature_alpha=2.0,
    )
    branches = extract_branches(mask_path, params)
    img = nib.load(str(mask_path))
    affine = img.affine
    origin = np.asarray(affine[:3, 3], dtype=float)
    spacing = np.abs(np.diag(affine)[:3]).astype(float)

    polylines = []
    for branch in branches:
        if branch.voxel_points.shape[0] < 2:
            continue
        points = origin[None, :] + branch.voxel_points.astype(float) * spacing[None, :]
        polylines.append(points.astype(np.float32))

    stats = write_vtp_polylines(polylines, output_path)
    return {
        "seed_mode": "mask_skeleton",
        "branch_count": len(polylines),
        "points": int(stats["points"]),
        "lines": int(stats["lines"]),
    }


def _read_vtp_polylines(vtp_path: Path) -> List[np.ndarray]:
    poly = read_polydata(vtp_path)
    pts = poly.GetPoints()
    if pts is None:
        return []
    lines = poly.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    polylines: List[np.ndarray] = []
    while lines.GetNextCell(id_list):
        if id_list.GetNumberOfIds() < 2:
            continue
        line = np.array([pts.GetPoint(id_list.GetId(i)) for i in range(id_list.GetNumberOfIds())], dtype=float)
        polylines.append(line)
    return polylines


def _resample_polyline(points: np.ndarray, step_mm: float) -> np.ndarray:
    if points.shape[0] < 2 or step_mm <= 0:
        return points
    diffs = np.diff(points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.zeros(points.shape[0], dtype=float)
    s[1:] = np.cumsum(seg_len)
    total = float(s[-1])
    if total <= step_mm:
        return points
    targets = np.arange(0.0, total, step_mm, dtype=float)
    if targets.size == 0 or abs(targets[-1] - total) > 1e-6:
        targets = np.append(targets, total)
    resampled = np.vstack([np.interp(targets, s, points[:, d]) for d in range(3)]).T
    return resampled.astype(np.float32)


def densify_centerline_vtp(vtp_path: Path, step_mm: float) -> Dict[str, int]:
    polylines = _read_vtp_polylines(vtp_path)
    if not polylines:
        return {"lines": 0, "points": 0}

    out_polylines = [_resample_polyline(poly, step_mm) for poly in polylines]

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for poly in out_polylines:
        if poly.shape[0] < 2:
            continue
        start_id = points.GetNumberOfPoints()
        for p in poly:
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(poly.shape[0])
        for i in range(poly.shape[0]):
            line.GetPointIds().SetId(i, start_id + i)
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path))
    writer.SetInputData(polydata)
    writer.Write()
    return {"lines": int(lines.GetNumberOfCells()), "points": int(points.GetNumberOfPoints())}


def centerlines_from_surface_with_gt(
    surface_path: Path, output_path: Path, mask_path: Path, gt_centerline: Path
) -> dict:
    surface = read_polydata(surface_path)
    endpoints = centerline_endpoints_vtp(gt_centerline)
    if endpoints.size == 0:
        raise ValueError("No endpoints found in GT centerline.")
    endpoints = project_points_to_surface(surface, endpoints)
    source, targets = select_source_target(endpoints, mask_path)

    cmd = [
        "conda",
        "run",
        "-n",
        "vessel_seg",
        "vmtkcenterlines",
        "-ifile",
        str(surface_path),
        "-ofile",
        str(output_path),
        "-seedselector",
        "pointlist",
        "-sourcepoints",
        *[f"{v:.6f}" for v in source.tolist()],
        "-targetpoints",
        *[f"{v:.6f}" for t in targets for v in t.tolist()],
        "-endpoints",
        "1",
    ]
    run(cmd)

    return {
        "seed_mode": "gt_endpoints",
        "endpoints": int(endpoints.shape[0]),
        "source_point": source.tolist(),
        "target_points": [t.tolist() for t in targets],
    }


def centerlines_from_surface(surface_path: Path, output_path: Path, mask_path: Path) -> dict:
    surface = read_polydata(surface_path)
    profiles = extract_open_profiles(surface)
    seed_mode = "openprofiles"
    if profiles:
        source, targets = build_seed_points(profiles)
        cmd = [
            "conda",
            "run",
            "-n",
            "vessel_seg",
            "vmtkcenterlines",
            "-ifile",
            str(surface_path),
            "-ofile",
            str(output_path),
            "-seedselector",
            "pointlist",
            "-sourcepoints",
            *[f"{v:.6f}" for v in source.tolist()],
            "-targetpoints",
            *[f"{v:.6f}" for t in targets for v in t.tolist()],
            "-endpoints",
            "1",
        ]
        run(cmd)
        return {
            "seed_mode": seed_mode,
            "profiles": len(profiles),
            "source_point": source.tolist(),
            "target_points": [t.tolist() for t in targets],
        }

    # Fallback: use in-repo mask skeleton extraction when surface has no open profiles.
    report = centerlines_from_mask_skeleton(mask_path, output_path)
    report["profiles"] = 0
    report["surface_fallback"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VMTK centerlines from a mask.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask (NIfTI).")
    parser.add_argument("--out", type=Path, required=True, help="Output centerline VTP.")
    parser.add_argument("--surface", type=Path, default=None, help="Optional surface VTP path.")
    parser.add_argument("--level", type=float, default=0.5, help="Iso-level for marching cubes.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report.")
    parser.add_argument(
        "--decimate",
        type=float,
        default=0.0,
        help="Surface decimation reduction fraction (0-1). 0 disables.",
    )
    parser.add_argument(
        "--gt-centerline",
        type=Path,
        default=None,
        help="Optional GT centerline VTP to seed vmtkcenterlines.",
    )
    parser.add_argument(
        "--resample-step-mm",
        type=float,
        default=0.5,
        help="Resample polyline spacing in mm for dense output (<=0 disables).",
    )
    args = parser.parse_args()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.surface is None:
        surface_path = out.with_suffix(".surface.vtp")
        marching_cubes(args.mask, surface_path, args.level)
        cleanup_surface = True
    else:
        surface_path = args.surface
        cleanup_surface = False

    cleanup_decim = False
    if args.decimate and args.decimate > 0.0:
        decim_path = out.with_suffix(".surface_decim.vtp")
        decimate_surface(surface_path, decim_path, args.decimate)
        surface_path = decim_path
        cleanup_decim = True

    if args.gt_centerline is not None:
        report = centerlines_from_surface_with_gt(surface_path, out, args.mask, args.gt_centerline)
    else:
        report = centerlines_from_surface(surface_path, out, args.mask)

    if args.resample_step_mm is not None and args.resample_step_mm > 0:
        dense_stats = densify_centerline_vtp(out, args.resample_step_mm)
        report["resample_step_mm"] = float(args.resample_step_mm)
        report["resampled_points"] = dense_stats["points"]
        report["resampled_lines"] = dense_stats["lines"]

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))

    if cleanup_decim:
        try:
            surface_path.unlink()
        except OSError:
            pass

    if cleanup_surface:
        try:
            (out.with_suffix(".surface.vtp")).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
