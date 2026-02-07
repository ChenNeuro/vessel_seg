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
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

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
    coords = origin + spacing * np.vstack([i, j, k]).T

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(coords.shape[0])
    for n, (x, y, z) in enumerate(coords):
        points.SetPoint(n, float(x), float(y), float(z))

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(poly)
    writer.Write()
    return coords.shape[0]


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

    # Fallback: use vmtkcenterlineimage for closed surfaces.
    seed_mode = "centerlineimage"
    spacing_candidates = [
        (0.3, 0.3, 0.3),
        mask_spacing(mask_path),
        (0.6, 0.6, 0.6),
    ]
    last_error = None
    for spacing in spacing_candidates:
        tmp_image = output_path.with_suffix(".centerlineimage.vti")
        cmd = [
            "conda",
            "run",
            "-n",
            "vessel_seg",
            "vmtkcenterlineimage",
            "-ifile",
            str(surface_path),
            "-ofile",
            str(tmp_image),
            "-spacing",
            f"{spacing[0]:.6f}",
            f"{spacing[1]:.6f}",
            f"{spacing[2]:.6f}",
        ]
        try:
            run(cmd)
            num_points = centerlineimage_to_vtp(tmp_image, output_path)
            try:
                tmp_image.unlink()
            except OSError:
                pass
            return {
                "seed_mode": seed_mode,
                "profiles": 0,
                "points": num_points,
                "spacing": spacing,
            }
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_error = exc
            try:
                tmp_image.unlink()
            except OSError:
                pass
            continue

    raise RuntimeError(f"centerlineimage failed after retries: {last_error}") from last_error


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
