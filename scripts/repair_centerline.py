#!/usr/bin/env python3
"""Repair broken centerlines by connecting endpoints using a probability-guided path.

Inputs:
  --prob : probability map (.nii/.nii.gz), values in [0,1]
  --vtp  : initial centerlines VTP (e.g., from VMTK)
Output:
  --out  : repaired VTP with added bridge polylines

Example:
  python scripts/repair_centerline.py \
    --prob ASOCA2020/Normal/Prob/Normal_1_prob.nii.gz \
    --vtp ASOCA2020/Normal/Centerlines/Normal_1.vtp \
    --out outputs/Normal_1/centerline_repaired.vtp \
    --prob_thresh 0.2 --max_dist 10 --max_bridge_len 25 \
    --max_angle_deg 75 --w_prob 1.0 --w_dist 0.6 --outside_penalty 10.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage

try:
    from skimage.graph import route_through_array
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scikit-image is required for path routing. Install with `pip install scikit-image`."
    ) from exc

try:
    import vtk  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("vtk is required to read/write VTP. Install vtk (e.g., via conda-forge).") from exc


@dataclass
class Endpoint:
    branch_id: int
    end_idx: int  # 0 for start, -1 for end
    point: np.ndarray  # world coords (3,)
    tangent: np.ndarray  # unit vector (3,)


def read_vtp_centerlines(vtp_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray | None]]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    radius_arr = poly.GetPointData().GetArray("MaximumInscribedSphereRadius")
    lines = poly.GetLines()
    lines.InitTraversal()
    branches: List[np.ndarray] = []
    branch_radii: List[np.ndarray | None] = []
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        if ids.GetNumberOfIds() < 2:
            continue
        coords = np.array([pts.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
        branches.append(coords)
        if radius_arr is not None:
            radii = np.array(
                [radius_arr.GetTuple1(ids.GetId(i)) for i in range(ids.GetNumberOfIds())],
                dtype=float,
            )
            branch_radii.append(radii)
        else:
            branch_radii.append(None)
    if not branches:
        raise ValueError(f"No polylines found in {vtp_path}")
    return branches, branch_radii


def write_vtp_centerlines(polylines: List[np.ndarray], out_path: Path, bridge_flags: List[int]) -> None:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    cell_flags = vtk.vtkIntArray()
    cell_flags.SetName("is_bridge")

    for i, poly in enumerate(polylines):
        start_id = points.GetNumberOfPoints()
        for p in poly:
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(poly))
        for j in range(len(poly)):
            line.GetPointIds().SetId(j, start_id + j)
        lines.InsertNextCell(line)
        cell_flags.InsertNextValue(int(bridge_flags[i]))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetCellData().AddArray(cell_flags)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(polydata)
    writer.Write()


def endpoint_from_polyline(poly: np.ndarray, branch_id: int, end_idx: int) -> Endpoint:
    if end_idx == 0:
        p0, p1 = poly[0], poly[1]
        tangent = p0 - p1  # outward
    else:
        p0, p1 = poly[-1], poly[-2]
        tangent = p0 - p1  # outward
    norm = np.linalg.norm(tangent) + 1e-8
    tangent = tangent / norm
    return Endpoint(branch_id=branch_id, end_idx=end_idx, point=p0.astype(float), tangent=tangent)


def collect_endpoints(polylines: List[np.ndarray]) -> List[Endpoint]:
    endpoints: List[Endpoint] = []
    for bid, poly in enumerate(polylines):
        if poly.shape[0] < 2:
            continue
        endpoints.append(endpoint_from_polyline(poly, bid, 0))
        endpoints.append(endpoint_from_polyline(poly, bid, -1))
    return endpoints


def angle_deg_between(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def build_cost_volume(
    prob: np.ndarray,
    spacing: Tuple[float, float, float],
    prob_thresh: float,
    w_prob: float,
    w_dist: float,
    outside_penalty: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = np.clip(prob, 0.0, 1.0)
    mask = prob >= prob_thresh
    dist = ndimage.distance_transform_edt(mask, sampling=spacing)
    cost = w_prob * (1.0 - prob) + w_dist * (1.0 / (dist + 1e-3))
    cost = cost.astype(np.float32)
    if outside_penalty > 0:
        cost = cost + (~mask) * outside_penalty
    return cost, mask, dist


def world_to_voxel(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    # VMTK centerlines are typically written in image-origin + abs(spacing)*index
    # coordinates, while NIfTI affine may contain axis flips. Use absolute
    # spacing convention here to stay consistent with VMTK/polyline coordinates.
    origin = np.asarray(affine[:3, 3], dtype=float)
    spacing = np.sqrt((np.asarray(affine[:3, :3], dtype=float) ** 2).sum(axis=0))
    spacing = np.where(spacing > 0, spacing, 1.0)
    return (points - origin[None, :]) / spacing[None, :]


def voxel_to_world(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    origin = np.asarray(affine[:3, 3], dtype=float)
    spacing = np.sqrt((np.asarray(affine[:3, :3], dtype=float) ** 2).sum(axis=0))
    spacing = np.where(spacing > 0, spacing, 1.0)
    return origin[None, :] + points * spacing[None, :]


def clamp_index(idx: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    idx = np.round(idx).astype(int)
    idx[0] = int(np.clip(idx[0], 0, shape[0] - 1))
    idx[1] = int(np.clip(idx[1], 0, shape[1] - 1))
    idx[2] = int(np.clip(idx[2], 0, shape[2] - 1))
    return idx


def path_length_mm(path_vox: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    if path_vox.shape[0] < 2:
        return 0.0
    diffs = np.diff(path_vox.astype(float), axis=0)
    diffs_mm = diffs * np.array(spacing)[None, :]
    return float(np.linalg.norm(diffs_mm, axis=1).sum())


def smooth_path(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or points.shape[0] < 3:
        return points
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.vstack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)]).T
    return smoothed


def smooth_path_preserve_endpoints(points: np.ndarray, window: int) -> np.ndarray:
    smoothed = smooth_path(points, window)
    if smoothed.shape[0] >= 1:
        smoothed[0] = points[0]
        smoothed[-1] = points[-1]
    return smoothed


def max_curvature(path_world: np.ndarray) -> float:
    if path_world.shape[0] < 3:
        return 0.0
    segs = np.diff(path_world, axis=0)
    seg_lens = np.linalg.norm(segs, axis=1) + 1e-8
    tangents = segs / seg_lens[:, None]
    angles = []
    lengths = []
    for i in range(len(tangents) - 1):
        dot = float(np.clip(np.dot(tangents[i], tangents[i + 1]), -1.0, 1.0))
        ang = float(np.arccos(dot))
        angles.append(ang)
        lengths.append(0.5 * (seg_lens[i] + seg_lens[i + 1]))
    if not angles:
        return 0.0
    curv = np.array(angles) / (np.array(lengths) + 1e-8)
    return float(np.max(curv))


def polyline_length_mm(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def snap_nearby_endpoints(
    polylines: List[np.ndarray],
    *,
    snap_tol_mm: float,
    blend_steps: int,
) -> Tuple[List[np.ndarray], dict]:
    if snap_tol_mm <= 0 or not polylines:
        return polylines, {"clusters": 0, "snapped_endpoints": 0}

    records = []
    for bid, poly in enumerate(polylines):
        if poly.shape[0] == 0:
            continue
        records.append({"branch_id": bid, "end_idx": 0, "point": poly[0].copy()})
        if poly.shape[0] > 1:
            records.append({"branch_id": bid, "end_idx": -1, "point": poly[-1].copy()})

    n = len(records)
    if n < 2:
        return polylines, {"clusters": 0, "snapped_endpoints": 0}

    # Use seed-centric grouping (no transitive chaining) to avoid collapsing
    # distant bifurcations through a chain of near neighbors.
    remaining = set(range(n))
    clusters: List[List[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        seed_pt = records[seed]["point"]
        cluster = [seed]
        attached = []
        for idx in remaining:
            if float(np.linalg.norm(records[idx]["point"] - seed_pt)) <= snap_tol_mm:
                attached.append(idx)
        for idx in attached:
            remaining.remove(idx)
            cluster.append(idx)
        clusters.append(cluster)

    out = [poly.copy() for poly in polylines]
    cluster_count = 0
    snapped_count = 0
    blend_steps = max(int(blend_steps), 0)

    for members in clusters:
        if len(members) < 2:
            continue
        pts = np.array([records[m]["point"] for m in members], dtype=float)
        centroid = pts.mean(axis=0)
        cluster_count += 1
        snapped_count += len(members)

        for m in members:
            rec = records[m]
            bid = int(rec["branch_id"])
            end_idx = int(rec["end_idx"])
            poly = out[bid]
            if poly.shape[0] == 0:
                continue

            if end_idx == 0:
                poly[0] = centroid
                if blend_steps > 0 and poly.shape[0] > 1:
                    span = min(blend_steps, poly.shape[0] - 1)
                    for k in range(1, span + 1):
                        alpha = float(span + 1 - k) / float(span + 1)
                        poly[k] = alpha * poly[k] + (1.0 - alpha) * centroid
            else:
                poly[-1] = centroid
                if blend_steps > 0 and poly.shape[0] > 1:
                    span = min(blend_steps, poly.shape[0] - 1)
                    for k in range(1, span + 1):
                        alpha = float(span + 1 - k) / float(span + 1)
                        idx = -1 - k
                        poly[idx] = alpha * poly[idx] + (1.0 - alpha) * centroid

    return out, {"clusters": cluster_count, "snapped_endpoints": snapped_count}


def regularize_polylines_curvature(
    polylines: List[np.ndarray],
    *,
    smooth_window: int,
    max_curv: float | None,
    iterations: int,
) -> Tuple[List[np.ndarray], dict]:
    out = []
    max_before = 0.0
    max_after = 0.0
    adjusted = 0
    iterations = max(int(iterations), 0)

    for poly in polylines:
        work = poly.copy()
        if work.shape[0] < 3:
            out.append(work)
            continue

        before = max_curvature(work)
        after = before
        if max_curv is not None and max_curv > 0:
            n_iter = 0
            while after > max_curv and n_iter < iterations:
                work = smooth_path_preserve_endpoints(work, smooth_window)
                after = max_curvature(work)
                n_iter += 1
            if n_iter > 0:
                adjusted += 1
        else:
            work = smooth_path_preserve_endpoints(work, smooth_window)
            after = max_curvature(work)
            if after < before:
                adjusted += 1

        max_before = max(max_before, before)
        max_after = max(max_after, after)
        out.append(work)

    return out, {"adjusted_branches": adjusted, "max_curvature_before": max_before, "max_curvature_after": max_after}


def sample_radius(dist: np.ndarray, world_pt: np.ndarray, affine: np.ndarray) -> float:
    vox = world_to_voxel(world_pt[None, :], affine)[0]
    idx = clamp_index(vox, dist.shape)
    return float(dist[tuple(idx.tolist())])


def endpoint_radius(
    endpoint: Endpoint,
    branch_radii: List[np.ndarray | None],
    dist: np.ndarray,
    affine: np.ndarray,
) -> float:
    radii = branch_radii[endpoint.branch_id]
    if radii is not None and radii.size > 0:
        return float(radii[0] if endpoint.end_idx == 0 else radii[-1])
    return sample_radius(dist, endpoint.point, affine)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair centerlines by connecting endpoints.")
    parser.add_argument("--prob", type=Path, required=True, help="Probability map (NIfTI).")
    parser.add_argument("--vtp", type=Path, required=True, help="Input centerline VTP (from VMTK).")
    parser.add_argument("--out", type=Path, required=True, help="Output repaired VTP.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report.")
    parser.add_argument("--prob_thresh", type=float, default=0.2)
    parser.add_argument("--max_dist", type=float, default=10.0, help="Max endpoint distance (mm).")
    parser.add_argument("--max_bridge_len", type=float, default=25.0, help="Max allowed bridge length (mm).")
    parser.add_argument("--max_angle_deg", type=float, default=90.0, help="Max angle between outward tangents (deg).")
    parser.add_argument("--max_pairs", type=int, default=50)
    parser.add_argument("--w_prob", type=float, default=1.0)
    parser.add_argument("--w_dist", type=float, default=0.6)
    parser.add_argument("--outside_penalty", type=float, default=10.0)
    parser.add_argument("--smooth_window", type=int, default=5, help="Moving-average window for bridge (odd).")
    parser.add_argument("--max_curvature", type=float, default=0.4, help="Max allowed curvature (1/mm) after smoothing.")
    parser.add_argument("--murray_exp", type=float, default=3.0, help="Exponent for Murray-style radius consistency.")
    parser.add_argument("--murray_tol", type=float, default=0.5, help="Max allowed Murray deviation (0-1).")
    parser.add_argument(
        "--junction_snap_tol",
        type=float,
        default=1.0,
        help="Snap nearby branch endpoints within this distance (mm). <=0 disables.",
    )
    parser.add_argument(
        "--junction_blend_steps",
        type=int,
        default=3,
        help="Number of interior points blended from each snapped endpoint.",
    )
    parser.add_argument(
        "--regularize_curvature",
        type=float,
        default=0.45,
        help="Post-repair max branch curvature (1/mm). <=0 disables curvature cap.",
    )
    parser.add_argument(
        "--regularize_iterations",
        type=int,
        default=3,
        help="Max smoothing iterations during curvature regularization.",
    )
    parser.add_argument(
        "--min_branch_length",
        type=float,
        default=0.0,
        help="Drop repaired branches shorter than this length (mm). <=0 keeps all.",
    )
    args = parser.parse_args()

    prob_img = nib.load(str(args.prob))
    prob = prob_img.get_fdata().astype(np.float32)
    spacing = prob_img.header.get_zooms()[:3]
    affine = prob_img.affine

    cost, _, dist_map = build_cost_volume(
        prob=prob,
        spacing=spacing,
        prob_thresh=args.prob_thresh,
        w_prob=args.w_prob,
        w_dist=args.w_dist,
        outside_penalty=args.outside_penalty,
    )

    polylines, branch_radii = read_vtp_centerlines(args.vtp)
    endpoints = collect_endpoints(polylines)

    # Build candidate pairs
    pairs = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            a = endpoints[i]
            b = endpoints[j]
            if a.branch_id == b.branch_id:
                continue
            endpoint_dist = float(np.linalg.norm(a.point - b.point))
            if endpoint_dist > args.max_dist:
                continue
            angle = angle_deg_between(a.tangent, -b.tangent)
            if angle > args.max_angle_deg:
                continue
            pairs.append((endpoint_dist, angle, i, j))
    pairs.sort(key=lambda x: (x[0], x[1]))
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    used = set()
    bridges: List[np.ndarray] = []
    bridge_meta = []
    shape = prob.shape

    for endpoint_dist, angle, i, j in pairs:
        if i in used or j in used:
            continue
        a = endpoints[i]
        b = endpoints[j]
        start_vox = clamp_index(world_to_voxel(a.point[None, :], affine)[0], shape)
        end_vox = clamp_index(world_to_voxel(b.point[None, :], affine)[0], shape)

        try:
            path, cost_val = route_through_array(
                cost,
                tuple(start_vox.tolist()),
                tuple(end_vox.tolist()),
                fully_connected=True,
                geometric=True,
            )
        except Exception:
            continue
        path_vox = np.array(path, dtype=float)
        length_mm = path_length_mm(path_vox, spacing)
        if length_mm <= 0 or length_mm > args.max_bridge_len:
            continue

        path_world = voxel_to_world(path_vox, affine)
        path_world = smooth_path(path_world, args.smooth_window)
        curvature = max_curvature(path_world)
        if args.max_curvature is not None and curvature > args.max_curvature:
            continue

        ra = endpoint_radius(a, branch_radii, dist_map, affine)
        rb = endpoint_radius(b, branch_radii, dist_map, affine)
        r_parent = max(ra, rb)
        r_child = min(ra, rb)
        murray_dev = abs((r_parent ** args.murray_exp) - (r_child ** args.murray_exp)) / (
            r_parent ** args.murray_exp + 1e-8
        )
        if args.murray_tol is not None and murray_dev > args.murray_tol:
            continue

        bridges.append(path_world)
        bridge_meta.append(
            {
                "endpoint_a": {"branch": a.branch_id, "end_idx": a.end_idx, "point": a.point.tolist()},
                "endpoint_b": {"branch": b.branch_id, "end_idx": b.end_idx, "point": b.point.tolist()},
                "dist_mm": endpoint_dist,
                "angle_deg": angle,
                "path_len_mm": length_mm,
                "max_curvature": curvature,
                "radius_a": ra,
                "radius_b": rb,
                "murray_deviation": murray_dev,
                "cost": float(cost_val),
            }
        )
        used.add(i)
        used.add(j)

    out_polylines = polylines + bridges
    bridge_flags = [0] * len(polylines) + [1] * len(bridges)

    out_polylines, snap_stats = snap_nearby_endpoints(
        out_polylines,
        snap_tol_mm=args.junction_snap_tol,
        blend_steps=args.junction_blend_steps,
    )

    reg_cap = args.regularize_curvature if args.regularize_curvature and args.regularize_curvature > 0 else None
    out_polylines, regularize_stats = regularize_polylines_curvature(
        out_polylines,
        smooth_window=args.smooth_window,
        max_curv=reg_cap,
        iterations=args.regularize_iterations,
    )

    removed_short = 0
    if args.min_branch_length and args.min_branch_length > 0:
        keep_polys: List[np.ndarray] = []
        keep_flags: List[int] = []
        for poly, flag in zip(out_polylines, bridge_flags):
            if polyline_length_mm(poly) < args.min_branch_length:
                removed_short += 1
                continue
            keep_polys.append(poly)
            keep_flags.append(flag)
        out_polylines = keep_polys
        bridge_flags = keep_flags

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_vtp_centerlines(out_polylines, args.out, bridge_flags)
    print(f"Saved repaired VTP -> {args.out} | added bridges: {len(bridges)}")

    if args.report is not None:
        payload = {
            "prob": str(args.prob),
            "vtp": str(args.vtp),
            "out": str(args.out),
            "num_branches": len(polylines),
            "num_bridges": len(bridges),
            "params": {
                "prob_thresh": args.prob_thresh,
                "max_dist": args.max_dist,
                "max_bridge_len": args.max_bridge_len,
                "max_angle_deg": args.max_angle_deg,
                "w_prob": args.w_prob,
                "w_dist": args.w_dist,
                "outside_penalty": args.outside_penalty,
                "smooth_window": args.smooth_window,
                "max_curvature": args.max_curvature,
                "murray_exp": args.murray_exp,
                "murray_tol": args.murray_tol,
                "junction_snap_tol": args.junction_snap_tol,
                "junction_blend_steps": args.junction_blend_steps,
                "regularize_curvature": args.regularize_curvature,
                "regularize_iterations": args.regularize_iterations,
                "min_branch_length": args.min_branch_length,
            },
            "snap_stats": snap_stats,
            "regularize_stats": regularize_stats,
            "removed_short_branches": removed_short,
            "bridges": bridge_meta,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report -> {args.report}")


if __name__ == "__main__":
    main()
