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
from typing import Callable, List, Tuple

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


@dataclass
class CoordinateTransform:
    mode: str
    world_to_voxel: Callable[[np.ndarray], np.ndarray]
    voxel_to_world: Callable[[np.ndarray], np.ndarray]


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


def write_vtp_centerlines(
    polylines: List[np.ndarray],
    out_path: Path,
    bridge_flags: List[int],
    *,
    alignment_shift_world: np.ndarray | None = None,
    coordinate_mode: str | None = None,
    merge_point_tol_mm: float = 1e-6,
) -> dict:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    cell_flags = vtk.vtkIntArray()
    cell_flags.SetName("is_bridge")
    point_id_map: dict[tuple[int, int, int], int] = {}
    use_merge = float(merge_point_tol_mm) > 0
    inv_tol = 1.0 / float(merge_point_tol_mm) if use_merge else 0.0

    for i, poly in enumerate(polylines):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(poly))
        for j in range(len(poly)):
            p = poly[j]
            if use_merge:
                key = (
                    int(np.round(float(p[0]) * inv_tol)),
                    int(np.round(float(p[1]) * inv_tol)),
                    int(np.round(float(p[2]) * inv_tol)),
                )
                pid = point_id_map.get(key)
                if pid is None:
                    pid = int(points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
                    point_id_map[key] = pid
            else:
                pid = int(points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
            line.GetPointIds().SetId(j, pid)
        lines.InsertNextCell(line)
        cell_flags.InsertNextValue(int(bridge_flags[i]))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetCellData().AddArray(cell_flags)

    field_data = polydata.GetFieldData()
    if alignment_shift_world is not None:
        shift_arr = vtk.vtkDoubleArray()
        shift_arr.SetName("alignment_shift_world")
        shift_arr.SetNumberOfComponents(3)
        shift_arr.InsertNextTuple(
            (
                float(alignment_shift_world[0]),
                float(alignment_shift_world[1]),
                float(alignment_shift_world[2]),
            )
        )
        field_data.AddArray(shift_arr)
    if coordinate_mode is not None:
        mode_arr = vtk.vtkStringArray()
        mode_arr.SetName("coordinate_mode")
        mode_arr.InsertNextValue(str(coordinate_mode))
        field_data.AddArray(mode_arr)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(polydata)
    writer.Write()
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(polydata)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return {
        "points": int(polydata.GetNumberOfPoints()),
        "lines": int(polydata.GetNumberOfLines()),
        "cells": int(polydata.GetNumberOfCells()),
        "components": int(conn.GetNumberOfExtractedRegions()),
    }


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


def make_coordinate_transform(affine: np.ndarray, mode: str) -> CoordinateTransform:
    mode = mode.lower()
    if mode not in {"abs_spacing", "affine"}:
        raise ValueError(f"Unsupported coordinate mode: {mode}")

    if mode == "abs_spacing":
        origin = np.asarray(affine[:3, 3], dtype=float)
        spacing = np.sqrt((np.asarray(affine[:3, :3], dtype=float) ** 2).sum(axis=0))
        spacing = np.where(spacing > 0, spacing, 1.0)

        def world_to_voxel_fn(points: np.ndarray) -> np.ndarray:
            return (points - origin[None, :]) / spacing[None, :]

        def voxel_to_world_fn(points: np.ndarray) -> np.ndarray:
            return origin[None, :] + points * spacing[None, :]

        return CoordinateTransform(mode=mode, world_to_voxel=world_to_voxel_fn, voxel_to_world=voxel_to_world_fn)

    inv_affine = np.linalg.inv(affine)

    def world_to_voxel_fn(points: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1), dtype=float)
        hom = np.hstack([points.astype(float), ones])
        vox = (inv_affine @ hom.T).T[:, :3]
        return vox

    def voxel_to_world_fn(points: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1), dtype=float)
        hom = np.hstack([points.astype(float), ones])
        world = (affine @ hom.T).T[:, :3]
        return world

    return CoordinateTransform(mode=mode, world_to_voxel=world_to_voxel_fn, voxel_to_world=voxel_to_world_fn)


def clamp_index(idx: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    idx = np.round(idx).astype(int)
    idx[0] = int(np.clip(idx[0], 0, shape[0] - 1))
    idx[1] = int(np.clip(idx[1], 0, shape[1] - 1))
    idx[2] = int(np.clip(idx[2], 0, shape[2] - 1))
    return idx


def _in_bounds_idx(idx: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    return (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < shape[0])
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < shape[1])
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < shape[2])
    )


def _mask_center_world(mask: np.ndarray, transform: CoordinateTransform) -> np.ndarray:
    nz = np.argwhere(mask)
    if nz.size == 0:
        center_vox = 0.5 * (np.asarray(mask.shape, dtype=float) - 1.0)
    else:
        mins = nz.min(axis=0).astype(float)
        maxs = nz.max(axis=0).astype(float)
        center_vox = 0.5 * (mins + maxs)
    return transform.voxel_to_world(center_vox[None, :])[0]


def alignment_score(
    points_world: np.ndarray,
    *,
    prob: np.ndarray,
    mask: np.ndarray,
    transform: CoordinateTransform,
    outside_penalty: float,
) -> dict:
    if points_world.size == 0:
        return {"score": float("inf"), "inside_ratio": 0.0, "mean_prob": 0.0}

    vox = transform.world_to_voxel(points_world)
    idx = np.round(vox).astype(int)
    in_bounds = _in_bounds_idx(idx, prob.shape)
    if not np.any(in_bounds):
        return {"score": float("inf"), "inside_ratio": 0.0, "mean_prob": 0.0}

    idx_in = idx[in_bounds]
    prob_vals = prob[idx_in[:, 0], idx_in[:, 1], idx_in[:, 2]]
    mask_hits = mask[idx_in[:, 0], idx_in[:, 1], idx_in[:, 2]]

    outside_frac = 1.0 - float(in_bounds.mean())
    inside_ratio = float(mask_hits.mean()) if mask_hits.size else 0.0
    mean_prob = float(prob_vals.mean()) if prob_vals.size else 0.0

    # Lower is better. Balance outside penalty and inside-probability consistency.
    score = outside_frac * outside_penalty + (1.0 - inside_ratio) + (1.0 - mean_prob)
    return {"score": float(score), "inside_ratio": inside_ratio, "mean_prob": mean_prob}


def choose_coordinate_mode(
    points_world: np.ndarray,
    *,
    prob: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    outside_penalty: float,
) -> tuple[CoordinateTransform, dict]:
    cands = {}
    for mode in ("abs_spacing", "affine"):
        tr = make_coordinate_transform(affine, mode)
        sc = alignment_score(
            points_world,
            prob=prob,
            mask=mask,
            transform=tr,
            outside_penalty=outside_penalty,
        )
        cands[mode] = {"transform": tr, **sc}

    # Primary: lower alignment score; secondary: deterministic mode name.
    best_mode = min(cands.keys(), key=lambda m: (cands[m]["score"], m))
    info = {
        "selected_mode": best_mode,
        "candidates": {
            m: {
                "score": float(v["score"]),
                "inside_ratio": float(v["inside_ratio"]),
                "mean_prob": float(v["mean_prob"]),
            }
            for m, v in cands.items()
        },
    }
    return cands[best_mode]["transform"], info


def estimate_alignment_shift(
    points_world: np.ndarray,
    *,
    prob: np.ndarray,
    mask: np.ndarray,
    transform: CoordinateTransform,
    outside_penalty: float,
    max_points: int,
    steps_mm: tuple[float, ...],
) -> tuple[np.ndarray, dict]:
    if points_world.size == 0:
        return np.zeros(3, dtype=float), {
            "enabled": True,
            "best_shift_world": [0.0, 0.0, 0.0],
            "best_score": None,
            "inside_ratio": None,
            "mean_prob": None,
            "num_samples": 0,
            "steps_mm": list(steps_mm),
        }

    sample = points_world
    if max_points > 0 and points_world.shape[0] > max_points:
        ids = np.linspace(0, points_world.shape[0] - 1, num=max_points, dtype=int)
        sample = points_world[ids]

    cl_center = 0.5 * (sample.min(axis=0) + sample.max(axis=0))
    mask_center = _mask_center_world(mask, transform)
    init_shift = mask_center - cl_center

    def score(shift: np.ndarray) -> dict:
        return alignment_score(
            sample + shift[None, :],
            prob=prob,
            mask=mask,
            transform=transform,
            outside_penalty=outside_penalty,
        )

    best_shift = init_shift.copy()
    best = score(best_shift)
    for step in steps_mm:
        improved = True
        while improved:
            improved = False
            for axis in range(3):
                for delta in (-step, step):
                    trial = best_shift.copy()
                    trial[axis] += delta
                    sc = score(trial)
                    if sc["score"] + 1e-12 < best["score"]:
                        best = sc
                        best_shift = trial
                        improved = True

    info = {
        "enabled": True,
        "init_shift_world": init_shift.tolist(),
        "best_shift_world": best_shift.tolist(),
        "best_score": float(best["score"]),
        "inside_ratio": float(best["inside_ratio"]),
        "mean_prob": float(best["mean_prob"]),
        "num_samples": int(sample.shape[0]),
        "steps_mm": list(steps_mm),
    }
    return best_shift.astype(float), info


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


def densify_polyline(points: np.ndarray, max_step_mm: float) -> np.ndarray:
    if max_step_mm <= 0 or points.shape[0] < 2:
        return points
    out = [points[0]]
    for i in range(points.shape[0] - 1):
        p0 = points[i]
        p1 = points[i + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= max_step_mm:
            out.append(p1)
            continue
        n = int(np.ceil(seg_len / max_step_mm))
        for k in range(1, n + 1):
            t = float(k) / float(n)
            out.append(p0 * (1.0 - t) + p1 * t)
    return np.asarray(out, dtype=float)


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


def sample_radius(dist: np.ndarray, world_pt: np.ndarray, transform: CoordinateTransform) -> float:
    vox = transform.world_to_voxel(world_pt[None, :])[0]
    idx = clamp_index(vox, dist.shape)
    return float(dist[tuple(idx.tolist())])


def endpoint_radius(
    endpoint: Endpoint,
    branch_radii: List[np.ndarray | None],
    dist: np.ndarray,
    transform: CoordinateTransform,
) -> float:
    radii = branch_radii[endpoint.branch_id]
    if radii is not None and radii.size > 0:
        return float(radii[0] if endpoint.end_idx == 0 else radii[-1])
    return sample_radius(dist, endpoint.point, transform)


def build_pair_candidates(
    endpoints: List[Endpoint],
    branch_lengths: List[float],
    *,
    max_dist: float,
    max_angle_deg: float,
    max_pairs: int,
) -> List[dict]:
    pair_candidates: List[dict] = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            a = endpoints[i]
            b = endpoints[j]
            if a.branch_id == b.branch_id:
                continue
            endpoint_dist = float(np.linalg.norm(a.point - b.point))
            if endpoint_dist > max_dist:
                continue
            angle = angle_deg_between(a.tangent, -b.tangent)
            if angle > max_angle_deg:
                continue

            # Prefer endpoint pairs that belong to longer branches to avoid
            # repeatedly selecting tiny noisy fragments.
            len_a = float(branch_lengths[a.branch_id]) if a.branch_id < len(branch_lengths) else 0.0
            len_b = float(branch_lengths[b.branch_id]) if b.branch_id < len(branch_lengths) else 0.0
            pre_score = endpoint_dist + 0.15 * angle - 0.01 * (len_a + len_b)
            pair_candidates.append(
                {
                    "i": i,
                    "j": j,
                    "dist_mm": endpoint_dist,
                    "angle_deg": angle,
                    "pre_score": float(pre_score),
                }
            )

    pair_candidates.sort(
        key=lambda c: (
            c["pre_score"],
            c["dist_mm"],
            c["angle_deg"],
            endpoints[c["i"]].branch_id,
            endpoints[c["j"]].branch_id,
            endpoints[c["i"]].end_idx,
            endpoints[c["j"]].end_idx,
        )
    )
    if max_pairs > 0:
        pair_candidates = pair_candidates[:max_pairs]
    return pair_candidates


def build_valid_routes(
    pair_candidates: List[dict],
    *,
    endpoints: List[Endpoint],
    transform: CoordinateTransform,
    cost: np.ndarray,
    spacing: Tuple[float, float, float],
    shape: Tuple[int, int, int],
    max_bridge_len: float,
    max_curvature_allowed: float | None,
    smooth_window: int,
    bridge_smooth_iterations: int,
    branch_radii: List[np.ndarray | None],
    dist_map: np.ndarray,
    murray_exp: float,
    murray_tol: float | None,
) -> tuple[List[dict], dict]:
    valid_routes: List[dict] = []
    rejected_reasons = {
        "routing_failed": 0,
        "bridge_len_outside_range": 0,
        "curvature_too_high": 0,
        "murray_too_high": 0,
    }

    for cand in pair_candidates:
        i = int(cand["i"])
        j = int(cand["j"])
        a = endpoints[i]
        b = endpoints[j]
        start_vox = clamp_index(transform.world_to_voxel(a.point[None, :])[0], shape)
        end_vox = clamp_index(transform.world_to_voxel(b.point[None, :])[0], shape)

        try:
            path, cost_val = route_through_array(
                cost,
                tuple(start_vox.tolist()),
                tuple(end_vox.tolist()),
                fully_connected=True,
                geometric=True,
            )
        except Exception:
            rejected_reasons["routing_failed"] += 1
            continue

        path_vox = np.array(path, dtype=float)
        length_mm = path_length_mm(path_vox, spacing)
        if length_mm <= 0 or length_mm > max_bridge_len:
            rejected_reasons["bridge_len_outside_range"] += 1
            continue

        path_world = transform.voxel_to_world(path_vox)
        path_world = smooth_path_preserve_endpoints(path_world, smooth_window)
        curvature = max_curvature(path_world)
        if max_curvature_allowed is not None and curvature > max_curvature_allowed:
            work = path_world.copy()
            for _ in range(max(int(bridge_smooth_iterations), 0)):
                work = smooth_path_preserve_endpoints(work, smooth_window)
                curvature = max_curvature(work)
                if curvature <= max_curvature_allowed:
                    path_world = work
                    break
            if curvature > max_curvature_allowed:
                rejected_reasons["curvature_too_high"] += 1
                continue

        ra = endpoint_radius(a, branch_radii, dist_map, transform)
        rb = endpoint_radius(b, branch_radii, dist_map, transform)
        r_parent = max(ra, rb)
        r_child = min(ra, rb)
        murray_dev = abs((r_parent ** murray_exp) - (r_child ** murray_exp)) / (r_parent ** murray_exp + 1e-8)
        if murray_tol is not None and murray_dev > murray_tol:
            rejected_reasons["murray_too_high"] += 1
            continue

        # Global route score for deterministic selection (lower is better).
        route_score = (
            float(cost_val) / (length_mm + 1e-6)
            + 0.10 * float(cand["dist_mm"])
            + 0.01 * float(cand["angle_deg"])
            + 0.50 * float(murray_dev)
            + 0.20 * float(curvature)
        )
        valid_routes.append(
            {
                "i": i,
                "j": j,
                "path_world": path_world,
                "score": float(route_score),
                "dist_mm": float(cand["dist_mm"]),
                "angle_deg": float(cand["angle_deg"]),
                "path_len_mm": float(length_mm),
                "max_curvature": float(curvature),
                "radius_a": float(ra),
                "radius_b": float(rb),
                "murray_deviation": float(murray_dev),
                "cost": float(cost_val),
            }
        )

    valid_routes.sort(
        key=lambda c: (
            c["score"],
            c["dist_mm"],
            c["angle_deg"],
            endpoints[c["i"]].branch_id,
            endpoints[c["j"]].branch_id,
            endpoints[c["i"]].end_idx,
            endpoints[c["j"]].end_idx,
        )
    )
    return valid_routes, rejected_reasons


def merge_rejected_counts(target: dict, source: dict) -> dict:
    out = dict(target)
    for key, value in source.items():
        out[key] = int(out.get(key, 0)) + int(value)
    return out


def regularize_selected_polylines(
    polylines: List[np.ndarray],
    *,
    selected_indices: List[int],
    smooth_window: int,
    max_curv: float | None,
    iterations: int,
) -> tuple[List[np.ndarray], dict]:
    if not selected_indices:
        return polylines, {"adjusted_branches": 0, "max_curvature_before": 0.0, "max_curvature_after": 0.0, "scope_count": 0}

    subset = [polylines[i] for i in selected_indices]
    reg_subset, stats = regularize_polylines_curvature(
        subset,
        smooth_window=smooth_window,
        max_curv=max_curv,
        iterations=iterations,
    )
    out = list(polylines)
    for idx, poly in zip(selected_indices, reg_subset):
        out[idx] = poly
    stats = dict(stats)
    stats["scope_count"] = int(len(selected_indices))
    return out, stats


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
    parser.add_argument(
        "--coord_mode",
        choices=["auto", "abs_spacing", "affine"],
        default="auto",
        help="Coordinate conversion between VTP world and probability voxel grid.",
    )
    parser.add_argument(
        "--align_centerline",
        action="store_true",
        default=True,
        help="Estimate and apply a global translation shift to align centerline with probability mask.",
    )
    parser.add_argument(
        "--no_align_centerline",
        dest="align_centerline",
        action="store_false",
        help="Disable global translation alignment.",
    )
    parser.add_argument(
        "--align_max_points",
        type=int,
        default=8000,
        help="Max centerline points used for alignment search.",
    )
    parser.add_argument(
        "--align_outside_penalty",
        type=float,
        default=20.0,
        help="Penalty weight for points outside volume during alignment search.",
    )
    parser.add_argument(
        "--align_steps",
        type=str,
        default="10,5,2,1,0.5,0.25",
        help="Comma-separated translation search steps in mm.",
    )
    parser.add_argument(
        "--densify_step_mm",
        type=float,
        default=0.5,
        help="Densify each output polyline so adjacent points are <= this distance (mm). <=0 disables.",
    )
    parser.add_argument(
        "--merge_point_tol",
        type=float,
        default=1e-6,
        help="Merge points closer than this tolerance (mm) when writing VTP. <=0 disables.",
    )
    parser.add_argument(
        "--bridge_min_count",
        type=int,
        default=1,
        help="If selected bridges are fewer than this, run one relaxed bridging pass.",
    )
    parser.add_argument(
        "--bridge_relax_if_none",
        action="store_true",
        default=True,
        help="Enable relaxed fallback pass when strict bridging is insufficient.",
    )
    parser.add_argument(
        "--no_bridge_relax_if_none",
        dest="bridge_relax_if_none",
        action="store_false",
        help="Disable relaxed fallback pass.",
    )
    parser.add_argument("--bridge_relax_dist_factor", type=float, default=1.5)
    parser.add_argument("--bridge_relax_angle_add", type=float, default=20.0)
    parser.add_argument("--bridge_relax_len_factor", type=float, default=1.4)
    parser.add_argument("--bridge_relax_curvature_factor", type=float, default=1.5)
    parser.add_argument("--bridge_relax_murray_add", type=float, default=0.15)
    parser.add_argument(
        "--bridge_smooth_iterations",
        type=int,
        default=4,
        help="Additional smoothing iterations for candidate bridge path before curvature rejection.",
    )
    parser.add_argument(
        "--regularize_scope",
        choices=["all", "bridges", "none"],
        default="bridges",
        help="Apply post-repair curvature regularization to all branches, bridges only, or disable it.",
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

    mask_for_align = prob >= args.prob_thresh
    polylines, branch_radii = read_vtp_centerlines(args.vtp)
    raw_points = np.vstack(polylines) if polylines else np.zeros((0, 3), dtype=float)

    if args.coord_mode == "auto":
        transform, coord_mode_info = choose_coordinate_mode(
            raw_points,
            prob=prob,
            mask=mask_for_align,
            affine=affine,
            outside_penalty=args.align_outside_penalty,
        )
    else:
        transform = make_coordinate_transform(affine, args.coord_mode)
        sc = alignment_score(
            raw_points,
            prob=prob,
            mask=mask_for_align,
            transform=transform,
            outside_penalty=args.align_outside_penalty,
        )
        coord_mode_info = {
            "selected_mode": transform.mode,
            "candidates": {
                transform.mode: {
                    "score": float(sc["score"]),
                    "inside_ratio": float(sc["inside_ratio"]),
                    "mean_prob": float(sc["mean_prob"]),
                }
            },
        }

    align_steps = tuple(float(x) for x in args.align_steps.split(",") if x.strip())
    if args.align_centerline:
        align_shift, alignment_info = estimate_alignment_shift(
            raw_points,
            prob=prob,
            mask=mask_for_align,
            transform=transform,
            outside_penalty=args.align_outside_penalty,
            max_points=args.align_max_points,
            steps_mm=align_steps,
        )
    else:
        align_shift = np.zeros(3, dtype=float)
        alignment_info = {
            "enabled": False,
            "best_shift_world": [0.0, 0.0, 0.0],
            "best_score": None,
            "inside_ratio": None,
            "mean_prob": None,
            "num_samples": int(raw_points.shape[0]),
            "steps_mm": list(align_steps),
        }

    if np.linalg.norm(align_shift) > 0:
        polylines = [poly + align_shift[None, :] for poly in polylines]

    endpoints = collect_endpoints(polylines)
    branch_lengths = [polyline_length_mm(poly) for poly in polylines]
    shape = prob.shape

    pass_summaries = []
    pair_candidates = build_pair_candidates(
        endpoints,
        branch_lengths,
        max_dist=float(args.max_dist),
        max_angle_deg=float(args.max_angle_deg),
        max_pairs=int(args.max_pairs),
    )
    valid_routes, rejected_reasons = build_valid_routes(
        pair_candidates,
        endpoints=endpoints,
        transform=transform,
        cost=cost,
        spacing=spacing,
        shape=shape,
        max_bridge_len=float(args.max_bridge_len),
        max_curvature_allowed=float(args.max_curvature) if args.max_curvature is not None else None,
        smooth_window=int(args.smooth_window),
        bridge_smooth_iterations=int(args.bridge_smooth_iterations),
        branch_radii=branch_radii,
        dist_map=dist_map,
        murray_exp=float(args.murray_exp),
        murray_tol=float(args.murray_tol) if args.murray_tol is not None else None,
    )
    pass_summaries.append(
        {
            "name": "strict",
            "candidate_pairs_prefiltered": len(pair_candidates),
            "valid_routes": len(valid_routes),
            "rejected_reasons": rejected_reasons,
            "params": {
                "max_dist": float(args.max_dist),
                "max_angle_deg": float(args.max_angle_deg),
                "max_bridge_len": float(args.max_bridge_len),
                "max_curvature": float(args.max_curvature) if args.max_curvature is not None else None,
                "murray_tol": float(args.murray_tol) if args.murray_tol is not None else None,
            },
        }
    )

    need_relaxed_pass = args.bridge_relax_if_none and (len(valid_routes) < int(args.bridge_min_count))
    if need_relaxed_pass:
        relaxed_max_dist = float(args.max_dist) * float(args.bridge_relax_dist_factor)
        relaxed_max_angle = min(180.0, float(args.max_angle_deg) + float(args.bridge_relax_angle_add))
        relaxed_max_len = float(args.max_bridge_len) * float(args.bridge_relax_len_factor)
        relaxed_max_curv = (
            float(args.max_curvature) * float(args.bridge_relax_curvature_factor)
            if args.max_curvature is not None
            else None
        )
        relaxed_murray_tol = None
        if args.murray_tol is not None:
            relaxed_murray_tol = min(1.0, float(args.murray_tol) + float(args.bridge_relax_murray_add))

        relaxed_pairs = build_pair_candidates(
            endpoints,
            branch_lengths,
            max_dist=relaxed_max_dist,
            max_angle_deg=relaxed_max_angle,
            max_pairs=int(args.max_pairs),
        )
        relaxed_routes, relaxed_rejected = build_valid_routes(
            relaxed_pairs,
            endpoints=endpoints,
            transform=transform,
            cost=cost,
            spacing=spacing,
            shape=shape,
            max_bridge_len=relaxed_max_len,
            max_curvature_allowed=relaxed_max_curv,
            smooth_window=int(args.smooth_window),
            bridge_smooth_iterations=int(args.bridge_smooth_iterations),
            branch_radii=branch_radii,
            dist_map=dist_map,
            murray_exp=float(args.murray_exp),
            murray_tol=relaxed_murray_tol,
        )
        valid_routes.extend(relaxed_routes)
        rejected_reasons = merge_rejected_counts(rejected_reasons, relaxed_rejected)
        pass_summaries.append(
            {
                "name": "relaxed",
                "candidate_pairs_prefiltered": len(relaxed_pairs),
                "valid_routes": len(relaxed_routes),
                "rejected_reasons": relaxed_rejected,
                "params": {
                    "max_dist": relaxed_max_dist,
                    "max_angle_deg": relaxed_max_angle,
                    "max_bridge_len": relaxed_max_len,
                    "max_curvature": relaxed_max_curv,
                    "murray_tol": relaxed_murray_tol,
                },
            }
        )
        valid_routes.sort(
            key=lambda c: (
                c["score"],
                c["dist_mm"],
                c["angle_deg"],
                endpoints[c["i"]].branch_id,
                endpoints[c["j"]].branch_id,
                endpoints[c["i"]].end_idx,
                endpoints[c["j"]].end_idx,
            )
        )

    used = set()
    bridges: List[np.ndarray] = []
    bridge_meta = []
    for route in valid_routes:
        i = int(route["i"])
        j = int(route["j"])
        if i in used or j in used:
            continue
        a = endpoints[i]
        b = endpoints[j]
        bridges.append(route["path_world"])
        bridge_meta.append(
            {
                "endpoint_a": {"branch": a.branch_id, "end_idx": a.end_idx, "point": a.point.tolist()},
                "endpoint_b": {"branch": b.branch_id, "end_idx": b.end_idx, "point": b.point.tolist()},
                "dist_mm": route["dist_mm"],
                "angle_deg": route["angle_deg"],
                "path_len_mm": route["path_len_mm"],
                "max_curvature": route["max_curvature"],
                "radius_a": route["radius_a"],
                "radius_b": route["radius_b"],
                "murray_deviation": route["murray_deviation"],
                "cost": route["cost"],
                "score": route["score"],
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
    if args.regularize_scope == "none":
        regularize_stats = {
            "adjusted_branches": 0,
            "max_curvature_before": 0.0,
            "max_curvature_after": 0.0,
            "scope_count": 0,
            "scope": "none",
        }
    else:
        if args.regularize_scope == "all":
            target_indices = list(range(len(out_polylines)))
        else:
            target_indices = [idx for idx, flag in enumerate(bridge_flags) if int(flag) == 1]
        out_polylines, regularize_stats = regularize_selected_polylines(
            out_polylines,
            selected_indices=target_indices,
            smooth_window=args.smooth_window,
            max_curv=reg_cap,
            iterations=args.regularize_iterations,
        )
        regularize_stats["scope"] = args.regularize_scope

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

    if args.densify_step_mm and args.densify_step_mm > 0:
        out_polylines = [densify_polyline(poly, args.densify_step_mm) for poly in out_polylines]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_topology = write_vtp_centerlines(
        out_polylines,
        args.out,
        bridge_flags,
        alignment_shift_world=align_shift,
        coordinate_mode=transform.mode,
        merge_point_tol_mm=args.merge_point_tol,
    )
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
                "coord_mode": args.coord_mode,
                "align_centerline": args.align_centerline,
                "align_max_points": args.align_max_points,
                "align_outside_penalty": args.align_outside_penalty,
                "align_steps": args.align_steps,
                "densify_step_mm": args.densify_step_mm,
                "merge_point_tol": args.merge_point_tol,
                "bridge_min_count": args.bridge_min_count,
                "bridge_relax_if_none": args.bridge_relax_if_none,
                "bridge_relax_dist_factor": args.bridge_relax_dist_factor,
                "bridge_relax_angle_add": args.bridge_relax_angle_add,
                "bridge_relax_len_factor": args.bridge_relax_len_factor,
                "bridge_relax_curvature_factor": args.bridge_relax_curvature_factor,
                "bridge_relax_murray_add": args.bridge_relax_murray_add,
                "bridge_smooth_iterations": args.bridge_smooth_iterations,
                "regularize_scope": args.regularize_scope,
            },
            "coordinate_mode_info": coord_mode_info,
            "alignment": alignment_info,
            "alignment_applied_to_output": True,
            "snap_stats": snap_stats,
            "regularize_stats": regularize_stats,
            "output_topology": output_topology,
            "removed_short_branches": removed_short,
            "candidate_stats": {
                "candidate_pairs_prefiltered": len(pair_candidates),
                "valid_routes": len(valid_routes),
                "selected_routes": len(bridges),
                "rejected_reasons": rejected_reasons,
                "passes": pass_summaries,
            },
            "bridges": bridge_meta,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report -> {args.report}")


if __name__ == "__main__":
    main()
