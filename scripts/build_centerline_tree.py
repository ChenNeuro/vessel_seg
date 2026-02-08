"""
Build a centerline tree from a VTP centerline file.

Steps:
- Read polylines from VTP (e.g., ASOCA2020/Normal/Centerlines/Normal_1.vtp).
- Extract start/end points, lengths, and point coordinates.
- Cluster branches that share the same start point (within a small threshold).
  Pick the longest in each cluster as the root of that cluster; the others
  become children at lambda=0 of that root.
- For remaining unassigned branches, attach to the already-built tree by
  projecting the branch start onto each candidate parent polyline. The best
  (closest) attachment within a distance threshold is selected.
- Compute relative position lambda (fractional arc-length along parent) and
  branching angles (theta: angle to parent tangent; phi: azimuth in parent
  normal-binormal plane).

Outputs:
- Prints a concise tree summary.
- Saves JSON to outputs/<name>_tree.json with branch info and relations.
"""

from __future__ import annotations

import json
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import vtk


ATTACH_DIST_THRESHOLD = 3.0  # mm, max distance to attach a child to a parent
START_CLUSTER_EPS = 1e-3  # mm, for trimming identical starts
ROOT_CLUSTER_EPS = 1.0  # mm, for grouping starts when defining roots
OVERLAP_DIST_EPS = 0.5  # mm, points within this to root are considered overlapping


@dataclass
class Branch:
    branch_id: int
    coords: np.ndarray  # (N, 3)
    length: float
    start: np.ndarray
    end: np.ndarray
    centroid: np.ndarray
    radius: Optional[np.ndarray] = None


@dataclass
class Attachment:
    parent: Optional[int]  # None for root
    lambda_pos: Optional[float]  # fraction along parent length
    theta_deg: Optional[float]
    phi_deg: Optional[float]


def read_branches(vtp_path: Path) -> List[Branch]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()

    points = poly.GetPoints()
    radius_arr = poly.GetPointData().GetArray("MaximumInscribedSphereRadius")
    lines = poly.GetLines()
    lines.InitTraversal()

    branches: List[Branch] = []
    cid = 0
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        if ids.GetNumberOfIds() < 2:
            cid += 1
            continue

        coords = np.array([points.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
        start = coords[0]
        end = coords[-1]
        centroid = coords.mean(axis=0)
        diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        length = float(np.sum(diffs))
        radii = (
            np.array([radius_arr.GetTuple1(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
            if radius_arr is not None
            else None
        )
        branches.append(
            Branch(
                branch_id=cid,
                coords=coords,
                length=length,
                start=start,
                end=end,
                centroid=centroid,
                radius=radii,
            )
        )
        cid += 1
    return branches


def cluster_starts(branches: List[Branch], eps: float) -> List[List[int]]:
    """Cluster branches whose start points are within eps."""
    clusters: List[List[int]] = []
    for b in branches:
        placed = False
        for c in clusters:
            ref = branches[c[0]].start
            if np.linalg.norm(b.start - ref) <= eps:
                c.append(b.branch_id)
                placed = True
                break
        if not placed:
            clusters.append([b.branch_id])
    return clusters


def project_point_to_polyline(p: np.ndarray, poly: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Project point p to polyline (Nx3).
    Returns: (s_proj, dist, proj_point, tangent_at_proj)
    s_proj: arc-length position along polyline (mm)
    dist: Euclidean distance (mm)
    proj_point: coordinates on polyline
    tangent_at_proj: unit tangent vector at projection (approximated by segment direction)
    """
    best_dist = float("inf")
    best_s = 0.0
    best_point = poly[0]
    best_tangent = np.array([1.0, 0.0, 0.0])
    s_acc = 0.0

    for i in range(len(poly) - 1):
        a = poly[i]
        b = poly[i + 1]
        seg = b - a
        seg_len2 = np.dot(seg, seg)
        if seg_len2 == 0:
            continue
        t = np.clip(np.dot(p - a, seg) / seg_len2, 0.0, 1.0)
        proj = a + t * seg
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best_s = s_acc + math.sqrt(seg_len2) * t
            best_point = proj
            if np.linalg.norm(seg) > 0:
                best_tangent = seg / np.linalg.norm(seg)
        s_acc += math.sqrt(seg_len2)

    return best_s, best_dist, best_point, best_tangent


def trim_overlap(branch: Branch, ref: Branch, eps: float) -> Optional[Branch]:
    """
    Remove the overlapping prefix of `branch` that coincides with `ref` (same start cluster).
    If the whole branch overlaps, return None (drop it).
    """
    idx_start = 0
    for i, p in enumerate(branch.coords):
        _, dist, _, _ = project_point_to_polyline(p, ref.coords)
        if dist > eps:
            idx_start = i
            break
    else:
        # Entire branch overlaps
        return None

    coords = branch.coords[idx_start:]
    if len(coords) < 2:
        return None
    radii = branch.radius[idx_start:] if branch.radius is not None else None
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    length = float(np.sum(diffs))
    if length < 1e-3:
        return None
    return Branch(
        branch_id=branch.branch_id,
        coords=coords,
        length=length,
        start=coords[0],
        end=coords[-1],
        centroid=coords.mean(axis=0),
        radius=radii,
    )


def make_basis(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct an orthonormal frame given tangent."""
    t = tangent / np.linalg.norm(tangent)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    n = np.cross(t, ref)
    n = n / np.linalg.norm(n)
    b = np.cross(t, n)
    return t, n, b


def attachment_from_projection(
    child_dir: np.ndarray, parent_tangent: np.ndarray
) -> Tuple[float, float]:
    """Compute theta/phi in degrees between child_dir and parent frame."""
    t, n, b = make_basis(parent_tangent)
    child_unit = child_dir / np.linalg.norm(child_dir)
    # Angle to tangent
    theta = math.degrees(math.acos(np.clip(np.dot(child_unit, t), -1.0, 1.0)))
    # Project onto normal-binormal plane
    proj = child_unit - np.dot(child_unit, t) * t
    if np.linalg.norm(proj) < 1e-8:
        phi = 0.0
    else:
        proj /= np.linalg.norm(proj)
        x = np.dot(proj, n)
        y = np.dot(proj, b)
        phi = math.degrees(math.atan2(y, x))
    return theta, phi


def build_tree(branches: List[Branch]):
    """Greedy tree: start from longest branch as root; attach others by nearest projection, else new root."""
    id_to_branch = {b.branch_id: b for b in branches}
    attachments = {b.branch_id: Attachment(parent=None, lambda_pos=None, theta_deg=None, phi_deg=None) for b in branches}

    # Sort branches by length (longest first)
    sorted_branches = sorted(branches, key=lambda b: b.length, reverse=True)
    if not sorted_branches:
        return [], attachments

    roots = [sorted_branches[0].branch_id]
    assigned = {sorted_branches[0].branch_id}

    for child in sorted_branches[1:]:
        best_parent = None
        best = None  # (dist, lambda_pos, theta, phi)
        for pid in assigned:
            parent = id_to_branch[pid]
            s_proj, dist, _, tangent = project_point_to_polyline(child.start, parent.coords)
            if dist <= ATTACH_DIST_THRESHOLD:
                child_dir = child.coords[1] - child.coords[0]
                if np.linalg.norm(child_dir) < 1e-8:
                    continue
                theta, phi = attachment_from_projection(child_dir, tangent)
                lambda_pos = s_proj / parent.length if parent.length > 0 else 0.0
                if (best is None) or (dist < best[0]):
                    best_parent = pid
                    best = (dist, lambda_pos, theta, phi)
        if best_parent is not None:
            attachments[child.branch_id] = Attachment(
                parent=best_parent,
                lambda_pos=best[1],
                theta_deg=best[2],
                phi_deg=best[3],
            )
            assigned.add(child.branch_id)
        else:
            roots.append(child.branch_id)
            assigned.add(child.branch_id)

    return roots, attachments


def main():
    parser = argparse.ArgumentParser(description="Build centerline tree from VTP.")
    parser.add_argument("--vtp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Output tree JSON (default outputs/<case>/tree.json)")
    parser.add_argument("--case", type=str, default="Normal_1", help="Case name for default outputs")
    args = parser.parse_args()

    out_json = args.out or Path(f"outputs/{args.case}/tree.json")
    out_json.parent.mkdir(exist_ok=True, parents=True)

    branches = read_branches(args.vtp)
    roots, attachments = build_tree(branches)

    tree = {
        "vtp": str(args.vtp),
        "roots": roots,
        "branches": [
            {
                "branch_id": b.branch_id,
                "length_mm": b.length,
                "start": b.start.tolist(),
                "end": b.end.tolist(),
                "centroid": b.centroid.tolist(),
                "attachment": asdict(attachments[b.branch_id]),
            }
            for b in branches
        ],
    }
    out_json.write_text(json.dumps(tree, indent=2))

    print(f"Roots: {roots}")
    print("Branches (id -> parent, lambda, theta, phi):")
    for b in sorted(branches, key=lambda x: x.length, reverse=True):
        att = attachments[b.branch_id]
        print(
            f"{b.branch_id:2d} len={b.length:7.2f} mm | "
            f"parent={att.parent} lambda={att.lambda_pos} "
            f"theta={att.theta_deg} phi={att.phi_deg}"
        )
    print(f"Saved tree JSON to {out_json}")


if __name__ == "__main__":
    main()
