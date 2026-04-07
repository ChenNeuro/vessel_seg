from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
import vtk


"""
Sample prior radii onto existing branch centerlines and export per-branch + merged VTP meshes.

Inputs:
  --prior fgpm_model.npz (coeffs_mean/std, tk, angles, order)
  --branches_dir outputs/<case>/branches (each branch has centerline.npy, optional radii.npz)
  --tree (optional): tree.json to get parent relationships for overlap trimming
Outputs:
  --out_dir: saves branch_*.vtp and merged.vtp
"""

def load_centerline(path: Path) -> np.ndarray:
    return np.load(path)


def resample_centerline(centerline: np.ndarray, K: int) -> np.ndarray:
    """Resample polyline to K points by arc-length."""
    pts = np.asarray(centerline, dtype=float)
    if len(pts) < 2:
        return np.repeat(pts[:1], K, axis=0)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    target = np.linspace(0.0, s[-1], K)
    resampled = []
    j = 0
    for t in target:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        if j + 1 >= len(s):
            resampled.append(pts[-1])
            continue
        w = (t - s[j]) / (s[j + 1] - s[j] + 1e-8)
        p = pts[j] * (1 - w) + pts[j + 1] * w
        resampled.append(p)
    return np.vstack(resampled)


def centerline_length(centerline: np.ndarray) -> float:
    if len(centerline) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())


def extend_centerline(centerline: np.ndarray, percent: float) -> np.ndarray:
    """Extend distal end by given percent of length along last segment direction."""
    if percent <= 0 or len(centerline) < 2:
        return centerline
    pts = np.asarray(centerline, dtype=float)
    L = centerline_length(pts)
    if L < 1e-6:
        return pts
    extend_len = L * (percent / 100.0)
    direction = pts[-1] - pts[-2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return pts
    direction = direction / norm
    new_end = pts[-1] + extend_len * direction
    return np.vstack([pts, new_end])


def fourier_reconstruct(coeffs: np.ndarray, angles: np.ndarray) -> np.ndarray:
    theta = angles.reshape(-1)
    order = (len(coeffs) - 1) // 2
    r = np.full_like(theta, coeffs[0], dtype=float)
    for n in range(1, order + 1):
        a_n = coeffs[2 * n - 1]
        b_n = coeffs[2 * n]
        r += a_n * np.cos(n * theta) + b_n * np.sin(n * theta)
    return r


def parallel_transport_frames(centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(centerline, dtype=float)
    K = len(pts)
    tangents = np.zeros((K, 3), dtype=float)
    normals = np.zeros((K, 3), dtype=float)
    for i in range(K):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == K - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        tangents[i] = t / (np.linalg.norm(t) + 1e-8)
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
        n_prev = normals[i - 1]
        n_rot = (
            n_prev * np.cos(angle)
            + np.cross(axis, n_prev) * np.sin(angle)
            + axis * np.dot(axis, n_prev) * (1 - np.cos(angle))
        )
        normals[i] = n_rot / (np.linalg.norm(n_rot) + 1e-8)
        binorm = np.cross(tangents[i], normals[i])
    return tangents, normals


def build_mesh(centerline: np.ndarray, radii: np.ndarray, angles: np.ndarray, add_caps: bool) -> vtk.vtkPolyData:
    K, M = radii.shape
    tangents, normals = parallel_transport_frames(centerline)
    points = vtk.vtkPoints()
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
            v0 = ids[k, j]
            v1 = ids[k, jp]
            v2 = ids[k + 1, jp]
            v3 = ids[k + 1, j]
            for tri in [(v0, v1, v2), (v0, v2, v3)]:
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0, tri[0])
                cell.GetPointIds().SetId(1, tri[1])
                cell.GetPointIds().SetId(2, tri[2])
                polys.InsertNextCell(cell)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    if add_caps:
        # proximal cap
        center_id_prox = points.InsertNextPoint(*centerline[0].tolist())
        for j in range(M):
            jp = (j + 1) % M
            cell = vtk.vtkTriangle()
            cell.GetPointIds().SetId(0, ids[0, j])
            cell.GetPointIds().SetId(1, ids[0, jp])
            cell.GetPointIds().SetId(2, center_id_prox)
            polys.InsertNextCell(cell)
        # distal cap
        center_id_dist = points.InsertNextPoint(*centerline[-1].tolist())
        for j in range(M):
            jp = (j + 1) % M
            cell = vtk.vtkTriangle()
            cell.GetPointIds().SetId(0, ids[K - 1, jp])
            cell.GetPointIds().SetId(1, ids[K - 1, j])
            cell.GetPointIds().SetId(2, center_id_dist)
            polys.InsertNextCell(cell)
    return polydata


def append_polys(polys: List[vtk.vtkPolyData]) -> vtk.vtkPolyData:
    append = vtk.vtkAppendPolyData()
    for p in polys:
        append.AddInputData(p)
    append.Update()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(append.GetOutput())
    cleaner.Update()
    return cleaner.GetOutput()


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def point_to_polyline_distance(p: np.ndarray, polyline: np.ndarray) -> float:
    best = float("inf")
    for i in range(len(polyline) - 1):
        d = point_to_segment_distance(p, polyline[i], polyline[i + 1])
        if d < best:
            best = d
    return best


def trim_overlap(
    child: np.ndarray,
    parent: np.ndarray,
    tol: float,
    consecutive: int = 2,
    min_keep: int = 6,
) -> np.ndarray:
    """Trim proximal child portion that lies close to parent (< tol)."""
    if len(child) < min_keep or len(parent) < 2:
        return child
    dists = [point_to_polyline_distance(p, parent) for p in child]
    start_idx = 0
    window = max(1, consecutive)
    for i in range(0, len(child) - window + 1):
        if all(d > tol for d in dists[i : i + window]):
            start_idx = i
            break
    trimmed = child[start_idx:]
    if len(trimmed) < min_keep:
        return child
    return trimmed


def load_parent_map(tree_path: Optional[Path]) -> Dict[int, Optional[int]]:
    if tree_path is None:
        return {}
    data = json.loads(tree_path.read_text())
    parent_map: Dict[int, Optional[int]] = {}
    for b in data.get("branches", []):
        bid = int(b["branch_id"])
        parent = b.get("attachment", {}).get("parent", None)
        parent_map[bid] = None if parent is None else int(parent)
    return parent_map


def parse_branch_id(branch_dir: Path) -> Optional[int]:
    try:
        return int(branch_dir.name.split("_")[1])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Sample prior radii onto branch centerlines and export VTP.")
    parser.add_argument("--prior", type=Path, required=True, help="fgpm_model.npz")
    parser.add_argument("--branches_dir", type=Path, required=True, help="Dir containing branch_<id>/centerline.npy (and optional radii.npz)")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--noise_scale", type=float, default=0.3, help="Std multiplier for coeff sampling")
    parser.add_argument("--scale_from_data", action="store_true", help="Scale proximal radius to match branch data if available")
    parser.add_argument("--K", type=int, default=None, help="Override number of samples along centerline (default: use prior K)")
    parser.add_argument("--add_caps", action="store_true", help="Add end caps to close vessel ends")
    parser.add_argument("--extend_short_percent", type=float, default=10.0, help="Extend short branches by this percent of their length (default 10%).")
    parser.add_argument("--extend_long_percent", type=float, default=0.0, help="Extend long branches by this percent (default 0%).")
    parser.add_argument("--length_quantile", type=float, default=0.33, help="Quantile to split short vs long (default 0.33).")
    parser.add_argument("--tree", type=Path, default=None, help="tree.json to provide parent relationships for overlap trimming.")
    parser.add_argument("--overlap_tol", type=float, default=1.0, help="Distance (mm) threshold to stop trimming child branch.")
    parser.add_argument("--overlap_consecutive", type=int, default=2, help="Consecutive points above tol to declare divergence.")
    parser.add_argument("--min_keep", type=int, default=6, help="Minimum points to keep after trimming.")
    args = parser.parse_args()

    prior = np.load(args.prior, allow_pickle=True)
    coeffs_mean = prior["coeffs_mean"]
    coeffs_std = prior["coeffs_std"]
    tk_prior = prior["tk"]
    angles = prior["angles"]
    order = int(prior["order"])
    K = args.K or coeffs_mean.shape[0]
    C = coeffs_mean.shape[1]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    branch_dirs = sorted(args.branches_dir.glob("branch_*"))
    lengths = {}
    parent_map = load_parent_map(args.tree)

    for branch_dir in branch_dirs:
        cl_path = branch_dir / "centerline.npy"
        if not cl_path.exists():
            continue
        centerline = load_centerline(cl_path)
        lengths[branch_dir.name] = centerline_length(centerline)

    # length threshold
    if lengths:
        len_vals = np.array(list(lengths.values()), dtype=float)
        thresh = float(np.quantile(len_vals, args.length_quantile))
    else:
        thresh = 0.0

    meshes = []
    for branch_dir in branch_dirs:
        cl_path = branch_dir / "centerline.npy"
        if not cl_path.exists():
            continue
        centerline = load_centerline(cl_path)
        bid = parse_branch_id(branch_dir)
        # trim overlap with parent if available
        if bid is not None and bid in parent_map and parent_map[bid] is not None:
            parent_id = parent_map[bid]
            parent_cl = args.branches_dir / f"branch_{parent_id}" / "centerline.npy"
            if parent_cl.exists():
                parent_centerline = load_centerline(parent_cl)
                trimmed = trim_overlap(
                    centerline,
                    parent_centerline,
                    tol=args.overlap_tol,
                    consecutive=args.overlap_consecutive,
                    min_keep=args.min_keep,
                )
                if len(trimmed) != len(centerline):
                    print(f"Trimmed overlap for branch {bid}: {len(centerline)} -> {len(trimmed)} points")
                centerline = trimmed

        L = centerline_length(centerline)
        percent = args.extend_short_percent if L <= thresh else args.extend_long_percent
        centerline_ext = extend_centerline(centerline, percent=percent)
        centerline_rs = resample_centerline(centerline_ext, K)
        # scale to data
        scale = 1.0
        if args.scale_from_data and (branch_dir / "radii.npz").exists():
            data = np.load(branch_dir / "radii.npz")
            r_branch = data["radii"]
            if r_branch.size > 0:
                scale = float(np.nanmean(r_branch[0]))
        # sample coeffs
        # interp coeffs_mean/std to new K if overridden
        if K != coeffs_mean.shape[0]:
            xi_prior = np.linspace(0, 1, coeffs_mean.shape[0])
            xi_new = np.linspace(0, 1, K)
            coeffs_mean_interp = np.array([np.interp(xi_new, xi_prior, coeffs_mean[:, j]) for j in range(C)]).T
            coeffs_std_interp = np.array([np.interp(xi_new, xi_prior, coeffs_std[:, j]) for j in range(C)]).T
        else:
            coeffs_mean_interp = coeffs_mean
            coeffs_std_interp = coeffs_std
        noise = np.random.randn(K, C) * args.noise_scale * coeffs_std_interp
        coeffs = coeffs_mean_interp + noise
        radii = np.zeros((K, len(angles)), dtype=float)
        for k in range(K):
            radii[k] = fourier_reconstruct(coeffs[k], angles)
        # scale proximal
        if scale != 1.0:
            src = np.nanmean(radii[0])
            if src > 1e-6:
                radii *= (scale / src)
        mesh = build_mesh(centerline_rs, radii, angles, add_caps=args.add_caps)
        out_vtp = args.out_dir / f"{branch_dir.name}.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(out_vtp))
        writer.SetInputData(mesh)
        writer.Write()
        meshes.append(mesh)
        print(f"Wrote {out_vtp} | points={mesh.GetNumberOfPoints()}")

    if meshes:
        merged = append_polys(meshes)
        merged_vtp = args.out_dir / "merged.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(merged_vtp))
        writer.SetInputData(merged)
        writer.Write()
        print(f"Wrote merged mesh to {merged_vtp} | points={merged.GetNumberOfPoints()} polys={merged.GetNumberOfPolys()}")


if __name__ == "__main__":
    main()
