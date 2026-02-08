"""
Build branch-wise training samples (meta + radius tensor) from centerlines and mask.

For each branch in a VTP centerline file:
1) Resample centerline to K points evenly in arc-length (normalized t in [0,1]).
2) At each resampled point, construct a local orthonormal frame (tangent T, normals N,B).
3) Sample M angular directions in the normal-binormal plane and cast rays into the mask to
   measure radius = distance to mask boundary along that direction.
4) Save meta (branch_id, length_mm, parent_id, lambda, tk, angles) and radius tensor Rb∈R^{K×M}.

Outputs:
- A .npz file with arrays: radii (B,K,M), tk (K,), angles (M,), meta (list of dicts).

Usage:
  conda activate vessel_seg
  python scripts/build_branch_dataset.py \
      --vtp ASOCA2020/Normal/Centerlines/Normal_1.vtp \
      --mask ASOCA2020/Normal/Annotations/Normal_1.nrrd \
      --tree outputs/normal1_tree.json \
      --out outputs/normal1_branch_dataset.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import vtk


def read_centerlines(vtp_path: Path) -> Dict[int, np.ndarray]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    lines = poly.GetLines()
    lines.InitTraversal()
    branches = {}
    cid = 0
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        if ids.GetNumberOfIds() < 2:
            cid += 1
            continue
        coords = np.array([pts.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
        branches[cid] = coords
        cid += 1
    return branches


def resample_polyline(coords: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample polyline to k points evenly spaced in arc-length. Returns (points, t in [0,1])."""
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(diffs)])
    if s[-1] == 0:
        return np.repeat(coords[:1], k, axis=0), np.linspace(0, 1, k)
    target_s = np.linspace(0, s[-1], k)
    resampled = []
    j = 0
    for ts in target_s:
        while j + 1 < len(s) and s[j + 1] < ts:
            j += 1
        if j + 1 >= len(s):
            resampled.append(coords[-1])
            continue
        ratio = (ts - s[j]) / (s[j + 1] - s[j] + 1e-8)
        p = coords[j] * (1 - ratio) + coords[j + 1] * ratio
        resampled.append(p)
    resampled = np.vstack(resampled)
    t = target_s / (s[-1] + 1e-8)
    return resampled, t


def make_frame(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = tangent / (np.linalg.norm(tangent) + 1e-8)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    n = np.cross(t, ref)
    n = n / (np.linalg.norm(n) + 1e-8)
    b = np.cross(t, n)
    b = b / (np.linalg.norm(b) + 1e-8)
    return t, n, b


def ray_radius_phys(img: sitk.Image, mask_arr: np.ndarray, p: np.ndarray, u: np.ndarray, max_r_mm: float = 20.0) -> float:
    """Cast ray in physical space; return distance to exit mask."""
    spacing = np.array(img.GetSpacing())
    step = min(spacing) * 0.5
    r = 0.0
    while r < max_r_mm:
        pos = p + r * u
        idx = img.TransformPhysicalPointToIndex(tuple(pos.tolist()))
        if any([idx[d] < 0 or idx[d] >= img.GetSize()[d] for d in range(3)]):
            break
        if mask_arr[idx[2], idx[1], idx[0]] <= 0:
            break
        r += step
    return r


def trim_start(coords: np.ndarray, offset_mm: float, offset_percent: float | None = None) -> Tuple[np.ndarray, float]:
    """Trim branch start by fixed mm or percent. Returns trimmed coords and the applied mm offset."""
    if offset_percent is not None and offset_percent > 0:
        diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total_len = float(np.sum(diffs))
        offset_mm = max(offset_mm, total_len * offset_percent)
    if offset_mm <= 0:
        return coords, 0.0
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(diffs)])
    if s[-1] <= offset_mm:
        return coords[-1:][:1], offset_mm
    # find segment where offset lies
    j = np.searchsorted(s, offset_mm) - 1
    j = max(0, min(j, len(diffs) - 1))
    ratio = (offset_mm - s[j]) / (diffs[j] + 1e-8)
    new_start = coords[j] * (1 - ratio) + coords[j + 1] * ratio
    return np.vstack([new_start, coords[j + 1 :]]), offset_mm


def align_centerline_to_mask(
    coords_all: np.ndarray,
    img: sitk.Image,
    max_points: int = 8000,
    penalty_outside: float = 30.0,
    steps: Tuple[float, ...] = (10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1),
) -> Tuple[np.ndarray, Dict]:
    """Find translation that maximises overlap with the mask using a simple coordinate descent on distance map."""
    mask = img > 0
    dist_img = sitk.SignedMaurerDistanceMap(
        mask, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    dist_arr = sitk.GetArrayFromImage(dist_img)  # z,y,x
    size = np.array(img.GetSize())
    mask_min = np.array(img.TransformIndexToPhysicalPoint((0, 0, 0)))
    mask_max = np.array(img.TransformIndexToPhysicalPoint(tuple((size - 1).tolist())))
    mask_center = (mask_min + mask_max) / 2

    cl_center = (coords_all.min(axis=0) + coords_all.max(axis=0)) / 2
    init_shift = mask_center - cl_center

    sample = coords_all
    if max_points and coords_all.shape[0] > max_points:
        idx = np.linspace(0, coords_all.shape[0] - 1, num=max_points, dtype=int)
        sample = coords_all[idx]

    def score(shift: np.ndarray) -> Tuple[float, int]:
        pts = sample + shift
        penalty = 0.0
        inside = 0
        for p in pts:
            idx = img.TransformPhysicalPointToIndex(tuple(p))
            if all(0 <= idx[d] < size[d] for d in range(3)):
                d = float(dist_arr[idx[2], idx[1], idx[0]])
                if d > 0:
                    penalty += d
                else:
                    inside += 1
            else:
                penalty += penalty_outside
        return penalty / len(sample), inside

    best_shift = init_shift.copy()
    best_score, best_inside = score(best_shift)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for axis in range(3):
                for delta in (-step, step):
                    trial = best_shift.copy()
                    trial[axis] += delta
                    sc, inside = score(trial)
                    if sc < best_score:
                        best_score = sc
                        best_shift = trial
                        best_inside = inside
                        improved = True

    info = {
        "mask_center": mask_center.tolist(),
        "cl_center": cl_center.tolist(),
        "init_shift": init_shift.tolist(),
        "best_shift": best_shift.tolist(),
        "best_score_mean_penalty": float(best_score),
        "inside_samples": int(best_inside),
        "num_samples": int(sample.shape[0]),
        "num_points": int(coords_all.shape[0]),
        "steps": list(steps),
        "mode": "distance_map_search",
    }
    return best_shift, info


def sample_branch_radii(coords: np.ndarray, img: sitk.Image, mask_arr: np.ndarray, shift: np.ndarray, k: int, m: int, max_r: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return radii tensor (k,m), tk (k,), angles (m,), resampled pts (k,3) in physical space."""
    coords_shift = coords + shift
    pts, tk = resample_polyline(coords_shift, k)
    diffs = np.diff(pts, axis=0, prepend=pts[:1])
    # angles
    angles = np.linspace(0, 2 * math.pi, m, endpoint=False)
    radii = np.zeros((k, m), dtype=np.float32)
    spacing = np.array(img.GetSpacing())

    for i in range(k):
        tangent = diffs[i]
        if np.linalg.norm(tangent) < 1e-6 and i + 1 < k:
            tangent = pts[i + 1] - pts[i]
        t, n, b = make_frame(tangent)
        for j, ang in enumerate(angles):
            u = math.cos(ang) * n + math.sin(ang) * b
            radii[i, j] = ray_radius_phys(img, mask_arr, pts[i], u, max_r_mm=max_r)
    return radii, tk, angles, pts


def load_tree_meta(tree_json: Path) -> Dict[int, Dict]:
    if not tree_json.exists():
        return {}
    data = json.loads(tree_json.read_text())
    meta = {}
    for b in data.get("branches", []):
        meta[b["branch_id"]] = b["attachment"]
    return meta


def main():
    parser = argparse.ArgumentParser(description="Build branch dataset (meta + radii tensor).")
    parser.add_argument("--vtp", type=Path, required=True, help="Centerline VTP")
    parser.add_argument("--mask", type=Path, required=True, help="Segmentation mask (NRRD/NIfTI)")
    parser.add_argument("--tree", type=Path, default=None, help="Tree JSON with parent/lambda (optional)")
    parser.add_argument("--out", type=Path, default=None, help="Output npz (default outputs/<case>/branch_dataset.npz)")
    parser.add_argument("--branch_dir", type=Path, default=None, help="Optional per-branch output dir (default outputs/<case>/branches)")
    parser.add_argument("--case", type=str, default="Normal_1", help="Case name for default outputs")
    parser.add_argument("--K", type=int, default=32, help="Number of arc-length samples per branch")
    parser.add_argument("--M", type=int, default=32, help="Number of angle samples per section")
    parser.add_argument("--max_r", type=float, default=20.0, help="Max search radius in mm")
    parser.add_argument("--start_offset", type=float, default=3.0, help="Trim this length (mm) from branch start to avoid node artifacts")
    parser.add_argument("--start_offset_percent", type=float, default=None, help="Trim this fraction (e.g., 2.5 for 2.5%% of branch length). Applied on top of --start_offset.")
    parser.add_argument("--align", action="store_true", help="Search translation to align centerline to mask (distance-map guided).")
    parser.add_argument("--no_align", dest="align", action="store_false")
    parser.set_defaults(align=True)
    parser.add_argument("--align_max_points", type=int, default=8000, help="Max points sampled for alignment search (speed).")
    args = parser.parse_args()

    out_path = args.out or Path(f"outputs/{args.case}/branch_dataset.npz")
    branch_dir = args.branch_dir or Path(f"outputs/{args.case}/branches")

    branches = read_centerlines(args.vtp)
    img = sitk.ReadImage(str(args.mask))
    mask_arr = sitk.GetArrayFromImage(img)
    tree_meta = load_tree_meta(args.tree) if args.tree else {}
    tk_default = np.linspace(0, 1, args.K)
    angles_default = np.linspace(0, 2 * math.pi, args.M, endpoint=False)

    radii_list = []
    meta_list = []
    bid_map = []

    # alignment: distance-map search for translation
    all_pts = np.vstack(list(branches.values()))
    if args.align:
        shift_global, align_info = align_centerline_to_mask(
            all_pts, img, max_points=args.align_max_points
        )
        print(
            f"[align] shift={shift_global} | samples={align_info['num_samples']} "
            f"inside={align_info['inside_samples']}/{align_info['num_samples']} "
            f"score={align_info['best_score_mean_penalty']:.4f}"
        )
    else:
        zz, yy, xx = np.argwhere(mask_arr > 0).T
        origin = np.array(img.GetOrigin())
        spacing = np.array(img.GetSpacing())
        mask_min_phys = origin + spacing * np.array([xx.min(), yy.min(), zz.min()])
        mask_max_phys = origin + spacing * np.array([xx.max(), yy.max(), zz.max()])
        mask_center = (mask_min_phys + mask_max_phys) / 2
        cl_center_global = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2
        shift_global = mask_center - cl_center_global
        align_info = {
            "mask_center": mask_center.tolist(),
            "cl_center": cl_center_global.tolist(),
            "init_shift": shift_global.tolist(),
            "best_shift": shift_global.tolist(),
            "best_score_mean_penalty": None,
            "inside_samples": None,
            "num_samples": int(all_pts.shape[0]),
            "num_points": int(all_pts.shape[0]),
            "steps": [],
            "mode": "center_alignment",
        }

    for bid, coords in branches.items():
        trimmed, applied_offset_mm = trim_start(
            coords, args.start_offset, offset_percent=(args.start_offset_percent / 100.0) if args.start_offset_percent else None
        )
        if len(trimmed) < 2:
            continue
        radii, tk, angles, pts_rs = sample_branch_radii(trimmed, img, mask_arr, shift_global, k=args.K, m=args.M, max_r=args.max_r)
        length = float(np.sum(np.linalg.norm(np.diff(trimmed, axis=0), axis=1)))
        att = tree_meta.get(bid, {"parent": None, "lambda_pos": None})
        meta_list.append(
            {
                "branch_id": bid,
                "length_mm": length,
                "parent": att.get("parent"),
                "lambda": att.get("lambda_pos"),
                "start_offset_mm": applied_offset_mm,
                "start_offset_percent": args.start_offset_percent,
            }
        )
        radii_list.append(radii)
        bid_map.append(bid)
        # per-branch export
        if branch_dir:
            bdir = branch_dir / f"branch_{bid}"
            bdir.mkdir(parents=True, exist_ok=True)
            np.save(bdir / "centerline.npy", trimmed + shift_global)
            np.savez(
                bdir / "radii.npz",
                radii=radii,
                tk=tk,
                angles=angles,
                length_mm=length,
                parent=att.get("parent"),
                lambda_pos=att.get("lambda_pos"),
                start_offset_mm=applied_offset_mm,
                start_offset_percent=args.start_offset_percent,
                alignment_shift=shift_global,
            )

    radii_arr = np.stack(radii_list, axis=0) if radii_list else np.zeros((0, args.K, args.M), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tk_save = tk if radii_list else tk_default
    angles_save = angles if radii_list else angles_default
    np.savez(
        out_path,
        radii=radii_arr,
        tk=tk_save,
        angles=angles_save,
        meta=np.array(meta_list, dtype=object),
        branch_ids=np.array(bid_map),
        alignment_shift=shift_global,
        alignment_info=np.array([align_info], dtype=object),
    )
    (out_path.parent / "alignment.json").write_text(json.dumps(align_info, indent=2))
    print(f"Saved branch dataset to {out_path} | branches={len(bid_map)}, K={args.K}, M={args.M}")


if __name__ == "__main__":
    main()
