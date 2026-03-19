"""Clinical vessel completion workflow: centerline, wall features, modeling, longitudinal sections, and dashboard.

This module ties together existing coronary geometry tooling into a single,
maintainable workflow focused on three deliverables:

1. Vessel completion from mask/CT:
   segmentation -> centerline branches -> wall polar features -> modeled meshes
2. Longitudinal vessel sections:
   centerline-aligned slab views along selected branches
3. Common coronary measurements:
   length, diameter, area, tortuosity, curvature, stenosis proxies, and wall HU

All stable logic lives here; scripts should be thin wrappers around this module.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from vessel_seg.shape import (
    BranchProfile,
    Branch,
    CentrelineParams,
    ProfileParams,
    ReconstructionParams,
    _arclength,
    _compute_length,
    _deduplicate_points,
    _interpolate_profile_matrix,
    _local_frames,
    _resample_curve,
    _resample_polar_profiles,
    compute_polar_profiles,
    export_branch_features,
    extract_branches,
)


SMALL_EPS = 1e-8


@dataclass
class ClinicalCompletionConfig:
    case_id: str
    ct_path: Path
    mask_path: Path
    out_dir: Path
    centerline_backend: str = "vessel_sort"
    min_length_mm: float = 10.0
    max_components: int = 2
    closing_iterations: int = 2
    smooth_sigma_mm: float = 0.8
    adaptive_min_step_mm: float = 0.6
    adaptive_max_step_mm: float = 2.5
    adaptive_curvature_alpha: float = 2.0
    num_samples: int = 120
    num_angle_bins: int = 72
    patch_radius_mm: float = 3.5
    half_thickness_mm: float = 1.2
    radius_clip_factor: float = 1.8
    endpoint_trim_mm: float = 1.5
    junction_distance_mm: float = 1.5
    junction_inherit_mm: float = 1.5
    pca_components: int = 8
    confidence_threshold: float = 0.4
    mesh_target_samples: int | None = 140
    mesh_smoothing: float = 0.0
    mesh_min_valid: int = 3
    mesh_interp_kind: str = "cubic"
    mesh_angular_upsample: int = 3
    mesh_angular_smoothing: float = 1.0
    mesh_min_radius_ratio: float = 0.08
    mesh_angular_gap_fill: int = 2
    mesh_axial_gap_fill: int = 1
    top_unfold_branches: int = 3
    longitudinal_width_mm: float = 8.0
    longitudinal_slab_mm: float = 2.0
    longitudinal_out_width_px: int = 96
    wall_sample_relative_radii: tuple[float, ...] = (0.7, 0.85, 1.0)
    high_hu_threshold: float = 350.0


@dataclass
class BranchMetric:
    branch_name: str
    length_mm: float
    mean_radius_mm: float
    min_radius_mm: float
    max_radius_mm: float
    mean_diameter_mm: float
    min_diameter_mm: float
    max_diameter_mm: float
    mean_area_mm2: float
    min_area_mm2: float
    tortuosity_index: float
    curvature_mean: float
    curvature_max: float
    proximal_reference_radius_mm: float
    stenosis_pct: float
    eccentricity_mean: float
    branch_confidence: float
    wall_hu_mean: float | None
    wall_hu_std: float | None
    high_hu_wall_fraction: float | None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _infer_case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _import_vessel_sort_segmentation():
    vessel_sort_root = Path(__file__).resolve().parents[2] / "vessel_sort"
    if not vessel_sort_root.exists():
        raise FileNotFoundError(f"vessel_sort repository not found: {vessel_sort_root}")
    if str(vessel_sort_root) not in sys.path:
        sys.path.insert(0, str(vessel_sort_root))
    return importlib.import_module("segmentation")


def _extract_branches_via_shape(mask_path: Path, params: CentrelineParams) -> tuple[list[Branch], dict[str, Any]]:
    branches = extract_branches(mask_path, params)
    info = {
        "backend": "shape",
        "branch_count": len(branches),
        "branch_names": [branch.name for branch in branches],
        "lengths_mm": [float(branch.length_mm) for branch in branches],
    }
    return branches, info


def _extract_branches_via_vessel_sort(
    mask_path: Path,
    case_id: str,
    params: CentrelineParams,
    output_dir: Path,
) -> tuple[list[Branch], dict[str, Any]]:
    segmentation = _import_vessel_sort_segmentation()
    tree_module = importlib.import_module("segmentation.tree")
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_path = output_dir / "tree.json"
    branches_dir = output_dir / "branches"
    observed_branches = extract_branches(mask_path, params)
    sort_branches = {}
    for idx, branch in enumerate(observed_branches):
        coords = np.asarray(branch.world_points, dtype=np.float64)
        coords = _deduplicate_points(coords)
        if coords.shape[0] < 2:
            continue
        sort_branches[idx] = tree_module.Branch(
            branch_id=idx,
            coords=coords,
            length_mm=_compute_length(coords),
            start_mm=coords[0],
            end_mm=coords[-1],
        )

    sort_branches, dropped_start = segmentation.trim_overlap_start_clusters(
        sort_branches,
        start_cluster_eps=1e-3,
        overlap_eps=0.5,
        method="auto",
        min_run=3,
    )
    tree = segmentation.build_tree_from_branches_vessel_seg(
        sort_branches,
        case_id=case_id,
        vtp_path=mask_path,
        attach_dist_mm=3.0,
    )
    trimmed, dropped_overlap = segmentation.trim_overlap_branches(
        tree,
        overlap_eps=0.5,
        method="auto",
        min_run=3,
    )
    tree = segmentation.build_tree_from_branches_vessel_seg(
        trimmed,
        case_id=case_id,
        vtp_path=mask_path,
        attach_dist_mm=3.0,
    )
    segmentation.save_tree_json(tree, tree_path)
    segmentation.export_branches(tree, branches_dir, tree_json=tree_path)

    mask_image = nib.load(str(mask_path))
    inv_affine = np.linalg.inv(mask_image.affine)
    sorted_branches = sorted(tree.branches.values(), key=lambda branch: (-branch.length_mm, branch.branch_id))

    shape_branches: list[Branch] = []
    branch_records: list[dict[str, Any]] = []
    for branch in sorted_branches:
        edge = tree.edges.get(branch.branch_id)
        name = f"Branch_{int(branch.branch_id):02d}"
        coords = _deduplicate_points(np.asarray(branch.coords, dtype=np.float64))
        if coords.shape[0] < 2:
            continue
        voxel_points = nib.affines.apply_affine(inv_affine, coords).astype(np.float64)
        shape_branch = Branch(
            name=name,
            world_points=coords,
            voxel_points=voxel_points,
            length_mm=_compute_length(coords),
            source="vessel_sort",
        )
        if shape_branch.length_mm <= 0:
            continue
        shape_branches.append(shape_branch)
        branch_records.append(
            {
                "branch_id": int(branch.branch_id),
                "name": name,
                "length_mm": float(shape_branch.length_mm),
                "parent_id": None if edge is None else edge.parent_id,
                "lambda_pos": None if edge is None else edge.lambda_pos,
                "theta_deg": None if edge is None else edge.theta_deg,
                "phi_deg": None if edge is None else edge.phi_deg,
            }
        )

    info = {
        "backend": "vessel_sort",
        "branch_count": len(shape_branches),
        "branch_names": [branch.name for branch in shape_branches],
        "lengths_mm": [float(branch.length_mm) for branch in shape_branches],
        "tree_json": str(tree_path),
        "branches_dir": str(branches_dir),
        "source_extractor": "vessel_seg.shape.extract_branches",
        "source_branch_count": len(observed_branches),
        "coordinate_transform": "native nibabel RAS world coordinates",
        "dropped_start_clusters": {str(key): int(value) for key, value in dropped_start.items()},
        "dropped_parent_overlap": {str(key): int(value) for key, value in dropped_overlap.items()},
        "branch_records": branch_records,
    }
    return shape_branches, info


def _compute_curvature(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    curvature = np.zeros(n, dtype=np.float64)
    if n < 3:
        return curvature
    for idx in range(1, n - 1):
        v1 = pts[idx] - pts[idx - 1]
        v2 = pts[idx + 1] - pts[idx]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < SMALL_EPS or norm2 < SMALL_EPS:
            continue
        cross = np.linalg.norm(np.cross(v1, v2))
        denom = max(norm1 * norm2 * np.linalg.norm(v1 + v2), SMALL_EPS)
        curvature[idx] = 2.0 * cross / denom
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def _slice_area_mm2(radii: np.ndarray) -> float:
    row = np.asarray(radii, dtype=np.float64)
    if row.size < 3 or not np.any(row > 0):
        return 0.0
    dtheta = 2.0 * math.pi / float(row.size)
    return float(0.5 * np.sum(np.square(np.clip(row, 0.0, None))) * dtheta)


def _sample_trilinear_affine(volume: np.ndarray, inv_affine: np.ndarray, world_point: np.ndarray) -> float | None:
    ijk = nib.affines.apply_affine(inv_affine, np.asarray(world_point, dtype=np.float64))
    i, j, k = ijk
    if (
        i < 0
        or j < 0
        or k < 0
        or i > volume.shape[0] - 1
        or j > volume.shape[1] - 1
        or k > volume.shape[2] - 1
    ):
        return None

    i0 = int(np.floor(i))
    j0 = int(np.floor(j))
    k0 = int(np.floor(k))
    i1 = min(i0 + 1, volume.shape[0] - 1)
    j1 = min(j0 + 1, volume.shape[1] - 1)
    k1 = min(k0 + 1, volume.shape[2] - 1)
    di = float(i - i0)
    dj = float(j - j0)
    dk = float(k - k0)

    c000 = float(volume[i0, j0, k0])
    c001 = float(volume[i0, j0, k1])
    c010 = float(volume[i0, j1, k0])
    c011 = float(volume[i0, j1, k1])
    c100 = float(volume[i1, j0, k0])
    c101 = float(volume[i1, j0, k1])
    c110 = float(volume[i1, j1, k0])
    c111 = float(volume[i1, j1, k1])

    c00 = c000 * (1.0 - dk) + c001 * dk
    c01 = c010 * (1.0 - dk) + c011 * dk
    c10 = c100 * (1.0 - dk) + c101 * dk
    c11 = c110 * (1.0 - dk) + c111 * dk
    c0 = c00 * (1.0 - dj) + c01 * dj
    c1 = c10 * (1.0 - dj) + c11 * dj
    return float(c0 * (1.0 - di) + c1 * di)


def _fit_pca_model(
    branch_profiles: Iterable[BranchProfile],
    *,
    num_components: int,
    confidence_threshold: float,
) -> dict[str, np.ndarray]:
    slices: list[np.ndarray] = []
    for profile in branch_profiles:
        mask = profile.slice_confidence >= confidence_threshold
        if not np.any(mask):
            continue
        samples = profile.normalized_profiles[mask]
        valid = samples[(samples > 0).any(axis=1)]
        if valid.size:
            slices.append(valid)

    if not slices:
        raise RuntimeError(
            "No cross-sections met the confidence threshold. Lower --confidence-threshold or adjust sampling."
        )

    data = np.vstack(slices).astype(np.float32)
    mean = data.mean(axis=0)
    centred = data - mean
    _, s, vh = np.linalg.svd(centred, full_matrices=False)
    components = vh[: num_components].astype(np.float32)
    denom = max(data.shape[0] - 1, 1)
    explained_variance = (s**2) / denom
    total = float(np.sum(explained_variance))
    if total <= 0:
        explained_ratio = np.zeros(components.shape[0], dtype=np.float32)
    else:
        explained_ratio = (explained_variance[: components.shape[0]] / total).astype(np.float32)
    return {
        "mean": mean.astype(np.float32),
        "components": components,
        "explained_variance_ratio": explained_ratio,
        "confidence_threshold": np.float32(confidence_threshold),
    }


def _reconstruct_profiles_from_pca(
    branch_profiles: Iterable[BranchProfile],
    model: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    mean = model["mean"]
    components = model["components"]
    threshold = float(model["confidence_threshold"])
    num_components = int(components.shape[0])

    reconstructions: dict[str, dict[str, np.ndarray]] = {}
    for profile in branch_profiles:
        normalized = profile.normalized_profiles.astype(np.float32)
        recon_norm = normalized.copy()
        coeffs = np.zeros((normalized.shape[0], num_components), dtype=np.float32)

        mask = profile.slice_confidence >= threshold
        if np.any(mask):
            samples = normalized[mask]
            centred = samples - mean
            projected = centred @ components.T
            rebuilt = projected @ components + mean
            rebuilt = np.clip(rebuilt, 0.0, None)
            recon_norm[mask] = rebuilt
            coeffs[mask] = projected

        recon_raw = recon_norm * profile.mean_radius[:, None]
        mse = float(np.mean((recon_raw - profile.raw_profiles) ** 2))
        mae = float(np.mean(np.abs(recon_raw - profile.raw_profiles)))

        reconstructions[profile.branch.name] = {
            "normalized": recon_norm.astype(np.float32),
            "raw": recon_raw.astype(np.float32),
            "coefficients": coeffs,
            "mse": np.float32(mse),
            "mae": np.float32(mae),
        }
    return reconstructions


def _save_modeled_features(
    branch_profiles: Sequence[BranchProfile],
    reconstructions: Mapping[str, Mapping[str, np.ndarray]],
    model: Mapping[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_summary: list[dict[str, Any]] = []
    feature_matrix: list[np.ndarray] = []

    for profile in branch_profiles:
        branch_name = profile.branch.name
        recon = reconstructions[branch_name]
        branch_path = output_dir / f"{branch_name}.npz"
        np.savez(
            branch_path,
            branch_name=branch_name,
            samples_world=profile.samples_world.astype(np.float32),
            samples_voxel=profile.samples_voxel.astype(np.float32),
            tangents=profile.tangents.astype(np.float32),
            normals=profile.normals.astype(np.float32),
            binormals=profile.binormals.astype(np.float32),
            raw_profiles=recon["raw"].astype(np.float32),
            normalized_profiles=recon["normalized"].astype(np.float32),
            original_raw_profiles=profile.raw_profiles.astype(np.float32),
            original_normalized_profiles=profile.normalized_profiles.astype(np.float32),
            mean_radius=profile.mean_radius.astype(np.float32),
            slice_confidence=profile.slice_confidence.astype(np.float32),
            branch_confidence=np.float32(profile.branch_confidence),
            feature_vector=profile.feature_vector.astype(np.float32),
            angles=profile.angles.astype(np.float32),
            length_mm=np.float32(profile.branch.length_mm),
            pca_coefficients=recon["coefficients"],
            reconstruction_mse=np.float32(recon["mse"]),
            reconstruction_mae=np.float32(recon["mae"]),
        )
        branch_summary.append(
            {
                "name": branch_name,
                "length_mm": float(profile.branch.length_mm),
                "confidence": float(profile.branch_confidence),
                "feature_file": branch_path.name,
                "reconstruction_mse": float(recon["mse"]),
                "reconstruction_mae": float(recon["mae"]),
            }
        )
        feature_matrix.append(profile.feature_vector.astype(np.float32))

    feature_matrix_array = np.vstack(feature_matrix) if feature_matrix else np.zeros((0, 1), dtype=np.float32)
    global_descriptor = (
        feature_matrix_array.mean(axis=0) if feature_matrix_array.size else np.zeros(1, dtype=np.float32)
    )
    np.save(output_dir / "global_descriptor.npy", global_descriptor.astype(np.float32))
    np.savez(
        output_dir / "pca_model.npz",
        mean=model["mean"],
        components=model["components"],
        explained_variance_ratio=model["explained_variance_ratio"],
        confidence_threshold=model["confidence_threshold"],
    )
    summary = {
        "branch_count": len(branch_summary),
        "branches": branch_summary,
        "global_descriptor": "global_descriptor.npy",
        "model": {
            "components": int(model["components"].shape[0]),
            "explained_variance_ratio": model["explained_variance_ratio"].tolist(),
            "confidence_threshold": float(model["confidence_threshold"]),
            "model_file": "pca_model.npz",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _reconstruct_single_branch_mesh(
    branch_file: Path,
    recon_params: ReconstructionParams,
):
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyvista is required for branch mesh reconstruction.") from exc

    data = np.load(branch_file)
    samples_world = data["samples_world"]
    tangents = data.get("tangents")
    normals = data.get("normals")
    binormals = data.get("binormals")
    raw_profiles = data["raw_profiles"]
    angles = data["angles"]

    original_samples = samples_world.shape[0]
    target = (
        int(recon_params.target_samples)
        if recon_params.target_samples and recon_params.target_samples > 1
        else original_samples
    )

    if target != original_samples:
        samples_world = _resample_curve(samples_world, target)
        tangents, normals, binormals = _local_frames(samples_world)
        s_old = _arclength(data["samples_world"])
        s_new = np.linspace(s_old[0], s_old[-1], target, dtype=np.float64)
        raw_profiles = _interpolate_profile_matrix(
            raw_profiles,
            s_old,
            s_new,
            smoothing_factor=recon_params.smoothing_factor,
            min_valid=recon_params.min_valid_slices,
            kind=recon_params.interpolation_kind,
        )
    elif recon_params.smoothing_factor > 0:
        s_old = _arclength(samples_world)
        raw_profiles = _interpolate_profile_matrix(
            raw_profiles,
            s_old,
            s_old,
            smoothing_factor=recon_params.smoothing_factor,
            min_valid=recon_params.min_valid_slices,
            kind=recon_params.interpolation_kind,
        )

    raw_profiles, angles = _resample_polar_profiles(
        raw_profiles,
        angles,
        upsample_factor=recon_params.angular_upsample,
        smoothing_sigma=recon_params.angular_smoothing,
        min_radius_ratio=recon_params.min_radius_ratio,
        angular_gap_fill_bins=recon_params.angular_gap_fill_bins,
    )

    if recon_params.axial_gap_fill > 0:
        row_valid = raw_profiles.sum(axis=1) > 0
        if not np.all(row_valid):
            num_rows = raw_profiles.shape[0]
            idx = 0
            while idx < num_rows:
                if row_valid[idx]:
                    idx += 1
                    continue
                gap_start = idx
                while idx < num_rows and not row_valid[idx]:
                    idx += 1
                gap_end = idx
                gap_len = gap_end - gap_start
                if gap_len <= recon_params.axial_gap_fill:
                    prev_idx = gap_start - 1 if gap_start > 0 else None
                    next_idx = gap_end if gap_end < num_rows else None
                    if prev_idx is not None and next_idx is not None:
                        for k in range(gap_len):
                            alpha = (k + 1) / (gap_len + 1)
                            raw_profiles[gap_start + k] = (
                                (1.0 - alpha) * raw_profiles[prev_idx] + alpha * raw_profiles[next_idx]
                            )
                    elif prev_idx is not None:
                        raw_profiles[gap_start:gap_end] = raw_profiles[prev_idx]
                    elif next_idx is not None:
                        raw_profiles[gap_start:gap_end] = raw_profiles[next_idx]

    raw_profiles = raw_profiles.astype(np.float32, copy=False)
    if tangents is None or normals is None or binormals is None or tangents.shape[0] != samples_world.shape[0]:
        tangents, normals, binormals = _local_frames(samples_world)

    num_samples, num_bins = raw_profiles.shape
    num_bins_ext = num_bins + 1
    points = np.tile(samples_world[np.newaxis, :, :], (num_bins_ext, 1, 1)).astype(np.float32)
    valid_points = np.zeros((num_bins_ext, num_samples), dtype=bool)

    for sample_idx in range(num_samples):
        center = samples_world[sample_idx]
        normal = normals[sample_idx]
        binormal = binormals[sample_idx]
        radii = raw_profiles[sample_idx]
        valid = radii > 0
        if valid.sum() < 2:
            continue
        theta_valid = angles[valid]
        radii_valid = radii[valid]
        order = np.argsort(theta_valid)
        theta_valid = theta_valid[order]
        radii_valid = radii_valid[order]

        if theta_valid[-1] - theta_valid[0] >= 2 * math.pi - 1e-3:
            theta_periodic = theta_valid
            radii_periodic = radii_valid
        else:
            theta_periodic = np.concatenate(
                [theta_valid - 2 * math.pi, theta_valid, theta_valid + 2 * math.pi]
            )
            radii_periodic = np.concatenate([radii_valid, radii_valid, radii_valid])

        interpolated = np.interp(
            np.linspace(theta_valid[0], theta_valid[0] + 2 * math.pi, num_bins + 1, endpoint=True),
            theta_periodic,
            radii_periodic,
        )
        theta_dense = np.linspace(theta_valid[0], theta_valid[0] + 2 * math.pi, num_bins + 1, endpoint=True)
        points_slice = (
            center
            + interpolated[:, None] * np.cos(theta_dense)[:, None] * normal
            + interpolated[:, None] * np.sin(theta_dense)[:, None] * binormal
        )
        points[:num_bins, sample_idx] = points_slice[:-1]
        points[num_bins, sample_idx] = points_slice[-1]
        valid_points[: num_bins + 1, sample_idx] = interpolated > 0

    grid = pv.StructuredGrid()
    grid.points = points.reshape(-1, 3, order="F")
    grid.dimensions = (num_bins_ext, num_samples, 1)
    grid["valid"] = valid_points.reshape(-1, order="F").astype(np.uint8)
    surface = grid.extract_surface(algorithm="dataset_surface").triangulate()
    if "valid" in surface.point_data:
        mask = surface.point_data["valid"].astype(bool)
        surface = surface.extract_points(mask, adjacent_cells=True)
    if hasattr(surface, "point_data") and "valid" in surface.point_data:
        surface.point_data.pop("valid")
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface(algorithm="dataset_surface")
    return surface.clean()


def reconstruct_branch_meshes(
    features_dir: Path,
    output_dir: Path,
    recon_params: ReconstructionParams,
) -> dict[str, Any]:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyvista is required for mesh reconstruction.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = features_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {features_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    branch_meshes: list[dict[str, Any]] = []
    merged = None
    for branch_meta in summary.get("branches", []):
        branch_name = str(branch_meta["name"])
        branch_file = features_dir / str(branch_meta["feature_file"])
        mesh = _reconstruct_single_branch_mesh(branch_file, recon_params)
        if mesh.n_points < 3 or mesh.n_cells == 0:
            continue
        out_path = output_dir / f"{branch_name}.vtp"
        mesh.save(str(out_path))
        if merged is None:
            merged = mesh
        else:
            merged = merged.merge(mesh, merge_points=True, tolerance=1e-3)
        branch_meshes.append(
            {
                "branch_name": branch_name,
                "mesh_file": out_path.name,
                "n_points": int(mesh.n_points),
                "n_cells": int(mesh.n_cells),
                "length_mm": float(branch_meta.get("length_mm", 0.0)),
            }
        )

    if merged is None:
        raise RuntimeError("No branch meshes reconstructed.")
    merged = merged.clean()
    merged_path = output_dir / "merged.vtp"
    merged.save(str(merged_path))
    payload = {
        "branch_count": len(branch_meshes),
        "branch_meshes": branch_meshes,
        "merged_mesh": merged_path.name,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compute_wall_unfolding_map(
    ct_volume: np.ndarray,
    affine: np.ndarray,
    profile: BranchProfile,
    *,
    relative_radii: Sequence[float],
) -> np.ndarray:
    inv_affine = np.linalg.inv(affine)
    raw = profile.raw_profiles.astype(np.float64)
    wall_map = np.full(raw.T.shape, np.nan, dtype=np.float32)

    for sample_idx in range(raw.shape[0]):
        center = profile.samples_world[sample_idx]
        normal = profile.normals[sample_idx]
        binormal = profile.binormals[sample_idx]
        for angle_idx, theta in enumerate(profile.angles.astype(np.float64)):
            radius = float(raw[sample_idx, angle_idx])
            if radius <= 0:
                continue
            direction = math.cos(theta) * normal + math.sin(theta) * binormal
            values: list[float] = []
            for relative in relative_radii:
                point = center + float(relative) * radius * direction
                value = _sample_trilinear_affine(ct_volume, inv_affine, point)
                if value is not None:
                    values.append(value)
            if values:
                wall_map[angle_idx, sample_idx] = float(np.mean(values))
    return wall_map


def compute_longitudinal_section_map(
    ct_volume: np.ndarray,
    affine: np.ndarray,
    profile: BranchProfile,
    *,
    width_mm: float,
    slab_mm: float,
    out_width_px: int,
) -> np.ndarray:
    inv_affine = np.linalg.inv(affine)
    num_samples = profile.samples_world.shape[0]
    out_width_px = max(int(out_width_px), 8)
    width_mm = max(float(width_mm), 1.0)
    slab_mm = max(float(slab_mm), 0.0)

    lateral_offsets = np.linspace(-0.5 * width_mm, 0.5 * width_mm, out_width_px, dtype=np.float64)
    voxel_sizes = nib.affines.voxel_sizes(affine)
    base_spacing = max(float(min(voxel_sizes)), 0.25)
    if slab_mm <= SMALL_EPS:
        slab_offsets = np.array([0.0], dtype=np.float64)
    else:
        slab_samples = max(int(math.ceil(slab_mm / base_spacing)), 3)
        slab_offsets = np.linspace(-0.5 * slab_mm, 0.5 * slab_mm, slab_samples, dtype=np.float64)

    section = np.full((out_width_px, num_samples), np.nan, dtype=np.float32)
    for sample_idx in range(num_samples):
        center = profile.samples_world[sample_idx]
        normal = profile.normals[sample_idx]
        binormal = profile.binormals[sample_idx]
        for width_idx, offset in enumerate(lateral_offsets):
            base_point = center + offset * normal
            values: list[float] = []
            for slab_offset in slab_offsets:
                point = base_point + slab_offset * binormal
                value = _sample_trilinear_affine(ct_volume, inv_affine, point)
                if value is not None:
                    values.append(value)
            if values:
                section[width_idx, sample_idx] = float(np.max(values))
    return section


def compute_branch_metrics(
    profile: BranchProfile,
    wall_map: np.ndarray | None = None,
    *,
    high_hu_threshold: float,
) -> BranchMetric:
    mean_radius = profile.mean_radius.astype(np.float64)
    valid_radius = mean_radius[mean_radius > 0]
    if valid_radius.size == 0:
        valid_radius = np.array([0.0], dtype=np.float64)

    areas = np.array([_slice_area_mm2(row) for row in profile.raw_profiles], dtype=np.float64)
    valid_areas = areas[areas > 0]
    if valid_areas.size == 0:
        valid_areas = np.array([0.0], dtype=np.float64)

    endpoint_dist = float(np.linalg.norm(profile.samples_world[-1] - profile.samples_world[0]))
    tortuosity = profile.branch.length_mm / max(endpoint_dist, SMALL_EPS) - 1.0 if endpoint_dist > 0 else 0.0
    curvature = _compute_curvature(profile.samples_world)
    proximal_count = max(3, int(math.ceil(profile.samples_world.shape[0] * 0.2)))
    proximal_radius = mean_radius[:proximal_count]
    proximal_valid = proximal_radius[proximal_radius > 0]
    proximal_reference = float(proximal_valid.mean()) if proximal_valid.size else float(valid_radius.mean())
    stenosis_pct = 100.0 * max(0.0, 1.0 - float(valid_radius.min()) / max(proximal_reference, SMALL_EPS))

    eccentricity_values = []
    for row in profile.raw_profiles:
        valid = row[row > 0]
        if valid.size < 3:
            continue
        eccentricity_values.append(float(valid.max()) / max(float(valid.min()), SMALL_EPS))
    eccentricity_mean = float(np.mean(eccentricity_values)) if eccentricity_values else 1.0

    if wall_map is not None and np.isfinite(wall_map).any():
        finite = wall_map[np.isfinite(wall_map)]
        wall_hu_mean = float(np.mean(finite))
        wall_hu_std = float(np.std(finite))
        high_hu_fraction = float(np.mean(finite >= high_hu_threshold))
    else:
        wall_hu_mean = None
        wall_hu_std = None
        high_hu_fraction = None

    return BranchMetric(
        branch_name=profile.branch.name,
        length_mm=float(profile.branch.length_mm),
        mean_radius_mm=float(valid_radius.mean()),
        min_radius_mm=float(valid_radius.min()),
        max_radius_mm=float(valid_radius.max()),
        mean_diameter_mm=float(valid_radius.mean() * 2.0),
        min_diameter_mm=float(valid_radius.min() * 2.0),
        max_diameter_mm=float(valid_radius.max() * 2.0),
        mean_area_mm2=float(valid_areas.mean()),
        min_area_mm2=float(valid_areas.min()),
        tortuosity_index=float(max(tortuosity, 0.0)),
        curvature_mean=float(np.mean(curvature)),
        curvature_max=float(np.max(curvature)),
        proximal_reference_radius_mm=float(proximal_reference),
        stenosis_pct=float(stenosis_pct),
        eccentricity_mean=float(eccentricity_mean),
        branch_confidence=float(profile.branch_confidence),
        wall_hu_mean=_ensure_float(wall_hu_mean),
        wall_hu_std=_ensure_float(wall_hu_std),
        high_hu_wall_fraction=_ensure_float(high_hu_fraction),
    )


def _write_metrics(branch_metrics: Sequence[BranchMetric], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "branch_metrics.json"
    csv_path = output_dir / "branch_metrics.csv"
    rows = [asdict(metric) for metric in branch_metrics]
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else [field.name for field in BranchMetric.__dataclass_fields__.values()]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def _summarise_case_metrics(branch_metrics: Sequence[BranchMetric]) -> dict[str, Any]:
    if not branch_metrics:
        return {
            "branch_count": 0,
            "total_length_mm": 0.0,
        }
    total_length = float(sum(metric.length_mm for metric in branch_metrics))
    weighted_mean_diameter = float(
        sum(metric.mean_diameter_mm * metric.length_mm for metric in branch_metrics) / max(total_length, SMALL_EPS)
    )
    global_min_diameter = min(branch_metrics, key=lambda metric: metric.min_diameter_mm)
    max_stenosis = max(branch_metrics, key=lambda metric: metric.stenosis_pct)
    max_curvature = max(branch_metrics, key=lambda metric: metric.curvature_max)
    top_lengths = sorted(branch_metrics, key=lambda metric: metric.length_mm, reverse=True)[:5]
    high_hu_values = [metric.high_hu_wall_fraction for metric in branch_metrics if metric.high_hu_wall_fraction is not None]
    wall_hu_values = [metric.wall_hu_mean for metric in branch_metrics if metric.wall_hu_mean is not None]
    return {
        "branch_count": len(branch_metrics),
        "total_length_mm": total_length,
        "weighted_mean_diameter_mm": weighted_mean_diameter,
        "global_min_diameter_mm": float(global_min_diameter.min_diameter_mm),
        "global_min_diameter_branch": global_min_diameter.branch_name,
        "max_stenosis_pct": float(max_stenosis.stenosis_pct),
        "max_stenosis_branch": max_stenosis.branch_name,
        "max_curvature": float(max_curvature.curvature_max),
        "max_curvature_branch": max_curvature.branch_name,
        "mean_wall_hu": float(np.mean(wall_hu_values)) if wall_hu_values else None,
        "mean_high_hu_wall_fraction": float(np.mean(high_hu_values)) if high_hu_values else None,
        "top_length_branches": [metric.branch_name for metric in top_lengths],
    }


def _read_vtp_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vtk is required to read VTP meshes.") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    points = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())], dtype=np.float64)
    triangles: list[list[int]] = []
    polys = poly.GetPolys()
    polys.InitTraversal()
    ids = vtk.vtkIdList()
    while polys.GetNextCell(ids):
        n_ids = ids.GetNumberOfIds()
        if n_ids < 3:
            continue
        face = [int(ids.GetId(i)) for i in range(n_ids)]
        for idx in range(1, n_ids - 1):
            triangles.append([face[0], face[idx], face[idx + 1]])
    if not triangles:
        return points, np.zeros((0, 3), dtype=np.int32)
    return points, np.asarray(triangles, dtype=np.int32)


def _set_equal_aspect_3d(ax: Axes3D, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    ranges = maxs - mins
    max_range = float(ranges.max()) / 2.0
    center = (maxs + mins) / 2.0
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)


def _render_coronary_overview(
    ax: Axes3D,
    mesh_dir: Path,
    branch_metrics: Sequence[BranchMetric],
    *,
    legend_limit: int = 8,
    max_triangles_per_branch: int = 2500,
) -> None:
    colors = cm.get_cmap("tab20", max(len(branch_metrics), 1))
    all_points: list[np.ndarray] = []
    for idx, metric in enumerate(branch_metrics):
        mesh_path = mesh_dir / f"{metric.branch_name}.vtp"
        if not mesh_path.exists():
            continue
        points, faces = _read_vtp_mesh(mesh_path)
        if points.size == 0:
            continue
        all_points.append(points)
        if faces.size:
            if faces.shape[0] > max_triangles_per_branch:
                stride = max(int(math.ceil(faces.shape[0] / max_triangles_per_branch)), 1)
                faces = faces[::stride]
            ax.plot_trisurf(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                triangles=faces,
                color=colors(idx),
                linewidth=0.0,
                shade=False,
                alpha=0.95,
                antialiased=False,
            )
        else:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors(idx), linewidth=1.2)

    ax.view_init(elev=22.0, azim=-58.0)
    ax.set_title("Coronary Completion (Modeled Vessel Walls)", fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    if all_points:
        _set_equal_aspect_3d(ax, np.vstack(all_points))

    handles = []
    labels = []
    for idx, metric in enumerate(branch_metrics[:legend_limit]):
        handles.append(plt.Line2D([0], [0], color=colors(idx), lw=4))
        labels.append(metric.branch_name)
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=8,
            frameon=False,
        )


def _render_longitudinal_sections(
    figure: plt.Figure,
    spec,
    selected_profiles: Sequence[BranchProfile],
    selected_maps: Mapping[str, np.ndarray],
    selected_metrics: Mapping[str, BranchMetric],
    *,
    width_mm: float,
) -> None:
    inner = GridSpecFromSubplotSpec(len(selected_profiles), 1, subplot_spec=spec, hspace=0.22)
    colors = cm.get_cmap("tab10", max(len(selected_profiles), 1))
    for row_idx, profile in enumerate(selected_profiles):
        ax = figure.add_subplot(inner[row_idx, 0])
        section_map = selected_maps[profile.branch.name]
        metric = selected_metrics[profile.branch.name]
        extent = [0.0, float(profile.branch.length_mm), -0.5 * width_mm, 0.5 * width_mm]
        valid = section_map[np.isfinite(section_map)]
        if valid.size:
            vmin = float(np.percentile(valid, 5))
            vmax = float(np.percentile(valid, 95))
        else:
            vmin, vmax = 0.0, 1.0
        ax.imshow(
            section_map,
            cmap="gray",
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        s_mm = np.linspace(0.0, float(profile.branch.length_mm), profile.mean_radius.shape[0], dtype=np.float64)
        color = colors(row_idx)
        ax.plot(s_mm, profile.mean_radius, color=color, linewidth=1.2)
        ax.plot(s_mm, -profile.mean_radius, color=color, linewidth=1.2)
        ax.set_ylabel("Width (mm)", fontsize=8)
        if row_idx == len(selected_profiles) - 1:
            ax.set_xlabel("Arc length (mm)", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_title(
            f"{profile.branch.name} | Longitudinal section | L={metric.length_mm:.1f}mm  MinD={metric.min_diameter_mm:.2f}mm  Stenosis={metric.stenosis_pct:.1f}%",
            fontsize=9,
            loc="left",
        )
        ax.tick_params(axis="both", labelsize=7)


def _render_metrics_panel(
    ax: plt.Axes,
    case_id: str,
    case_summary: Mapping[str, Any],
    top_metrics: Sequence[BranchMetric],
) -> None:
    ax.axis("off")
    ax.set_title("Coronary Clinical Summary", fontsize=13, loc="left", pad=10)

    lines = [
        f"Case: {case_id}",
        f"Branches: {case_summary.get('branch_count', 0)}",
        f"Total length: {case_summary.get('total_length_mm', 0.0):.1f} mm",
        f"Weighted mean diameter: {case_summary.get('weighted_mean_diameter_mm', 0.0):.2f} mm",
        f"Global min diameter: {case_summary.get('global_min_diameter_mm', 0.0):.2f} mm",
        f"Min-diameter branch: {case_summary.get('global_min_diameter_branch', '-')}",
        f"Max stenosis: {case_summary.get('max_stenosis_pct', 0.0):.1f} %",
        f"Max-stenosis branch: {case_summary.get('max_stenosis_branch', '-')}",
        f"Max curvature: {case_summary.get('max_curvature', 0.0):.3f}",
        f"Curvature branch: {case_summary.get('max_curvature_branch', '-')}",
    ]
    if case_summary.get("mean_wall_hu") is not None:
        lines.append(f"Mean wall HU: {float(case_summary['mean_wall_hu']):.1f}")
    if case_summary.get("mean_high_hu_wall_fraction") is not None:
        lines.append(f"High-HU wall fraction: {100.0 * float(case_summary['mean_high_hu_wall_fraction']):.1f} %")

    y = 0.98
    for line in lines:
        ax.text(0.02, y, line, fontsize=10, va="top", ha="left")
        y -= 0.055

    ax.text(0.02, y - 0.01, "Top branches by stenosis proxy", fontsize=11, weight="bold", va="top")
    y -= 0.08
    for metric in top_metrics:
        ax.text(
            0.02,
            y,
            (
                f"{metric.branch_name}: Stenosis {metric.stenosis_pct:.1f}% | "
                f"MinD {metric.min_diameter_mm:.2f} mm | "
                f"Tort {metric.tortuosity_index:.3f}"
            ),
            fontsize=9,
            va="top",
            ha="left",
        )
        y -= 0.05

    if top_metrics:
        inset = ax.inset_axes([0.05, 0.05, 0.9, 0.28])
        labels = [metric.branch_name for metric in top_metrics]
        values = [metric.stenosis_pct for metric in top_metrics]
        positions = np.arange(len(labels))
        inset.barh(positions, values[::-1], color="#3b82f6")
        inset.set_yticks(positions)
        inset.set_yticklabels(labels[::-1])
        inset.set_xlabel("Stenosis proxy (%)", fontsize=8)
        inset.tick_params(axis="both", labelsize=8)
        inset.grid(axis="x", linestyle=":", alpha=0.4)


def create_clinical_dashboard(
    case_id: str,
    mesh_dir: Path,
    branch_profiles: Sequence[BranchProfile],
    longitudinal_sections: Mapping[str, np.ndarray],
    branch_metrics: Sequence[BranchMetric],
    output_dir: Path,
    *,
    top_unfold_branches: int = 3,
    longitudinal_width_mm: float = 8.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = [
        metric.branch_name
        for metric in sorted(branch_metrics, key=lambda item: item.length_mm, reverse=True)[: max(int(top_unfold_branches), 1)]
    ]
    selected_profiles = [profile for profile in branch_profiles if profile.branch.name in selected_names]
    selected_metrics = {metric.branch_name: metric for metric in branch_metrics if metric.branch_name in selected_names}
    selected_maps = {name: longitudinal_sections[name] for name in selected_names if name in longitudinal_sections}
    case_summary = _summarise_case_metrics(branch_metrics)
    top_stenosis = sorted(branch_metrics, key=lambda metric: metric.stenosis_pct, reverse=True)[:5]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.7, 1.0], height_ratios=[1.1, 1.0], wspace=0.18, hspace=0.12)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    _render_coronary_overview(ax3d, mesh_dir, branch_metrics)
    _render_longitudinal_sections(
        fig,
        gs[1, 0],
        selected_profiles,
        selected_maps,
        selected_metrics,
        width_mm=longitudinal_width_mm,
    )
    ax_metrics = fig.add_subplot(gs[:, 1])
    _render_metrics_panel(ax_metrics, case_id, case_summary, top_stenosis)
    fig.suptitle("Coronary Completion Dashboard", fontsize=16, weight="bold")
    dashboard_path = output_dir / "dashboard.png"
    fig.savefig(dashboard_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return dashboard_path


def run_clinical_completion_workflow(config: ClinicalCompletionConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    features_original_dir = config.out_dir / "features_original"
    features_modeled_dir = config.out_dir / "features_modeled"
    meshes_original_dir = config.out_dir / "meshes_original"
    meshes_modeled_dir = config.out_dir / "meshes_modeled"
    unfoldings_dir = config.out_dir / "unfoldings"
    longitudinal_dir = config.out_dir / "longitudinal_sections"
    centerline_sort_dir = config.out_dir / "centerline_sort"

    stage_durations: dict[str, float] = {}
    stage_outputs: dict[str, Any] = {}

    t0 = time.perf_counter()
    centreline_params = CentrelineParams(
        min_length_mm=config.min_length_mm,
        max_components=config.max_components,
        closing_iterations=config.closing_iterations,
        smooth_sigma_mm=config.smooth_sigma_mm,
        adaptive_min_step_mm=config.adaptive_min_step_mm,
        adaptive_max_step_mm=config.adaptive_max_step_mm,
        curvature_alpha=config.adaptive_curvature_alpha,
    )
    if config.centerline_backend == "vessel_sort":
        branches, extraction_info = _extract_branches_via_vessel_sort(
            config.mask_path,
            config.case_id,
            centreline_params,
            centerline_sort_dir,
        )
    elif config.centerline_backend == "shape":
        branches, extraction_info = _extract_branches_via_shape(config.mask_path, centreline_params)
    else:
        raise ValueError(f"Unsupported centerline backend: {config.centerline_backend}")
    stage_durations["centerline_extraction"] = time.perf_counter() - t0
    stage_outputs["centerline_extraction"] = extraction_info

    t0 = time.perf_counter()
    profile_params = ProfileParams(
        num_samples=config.num_samples,
        num_angle_bins=config.num_angle_bins,
        patch_radius_mm=config.patch_radius_mm,
        half_thickness_mm=config.half_thickness_mm,
        radius_clip_factor=config.radius_clip_factor,
        endpoint_trim_mm=config.endpoint_trim_mm,
        junction_distance_mm=config.junction_distance_mm,
        junction_inherit_mm=config.junction_inherit_mm,
    )
    profiles = compute_polar_profiles(config.mask_path, branches, profile_params)
    features_summary = export_branch_features(profiles, features_original_dir)
    stage_durations["wall_feature_extraction"] = time.perf_counter() - t0
    stage_outputs["wall_feature_extraction"] = features_summary

    t0 = time.perf_counter()
    pca_model = _fit_pca_model(
        profiles,
        num_components=config.pca_components,
        confidence_threshold=config.confidence_threshold,
    )
    modeled_profiles = _reconstruct_profiles_from_pca(profiles, pca_model)
    modeled_summary = _save_modeled_features(profiles, modeled_profiles, pca_model, features_modeled_dir)
    stage_durations["shape_modeling"] = time.perf_counter() - t0
    stage_outputs["shape_modeling"] = modeled_summary

    recon_params = ReconstructionParams(
        target_samples=config.mesh_target_samples,
        smoothing_factor=config.mesh_smoothing,
        min_valid_slices=config.mesh_min_valid,
        interpolation_kind=config.mesh_interp_kind,
        angular_upsample=config.mesh_angular_upsample,
        angular_smoothing=config.mesh_angular_smoothing,
        min_radius_ratio=config.mesh_min_radius_ratio,
        angular_gap_fill_bins=config.mesh_angular_gap_fill,
        axial_gap_fill=config.mesh_axial_gap_fill,
    )

    t0 = time.perf_counter()
    mesh_summary_original = reconstruct_branch_meshes(features_original_dir, meshes_original_dir, recon_params)
    mesh_summary_modeled = reconstruct_branch_meshes(features_modeled_dir, meshes_modeled_dir, recon_params)
    stage_durations["mesh_reconstruction"] = time.perf_counter() - t0
    stage_outputs["mesh_reconstruction"] = {
        "original": mesh_summary_original,
        "modeled": mesh_summary_modeled,
    }

    t0 = time.perf_counter()
    ct_image = nib.load(str(config.ct_path))
    ct_volume = np.asarray(ct_image.dataobj, dtype=np.float32)
    wall_maps: dict[str, np.ndarray] = {}
    longitudinal_sections: dict[str, np.ndarray] = {}
    branch_metrics: list[BranchMetric] = []
    unfoldings_dir.mkdir(parents=True, exist_ok=True)
    longitudinal_dir.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        wall_map = compute_wall_unfolding_map(
            ct_volume,
            ct_image.affine,
            profile,
            relative_radii=config.wall_sample_relative_radii,
        )
        longitudinal_section = compute_longitudinal_section_map(
            ct_volume,
            ct_image.affine,
            profile,
            width_mm=config.longitudinal_width_mm,
            slab_mm=config.longitudinal_slab_mm,
            out_width_px=config.longitudinal_out_width_px,
        )
        wall_maps[profile.branch.name] = wall_map
        longitudinal_sections[profile.branch.name] = longitudinal_section
        np.save(unfoldings_dir / f"{profile.branch.name}_wall_map.npy", wall_map)
        np.save(longitudinal_dir / f"{profile.branch.name}_longitudinal.npy", longitudinal_section)
        branch_metrics.append(
            compute_branch_metrics(
                profile,
                wall_map,
                high_hu_threshold=config.high_hu_threshold,
            )
        )
    branch_metrics.sort(key=lambda item: item.length_mm, reverse=True)
    metrics_json_path, metrics_csv_path = _write_metrics(branch_metrics, config.out_dir)
    case_summary = _summarise_case_metrics(branch_metrics)
    case_summary_path = config.out_dir / "case_summary.json"
    case_summary_path.write_text(json.dumps(case_summary, indent=2), encoding="utf-8")
    np.savez(config.out_dir / "wall_maps.npz", **wall_maps)
    np.savez(config.out_dir / "longitudinal_sections.npz", **longitudinal_sections)
    stage_durations["metric_analysis"] = time.perf_counter() - t0
    stage_outputs["metric_analysis"] = {
        "branch_metrics_json": str(metrics_json_path),
        "branch_metrics_csv": str(metrics_csv_path),
        "case_summary_json": str(case_summary_path),
        "wall_maps_npz": str(config.out_dir / "wall_maps.npz"),
        "longitudinal_sections_npz": str(config.out_dir / "longitudinal_sections.npz"),
        "longitudinal_sections_dir": str(longitudinal_dir),
    }

    t0 = time.perf_counter()
    dashboard_path = create_clinical_dashboard(
        config.case_id,
        meshes_modeled_dir,
        profiles,
        longitudinal_sections,
        branch_metrics,
        config.out_dir,
        top_unfold_branches=config.top_unfold_branches,
        longitudinal_width_mm=config.longitudinal_width_mm,
    )
    stage_durations["dashboard"] = time.perf_counter() - t0
    stage_outputs["dashboard"] = {
        "dashboard_png": str(dashboard_path),
    }

    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": _now_utc(),
        "case_id": config.case_id,
        "inputs": {
            "ct_path": str(config.ct_path),
            "mask_path": str(config.mask_path),
        },
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(config).items()
        },
        "stages": stage_outputs,
        "durations_sec": stage_durations,
        "case_summary": case_summary,
    }
    report_path = config.out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def add_clinical_completion_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--ct", type=Path, default=Path("ASOCA2020/Normal/CTCA_nii/Normal_1.nii.gz"))
    parser.add_argument("--mask", type=Path, default=Path("ASOCA2020/Normal/Annotations_nii/Normal_1.nii.gz"))
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: outputs_reorganized/cases/<case>/analysis/clinical_completion_default",
    )
    parser.add_argument("--centerline-backend", choices=["vessel_sort", "shape"], default="vessel_sort")
    parser.add_argument("--min-length-mm", type=float, default=10.0)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--closing-iterations", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--num-angle-bins", type=int, default=72)
    parser.add_argument("--patch-radius-mm", type=float, default=3.5)
    parser.add_argument("--half-thickness-mm", type=float, default=1.2)
    parser.add_argument("--endpoint-trim-mm", type=float, default=1.5)
    parser.add_argument("--radius-clip-factor", type=float, default=1.8)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--mesh-target-samples", type=int, default=140)
    parser.add_argument("--mesh-angular-upsample", type=int, default=3)
    parser.add_argument("--mesh-angular-smoothing", type=float, default=1.0)
    parser.add_argument("--mesh-min-radius-ratio", type=float, default=0.08)
    parser.add_argument("--mesh-angular-gap-fill", type=int, default=2)
    parser.add_argument("--mesh-axial-gap-fill", type=int, default=1)
    parser.add_argument("--top-unfold-branches", type=int, default=3)
    parser.add_argument("--longitudinal-width-mm", type=float, default=8.0)
    parser.add_argument("--longitudinal-slab-mm", type=float, default=2.0)
    parser.add_argument("--longitudinal-out-width-px", type=int, default=96)
    parser.add_argument("--high-hu-threshold", type=float, default=350.0)
    return parser


def run_clinical_completion_command(args: argparse.Namespace) -> Path:
    ct_path = args.ct.resolve() if not args.ct.is_absolute() else args.ct
    mask_path = args.mask.resolve() if not args.mask.is_absolute() else args.mask
    case_id = args.case or _infer_case_id(mask_path)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None and args.out_dir.is_absolute()
        else (Path.cwd() / args.out_dir).resolve()
        if args.out_dir is not None
        else (Path.cwd() / "outputs_reorganized" / "cases" / case_id / "analysis" / "clinical_completion_default")
    )
    config = ClinicalCompletionConfig(
        case_id=case_id,
        ct_path=ct_path,
        mask_path=mask_path,
        out_dir=out_dir,
        centerline_backend=args.centerline_backend,
        min_length_mm=args.min_length_mm,
        max_components=args.max_components,
        closing_iterations=args.closing_iterations,
        num_samples=args.num_samples,
        num_angle_bins=args.num_angle_bins,
        patch_radius_mm=args.patch_radius_mm,
        half_thickness_mm=args.half_thickness_mm,
        endpoint_trim_mm=args.endpoint_trim_mm,
        radius_clip_factor=args.radius_clip_factor,
        pca_components=args.pca_components,
        confidence_threshold=args.confidence_threshold,
        mesh_target_samples=args.mesh_target_samples,
        mesh_angular_upsample=args.mesh_angular_upsample,
        mesh_angular_smoothing=args.mesh_angular_smoothing,
        mesh_min_radius_ratio=args.mesh_min_radius_ratio,
        mesh_angular_gap_fill=args.mesh_angular_gap_fill,
        mesh_axial_gap_fill=args.mesh_axial_gap_fill,
        top_unfold_branches=args.top_unfold_branches,
        longitudinal_width_mm=args.longitudinal_width_mm,
        longitudinal_slab_mm=args.longitudinal_slab_mm,
        longitudinal_out_width_px=args.longitudinal_out_width_px,
        high_hu_threshold=args.high_hu_threshold,
    )
    payload = run_clinical_completion_workflow(config)
    print(json.dumps(payload["case_summary"], indent=2))
    return out_dir
