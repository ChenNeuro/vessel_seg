"""Branch extraction, polar profiling, and reconstruction utilities."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy import ndimage, interpolate
from scipy.ndimage import gaussian_filter1d

try:
    from skimage.morphology import skeletonize
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "scikit-image is required for skeletonisation. Install with `pip install scikit-image`."
    ) from exc

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("networkx is required. Install with `pip install networkx`.") from exc


SMALL_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class CentrelineParams:
    """Tunable parameters for centreline extraction and resampling."""

    min_length_mm: float = 5.0
    short_bridge_max_mm: float = 6.0
    closing_iterations: int = 1
    smooth_sigma_mm: float = 0.8
    adaptive_min_step_mm: float = 0.6
    adaptive_max_step_mm: float = 2.5
    curvature_alpha: float = 2.0


@dataclass
class ProfileParams:
    """Configuration for cross-sectional polar sampling."""

    num_samples: int = 100
    num_angle_bins: int = 72
    patch_radius_mm: float = 3.0
    half_thickness_mm: float = 1.0
    radius_clip_factor: float = 0.0
    endpoint_trim_mm: float = 0.0
    junction_distance_mm: float = 1.5
    junction_inherit_mm: float = 1.5


@dataclass
class ReconstructionParams:
    """Parameters controlling curve sweep reconstruction."""

    target_samples: Optional[int] = None
    smoothing_factor: float = 0.0
    min_valid_slices: int = 3
    interpolation_kind: str = "cubic"
    angular_upsample: int = 1
    angular_smoothing: float = 0.0
    min_radius_ratio: float = 0.05
    angular_gap_fill_bins: int = 0
    axial_gap_fill: int = 0


@dataclass
class Branch:
    """Representation of a single vessel branch."""

    name: str
    world_points: np.ndarray  # (N, 3)
    voxel_points: np.ndarray  # (N, 3)
    length_mm: float
    source: str = "observed"


@dataclass
class BranchProfile:
    """Per-branch sampling and polar descriptor bundle."""

    branch: Branch
    samples_world: np.ndarray  # (S, 3)
    samples_voxel: np.ndarray  # (S, 3)
    tangents: np.ndarray  # (S, 3)
    normals: np.ndarray  # (S, 3)
    binormals: np.ndarray  # (S, 3)
    raw_profiles: np.ndarray  # (S, A)
    normalized_profiles: np.ndarray  # (S, A)
    mean_radius: np.ndarray  # (S,)
    slice_confidence: np.ndarray  # (S,)
    branch_confidence: float
    feature_vector: np.ndarray
    angles: np.ndarray  # (A,)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num <= 1:
        return mask
    counts = ndimage.sum(mask, labeled, index=range(1, num + 1))
    largest_label = int(np.argmax(counts) + 1)
    return labeled == largest_label


def _binary_cleanup(mask: np.ndarray, iterations: int) -> np.ndarray:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    closed = ndimage.binary_closing(mask, structure=structure, iterations=iterations)
    filled = ndimage.binary_fill_holes(closed)
    return _largest_component(filled)


def _iter_neighbors(coord: Sequence[int]) -> Iterable[Tuple[int, int, int]]:
    x, y, z = coord
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                yield x + dx, y + dy, z + dz


def _build_graph(skeleton: np.ndarray) -> nx.Graph:
    coords = np.argwhere(skeleton)
    index_map = -np.ones(skeleton.shape, dtype=np.int32)
    index_map[tuple(coords.T)] = np.arange(coords.shape[0], dtype=np.int32)
    graph = nx.Graph()
    for idx, (x, y, z) in enumerate(coords):
        graph.add_node(idx, coord=np.array([x, y, z], dtype=np.int32))
        for nx_, ny_, nz_ in _iter_neighbors((x, y, z)):
            if (
                0 <= nx_ < skeleton.shape[0]
                and 0 <= ny_ < skeleton.shape[1]
                and 0 <= nz_ < skeleton.shape[2]
            ):
                neighbor_index = index_map[nx_, ny_, nz_]
                if neighbor_index >= 0 and neighbor_index > idx:
                    graph.add_edge(idx, neighbor_index)
    return graph


def _trace_branches(graph: nx.Graph) -> List[List[int]]:
    if graph.number_of_nodes() == 0:
        return []

    special = [n for n in graph.nodes if graph.degree[n] != 2]
    if not special:
        special = [next(iter(graph.nodes))]

    visited_edges = set()
    branches: List[List[int]] = []

    for start in special:
        for neighbor in graph.neighbors(start):
            edge = tuple(sorted((start, neighbor)))
            if edge in visited_edges:
                continue
            path = [start]
            prev = start
            current = neighbor
            visited_edges.add(edge)

            while True:
                path.append(current)
                if graph.degree[current] != 2:
                    break
                next_nodes = [n for n in graph.neighbors(current) if n != prev]
                if not next_nodes:
                    break
                prev, current = current, next_nodes[0]
                visited_edges.add(tuple(sorted((prev, current))))
            branches.append(path)
    return branches


def _compute_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _prune_short_bridge_segments(
    branch_items: List[Dict[str, object]],
    *,
    short_bridge_max_mm: float,
) -> List[Dict[str, object]]:
    """Remove very short junction-to-junction connector stubs.

    Skeleton graphs around bifurcations may introduce tiny bridge segments
    between neighboring high-degree nodes. These segments are usually not
    meaningful anatomical branches and can inflate branch count by one.
    """

    if short_bridge_max_mm <= 0.0 or len(branch_items) < 3:
        return branch_items

    items = list(branch_items)
    while len(items) >= 3:
        endpoint_degree: Dict[int, int] = {}
        for item in items:
            start_node = int(item["start_node"])
            end_node = int(item["end_node"])
            endpoint_degree[start_node] = endpoint_degree.get(start_node, 0) + 1
            endpoint_degree[end_node] = endpoint_degree.get(end_node, 0) + 1

        candidates: List[Tuple[int, float]] = []
        for idx, item in enumerate(items):
            branch = item["branch"]
            assert isinstance(branch, Branch)
            start_node = int(item["start_node"])
            end_node = int(item["end_node"])
            if (
                branch.length_mm <= short_bridge_max_mm
                and endpoint_degree.get(start_node, 0) >= 2
                and endpoint_degree.get(end_node, 0) >= 2
            ):
                candidates.append((idx, branch.length_mm))

        if not candidates:
            break

        # Remove the shortest candidate first, then recompute incidence.
        remove_idx, _ = min(candidates, key=lambda pair: (pair[1], pair[0]))
        items.pop(remove_idx)

    return items


def _deduplicate_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mask = np.ones(points.shape[0], dtype=bool)
    mask[1:] = diffs > SMALL_EPS
    deduped = points[mask]
    if deduped.shape[0] < 2:
        return points[:2] if points.shape[0] >= 2 else points
    return deduped


def _resample_curve(points: np.ndarray, num_samples: int) -> np.ndarray:
    num_samples = max(int(num_samples), 1)
    if points.shape[0] == 0:
        return np.zeros((num_samples, 3), dtype=np.float64)
    if points.shape[0] == 1 or num_samples == 1:
        return np.repeat(points[:1], num_samples, axis=0)

    distances = np.zeros(points.shape[0], dtype=np.float64)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    if distances[-1] < SMALL_EPS:
        return np.repeat(points[:1], num_samples, axis=0)

    target = np.linspace(0.0, distances[-1], num_samples, dtype=np.float64)
    resampled = np.vstack(
        [np.interp(target, distances, points[:, dim]) for dim in range(points.shape[1])]
    ).T
    return resampled


def _smooth_centerline(points: np.ndarray, sigma_mm: float) -> np.ndarray:
    if sigma_mm <= 0 or points.shape[0] < 3:
        return points
    s = np.zeros(points.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    mean_spacing = float(np.mean(np.diff(s))) if points.shape[0] > 1 else 1.0
    if mean_spacing < SMALL_EPS:
        return points
    sigma_idx = max(sigma_mm / mean_spacing, 1e-3)
    smoothed = gaussian_filter1d(points.astype(np.float64), sigma=sigma_idx, axis=0, mode="nearest")
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def _compute_curvature(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    curvature = np.zeros(n, dtype=np.float64)
    if n < 3:
        return curvature
    for i in range(1, n - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < SMALL_EPS or norm2 < SMALL_EPS:
            continue
        cross = np.linalg.norm(np.cross(v1, v2))
        sum_vec = v1 + v2
        norm_sum = np.linalg.norm(sum_vec)
        if norm_sum < SMALL_EPS:
            continue
        curvature[i] = 2.0 * cross / max(norm1 * norm2 * norm_sum, SMALL_EPS)
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def _adaptive_resample_curve(
    points: np.ndarray,
    min_step_mm: float,
    max_step_mm: float,
    curvature_alpha: float,
) -> np.ndarray:
    if (
        points.shape[0] < 3
        or min_step_mm <= 0
        or max_step_mm <= min_step_mm
        or curvature_alpha <= 0
    ):
        return points

    points = _deduplicate_points(points)
    if points.shape[0] < 3:
        return points

    s = np.zeros(points.shape[0], dtype=np.float64)
    s[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    unique_s, unique_idx = np.unique(s, return_index=True)
    if unique_s.shape[0] < s.shape[0]:
        points = points[unique_idx]
        s = unique_s

    total_length = float(s[-1])
    if total_length < min_step_mm:
        return points

    curvature = _compute_curvature(points)
    mean_curvature = float(np.mean(curvature)) + SMALL_EPS
    norm_curvature = curvature / mean_curvature
    step_sizes = np.clip(
        max_step_mm / (1.0 + curvature_alpha * norm_curvature),
        min_step_mm,
        max_step_mm,
    )

    step_interp = interpolate.interp1d(
        s,
        step_sizes,
        kind="linear",
        fill_value=(step_sizes[0], step_sizes[-1]),
        bounds_error=False,
        assume_sorted=True,
    )

    samples = [0.0]
    current = 0.0
    while current < total_length:
        step = float(step_interp(current))
        if step <= SMALL_EPS:
            step = min_step_mm
        current += step
        if current >= total_length:
            break
        samples.append(current)
    samples.append(total_length)
    samples = np.array(samples, dtype=np.float64)

    resampled = np.vstack(
        [np.interp(samples, s, points[:, dim]) for dim in range(points.shape[1])]
    ).T
    return resampled


def _local_frames(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangents = np.zeros_like(points)
    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)

    for idx in range(points.shape[0]):
        if idx == 0:
            forward = points[idx + 1] - points[idx]
        elif idx == points.shape[0] - 1:
            forward = points[idx] - points[idx - 1]
        else:
            forward = points[idx + 1] - points[idx - 1]

        norm_forward = np.linalg.norm(forward)
        if norm_forward < SMALL_EPS:
            forward = np.array([1.0, 0.0, 0.0])
            norm_forward = 1.0

        tangent = forward / norm_forward
        reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tangent, reference)) > 0.95:
            reference = np.array([0.0, 1.0, 0.0])

        normal = np.cross(tangent, reference)
        norm_normal = np.linalg.norm(normal)
        if norm_normal < SMALL_EPS:
            normal = np.array([1.0, 0.0, 0.0])
            norm_normal = 1.0
        normal /= norm_normal

        binormal = np.cross(tangent, normal)
        norm_binormal = np.linalg.norm(binormal)
        if norm_binormal < SMALL_EPS:
            binormal = np.array([0.0, 1.0, 0.0])
            norm_binormal = 1.0
        binormal /= norm_binormal

        tangents[idx] = tangent
        normals[idx] = normal
        binormals[idx] = binormal

    return tangents, normals, binormals


def _interpolate_profile_matrix(
    profiles: np.ndarray,
    s_old: np.ndarray,
    s_new: np.ndarray,
    *,
    smoothing_factor: float,
    min_valid: int,
    kind: str,
) -> np.ndarray:
    num_new = s_new.shape[0]
    num_bins = profiles.shape[1]
    resampled = np.zeros((num_new, num_bins), dtype=np.float32)

    for angle_idx in range(num_bins):
        values = profiles[:, angle_idx]
        valid_mask = values > 0
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            continue

        x = s_old[valid_mask]
        y = values[valid_mask]

        if valid_count == 1:
            resampled[:, angle_idx] = float(y[0])
            continue

        try:
            if smoothing_factor > 0 and valid_count >= max(min_valid, 4):
                spline = interpolate.UnivariateSpline(x, y, s=smoothing_factor * valid_count)
                resampled[:, angle_idx] = np.clip(spline(s_new), 0.0, None)
            else:
                interp_kind = "linear" if valid_count < 3 else kind
                interpolator = interpolate.interp1d(
                    x,
                    y,
                    kind=interp_kind,
                    fill_value=(float(y[0]), float(y[-1])),
                    bounds_error=False,
                    assume_sorted=True,
                )
                resampled[:, angle_idx] = np.clip(interpolator(s_new), 0.0, None)
        except Exception:
            interpolator = interpolate.interp1d(
                x,
                y,
                kind="linear",
                fill_value=(float(y[0]), float(y[-1])),
                bounds_error=False,
                assume_sorted=True,
            )
            resampled[:, angle_idx] = np.clip(interpolator(s_new), 0.0, None)

    return resampled


def _arclength(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


# --------------------------------------------------------------------------- #
# Polar sampling helpers
# --------------------------------------------------------------------------- #


def _compute_profiles_for_branch(
    mask: np.ndarray,
    affine: np.ndarray,
    branch: Branch,
    *,
    num_samples: int,
    num_angle_bins: int,
    patch_radius_mm: float,
    half_thickness_mm: float,
    radius_clip_factor: float,
    endpoint_trim_mm: float,
) -> BranchProfile:
    inv_affine = np.linalg.inv(affine)
    samples_world = _resample_curve(branch.world_points, num_samples)
    samples_voxel = apply_affine(inv_affine, samples_world)
    tangents, normals, binormals = _local_frames(samples_world)

    spacing = np.array(nib.affines.voxel_sizes(affine), dtype=np.float32)

    n_samples = samples_world.shape[0]
    angles = np.linspace(-math.pi, math.pi, num_angle_bins, endpoint=False)
    raw_profiles = np.zeros((n_samples, num_angle_bins), dtype=np.float32)
    mean_radius = np.zeros(n_samples, dtype=np.float32)
    slice_confidence = np.zeros(n_samples, dtype=np.float32)

    radius_vox = patch_radius_mm / np.maximum(spacing, SMALL_EPS)
    half_thickness = half_thickness_mm

    shape = mask.shape
    arclengths = _arclength(samples_world)
    total_length = float(arclengths[-1]) if arclengths.size else 0.0

    for idx in range(n_samples):
        center_world = samples_world[idx]
        center_voxel = samples_voxel[idx]
        tangent = tangents[idx]
        normal = normals[idx]
        binormal = binormals[idx]

        base = np.round(center_voxel).astype(int)
        delta = np.ceil(radius_vox).astype(int)

        bin_values = np.full(num_angle_bins, 0.0, dtype=np.float32)

        for dx in range(-delta[0], delta[0] + 1):
            vx = base[0] + dx
            if vx < 0 or vx >= shape[0]:
                continue
            for dy in range(-delta[1], delta[1] + 1):
                vy = base[1] + dy
                if vy < 0 or vy >= shape[1]:
                    continue
                for dz in range(-delta[2], delta[2] + 1):
                    vz = base[2] + dz
                    if vz < 0 or vz >= shape[2]:
                        continue
                    if not mask[vx, vy, vz]:
                        continue

                    point_world = apply_affine(affine, [vx, vy, vz])
                    offset = point_world - center_world
                    axial = float(np.dot(offset, tangent))
                    if abs(axial) > half_thickness:
                        continue

                    radial_n = float(np.dot(offset, normal))
                    radial_b = float(np.dot(offset, binormal))
                    radius = math.hypot(radial_n, radial_b)
                    if radius > patch_radius_mm + 1e-3:
                        continue

                    theta = math.atan2(radial_b, radial_n)
                    bin_idx = int(((theta + math.pi) / (2 * math.pi)) * num_angle_bins) % num_angle_bins
                    bin_values[bin_idx] = max(bin_values[bin_idx], radius)

        if endpoint_trim_mm > 0 and (
            arclengths[idx] <= endpoint_trim_mm or (total_length - arclengths[idx]) <= endpoint_trim_mm
        ):
            slice_confidence[idx] = 0.0
            raw_profiles[idx] = bin_values
            continue

        valid_mask = bin_values > 0
        if radius_clip_factor > 0 and valid_mask.any():
            median_radius = float(np.median(bin_values[valid_mask]))
            if median_radius > 0:
                clip_threshold = radius_clip_factor * median_radius
                bin_values = np.minimum(bin_values, clip_threshold, dtype=np.float32)
                valid_mask = bin_values > 0

        if valid_mask.any():
            mean_radius[idx] = float(bin_values[valid_mask].mean())
            slice_confidence[idx] = float(valid_mask.mean())
        else:
            slice_confidence[idx] = 0.0
        raw_profiles[idx] = bin_values

    filled_profiles = raw_profiles.copy()
    for angle_idx in range(num_angle_bins):
        column = raw_profiles[:, angle_idx]
        valid_idx = np.where(column > 0)[0]
        if valid_idx.size == 0:
            continue
        missing_idx = np.where(column == 0)[0]
        if missing_idx.size == 0:
            continue
        filled_profiles[missing_idx, angle_idx] = np.interp(missing_idx, valid_idx, column[valid_idx])

    normalized_profiles = np.zeros_like(filled_profiles)
    for idx in range(n_samples):
        row = filled_profiles[idx]
        valid = row > 0
        if not valid.any():
            continue
        slice_mean = mean_radius[idx] if mean_radius[idx] > 0 else float(row[valid].mean())
        if slice_mean > 0:
            normalized_profiles[idx] = row / slice_mean

    branch_confidence = float(slice_confidence.mean())
    feature_vector = _compute_branch_feature_vector(samples_world, filled_profiles, mean_radius)

    return BranchProfile(
        branch=branch,
        samples_world=samples_world,
        samples_voxel=samples_voxel,
        tangents=tangents,
        normals=normals,
        binormals=binormals,
        raw_profiles=filled_profiles,
        normalized_profiles=normalized_profiles,
        mean_radius=mean_radius,
        slice_confidence=slice_confidence,
        branch_confidence=branch_confidence,
        feature_vector=feature_vector,
        angles=angles.astype(np.float32),
    )


def _resample_polar_profiles(
    profiles: np.ndarray,
    angles: np.ndarray,
    *,
    upsample_factor: int,
    smoothing_sigma: float,
    min_radius_ratio: float,
    angular_gap_fill_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    upsample_factor = max(int(upsample_factor), 1)
    if upsample_factor == 1 and smoothing_sigma <= 0 and angular_gap_fill_bins <= 0:
        return profiles.astype(np.float32, copy=False), angles.astype(np.float32, copy=False)

    num_samples, num_bins = profiles.shape
    target_bins = max(num_bins * upsample_factor, 1)

    angles_mod = np.mod(angles + 2 * math.pi, 2 * math.pi)
    sort_idx = np.argsort(angles_mod)
    angles_sorted = angles_mod[sort_idx]

    target_angles_mod = np.linspace(0.0, 2 * math.pi, target_bins, endpoint=False, dtype=np.float64)
    target_angles = target_angles_mod.copy()
    target_angles[target_angles >= math.pi] -= 2 * math.pi

    resampled = np.zeros((num_samples, target_bins), dtype=np.float32)

    for sample_idx in range(num_samples):
        row = profiles[sample_idx, sort_idx]
        valid_mask = row > 0
        valid_count = int(valid_mask.sum())

        if valid_count == 0:
            continue
        if valid_count == 1:
            resampled[sample_idx] = row[valid_mask][0]
            continue

        theta_valid = angles_sorted[valid_mask].astype(np.float64)
        radii_valid = row[valid_mask].astype(np.float64)

        base_angle = theta_valid[0]
        theta_shifted = np.mod(theta_valid - base_angle, 2 * math.pi)
        theta_periodic = np.concatenate([theta_shifted, [theta_shifted[0] + 2 * math.pi]])
        radii_periodic = np.concatenate([radii_valid, [radii_valid[0]]])

        interpolator = interpolate.interp1d(
            theta_periodic,
            radii_periodic,
            kind="cubic" if valid_count >= 4 else "linear",
            assume_sorted=True,
            copy=False,
        )
        target_shifted = np.mod(target_angles_mod - base_angle, 2 * math.pi)
        sampled = interpolator(target_shifted)

        theta_extended = np.concatenate(
            [theta_shifted, theta_shifted + 2 * math.pi, theta_shifted - 2 * math.pi]
        )
        diff = np.abs(((target_shifted[:, None] - theta_extended[None, :]) + math.pi) % (2 * math.pi) - math.pi)
        angular_mask = (diff.min(axis=1) <= (2 * math.pi / max(num_bins, 1)) * 1.1)

        if angular_gap_fill_bins > 0 and angular_mask.any():
            structure = np.ones(max(int(angular_gap_fill_bins), 1), dtype=bool)
            angular_mask = ndimage.binary_closing(angular_mask, structure=structure)

        mask_float = angular_mask.astype(np.float32)

        if smoothing_sigma > 0:
            smooth_sigma = smoothing_sigma * upsample_factor
            weighted = gaussian_filter1d(sampled * mask_float, sigma=smooth_sigma, mode="wrap")
            norm = gaussian_filter1d(mask_float, sigma=smooth_sigma, mode="wrap")
            safe_norm = np.where(norm > 1e-6, norm, 1.0)
            sampled = weighted / safe_norm
            mask_float = np.where(norm > 1e-6, mask_float, 0.0)

        mean_radius = float(radii_valid.mean())
        min_radius = min_radius_ratio * mean_radius if mean_radius > 0 else 0.0
        sampled = np.where(mask_float > 0, np.clip(sampled, min_radius, None), 0.0)

        resampled[sample_idx] = sampled.astype(np.float32, copy=False)

    return resampled, target_angles.astype(np.float32, copy=False)


def _pad_array(arr: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if arr.size >= length:
        return arr[:length]
    return np.pad(arr, (0, length - arr.size), constant_values=0.0)


def _compute_branch_feature_vector(
    samples_world: np.ndarray,
    raw_profiles: np.ndarray,
    mean_radius: np.ndarray,
) -> np.ndarray:
    length_feature = np.array([_compute_length(samples_world)], dtype=np.float32)
    radius_stats = np.array(
        [
            float(mean_radius.mean()) if mean_radius.size else 0.0,
            float(mean_radius.min(initial=0.0)),
            float(mean_radius.max(initial=0.0)),
        ],
        dtype=np.float32,
    )

    centred = samples_world - samples_world.mean(axis=0)
    try:
        _, s, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        s = np.zeros(3, dtype=np.float32)
        vh = np.eye(3, dtype=np.float32)
    centreline_singular = _pad_array(s, 3)
    centreline_axes = _pad_array(vh.reshape(-1), 9)

    fft_components = []
    for row in raw_profiles:
        if (row > 0).any():
            fft_vals = np.fft.rfft(row)
            fft_components.append(np.abs(fft_vals[:8]))
    fourier_feature = (
        np.mean(fft_components, axis=0).astype(np.float32) if fft_components else np.zeros(8, dtype=np.float32)
    )

    return np.concatenate(
        [length_feature, radius_stats, centreline_singular, centreline_axes, fourier_feature]
    ).astype(np.float32)


def _refresh_branch_profile(profile: BranchProfile) -> None:
    raw = profile.raw_profiles.astype(np.float32, copy=False)
    num_samples = raw.shape[0]
    normalized = np.zeros_like(raw)
    mean_radius = np.zeros(num_samples, dtype=np.float32)
    slice_confidence = np.zeros(num_samples, dtype=np.float32)

    for idx, row in enumerate(raw):
        valid = row > 0
        if not valid.any():
            continue
        mean = float(row[valid].mean())
        mean_radius[idx] = mean
        slice_confidence[idx] = float(valid.mean())
        if mean > 0:
            normalized[idx] = row / mean

    profile.raw_profiles = raw
    profile.normalized_profiles = normalized
    profile.mean_radius = mean_radius
    profile.slice_confidence = slice_confidence
    profile.branch_confidence = float(slice_confidence.mean())
    profile.feature_vector = _compute_branch_feature_vector(profile.samples_world, raw, mean_radius)


# --------------------------------------------------------------------------- #
# Junction harmonisation
# --------------------------------------------------------------------------- #


def _harmonise_branch_junctions(
    profiles: List[BranchProfile],
    *,
    distance_threshold_mm: float,
    inherit_length_mm: float,
) -> None:
    if distance_threshold_mm <= 0 or inherit_length_mm <= 0:
        return

    metas: List[Dict[str, object]] = []
    endpoints: List[Tuple[int, int, np.ndarray]] = []

    for idx, profile in enumerate(profiles):
        samples = profile.samples_world
        if samples.shape[0] == 0:
            arclengths = np.zeros(0, dtype=np.float64)
            total_length = 0.0
        else:
            arclengths = _arclength(samples)
            total_length = float(arclengths[-1])

        valid_radius = profile.mean_radius[profile.mean_radius > 0]
        median_radius = float(np.median(valid_radius)) if valid_radius.size else 0.0

        metas.append(
            {
                "profile": profile,
                "arclengths": arclengths,
                "length": total_length,
                "median_radius": median_radius,
            }
        )

        if samples.shape[0]:
            endpoints.append((idx, 0, samples[0]))
            endpoints.append((idx, samples.shape[0] - 1, samples[-1]))

    processed: set[Tuple[int, int]] = set()

    for i in range(len(endpoints)):
        prof_i, sample_i, point_i = endpoints[i]
        for j in range(i + 1, len(endpoints)):
            prof_j, sample_j, point_j = endpoints[j]
            if prof_i == prof_j:
                continue
            distance = float(np.linalg.norm(point_i - point_j))
            if distance > distance_threshold_mm:
                continue

            meta_i = metas[prof_i]
            meta_j = metas[prof_j]
            length_i = meta_i["length"]
            length_j = meta_j["length"]

            parent_idx = prof_i
            child_idx = prof_j
            parent_sample = sample_i
            child_sample = sample_j

            if length_j > length_i:
                parent_idx, child_idx = child_idx, parent_idx
                parent_sample, child_sample = child_sample, parent_sample

            max_length = max(length_i, length_j)
            rel_diff = abs(length_i - length_j) / max_length if max_length > 0 else 0.0

            if rel_diff < 0.1:
                median_i = meta_i["median_radius"]
                median_j = meta_j["median_radius"]
                if median_j > median_i:
                    parent_idx, child_idx = child_idx, parent_idx
                    parent_sample, child_sample = child_sample, sample_i if child_idx == prof_i else sample_j

            key = (child_idx, child_sample)
            if key in processed:
                continue
            processed.add(key)

            parent_profile = metas[parent_idx]["profile"]
            child_profile = metas[child_idx]["profile"]
            if parent_profile.raw_profiles.size == 0 or child_profile.raw_profiles.size == 0:
                continue

            junction_point = point_j if child_idx == prof_j else point_i
            parent_points = parent_profile.samples_world
            parent_sample_idx = int(np.argmin(np.linalg.norm(parent_points - junction_point, axis=1)))
            parent_row = parent_profile.raw_profiles[parent_sample_idx]
            if not (parent_row > 0).any():
                continue

            child_meta = metas[child_idx]
            child_arclengths = child_meta["arclengths"]
            if child_arclengths.size == 0:
                continue

            inherit_limit = max(inherit_length_mm, 1e-3)
            if child_sample == 0:
                indices = np.where(child_arclengths <= inherit_limit)[0]
            else:
                indices = np.where((child_arclengths[-1] - child_arclengths) <= inherit_limit)[0]

            if indices.size == 0:
                indices = np.array([0 if child_sample == 0 else child_profile.raw_profiles.shape[0] - 1])

            for idx_sample in indices:
                if child_sample == 0:
                    dist = child_arclengths[idx_sample]
                else:
                    dist = child_arclengths[-1] - child_arclengths[idx_sample]
                weight = max(0.0, 1.0 - dist / inherit_limit)
                if weight <= 0:
                    continue
                child_row = child_profile.raw_profiles[idx_sample]
                blended = weight * parent_row + (1.0 - weight) * child_row
                child_profile.raw_profiles[idx_sample] = np.maximum(blended, 0.0).astype(np.float32)

            _refresh_branch_profile(child_profile)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def extract_branches(seg_path: str | Path, params: CentrelineParams | None = None) -> List[Branch]:
    params = params or CentrelineParams()
    seg_path = Path(seg_path)
    image = nib.load(str(seg_path))
    mask = image.get_fdata() > 0.5
    cleaned = _binary_cleanup(mask, params.closing_iterations)
    skeleton = skeletonize(cleaned)
    graph = _build_graph(skeleton)
    branches_idx = _trace_branches(graph)

    affine = image.affine
    inv_affine = np.linalg.inv(affine)
    spacing = image.header.get_zooms()[:3]
    branch_items: List[Dict[str, object]] = []

    for idx, path in enumerate(branches_idx):
        coords = np.vstack([graph.nodes[node]["coord"] for node in path])
        world_points = apply_affine(affine, coords).astype(np.float64)
        world_points = _deduplicate_points(world_points)
        if params.smooth_sigma_mm > 0:
            world_points = _smooth_centerline(world_points, params.smooth_sigma_mm)
        if (
            params.adaptive_max_step_mm > params.adaptive_min_step_mm
            and params.adaptive_min_step_mm > 0
            and params.curvature_alpha > 0
        ):
            world_points = _adaptive_resample_curve(
                world_points,
                params.adaptive_min_step_mm,
                params.adaptive_max_step_mm,
                params.curvature_alpha,
            )
        world_points = _deduplicate_points(world_points)
        length = _compute_length(world_points)
        if length < params.min_length_mm:
            continue
        voxel_points = apply_affine(inv_affine, world_points).astype(np.float64)
        branch = Branch(
            name=f"Branch_{idx:02d}",
            world_points=world_points,
            voxel_points=voxel_points,
            length_mm=length,
            source="observed",
        )
        branch_items.append(
            {
                "branch": branch,
                "start_node": int(path[0]),
                "end_node": int(path[-1]),
            }
        )

    branch_items = _prune_short_bridge_segments(
        branch_items,
        short_bridge_max_mm=params.short_bridge_max_mm,
    )
    branches = [item["branch"] for item in branch_items]

    if not branches:
        raise RuntimeError("No branches extracted. Verify segmentation quality and preprocessing steps.")

    branches.sort(key=lambda br: br.length_mm, reverse=True)
    spacing_mean = float(np.mean(spacing))
    for order, branch in enumerate(branches):
        branch.name = f"Branch_{order:02d}"
        if spacing_mean < 0.5 and branch.length_mm > 40:
            branch.name = f"Major_{order:02d}"

    return branches


def compute_polar_profiles(
    seg_path: str | Path,
    branches: List[Branch],
    profile_params: ProfileParams | None = None,
) -> List[BranchProfile]:
    params = profile_params or ProfileParams()
    seg_path = Path(seg_path)
    image = nib.load(str(seg_path))
    mask = image.get_fdata() > 0.5

    profiles: List[BranchProfile] = []
    for branch in branches:
        profile = _compute_profiles_for_branch(
            mask,
            image.affine,
            branch,
            num_samples=params.num_samples,
            num_angle_bins=params.num_angle_bins,
            patch_radius_mm=params.patch_radius_mm,
            half_thickness_mm=params.half_thickness_mm,
            radius_clip_factor=params.radius_clip_factor,
            endpoint_trim_mm=params.endpoint_trim_mm,
        )
        profiles.append(profile)

    _harmonise_branch_junctions(
        profiles,
        distance_threshold_mm=params.junction_distance_mm,
        inherit_length_mm=params.junction_inherit_mm,
    )
    return profiles


def export_branch_features(branch_profiles: List[BranchProfile], output_dir: str | Path) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_branches = []
    feature_matrix = []

    for profile in branch_profiles:
        branch_name = profile.branch.name
        branch_path = output_dir / f"{branch_name}.npz"
        np.savez(
            branch_path,
            branch_name=branch_name,
            samples_world=profile.samples_world,
            samples_voxel=profile.samples_voxel,
            tangents=profile.tangents,
            normals=profile.normals,
            binormals=profile.binormals,
            raw_profiles=profile.raw_profiles,
            normalized_profiles=profile.normalized_profiles,
            mean_radius=profile.mean_radius,
            slice_confidence=profile.slice_confidence,
            branch_confidence=profile.branch_confidence,
            feature_vector=profile.feature_vector,
            angles=profile.angles,
            length_mm=profile.branch.length_mm,
        )

        summary_branches.append(
            {
                "name": branch_name,
                "length_mm": profile.branch.length_mm,
                "confidence": profile.branch_confidence,
                "feature_file": branch_path.name,
            }
        )
        feature_matrix.append(profile.feature_vector)

    feature_matrix = np.vstack(feature_matrix) if feature_matrix else np.zeros((0, 1))
    global_descriptor = feature_matrix.mean(axis=0) if feature_matrix.size else np.zeros(1)
    np.save(output_dir / "global_descriptor.npy", global_descriptor)

    summary = {
        "branch_count": len(summary_branches),
        "branches": summary_branches,
        "global_descriptor": "global_descriptor.npy",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def reconstruct_from_features(
    features_dir: str | Path,
    output_mesh: Optional[str | Path] = None,
    recon_params: ReconstructionParams | None = None,
) -> "pyvista.PolyData":
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyvista is required for reconstruction. Install with `pip install pyvista`.") from exc

    params = recon_params or ReconstructionParams()
    features_dir = Path(features_dir)
    summary_path = features_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {features_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    combined_mesh: Optional[pv.PolyData] = None
    for branch_meta in summary.get("branches", []):
        branch_file = features_dir / branch_meta["feature_file"]
        data = np.load(branch_file)
        samples_world = data["samples_world"]
        tangents = data.get("tangents")
        normals = data.get("normals")
        binormals = data.get("binormals")
        raw_profiles = data["raw_profiles"]
        angles = data["angles"]

        original_samples = samples_world.shape[0]
        target = int(params.target_samples) if params.target_samples and params.target_samples > 1 else original_samples

        if target != original_samples:
            samples_world = _resample_curve(samples_world, target)
            tangents, normals, binormals = _local_frames(samples_world)
            s_old = _arclength(data["samples_world"])
            s_new = np.linspace(s_old[0], s_old[-1], target, dtype=np.float64)
            raw_profiles = _interpolate_profile_matrix(
                raw_profiles,
                s_old,
                s_new,
                smoothing_factor=params.smoothing_factor,
                min_valid=params.min_valid_slices,
                kind=params.interpolation_kind,
            )
        elif params.smoothing_factor > 0:
            s_old = _arclength(samples_world)
            raw_profiles = _interpolate_profile_matrix(
                raw_profiles,
                s_old,
                s_old,
                smoothing_factor=params.smoothing_factor,
                min_valid=params.min_valid_slices,
                kind=params.interpolation_kind,
            )

        raw_profiles, angles = _resample_polar_profiles(
            raw_profiles,
            angles,
            upsample_factor=params.angular_upsample,
            smoothing_sigma=params.angular_smoothing,
            min_radius_ratio=params.min_radius_ratio,
            angular_gap_fill_bins=params.angular_gap_fill_bins,
        )

        if params.axial_gap_fill > 0:
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
                    if gap_len <= params.axial_gap_fill:
                        prev_idx = gap_start - 1 if gap_start > 0 else None
                        next_idx = gap_end if gap_end < num_rows else None
                        if prev_idx is not None and next_idx is not None:
                            for k in range(gap_len):
                                t = (k + 1) / (gap_len + 1)
                                raw_profiles[gap_start + k] = (
                                    (1 - t) * raw_profiles[prev_idx] + t * raw_profiles[next_idx]
                                )
                            row_valid[gap_start:gap_end] = True
                        elif prev_idx is not None:
                            raw_profiles[gap_start:gap_end] = raw_profiles[prev_idx]
                            row_valid[gap_start:gap_end] = True
                        elif next_idx is not None:
                            raw_profiles[gap_start:gap_end] = raw_profiles[next_idx]
                            row_valid[gap_start:gap_end] = True
                raw_profiles = np.clip(raw_profiles, 0.0, None)

        raw_profiles = raw_profiles.astype(np.float32, copy=False)

        if tangents is None or normals is None or binormals is None or tangents.shape[0] != samples_world.shape[0]:
            tangents, normals, binormals = _local_frames(samples_world)

        num_samples, num_bins = raw_profiles.shape
        num_bins_ext = num_bins + 1

        points = np.tile(samples_world[np.newaxis, :, :], (num_bins_ext, 1, 1)).astype(np.float32)
        valid_points = np.zeros((num_bins_ext, num_samples), dtype=bool)
        for i in range(num_samples):
            center = samples_world[i]
            normal = normals[i]
            binormal = binormals[i]
            radii = raw_profiles[i]
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

            tck = interpolate.splrep(theta_periodic, radii_periodic, s=0, per=True, k=3)
            theta_dense = np.linspace(theta_valid[0], theta_valid[0] + 2 * math.pi, num_bins + 1, endpoint=True)
            radii_dense = np.clip(interpolate.splev(theta_dense, tck), 0.0, None)

            points_slice = (
                center
                + radii_dense[:, None] * np.cos(theta_dense)[:, None] * normal
                + radii_dense[:, None] * np.sin(theta_dense)[:, None] * binormal
            )
            points[: num_bins, i] = points_slice[:-1]
            points[num_bins, i] = points_slice[-1]
            valid_points[: num_bins + 1, i] = radii_dense > 0

        grid = pv.StructuredGrid()
        grid.points = points.reshape(-1, 3, order="F")
        grid.dimensions = (num_bins_ext, num_samples, 1)
        grid["valid"] = valid_points.reshape(-1, order="F").astype(np.uint8)
        surface = grid.extract_surface().triangulate()
        if "valid" in surface.point_data:
            mask = surface.point_data["valid"].astype(bool)
            surface = surface.extract_points(mask, adjacent_cells=True)
        if hasattr(surface, "point_data") and "valid" in surface.point_data:
            surface.point_data.pop("valid")
        if not isinstance(surface, pv.PolyData):
            surface = surface.extract_surface()
        surface = surface.clean()
        if surface.n_points < 3 or surface.n_cells == 0:
            continue

        if combined_mesh is None:
            combined_mesh = surface
        else:
            combined_mesh = combined_mesh.merge(surface, merge_points=True, tolerance=1e-3)

    if combined_mesh is None:
        raise RuntimeError("No branch meshes reconstructed.")

    combined_mesh = combined_mesh.clean()
    if output_mesh:
        combined_mesh.save(str(output_mesh))
    return combined_mesh


# --------------------------------------------------------------------------- #
# CLI utilities
# --------------------------------------------------------------------------- #


def _run_extract(args: argparse.Namespace) -> None:
    centreline_params = CentrelineParams(
        min_length_mm=args.min_length,
        short_bridge_max_mm=args.short_bridge_max_mm,
        closing_iterations=args.closing_iterations,
        smooth_sigma_mm=args.smooth_sigma_mm,
        adaptive_min_step_mm=args.adaptive_min_step,
        adaptive_max_step_mm=args.adaptive_max_step,
        curvature_alpha=args.adaptive_curvature_alpha,
    )
    branches = extract_branches(args.seg, centreline_params)

    profile_params = ProfileParams(
        num_samples=args.num_samples,
        num_angle_bins=args.num_angle_bins,
        patch_radius_mm=args.patch_radius,
        half_thickness_mm=args.half_thickness,
        radius_clip_factor=args.radius_clip_factor,
        endpoint_trim_mm=args.endpoint_trim_mm,
        junction_distance_mm=args.junction_distance,
        junction_inherit_mm=args.junction_inherit,
    )
    profiles = compute_polar_profiles(args.seg, branches, profile_params)
    summary = export_branch_features(profiles, args.out)
    print(json.dumps(summary, indent=2))


def _run_reconstruct(args: argparse.Namespace) -> None:
    recon_params = ReconstructionParams(
        target_samples=args.target_samples,
        smoothing_factor=args.smoothing_factor,
        min_valid_slices=args.min_valid_slices,
        interpolation_kind=args.interpolation_kind,
        angular_upsample=args.angular_upsample,
        angular_smoothing=args.angular_smoothing,
        min_radius_ratio=args.min_radius_ratio,
        angular_gap_fill_bins=args.angular_gap_fill,
        axial_gap_fill=args.axial_gap_fill,
    )
    mesh = reconstruct_from_features(args.features, args.output, recon_params)
    if args.output:
        print(f"Saved mesh to {args.output}")
    else:
        print("Reconstruction completed.", mesh.n_points, "points.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coronary branch extraction and reconstruction toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract branches and polar profiles.")
    extract_parser.add_argument("--seg", required=True, help="Path to segmentation NIfTI file.")
    extract_parser.add_argument("--out", required=True, help="Output directory for features.")
    extract_parser.add_argument("--min-length", type=float, default=5.0, help="Minimum branch length to keep (mm).")
    extract_parser.add_argument(
        "--short-bridge-max-mm",
        type=float,
        default=6.0,
        help="Remove short junction-to-junction bridge segments up to this length (<=0 disables).",
    )
    extract_parser.add_argument(
        "--closing-iterations",
        type=int,
        default=1,
        help="Binary closing iterations for mask cleanup.",
    )
    extract_parser.add_argument(
        "--smooth-sigma-mm",
        type=float,
        default=0.8,
        help="Gaussian smoothing sigma (mm) applied along each centreline.",
    )
    extract_parser.add_argument(
        "--adaptive-min-step",
        type=float,
        default=0.6,
        help="Minimum sampling step (mm) for curvature-adaptive resampling.",
    )
    extract_parser.add_argument(
        "--adaptive-max-step",
        type=float,
        default=2.5,
        help="Maximum sampling step (mm) for curvature-adaptive resampling.",
    )
    extract_parser.add_argument(
        "--adaptive-curvature-alpha",
        type=float,
        default=2.0,
        help="Curvature weight controlling adaptive resampling density.",
    )
    extract_parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of resampled points per branch.",
    )
    extract_parser.add_argument(
        "--num-angle-bins",
        type=int,
        default=72,
        help="Number of polar angle bins per cross-section.",
    )
    extract_parser.add_argument(
        "--patch-radius",
        type=float,
        default=3.0,
        help="Cross-section radius around the centreline sample (mm).",
    )
    extract_parser.add_argument(
        "--half-thickness",
        type=float,
        default=1.0,
        help="Half thickness of sampling slab along the tangent direction (mm).",
    )
    extract_parser.add_argument(
        "--radius-clip-factor",
        type=float,
        default=0.0,
        help="Max radius multiplier relative to slice median (0 disables clipping).",
    )
    extract_parser.add_argument(
        "--endpoint-trim-mm",
        type=float,
        default=0.0,
        help="Trim distance (mm) from branch endpoints to suppress junction artefacts.",
    )
    extract_parser.add_argument(
        "--junction-distance",
        type=float,
        default=1.5,
        help="Distance threshold (mm) for blending branch junction cross-sections.",
    )
    extract_parser.add_argument(
        "--junction-inherit",
        type=float,
        default=1.5,
        help="Arclength (mm) over which child branches inherit parent profiles near a junction.",
    )
    extract_parser.set_defaults(func=_run_extract)

    recon_parser = subparsers.add_parser("reconstruct", help="Reconstruct surface mesh from saved features.")
    recon_parser.add_argument(
        "--features",
        required=True,
        help="Directory containing per-branch feature `.npz` files and summary.json.",
    )
    recon_parser.add_argument("--output", help="Optional path to save reconstructed mesh (.vtp/.stl).")
    recon_parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Resampled points per branch before sweeping.",
    )
    recon_parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.0,
        help="Spline smoothing factor applied along branch arclength before sweep.",
    )
    recon_parser.add_argument(
        "--min-valid-slices",
        type=int,
        default=3,
        help="Minimum valid slices required to fit arclength smoothing spline.",
    )
    recon_parser.add_argument(
        "--interpolation-kind",
        choices=("linear", "quadratic", "cubic"),
        default="cubic",
        help="Fallback interpolation scheme when smoothing is disabled.",
    )
    recon_parser.add_argument(
        "--angular-upsample",
        type=int,
        default=1,
        help="Angular upsampling factor per slice before sweep.",
    )
    recon_parser.add_argument(
        "--angular-smoothing",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma (in angular bins) applied after upsampling.",
    )
    recon_parser.add_argument(
        "--min-radius-ratio",
        type=float,
        default=0.05,
        help="Minimum radius per slice as fraction of its mean radius to avoid spikes.",
    )
    recon_parser.add_argument(
        "--angular-gap-fill",
        type=int,
        default=0,
        help="Maximum angular gap (in bins) to fill within a slice during sweep.",
    )
    recon_parser.add_argument(
        "--axial-gap-fill",
        type=int,
        default=0,
        help="Maximum number of consecutive empty slices to interpolate along the branch.",
    )
    recon_parser.set_defaults(func=_run_reconstruct)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
