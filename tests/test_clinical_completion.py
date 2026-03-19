from __future__ import annotations

import math

import numpy as np

from vessel_seg.clinical_completion import compute_branch_metrics, compute_longitudinal_section_map
from vessel_seg.shape import Branch, BranchProfile


def _toy_profile(radius: float = 2.0) -> BranchProfile:
    num_samples = 5
    num_angles = 8
    samples = np.stack(
        [np.linspace(0.0, 4.0, num_samples), np.zeros(num_samples), np.zeros(num_samples)],
        axis=1,
    ).astype(np.float32)
    tangents = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (num_samples, 1))
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (num_samples, 1))
    binormals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (num_samples, 1))
    raw = np.full((num_samples, num_angles), radius, dtype=np.float32)
    normalized = np.ones_like(raw)
    mean_radius = np.full(num_samples, radius, dtype=np.float32)
    confidence = np.ones(num_samples, dtype=np.float32)
    branch = Branch(
        name="ToyBranch",
        world_points=samples,
        voxel_points=samples,
        length_mm=4.0,
        source="test",
    )
    return BranchProfile(
        branch=branch,
        samples_world=samples,
        samples_voxel=samples,
        tangents=tangents,
        normals=normals,
        binormals=binormals,
        raw_profiles=raw,
        normalized_profiles=normalized,
        mean_radius=mean_radius,
        slice_confidence=confidence,
        branch_confidence=1.0,
        feature_vector=np.zeros(4, dtype=np.float32),
        angles=np.linspace(-math.pi, math.pi, num_angles, endpoint=False, dtype=np.float32),
    )


def test_compute_branch_metrics_geometry() -> None:
    profile = _toy_profile(radius=2.0)
    metrics = compute_branch_metrics(profile, high_hu_threshold=350.0)
    assert metrics.branch_name == "ToyBranch"
    assert metrics.mean_diameter_mm == 4.0
    assert metrics.min_diameter_mm == 4.0
    assert metrics.length_mm == 4.0
    assert abs(metrics.mean_area_mm2 - math.pi * 4.0) < 0.5
    assert abs(metrics.tortuosity_index) < 1e-6
    assert metrics.stenosis_pct == 0.0


def test_compute_branch_metrics_wall_stats() -> None:
    profile = _toy_profile(radius=1.5)
    wall_map = np.full((8, 5), 200.0, dtype=np.float32)
    wall_map[:2, :2] = 400.0
    metrics = compute_branch_metrics(profile, wall_map=wall_map, high_hu_threshold=350.0)
    assert metrics.wall_hu_mean is not None
    assert metrics.wall_hu_mean > 200.0
    assert metrics.wall_hu_std is not None
    assert metrics.high_hu_wall_fraction is not None
    assert 0.0 < metrics.high_hu_wall_fraction < 1.0


def test_compute_longitudinal_section_map_shape_and_values() -> None:
    profile = _toy_profile(radius=1.5)
    volume = np.full((16, 16, 16), 123.0, dtype=np.float32)
    affine = np.eye(4, dtype=np.float64)
    section = compute_longitudinal_section_map(
        volume,
        affine,
        profile,
        width_mm=6.0,
        slab_mm=2.0,
        out_width_px=32,
    )
    assert section.shape == (32, profile.samples_world.shape[0])
    finite = section[np.isfinite(section)]
    assert finite.size > 0
    assert np.allclose(finite, 123.0)
