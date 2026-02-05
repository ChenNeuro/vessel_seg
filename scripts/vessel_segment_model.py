"""
Vessel segment modeling: exponential taper + residual radius + fixed-length features.

This module:
- Computes arc-length and normalized coordinate xi ∈ [0, 1].
- Fits an exponential taper baseline r(xi) = r0 * exp(-alpha * xi).
- Computes residual radii.
- Builds a fixed-length feature vector for ML (length, r0, alpha, sampled residuals).

Robustness:
- Ignores invalid radii (≤0 or NaN) when fitting.
- Handles very short segments.
- Interpolates residuals to a fixed grid, tolerating small NaN counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


SMALL_EPS = 1e-8


@dataclass
class TaperModel:
    r0: float
    alpha: float


def compute_arc_length_and_xi(centerline_points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Given centerline points (N,3), compute cumulative arc-length s, total length L, and xi=s/L.
    """
    pts = np.asarray(centerline_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("centerline_points must have shape (N,3)")
    if len(pts) < 2:
        s = np.zeros(len(pts), dtype=float)
        return s, 0.0, s
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(diffs)])
    L = float(s[-1])
    xi = s / (L + SMALL_EPS)
    return s, L, xi


def fit_exponential_taper(xi: np.ndarray, radii: np.ndarray) -> Tuple[float, float]:
    """
    Fit r(xi) = r0 * exp(-alpha * xi) via linear regression in log-space.
    Returns (r0, alpha).
    """
    xi = np.asarray(xi, dtype=float).reshape(-1)
    r = np.asarray(radii, dtype=float).reshape(-1)
    valid = np.isfinite(r) & (r > 0) & np.isfinite(xi)
    if valid.sum() < 2:
        return float(np.nan), float(np.nan)
    x = xi[valid]
    y = np.log(r[valid] + SMALL_EPS)
    # Linear regression y ≈ a0 + a1*x
    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a0, a1 = coef
    r0 = math.exp(a0)
    alpha = -a1
    return r0, alpha


def evaluate_exponential_taper(xi: np.ndarray, r0: float, alpha: float) -> np.ndarray:
    """Evaluate r_baseline(xi) = r0 * exp(-alpha * xi)."""
    xi = np.asarray(xi, dtype=float)
    return r0 * np.exp(-alpha * xi)


def compute_radius_residuals(xi: np.ndarray, radii: np.ndarray, r0: float, alpha: float) -> np.ndarray:
    """Compute delta_r = r - r_baseline(xi)."""
    xi = np.asarray(xi, dtype=float)
    r = np.asarray(radii, dtype=float)
    baseline = evaluate_exponential_taper(xi, r0, alpha)
    return r - baseline


def build_segment_feature_vector(
    centerline_points: np.ndarray,
    radii: np.ndarray,
    num_residual_samples: int = 64,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a fixed-length feature vector for a vessel segment.

    features = [L, r0, alpha, sampled_delta_r (num_residual_samples)]
    """
    s, L, xi = compute_arc_length_and_xi(centerline_points)
    r0, alpha = fit_exponential_taper(xi, radii)
    if not np.isfinite(r0) or not np.isfinite(alpha):
        # Fallback: use first radius, zero taper
        r0 = float(np.nanmean(radii))
        alpha = 0.0
    delta_r = compute_radius_residuals(xi, radii, r0, alpha)

    # Interpolate residuals to a fixed grid in xi
    xi_grid = np.linspace(0.0, 1.0, num_residual_samples)
    # Handle NaNs by temporary filling with zero before interp
    delta_clean = delta_r.copy()
    nan_mask = ~np.isfinite(delta_clean)
    if nan_mask.any():
        delta_clean[nan_mask] = 0.0
    sampled = np.interp(xi_grid, xi, delta_clean)

    features = np.concatenate([[L, r0, alpha], sampled.astype(float)])
    meta = {"L": float(L), "r0": float(r0), "alpha": float(alpha), "num_samples": int(num_residual_samples)}
    return features, meta


if __name__ == "__main__":
    # Demo: synthetic straight vessel with exponential taper + sinusoidal perturbation
    N = 100
    # Centerline: straight along x
    x = np.linspace(0, 50, N)
    centerline = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    # Ground-truth taper
    r0_true = 3.0
    alpha_true = 0.8
    _, _, xi = compute_arc_length_and_xi(centerline)
    radii_baseline = evaluate_exponential_taper(xi, r0_true, alpha_true)
    # Add small sinusoidal residual
    residual = 0.2 * np.sin(4 * np.pi * xi)
    radii_noisy = radii_baseline + residual

    # Run pipeline
    s, L, xi = compute_arc_length_and_xi(centerline)
    r0_fit, alpha_fit = fit_exponential_taper(xi, radii_noisy)
    delta_r = compute_radius_residuals(xi, radii_noisy, r0_fit, alpha_fit)
    features, meta = build_segment_feature_vector(centerline, radii_noisy, num_residual_samples=64)

    print(f"GT r0={r0_true:.3f}, alpha={alpha_true:.3f}")
    print(f"Fit r0={r0_fit:.3f}, alpha={alpha_fit:.3f}")
    print(f"Length L={L:.3f}")
    print(f"Feature vector shape: {features.shape}, meta: {meta}")
