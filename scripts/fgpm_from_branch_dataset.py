"""
Fit a simple Fourier + GP taper prior from a branch_dataset npz, and reconstruct/visualize mean shape.

Pipeline:
1) Load branch_dataset npz (radii (B,K,M), tk (K,), angles (M,)).
2) Convert each slice radius r(theta) to Fourier coefficients up to order N.
3) Aggregate coefficients across branches (mean), optionally smooth along tk with an RBF GP.
4) Reconstruct mean radius profile and save npz/visualization.

This is a light-weight approximation of FGPM (Fourier + GP) suitable for quick exploration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def fourier_coeffs(r: np.ndarray, angles: np.ndarray, order: int) -> np.ndarray:
    """
    Compute Fourier coefficients for a single slice radius array r(theta).
    Returns [a0, a1..N, b1..N] of length (2*order + 1).
    """
    theta = angles.reshape(-1)
    r = r.reshape(-1)
    coeffs = [np.nanmean(r)]  # a0
    for n in range(1, order + 1):
        coeffs.append(2 * np.nanmean(r * np.cos(n * theta)))
        coeffs.append(2 * np.nanmean(r * np.sin(n * theta)))
    return np.array(coeffs, dtype=float)


def fourier_reconstruct(coeffs: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Reconstruct r(theta) from coeffs [a0, a1..N, b1..N].
    Returns array shape (len(angles),).
    """
    theta = angles.reshape(-1)
    order = (len(coeffs) - 1) // 2
    r = np.full_like(theta, coeffs[0], dtype=float)
    for n in range(1, order + 1):
        a_n = coeffs[2 * n - 1]
        b_n = coeffs[2 * n]
        r += a_n * np.cos(n * theta) + b_n * np.sin(n * theta)
    return r


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, amp: float, lengthscale: float) -> np.ndarray:
    x1 = x1[:, None]
    x2 = x2[None, :]
    dist2 = (x1 - x2) ** 2
    return (amp**2) * np.exp(-0.5 * dist2 / (lengthscale**2 + 1e-12))


def gp_smooth(y: np.ndarray, x: np.ndarray, lengthscale: float, noise: float) -> np.ndarray:
    """
    Smooth y(x) with an RBF GP prior. Returns posterior mean at x.
    amp is set to std(y); noise is additive diagonal term.
    """
    amp = float(np.nanstd(y) + 1e-6)
    x = x.reshape(-1)
    y = y.reshape(-1)
    K_base = rbf_kernel(x, x, amp=amp, lengthscale=lengthscale)
    K = K_base + noise * np.eye(len(x))
    alpha = np.linalg.solve(K, y)
    mean = K_base @ alpha
    return mean


def fit_model(radii: np.ndarray, tk: np.ndarray, angles: np.ndarray, order: int, lengthscale: float, noise: float):
    """
    Fit mean (and std) Fourier coefficients across branches, with GP smoothing along tk.
    Returns mean_coeffs, std_coeffs (both K,C) and reconstructed mean radii (K,M).
    """
    B, K, M = radii.shape
    C = 2 * order + 1
    coeffs_all = np.zeros((B, K, C), dtype=float)
    for b in range(B):
        for k in range(K):
            coeffs_all[b, k] = fourier_coeffs(radii[b, k], angles, order=order)
    mean_coeffs = np.nanmean(coeffs_all, axis=0)  # (K, C)
    std_coeffs = np.nanstd(coeffs_all, axis=0)    # (K, C)
    # GP smooth each coefficient over tk
    smoothed = np.zeros_like(mean_coeffs)
    smoothed_std = np.zeros_like(std_coeffs)
    for j in range(C):
        smoothed[:, j] = gp_smooth(mean_coeffs[:, j], tk, lengthscale=lengthscale, noise=noise)
        smoothed_std[:, j] = gp_smooth(std_coeffs[:, j], tk, lengthscale=lengthscale, noise=noise)

    # reconstruct mean radii
    radii_mean = np.zeros((K, M), dtype=float)
    for k in range(K):
        radii_mean[k] = fourier_reconstruct(smoothed[k], angles)
    return smoothed, smoothed_std, radii_mean


def save_visualization(radii_mean: np.ndarray, tk: np.ndarray, angles: np.ndarray, out_png: Path):
    plt.figure(figsize=(6, 4))
    # show radius as a function of xi (tk) and angle index
    im = plt.imshow(radii_mean, aspect="auto", origin="lower", extent=[0, len(angles), float(tk[0]), float(tk[-1])], cmap="magma")
    plt.colorbar(im, label="Radius (mm)")
    plt.xlabel("Angle bin")
    plt.ylabel("tk (0-1)")
    plt.title("Mean reconstructed radii")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fit Fourier+GP prior from branch_dataset npz.")
    parser.add_argument("--npz", type=Path, required=True, help="Branch dataset npz (can be aggregated).")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--order", type=int, default=6, help="Fourier order.")
    parser.add_argument("--lengthscale", type=float, default=0.15, help="GP RBF lengthscale on tk (0-1).")
    parser.add_argument("--noise", type=float, default=1e-3, help="GP noise term.")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    radii = data["radii"]  # (B,K,M)
    tk = data["tk"]
    angles = data["angles"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coeffs_mean, coeffs_std, radii_mean = fit_model(radii, tk, angles, order=args.order, lengthscale=args.lengthscale, noise=args.noise)
    np.savez(
        args.out_dir / "fgpm_model.npz",
        coeffs_mean=coeffs_mean,
        coeffs_std=coeffs_std,
        tk=tk,
        angles=angles,
        order=args.order,
        lengthscale=args.lengthscale,
        noise=args.noise,
        radii_mean=radii_mean,
    )
    save_visualization(radii_mean, tk, angles, args.out_dir / "radii_mean.png")
    print(f"Saved model to {args.out_dir}, coeffs_mean shape={coeffs_mean.shape}, radii_mean shape={radii_mean.shape}")


if __name__ == "__main__":
    main()
