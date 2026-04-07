"""
Sample a synthetic vessel from a FGPM prior (Fourier+GP) given desired length and optional initial section.

Inputs:
  --prior fgpm_model.npz (from fgpm_from_branch_dataset.py)
  --length_mm <float> desired physical length
  --noise_scale <float> multiplier on coeffs_std (0 => mean only)
  optional: --init_profile npy (angles length M) to anchor first section scale,
            or --r0 <float> to set mean radius at proximal end.

Outputs:
  npz containing: centerline (K,3), tk (K,), angles (M,), radii (K,M)
  heatmap PNG of radii along tk/angle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def fourier_reconstruct(coeffs: np.ndarray, angles: np.ndarray) -> np.ndarray:
    theta = angles.reshape(-1)
    order = (len(coeffs) - 1) // 2
    r = np.full_like(theta, coeffs[0], dtype=float)
    for n in range(1, order + 1):
        a_n = coeffs[2 * n - 1]
        b_n = coeffs[2 * n]
        r += a_n * np.cos(n * theta) + b_n * np.sin(n * theta)
    return r


def scale_to_init(radii: np.ndarray, init_profile: np.ndarray | None, r0: float | None) -> np.ndarray:
    radii = radii.copy()
    if init_profile is not None:
        tgt = np.nanmean(init_profile)
    elif r0 is not None:
        tgt = float(r0)
    else:
        return radii
    src = np.nanmean(radii[0])
    if src > 1e-6:
        scale = tgt / src
        radii *= scale
    return radii


def main():
    parser = argparse.ArgumentParser(description="Sample vessel from FGPM prior.")
    parser.add_argument("--prior", type=Path, required=True, help="fgpm_model.npz path")
    parser.add_argument("--length_mm", type=float, required=True, help="Desired vessel length (mm)")
    parser.add_argument("--noise_scale", type=float, default=0.5, help="Scale for coeffs_std noise (0 => mean only)")
    parser.add_argument("--init_profile", type=Path, default=None, help="Optional npy of shape (M,) for proximal profile")
    parser.add_argument("--r0", type=float, default=None, help="Optional proximal mean radius override")
    parser.add_argument("--K", type=int, default=None, help="Override number of samples along centerline (higher=>更高采样密度)")
    parser.add_argument("--out", type=Path, default=Path("outputs/sample_vessel.npz"))
    args = parser.parse_args()

    prior = np.load(args.prior)
    coeffs_mean = prior["coeffs_mean"]
    coeffs_std = prior["coeffs_std"]
    tk = prior["tk"]
    angles = prior["angles"]
    order = int(prior["order"])

    K_orig, C = coeffs_mean.shape
    K = args.K or K_orig
    if K != K_orig:
        tk_new = np.linspace(0, 1, K)
        coeffs_mean = np.array([np.interp(tk_new, tk, coeffs_mean[:, j]) for j in range(C)]).T
        coeffs_std = np.array([np.interp(tk_new, tk, coeffs_std[:, j]) for j in range(C)]).T
        tk = tk_new

    # sample coeffs
    noise = np.random.randn(K, C) * args.noise_scale * coeffs_std
    coeffs = coeffs_mean + noise
    # reconstruct radii
    radii = np.zeros((K, len(angles)), dtype=float)
    for k in range(K):
        radii[k] = fourier_reconstruct(coeffs[k], angles)

    # scale to desired initial profile / r0
    init_prof = np.load(args.init_profile) if args.init_profile is not None else None
    radii = scale_to_init(radii, init_prof, args.r0)

    # build straight centerline of given length
    centerline = np.zeros((K, 3), dtype=float)
    centerline[:, 0] = np.linspace(0, args.length_mm, K)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, centerline=centerline, tk=tk, angles=angles, radii=radii, coeffs=coeffs, order=order)

    # heatmap
    plt.figure(figsize=(6, 4))
    im = plt.imshow(radii, aspect="auto", origin="lower", extent=[0, len(angles), float(tk[0]), float(tk[-1])], cmap="magma")
    plt.colorbar(im, label="Radius (mm)")
    plt.xlabel("Angle bin")
    plt.ylabel("tk (0-1)")
    plt.title("Sampled vessel radii")
    plt.tight_layout()
    plt.savefig(args.out.with_suffix(".png"), dpi=200)
    plt.close()

    print(f"Saved sampled vessel to {args.out} | centerline {centerline.shape}, radii {radii.shape}, noise_scale={args.noise_scale}")


if __name__ == "__main__":
    main()
