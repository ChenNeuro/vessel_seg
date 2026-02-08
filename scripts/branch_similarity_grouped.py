"""
Compute PCA/距离热力图，按分支长度分层（基于 quantile）。

示例：
  python scripts/branch_similarity_grouped.py \
    --npz outputs/aggregate/branch_dataset_normal_pct2p5.npz \
    --out_dir outputs/aggregate/branch_dataset_normal_pct2p5_groups \
    --quantiles 0.33 0.66 \
    --normalize branch_mean \
    --adaptive_beta_min 0.6 --adaptive_tau 1.0 \
    --pca_dim 8 --heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def compute_pca(X: np.ndarray, d: int):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:d]
    latents = Xc @ comps.T
    return latents, comps, X.mean(axis=0)


def normalize_radii(radii: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return radii
    if mode == "branch_mean":
        means = radii.mean(axis=(1, 2), keepdims=True) + 1e-8
        return radii / means
    if mode == "global_mean":
        mean = radii.mean() + 1e-8
        return radii / mean
    raise ValueError(f"Unknown normalize mode: {mode}")


def adaptive_compress(radii: np.ndarray, beta_min: float, tau: float) -> np.ndarray:
    r = np.clip(radii, 1e-6, None)
    beta = beta_min + (1.0 - beta_min) * np.exp(-np.square(r / (tau + 1e-8)))
    return np.power(r, beta)


def run_group(radii, lengths, meta, ids, args, group_name: str, out_root: Path):
    B, K, M = radii.shape
    if B < 2:
        return
    norm_radii = normalize_radii(radii, args.normalize)
    alpha_used = None
    if args.length_detrend and lengths is not None:
        valid = lengths > 0
        if np.any(valid):
            lengths_valid = lengths[valid]
            mean_r = norm_radii.mean(axis=(1, 2))[valid]
            if args.length_detrend.lower() == "auto":
                coeffs = np.polyfit(np.log(lengths_valid + 1e-8), np.log(mean_r + 1e-8), 1)
                alpha_used = coeffs[0]
            else:
                alpha_used = float(args.length_detrend)
            factors = np.ones_like(lengths)
            factors[valid] = np.power(lengths_valid, alpha_used)
            norm_radii = norm_radii / factors[:, None, None]
    adaptive_used = None
    if args.adaptive_beta_min is not None:
        adaptive_used = (args.adaptive_beta_min, args.adaptive_tau)
        norm_radii = adaptive_compress(norm_radii, args.adaptive_beta_min, args.adaptive_tau)

    X = norm_radii.reshape(B, K * M)
    d = min(args.pca_dim, min(X.shape))
    latents, comps, mean_vec = compute_pca(X, d=d)
    out_dir = out_root / f"{group_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dists = np.linalg.norm(latents[:, None, :] - latents[None, :, :], axis=2)
    np.save(out_dir / "pca_latents.npy", latents)
    np.save(out_dir / "pca_components.npy", comps)
    np.save(out_dir / "pca_mean.npy", mean_vec)
    with open(out_dir / "distances.csv", "w") as f:
        header = ",".join(["branch_id"] + [bid for bid in ids])
        f.write(header + "\n")
        for i, bid in enumerate(ids):
            row = ",".join([bid] + [f"{dists[i,j]:.4f}" for j in range(B)])
            f.write(row + "\n")
    if args.heatmap:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        vmin = np.quantile(dists, args.vmin_quantile)
        vmax = np.quantile(dists, args.vmax_quantile)
        im = ax.imshow(dists, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, shrink=0.7, label="PCA latent L2")
        plt.xticks(range(B), ids, rotation=90, fontsize=6)
        plt.yticks(range(B), ids, fontsize=6)
        plt.title(f"{group_name} similarity")
        lines = [
            f"B={B}, K={K}, M={M}, d={d}",
            f"norm={args.normalize}",
            f"r μ={norm_radii.mean():.2f}, σ={norm_radii.std():.2f}",
        ]
        if lengths is not None and lengths.size > 0:
            lines.append(f"L μ={lengths.mean():.1f}, σ={lengths.std():.1f}")
        if alpha_used is not None:
            lines.append(f"L detrend α={alpha_used:.3f}")
        if adaptive_used is not None:
            lines.append(f"adaptive beta_min={adaptive_used[0]:.2f}, tau={adaptive_used[1]:.2f}")
        ax.text(
            0.98,
            0.02,
            "\n".join(lines),
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        plt.tight_layout()
        plt.savefig(out_dir / "distance_heatmap.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Length-grouped branch similarity & PCA.")
    parser.add_argument("--npz", type=Path, required=True, help="Aggregated branch_dataset npz.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output root dir for grouped results.")
    parser.add_argument("--quantiles", type=float, nargs="*", default=[0.33, 0.66], help="Quantiles for length split (0-1).")
    parser.add_argument("--pca_dim", type=int, default=8)
    parser.add_argument("--normalize", type=str, default="branch_mean", choices=["none", "branch_mean", "global_mean"])
    parser.add_argument("--length_detrend", type=str, default=None, help="Optional alpha or 'auto' for L^alpha detrend.")
    parser.add_argument("--adaptive_beta_min", type=float, default=None, help="Adaptive compression min exponent.")
    parser.add_argument("--adaptive_tau", type=float, default=1.0, help="Adaptive compression tau.")
    parser.add_argument("--vmin_quantile", type=float, default=0.05)
    parser.add_argument("--vmax_quantile", type=float, default=0.95)
    parser.add_argument("--heatmap", action="store_true")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    radii = data["radii"]
    meta = data["meta"]
    ids = data["branch_ids"]
    lengths = np.array([m.get("length_mm", np.nan) for m in meta], dtype=float)
    if lengths.size == 0:
        print("No lengths found in meta.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # quantile bounds
    qs = sorted(args.quantiles)
    bounds = [-np.inf] + list(np.quantile(lengths, qs)) + [np.inf]
    names = []
    for i in range(len(bounds) - 1):
        if i == 0:
            names.append("short")
        elif i == len(bounds) - 2:
            names.append("long")
        else:
            names.append(f"group{i}")

    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (lengths > lo) & (lengths <= hi)
        if mask.sum() < 2:
            continue
        run_group(radii[mask], lengths[mask], [meta[j] for j in range(len(meta)) if mask[j]], [ids[j] for j in range(len(ids)) if mask[j]], args, names[i], args.out_dir)


if __name__ == "__main__":
    main()
