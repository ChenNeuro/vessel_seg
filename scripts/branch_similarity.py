"""
Compute branch similarity and PCA latents from a branch_dataset npz.

Inputs: npz from build_branch_dataset.py (radii (B,K,M), meta, branch_ids).
Outputs:
- distances.csv: pairwise L2 distances on flattened radii or PCA latents
- pca_latents.npy: (B, d) latent vectors
- optional heatmap PNG

Usage:
  conda activate vessel_seg
  python scripts/branch_similarity.py \
      --npz outputs/normal1_branch_dataset.npz \
      --out_dir outputs/normal1_similarity \
      --pca_dim 8
"""

from pathlib import Path
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt


def normalize_radii(radii: np.ndarray, mode: str) -> np.ndarray:
    """Normalize radii tensor (B,K,M)."""
    if mode == "none":
        return radii
    if mode == "branch_mean":
        means = radii.mean(axis=(1, 2), keepdims=True) + 1e-8
        return radii / means
    if mode == "global_mean":
        mean = radii.mean() + 1e-8
        return radii / mean
    raise ValueError(f"Unknown normalize mode: {mode}")


def compute_pca(X: np.ndarray, d: int):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:d]
    latents = Xc @ comps.T
    return latents, comps, X.mean(axis=0)


def adaptive_compress(radii: np.ndarray, beta_min: float, tau: float) -> np.ndarray:
    """
    Adaptive exponent compression: bigger radii get more compression (smaller exponent), small radii stay closer to 1.
    radii are assumed already normalized (e.g., branch_mean).
    """
    r = np.clip(radii, 1e-6, None)
    beta = beta_min + (1.0 - beta_min) * np.exp(-np.square(r / (tau + 1e-8)))
    return np.power(r, beta)


def fit_exponential_taper(xi: np.ndarray, radii: np.ndarray) -> tuple[float, float]:
    """Fit r(xi)=r0*exp(-alpha*xi); returns (r0, alpha)."""
    xi = np.asarray(xi, dtype=float).reshape(-1)
    r = np.asarray(radii, dtype=float).reshape(-1)
    valid = np.isfinite(r) & (r > 0) & np.isfinite(xi)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    x = xi[valid]
    y = np.log(r[valid] + 1e-8)
    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a0, a1 = coef
    r0 = float(np.exp(a0))
    alpha = float(-a1)
    return r0, alpha


def evaluate_exponential_taper(xi: np.ndarray, r0: float, alpha: float) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    return r0 * np.exp(-alpha * xi)


def remove_taper(radii: np.ndarray, tk: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Remove per-branch taper using mean radius along angles.
    radii: (B,K,M), tk: (K,)
    Returns adjusted radii and stats dict.
    """
    B, K, M = radii.shape
    tk = np.asarray(tk, dtype=float).reshape(-1)
    if tk.shape[0] != K:
        raise ValueError("tk length must match K dimension of radii")
    r0_list = []
    alpha_list = []
    adjusted = np.empty_like(radii)
    for i in range(B):
        r_branch = radii[i]
        mean_r = np.nanmean(r_branch, axis=1)
        r0, alpha = fit_exponential_taper(tk, mean_r)
        if not np.isfinite(r0):
            r0, alpha = float(np.nanmean(mean_r)), 0.0
        baseline = evaluate_exponential_taper(tk, r0, alpha)
        adjusted[i] = r_branch - baseline[:, None]
        r0_list.append(r0)
        alpha_list.append(alpha)
    stats = {
        "r0_mean": float(np.nanmean(r0_list)),
        "alpha_mean": float(np.nanmean(alpha_list)),
        "r0_std": float(np.nanstd(r0_list)),
        "alpha_std": float(np.nanstd(alpha_list)),
    }
    return adjusted, stats


def main():
    parser = argparse.ArgumentParser(description="Branch similarity & PCA.")
    parser.add_argument("--npz", type=Path, default=None, help="branch_dataset npz")
    parser.add_argument("--case", type=str, default="Normal_1", help="Case name for default paths")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output dir (default outputs/<case>/similarity)")
    parser.add_argument("--pca_dim", type=int, default=8)
    parser.add_argument("--heatmap", action="store_true", help="Save distance heatmap PNG")
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "branch_mean", "global_mean"],
        help="Normalization for radii before PCA.",
    )
    parser.add_argument(
        "--length_detrend",
        type=str,
        default=None,
        help="Optional length-radius幂次去趋势系数，提供数值或 'auto' 使用 log-log 回归拟合 alpha 并除以 L^alpha。",
    )
    parser.add_argument("--length_log", action="store_true", help="Apply log1p to branch lengths before stats / detrend fitting.")
    parser.add_argument("--adaptive_beta_min", type=float, default=None, help="Enable adaptive radius compression with this minimum exponent (e.g., 0.6).")
    parser.add_argument("--adaptive_tau", type=float, default=1.0, help="Tau controlling transition for adaptive compression (in normalized radius units).")
    parser.add_argument("--vmin_quantile", type=float, default=0.05, help="Quantile for heatmap lower bound (e.g., 0.05).")
    parser.add_argument("--vmax_quantile", type=float, default=0.95, help="Quantile for heatmap upper bound (e.g., 0.95).")
    parser.add_argument("--remove_taper", action="store_true", help="Fit per-branch exponential taper along tk and subtract baseline before PCA.")
    args = parser.parse_args()

    if args.npz is None:
        args.npz = Path(f"outputs/{args.case}/branch_dataset.npz")
    if args.out_dir is None:
        args.out_dir = Path(f"outputs/{args.case}/similarity")

    data = np.load(args.npz, allow_pickle=True)
    radii = data["radii"]  # (B,K,M)
    branch_ids = data["branch_ids"]
    meta = data.get("meta", None)
    norm_radii = normalize_radii(radii, args.normalize)
    r_mean = float(norm_radii.mean())
    r_std = float(norm_radii.std())
    lengths = None
    alpha_used = None
    taper_stats = None
    if args.remove_taper:
        norm_radii, taper_stats = remove_taper(norm_radii, data["tk"])
    if meta is not None and len(meta) > 0:
        try:
            lengths = np.array([m.get("length_mm") for m in meta if "length_mm" in m], dtype=float)
            if args.length_log:
                lengths = np.log1p(lengths)
            if args.length_detrend:
                valid = lengths > 0
                lengths_valid = lengths[valid]
                mean_r = norm_radii.mean(axis=(1, 2))[valid]
                if args.length_detrend.lower() == "auto":
                    # log-log 回归拟合 alpha
                    coeffs = np.polyfit(np.log(lengths_valid + 1e-8), np.log(mean_r + 1e-8), 1)
                    alpha_used = coeffs[0]
                else:
                    alpha_used = float(args.length_detrend)
                factors = np.ones_like(lengths)
                factors[valid] = np.power(lengths_valid, alpha_used)
                norm_radii = norm_radii / factors[:, None, None]
        except Exception:
            lengths = None
    adaptive_used = None
    if args.adaptive_beta_min is not None:
        adaptive_used = (args.adaptive_beta_min, args.adaptive_tau)
        norm_radii = adaptive_compress(norm_radii, args.adaptive_beta_min, args.adaptive_tau)

    B, K, M = radii.shape
    X = norm_radii.reshape(B, K * M)
    latents, comps, mean_vec = compute_pca(X, d=min(args.pca_dim, min(X.shape)))
    dists = np.linalg.norm(latents[:, None, :] - latents[None, :, :], axis=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "pca_latents.npy", latents)
    np.save(args.out_dir / "pca_components.npy", comps)
    np.save(args.out_dir / "pca_mean.npy", mean_vec)
    # save distances as CSV
    with open(args.out_dir / "distances.csv", "w") as f:
        header = ",".join(["branch_id"] + [str(bid) for bid in branch_ids])
        f.write(header + "\n")
        for i, bid in enumerate(branch_ids):
            row = ",".join([str(bid)] + [f"{dists[i,j]:.4f}" for j in range(B)])
            f.write(row + "\n")

    if args.heatmap:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(dists, cmap="viridis")
        plt.colorbar(im, shrink=0.7, label="PCA latent L2")
        plt.xticks(range(B), branch_ids, rotation=90)
        plt.yticks(range(B), branch_ids)
        plt.title("Branch similarity (PCA latent L2)")
        # annotate key params on top-right
        offset_mm = None
        offset_pct = None
        if meta is not None and len(meta) > 0:
            try:
                offset_mm_set = {m.get("start_offset_mm") for m in meta if "start_offset_mm" in m}
                offset_pct_set = {m.get("start_offset_percent") for m in meta if "start_offset_percent" in m}
                offset_mm = offset_mm_set.pop() if len(offset_mm_set) == 1 else None
                offset_pct = offset_pct_set.pop() if len(offset_pct_set) == 1 else None
            except Exception:
                pass
        lines = [
            f"K={K}, M={M}, d={latents.shape[1]}",
            f"norm={args.normalize}",
            f"r μ={r_mean:.2f}, σ={r_std:.2f}",
        ]
        if offset_mm is not None or offset_pct is not None:
            lines.append(
                "offset=" +
                (f"{offset_mm:.2f}mm" if offset_mm is not None else "") +
                (" / " if offset_mm is not None and offset_pct is not None else "") +
                (f"{offset_pct}%" if offset_pct is not None else "")
            )
        if lengths is not None and lengths.size > 0:
            suffix = " (log1p)" if args.length_log else " mm"
            lines.append(f"L μ={lengths.mean():.1f}{suffix}, σ={lengths.std():.1f}")
        if alpha_used is not None:
            lines.append(f"L detrend α={alpha_used:.3f}")
        if adaptive_used is not None:
            lines.append(f"adaptive beta_min={adaptive_used[0]:.2f}, tau={adaptive_used[1]:.2f}")
        if taper_stats is not None:
            lines.append(f"taper r0 μ={taper_stats['r0_mean']:.2f}, α μ={taper_stats['alpha_mean']:.2f}")
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
        plt.savefig(args.out_dir / "distance_heatmap.png", dpi=200)
        plt.close()

    print(f"Saved latents and distances to {args.out_dir} (branches={B}, K={K}, M={M})")


if __name__ == "__main__":
    main()
