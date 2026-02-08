"""
Aggregate multiple branch_dataset npz files into a single bundle (for cross-case PCA/similarity).

Example:
  python scripts/aggregate_branch_datasets.py \\
    --npz_glob 'outputs/Normal_*/branch_dataset*.npz' \\
    --out outputs/aggregate/branch_dataset_all.npz \\
    --pca_dim 8 --heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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
    """
    Adaptive exponent compression: bigger radii get more压缩（较小指数），小半径保留。
    radii 假定已归一化（如 branch_mean）。
    """
    r = np.clip(radii, 1e-6, None)
    beta = beta_min + (1.0 - beta_min) * np.exp(-np.square(r / (tau + 1e-8)))
    return np.power(r, beta)


def main():
    parser = argparse.ArgumentParser(description="Aggregate branch_dataset npz files across cases.")
    parser.add_argument("--npz_glob", type=str, default="outputs/Normal_*/branch_dataset*.npz", help="Glob to find npz files when no explicit list is provided.")
    parser.add_argument("npzs", nargs="*", help="Optional explicit list of npz files.")
    parser.add_argument("--out", type=Path, default=Path("outputs/aggregate/branch_dataset_all.npz"), help="Output npz path.")
    parser.add_argument("--pca_dim", type=int, default=None, help="Optional PCA dimension to compute on aggregated radii.")
    parser.add_argument("--heatmap", action="store_true", help="If PCA is computed, also save distance heatmap.")
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
    parser.add_argument("--length_log", action="store_true", help="Apply log1p to lengths before stats / detrend fitting.")
    parser.add_argument("--adaptive_beta_min", type=float, default=None, help="Enable adaptive radius compression with this minimum exponent (e.g., 0.6).")
    parser.add_argument("--adaptive_tau", type=float, default=1.0, help="Tau controlling transition for adaptive compression.")
    parser.add_argument("--vmin_quantile", type=float, default=0.05, help="Quantile for heatmap lower bound.")
    parser.add_argument("--vmax_quantile", type=float, default=0.95, help="Quantile for heatmap upper bound.")
    parser.add_argument(
        "--length_group_quantiles",
        type=float,
        nargs="*",
        default=None,
        help="Optional quantiles to split lengths for分层建模，例：0.33 0.66 → short/mid/long。",
    )
    args = parser.parse_args()

    files: List[Path]
    if args.npzs:
        files = [Path(p) for p in args.npzs]
    else:
        files = sorted(Path(".").glob(args.npz_glob))
    if not files:
        print("No npz files found to aggregate.")
        return

    radii_all = []
    meta_all = []
    branch_ids_all = []
    case_ids = []
    align_shifts = []
    tk_ref = None
    angles_ref = None

    for npz_path in files:
        data = np.load(npz_path, allow_pickle=True)
        radii = data["radii"]
        if radii.size == 0:
            continue
        if tk_ref is None:
            tk_ref = data["tk"]
            angles_ref = data["angles"]
        else:
            if not np.allclose(tk_ref, data["tk"]):
                print(f"Warning: tk mismatch in {npz_path}, using first as reference.")
            if not np.allclose(angles_ref, data["angles"]):
                print(f"Warning: angles mismatch in {npz_path}, using first as reference.")
        meta = data["meta"]
        case = npz_path.parent.name
        branch_ids = data["branch_ids"]
        shift = data.get("alignment_shift", None)

        radii_all.append(radii)
        meta_with_case = []
        for m in meta:
            m = dict(m)
            m["case"] = case
            meta_with_case.append(m)
        meta_all.extend(meta_with_case)
        case_ids.extend([case] * radii.shape[0])
        branch_ids_all.extend([f"{case}_{bid}" for bid in branch_ids])
        if shift is not None:
            align_shifts.append(shift)

    if not radii_all:
        print("No radii found after loading npz files.")
        return

    radii_concat = np.concatenate(radii_all, axis=0)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        radii=radii_concat,
        tk=tk_ref,
        angles=angles_ref,
        meta=np.array(meta_all, dtype=object),
        branch_ids=np.array(branch_ids_all),
        cases=np.array(case_ids),
        source_files=np.array([str(p) for p in files]),
        alignment_shifts=np.array(align_shifts, dtype=object) if align_shifts else None,
    )
    print(f"Saved aggregated dataset to {out_path} | branches={radii_concat.shape[0]}, K={radii_concat.shape[1]}, M={radii_concat.shape[2]}")

    if args.pca_dim:
        B, K, M = radii_concat.shape
        norm_radii = normalize_radii(radii_concat, args.normalize)
        lengths = np.array([m.get("length_mm", np.nan) for m in meta_all], dtype=float) if meta_all else None
        if args.length_log and lengths is not None:
            lengths = np.log1p(lengths)
        alpha_used = None
        if args.length_detrend and lengths is not None:
            valid = lengths > 0
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
        out_dir = out_path.parent / f"{out_path.stem}_similarity"
        out_dir.mkdir(parents=True, exist_ok=True)
        dists = np.linalg.norm(latents[:, None, :] - latents[None, :, :], axis=2)
        np.save(out_dir / "pca_latents.npy", latents)
        np.save(out_dir / "pca_components.npy", comps)
        np.save(out_dir / "pca_mean.npy", mean_vec)
        r_mean = float(norm_radii.mean())
        r_std = float(norm_radii.std())
        with open(out_dir / "distances.csv", "w") as f:
            header = ",".join(["branch_id"] + [bid for bid in branch_ids_all])
            f.write(header + "\n")
            for i, bid in enumerate(branch_ids_all):
                row = ",".join([bid] + [f"{dists[i,j]:.4f}" for j in range(B)])
                f.write(row + "\n")
        if args.heatmap:
            plt.figure(figsize=(8, 7))
            ax = plt.gca()
            im = ax.imshow(dists, cmap="viridis")
            plt.colorbar(im, shrink=0.7, label="PCA latent L2")
            plt.xticks(range(B), branch_ids_all, rotation=90, fontsize=6)
            plt.yticks(range(B), branch_ids_all, fontsize=6)
            plt.title("Aggregated branch similarity (PCA latent L2)")
            offset_pct = None
            if meta_all:
                pct_set = {m.get("start_offset_percent") for m in meta_all if "start_offset_percent" in m}
                offset_pct = pct_set.pop() if len(pct_set) == 1 else None
            lines = [
                f"B={B}, K={K}, M={M}, d={d}",
                f"norm={args.normalize}",
                f"r μ={r_mean:.2f}, σ={r_std:.2f}",
            ]
            if offset_pct is not None:
                lines.append(f"offset={offset_pct}%")
            if lengths is not None and lengths.size > 0:
                suffix = " (log1p)" if args.length_log else " mm"
                lines.append(f"L μ={lengths.mean():.1f}{suffix}, σ={lengths.std():.1f}")
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
        print(f"PCA+distances saved to {out_dir} (B={B}, K={K}, M={M}, d={d})")


if __name__ == "__main__":
    main()
