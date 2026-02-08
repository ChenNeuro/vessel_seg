#!/usr/bin/env python3
"""Compute PCA reconstruction error on Normal cases with a fixed train/val/test split.

Example:
  python scripts/benchmark_pca_split.py \
    --npz_glob 'outputs/Normal_*/branch_dataset.npz' \
    --out outputs/benchmark/pca_normal_split.json \
    --pca_dim 8 --normalize branch_mean --split 0.8 0.1 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_case_id(name: str) -> Tuple[int, str]:
    if name.startswith("Normal_"):
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix), name
    return 10**9, name


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


def compute_pca(X: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:d]
    return mean, comps


def reconstruction_errors(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xc = X - mean
    coeffs = Xc @ comps.T
    recon = mean + coeffs @ comps
    mse = np.mean((X - recon) ** 2, axis=1)
    rmse = np.sqrt(mse)
    return mse, rmse


def load_case_npz(path: Path, normalize: str, ref_shapes: Tuple[int, int] | None) -> Tuple[np.ndarray, Tuple[int, int]]:
    data = np.load(path, allow_pickle=True)
    radii = data["radii"]
    if radii.ndim != 3:
        raise ValueError(f"Unexpected radii shape in {path}: {radii.shape}")
    k, m = radii.shape[1], radii.shape[2]
    if ref_shapes is not None and (k, m) != ref_shapes:
        raise ValueError(f"K/M mismatch in {path}: {(k, m)} vs {ref_shapes}")
    radii = normalize_radii(radii, normalize)
    X = radii.reshape(radii.shape[0], k * m)
    return X, (k, m)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA reconstruction benchmark with fixed split.")
    parser.add_argument("--npz_glob", type=str, default="outputs/Normal_*/branch_dataset.npz")
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark/pca_normal_split.json"))
    parser.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--pca_dim", type=int, default=8)
    parser.add_argument(
        "--normalize",
        type=str,
        default="branch_mean",
        choices=["none", "branch_mean", "global_mean"],
    )
    parser.add_argument("--save_pca_dir", type=Path, default=None, help="Save mean/components to this directory.")
    args = parser.parse_args()

    files = sorted(Path(".").glob(args.npz_glob))
    if not files:
        raise SystemExit(f"No npz files matched: {args.npz_glob}")

    cases = []
    for p in files:
        case = p.parent.name
        cases.append((parse_case_id(case), case, p))
    cases.sort(key=lambda x: x[0])

    case_names = [c[1] for c in cases]
    n = len(cases)
    split = np.array(args.split, dtype=float)
    if split.sum() <= 0:
        raise SystemExit("Split ratios must be positive.")
    split = split / split.sum()
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise SystemExit(f"Invalid split for n={n}: train={n_train}, val={n_val}, test={n_test}")

    train_cases = cases[:n_train]
    val_cases = cases[n_train : n_train + n_val]
    test_cases = cases[n_train + n_val :]

    ref_shapes = None
    X_train_list = []
    for _, case, path in train_cases:
        X, ref_shapes = load_case_npz(path, args.normalize, ref_shapes)
        X_train_list.append(X)
    X_train = np.concatenate(X_train_list, axis=0)

    d = min(args.pca_dim, X_train.shape[1], X_train.shape[0])
    mean, comps = compute_pca(X_train, d)
    mse_train, rmse_train = reconstruction_errors(X_train, mean, comps)

    def eval_split(split_cases: List[Tuple[Tuple[int, str], str, Path]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for _, case, path in split_cases:
            X, _ = load_case_npz(path, args.normalize, ref_shapes)
            mse, rmse = reconstruction_errors(X, mean, comps)
            out[case] = {
                "num_branches": int(X.shape[0]),
                "mse_mean": float(mse.mean()),
                "mse_std": float(mse.std()),
                "rmse_mean": float(rmse.mean()),
                "rmse_std": float(rmse.std()),
            }
        return out

    val_metrics = eval_split(val_cases)
    test_metrics = eval_split(test_cases)

    payload = {
        "split": {
            "ratios": [float(x) for x in split.tolist()],
            "train_cases": [c[1] for c in train_cases],
            "val_cases": [c[1] for c in val_cases],
            "test_cases": [c[1] for c in test_cases],
        },
        "pca": {
            "dim": int(d),
            "normalize": args.normalize,
            "train_branches": int(X_train.shape[0]),
            "train_mse_mean": float(mse_train.mean()),
            "train_mse_std": float(mse_train.std()),
            "train_rmse_mean": float(rmse_train.mean()),
            "train_rmse_std": float(rmse_train.std()),
        },
        "val": val_metrics,
        "test": test_metrics,
        "case_order": case_names,
        "npz_glob": args.npz_glob,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved PCA benchmark -> {args.out}")

    if args.save_pca_dir is not None:
        args.save_pca_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.save_pca_dir / "pca_mean.npy", mean.squeeze(0))
        np.save(args.save_pca_dir / "pca_components.npy", comps)
        print(f"Saved PCA mean/components -> {args.save_pca_dir}")


if __name__ == "__main__":
    main()
