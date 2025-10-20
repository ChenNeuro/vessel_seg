"""End-to-end coronary dimensionality reduction demo on ASOCA segmentations.

Steps
-----
1. Extract centreline branches from a binary segmentation.
2. Sample cross-sectional polar profiles along each branch.
3. Fit a PCA model on confident cross-sections to obtain a low-dimensional embedding.
4. Reconstruct full cross-sections from the PCA coefficients and export artefacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.shape import (
    BranchProfile,
    compute_polar_profiles,
    export_branch_features,
    extract_branches,
    reconstruct_from_features,
)


def _fit_pca(
    branch_profiles: Iterable[BranchProfile],
    *,
    num_components: int,
    confidence_threshold: float,
) -> Dict[str, np.ndarray]:
    """Fit a PCA basis over high-confidence polar profiles."""

    slices: List[np.ndarray] = []
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
            "No cross-sections met the confidence threshold. "
            "Consider lowering `--confidence-threshold`."
        )

    data = np.vstack(slices).astype(np.float32)
    mean = data.mean(axis=0)
    centred = data - mean
    u, s, vh = np.linalg.svd(centred, full_matrices=False)
    components = vh[: num_components].astype(np.float32)

    denom = max(data.shape[0] - 1, 1)
    explained_variance = (s ** 2) / denom
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


def _reconstruct_profiles(
    branch_profiles: Iterable[BranchProfile],
    model: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Project cross-sections into PCA space and reconstruct the contours."""

    mean = model["mean"]
    components = model["components"]
    threshold = float(model["confidence_threshold"])
    num_components = components.shape[0]

    reconstructions: Dict[str, Dict[str, np.ndarray]] = {}
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
        mse = np.mean((recon_raw - profile.raw_profiles) ** 2)
        mae = np.mean(np.abs(recon_raw - profile.raw_profiles))

        reconstructions[profile.branch.name] = {
            "normalized": recon_norm,
            "raw": recon_raw,
            "coefficients": coeffs,
            "mse": np.float32(mse),
            "mae": np.float32(mae),
        }

    return reconstructions


def _save_ml_features(
    branch_profiles: Iterable[BranchProfile],
    reconstructions: Mapping[str, Mapping[str, np.ndarray]],
    model: Mapping[str, np.ndarray],
    output_dir: Path,
) -> Dict[str, object]:
    """Persist ML-driven reconstructions in the same layout as `export_branch_features`."""

    output_dir.mkdir(parents=True, exist_ok=True)
    branch_summary: List[Dict[str, object]] = []
    feature_matrix: List[np.ndarray] = []

    for profile in branch_profiles:
        branch_name = profile.branch.name
        recon = reconstructions[branch_name]
        branch_path = output_dir / f"{branch_name}.npz"
        np.savez(
            branch_path,
            branch_name=branch_name,
            samples_world=profile.samples_world,
            samples_voxel=profile.samples_voxel,
            tangents=profile.tangents,
            normals=profile.normals,
            binormals=profile.binormals,
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
        feature_matrix.append(profile.feature_vector)

    feature_matrix = np.vstack(feature_matrix) if feature_matrix else np.zeros((0, 1), dtype=np.float32)
    global_descriptor = feature_matrix.mean(axis=0) if feature_matrix.size else np.zeros(1, dtype=np.float32)
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
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _run_pipeline(args: argparse.Namespace) -> None:
    seg_path = Path(args.seg).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    features_dir = output_dir / "features_original"
    ml_dir = output_dir / "features_ml"

    branches = extract_branches(
        seg_path,
        min_branch_length_mm=args.min_length,
        closing_iterations=args.closing_iterations,
        smooth_sigma_mm=args.smooth_sigma_mm,
        adaptive_min_step_mm=args.adaptive_min_step,
        adaptive_max_step_mm=args.adaptive_max_step,
        curvature_alpha=args.adaptive_curvature_alpha,
    )
    profiles = compute_polar_profiles(
        seg_path,
        branches,
        num_samples=args.num_samples,
        num_angle_bins=args.num_angle_bins,
        patch_radius_mm=args.patch_radius,
        half_thickness_mm=args.half_thickness,
        radius_clip_factor=args.radius_clip_factor,
        endpoint_trim_mm=args.endpoint_trim_mm,
    )
    summary_original = export_branch_features(profiles, features_dir)

    model = _fit_pca(
        profiles,
        num_components=args.components,
        confidence_threshold=args.confidence_threshold,
    )
    recon = _reconstruct_profiles(profiles, model)
    summary_ml = _save_ml_features(profiles, recon, model, ml_dir)

    result = {
        "original_features": {
            "dir": str(features_dir),
            "branch_count": summary_original.get("branch_count", 0),
        },
        "ml_reconstruction": {
            "dir": str(ml_dir),
            "branch_count": summary_ml.get("branch_count", 0),
            "explained_variance_ratio": summary_ml["model"]["explained_variance_ratio"],
        },
    }

    if args.mesh:
        mesh_path_original = output_dir / "mesh_original.vtp"
        mesh_path_ml = output_dir / "mesh_ml.vtp"
        reconstruct_from_features(
            features_dir,
            mesh_path_original,
            target_samples=args.mesh_target_samples,
            smoothing_factor=args.mesh_smoothing,
            min_valid_slices=args.mesh_min_valid,
            interpolation_kind=args.mesh_interp_kind,
            angular_upsample=args.mesh_angular_upsample,
            angular_smoothing=args.mesh_angular_smoothing,
            min_radius_ratio=args.mesh_min_radius_ratio,
            angular_gap_fill_bins=args.mesh_angular_gap_fill,
            axial_gap_fill=args.mesh_axial_gap_fill,
        )
        reconstruct_from_features(
            ml_dir,
            mesh_path_ml,
            target_samples=args.mesh_target_samples,
            smoothing_factor=args.mesh_smoothing,
            min_valid_slices=args.mesh_min_valid,
            interpolation_kind=args.mesh_interp_kind,
            angular_upsample=args.mesh_angular_upsample,
            angular_smoothing=args.mesh_angular_smoothing,
            min_radius_ratio=args.mesh_min_radius_ratio,
            angular_gap_fill_bins=args.mesh_angular_gap_fill,
            axial_gap_fill=args.mesh_axial_gap_fill,
        )
        result["meshes"] = {
            "original": str(mesh_path_original),
            "ml_reconstruction": str(mesh_path_ml),
        }

    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the coronary dimensionality reduction demo on ASOCA data."
    )
    parser.add_argument("--seg", required=True, help="Path to ASOCA binary coronary mask (.nii.gz).")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store outputs (features, PCA model, optional meshes).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=120,
        help="Number of resampled points along each branch.",
    )
    parser.add_argument(
        "--num-angle-bins",
        type=int,
        default=72,
        help="Number of angular bins for polar sampling.",
    )
    parser.add_argument(
        "--patch-radius",
        type=float,
        default=3.5,
        help="Radius (mm) of cross-sectional sampling window.",
    )
    parser.add_argument(
        "--half-thickness",
        type=float,
        default=1.2,
        help="Half thickness (mm) of sampling slab along the tangent direction.",
    )
    parser.add_argument(
        "--radius-clip-factor",
        type=float,
        default=1.8,
        help="Max radius multiplier relative to slice median (0 disables clipping).",
    )
    parser.add_argument(
        "--endpoint-trim-mm",
        type=float,
        default=1.5,
        help="Trim distance (mm) from each branch endpoint to suppress bifurcation artefacts.",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=10.0,
        help="Minimum branch length to keep (mm).",
    )
    parser.add_argument(
        "--closing-iterations",
        type=int,
        default=2,
        help="Binary closing iterations for mask cleanup.",
    )
    parser.add_argument(
        "--smooth-sigma-mm",
        type=float,
        default=0.8,
        help="Gaussian smoothing sigma (mm) applied along extracted centrelines.",
    )
    parser.add_argument(
        "--adaptive-min-step",
        type=float,
        default=0.6,
        help="Minimum sampling step (mm) for curvature-adaptive centreline resampling.",
    )
    parser.add_argument(
        "--adaptive-max-step",
        type=float,
        default=2.5,
        help="Maximum sampling step (mm) for curvature-adaptive centreline resampling.",
    )
    parser.add_argument(
        "--adaptive-curvature-alpha",
        type=float,
        default=2.0,
        help="Curvature weight controlling adaptive resampling density.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=8,
        help="Number of PCA components for cross-section reconstruction.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Minimum slice confidence to include in PCA fitting.",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="If set, export PyVista meshes for original and ML-reconstructed profiles.",
    )
    parser.add_argument(
        "--mesh-target-samples",
        type=int,
        default=None,
        help="Optional resampling density for centreline sweep when exporting meshes.",
    )
    parser.add_argument(
        "--mesh-smoothing",
        type=float,
        default=0.0,
        help="Spline smoothing factor applied to polar profiles before sweep.",
    )
    parser.add_argument(
        "--mesh-min-valid",
        type=int,
        default=3,
        help="Minimum number of valid slices required to apply smoothing.",
    )
    parser.add_argument(
        "--mesh-interp-kind",
        choices=("linear", "quadratic", "cubic"),
        default="cubic",
        help="Fallback interpolation scheme when smoothing is disabled.",
    )
    parser.add_argument(
        "--mesh-angular-upsample",
        type=int,
        default=3,
        help="Angular upsampling factor per slice before sweep.",
    )
    parser.add_argument(
        "--mesh-angular-smoothing",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma (in angular bins) applied after upsampling.",
    )
    parser.add_argument(
        "--mesh-min-radius-ratio",
        type=float,
        default=0.08,
        help="Minimum radius per slice as fraction of its mean radius to avoid spikes.",
    )
    parser.add_argument(
        "--mesh-angular-gap-fill",
        type=int,
        default=2,
        help="Maximum angular gap (in bins) to fill within a slice during sweep.",
    )
    parser.add_argument(
        "--mesh-axial-gap-fill",
        type=int,
        default=1,
        help="Maximum number of consecutive empty slices to interpolate along the branch.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover
    main()
