"""Fourier-Gaussian-process probabilistic shape modelling and inference utilities.

This module adapts the Bayesian fusion strategy from
Wang et al., “An Efficient Muscle Segmentation Method via Bayesian Fusion of Probabilistic
Shape Modeling and Deep Edge Detection” (IEEE TBME, 2024) to the vessel segmentation
toolkit. It provides:

* Fourier-based radial modelling of branch cross-sections.
* Gaussian-process axial modelling to capture longitudinal correlations.
* MAP inference that fuses sparse annotations with edge-confidence maps.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy import linalg, ndimage, optimize

SMALL_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Fourier descriptors
# --------------------------------------------------------------------------- #


def _validate_angles(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 1:
        raise ValueError("angles must be 1-D.")
    if angles.size < 4:
        raise ValueError("angles must contain at least 4 samples for Fourier fitting.")
    return angles


def fourier_coefficients(
    radii: np.ndarray,
    angles: np.ndarray,
    order: int,
) -> np.ndarray:
    """Compute Fourier series coefficients up to the given order.

    The discrete formulation mirrors Eq. (2) in the reference paper.
    """

    radii = np.asarray(radii, dtype=np.float64)
    if radii.shape != angles.shape:
        raise ValueError("radii and angles must share the same shape.")
    if order < 1:
        raise ValueError("order must be >= 1.")

    angles = _validate_angles(angles)
    delta = 2.0 * math.pi / float(angles.size)
    weights = np.full_like(angles, delta)

    coeffs = [float((weights * radii).sum() / math.pi)]
    for n in range(1, order + 1):
        cos_term = np.cos(n * angles)
        sin_term = np.sin(n * angles)
        an = float((weights * radii * cos_term).sum() / math.pi)
        bn = float((weights * radii * sin_term).sum() / math.pi)
        coeffs.extend([an, bn])
    return np.asarray(coeffs, dtype=np.float64)


def radii_from_coefficients(coeffs: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Reconstruct radii samples from Fourier coefficients."""

    coeffs = np.asarray(coeffs, dtype=np.float64)
    angles = _validate_angles(angles)
    order = (coeffs.size - 1) // 2
    result = np.full_like(angles, coeffs[0])
    idx = 1
    for n in range(1, order + 1):
        an = coeffs[idx]
        bn = coeffs[idx + 1]
        result += an * np.cos(n * angles) + bn * np.sin(n * angles)
        idx += 2
    return np.clip(result, 0.0, None)


def parameter_names(order: int) -> List[str]:
    names = ["a0"]
    for n in range(1, order + 1):
        names.append(f"a{n}")
        names.append(f"b{n}")
    return names


# --------------------------------------------------------------------------- #
# Gaussian process helpers
# --------------------------------------------------------------------------- #


def _vandermonde(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.vstack([x ** d for d in range(degree + 1)]).T


@dataclass
class PolynomialMean:
    degree: int
    coefficients: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        design = _vandermonde(x, self.degree)
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.coefficients = coeffs

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("PolynomialMean not fitted.")
        design = _vandermonde(x, self.degree)
        return design @ self.coefficients


@dataclass
class RBFKernel:
    variance: float
    length_scale: float

    def __post_init__(self) -> None:
        self.variance = max(float(self.variance), SMALL_EPS)
        self.length_scale = max(float(self.length_scale), SMALL_EPS)

    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        diff = x[:, None] - y[None, :]
        sq = diff * diff
        return (self.variance ** 2) * np.exp(-0.5 * sq / (self.length_scale ** 2))


@dataclass
class GaussianProcessAxis:
    degree: int = 2
    noise: float = 1e-2
    mean: PolynomialMean = field(init=False)
    kernel: RBFKernel = field(init=False)

    def fit(self, heights: np.ndarray, values: np.ndarray) -> None:
        heights = np.asarray(heights, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if heights.ndim != 1:
            raise ValueError("heights must be 1-D.")
        if heights.size != values.size:
            raise ValueError("heights and values must have the same length.")
        if heights.size < self.degree + 1:
            raise ValueError("Insufficient samples to fit the polynomial mean.")

        order = np.argsort(heights)
        heights = heights[order]
        values = values[order]

        self.mean = PolynomialMean(self.degree)
        self.mean.fit(heights, values)
        residual = values - self.mean.evaluate(heights)

        init_var = max(float(np.std(residual)), 1e-2)
        init_length = 0.2
        init_noise = max(float(np.std(residual) * 0.1), 1e-3)
        theta0 = np.log([init_var, init_length, init_noise])

        def objective(theta: np.ndarray) -> float:
            var = math.exp(theta[0])
            length = math.exp(theta[1])
            noise = math.exp(theta[2])
            kernel = RBFKernel(var, length)
            k_mat = kernel.matrix(heights, heights)
            np.fill_diagonal(k_mat, k_mat.diagonal() + noise ** 2)
            try:
                l_mat = np.linalg.cholesky(k_mat)
            except np.linalg.LinAlgError:
                return np.inf
            alpha = linalg.cho_solve((l_mat, True), residual)
            log_det = 2.0 * np.sum(np.log(np.diag(l_mat)))
            return 0.5 * residual.dot(alpha) + 0.5 * log_det + 0.5 * heights.size * math.log(2.0 * math.pi)

        result = optimize.minimize(objective, theta0, method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"GP optimisation failed: {result.message}")

        var = math.exp(result.x[0])
        length = math.exp(result.x[1])
        noise = math.exp(result.x[2])
        self.kernel = RBFKernel(var, length)
        self.noise = noise

    def prior(self, heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        heights = np.asarray(heights, dtype=np.float64)
        mean = self.mean.evaluate(heights)
        cov = self.kernel.matrix(heights, heights)
        np.fill_diagonal(cov, cov.diagonal() + self.noise ** 2)
        return mean, cov

    def posterior(
        self,
        query_heights: np.ndarray,
        obs_heights: Optional[np.ndarray] = None,
        obs_values: Optional[np.ndarray] = None,
        obs_noise: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        query_heights = np.asarray(query_heights, dtype=np.float64)
        if obs_heights is None or obs_values is None or obs_heights.size == 0:
            return self.prior(query_heights)

        obs_heights = np.asarray(obs_heights, dtype=np.float64)
        obs_values = np.asarray(obs_values, dtype=np.float64)
        if obs_heights.size != obs_values.size:
            raise ValueError("obs_heights and obs_values must match in length.")

        mu_query = self.mean.evaluate(query_heights)
        mu_obs = self.mean.evaluate(obs_heights)

        k_qq = self.kernel.matrix(query_heights, query_heights)
        k_qo = self.kernel.matrix(query_heights, obs_heights)
        k_oo = self.kernel.matrix(obs_heights, obs_heights)

        cov_obs = k_oo.copy()
        np.fill_diagonal(cov_obs, cov_obs.diagonal() + (self.noise ** 2 + obs_noise ** 2))

        try:
            cho = linalg.cho_factor(cov_obs, lower=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Covariance matrix ill-conditioned during posterior update.") from exc

        delta = obs_values - mu_obs
        gain = linalg.cho_solve(cho, delta, check_finite=False)
        posterior_mean = mu_query + k_qo @ gain
        cross = linalg.cho_solve(cho, k_qo.T, check_finite=False)
        posterior_cov = k_qq - k_qo @ cross
        np.fill_diagonal(posterior_cov, np.clip(np.diag(posterior_cov), SMALL_EPS, None))
        return posterior_mean, posterior_cov


# --------------------------------------------------------------------------- #
# FGPM branch model
# --------------------------------------------------------------------------- #


@dataclass
class FGPMBranchModel:
    order: int
    degree: int = 2
    param_models: List[GaussianProcessAxis] = field(default_factory=list)
    height_min: float = 0.0
    height_max: float = 1.0

    def fit(self, heights: np.ndarray, coeff_matrix: np.ndarray) -> None:
        heights = np.asarray(heights, dtype=np.float64)
        coeff_matrix = np.asarray(coeff_matrix, dtype=np.float64)
        if coeff_matrix.ndim != 2:
            raise ValueError("coeff_matrix must be 2-D.")
        if coeff_matrix.shape[0] != heights.size:
            raise ValueError("Number of rows in coeff_matrix must equal heights length.")
        param_count = coeff_matrix.shape[1]
        expected = 2 * self.order + 1
        if param_count != expected:
            raise ValueError(f"Expected {expected} parameters per slice, got {param_count}.")

        self.param_models = []
        for idx in range(param_count):
            gp = GaussianProcessAxis(self.degree)
            gp.fit(heights, coeff_matrix[:, idx])
            self.param_models.append(gp)
        self.height_min = float(heights.min(initial=0.0))
        self.height_max = float(heights.max(initial=1.0))

    def posterior(
        self,
        heights: np.ndarray,
        obs_heights: Optional[np.ndarray],
        obs_coeffs: Optional[np.ndarray],
        obs_noise: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        heights = np.asarray(heights, dtype=np.float64)
        param_count = len(self.param_models)
        mean = np.zeros((heights.size, param_count), dtype=np.float64)
        cov = np.zeros((param_count, heights.size, heights.size), dtype=np.float64)

        if obs_heights is None or obs_coeffs is None:
            obs_heights = None
            obs_coeffs = None

        for idx, gp in enumerate(self.param_models):
            obs_vals = obs_coeffs[:, idx] if obs_coeffs is not None else None
            mean[:, idx], cov[idx] = gp.posterior(heights, obs_heights, obs_vals, obs_noise=obs_noise)

        return mean, cov

    def sample(self, heights: np.ndarray, random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        rng = random_state or np.random.default_rng()
        heights = np.asarray(heights, dtype=np.float64)
        samples = np.zeros((heights.size, len(self.param_models)), dtype=np.float64)
        for idx, gp in enumerate(self.param_models):
            mu, cov = gp.prior(heights)
            samples[:, idx] = rng.multivariate_normal(mu, cov + np.eye(cov.shape[0]) * SMALL_EPS)
        return samples

    def to_dict(self) -> Dict[str, object]:
        packed = []
        for name, gp in zip(parameter_names(self.order), self.param_models):
            packed.append(
                {
                    "name": name,
                    "degree": gp.degree,
                    "poly": gp.mean.coefficients.tolist(),
                    "variance": gp.kernel.variance,
                    "length_scale": gp.kernel.length_scale,
                    "noise": gp.noise,
                }
            )
        return {
            "order": self.order,
            "degree": self.degree,
            "height_min": self.height_min,
            "height_max": self.height_max,
            "parameters": packed,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FGPMBranchModel":
        order = int(payload["order"])
        degree = int(payload.get("degree", 2))
        model = cls(order=order, degree=degree)
        model.param_models = []
        for entry in payload["parameters"]:
            gp = GaussianProcessAxis(degree=entry["degree"])
            gp.mean = PolynomialMean(entry["degree"])
            gp.mean.coefficients = np.asarray(entry["poly"], dtype=np.float64)
            gp.kernel = RBFKernel(entry["variance"], entry["length_scale"])
            gp.noise = float(entry["noise"])
            model.param_models.append(gp)
        model.height_min = float(payload.get("height_min", 0.0))
        model.height_max = float(payload.get("height_max", 1.0))
        return model


# --------------------------------------------------------------------------- #
# Training data utilities
# --------------------------------------------------------------------------- #


def _arclength(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=np.float64)
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def _load_branch_samples(
    features_dir: Path,
    branch_name: str,
    order: int,
    min_confidence: float,
) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
    summary = json.loads((features_dir / "summary.json").read_text(encoding="utf-8"))
    branch_entry = next((b for b in summary["branches"] if b["name"] == branch_name), None)
    if branch_entry is None:
        raise ValueError(f"Branch {branch_name} not available in {features_dir}.")
    branch_path = features_dir / branch_entry["feature_file"]
    data = np.load(branch_path)
    raw = data["raw_profiles"]
    confidences = data["slice_confidence"]
    angles = data["angles"]
    samples_world = data["samples_world"]
    s = _arclength(samples_world)
    if s.size == 0 or s[-1] < SMALL_EPS:
        raise ValueError(f"Branch {branch_name} has zero length in {features_dir}.")
    heights = s / s[-1]

    usable_heights: List[float] = []
    coeffs: List[np.ndarray] = []
    for idx, (profile, conf) in enumerate(zip(raw, confidences)):
        if conf < min_confidence:
            continue
        if not (profile > 0).any():
            continue
        usable_heights.append(float(heights[idx]))
        coeffs.append(fourier_coefficients(profile, angles, order))
    return usable_heights, coeffs, angles


def gather_training_set(
    feature_dirs: Sequence[str],
    branch_name: str,
    order: int,
    min_confidence: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    heights: List[float] = []
    coeffs: List[np.ndarray] = []
    angles: Optional[np.ndarray] = None
    for directory in feature_dirs:
        dir_path = Path(directory)
        h_list, c_list, local_angles = _load_branch_samples(dir_path, branch_name, order, min_confidence)
        heights.extend(h_list)
        coeffs.extend(c_list)
        if angles is None:
            angles = local_angles
        elif not np.allclose(angles, local_angles):
            raise ValueError("All feature directories must share identical angle sampling.")
    if not heights:
        raise RuntimeError("No valid slices collected for FGPM training.")
    return np.asarray(heights, dtype=np.float64), np.vstack(coeffs), angles


# --------------------------------------------------------------------------- #
# Edge confidence evaluation
# --------------------------------------------------------------------------- #


@dataclass
class SliceContext:
    center: np.ndarray
    normal: np.ndarray
    binormal: np.ndarray


class EdgeConfidenceEvaluator:
    def __init__(self, edge_path: str) -> None:
        image = nib.load(edge_path)
        self.volume = image.get_fdata().astype(np.float32)
        self.inv_affine = np.linalg.inv(image.affine)

    def evaluate(self, context: SliceContext, radii: np.ndarray, angles: np.ndarray) -> float:
        normals = np.outer(np.cos(angles), context.normal)
        binormals = np.outer(np.sin(angles), context.binormal)
        positions = context.center + normals * radii[:, None] + binormals * radii[:, None]
        vox = apply_affine(self.inv_affine, positions)
        samples = ndimage.map_coordinates(
            self.volume,
            [vox[:, 0], vox[:, 1], vox[:, 2]],
            order=1,
            mode="nearest",
            prefilter=False,
        )
        return float(np.clip(samples.mean(), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Annotation utilities
# --------------------------------------------------------------------------- #


def _load_annotations(
    annotation_path: Path,
    order: int,
    min_confidence: float,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(annotation_path)
    raw = data["raw_profiles"]
    angles = data["angles"]
    confidences = data["slice_confidence"]
    samples_world = data["samples_world"]
    lengths = _arclength(samples_world)
    if lengths.size == 0 or lengths[-1] < SMALL_EPS:
        raise ValueError("Annotation file missing valid centreline samples.")
    heights = lengths / lengths[-1]
    valid_mask = confidences >= min_confidence
    heights = heights[valid_mask]
    coeffs = [fourier_coefficients(profile, angles, order) for profile in raw[valid_mask]]
    if not len(coeffs):
        raise RuntimeError("No annotations passed the confidence threshold.")
    return heights, np.vstack(coeffs)


# --------------------------------------------------------------------------- #
# Annotation propagation
# --------------------------------------------------------------------------- #


def _extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def _insert_slice(volume: np.ndarray, axis: int, index: int, data: np.ndarray) -> None:
    slicer = [slice(None)] * volume.ndim
    slicer[axis] = index
    volume[tuple(slicer)] = data


def propagate_annotation(
    image_path: str,
    mask_path: str,
    source_index: int,
    target_index: int,
    axis: int,
    expand: int,
    output_path: str,
) -> None:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("SimpleITK is required for annotation propagation. Install with `pip install SimpleITK`.") from exc

    image = nib.load(image_path)
    mask_img = nib.load(mask_path)
    vol = image.get_fdata().astype(np.float32)
    mask = mask_img.get_fdata().astype(np.float32)

    source_slice = _extract_slice(vol, axis, source_index)
    target_slice = _extract_slice(vol, axis, target_index)
    source_mask = _extract_slice(mask, axis, source_index)

    coords = np.argwhere(source_mask > 0.5)
    if coords.size == 0:
        raise ValueError("Source slice does not contain annotations.")
    mins = np.maximum(coords.min(axis=0) - expand, 0)
    maxs = np.minimum(coords.max(axis=0) + expand + 1, source_mask.shape)

    slicer = tuple(slice(start, stop) for start, stop in zip(mins, maxs))
    moving_img = sitk.GetImageFromArray(source_slice[slicer])
    fixed_img = sitk.GetImageFromArray(target_slice[slicer])
    moving_mask = sitk.GetImageFromArray(source_mask[slicer])

    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(100)
    demons.SetStandardDeviations(1.0)
    displacement = demons.Execute(fixed_img, moving_img)
    transform = sitk.DisplacementFieldTransform(displacement)
    resampled = sitk.Resample(moving_mask, fixed_img, transform, sitk.sitkNearestNeighbor, 0.0, moving_mask.GetPixelID())
    warped = sitk.GetArrayFromImage(resampled)

    propagated_mask = mask.copy()
    target_mask_slice = np.zeros_like(source_mask)
    target_mask_slice[slicer] = warped
    _insert_slice(propagated_mask, axis, target_index, target_mask_slice)

    out_img = nib.Nifti1Image(propagated_mask, mask_img.affine, mask_img.header)
    nib.save(out_img, output_path)
    print(f"Saved propagated annotation to {output_path}")


# --------------------------------------------------------------------------- #
# MAP optimisation with edge fusion
# --------------------------------------------------------------------------- #


def map_estimate_per_slice(
    mean_coeff: np.ndarray,
    cov_diag: np.ndarray,
    context: SliceContext,
    evaluator: Optional[EdgeConfidenceEvaluator],
    angles: np.ndarray,
    edge_weight: float,
) -> np.ndarray:
    if evaluator is None or edge_weight <= 0:
        return mean_coeff

    def objective(coeffs_flat: np.ndarray) -> float:
        coeffs = coeffs_flat
        prior = 0.5 * np.sum((coeffs - mean_coeff) ** 2 / np.clip(cov_diag, SMALL_EPS, None))
        radii = radii_from_coefficients(coeffs, angles)
        confidence = evaluator.evaluate(context, radii, angles)
        return prior - edge_weight * math.log(max(confidence, 1e-6))

    result = optimize.minimize(
        objective,
        mean_coeff,
        method="L-BFGS-B",
    )
    if not result.success:
        return mean_coeff
    return result.x


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #


def _cmd_fit(args: argparse.Namespace) -> None:
    heights, coeffs, angles = gather_training_set(
        args.features_dirs,
        args.branch,
        args.order,
        args.min_confidence,
    )
    model = FGPMBranchModel(order=args.order, degree=args.degree)
    model.fit(heights, coeffs)
    payload = {
        "model": model.to_dict(),
        "branch": args.branch,
        "angles": angles.tolist(),
        "meta": {
            "feature_dirs": list(args.features_dirs),
            "min_confidence": args.min_confidence,
            "samples": coeffs.shape[0],
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved FGPM model for {args.branch} to {args.output}")


def _load_model(model_path: Path) -> Tuple[FGPMBranchModel, np.ndarray, Dict[str, object]]:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    model = FGPMBranchModel.from_dict(payload["model"])
    angles = np.asarray(payload["angles"], dtype=np.float64)
    return model, angles, payload.get("meta", {})


def _cmd_sample(args: argparse.Namespace) -> None:
    model, angles, _ = _load_model(Path(args.model))
    heights = np.linspace(0.0, 1.0, args.num_slices, dtype=np.float64)
    samples = model.sample(heights)
    radii = np.stack([radii_from_coefficients(row, angles) for row in samples], axis=0)
    np.savez(
        args.output,
        heights=heights,
        coeffs=samples,
        radii=radii,
        angles=angles,
    )
    print(f"Sampled {args.num_slices} slices from FGPM and saved to {args.output}")


def _cmd_infer(args: argparse.Namespace) -> None:
    model, angles, _ = _load_model(Path(args.model))

    annotations_h = None
    annotations_coeff = None
    if args.annotations:
        annotations_h, annotations_coeff = _load_annotations(Path(args.annotations), model.order, args.annotation_min_conf)

    branch_data = np.load(args.branch_features)
    samples_world = branch_data["samples_world"]
    normals = branch_data["normals"]
    binormals = branch_data["binormals"]
    s = _arclength(samples_world)
    if s.size == 0 or s[-1] < SMALL_EPS:
        raise ValueError("Branch features do not contain a valid centreline.")
    heights = s / s[-1]

    mean, cov = model.posterior(heights, annotations_h, annotations_coeff, obs_noise=args.annotation_noise)

    evaluator = EdgeConfidenceEvaluator(args.edge_map) if args.edge_map else None

    refined = np.zeros_like(mean)
    for idx in range(mean.shape[0]):
        context = SliceContext(center=samples_world[idx], normal=normals[idx], binormal=binormals[idx])
        cov_diag = np.array([cov[param_idx, idx, idx] for param_idx in range(cov.shape[0])], dtype=np.float64)
        refined[idx] = map_estimate_per_slice(
            mean[idx],
            cov_diag,
            context,
            evaluator,
            angles,
            edge_weight=args.edge_weight,
        )

    radii = np.stack([radii_from_coefficients(row, angles) for row in refined], axis=0)
    np.savez(
        args.output,
        heights=heights,
        coeffs=refined,
        radii=radii,
        angles=angles,
        mean=mean,
    )
    print(f"Saved inferred coefficients to {args.output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fourier-Gaussian-process modelling toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Train FGPM on branch profiles.")
    fit_parser.add_argument("--features-dirs", nargs="+", required=True, help="Feature directories produced by vessel_seg.shape.")
    fit_parser.add_argument("--branch", required=True, help="Branch name to model (exact match in summary.json).")
    fit_parser.add_argument("--order", type=int, default=6, help="Fourier order to retain.")
    fit_parser.add_argument("--degree", type=int, default=2, help="Polynomial degree for the mean function.")
    fit_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.2,
        help="Minimum slice confidence to keep during training.",
    )
    fit_parser.add_argument("--output", required=True, help="Path to save trained model (JSON).")
    fit_parser.set_defaults(func=_cmd_fit)

    sample_parser = subparsers.add_parser("sample", help="Generate random shapes from a trained FGPM.")
    sample_parser.add_argument("--model", required=True, help="Path to trained FGPM JSON.")
    sample_parser.add_argument("--num-slices", type=int, default=64, help="Number of slices to sample.")
    sample_parser.add_argument("--output", required=True, help="Output `.npz` path for sampled coefficients.")
    sample_parser.set_defaults(func=_cmd_sample)

    infer_parser = subparsers.add_parser("infer", help="Run Bayesian inference for a case.")
    infer_parser.add_argument("--model", required=True, help="Path to trained FGPM JSON.")
    infer_parser.add_argument(
        "--branch-features",
        required=True,
        help="NPZ file (from vessel_seg.shape extract) containing centreline frames for the target branch.",
    )
    infer_parser.add_argument(
        "--annotations",
        help="NPZ file with sparse manual annotations (output of vessel_seg.shape for partial masks).",
    )
    infer_parser.add_argument(
        "--annotation-min-conf",
        type=float,
        default=0.5,
        help="Confidence threshold for accepting annotation slices.",
    )
    infer_parser.add_argument(
        "--annotation-noise",
        type=float,
        default=0.5,
        help="Observation noise (mm) applied to annotation-derived coefficients.",
    )
    infer_parser.add_argument("--edge-map", help="Optional edge confidence volume (NIfTI).")
    infer_parser.add_argument(
        "--edge-weight",
        type=float,
        default=1.0,
        help="Weight applied to the edge likelihood during MAP.",
    )
    infer_parser.add_argument("--output", required=True, help="Output `.npz` for inferred coefficients.")
    infer_parser.set_defaults(func=_cmd_infer)

    prop_parser = subparsers.add_parser("propagate", help="Propagate annotations between adjacent slices.")
    prop_parser.add_argument("--image", required=True, help="Intensity volume (NIfTI).")
    prop_parser.add_argument("--mask", required=True, help="Binary mask with annotated slice.")
    prop_parser.add_argument("--source", type=int, required=True, help="Annotated slice index.")
    prop_parser.add_argument("--target", type=int, required=True, help="Slice index to propagate to.")
    prop_parser.add_argument("--axis", type=int, default=2, choices=(0, 1, 2), help="Axis representing slice dimension.")
    prop_parser.add_argument("--expand", type=int, default=8, help="ROI padding (voxels).")
    prop_parser.add_argument("--output-mask", required=True, help="Output NIfTI for propagated mask.")
    prop_parser.set_defaults(
        func=lambda args: propagate_annotation(
            args.image,
            args.mask,
            args.source,
            args.target,
            args.axis,
            args.expand,
            args.output_mask,
        )
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
