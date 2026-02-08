"""Stage executors for quantitative 5-step coronary pipeline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .io import (
    boundary_points_from_mask,
    load_binary_mask,
    load_feature_summary,
    load_global_descriptor,
    read_mesh_points,
    read_vtp_points,
    save_json,
    save_single_row_csv,
    skeleton_points_from_mask,
)
from .metrics import centerline_metrics, feature_descriptor_metrics, segmentation_metrics


def _stage_output_paths(out_dir: Path, stage_name: str) -> tuple[Path, Path]:
    stage_dir = out_dir / stage_name
    return stage_dir / "metrics.json", stage_dir / "metrics.csv"


def evaluate_step1_segmentation(pred_mask_path: Path, gt_mask_path: Path, out_dir: Path) -> dict[str, Any]:
    pred_mask, pred_affine, pred_spacing = load_binary_mask(pred_mask_path)
    gt_mask, _, gt_spacing = load_binary_mask(gt_mask_path)

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Mask shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")
    if tuple(pred_spacing) != tuple(gt_spacing):
        # Keep deterministic behavior: always evaluate in prediction grid.
        pass

    metrics = segmentation_metrics(pred_mask, gt_mask, spacing=pred_spacing)
    payload = {
        "stage": "step1_segmentation",
        "pred_mask": str(pred_mask_path),
        "gt_mask": str(gt_mask_path),
        "spacing_mm": list(pred_spacing),
        "metrics": metrics,
    }

    json_path, csv_path = _stage_output_paths(out_dir, "step1_segmentation")
    save_json(json_path, payload)
    save_single_row_csv(csv_path, {"stage": payload["stage"], **metrics})
    return payload


def evaluate_step2_centerline_from_mask(
    seg_mask_path: Path,
    gt_centerline_vtp: Path,
    out_dir: Path,
    thr_mm: float = 1.0,
    backend: str = "skeleton",
) -> dict[str, Any]:
    stage_dir = out_dir / "step2_centerline"
    stage_dir.mkdir(parents=True, exist_ok=True)

    backend = backend.lower()
    pred_vtp_path: Optional[Path] = None
    if backend == "skeleton":
        seg_mask, seg_affine, _ = load_binary_mask(seg_mask_path)
        pred_points = skeleton_points_from_mask(seg_mask, seg_affine)
    elif backend == "vmtk":
        pred_vtp_path = stage_dir / "pred_centerline.vtp"
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "vmtk_extract_centerlines.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--mask",
            str(seg_mask_path),
            "--out",
            str(pred_vtp_path),
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "step2 backend=vmtk failed.\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout tail: {completed.stdout[-2000:]}\n"
                f"stderr tail: {completed.stderr[-2000:]}"
            )
        pred_points = read_vtp_points(pred_vtp_path)
    else:
        raise ValueError(f"Unsupported step2 backend: {backend}. Use 'skeleton' or 'vmtk'.")

    gt_points = read_vtp_points(gt_centerline_vtp)

    metrics = centerline_metrics(pred_points, gt_points, thr_mm=thr_mm)
    np.save(stage_dir / "pred_centerline_points.npy", pred_points)

    payload = {
        "stage": "step2_centerline",
        "backend": backend,
        "seg_mask": str(seg_mask_path),
        "gt_centerline": str(gt_centerline_vtp),
        "pred_centerline_vtp": str(pred_vtp_path) if pred_vtp_path is not None else None,
        "metrics": metrics,
    }
    save_json(stage_dir / "metrics.json", payload)
    save_single_row_csv(stage_dir / "metrics.csv", {"stage": payload["stage"], **metrics})
    return payload


def evaluate_step3_repair(
    baseline_centerline_vtp: Optional[Path],
    repaired_centerline_vtp: Path,
    gt_centerline_vtp: Path,
    out_dir: Path,
    thr_mm: float = 1.0,
) -> dict[str, Any]:
    repaired_points = read_vtp_points(repaired_centerline_vtp)
    gt_points = read_vtp_points(gt_centerline_vtp)
    repaired_metrics = centerline_metrics(repaired_points, gt_points, thr_mm=thr_mm)

    payload: dict[str, Any] = {
        "stage": "step3_repair",
        "repaired_centerline": str(repaired_centerline_vtp),
        "gt_centerline": str(gt_centerline_vtp),
        "metrics": repaired_metrics,
    }

    if baseline_centerline_vtp is not None:
        baseline_points = read_vtp_points(baseline_centerline_vtp)
        baseline_metrics = centerline_metrics(baseline_points, gt_points, thr_mm=thr_mm)
        payload["baseline_centerline"] = str(baseline_centerline_vtp)
        payload["baseline_metrics"] = baseline_metrics
        payload["delta"] = {
            "pred2gt_mean_delta": _delta(baseline_metrics.get("pred2gt_mean"), repaired_metrics.get("pred2gt_mean")),
            "pred2gt_p95_delta": _delta(baseline_metrics.get("pred2gt_p95"), repaired_metrics.get("pred2gt_p95")),
            f"coverage_pred@{thr_mm:g}mm_delta": _delta(
                baseline_metrics.get(f"coverage_pred@{thr_mm:g}mm"),
                repaired_metrics.get(f"coverage_pred@{thr_mm:g}mm"),
                higher_better=True,
            ),
        }

    json_path, csv_path = _stage_output_paths(out_dir, "step3_repair")
    save_json(json_path, payload)
    save_single_row_csv(csv_path, {"stage": payload["stage"], **repaired_metrics})
    return payload


def evaluate_step4_features(pred_features_dir: Path, gt_features_dir: Path, out_dir: Path) -> dict[str, Any]:
    pred_summary = load_feature_summary(pred_features_dir)
    gt_summary = load_feature_summary(gt_features_dir)

    pred_desc = load_global_descriptor(pred_features_dir, pred_summary)
    gt_desc = load_global_descriptor(gt_features_dir, gt_summary)

    metrics = {
        "pred_branch_count": int(pred_summary.get("branch_count", 0)),
        "gt_branch_count": int(gt_summary.get("branch_count", 0)),
        "branch_count_abs_diff": int(abs(int(pred_summary.get("branch_count", 0)) - int(gt_summary.get("branch_count", 0)))),
        **feature_descriptor_metrics(pred_desc, gt_desc),
    }

    payload = {
        "stage": "step4_features",
        "pred_features": str(pred_features_dir),
        "gt_features": str(gt_features_dir),
        "metrics": metrics,
    }

    json_path, csv_path = _stage_output_paths(out_dir, "step4_features")
    save_json(json_path, payload)
    save_single_row_csv(csv_path, {"stage": payload["stage"], **metrics})
    return payload


def evaluate_step5_render(
    gt_mask_path: Path,
    out_dir: Path,
    *,
    pred_mesh_path: Optional[Path] = None,
    pred_features_dir: Optional[Path] = None,
    thr_mm: float = 1.0,
) -> dict[str, Any]:
    if pred_mesh_path is None and pred_features_dir is None:
        raise ValueError("Provide either pred_mesh_path or pred_features_dir for step5.")

    if pred_mesh_path is None:
        assert pred_features_dir is not None
        # Lazy import so step1/2/3/4 do not require reconstruction dependencies.
        from vessel_seg.shape import reconstruct_from_features

        with tempfile.TemporaryDirectory(prefix="render_step5_") as tmp_dir:
            tmp_mesh = Path(tmp_dir) / "reconstructed_pred.vtp"
            reconstruct_from_features(pred_features_dir, tmp_mesh)
            pred_points = read_mesh_points(tmp_mesh)
    else:
        pred_points = read_mesh_points(pred_mesh_path)

    gt_mask, gt_affine, _ = load_binary_mask(gt_mask_path)
    gt_boundary_points = boundary_points_from_mask(gt_mask, gt_affine)

    metrics = centerline_metrics(pred_points, gt_boundary_points, thr_mm=thr_mm)
    payload = {
        "stage": "step5_render",
        "pred_mesh": str(pred_mesh_path) if pred_mesh_path else None,
        "pred_features": str(pred_features_dir) if pred_features_dir else None,
        "gt_mask": str(gt_mask_path),
        "metrics": metrics,
    }

    json_path, csv_path = _stage_output_paths(out_dir, "step5_render")
    save_json(json_path, payload)
    save_single_row_csv(csv_path, {"stage": payload["stage"], **metrics})
    return payload


def _delta(before: Any, after: Any, higher_better: bool = False) -> Optional[float]:
    if before is None or after is None:
        return None
    if higher_better:
        return float(after - before)
    return float(before - after)
