"""Quantitative metrics used across segmentation/centerline/feature/render stages."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def _safe_stat(values: np.ndarray, fn) -> Optional[float]:
    if values.size == 0:
        return None
    return float(fn(values))


def summarize_distances(dist: np.ndarray) -> dict[str, Optional[float]]:
    if dist.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
    }


def _skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    """对二值体做骨架化，优先使用 3D skeletonize，缺失时退化到逐层 2D。"""
    volume = np.asarray(mask, dtype=bool)
    try:
        from skimage.morphology import skeletonize_3d  # type: ignore

        return skeletonize_3d(volume).astype(bool)
    except Exception:
        try:
            from skimage.morphology._skeletonize_3d import skeletonize_3d  # type: ignore

            return skeletonize_3d(volume).astype(bool)
        except Exception:
            from skimage.morphology import skeletonize  # type: ignore

            slices = [skeletonize(volume[:, :, i]) for i in range(volume.shape[2])]
            return np.stack(slices, axis=2).astype(bool)


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    _, num = ndimage.label(mask.astype(bool), structure=structure)
    return int(num)


def _largest_component_ratio(mask: np.ndarray, connectivity: int) -> Optional[float]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labels, num = ndimage.label(mask.astype(bool), structure=structure)
    if num == 0:
        return None
    counts = np.bincount(labels.ravel())
    foreground = counts[1:]
    if foreground.size == 0:
        return None
    return float(foreground.max() / foreground.sum())


def segmentation_consistency_metrics(mask: np.ndarray) -> dict[str, Any]:
    """返回不依赖 GT 的分割一致性统计。"""
    pred = np.asarray(mask, dtype=bool)
    skeleton = _skeletonize_binary(pred)
    return {
        "mask_components_6": _component_count(pred, connectivity=1),
        "mask_components_26": _component_count(pred, connectivity=3),
        "mask_largest_component_ratio_26": _largest_component_ratio(pred, connectivity=3),
        "skeleton_voxels": int(skeleton.sum(dtype=np.int64)),
        "skeleton_components_6": _component_count(skeleton, connectivity=1),
        "skeleton_components_26": _component_count(skeleton, connectivity=3),
    }


def cldice_metric(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, Optional[float]]:
    """计算管状结构常用的 clDice 及其中间项。"""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    if not pred.any() and not gt.any():
        return {"cldice": 1.0, "topology_precision": 1.0, "topology_sensitivity": 1.0}

    pred_skeleton = _skeletonize_binary(pred)
    gt_skeleton = _skeletonize_binary(gt)

    pred_skeleton_count = int(pred_skeleton.sum(dtype=np.int64))
    gt_skeleton_count = int(gt_skeleton.sum(dtype=np.int64))

    topology_precision = None
    if pred_skeleton_count > 0:
        topology_precision = float(np.logical_and(pred_skeleton, gt).sum(dtype=np.int64) / pred_skeleton_count)

    topology_sensitivity = None
    if gt_skeleton_count > 0:
        topology_sensitivity = float(np.logical_and(gt_skeleton, pred).sum(dtype=np.int64) / gt_skeleton_count)

    if topology_precision is None or topology_sensitivity is None:
        cldice = None
    elif topology_precision + topology_sensitivity == 0.0:
        cldice = 0.0
    else:
        cldice = float(
            2.0 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity)
        )

    return {
        "cldice": cldice,
        "topology_precision": topology_precision,
        "topology_sensitivity": topology_sensitivity,
    }


def symmetric_point_distances(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.size == 0 or b.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    dist_a, _ = tree_b.query(a, k=1)
    dist_b, _ = tree_a.query(b, k=1)
    return dist_a.astype(float), dist_b.astype(float)


def centerline_metrics(pred_points: np.ndarray, gt_points: np.ndarray, thr_mm: float = 1.0) -> dict[str, Any]:
    dist_pred, dist_gt = symmetric_point_distances(pred_points, gt_points)

    stats_pred = summarize_distances(dist_pred)
    stats_gt = summarize_distances(dist_gt)

    coverage_pred = float((dist_pred <= thr_mm).mean()) if dist_pred.size else None
    coverage_gt = float((dist_gt <= thr_mm).mean()) if dist_gt.size else None

    return {
        "pred_points": int(pred_points.shape[0]),
        "gt_points": int(gt_points.shape[0]),
        "pred2gt_mean": stats_pred["mean"],
        "pred2gt_median": stats_pred["median"],
        "pred2gt_p95": stats_pred["p95"],
        "pred2gt_max": stats_pred["max"],
        "gt2pred_mean": stats_gt["mean"],
        "gt2pred_median": stats_gt["median"],
        "gt2pred_p95": stats_gt["p95"],
        "gt2pred_max": stats_gt["max"],
        f"coverage_pred@{thr_mm:g}mm": coverage_pred,
        f"coverage_gt@{thr_mm:g}mm": coverage_gt,
        "thr_mm": float(thr_mm),
    }


def _surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)

    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
    gt_surface = gt ^ ndimage.binary_erosion(gt, structure=structure, border_value=0)

    if not pred_surface.any() or not gt_surface.any():
        return np.array([], dtype=float), np.array([], dtype=float)

    dt_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    dist_pred_gt = dt_to_gt[pred_surface]
    dist_gt_pred = dt_to_pred[gt_surface]
    return dist_pred_gt.astype(float), dist_gt_pred.astype(float)


def segmentation_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, float, float],
) -> dict[str, Any]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    tp = np.logical_and(pred, gt).sum(dtype=np.int64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.int64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.int64)

    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn

    voxel_volume = float(np.prod(spacing))
    pred_volume = float(pred.sum(dtype=np.int64) * voxel_volume)
    gt_volume = float(gt.sum(dtype=np.int64) * voxel_volume)

    dist_pred_gt, dist_gt_pred = _surface_distances(pred, gt, spacing)
    all_surface = np.concatenate([dist_pred_gt, dist_gt_pred]) if dist_pred_gt.size and dist_gt_pred.size else np.array([], dtype=float)

    pred_consistency = segmentation_consistency_metrics(pred)
    gt_consistency = segmentation_consistency_metrics(gt)
    cldice = cldice_metric(pred, gt)

    return {
        "pred_voxels": int(pred.sum(dtype=np.int64)),
        "gt_voxels": int(gt.sum(dtype=np.int64)),
        "dice": float((2 * tp) / denom_dice) if denom_dice > 0 else 1.0,
        "iou": float(tp / denom_iou) if denom_iou > 0 else 1.0,
        "volume_pred_mm3": pred_volume,
        "volume_gt_mm3": gt_volume,
        "volume_abs_diff_mm3": float(abs(pred_volume - gt_volume)),
        "asd_mm": _safe_stat(all_surface, np.mean),
        "hd95_mm": _safe_stat(all_surface, lambda x: np.percentile(x, 95)),
        "hdmax_mm": _safe_stat(all_surface, np.max),
        **{f"pred_{key}": value for key, value in pred_consistency.items()},
        **{f"gt_{key}": value for key, value in gt_consistency.items()},
        "component_diff_26": abs(pred_consistency["mask_components_26"] - gt_consistency["mask_components_26"]),
        "skeleton_component_diff_26": abs(
            pred_consistency["skeleton_components_26"] - gt_consistency["skeleton_components_26"]
        ),
        **cldice,
    }


def feature_descriptor_metrics(pred_desc: np.ndarray, gt_desc: np.ndarray) -> dict[str, Optional[float]]:
    pred = np.asarray(pred_desc, dtype=float).ravel()
    gt = np.asarray(gt_desc, dtype=float).ravel()

    if pred.size == 0 or gt.size == 0:
        return {"descriptor_l1": None, "descriptor_l2": None, "descriptor_cosine": None}

    if pred.size != gt.size:
        min_len = min(pred.size, gt.size)
        pred = pred[:min_len]
        gt = gt[:min_len]

    l1 = float(np.mean(np.abs(pred - gt)))
    l2 = float(np.sqrt(np.mean((pred - gt) ** 2)))

    pred_norm = float(np.linalg.norm(pred))
    gt_norm = float(np.linalg.norm(gt))
    if pred_norm == 0.0 or gt_norm == 0.0:
        cosine = None
    else:
        cosine = float(np.dot(pred, gt) / (pred_norm * gt_norm))

    return {
        "descriptor_l1": l1,
        "descriptor_l2": l2,
        "descriptor_cosine": cosine,
    }
