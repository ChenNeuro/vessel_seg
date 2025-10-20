"""Compute evaluation metrics between ASOCA predictions and reference segmentations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_sum = gt.sum()
    pred_sum = pred.sum()
    if gt_sum + pred_sum == 0:
        return 1.0
    intersection = np.logical_and(gt, pred).sum()
    return 2.0 * intersection / (gt_sum + pred_sum)


def hd95(gt_surface: np.ndarray, pred_surface: np.ndarray) -> float:
    if gt_surface.size == 0 or pred_surface.size == 0:
        return float("inf")
    distances = [
        directed_hausdorff(gt_surface, pred_surface)[0],
        directed_hausdorff(pred_surface, gt_surface)[0],
    ]
    return np.percentile(distances, 95)


def _surface_points(mask: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    from scipy import ndimage as ndi

    structure = ndi.generate_binary_structure(3, 1)
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=1)
    surface = np.logical_xor(mask, eroded)
    coords = np.argwhere(surface)
    return coords * np.asarray(spacing)


def load_nifti(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    image = nib.load(str(path))
    data = image.get_fdata() > 0.5
    header = image.header
    spacing = header.get_zooms()[:3]
    return data.astype(np.uint8), spacing


def evaluate_case(gt_path: Path, pred_path: Path) -> Dict[str, float]:
    gt_mask, spacing = load_nifti(gt_path)
    pred_mask, _ = load_nifti(pred_path)
    dice = dice_coefficient(gt_mask, pred_mask)
    hd = hd95(_surface_points(gt_mask, spacing), _surface_points(pred_mask, spacing))
    return {"dice": dice, "hd95": hd}


def gather_pairs(gt_dir: Path, pred_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for gt_file in sorted(gt_dir.glob("*.nii*")):
        pred_file = pred_dir / gt_file.name
        if pred_file.exists():
            pairs.append((gt_file, pred_file))
    return pairs


def main(gt_dir: str, pred_dir: str, output: str) -> None:
    gt_dir_path = Path(gt_dir)
    pred_dir_path = Path(pred_dir)
    results = {}
    pairs = gather_pairs(gt_dir_path, pred_dir_path)
    dices = []
    hd95s = []
    for gt_path, pred_path in pairs:
        metrics = evaluate_case(gt_path, pred_path)
        results[gt_path.name] = metrics
        dices.append(metrics["dice"])
        hd95s.append(metrics["hd95"])
    summary = {
        "per_case": results,
        "mean_dice": float(np.mean(dices)) if dices else None,
        "mean_hd95": float(np.mean(hd95s)) if hd95s else None,
        "case_count": len(pairs),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ASOCA predictions with ground truth labels.")
    parser.add_argument("--gt", required=True, help="Directory containing ground truth NIfTI masks.")
    parser.add_argument("--pred", required=True, help="Directory containing predicted masks.")
    parser.add_argument(
        "--output",
        default="outputs/asoca_metrics.json",
        help="Destination JSON file for aggregated metrics.",
    )
    args = parser.parse_args()
    main(args.gt, args.pred, args.output)
