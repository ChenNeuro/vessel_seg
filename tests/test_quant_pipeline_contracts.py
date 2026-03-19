from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from vessel_seg.quant.pipeline import evaluate_step1_segmentation, evaluate_step4_features


def _write_mask(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.uint8), np.eye(4)), str(path))


def test_step1_writes_metrics_contract(tmp_path: Path) -> None:
    pred_mask = np.zeros((6, 6, 6), dtype=np.uint8)
    gt_mask = np.zeros((6, 6, 6), dtype=np.uint8)
    pred_mask[1:4, 1:4, 1:4] = 1
    gt_mask[2:5, 2:5, 2:5] = 1

    pred_path = tmp_path / "pred.nii.gz"
    gt_path = tmp_path / "gt.nii.gz"
    out_dir = tmp_path / "out"

    _write_mask(pred_path, pred_mask)
    _write_mask(gt_path, gt_mask)

    payload = evaluate_step1_segmentation(pred_path, gt_path, out_dir)

    json_path = out_dir / "step1_segmentation" / "metrics.json"
    csv_path = out_dir / "step1_segmentation" / "metrics.csv"

    assert json_path.exists()
    assert csv_path.exists()
    assert payload["stage"] == "step1_segmentation"
    assert payload["metrics"]["dice"] < 1.0

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["stage"] == "step1_segmentation"
    assert "hd95_mm" in saved["metrics"]

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["stage"] == "step1_segmentation"
    assert "dice" in row


def test_step4_writes_metrics_contract(tmp_path: Path) -> None:
    pred_dir = tmp_path / "pred_features"
    gt_dir = tmp_path / "gt_features"
    out_dir = tmp_path / "out"
    pred_dir.mkdir()
    gt_dir.mkdir()

    np.save(pred_dir / "global_descriptor.npy", np.array([1.0, 2.0, 3.0], dtype=float))
    np.save(gt_dir / "global_descriptor.npy", np.array([1.0, 2.5, 2.0], dtype=float))

    (pred_dir / "summary.json").write_text(
        json.dumps({"branch_count": 4, "global_descriptor": "global_descriptor.npy"}),
        encoding="utf-8",
    )
    (gt_dir / "summary.json").write_text(
        json.dumps({"branch_count": 6, "global_descriptor": "global_descriptor.npy"}),
        encoding="utf-8",
    )

    payload = evaluate_step4_features(pred_dir, gt_dir, out_dir)

    json_path = out_dir / "step4_features" / "metrics.json"
    csv_path = out_dir / "step4_features" / "metrics.csv"

    assert json_path.exists()
    assert csv_path.exists()
    assert payload["metrics"]["branch_count_abs_diff"] == 2

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["stage"] == "step4_features"
    assert "descriptor_l2" in saved["metrics"]

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["stage"] == "step4_features"
    assert row["branch_count_abs_diff"] == "2"
