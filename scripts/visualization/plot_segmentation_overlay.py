#!/usr/bin/env python3
"""Render CT slices with predicted and optional GT segmentation overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.io import load_mask, load_volume
from vessel_seg.quant.metrics import segmentation_metrics
from vessel_seg.visualization import save_segmentation_slices


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize coronary segmentation on CT slices.")
    parser.add_argument("--ct", type=Path, required=True, help="CT volume (.nii/.nii.gz).")
    parser.add_argument("--pred-mask", type=Path, required=True, help="Predicted binary mask (.nii/.nii.gz).")
    parser.add_argument("--gt-mask", type=Path, default=None, help="Optional ground-truth binary mask (.nii/.nii.gz).")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="Slice axis.")
    parser.add_argument("--num-slices", type=int, default=6, help="Number of representative slices.")
    parser.add_argument("--window-low", type=float, default=-200.0, help="CT window low bound.")
    parser.add_argument("--window-high", type=float, default=800.0, help="CT window high bound.")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional JSON path to dump segmentation metrics when --gt-mask is provided.",
    )
    args = parser.parse_args()

    volume = load_volume(args.ct)
    pred_mask = load_mask(args.pred_mask)

    metrics = None
    gt_mask_data = None
    if args.gt_mask is not None:
        gt_mask = load_mask(args.gt_mask)
        metrics = segmentation_metrics(pred_mask.data, gt_mask.data, pred_mask.spacing)
        gt_mask_data = gt_mask.data
        if args.metrics_json is not None:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    out_path = save_segmentation_slices(
        volume.data,
        pred_mask.data,
        args.out,
        gt_mask_data,
        axis=args.axis,
        num_slices=args.num_slices,
        window=(args.window_low, args.window_high),
        metrics=metrics,
        title=args.title,
    )

    print(f"saved: {out_path}")
    if metrics is not None:
        print(
            "metrics:",
            json.dumps(
                {
                    key: metrics[key]
                    for key in (
                        "dice",
                        "hd95_mm",
                        "cldice",
                        "pred_mask_components_26",
                        "pred_skeleton_components_26",
                    )
                    if key in metrics
                },
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
