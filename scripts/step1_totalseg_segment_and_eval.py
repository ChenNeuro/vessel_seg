#!/usr/bin/env python3
"""Step1: run TotalSegmentator from CT and evaluate against GT mask."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_seg.io import load_volume
from vessel_seg.quant.pipeline import evaluate_step1_segmentation
from vessel_seg.segmentation_interface import TotalSegmentationBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TotalSegmentator and evaluate segmentation quality.")
    parser.add_argument("--ct", type=Path, required=True, help="Input CT volume (.nii/.nii.gz).")
    parser.add_argument("--gt-mask", type=Path, required=True, help="Ground-truth binary mask (.nii/.nii.gz).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for prediction and metrics.")
    parser.add_argument(
        "--pred-file",
        type=str,
        default="coronary_arteries.nii.gz",
        help="Expected output filename inside TotalSeg output directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="coronary_arteries",
        help="TotalSegmentator task name (default: coronary_arteries).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable TotalSegmentator fast mode.",
    )
    parser.add_argument(
        "--totalseg-cmd",
        type=str,
        default="TotalSegmentator",
        help="TotalSegmentator executable name/path.",
    )
    parser.add_argument(
        "--totalseg-arg",
        action="append",
        default=[],
        help="Extra arg passed to TotalSegmentator. Repeat this flag for multiple args.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.fast and args.task == "coronary_arteries":
        raise SystemExit("TotalSegmentator task 'coronary_arteries' does not support --fast. Remove --fast.")

    volume = load_volume(args.ct)
    stage1_dir = args.out_dir / "step1_segmentation"
    ts_out = stage1_dir / "totalseg_output"

    backend = TotalSegmentationBackend(
        output_dir=ts_out,
        prediction_file=args.pred_file,
        command=args.totalseg_cmd,
        task=args.task,
        fast=args.fast,
        extra_args=args.totalseg_arg,
    )
    backend.predict_mask(volume)

    pred_mask = ts_out / args.pred_file
    payload = evaluate_step1_segmentation(pred_mask, args.gt_mask, args.out_dir)

    print("[step1] done")
    print("pred mask:", pred_mask)
    print("dice:", payload["metrics"].get("dice"))
    print("hd95_mm:", payload["metrics"].get("hd95_mm"))


if __name__ == "__main__":
    main()
