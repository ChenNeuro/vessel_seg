#!/usr/bin/env python3
"""Step2: extract/evaluate centerline from TotalSeg mask against GT centerline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_seg.quant.pipeline import evaluate_step2_centerline_from_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step2 centerline evaluation from TotalSeg mask.")
    parser.add_argument("--totalseg-mask", type=Path, required=True, help="TotalSeg binary vessel mask (.nii/.nii.gz).")
    parser.add_argument("--gt-centerline", type=Path, required=True, help="GT centerline VTP.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for step2 metrics.")
    parser.add_argument(
        "--backend",
        choices=["skeleton", "vmtk"],
        default="skeleton",
        help="Centerline extraction backend. 'skeleton' requires only mask; 'vmtk' uses VMTK pipeline.",
    )
    parser.add_argument("--thr", type=float, default=1.0, help="Coverage threshold in mm.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = evaluate_step2_centerline_from_mask(
        seg_mask_path=args.totalseg_mask,
        gt_centerline_vtp=args.gt_centerline,
        out_dir=args.out_dir,
        thr_mm=args.thr,
        backend=args.backend,
    )
    metrics = payload["metrics"]
    print("[step2] done")
    print("pred2gt_mean:", metrics.get("pred2gt_mean"))
    print("pred2gt_p95:", metrics.get("pred2gt_p95"))
    print(f"coverage_pred@{args.thr:g}mm:", metrics.get(f"coverage_pred@{args.thr:g}mm"))


if __name__ == "__main__":
    main()
