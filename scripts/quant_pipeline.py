#!/usr/bin/env python3
"""Unified quantitative pipeline entry for 5-step coronary workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_seg.quant.pipeline import (
    evaluate_step1_segmentation,
    evaluate_step2_centerline_from_mask,
    evaluate_step3_repair,
    evaluate_step4_features,
    evaluate_step5_render,
)


def _build_step1_parser(subparsers) -> None:
    parser = subparsers.add_parser("step1", help="Segmentation quantitative evaluation.")
    parser.add_argument("--pred-mask", type=Path, required=True)
    parser.add_argument("--gt-mask", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)


def _build_step2_parser(subparsers) -> None:
    parser = subparsers.add_parser("step2", help="Centerline extraction/evaluation from TotalSeg mask.")
    parser.add_argument("--seg-mask", type=Path, required=True, help="TotalSeg mask path.")
    parser.add_argument("--gt-centerline", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["skeleton", "vmtk"], default="skeleton")
    parser.add_argument("--thr", type=float, default=1.0)


def _build_step3_parser(subparsers) -> None:
    parser = subparsers.add_parser("step3", help="Centerline repair evaluation.")
    parser.add_argument("--repaired-centerline", type=Path, required=True)
    parser.add_argument("--gt-centerline", type=Path, required=True)
    parser.add_argument("--baseline-centerline", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thr", type=float, default=1.0)


def _build_step4_parser(subparsers) -> None:
    parser = subparsers.add_parser("step4", help="Vessel feature extraction comparison.")
    parser.add_argument("--pred-features", type=Path, required=True)
    parser.add_argument("--gt-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)


def _build_step5_parser(subparsers) -> None:
    parser = subparsers.add_parser("step5", help="Render reconstruction vs GT mask.")
    parser.add_argument("--gt-mask", type=Path, required=True)
    parser.add_argument("--pred-mesh", type=Path, default=None)
    parser.add_argument("--pred-features", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thr", type=float, default=1.0)


def _build_all_parser(subparsers) -> None:
    parser = subparsers.add_parser("all", help="Run step1~step5 in one command.")
    parser.add_argument("--pred-mask", type=Path, required=True)
    parser.add_argument("--gt-mask", type=Path, required=True)
    parser.add_argument("--gt-centerline", type=Path, required=True)
    parser.add_argument("--step2-backend", choices=["skeleton", "vmtk"], default="skeleton")
    parser.add_argument("--repaired-centerline", type=Path, required=True)
    parser.add_argument("--pred-features", type=Path, required=True)
    parser.add_argument("--gt-features", type=Path, required=True)
    parser.add_argument("--pred-mesh", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thr", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coronary 5-step quantitative pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_step1_parser(subparsers)
    _build_step2_parser(subparsers)
    _build_step3_parser(subparsers)
    _build_step4_parser(subparsers)
    _build_step5_parser(subparsers)
    _build_all_parser(subparsers)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "step1":
        evaluate_step1_segmentation(args.pred_mask, args.gt_mask, args.out_dir)
        print("[step1] completed")
        return

    if args.command == "step2":
        evaluate_step2_centerline_from_mask(
            args.seg_mask,
            args.gt_centerline,
            args.out_dir,
            thr_mm=args.thr,
            backend=args.backend,
        )
        print("[step2] completed")
        return

    if args.command == "step3":
        evaluate_step3_repair(
            baseline_centerline_vtp=args.baseline_centerline,
            repaired_centerline_vtp=args.repaired_centerline,
            gt_centerline_vtp=args.gt_centerline,
            out_dir=args.out_dir,
            thr_mm=args.thr,
        )
        print("[step3] completed")
        return

    if args.command == "step4":
        evaluate_step4_features(args.pred_features, args.gt_features, args.out_dir)
        print("[step4] completed")
        return

    if args.command == "step5":
        evaluate_step5_render(
            gt_mask_path=args.gt_mask,
            pred_mesh_path=args.pred_mesh,
            pred_features_dir=args.pred_features,
            out_dir=args.out_dir,
            thr_mm=args.thr,
        )
        print("[step5] completed")
        return

    if args.command == "all":
        evaluate_step1_segmentation(args.pred_mask, args.gt_mask, args.out_dir)
        evaluate_step2_centerline_from_mask(
            args.pred_mask,
            args.gt_centerline,
            args.out_dir,
            thr_mm=args.thr,
            backend=args.step2_backend,
        )
        evaluate_step3_repair(
            baseline_centerline_vtp=None,
            repaired_centerline_vtp=args.repaired_centerline,
            gt_centerline_vtp=args.gt_centerline,
            out_dir=args.out_dir,
            thr_mm=args.thr,
        )
        evaluate_step4_features(args.pred_features, args.gt_features, args.out_dir)
        evaluate_step5_render(
            gt_mask_path=args.gt_mask,
            pred_mesh_path=args.pred_mesh,
            pred_features_dir=args.pred_features,
            out_dir=args.out_dir,
            thr_mm=args.thr,
        )
        print("[all] step1~step5 completed")
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
