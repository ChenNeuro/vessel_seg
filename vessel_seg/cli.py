from __future__ import annotations

import argparse
from pathlib import Path

from vessel_seg.clinical_completion import add_clinical_completion_arguments, run_clinical_completion_command
from vessel_seg.quant.pipeline import (
    evaluate_step1_segmentation,
    evaluate_step2_centerline_from_mask,
    evaluate_step3_repair,
    evaluate_step4_features,
    evaluate_step5_render,
)
from vessel_seg.workflows import add_normal1_arguments, run_normal1_pipeline


def add_quant_subcommands(parser: argparse.ArgumentParser, *, dest: str) -> None:
    quant_subparsers = parser.add_subparsers(dest=dest, required=True)

    step1 = quant_subparsers.add_parser("step1", help="Segmentation quantitative evaluation.")
    step1.add_argument("--pred-mask", type=Path, required=True)
    step1.add_argument("--gt-mask", type=Path, required=True)
    step1.add_argument("--out-dir", type=Path, required=True)

    step2 = quant_subparsers.add_parser("step2", help="Centerline extraction/evaluation from mask.")
    step2.add_argument("--seg-mask", type=Path, required=True)
    step2.add_argument("--gt-centerline", type=Path, required=True)
    step2.add_argument("--out-dir", type=Path, required=True)
    step2.add_argument("--backend", choices=["skeleton", "vmtk"], default="skeleton")
    step2.add_argument("--thr", type=float, default=1.0)

    step3 = quant_subparsers.add_parser("step3", help="Centerline repair evaluation.")
    step3.add_argument("--repaired-centerline", type=Path, required=True)
    step3.add_argument("--gt-centerline", type=Path, required=True)
    step3.add_argument("--baseline-centerline", type=Path, default=None)
    step3.add_argument("--out-dir", type=Path, required=True)
    step3.add_argument("--thr", type=float, default=1.0)

    step4 = quant_subparsers.add_parser("step4", help="Vessel feature extraction comparison.")
    step4.add_argument("--pred-features", type=Path, required=True)
    step4.add_argument("--gt-features", type=Path, required=True)
    step4.add_argument("--out-dir", type=Path, required=True)

    step5 = quant_subparsers.add_parser("step5", help="Render reconstruction vs GT mask.")
    step5.add_argument("--gt-mask", type=Path, required=True)
    step5.add_argument("--pred-mesh", type=Path, default=None)
    step5.add_argument("--pred-features", type=Path, default=None)
    step5.add_argument("--out-dir", type=Path, required=True)
    step5.add_argument("--thr", type=float, default=1.0)


def build_quant_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coronary 5-step quantitative pipeline.")
    add_quant_subcommands(parser, dest="quant_command")
    return parser


def run_quant_command(args: argparse.Namespace) -> None:
    if args.quant_command == "step1":
        evaluate_step1_segmentation(args.pred_mask, args.gt_mask, args.out_dir)
        print("[step1] completed")
        return
    if args.quant_command == "step2":
        evaluate_step2_centerline_from_mask(args.seg_mask, args.gt_centerline, args.out_dir, thr_mm=args.thr, backend=args.backend)
        print("[step2] completed")
        return
    if args.quant_command == "step3":
        evaluate_step3_repair(args.baseline_centerline, args.repaired_centerline, args.gt_centerline, args.out_dir, thr_mm=args.thr)
        print("[step3] completed")
        return
    if args.quant_command == "step4":
        evaluate_step4_features(args.pred_features, args.gt_features, args.out_dir)
        print("[step4] completed")
        return
    if args.quant_command == "step5":
        evaluate_step5_render(
            gt_mask_path=args.gt_mask,
            pred_mesh_path=args.pred_mesh,
            pred_features_dir=args.pred_features,
            out_dir=args.out_dir,
            thr_mm=args.thr,
        )
        print("[step5] completed")
        return
    raise SystemExit(f"Unsupported quant command: {args.quant_command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for vessel_seg workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quant_parser = subparsers.add_parser(
        "quant",
        help="Quantitative pipeline commands.",
        description="Quantitative pipeline commands.",
    )
    add_quant_subcommands(quant_parser, dest="quant_command")

    normal1_parser = subparsers.add_parser(
        "normal1",
        help="Run the ASOCA Normal_1 end-to-end workflow.",
        description="Run ASOCA Normal_1 full step1~step5 pipeline in one command.",
    )
    add_normal1_arguments(normal1_parser)

    clinical_parser = subparsers.add_parser(
        "clinical-demo",
        help="Run coronary completion + longitudinal section + clinical dashboard workflow.",
        description="Run coronary completion + longitudinal section + clinical dashboard workflow.",
    )
    add_clinical_completion_arguments(clinical_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "quant":
        run_quant_command(args)
        return

    if args.command == "normal1":
        run_normal1_pipeline(args)
        return

    if args.command == "clinical-demo":
        run_clinical_completion_command(args)
        return

    raise SystemExit(f"Unsupported command: {args.command}")
