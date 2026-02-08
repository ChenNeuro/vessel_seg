#!/usr/bin/env python3
"""One-command runner for ASOCA Normal_1 step1~step5 quantitative pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_cmd(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(x) for x in cmd))


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    _print_cmd(cmd)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(cmd)}")


def _extend_extra_args(cmd: list[str], extras: list[str]) -> None:
    for chunk in extras:
        cmd.extend(shlex.split(chunk))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_path(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _vtp_stats(path: Path) -> dict[str, int | None]:
    try:
        import vtk  # type: ignore
    except Exception:
        return {"points": None, "lines": None}

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    return {
        "points": int(poly.GetNumberOfPoints()),
        "lines": int(poly.GetNumberOfLines()),
    }


def _skip_stage(skip_existing: bool, metrics_path: Path) -> bool:
    return skip_existing and metrics_path.exists()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ASOCA Normal_1 full step1~step5 pipeline in one command.",
    )

    parser.add_argument(
        "--ct",
        type=Path,
        default=Path("ASOCA2020/Normal/CTCA_nii/Normal_1.nii.gz"),
        help="Input CT NIfTI.",
    )
    parser.add_argument(
        "--gt-mask",
        type=Path,
        default=Path("ASOCA2020/Normal/Annotations_nii/Normal_1.nii.gz"),
        help="GT mask NIfTI.",
    )
    parser.add_argument(
        "--gt-centerline",
        type=Path,
        default=Path("ASOCA2020/Normal/Centerlines/Normal_1.vtp"),
        help="GT centerline VTP.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/quant/Normal_1"),
        help="Pipeline output directory.",
    )

    parser.add_argument("--task", default="coronary_arteries", help="TotalSegmentator task.")
    parser.add_argument("--pred-file", default="coronary_arteries.nii.gz", help="Pred mask file name.")
    parser.add_argument("--totalseg-cmd", default="TotalSegmentator", help="TotalSegmentator command path.")
    parser.add_argument("--totalseg-arg", action="append", default=[], help="Extra arg forwarded to TotalSegmentator.")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator --fast mode.")

    parser.add_argument(
        "--step2-backend",
        choices=["skeleton", "vmtk"],
        default="vmtk",
        help="Centerline extraction backend in step2.",
    )
    parser.add_argument("--thr", type=float, default=1.0, help="Distance threshold in mm for coverage metrics.")

    parser.add_argument("--repair-prob", type=Path, default=None, help="Probability map for repair; defaults to pred mask.")
    parser.add_argument("--prob-thresh", type=float, default=0.2)
    parser.add_argument("--max-dist", type=float, default=10.0)
    parser.add_argument("--max-bridge-len", type=float, default=25.0)
    parser.add_argument("--max-angle-deg", type=float, default=90.0)
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--w-prob", type=float, default=1.0)
    parser.add_argument("--w-dist", type=float, default=0.6)
    parser.add_argument("--outside-penalty", type=float, default=10.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--max-curvature", type=float, default=0.4)
    parser.add_argument("--murray-exp", type=float, default=3.0)
    parser.add_argument("--murray-tol", type=float, default=0.5)

    parser.add_argument("--pred-features-dir", type=Path, default=None, help="Override output dir for predicted features.")
    parser.add_argument("--gt-features-dir", type=Path, default=None, help="Override output dir for GT features.")
    parser.add_argument(
        "--pred-extract-arg",
        action="append",
        default=[],
        help="Extra arg for `python -m vessel_seg.shape extract` on predicted mask.",
    )
    parser.add_argument(
        "--gt-extract-arg",
        action="append",
        default=[],
        help="Extra arg for `python -m vessel_seg.shape extract` on GT mask.",
    )

    parser.add_argument("--skip-existing", action="store_true", help="Skip a stage when its metrics.json already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without executing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ct_path = (ROOT / args.ct).resolve() if not args.ct.is_absolute() else args.ct
    gt_mask_path = (ROOT / args.gt_mask).resolve() if not args.gt_mask.is_absolute() else args.gt_mask
    gt_centerline_path = (ROOT / args.gt_centerline).resolve() if not args.gt_centerline.is_absolute() else args.gt_centerline
    out_dir = (ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir

    pred_features_dir = args.pred_features_dir or (out_dir / "features_pred")
    gt_features_dir = args.gt_features_dir or (out_dir / "features_gt")

    if not args.dry_run:
        _ensure_path(ct_path, label="CT")
        _ensure_path(gt_mask_path, label="GT mask")
        _ensure_path(gt_centerline_path, label="GT centerline")

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "started_at_utc": _now_utc(),
        "root": str(ROOT),
        "config": {
            "ct": str(ct_path),
            "gt_mask": str(gt_mask_path),
            "gt_centerline": str(gt_centerline_path),
            "out_dir": str(out_dir),
            "task": args.task,
            "pred_file": args.pred_file,
            "step2_backend": args.step2_backend,
            "thr": args.thr,
            "skip_existing": bool(args.skip_existing),
            "dry_run": bool(args.dry_run),
        },
        "stages": {},
    }

    step1_metrics = out_dir / "step1_segmentation" / "metrics.json"
    step2_metrics = out_dir / "step2_centerline" / "metrics.json"
    step3_metrics = out_dir / "step3_repair" / "metrics.json"
    step4_metrics = out_dir / "step4_features" / "metrics.json"
    step5_metrics = out_dir / "step5_render" / "metrics.json"

    # Step1
    if _skip_stage(args.skip_existing, step1_metrics):
        print("[skip] step1 metrics exists")
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/step1_totalseg_segment_and_eval.py"),
            "--ct",
            str(ct_path),
            "--gt-mask",
            str(gt_mask_path),
            "--out-dir",
            str(out_dir),
            "--task",
            args.task,
            "--pred-file",
            args.pred_file,
            "--totalseg-cmd",
            args.totalseg_cmd,
        ]
        if args.fast:
            cmd.append("--fast")
        for one_arg in args.totalseg_arg:
            cmd.extend(["--totalseg-arg", one_arg])
        _run(cmd, dry_run=args.dry_run)

    pred_mask = out_dir / "step1_segmentation" / "totalseg_output" / args.pred_file
    if not args.dry_run:
        _ensure_path(pred_mask, label="Predicted mask")
    summary["stages"]["step1"] = _read_json(step1_metrics)

    # Step2
    if _skip_stage(args.skip_existing, step2_metrics):
        print("[skip] step2 metrics exists")
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/step2_centerline_totalseg.py"),
            "--totalseg-mask",
            str(pred_mask),
            "--gt-centerline",
            str(gt_centerline_path),
            "--out-dir",
            str(out_dir),
            "--backend",
            args.step2_backend,
            "--thr",
            str(args.thr),
        ]
        _run(cmd, dry_run=args.dry_run)

    baseline_centerline = out_dir / "step2_centerline" / "pred_centerline.vtp"
    needs_repair_run = not _skip_stage(args.skip_existing, step3_metrics)

    # Fallback: step2 sometimes outputs points-only VTP without line cells.
    if args.step2_backend == "vmtk" and baseline_centerline.exists() and not args.dry_run and needs_repair_run:
        stats = _vtp_stats(baseline_centerline)
        summary["stages"]["step2_centerline_vtp"] = stats
        if stats.get("lines") == 0:
            fallback_centerline = out_dir / "step2_centerline" / "pred_centerline_poly.vtp"
            fallback_report = out_dir / "step2_centerline" / "pred_centerline_poly_report.json"
            cmd = [
                sys.executable,
                str(ROOT / "scripts/vmtk_extract_centerlines.py"),
                "--mask",
                str(pred_mask),
                "--out",
                str(fallback_centerline),
                "--gt-centerline",
                str(gt_centerline_path),
                "--report",
                str(fallback_report),
            ]
            _run(cmd, dry_run=args.dry_run)
            baseline_centerline = fallback_centerline

    if not args.dry_run:
        _ensure_path(baseline_centerline, label="Step2 baseline centerline")
    summary["stages"]["step2"] = _read_json(step2_metrics)

    # Step3
    repaired_vtp = out_dir / "step3_repair" / "repaired.vtp"
    repair_report = out_dir / "step3_repair" / "repair_report.json"
    repair_prob = args.repair_prob
    if repair_prob is None:
        repair_prob = pred_mask
    elif not repair_prob.is_absolute():
        repair_prob = (ROOT / repair_prob).resolve()

    if _skip_stage(args.skip_existing, step3_metrics):
        print("[skip] step3 metrics exists")
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/repair_centerline.py"),
            "--prob",
            str(repair_prob),
            "--vtp",
            str(baseline_centerline),
            "--out",
            str(repaired_vtp),
            "--report",
            str(repair_report),
            "--prob_thresh",
            str(args.prob_thresh),
            "--max_dist",
            str(args.max_dist),
            "--max_bridge_len",
            str(args.max_bridge_len),
            "--max_angle_deg",
            str(args.max_angle_deg),
            "--max_pairs",
            str(args.max_pairs),
            "--w_prob",
            str(args.w_prob),
            "--w_dist",
            str(args.w_dist),
            "--outside_penalty",
            str(args.outside_penalty),
            "--smooth_window",
            str(args.smooth_window),
            "--max_curvature",
            str(args.max_curvature),
            "--murray_exp",
            str(args.murray_exp),
            "--murray_tol",
            str(args.murray_tol),
        ]
        _run(cmd, dry_run=args.dry_run)

        cmd = [
            sys.executable,
            str(ROOT / "scripts/quant_pipeline.py"),
            "step3",
            "--baseline-centerline",
            str(baseline_centerline),
            "--repaired-centerline",
            str(repaired_vtp),
            "--gt-centerline",
            str(gt_centerline_path),
            "--out-dir",
            str(out_dir),
            "--thr",
            str(args.thr),
        ]
        _run(cmd, dry_run=args.dry_run)

    summary["stages"]["step3"] = _read_json(step3_metrics)

    # Step4
    if _skip_stage(args.skip_existing, step4_metrics):
        print("[skip] step4 metrics exists")
    else:
        cmd = [
            sys.executable,
            "-m",
            "vessel_seg.shape",
            "extract",
            "--seg",
            str(pred_mask),
            "--out",
            str(pred_features_dir),
        ]
        _extend_extra_args(cmd, args.pred_extract_arg)
        _run(cmd, dry_run=args.dry_run)

        cmd = [
            sys.executable,
            "-m",
            "vessel_seg.shape",
            "extract",
            "--seg",
            str(gt_mask_path),
            "--out",
            str(gt_features_dir),
        ]
        _extend_extra_args(cmd, args.gt_extract_arg)
        _run(cmd, dry_run=args.dry_run)

        cmd = [
            sys.executable,
            str(ROOT / "scripts/quant_pipeline.py"),
            "step4",
            "--pred-features",
            str(pred_features_dir),
            "--gt-features",
            str(gt_features_dir),
            "--out-dir",
            str(out_dir),
        ]
        _run(cmd, dry_run=args.dry_run)

    summary["stages"]["step4"] = _read_json(step4_metrics)

    # Step5
    if _skip_stage(args.skip_existing, step5_metrics):
        print("[skip] step5 metrics exists")
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/quant_pipeline.py"),
            "step5",
            "--pred-features",
            str(pred_features_dir),
            "--gt-mask",
            str(gt_mask_path),
            "--out-dir",
            str(out_dir),
            "--thr",
            str(args.thr),
        ]
        _run(cmd, dry_run=args.dry_run)

    summary["stages"]["step5"] = _read_json(step5_metrics)
    summary["finished_at_utc"] = _now_utc()

    summary_path = out_dir / "pipeline_run_summary.json"
    if args.dry_run:
        print("[done] dry-run finished; no files were written.")
    else:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
