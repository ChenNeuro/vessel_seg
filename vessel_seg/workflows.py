from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vessel_seg.io import load_volume
from vessel_seg.quant.pipeline import (
    evaluate_step1_segmentation,
    evaluate_step2_centerline_from_mask,
    evaluate_step3_repair,
    evaluate_step4_features,
    evaluate_step5_render,
)
from vessel_seg.segmentation_interface import TotalSegmentationBackend


ROOT = Path(__file__).resolve().parents[1]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_cmd(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(part) for part in cmd))


def _run_command(cmd: list[str], *, dry_run: bool = False) -> None:
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
        import vtk
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


def add_normal1_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--ct", type=Path, default=Path("ASOCA2020/Normal/CTCA_nii/Normal_1.nii.gz"))
    parser.add_argument("--gt-mask", type=Path, default=Path("ASOCA2020/Normal/Annotations_nii/Normal_1.nii.gz"))
    parser.add_argument("--gt-centerline", type=Path, default=Path("ASOCA2020/Normal/Centerlines/Normal_1.vtp"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/quant/Normal_1"))
    parser.add_argument("--task", default="coronary_arteries", help="TotalSegmentator task.")
    parser.add_argument("--pred-file", default="coronary_arteries.nii.gz", help="Pred mask file name.")
    parser.add_argument("--totalseg-cmd", default="TotalSegmentator", help="TotalSegmentator command path.")
    parser.add_argument("--totalseg-arg", action="append", default=[], help="Extra arg forwarded to TotalSegmentator.")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator --fast mode.")
    parser.add_argument("--step2-backend", choices=["skeleton", "vmtk"], default="vmtk")
    parser.add_argument("--step2-skeleton-max-components", type=int, default=2)
    parser.add_argument("--thr", type=float, default=1.0, help="Distance threshold in mm for coverage metrics.")
    parser.add_argument("--repair-prob", type=Path, default=None, help="Probability map for repair; defaults to pred mask.")
    parser.add_argument("--prob-thresh", type=float, default=0.2)
    parser.add_argument("--max-dist", type=float, default=8.0)
    parser.add_argument("--max-bridge-len", type=float, default=20.0)
    parser.add_argument("--max-angle-deg", type=float, default=85.0)
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--w-prob", type=float, default=1.0)
    parser.add_argument("--w-dist", type=float, default=0.6)
    parser.add_argument("--outside-penalty", type=float, default=10.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--max-curvature", type=float, default=0.4)
    parser.add_argument("--murray-exp", type=float, default=3.0)
    parser.add_argument("--murray-tol", type=float, default=0.5)
    parser.add_argument("--coord-mode", choices=["auto", "abs_spacing", "affine"], default="auto")
    parser.add_argument("--no-align-centerline", action="store_true")
    parser.add_argument("--align-max-points", type=int, default=8000)
    parser.add_argument("--align-outside-penalty", type=float, default=20.0)
    parser.add_argument("--align-steps", default="10,5,2,1,0.5,0.25")
    parser.add_argument("--densify-step-mm", type=float, default=0.5)
    parser.add_argument("--bridge-min-count", type=int, default=1)
    parser.add_argument("--no-bridge-relax-if-none", action="store_true")
    parser.add_argument("--bridge-relax-dist-factor", type=float, default=1.5)
    parser.add_argument("--bridge-relax-angle-add", type=float, default=20.0)
    parser.add_argument("--bridge-relax-len-factor", type=float, default=1.4)
    parser.add_argument("--bridge-relax-curvature-factor", type=float, default=1.5)
    parser.add_argument("--bridge-relax-murray-add", type=float, default=0.15)
    parser.add_argument("--bridge-smooth-iterations", type=int, default=4)
    parser.add_argument("--regularize-scope", choices=["all", "bridges", "none"], default="bridges")
    parser.add_argument("--pred-features-dir", type=Path, default=None)
    parser.add_argument("--gt-features-dir", type=Path, default=None)
    parser.add_argument("--pred-extract-arg", action="append", default=[])
    parser.add_argument("--gt-extract-arg", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_normal1_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ASOCA Normal_1 full step1~step5 pipeline in one command.",
    )
    add_normal1_arguments(parser)
    return parser


def run_normal1_pipeline(args: argparse.Namespace) -> Path | None:
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
            "step2_skeleton_max_components": args.step2_skeleton_max_components,
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

    if _skip_stage(args.skip_existing, step1_metrics):
        print("[skip] step1 metrics exists")
    else:
        if args.fast and args.task == "coronary_arteries":
            raise SystemExit("TotalSegmentator task 'coronary_arteries' does not support --fast. Remove --fast.")
        if args.dry_run:
            cmd = [
                args.totalseg_cmd,
                "-i",
                str(ct_path),
                "-o",
                str(out_dir / "step1_segmentation" / "totalseg_output"),
                "--task",
                args.task,
            ]
            if args.fast:
                cmd.append("--fast")
            _extend_extra_args(cmd, args.totalseg_arg)
            _print_cmd(cmd)
        else:
            volume = load_volume(ct_path)
            stage1_dir = out_dir / "step1_segmentation"
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
            evaluate_step1_segmentation(pred_mask, gt_mask_path, out_dir)

    pred_mask = out_dir / "step1_segmentation" / "totalseg_output" / args.pred_file
    if not args.dry_run:
        _ensure_path(pred_mask, label="Predicted mask")
    summary["stages"]["step1"] = _read_json(step1_metrics)

    if _skip_stage(args.skip_existing, step2_metrics):
        print("[skip] step2 metrics exists")
    else:
        if args.dry_run:
            cmd = [
                sys.executable,
                "-m",
                "vessel_seg",
                "quant",
                "step2",
                "--seg-mask",
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
            _print_cmd(cmd)
        else:
            evaluate_step2_centerline_from_mask(
                pred_mask,
                gt_centerline_path,
                out_dir,
                thr_mm=args.thr,
                backend=args.step2_backend,
            )

    baseline_centerline = out_dir / "step2_centerline" / "pred_centerline.vtp"
    needs_repair_run = not _skip_stage(args.skip_existing, step3_metrics)
    if args.step2_backend == "vmtk" and baseline_centerline.exists() and not args.dry_run and needs_repair_run:
        stats = _vtp_stats(baseline_centerline)
        summary["stages"]["step2_centerline_vtp"] = stats
        if stats.get("lines") == 0:
            fallback_centerline = out_dir / "step2_centerline" / "pred_centerline_poly.vtp"
            fallback_report = out_dir / "step2_centerline" / "pred_centerline_poly_report.json"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "vmtk_extract_centerlines.py"),
                "--mask",
                str(pred_mask),
                "--out",
                str(fallback_centerline),
                "--report",
                str(fallback_report),
                "--skeleton-max-components",
                str(max(1, int(args.step2_skeleton_max_components))),
            ]
            _run_command(cmd, dry_run=args.dry_run)
            fallback_stats = _vtp_stats(fallback_centerline)
            summary["stages"]["step2_centerline_poly_vtp"] = fallback_stats
            if (fallback_stats.get("lines") or 0) > 0:
                baseline_centerline = fallback_centerline
            else:
                graph_centerline = out_dir / "step2_centerline" / "pred_centerline_graph.vtp"
                graph_report = out_dir / "step2_centerline" / "pred_centerline_graph_report.json"
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "reconstruct_centerline_lines.py"),
                    "--in-vtp",
                    str(baseline_centerline),
                    "--out-vtp",
                    str(graph_centerline),
                    "--report",
                    str(graph_report),
                ]
                _run_command(cmd, dry_run=args.dry_run)
                graph_stats = _vtp_stats(graph_centerline)
                summary["stages"]["step2_centerline_graph_vtp"] = graph_stats
                if (graph_stats.get("lines") or 0) > 0:
                    baseline_centerline = graph_centerline

    if not args.dry_run:
        _ensure_path(baseline_centerline, label="Step2 baseline centerline")
    summary["stages"]["step2"] = _read_json(step2_metrics)

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
            str(ROOT / "scripts" / "repair_centerline.py"),
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
            "--coord_mode",
            str(args.coord_mode),
            "--align_max_points",
            str(args.align_max_points),
            "--align_outside_penalty",
            str(args.align_outside_penalty),
            "--align_steps",
            str(args.align_steps),
            "--densify_step_mm",
            str(args.densify_step_mm),
            "--bridge_min_count",
            str(args.bridge_min_count),
            "--bridge_relax_dist_factor",
            str(args.bridge_relax_dist_factor),
            "--bridge_relax_angle_add",
            str(args.bridge_relax_angle_add),
            "--bridge_relax_len_factor",
            str(args.bridge_relax_len_factor),
            "--bridge_relax_curvature_factor",
            str(args.bridge_relax_curvature_factor),
            "--bridge_relax_murray_add",
            str(args.bridge_relax_murray_add),
            "--bridge_smooth_iterations",
            str(args.bridge_smooth_iterations),
            "--regularize_scope",
            str(args.regularize_scope),
        ]
        if args.no_align_centerline:
            cmd.append("--no_align_centerline")
        if args.no_bridge_relax_if_none:
            cmd.append("--no_bridge_relax_if_none")
        _run_command(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            evaluate_step3_repair(
                baseline_centerline_vtp=baseline_centerline,
                repaired_centerline_vtp=repaired_vtp,
                gt_centerline_vtp=gt_centerline_path,
                out_dir=out_dir,
                thr_mm=args.thr,
            )

    summary["stages"]["step3"] = _read_json(step3_metrics)

    if _skip_stage(args.skip_existing, step4_metrics):
        print("[skip] step4 metrics exists")
    else:
        pred_cmd = [sys.executable, "-m", "vessel_seg.shape", "extract", "--seg", str(pred_mask), "--out", str(pred_features_dir)]
        _extend_extra_args(pred_cmd, args.pred_extract_arg)
        _run_command(pred_cmd, dry_run=args.dry_run)

        gt_cmd = [sys.executable, "-m", "vessel_seg.shape", "extract", "--seg", str(gt_mask_path), "--out", str(gt_features_dir)]
        _extend_extra_args(gt_cmd, args.gt_extract_arg)
        _run_command(gt_cmd, dry_run=args.dry_run)

        if not args.dry_run:
            evaluate_step4_features(pred_features_dir, gt_features_dir, out_dir)

    summary["stages"]["step4"] = _read_json(step4_metrics)

    if _skip_stage(args.skip_existing, step5_metrics):
        print("[skip] step5 metrics exists")
    else:
        if not args.dry_run:
            evaluate_step5_render(
                gt_mask_path=gt_mask_path,
                pred_features_dir=pred_features_dir,
                out_dir=out_dir,
                thr_mm=args.thr,
            )
        else:
            cmd = [
                sys.executable,
                "-m",
                "vessel_seg",
                "quant",
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
            _print_cmd(cmd)

    summary["stages"]["step5"] = _read_json(step5_metrics)
    summary["finished_at_utc"] = _now_utc()

    summary_path = out_dir / "pipeline_run_summary.json"
    if args.dry_run:
        print("[done] dry-run finished; no files were written.")
        return None

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] summary: {summary_path}")
    return summary_path
