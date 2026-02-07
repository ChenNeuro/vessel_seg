#!/usr/bin/env python3
"""Collect reproducible benchmark artifacts for refactor gating.

This script can:
1. Run a command multiple times and measure wall-time and peak RSS (best effort).
2. Parse a centerline CSV summary (e.g., compare_summary.csv) into aggregate metrics.
3. Emit a single JSON file used by compare_benchmark.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _summary(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    mean_val = statistics.fmean(values)
    std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
    p50 = statistics.median(values)
    ordered = sorted(values)
    p95_idx = int(round(0.95 * (len(ordered) - 1)))
    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "min": float(ordered[0]),
        "p50": float(p50),
        "p95": float(ordered[p95_idx]),
        "max": float(ordered[-1]),
    }


def _read_proc_rss_mb(pid: int) -> Optional[float]:
    status_path = Path("/proc") / str(pid) / "status"
    if not status_path.exists():
        return None
    try:
        text = status_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    kb = float(parts[1])
                    return kb / 1024.0
                except ValueError:
                    return None
    return None


def _run_single_command(
    command: List[str],
    cwd: Path,
    env: Optional[Dict[str, str]],
    sample_interval: float,
) -> Dict[str, Any]:
    start_utc = _now_utc()
    t0 = time.perf_counter()

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    peak_rss_mb: Optional[float] = None
    while proc.poll() is None:
        rss_mb = _read_proc_rss_mb(proc.pid)
        if rss_mb is not None:
            if peak_rss_mb is None:
                peak_rss_mb = rss_mb
            else:
                peak_rss_mb = max(peak_rss_mb, rss_mb)
        time.sleep(sample_interval)

    stdout, stderr = proc.communicate()
    wall_time = time.perf_counter() - t0
    end_utc = _now_utc()

    result: Dict[str, Any] = {
        "command": command,
        "cwd": str(cwd),
        "start_utc": start_utc,
        "end_utc": end_utc,
        "returncode": int(proc.returncode),
        "wall_time_sec": float(wall_time),
        "peak_rss_mb": None if peak_rss_mb is None else float(peak_rss_mb),
    }
    if proc.returncode != 0:
        result["stdout_tail"] = stdout[-4000:]
        result["stderr_tail"] = stderr[-4000:]
    return result


def _git_value(repo_root: Path, args: List[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    return text if text else None


def _collect_csv_metrics(csv_path: Path) -> Dict[str, Any]:
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows_raw = list(reader)
    if not rows_raw:
        return {"source_csv": str(csv_path), "case_count": 0, "aggregates": {}, "per_case": []}

    numeric_cols: List[str] = []
    for key in rows_raw[0].keys():
        if key == "case":
            continue
        col_values = [_parse_float(row.get(key)) for row in rows_raw]
        if any(v is not None for v in col_values):
            numeric_cols.append(key)

    per_case: List[Dict[str, Any]] = []
    for row in rows_raw:
        parsed: Dict[str, Any] = {"case": row.get("case")}
        for col in numeric_cols:
            parsed[col] = _parse_float(row.get(col))
        per_case.append(parsed)

    aggregates: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        values = [item[col] for item in per_case if item[col] is not None]
        stats = _summary([float(v) for v in values])
        if stats is not None:
            aggregates[col] = stats

    worst_cases: Dict[str, List[Dict[str, Any]]] = {}
    if "pred2gt_mean" in numeric_cols:
        sorted_cases = sorted(
            [row for row in per_case if row["pred2gt_mean"] is not None],
            key=lambda x: x["pred2gt_mean"],
            reverse=True,
        )[:5]
        worst_cases["pred2gt_mean_top5"] = sorted_cases
    if "coverage_pred@1mm" in numeric_cols:
        sorted_cases = sorted(
            [row for row in per_case if row["coverage_pred@1mm"] is not None],
            key=lambda x: x["coverage_pred@1mm"],
        )[:5]
        worst_cases["coverage_pred@1mm_bottom5"] = sorted_cases

    return {
        "source_csv": str(csv_path),
        "case_count": len(per_case),
        "aggregates": aggregates,
        "per_case": per_case,
        "worst_cases": worst_cases,
    }


def _parse_env_items(items: List[str]) -> Dict[str, str]:
    env_map: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env item: {item!r}. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env item: {item!r}. KEY cannot be empty.")
        env_map[key] = value
    return env_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect benchmark JSON for refactor gating.")
    parser.add_argument("--name", required=True, help="Benchmark artifact name, e.g. baseline_normal20.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--centerline-csv",
        type=Path,
        default=None,
        help="Optional CSV summary (e.g. outputs/vmtk_centerlines/compare_summary.csv).",
    )
    parser.add_argument(
        "--cmd",
        type=str,
        default=None,
        help="Optional command to benchmark. Example: \"python scripts/run_pipeline.py ...\"",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of benchmark repeats.")
    parser.add_argument("--cwd", type=Path, default=Path("."), help="Working directory for command execution.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment variable KEY=VALUE. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.05,
        help="Memory sampling interval in seconds while command is running.",
    )
    parser.add_argument(
        "--allow-nonzero",
        action="store_true",
        help="Do not fail the collector if benchmark command returns non-zero.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.sample_interval <= 0:
        raise SystemExit("--sample-interval must be > 0")

    cwd = args.cwd.expanduser().resolve()
    output = args.output.expanduser().resolve()

    env = dict(os.environ)
    env.update(_parse_env_items(args.env))

    command: Optional[List[str]] = shlex.split(args.cmd) if args.cmd else None

    runs: List[Dict[str, Any]] = []
    if command is not None:
        for idx in range(args.repeat):
            run = _run_single_command(command, cwd=cwd, env=env, sample_interval=args.sample_interval)
            run["index"] = idx + 1
            runs.append(run)
            if run["returncode"] != 0 and not args.allow_nonzero:
                break

    wall_values = [run["wall_time_sec"] for run in runs if run.get("wall_time_sec") is not None]
    mem_values = [run["peak_rss_mb"] for run in runs if run.get("peak_rss_mb") is not None]
    runtime_summary = {
        "wall_time_sec": _summary([float(v) for v in wall_values]),
        "peak_rss_mb": _summary([float(v) for v in mem_values]),
    }

    centerline_summary = None
    if args.centerline_csv is not None:
        centerline_csv = args.centerline_csv.expanduser().resolve()
        if not centerline_csv.exists():
            raise SystemExit(f"--centerline-csv not found: {centerline_csv}")
        centerline_summary = _collect_csv_metrics(centerline_csv)

    repo_root = cwd
    git_commit = _git_value(repo_root, ["rev-parse", "HEAD"])
    git_dirty_raw = _git_value(repo_root, ["status", "--porcelain"])
    git_dirty = bool(git_dirty_raw) if git_dirty_raw is not None else None

    payload = {
        "schema_version": "1.0",
        "name": args.name,
        "created_at_utc": _now_utc(),
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "hostname": platform.node(),
        },
        "git": {
            "commit": git_commit,
            "dirty": git_dirty,
        },
        "command_config": {
            "cmd": command,
            "cwd": str(cwd),
            "repeat": args.repeat,
            "sample_interval_sec": args.sample_interval,
        },
        "runs": runs,
        "runtime_summary": runtime_summary,
        "centerline_summary": centerline_summary,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    nonzero = [run for run in runs if run.get("returncode", 0) != 0]
    if nonzero and not args.allow_nonzero:
        raise SystemExit(
            f"Benchmark command failed on {len(nonzero)} run(s). "
            f"JSON still written to: {output}"
        )

    print(f"Wrote benchmark artifact: {output}")


if __name__ == "__main__":
    main()
