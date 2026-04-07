#!/usr/bin/env python3
"""Batch-generate per-case overview images and a gallery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.config import ProjectPaths
from vessel_seg.pipeline.batch_overview import BatchOverviewConfig, run_batch_overview


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate overview.png for multiple ASOCA cases.")
    parser.add_argument(
        "--cohort",
        action="append",
        choices=["Normal", "Diseased"],
        help="Cohort to include. Repeatable. Default: Normal.",
    )
    parser.add_argument("--case-id", action="append", default=None, help="Specific case id to include. Repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of cases.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs_reorganized_runs"), help="Pipeline output root.")
    parser.add_argument("--report-tag", type=str, default="overview_batch", help="Subdirectory name for summary/gallery.")
    parser.add_argument("--gallery-cols", type=int, default=4, help="Number of columns in gallery.")
    parser.add_argument("--force", action="store_true", help="Re-run cases even if overview.png already exists.")
    parser.add_argument(
        "--semantic-prior",
        type=Path,
        default=None,
        help="Optional semantic geometry prior JSON used by stage3 semantic topology.",
    )
    args = parser.parse_args()

    project_paths = ProjectPaths.from_root(REPO_ROOT)
    config = BatchOverviewConfig(
        cohorts=tuple(args.cohort or ["Normal"]),
        case_ids=tuple(args.case_id or []),
        output_root=args.output_root,
        report_tag=args.report_tag,
        skip_existing=not args.force,
        limit=args.limit,
        gallery_cols=args.gallery_cols,
        semantic_prior_path=args.semantic_prior,
    )
    result = run_batch_overview(project_paths, config)

    print(f"summary_csv: {result.summary_csv}")
    print(f"gallery_png: {result.gallery_png}")
    print(f"failures_json: {result.failures_json}")
    print(f"cases_ok: {len(result.rows)}")
    print(f"cases_failed: {len(result.failures)}")


if __name__ == "__main__":
    main()
