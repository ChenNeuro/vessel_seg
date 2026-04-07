#!/usr/bin/env python3
"""Analyze semantic topology consistency from a batch summary csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.pipeline.semantic_consistency import analyze_semantic_consistency


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze semantic consistency for a batch overview summary.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Summary csv produced by generate_case_overviews.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for report json and anomaly csv.")
    parser.add_argument("--score-threshold", type=float, default=75.0, help="Semantic score threshold for anomalies.")
    parser.add_argument("--branch-threshold", type=int, default=12, help="Branch count threshold for anomalies.")
    args = parser.parse_args()

    report = analyze_semantic_consistency(
        args.summary_csv if args.summary_csv.is_absolute() else (REPO_ROOT / args.summary_csv).resolve(),
        output_dir=None if args.output_dir is None else (args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir).resolve()),
        anomaly_score_threshold=args.score_threshold,
        anomaly_branch_threshold=args.branch_threshold,
    )

    print(f"report_json: {report.report_json}")
    print(f"anomalies_csv: {report.anomalies_csv}")
    print(f"total_cases: {report.total_cases}")
    print(f"mean_semantic_score: {report.mean_semantic_score}")
    print(f"mean_semantic_jaccard: {report.mean_semantic_jaccard}")
    print(f"anomaly_cases: {len(report.anomalies)}")


if __name__ == "__main__":
    main()
