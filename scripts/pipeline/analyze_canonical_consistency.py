#!/usr/bin/env python3
"""Analyze canonical naming consistency from a batch summary csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.pipeline.canonical_consistency import analyze_canonical_consistency


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze canonical naming consistency for a batch overview summary.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Summary csv produced by generate_case_overviews.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for report json and csv outputs.")
    parser.add_argument("--min-case-count", type=int, default=2, help="Minimum case count required for a canonical label to be analyzed.")
    parser.add_argument("--similarity-threshold", type=float, default=0.70, help="Overall prototype similarity threshold for case anomalies.")
    parser.add_argument("--shape-threshold", type=float, default=0.50, help="Shape similarity threshold for case anomalies.")
    args = parser.parse_args()

    report = analyze_canonical_consistency(
        args.summary_csv if args.summary_csv.is_absolute() else (REPO_ROOT / args.summary_csv).resolve(),
        output_dir=None if args.output_dir is None else (args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir).resolve()),
        min_case_count=args.min_case_count,
        anomaly_similarity_threshold=args.similarity_threshold,
        anomaly_shape_threshold=args.shape_threshold,
    )

    print(f"report_json: {report.report_json}")
    print(f"label_stats_csv: {report.label_stats_csv}")
    print(f"anomalies_csv: {report.anomalies_csv}")
    print(f"total_cases: {report.total_cases}")
    print(f"total_labels: {report.total_labels}")
    print(f"mean_pairwise_similarity: {report.mean_pairwise_similarity}")
    print(f"mean_shape_similarity: {report.mean_shape_similarity}")
    print(f"anomaly_cases: {len({item.case_id for item in report.anomalies})}")


if __name__ == "__main__":
    main()
