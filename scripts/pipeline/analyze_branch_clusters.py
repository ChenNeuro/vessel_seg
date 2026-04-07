#!/usr/bin/env python3
"""Analyze global branch clusters from a batch summary csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.pipeline.branch_clustering import analyze_branch_clusters


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze global unlabeled branch clusters for a batch overview summary.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Summary csv produced by generate_case_overviews.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for report json and csv outputs.")
    parser.add_argument("--clustering-mode", choices=["side_first", "global"], default="side_first", help="Cluster all branches together or first partition by left/right systems.")
    parser.add_argument("--side-assignments-csv", type=Path, default=None, help="Optional branch assignments csv with case_id/branch_id/side_group. When provided, side grouping uses this file instead of internal root membership.")
    parser.add_argument("--similarity-threshold", type=float, default=0.80, help="Average-linkage similarity threshold used by agglomerative clustering.")
    parser.add_argument("--target-cluster-count", type=int, default=None, help="Optional target number of clusters. When set, clustering keeps merging until this count is reached.")
    parser.add_argument("--match-max-case-branches", action="store_true", help="Set target cluster count to the maximum num_branches found in summary csv.")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Clusters smaller than this are marked as outlier clusters.")
    parser.add_argument("--outlier-threshold", type=float, default=0.70, help="Prototype similarity threshold used to flag outlier samples.")
    args = parser.parse_args()

    report = analyze_branch_clusters(
        args.summary_csv if args.summary_csv.is_absolute() else (REPO_ROOT / args.summary_csv).resolve(),
        output_dir=None if args.output_dir is None else (args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir).resolve()),
        clustering_mode=args.clustering_mode,
        side_assignments_csv=None if args.side_assignments_csv is None else (args.side_assignments_csv if args.side_assignments_csv.is_absolute() else (REPO_ROOT / args.side_assignments_csv).resolve()),
        similarity_threshold=args.similarity_threshold,
        target_cluster_count=args.target_cluster_count,
        match_max_case_branches=args.match_max_case_branches,
        min_cluster_size=args.min_cluster_size,
        outlier_similarity_threshold=args.outlier_threshold,
    )

    print(f"report_json: {report.report_json}")
    print(f"cluster_summary_csv: {report.cluster_summary_csv}")
    print(f"assignments_csv: {report.assignments_csv}")
    print(f"outliers_csv: {report.outliers_csv}")
    print(f"total_cases: {report.total_cases}")
    print(f"total_samples: {report.total_samples}")
    print(f"clustering_mode: {report.clustering_mode}")
    print(f"side_assignments_csv: {report.side_assignments_csv}")
    print(f"side_override_count: {report.side_override_count}")
    print(f"cluster_count: {report.cluster_count}")
    print(f"target_cluster_count: {report.target_cluster_count}")
    print(f"mean_assignment_similarity_to_prototype: {report.mean_assignment_similarity_to_prototype}")


if __name__ == "__main__":
    main()
