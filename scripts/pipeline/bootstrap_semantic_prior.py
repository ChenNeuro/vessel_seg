#!/usr/bin/env python3
"""Fit semantic geometry priors from existing stage3 trees."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.config import ProjectPaths
from vessel_seg.pipeline.batch_overview import BatchOverviewConfig, discover_asoca_cases
from vessel_seg.pipeline.semantic_topology import fit_semantic_geometry_prior_from_tree_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit semantic geometry prior from existing stage3 tree outputs.")
    parser.add_argument(
        "--cohort",
        action="append",
        choices=["Normal", "Diseased"],
        help="Cohort to include. Repeatable. Default: Normal.",
    )
    parser.add_argument("--case-id", action="append", default=None, help="Specific case id to include. Repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of cases.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs_reorganized_runs"), help="Pipeline output root.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for semantic prior json.",
    )
    parser.add_argument(
        "--min-samples-per-family",
        type=int,
        default=3,
        help="Minimum sample count required for each family prior.",
    )
    args = parser.parse_args()

    project_paths = ProjectPaths.from_root(REPO_ROOT)
    config = BatchOverviewConfig(
        cohorts=tuple(args.cohort or ["Normal"]),
        case_ids=tuple(args.case_id or []),
        output_root=args.output_root,
        limit=args.limit,
    )
    records = discover_asoca_cases(project_paths, config)
    output_root = (project_paths.root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    tree_paths = []
    for record in records:
        tree_path = output_root / "cases" / record.case_id / "stages" / "03_centerline_repair" / "tree.json"
        if tree_path.exists():
            tree_paths.append(tree_path)

    prior = fit_semantic_geometry_prior_from_tree_paths(
        tree_paths,
        min_samples_per_family=args.min_samples_per_family,
    )
    out_path = args.out if args.out.is_absolute() else (project_paths.root / args.out).resolve()
    prior.save(out_path)

    print(f"prior_json: {out_path}")
    print(f"tree_count: {len(tree_paths)}")
    print(f"family_count: {len(prior.families)}")
    for family in prior.families:
        print(f"{family.family}: samples={family.sample_count}")


if __name__ == "__main__":
    main()
