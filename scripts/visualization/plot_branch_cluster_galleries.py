#!/usr/bin/env python3
"""Render one large gallery per global branch cluster."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.pipeline.branch_cluster_gallery import (
    BranchClusterGalleryConfig,
    generate_branch_cluster_galleries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one gallery image per branch cluster.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Summary csv produced by generate_case_overviews.py")
    parser.add_argument("--assignments-csv", type=Path, required=True, help="Assignments csv produced by analyze_branch_clusters.py")
    parser.add_argument("--cluster-summary-csv", type=Path, required=True, help="Cluster summary csv produced by analyze_branch_clusters.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store gallery images.")
    parser.add_argument("--cluster-id", type=int, action="append", default=None, help="Optional cluster ids to render. Can be passed multiple times.")
    parser.add_argument("--cols", type=int, default=8, help="Number of case panels per row.")
    parser.add_argument("--panel-width", type=float, default=3.2, help="Width of each panel in inches.")
    parser.add_argument("--panel-height", type=float, default=3.2, help="Height of each panel in inches.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure dpi.")
    parser.add_argument("--elev", type=float, default=18.0, help="3D camera elevation angle.")
    parser.add_argument("--azim", type=float, default=-64.0, help="3D camera azimuth angle.")
    parser.add_argument("--dynamic-order", action="store_true", help="Use hit-first dynamic case ordering instead of fixed cohort/id grid.")
    parser.add_argument("--raw-world-coords", action="store_true", help="Render in raw world coordinates instead of aligned anatomical frame.")
    args = parser.parse_args()

    config = BranchClusterGalleryConfig(
        summary_csv=args.summary_csv if args.summary_csv.is_absolute() else (REPO_ROOT / args.summary_csv).resolve(),
        assignments_csv=args.assignments_csv if args.assignments_csv.is_absolute() else (REPO_ROOT / args.assignments_csv).resolve(),
        cluster_summary_csv=args.cluster_summary_csv if args.cluster_summary_csv.is_absolute() else (REPO_ROOT / args.cluster_summary_csv).resolve(),
        output_dir=args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir).resolve(),
        cluster_ids=tuple(args.cluster_id or ()),
        cols=args.cols,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        fixed_case_grid=not bool(args.dynamic_order),
        align_anatomical_frame=not bool(args.raw_world_coords),
    )
    result = generate_branch_cluster_galleries(config)
    print(f"output_dir: {result.output_dir}")
    print(f"index_csv: {result.index_csv}")
    print(f"image_count: {len(result.image_paths)}")


if __name__ == "__main__":
    main()
