#!/usr/bin/env python3
"""Run the standardized five-stage pipeline for one case."""

from __future__ import annotations

import argparse
from pathlib import Path

from vessel_seg.config import ProjectPaths
from vessel_seg.pipeline import CaseInputs, PipelineConfig, run_case_pipeline
from vessel_seg.pipeline.stages import (
    CenterlineExtractionStageConfig,
    CenterlineRepairStageConfig,
    CtSegmentationStageConfig,
    RenderingStageConfig,
    WallFeatureStageConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the rebuilt five-stage coronary pipeline.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--centerline-vtp", type=Path, default=None)
    parser.add_argument("--probability-map", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs_reorganized"))
    parser.add_argument("--seg-backend", choices=["existing_mask", "totalseg", "dummy"], default="existing_mask")
    parser.add_argument("--centerline-backend", choices=["mask_skeleton", "vtp_copy"], default="mask_skeleton")
    parser.add_argument("--repair-mode", choices=["topology_only", "probability_bridge"], default="topology_only")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_paths = ProjectPaths.from_root(Path(__file__).resolve().parents[2])
    case_inputs = CaseInputs(
        case_id=args.case_id,
        ct_path=args.ct.resolve(),
        mask_path=None if args.mask is None else args.mask.resolve(),
        centerline_vtp_path=None if args.centerline_vtp is None else args.centerline_vtp.resolve(),
        probability_map_path=None if args.probability_map is None else args.probability_map.resolve(),
    )
    pipeline_config = PipelineConfig(
        output_root=(project_paths.root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root,
        segmentation_backend=args.seg_backend,
        centerline_backend=args.centerline_backend,
        repair_mode=args.repair_mode,
        dry_run=args.dry_run,
    )
    summary = run_case_pipeline(
        case_inputs,
        project_paths=project_paths,
        pipeline_config=pipeline_config,
        ct_config=CtSegmentationStageConfig(backend=args.seg_backend),
        extraction_config=CenterlineExtractionStageConfig(backend=args.centerline_backend),
        repair_config=CenterlineRepairStageConfig(mode=args.repair_mode),
        wall_config=WallFeatureStageConfig(),
        rendering_config=RenderingStageConfig(),
    )
    print(summary.output_root / "cases" / args.case_id / "pipeline_summary.json")


if __name__ == "__main__":
    main()
