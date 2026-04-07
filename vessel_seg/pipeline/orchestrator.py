"""Orchestrator for the standardized five-stage coronary pipeline."""

from __future__ import annotations

from pathlib import Path
import json

from vessel_seg.config import ProjectPaths

from .contracts import CaseInputs, PipelineConfig, PipelineRunSummary
from .layout import build_pipeline_layout
from .stages import (
    CenterlineExtractionStageConfig,
    CenterlineRepairStageConfig,
    CtSegmentationStageConfig,
    RenderingStageConfig,
    WallFeatureStageConfig,
    run_centerline_extraction_stage,
    run_centerline_repair_stage,
    run_ct_segmentation_stage,
    run_rendering_stage,
    run_wall_feature_stage,
)


def run_case_pipeline(
    case_inputs: CaseInputs,
    *,
    project_paths: ProjectPaths,
    pipeline_config: PipelineConfig | None = None,
    ct_config: CtSegmentationStageConfig | None = None,
    extraction_config: CenterlineExtractionStageConfig | None = None,
    repair_config: CenterlineRepairStageConfig | None = None,
    wall_config: WallFeatureStageConfig | None = None,
    rendering_config: RenderingStageConfig | None = None,
) -> PipelineRunSummary:
    pipeline_config = pipeline_config or PipelineConfig(output_root=project_paths.outputs_dir)
    ct_config = ct_config or CtSegmentationStageConfig(backend=pipeline_config.segmentation_backend)
    extraction_config = extraction_config or CenterlineExtractionStageConfig(backend=pipeline_config.centerline_backend)
    repair_config = repair_config or CenterlineRepairStageConfig(mode=pipeline_config.repair_mode)
    wall_config = wall_config or WallFeatureStageConfig()
    rendering_config = rendering_config or RenderingStageConfig()

    layout = build_pipeline_layout(pipeline_config.output_root, case_inputs.case_id, project_paths.root)
    layout.ensure()
    summary = PipelineRunSummary(case_id=case_inputs.case_id, output_root=pipeline_config.output_root, inputs=case_inputs)

    segmentation_artifact = run_ct_segmentation_stage(case_inputs, layout, pipeline_config, ct_config)
    summary.stages.append(segmentation_artifact)

    extraction_artifact = run_centerline_extraction_stage(case_inputs, layout, segmentation_artifact, extraction_config)
    summary.stages.append(extraction_artifact)

    repair_artifact = run_centerline_repair_stage(case_inputs, layout, extraction_artifact, pipeline_config, repair_config)
    summary.stages.append(repair_artifact)

    wall_artifact = run_wall_feature_stage(case_inputs, layout, segmentation_artifact, repair_artifact, pipeline_config, wall_config)
    summary.stages.append(wall_artifact)

    rendering_artifact = run_rendering_stage(case_inputs, layout, repair_artifact, wall_artifact, rendering_config)
    summary.stages.append(rendering_artifact)

    summary_path = layout.case_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
