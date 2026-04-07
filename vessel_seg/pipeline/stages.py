"""Five standardized pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import shutil
import subprocess
import sys

import numpy as np

from vessel_seg.centerline import extract_centerlines_from_mask
from vessel_seg.io import load_mask, load_volume, save_nifti
from vessel_seg.pipeline.naming import apply_canonical_naming
from vessel_seg.pipeline.semantic_topology import apply_semantic_topology
from vessel_seg.quant.metrics import segmentation_consistency_metrics, segmentation_metrics
from vessel_seg.segmentation_interface import (
    DummySegmentationBackend,
    TotalSegmentationBackend,
    run_segmentation,
)

from .contracts import CaseInputs, PipelineConfig, StageArtifact
from .layout import PipelineLayout, write_stage_manifest


@dataclass
class CtSegmentationStageConfig:
    backend: str = "existing_mask"
    min_component_size: int | None = None
    totalseg_command: str = "TotalSegmentator"
    task: str | None = None
    fast: bool = False


@dataclass
class CenterlineExtractionStageConfig:
    backend: str = "mask_skeleton"


@dataclass
class CenterlineRepairStageConfig:
    mode: str = "topology_only"
    distance_threshold_mm: float = 3.0
    semantic_prior_path: Path | None = None


@dataclass
class WallFeatureStageConfig:
    k_samples: int = 64
    angle_bins: int = 32
    start_offset_mm: float = 3.0


@dataclass
class RenderingStageConfig:
    figure_name: str = "overview.png"


def _run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    if dry_run:
        return
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Legacy command failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout: {completed.stdout[-2000:]}\n"
            f"stderr: {completed.stderr[-2000:]}"
        )


def _write_centerline_vtp(branches: list[np.ndarray], out_path: Path) -> None:
    try:
        import vtk
    except Exception as exc:  # pragma: no cover
        raise ImportError("vtk is required to write VTP files.") from exc

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    point_offset = 0
    for branch in branches:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(branch.shape[0])
        for idx, point in enumerate(branch):
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            line.GetPointIds().SetId(idx, point_offset + idx)
        point_offset += branch.shape[0]
        lines.InsertNextCell(line)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(poly)
    writer.Write()


def _write_centerline_manifest(branches_dir: Path, branches: list[np.ndarray]) -> Path:
    branches_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for branch_id, branch in enumerate(branches):
        branch_path = branches_dir / f"branch_{branch_id}.npy"
        np.save(branch_path, branch.astype(np.float32))
        records.append(
            {
                "branch_id": branch_id,
                "num_points": int(branch.shape[0]),
                "length_mm": float(np.linalg.norm(np.diff(branch, axis=0), axis=1).sum()) if branch.shape[0] > 1 else 0.0,
                "centerline_path": str(branch_path),
            }
        )
    manifest_path = branches_dir / "branches.json"
    manifest_path.write_text(json.dumps({"branches": records}, indent=2), encoding="utf-8")
    return manifest_path


def run_ct_segmentation_stage(
    case_inputs: CaseInputs,
    layout: PipelineLayout,
    pipeline_config: PipelineConfig,
    stage_config: CtSegmentationStageConfig,
) -> StageArtifact:
    stage_dir = layout.stage_dir("ct_segmentation")
    stage_dir.mkdir(parents=True, exist_ok=True)
    mask_path = stage_dir / "mask.nii.gz"

    if pipeline_config.dry_run:
        artifact = StageArtifact(
            stage_name="ct_segmentation",
            stage_dir=stage_dir,
            manifest_path=stage_dir / "manifest.json",
            primary_path=mask_path,
            metadata={"backend": stage_config.backend, "dry_run": True},
        )
        write_stage_manifest(artifact)
        return artifact

    if case_inputs.mask_path is not None and stage_config.backend == "existing_mask":
        shutil.copy2(case_inputs.mask_path, mask_path)
        metadata = {"backend": "existing_mask", "source_mask": str(case_inputs.mask_path)}
    else:
        volume = load_volume(case_inputs.ct_path)
        if stage_config.backend == "dummy":
            backend = DummySegmentationBackend()
        elif stage_config.backend == "totalseg":
            backend = TotalSegmentationBackend(
                output_dir=stage_dir / "totalseg_output",
                command=stage_config.totalseg_command,
                task=stage_config.task,
                fast=stage_config.fast,
            )
        else:
            raise ValueError(f"Unsupported segmentation backend: {stage_config.backend}")
        mask = run_segmentation(
            volume,
            backend,
            postprocess=stage_config.min_component_size is not None,
            min_size=stage_config.min_component_size,
        )
        save_nifti(mask.astype(np.uint8), {"affine": volume.affine}, mask_path)
        metadata = {"backend": stage_config.backend}

    pred_mask = load_mask(mask_path)
    metadata["consistency"] = segmentation_consistency_metrics(pred_mask.data)
    if case_inputs.mask_path is not None:
        gt_mask = load_mask(case_inputs.mask_path)
        metadata["evaluation"] = segmentation_metrics(
            pred_mask.data,
            gt_mask.data,
            pred_mask.spacing,
        )

    artifact = StageArtifact(
        stage_name="ct_segmentation",
        stage_dir=stage_dir,
        manifest_path=stage_dir / "manifest.json",
        primary_path=mask_path,
        metadata=metadata,
    )
    write_stage_manifest(artifact)
    return artifact


def run_centerline_extraction_stage(
    case_inputs: CaseInputs,
    layout: PipelineLayout,
    extraction_artifact: StageArtifact,
    stage_config: CenterlineExtractionStageConfig,
) -> StageArtifact:
    stage_dir = layout.stage_dir("centerline_extraction")
    stage_dir.mkdir(parents=True, exist_ok=True)
    vtp_path = stage_dir / "centerlines.vtp"
    branches_dir = stage_dir / "branches"

    if extraction_artifact.metadata.get("dry_run"):
        artifact = StageArtifact(
            stage_name="centerline_extraction",
            stage_dir=stage_dir,
            manifest_path=stage_dir / "manifest.json",
            primary_path=vtp_path,
            secondary_paths={"branches_dir": branches_dir},
            metadata={"backend": stage_config.backend, "dry_run": True},
        )
        write_stage_manifest(artifact)
        return artifact

    if case_inputs.centerline_vtp_path is not None and stage_config.backend == "vtp_copy":
        shutil.copy2(case_inputs.centerline_vtp_path, vtp_path)
        metadata = {"backend": "vtp_copy", "source_vtp": str(case_inputs.centerline_vtp_path)}
    else:
        mask_volume = load_mask(extraction_artifact.primary_path)
        tree = extract_centerlines_from_mask(mask_volume.data, mask_volume.spacing)
        branches = [np.asarray(branch.centerline, dtype=np.float32) for branch in tree.iter_branches()]
        _write_centerline_vtp(branches, vtp_path)
        branch_manifest_path = _write_centerline_manifest(branches_dir, branches)
        metadata = {
            "backend": stage_config.backend,
            "num_branches": len(branches),
            "branch_manifest": str(branch_manifest_path),
        }

    artifact = StageArtifact(
        stage_name="centerline_extraction",
        stage_dir=stage_dir,
        manifest_path=stage_dir / "manifest.json",
        primary_path=vtp_path,
        secondary_paths={"branches_dir": branches_dir},
        metadata=metadata,
    )
    write_stage_manifest(artifact)
    return artifact


def run_centerline_repair_stage(
    case_inputs: CaseInputs,
    layout: PipelineLayout,
    extraction_artifact: StageArtifact,
    pipeline_config: PipelineConfig,
    stage_config: CenterlineRepairStageConfig,
) -> StageArtifact:
    stage_dir = layout.stage_dir("centerline_repair")
    stage_dir.mkdir(parents=True, exist_ok=True)
    repaired_vtp = stage_dir / "centerline_repaired.vtp"
    tree_json = stage_dir / "tree.json"
    branch_names_json = stage_dir / "branch_names.json"
    semantic_topology_json = stage_dir / "semantic_topology.json"

    if pipeline_config.dry_run:
        artifact = StageArtifact(
            stage_name="centerline_repair",
            stage_dir=stage_dir,
            manifest_path=stage_dir / "manifest.json",
            primary_path=repaired_vtp,
            secondary_paths={
                "tree_json": tree_json,
                "branch_names_json": branch_names_json,
                "semantic_topology_json": semantic_topology_json,
            },
            metadata={"mode": stage_config.mode, "dry_run": True},
        )
        write_stage_manifest(artifact)
        return artifact

    source_vtp = extraction_artifact.primary_path
    if stage_config.mode == "probability_bridge":
        if case_inputs.probability_map_path is None:
            raise ValueError("probability_bridge mode requires probability_map_path.")
        command = [
            sys.executable,
            str(layout.repo_root / "scripts" / "repair_centerline.py"),
            "--prob",
            str(case_inputs.probability_map_path),
            "--vtp",
            str(source_vtp),
            "--out",
            str(repaired_vtp),
        ]
        _run_command(command, cwd=layout.repo_root, dry_run=pipeline_config.dry_run)
    else:
        if not pipeline_config.dry_run:
            shutil.copy2(source_vtp, repaired_vtp)

    build_tree_command = [
        sys.executable,
        str(layout.repo_root / "scripts" / "build_centerline_tree.py"),
        "--vtp",
        str(repaired_vtp),
        "--out",
        str(tree_json),
        "--case",
        case_inputs.case_id,
    ]
    _run_command(build_tree_command, cwd=layout.repo_root, dry_run=pipeline_config.dry_run)
    naming_result = apply_canonical_naming(tree_json, branch_names_json)
    semantic_result = apply_semantic_topology(
        tree_json,
        semantic_topology_json,
        geometry_prior_path=stage_config.semantic_prior_path,
    )

    artifact = StageArtifact(
        stage_name="centerline_repair",
        stage_dir=stage_dir,
        manifest_path=stage_dir / "manifest.json",
        primary_path=repaired_vtp,
        secondary_paths={
            "tree_json": tree_json,
            "branch_names_json": branch_names_json,
            "semantic_topology_json": semantic_topology_json,
        },
        metadata={
            "mode": stage_config.mode,
            "naming_version": naming_result.version,
            "num_named_branches": len(naming_result.branches),
            "systems": [system.system_name for system in naming_result.systems],
            "semantic_version": semantic_result.version,
            "dominance": semantic_result.dominance,
            "semantic_consistency_score": semantic_result.consistency.score,
            "semantic_prior_path": None if stage_config.semantic_prior_path is None else str(stage_config.semantic_prior_path),
        },
    )
    write_stage_manifest(artifact)
    return artifact


def run_wall_feature_stage(
    case_inputs: CaseInputs,
    layout: PipelineLayout,
    segmentation_artifact: StageArtifact,
    repair_artifact: StageArtifact,
    pipeline_config: PipelineConfig,
    stage_config: WallFeatureStageConfig,
) -> StageArtifact:
    stage_dir = layout.stage_dir("wall_features")
    stage_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = stage_dir / "branch_dataset.npz"
    branches_dir = stage_dir / "branches"

    if pipeline_config.dry_run:
        artifact = StageArtifact(
            stage_name="wall_features",
            stage_dir=stage_dir,
            manifest_path=stage_dir / "manifest.json",
            primary_path=dataset_path,
            secondary_paths={"branches_dir": branches_dir},
            metadata={"k_samples": stage_config.k_samples, "angle_bins": stage_config.angle_bins, "dry_run": True},
        )
        write_stage_manifest(artifact)
        return artifact

    command = [
        sys.executable,
        str(layout.repo_root / "scripts" / "build_branch_dataset.py"),
        "--vtp",
        str(repair_artifact.primary_path),
        "--mask",
        str(segmentation_artifact.primary_path),
        "--tree",
        str(repair_artifact.secondary_paths["tree_json"]),
        "--out",
        str(dataset_path),
        "--branch_dir",
        str(branches_dir),
        "--case",
        case_inputs.case_id,
        "--K",
        str(stage_config.k_samples),
        "--M",
        str(stage_config.angle_bins),
        "--start_offset",
        str(stage_config.start_offset_mm),
    ]
    _run_command(command, cwd=layout.repo_root, dry_run=pipeline_config.dry_run)

    metadata: dict[str, object] = {"k_samples": stage_config.k_samples, "angle_bins": stage_config.angle_bins}
    if dataset_path.exists():
        data = np.load(dataset_path, allow_pickle=True)
        radii = np.asarray(data["radii"])
        metadata["num_branches"] = int(radii.shape[0]) if radii.ndim == 3 else 0
        metadata["mean_radius_mm"] = float(radii.mean()) if radii.size else 0.0

    artifact = StageArtifact(
        stage_name="wall_features",
        stage_dir=stage_dir,
        manifest_path=stage_dir / "manifest.json",
        primary_path=dataset_path,
        secondary_paths={"branches_dir": branches_dir},
        metadata=metadata,
    )
    write_stage_manifest(artifact)
    return artifact


def run_rendering_stage(
    case_inputs: CaseInputs,
    layout: PipelineLayout,
    repair_artifact: StageArtifact,
    wall_artifact: StageArtifact,
    stage_config: RenderingStageConfig,
) -> StageArtifact:
    import matplotlib.pyplot as plt
    try:
        import vtk
    except Exception as exc:  # pragma: no cover
        raise ImportError("vtk is required for the rendering stage.") from exc

    stage_dir = layout.stage_dir("rendering")
    stage_dir.mkdir(parents=True, exist_ok=True)
    figure_path = stage_dir / stage_config.figure_name
    metrics_csv = stage_dir / "metrics.csv"

    if wall_artifact.metadata.get("dry_run"):
        artifact = StageArtifact(
            stage_name="rendering",
            stage_dir=stage_dir,
            manifest_path=stage_dir / "manifest.json",
            primary_path=figure_path,
            secondary_paths={"metrics_csv": metrics_csv},
            metadata={"dry_run": True},
        )
        write_stage_manifest(artifact)
        return artifact

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(repair_artifact.primary_path))
    reader.Update()
    poly = reader.GetOutput()
    points = poly.GetPoints()
    lines = poly.GetLines()
    branch_polylines: list[np.ndarray] = []
    lines.InitTraversal()
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        coords = np.asarray([points.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=np.float32)
        if coords.shape[0] >= 2:
            branch_polylines.append(coords)

    branch_lengths = [
        float(np.linalg.norm(np.diff(branch, axis=0), axis=1).sum())
        for branch in branch_polylines
    ]
    mean_length = float(np.mean(branch_lengths)) if branch_lengths else 0.0
    total_length = float(np.sum(branch_lengths)) if branch_lengths else 0.0
    mean_radius = 0.0
    if wall_artifact.primary_path is not None and wall_artifact.primary_path.exists():
        radii_data = np.load(wall_artifact.primary_path, allow_pickle=True)
        mean_radius = float(np.asarray(radii_data["radii"]).mean())

    branch_name_map: dict[int, str] = {}
    semantic_name_map: dict[int, str] = {}
    tree_json_path = repair_artifact.secondary_paths.get("tree_json")
    if tree_json_path is not None and tree_json_path.exists():
        tree_payload = json.loads(tree_json_path.read_text(encoding="utf-8"))
        for branch_meta in tree_payload.get("branches", []):
            branch_id = int(branch_meta["branch_id"])
            naming = branch_meta.get("naming") or {}
            semantic = branch_meta.get("semantic") or {}
            branch_name_map[branch_id] = str(naming.get("canonical_name") or f"branch_{branch_id}")
            semantic_name_map[branch_id] = str(semantic.get("semantic_name") or semantic.get("semantic_family") or "")

    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["case_id", case_inputs.case_id])
        writer.writerow(["num_branches", len(branch_polylines)])
        writer.writerow(["total_length_mm", total_length])
        writer.writerow(["mean_branch_length_mm", mean_length])
        writer.writerow(["mean_radius_mm", mean_radius])

    fig = plt.figure(figsize=(12, 6), dpi=180)
    ax_tree = fig.add_subplot(121, projection="3d")
    cmap = plt.get_cmap("tab20")
    for branch_id, branch in enumerate(branch_polylines):
        color = cmap(branch_id % 20)
        ax_tree.plot(branch[:, 0], branch[:, 1], branch[:, 2], color=color, linewidth=2.0)
        semantic_label = semantic_name_map.get(branch_id, "")
        canonical_label = branch_name_map.get(branch_id, f"branch_{branch_id}")
        label = semantic_label if semantic_label else canonical_label
        anchor = branch[len(branch) // 2]
        ax_tree.text(anchor[0], anchor[1], anchor[2], label, color=color, fontsize=7)
    ax_tree.set_title(f"{case_inputs.case_id} centerline overview")
    ax_tree.set_axis_off()

    ax_text = fig.add_subplot(122)
    ax_text.axis("off")
    branch_lines = []
    for branch_id, length_mm in enumerate(branch_lengths):
        canonical_label = branch_name_map.get(branch_id, f"branch_{branch_id}")
        semantic_label = semantic_name_map.get(branch_id, "")
        if semantic_label:
            branch_lines.append(f"{semantic_label} [{canonical_label}]: {length_mm:.1f} mm")
        else:
            branch_lines.append(f"{canonical_label}: {length_mm:.1f} mm")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"Case: {case_inputs.case_id}",
                f"Branches: {len(branch_polylines)}",
                f"Total length (mm): {total_length:.2f}",
                f"Mean branch length (mm): {mean_length:.2f}",
                f"Mean radius (mm): {mean_radius:.2f}",
                "",
                "Semantic names [canonical]:",
                *branch_lines,
            ]
        ),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    artifact = StageArtifact(
        stage_name="rendering",
        stage_dir=stage_dir,
        manifest_path=stage_dir / "manifest.json",
        primary_path=figure_path,
        secondary_paths={"metrics_csv": metrics_csv},
        metadata={"num_branches": len(branch_polylines)},
    )
    write_stage_manifest(artifact)
    return artifact
