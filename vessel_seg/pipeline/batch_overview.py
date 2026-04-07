"""Batch helpers for generating per-case overview figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from vessel_seg.config import ProjectPaths

from .contracts import CaseInputs, PipelineConfig
from .orchestrator import run_case_pipeline
from .stages import (
    CenterlineExtractionStageConfig,
    CenterlineRepairStageConfig,
    CtSegmentationStageConfig,
    RenderingStageConfig,
    WallFeatureStageConfig,
)


@dataclass(frozen=True)
class AsocaCaseRecord:
    """ASOCA 单病例输入路径。"""

    cohort: str
    case_id: str
    ct_path: Path
    mask_path: Path
    centerline_vtp_path: Path


@dataclass(frozen=True)
class BatchOverviewConfig:
    """批量生成 overview 图的运行配置。"""

    cohorts: tuple[str, ...] = ("Normal",)
    case_ids: tuple[str, ...] = ()
    output_root: Path = Path("outputs_reorganized_runs")
    report_tag: str = "overview_batch"
    skip_existing: bool = True
    limit: int | None = None
    gallery_cols: int = 4
    semantic_prior_path: Path | None = None


@dataclass
class BatchOverviewResult:
    """批量运行结果。"""

    rows: list[dict[str, object]]
    failures: list[dict[str, str]]
    summary_csv: Path
    gallery_png: Path | None
    failures_json: Path


def discover_asoca_cases(project_paths: ProjectPaths, config: BatchOverviewConfig) -> list[AsocaCaseRecord]:
    """从 ASOCA 根目录枚举病例。"""
    selected = set(config.case_ids)
    records: list[AsocaCaseRecord] = []
    for cohort in config.cohorts:
        cohort_root = project_paths.asoca_root / cohort
        ct_dir = cohort_root / "CTCA_nii"
        mask_dir = cohort_root / "Annotations_nii"
        vtp_dir = cohort_root / "Centerlines"
        for ct_path in sorted(ct_dir.glob("*.nii.gz")):
            case_id = ct_path.name.replace(".nii.gz", "")
            if selected and case_id not in selected:
                continue
            mask_path = mask_dir / f"{case_id}.nii.gz"
            centerline_vtp_path = vtp_dir / f"{case_id}.vtp"
            if not mask_path.exists() or not centerline_vtp_path.exists():
                continue
            records.append(
                AsocaCaseRecord(
                    cohort=cohort,
                    case_id=case_id,
                    ct_path=ct_path,
                    mask_path=mask_path,
                    centerline_vtp_path=centerline_vtp_path,
                )
            )
    if config.limit is not None:
        return records[: max(0, int(config.limit))]
    return records


def _report_dir(output_root: Path, report_tag: str) -> Path:
    return output_root / "analysis" / "overview_batches" / report_tag


def _load_metrics_csv(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics[str(row["metric"])] = str(row["value"])
    return metrics


def _collect_case_row(output_root: Path, record: AsocaCaseRecord) -> dict[str, object]:
    case_dir = output_root / "cases" / record.case_id
    tree_path = case_dir / "stages" / "03_centerline_repair" / "tree.json"
    metrics_path = case_dir / "stages" / "05_rendering" / "metrics.csv"
    overview_path = case_dir / "stages" / "05_rendering" / "overview.png"

    tree_payload = json.loads(tree_path.read_text(encoding="utf-8"))
    metrics_payload = _load_metrics_csv(metrics_path)
    branch_name_map = {}
    semantic_name_map = {}
    for branch in tree_payload.get("branches", []):
        naming = branch.get("naming") or {}
        semantic = branch.get("semantic") or {}
        branch_name_map[int(branch["branch_id"])] = str(naming.get("canonical_name") or f"branch_{branch['branch_id']}")
        semantic_name_map[int(branch["branch_id"])] = str(
            semantic.get("semantic_name") or semantic.get("semantic_family") or f"branch_{branch['branch_id']}"
        )
    systems = tree_payload.get("canonical_naming", {}).get("systems", [])
    semantic_payload = tree_payload.get("semantic_topology", {})
    semantic_consistency = semantic_payload.get("consistency", {})
    if systems:
        root_names = [str(system.get("system_name")) for system in systems]
    else:
        root_names = [branch_name_map.get(int(root_id), f"branch_{int(root_id)}") for root_id in tree_payload.get("roots", [])]
    name_signature = "|".join(branch_name_map[branch_id] for branch_id in sorted(branch_name_map))
    semantic_signature = "|".join(semantic_name_map[branch_id] for branch_id in sorted(semantic_name_map))
    return {
        "case_id": record.case_id,
        "cohort": record.cohort,
        "num_branches": int(metrics_payload.get("num_branches", len(tree_payload.get("branches", [])))),
        "num_roots": len(tree_payload.get("roots", [])),
        "roots": json.dumps(tree_payload.get("roots", []), ensure_ascii=False),
        "root_names": json.dumps(root_names, ensure_ascii=False),
        "name_signature": name_signature,
        "semantic_signature": semantic_signature,
        "dominance": str(semantic_payload.get("dominance", "Unknown")),
        "semantic_score": float(semantic_consistency.get("score", 0.0)),
        "semantic_unmatched_branches": int(semantic_consistency.get("unmatched_branches", 0)),
        "semantic_required_coverage": float(semantic_consistency.get("required_coverage", 0.0)),
        "semantic_warning_count": len(list(semantic_consistency.get("warnings", []))),
        "total_length_mm": float(metrics_payload.get("total_length_mm", 0.0)),
        "mean_branch_length_mm": float(metrics_payload.get("mean_branch_length_mm", 0.0)),
        "mean_radius_mm": float(metrics_payload.get("mean_radius_mm", 0.0)),
        "overview_path": str(overview_path),
    }


def _write_summary_csv(rows: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "cohort",
        "num_branches",
        "num_roots",
        "roots",
        "root_names",
        "name_signature",
        "semantic_signature",
        "dominance",
        "semantic_score",
        "semantic_unmatched_branches",
        "semantic_required_coverage",
        "semantic_warning_count",
        "total_length_mm",
        "mean_branch_length_mm",
        "mean_radius_mm",
        "overview_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_failures_json(failures: list[dict[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _build_gallery(rows: list[dict[str, object]], out_path: Path, cols: int) -> Path | None:
    valid_rows = [row for row in rows if Path(str(row["overview_path"])).exists()]
    if not valid_rows:
        return None

    cols = max(1, int(cols))
    n = len(valid_rows)
    rows_count = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.8 * cols, 3.8 * rows_count), dpi=180)
    axes_array = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes_array, valid_rows):
        image = plt.imread(str(row["overview_path"]))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{row['case_id']}\nbranches={row['num_branches']} roots={row['num_roots']}",
            fontsize=9,
        )

    for ax in axes_array[len(valid_rows):]:
        ax.axis("off")

    fig.suptitle(
        "ASOCA overview gallery\nSemantic labels are prior-based heuristics; canonical labels remain topology-stable ids.",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


def run_batch_overview(project_paths: ProjectPaths, config: BatchOverviewConfig) -> BatchOverviewResult:
    """批量运行多病例 overview 生成。"""
    output_root = (project_paths.root / config.output_root).resolve() if not config.output_root.is_absolute() else config.output_root
    report_dir = _report_dir(output_root, config.report_tag)
    records = discover_asoca_cases(project_paths, config)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for record in records:
        overview_path = output_root / "cases" / record.case_id / "stages" / "05_rendering" / "overview.png"
        try:
            if not (config.skip_existing and overview_path.exists()):
                run_case_pipeline(
                    CaseInputs(
                        case_id=record.case_id,
                        ct_path=record.ct_path,
                        mask_path=record.mask_path,
                        centerline_vtp_path=record.centerline_vtp_path,
                    ),
                    project_paths=project_paths,
                    pipeline_config=PipelineConfig(
                        output_root=output_root,
                        segmentation_backend="existing_mask",
                        centerline_backend="vtp_copy",
                        repair_mode="topology_only",
                    ),
                    ct_config=CtSegmentationStageConfig(backend="existing_mask"),
                    extraction_config=CenterlineExtractionStageConfig(backend="vtp_copy"),
                    repair_config=CenterlineRepairStageConfig(
                        mode="topology_only",
                        semantic_prior_path=config.semantic_prior_path,
                    ),
                    wall_config=WallFeatureStageConfig(),
                    rendering_config=RenderingStageConfig(),
                )
            rows.append(_collect_case_row(output_root, record))
        except Exception as exc:
            failures.append(
                {
                    "case_id": record.case_id,
                    "cohort": record.cohort,
                    "error": str(exc),
                }
            )

    rows.sort(key=lambda row: str(row["case_id"]))
    summary_csv = _write_summary_csv(rows, report_dir / "summary.csv")
    failures_json = _write_failures_json(failures, report_dir / "failures.json")
    gallery_png = _build_gallery(rows, report_dir / "gallery.png", config.gallery_cols)
    return BatchOverviewResult(
        rows=rows,
        failures=failures,
        summary_csv=summary_csv,
        gallery_png=gallery_png,
        failures_json=failures_json,
    )
