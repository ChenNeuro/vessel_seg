"""Visualization helpers for branch clustering galleries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np

from .semantic_topology import _build_descriptors, _build_frame, _infer_root_systems


@dataclass(frozen=True)
class BranchClusterGalleryConfig:
    """分支聚类大图生成配置。"""

    summary_csv: Path
    assignments_csv: Path
    cluster_summary_csv: Path
    output_dir: Path
    cluster_ids: tuple[int, ...] = ()
    cols: int = 8
    panel_width: float = 3.2
    panel_height: float = 3.2
    dpi: int = 220
    black_linewidth: float = 1.0
    red_linewidth: float = 2.8
    elev: float = 18.0
    azim: float = -64.0
    max_title_labels: int = 2
    fixed_case_grid: bool = True
    align_anatomical_frame: bool = True


@dataclass(frozen=True)
class BranchClusterGalleryResult:
    """批量 cluster gallery 产物。"""

    output_dir: Path
    index_csv: Path
    image_paths: tuple[Path, ...]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_branch_polylines(vtp_path: Path) -> list[np.ndarray]:
    try:
        import vtk
    except Exception as exc:  # pragma: no cover
        raise ImportError("vtk is required to render branch cluster galleries.") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    points = poly.GetPoints()
    lines = poly.GetLines()
    polylines: list[np.ndarray] = []
    lines.InitTraversal()
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        coords = np.asarray([points.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=np.float32)
        if coords.shape[0] >= 2:
            polylines.append(coords)
    return polylines


def _case_stage_paths(overview_path: Path) -> tuple[Path, Path]:
    case_dir = overview_path.parents[2]
    repair_dir = case_dir / "stages" / "03_centerline_repair"
    return repair_dir / "centerline_repaired.vtp", repair_dir / "tree.json"


def _align_polylines_to_anatomical_frame(
    polylines: list[np.ndarray],
    tree_payload: dict[str, object],
) -> list[np.ndarray]:
    descriptors, roots = _build_descriptors(tree_payload)
    frame = _build_frame(_infer_root_systems(roots, descriptors), descriptors)
    origin = np.asarray(frame.origin_mm, dtype=np.float32)
    lr_axis = np.asarray(frame.left_right_axis, dtype=np.float32)
    long_axis = np.asarray(frame.longitudinal_axis, dtype=np.float32)
    norm_axis = np.asarray(frame.normal_axis, dtype=np.float32)
    aligned: list[np.ndarray] = []
    for polyline in polylines:
        rel = polyline - origin
        coords = np.stack(
            [
                rel @ lr_axis,
                rel @ long_axis,
                rel @ norm_axis,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        aligned.append(coords)
    return aligned


def _case_bundle(summary_row: dict[str, str], *, align_anatomical_frame: bool) -> dict[str, object]:
    overview_path = Path(summary_row["overview_path"])
    repaired_vtp, tree_json = _case_stage_paths(overview_path)
    payload = json.loads(tree_json.read_text(encoding="utf-8"))
    canonical_map: dict[int, str] = {}
    for branch in list(payload.get("branches", [])):
        naming = dict(branch.get("naming") or {})
        canonical_map[int(branch["branch_id"])] = str(naming.get("canonical_name") or f"branch_{branch['branch_id']}")
    polylines = _read_branch_polylines(repaired_vtp)
    if align_anatomical_frame:
        polylines = _align_polylines_to_anatomical_frame(polylines, payload)
    return {
        "case_id": str(summary_row["case_id"]),
        "cohort": str(summary_row.get("cohort", "")),
        "polylines": polylines,
        "canonical_map": canonical_map,
    }


def _cluster_case_hits(assignments_rows: list[dict[str, str]]) -> dict[int, dict[str, list[dict[str, str]]]]:
    cluster_map: dict[int, dict[str, list[dict[str, str]]]] = {}
    for row in assignments_rows:
        cluster_id = int(row["cluster_id"])
        case_id = str(row["case_id"])
        cluster_map.setdefault(cluster_id, {}).setdefault(case_id, []).append(row)
    return cluster_map


def _case_grid_sort_key(summary_row: dict[str, str]) -> tuple[int, str, int, str]:
    cohort = str(summary_row.get("cohort", "") or "")
    case_id = str(summary_row.get("case_id", "") or "")
    cohort_rank = {
        "Normal": 0,
        "Diseased": 1,
    }.get(cohort, 99)
    match = re.match(r"^(.*?)(?:_(\d+))?$", case_id)
    prefix = case_id
    number = 10**9
    if match:
        prefix = str(match.group(1) or case_id)
        if match.group(2) is not None:
            number = int(match.group(2))
    return (cohort_rank, cohort or prefix, number, case_id)


def _ordered_case_ids_for_gallery(
    rows: list[dict[str, str]],
    *,
    fixed_case_grid: bool,
    cluster_cases: dict[str, list[dict[str, str]]] | None = None,
) -> list[str]:
    if fixed_case_grid:
        ordered_rows = sorted(rows, key=_case_grid_sort_key)
        return [str(row["case_id"]) for row in ordered_rows]
    cluster_cases = cluster_cases or {}
    ordered_case_ids = [str(row["case_id"]) for row in rows]
    ordered_hits = sorted(
        cluster_cases.items(),
        key=lambda item: (
            -len(item[1]),
            -max(float(row["overall_similarity_to_prototype"]) for row in item[1]),
            item[0],
        ),
    )
    present_case_ids = [case_id for case_id, _ in ordered_hits]
    absent_case_ids = [case_id for case_id in ordered_case_ids if case_id not in cluster_cases]
    return present_case_ids + absent_case_ids


def _top_hist_items(histogram_json: str, top_k: int = 5) -> str:
    try:
        histogram = json.loads(histogram_json)
    except json.JSONDecodeError:
        return histogram_json
    items = sorted(histogram.items(), key=lambda item: (-int(item[1]), item[0]))
    return ", ".join(f"{name}:{count}" for name, count in items[:top_k])


def _subplot_title(case_id: str, hit_rows: list[dict[str, str]], max_title_labels: int) -> str:
    if not hit_rows:
        return case_id
    canonical_names = sorted({str(row["canonical_name"]) for row in hit_rows})
    if len(canonical_names) <= max(1, int(max_title_labels)):
        return f"{case_id}\n" + ",".join(canonical_names)
    return f"{case_id}\n{len(canonical_names)} hits"


def _set_equal_3d_axes(ax, polylines: list[np.ndarray]) -> None:
    coords = np.concatenate(polylines, axis=0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _plot_case_tree(
    ax,
    polylines: list[np.ndarray],
    highlight_branch_ids: set[int],
    canonical_map: dict[int, str],
    title: str,
    config: BranchClusterGalleryConfig,
) -> None:
    for branch_id, polyline in enumerate(polylines):
        if branch_id in highlight_branch_ids:
            continue
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            polyline[:, 2],
            color="black",
            linewidth=float(config.black_linewidth),
            alpha=0.95,
        )
    for branch_id in sorted(highlight_branch_ids):
        if branch_id >= len(polylines):
            continue
        polyline = polylines[branch_id]
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            polyline[:, 2],
            color="red",
            linewidth=float(config.red_linewidth),
            alpha=1.0,
        )
        anchor = polyline[len(polyline) // 2]
        ax.text(
            anchor[0],
            anchor[1],
            anchor[2],
            canonical_map.get(branch_id, f"branch_{branch_id}"),
            color="red",
            fontsize=6,
        )
    _set_equal_3d_axes(ax, polylines)
    ax.view_init(elev=float(config.elev), azim=float(config.azim))
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def generate_branch_cluster_galleries(config: BranchClusterGalleryConfig) -> BranchClusterGalleryResult:
    """基于 cluster assignment 生成每个 cluster 的高亮 gallery。"""
    rows = _load_csv_rows(config.summary_csv)
    assignments_rows = _load_csv_rows(config.assignments_csv)
    cluster_summary_rows = {
        int(row["cluster_id"]): row
        for row in _load_csv_rows(config.cluster_summary_csv)
    }
    case_bundles = {
        str(row["case_id"]): _case_bundle(row, align_anatomical_frame=bool(config.align_anatomical_frame))
        for row in rows
    }
    cluster_hits = _cluster_case_hits(assignments_rows)
    cluster_ids = sorted(int(cluster_id) for cluster_id in cluster_hits)
    if config.cluster_ids:
        selected = set(int(cluster_id) for cluster_id in config.cluster_ids)
        cluster_ids = [cluster_id for cluster_id in cluster_ids if cluster_id in selected]

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    image_paths: list[Path] = []

    cols = max(1, int(config.cols))
    for cluster_id in cluster_ids:
        cluster_cases = cluster_hits.get(cluster_id, {})
        plotting_case_ids = _ordered_case_ids_for_gallery(
            rows,
            fixed_case_grid=bool(config.fixed_case_grid),
            cluster_cases=cluster_cases,
        )
        present_case_ids = [case_id for case_id in plotting_case_ids if case_id in cluster_cases]

        rows_count = int(math.ceil(len(plotting_case_ids) / float(cols)))
        fig = plt.figure(
            figsize=(cols * float(config.panel_width), rows_count * float(config.panel_height)),
            dpi=int(config.dpi),
        )

        for plot_index, case_id in enumerate(plotting_case_ids, start=1):
            bundle = case_bundles[case_id]
            hit_rows = cluster_cases.get(case_id, [])
            highlight_branch_ids = {int(row["branch_id"]) for row in hit_rows}
            title = _subplot_title(case_id, hit_rows, config.max_title_labels)
            ax = fig.add_subplot(rows_count, cols, plot_index, projection="3d")
            _plot_case_tree(
                ax,
                bundle["polylines"],  # type: ignore[arg-type]
                highlight_branch_ids,
                bundle["canonical_map"],  # type: ignore[arg-type]
                title,
                config,
            )

        summary_row = cluster_summary_rows.get(cluster_id, {})
        fig.suptitle(
            "\n".join(
                [
                    f"Branch Cluster {cluster_id:02d}",
                    (
                        f"size={summary_row.get('cluster_size', '?')} "
                        f"prototype={summary_row.get('prototype_case_id', '?')}:{summary_row.get('prototype_canonical_name', '?')} "
                        f"dominant={summary_row.get('dominant_canonical_name', '?')} "
                        f"({summary_row.get('dominant_canonical_ratio', '?')})"
                    ),
                    f"hist={_top_hist_items(str(summary_row.get('canonical_name_histogram', '{}')))}",
                    "Layout: fixed by cohort and case number" if config.fixed_case_grid else "Layout: hit-first dynamic order",
                    "Coords: aligned anatomical frame" if config.align_anatomical_frame else "Coords: raw world frame",
                    "Red: cluster members in each case; Black: all other branches",
                ]
            ),
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        image_path = output_dir / f"branch_cluster_{cluster_id:02d}.png"
        fig.savefig(image_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

        image_paths.append(image_path)
        index_rows.append(
            {
                "cluster_id": cluster_id,
                "image_path": str(image_path),
                "cluster_size": summary_row.get("cluster_size", ""),
                "prototype_case_id": summary_row.get("prototype_case_id", ""),
                "prototype_canonical_name": summary_row.get("prototype_canonical_name", ""),
                "dominant_canonical_name": summary_row.get("dominant_canonical_name", ""),
                "dominant_canonical_ratio": summary_row.get("dominant_canonical_ratio", ""),
                "present_case_count": len(present_case_ids),
            }
        )

    index_csv = output_dir / "branch_cluster_gallery_index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cluster_id",
            "image_path",
            "cluster_size",
            "prototype_case_id",
            "prototype_canonical_name",
            "dominant_canonical_name",
            "dominant_canonical_ratio",
            "present_case_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    return BranchClusterGalleryResult(
        output_dir=output_dir,
        index_csv=index_csv,
        image_paths=tuple(image_paths),
    )
