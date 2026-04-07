"""Visualization helpers for left/right side galleries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import matplotlib.pyplot as plt

from .branch_cluster_gallery import (
    _case_bundle,
    _load_csv_rows,
    _ordered_case_ids_for_gallery,
    _plot_case_tree,
    _top_hist_items,
    BranchClusterGalleryConfig,
)


@dataclass(frozen=True)
class BranchSideGalleryConfig:
    """左右系统大图生成配置。"""

    summary_csv: Path
    assignments_csv: Path
    output_dir: Path
    side_groups: tuple[str, ...] = ("LCA", "RCA")
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
class BranchSideGalleryResult:
    """左右系统大图生成结果。"""

    output_dir: Path
    index_csv: Path
    image_paths: tuple[Path, ...]


def _group_case_hits(
    assignments_rows: list[dict[str, str]],
    side_groups: tuple[str, ...],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {str(name): {} for name in side_groups}
    selected = set(str(name) for name in side_groups)
    for row in assignments_rows:
        side_group = str(row.get("side_group", "") or "")
        if side_group not in selected:
            continue
        case_id = str(row["case_id"])
        grouped.setdefault(side_group, {}).setdefault(case_id, []).append(row)
    return grouped


def _canonical_histogram_json(rows: list[dict[str, str]]) -> str:
    histogram: dict[str, int] = {}
    for row in rows:
        canonical_name = str(row.get("canonical_name", "") or "")
        histogram[canonical_name] = histogram.get(canonical_name, 0) + 1
    return json.dumps(
        dict(sorted(histogram.items(), key=lambda item: (-item[1], item[0]))),
        ensure_ascii=False,
    )


def generate_branch_side_galleries(config: BranchSideGalleryConfig) -> BranchSideGalleryResult:
    """基于 side_group 生成左右系统大图。"""
    rows = _load_csv_rows(config.summary_csv)
    assignments_rows = _load_csv_rows(config.assignments_csv)
    case_bundles = {
        str(row["case_id"]): _case_bundle(row, align_anatomical_frame=bool(config.align_anatomical_frame))
        for row in rows
    }
    side_case_hits = _group_case_hits(assignments_rows, config.side_groups)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_plot_config = BranchClusterGalleryConfig(
        summary_csv=config.summary_csv,
        assignments_csv=config.assignments_csv,
        cluster_summary_csv=config.summary_csv,
        output_dir=config.output_dir,
        cols=int(config.cols),
        panel_width=float(config.panel_width),
        panel_height=float(config.panel_height),
        dpi=int(config.dpi),
        black_linewidth=float(config.black_linewidth),
        red_linewidth=float(config.red_linewidth),
        elev=float(config.elev),
        azim=float(config.azim),
        max_title_labels=int(config.max_title_labels),
        fixed_case_grid=bool(config.fixed_case_grid),
        align_anatomical_frame=bool(config.align_anatomical_frame),
    )

    image_paths: list[Path] = []
    index_rows: list[dict[str, object]] = []
    cols = max(1, int(config.cols))
    for side_group in config.side_groups:
        side_rows = [row for row in assignments_rows if str(row.get("side_group", "") or "") == side_group]
        case_hits = side_case_hits.get(side_group, {})
        plotting_case_ids = _ordered_case_ids_for_gallery(
            rows,
            fixed_case_grid=bool(config.fixed_case_grid),
            cluster_cases=case_hits,
        )
        rows_count = int(math.ceil(len(plotting_case_ids) / float(cols)))
        fig = plt.figure(
            figsize=(cols * float(config.panel_width), rows_count * float(config.panel_height)),
            dpi=int(config.dpi),
        )
        for plot_index, case_id in enumerate(plotting_case_ids, start=1):
            bundle = case_bundles[case_id]
            hit_rows = case_hits.get(case_id, [])
            highlight_branch_ids = {int(row["branch_id"]) for row in hit_rows}
            title = case_id if not hit_rows else (f"{case_id}\n{len(highlight_branch_ids)} hits")
            ax = fig.add_subplot(rows_count, cols, plot_index, projection="3d")
            _plot_case_tree(
                ax,
                bundle["polylines"],  # type: ignore[arg-type]
                highlight_branch_ids,
                bundle["canonical_map"],  # type: ignore[arg-type]
                title,
                cluster_plot_config,
            )

        fig.suptitle(
            "\n".join(
                [
                    f"Branch Side Group {side_group}",
                    f"highlighted_samples={len(side_rows)} present_cases={len(case_hits)}",
                    f"hist={_top_hist_items(_canonical_histogram_json(side_rows))}",
                    "Layout: fixed by cohort and case number" if config.fixed_case_grid else "Layout: hit-first dynamic order",
                    "Coords: aligned anatomical frame" if config.align_anatomical_frame else "Coords: raw world frame",
                    "Red: side-group members in each case; Black: all other branches",
                ]
            ),
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        image_path = output_dir / f"branch_side_{side_group}.png"
        fig.savefig(image_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        image_paths.append(image_path)
        index_rows.append(
            {
                "side_group": side_group,
                "image_path": str(image_path),
                "highlighted_sample_count": len(side_rows),
                "present_case_count": len(case_hits),
                "canonical_name_histogram": _canonical_histogram_json(side_rows),
            }
        )

    index_csv = output_dir / "branch_side_gallery_index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "side_group",
            "image_path",
            "highlighted_sample_count",
            "present_case_count",
            "canonical_name_histogram",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    return BranchSideGalleryResult(
        output_dir=output_dir,
        index_csv=index_csv,
        image_paths=tuple(image_paths),
    )
