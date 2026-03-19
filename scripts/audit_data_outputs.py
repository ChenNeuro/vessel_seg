#!/usr/bin/env python3
"""Scan raw data and outputs layout, then emit a compact inventory report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR_RE = re.compile(r"^(Normal|Diseased)_\d+$")


@dataclass
class DirStat:
    path: str
    files: int
    size_mb: float


def file_size_bytes(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def collect_dir_stats(paths: Iterable[Path]) -> list[DirStat]:
    stats: list[DirStat] = []
    for path in sorted(paths):
        if not path.exists() or not path.is_dir():
            continue
        files = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                files += 1
        size_mb = file_size_bytes(path) / (1024 * 1024)
        stats.append(DirStat(path=str(path), files=files, size_mb=round(size_mb, 2)))
    return stats


def normalize_case_subdir(name: str) -> str:
    if name.startswith("branches_pct"):
        return "branches_pct*"
    if name.startswith("similarity_pct"):
        return "similarity_pct*"
    if name.startswith("prior_"):
        return "prior_*"
    if name.startswith("branches_official"):
        return "branches_official*"
    if name.startswith("branches_trim_overlap"):
        return "branches_trim_overlap*"
    return name


def scan_asoca(asoca_root: Path) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for cohort in ["Normal", "Diseased"]:
        cohort_root = asoca_root / cohort
        if not cohort_root.exists():
            continue
        cohort_stats: dict[str, int] = {}
        for subdir in sorted(p for p in cohort_root.iterdir() if p.is_dir()):
            cohort_stats[subdir.name] = sum(1 for p in subdir.iterdir() if p.is_file())
        summary[cohort] = cohort_stats
    return summary


def scan_outputs(outputs_root: Path) -> dict:
    top_level_files = sorted(p.name for p in outputs_root.iterdir() if p.is_file())
    top_level_dirs = sorted(p for p in outputs_root.iterdir() if p.is_dir())

    case_dirs = [p for p in top_level_dirs if p.name.startswith(("Normal_", "Diseased_"))]
    other_dirs = [p for p in top_level_dirs if p not in case_dirs]

    case_subdirs = Counter()
    for case_dir in case_dirs:
        for child in case_dir.iterdir():
            if child.is_dir():
                case_subdirs[normalize_case_subdir(child.name)] += 1

    quant_official = []
    quant_debug = []
    quant_root = outputs_root / "quant"
    if quant_root.exists():
        for child in sorted(p.name for p in quant_root.iterdir() if p.is_dir()):
            if CASE_DIR_RE.match(child):
                quant_official.append(child)
            else:
                quant_debug.append(child)

    largest_dirs = collect_dir_stats(other_dirs + case_dirs)
    largest_dirs = sorted(largest_dirs, key=lambda item: item.size_mb, reverse=True)[:25]

    return {
        "top_level_files": top_level_files,
        "case_dirs": sorted(p.name for p in case_dirs),
        "other_dirs": sorted(p.name for p in other_dirs),
        "case_subdir_patterns": dict(case_subdirs.most_common()),
        "quant_official_candidates": quant_official,
        "quant_debug_candidates": quant_debug,
        "largest_dirs": [asdict(item) for item in largest_dirs],
    }


def render_markdown(asoca_summary: dict, outputs_summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Data / Outputs Inventory")
    lines.append("")
    lines.append("## ASOCA2020")
    lines.append("")
    for cohort, stats in asoca_summary.items():
        lines.append(f"### {cohort}")
        lines.append("")
        for name, count in stats.items():
            lines.append(f"- `{name}`: {count} files")
        lines.append("")

    lines.append("## outputs/")
    lines.append("")
    lines.append(f"- Case directories: {len(outputs_summary['case_dirs'])}")
    lines.append(f"- Other top-level directories: {len(outputs_summary['other_dirs'])}")
    lines.append(f"- Top-level loose files: {len(outputs_summary['top_level_files'])}")
    lines.append("")

    if outputs_summary["top_level_files"]:
        lines.append("### Loose files at outputs root")
        lines.append("")
        for name in outputs_summary["top_level_files"]:
            lines.append(f"- `{name}`")
        lines.append("")

    lines.append("### Common case subdir patterns")
    lines.append("")
    for name, count in outputs_summary["case_subdir_patterns"].items():
        lines.append(f"- `{name}`: {count} occurrences across case dirs")
    lines.append("")

    lines.append("### quant candidates")
    lines.append("")
    lines.append("- Official-like runs:")
    for name in outputs_summary["quant_official_candidates"]:
        lines.append(f"  - `{name}`")
    lines.append("- Debug/experimental runs:")
    for name in outputs_summary["quant_debug_candidates"]:
        lines.append(f"  - `{name}`")
    lines.append("")

    lines.append("### Largest directories")
    lines.append("")
    for item in outputs_summary["largest_dirs"]:
        lines.append(f"- `{item['path']}`: {item['size_mb']} MB, {item['files']} files")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit current data/outputs layout.")
    parser.add_argument(
        "--asoca-root",
        type=Path,
        default=ROOT / "ASOCA2020",
        help="Path to ASOCA2020 raw data root.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT / "outputs",
        help="Path to outputs root.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "docs" / "generated" / "data_output_inventory.json",
        help="Where to write the JSON summary.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=ROOT / "docs" / "generated" / "data_output_inventory.md",
        help="Where to write the Markdown summary.",
    )
    args = parser.parse_args()

    asoca_summary = scan_asoca(args.asoca_root)
    outputs_summary = scan_outputs(args.outputs_root)
    report = {"asoca": asoca_summary, "outputs": outputs_summary}
    markdown = render_markdown(asoca_summary, outputs_summary)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.md_out.write_text(markdown, encoding="utf-8")

    print(f"Wrote JSON summary to {args.json_out}")
    print(f"Wrote Markdown summary to {args.md_out}")


if __name__ == "__main__":
    main()
