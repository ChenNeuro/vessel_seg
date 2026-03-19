#!/usr/bin/env python3
"""Non-destructively reorganize selected outputs into a cleaner target layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_RE = re.compile(r"^(Normal|Diseased)_\d+$")


@dataclass
class MigrationRecord:
    source: str
    target: str
    category: str
    action: str
    size_mb: float
    status: str


def size_mb(path: Path) -> float:
    if path.is_file():
        try:
            return round(path.stat().st_size / (1024 * 1024), 3)
        except OSError:
            return 0.0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return round(total / (1024 * 1024), 3)


def copy_path(src: Path, dst: Path, *, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return "skipped_exists"
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copied"


def classify_loose_output(file_path: Path) -> Path:
    name = file_path.name
    lowered = name.lower()

    if lowered.startswith("normal1_") or name.startswith("Normal_1_"):
        return Path("cases/Normal_1/legacy_root") / name
    if lowered.startswith("shape_diseased_1") or lowered.startswith("diseased_1"):
        return Path("cases/Diseased_1/legacy_root") / name
    if lowered.startswith("coronary_arteries") or lowered.startswith("preview_coronary_arteries"):
        return Path("archive/legacy_outputs_root/misc") / name
    return Path("archive/legacy_outputs_root/unclassified") / name


def classify_quant_dir(dir_path: Path) -> Path:
    name = dir_path.name
    if CASE_RE.match(name):
        return Path("quant/official") / name
    return Path("quant/experiments") / name


def branch_variant(name: str) -> str:
    if name == "branches":
        return "default"
    return name.removeprefix("branches_")


def similarity_variant(name: str) -> str:
    if name == "similarity":
        return "default"
    return name.removeprefix("similarity_")


def prior_variant(name: str) -> str:
    return name.removeprefix("prior_")


def dataset_variant(file_name: str) -> str:
    if file_name == "branch_dataset.npz":
        return "default"
    return file_name.removeprefix("branch_dataset_").removesuffix(".npz")


def tree_variant(file_name: str) -> str:
    if file_name == "tree.json":
        return "default"
    return file_name.removeprefix("tree_").removesuffix(".json")


def classify_case_dir_subpath(case_name: str, child_name: str) -> Path | None:
    if child_name.startswith("branches"):
        return Path("cases") / case_name / "branches" / branch_variant(child_name)
    if child_name.startswith("similarity"):
        return Path("cases") / case_name / "similarity" / similarity_variant(child_name)
    if child_name.startswith("prior_"):
        return Path("cases") / case_name / "priors" / prior_variant(child_name)
    if child_name == "cpr":
        return Path("cases") / case_name / "cpr" / "default"
    if child_name.startswith("branch_tree"):
        return Path("cases") / case_name / "topology" / child_name
    if child_name == "coronary_tree_guess":
        return Path("cases") / case_name / "topology" / child_name
    if child_name.startswith("fgpm"):
        return Path("cases") / case_name / "fgpm" / child_name
    return None


def classify_case_file_subpath(case_name: str, file_name: str) -> Path | None:
    if file_name == "alignment.json":
        return Path("cases") / case_name / "alignment" / "alignment.json"
    if file_name.startswith("branch_dataset") and file_name.endswith(".npz"):
        return Path("cases") / case_name / "datasets" / dataset_variant(file_name) / "branch_dataset.npz"
    if file_name.startswith("tree") and file_name.endswith(".json"):
        return Path("cases") / case_name / "tree" / tree_variant(file_name) / "tree.json"
    if file_name == "centerline_extracted.vtp":
        return Path("cases") / case_name / "centerline" / "extracted" / "centerline.vtp"
    if file_name == "centerline_repaired.vtp":
        return Path("cases") / case_name / "centerline" / "repaired" / "repaired.vtp"
    if file_name == "centerline_repair_report.json":
        return Path("cases") / case_name / "centerline" / "repaired" / "repair_report.json"
    if file_name == "prob_from_mask.nii.gz":
        return Path("cases") / case_name / "centerline" / "probability" / "prob_from_mask.nii.gz"
    if file_name.endswith(".png") or file_name.endswith(".npz"):
        return Path("cases") / case_name / "analysis" / file_name
    if file_name.endswith(".json") or file_name.endswith(".vtp"):
        return Path("cases") / case_name / "analysis" / file_name
    return None


def migrate_loose_outputs(outputs_root: Path, target_root: Path, *, overwrite: bool) -> list[MigrationRecord]:
    records: list[MigrationRecord] = []
    for src in sorted(p for p in outputs_root.iterdir() if p.is_file()):
        rel_target = classify_loose_output(src)
        dst = target_root / rel_target
        status = copy_path(src, dst, overwrite=overwrite)
        records.append(
            MigrationRecord(
                source=str(src),
                target=str(dst),
                category="loose_output_file",
                action="copy",
                size_mb=size_mb(src),
                status=status,
            )
        )
    return records


def migrate_quant(outputs_root: Path, target_root: Path, *, overwrite: bool) -> list[MigrationRecord]:
    records: list[MigrationRecord] = []
    quant_root = outputs_root / "quant"
    if not quant_root.exists():
        return records

    for src in sorted(p for p in quant_root.iterdir() if p.is_dir()):
        rel_target = classify_quant_dir(src)
        dst = target_root / rel_target
        status = copy_path(src, dst, overwrite=overwrite)
        category = "quant_official" if CASE_RE.match(src.name) else "quant_experiment"
        records.append(
            MigrationRecord(
                source=str(src),
                target=str(dst),
                category=category,
                action="copy",
                size_mb=size_mb(src),
                status=status,
            )
        )
    return records


def migrate_case_dirs(outputs_root: Path, target_root: Path, *, overwrite: bool) -> list[MigrationRecord]:
    records: list[MigrationRecord] = []
    case_dirs = sorted(p for p in outputs_root.iterdir() if p.is_dir() and CASE_RE.match(p.name))

    for case_dir in case_dirs:
        case_name = case_dir.name
        for child in sorted(case_dir.iterdir()):
            if child.is_dir():
                rel_target = classify_case_dir_subpath(case_name, child.name)
                if rel_target is None:
                    continue
                dst = target_root / rel_target
                status = copy_path(child, dst, overwrite=overwrite)
                records.append(
                    MigrationRecord(
                        source=str(child),
                        target=str(dst),
                        category="case_dir",
                        action="copy",
                        size_mb=size_mb(child),
                        status=status,
                    )
                )
            elif child.is_file():
                rel_target = classify_case_file_subpath(case_name, child.name)
                if rel_target is None:
                    continue
                dst = target_root / rel_target
                status = copy_path(child, dst, overwrite=overwrite)
                records.append(
                    MigrationRecord(
                        source=str(child),
                        target=str(dst),
                        category="case_file",
                        action="copy",
                        size_mb=size_mb(child),
                        status=status,
                    )
                )
    return records


def write_markdown(records: list[MigrationRecord], out_path: Path) -> None:
    lines = ["# Outputs Migration Report", ""]
    if not records:
        lines.append("No records.")
    else:
        grouped: dict[str, list[MigrationRecord]] = {}
        for record in records:
            grouped.setdefault(record.category, []).append(record)

        for category in sorted(grouped):
            lines.append(f"## {category}")
            lines.append("")
            for record in grouped[category]:
                lines.append(
                    f"- `{record.source}` -> `{record.target}` | {record.status} | {record.size_mb} MB"
                )
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_data_skeleton(root: Path) -> None:
    dirs = [
        root / "data" / "raw",
        root / "data" / "interim",
        root / "data" / "external",
        root / "data" / "processed",
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")

    readme = root / "data" / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Data Layout\n\n"
            "- `raw/`: immutable source datasets (e.g. ASOCA2020)\n"
            "- `interim/`: intermediate derived data\n"
            "- `processed/`: standardized data prepared for modeling\n"
            "- `external/`: third-party datasets not stored in repo\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy selected outputs into a cleaner target layout.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT / "outputs",
        help="Legacy outputs root.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "outputs_reorganized",
        help="Target root for reorganized outputs.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "docs" / "generated" / "outputs_migration_report.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "docs" / "generated" / "outputs_migration_report.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already copied targets.",
    )
    args = parser.parse_args()

    ensure_data_skeleton(ROOT)

    records: list[MigrationRecord] = []
    records.extend(migrate_loose_outputs(args.outputs_root, args.target_root, overwrite=args.overwrite))
    records.extend(migrate_quant(args.outputs_root, args.target_root, overwrite=args.overwrite))
    records.extend(migrate_case_dirs(args.outputs_root, args.target_root, overwrite=args.overwrite))

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps([asdict(record) for record in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(records, args.report_md)

    print(f"Migrated {len(records)} items into {args.target_root}")
    print(f"JSON report: {args.report_json}")
    print(f"Markdown report: {args.report_md}")


if __name__ == "__main__":
    main()
