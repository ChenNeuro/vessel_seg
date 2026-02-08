"""Routines for converting NRRD volumes to compressed NIfTI (`.nii.gz`) files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import SimpleITK as sitk


def _iter_nrrd_files(input_dir: Path) -> Iterable[Path]:
    """Yield all `.nrrd` files in `input_dir`, ignoring case."""
    for entry in input_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() == ".nrrd":
            yield entry


def convert_nrrd_to_nii(input_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> None:
    """Convert every NRRD volume in `input_dir` into a `.nii.gz` file in `output_dir`."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for in_path in _iter_nrrd_files(input_path):
        out_name = f"{in_path.stem}.nii.gz"
        out_path = output_path / out_name

        try:
            image = sitk.ReadImage(str(in_path))
            sitk.WriteImage(image, str(out_path))
            print(f"转换成功: {in_path} -> {out_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"转换失败: {in_path}, 错误: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI argument parser for batch conversions."""
    parser = argparse.ArgumentParser(
        description="Convert NRRD medical image volumes to NIfTI (.nii.gz) format."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing NRRD volumes.")
    parser.add_argument("output_dir", type=Path, help="Directory to receive converted NIfTI files.")
    return parser


def main() -> None:
    """Command-line entry point for the converter."""
    args = build_arg_parser().parse_args()

    convert_nrrd_to_nii(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
