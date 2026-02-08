"""Convert ASOCA2020 volumes into 2D PNG pairs for RCF fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import imageio.v3 as imageio
import nibabel as nib
import numpy as np
from skimage import color, feature, transform

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def _case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _gather_volumes(directories: Sequence[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for directory in directories:
        for ext in ("*.nii.gz", "*.nii"):
            for path in directory.glob(ext):
                mapping[_case_id(path)] = path
    return mapping


def _window_to_uint8(slice_data: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    low, high = window
    scaled = np.clip(slice_data, low, high)
    scaled = (scaled - low) / max(high - low, 1e-6)
    scaled = (scaled * 255.0).astype(np.uint8)
    return scaled


def _edge_from_mask(mask_slice: np.ndarray) -> np.ndarray:
    binary = mask_slice > 0.5
    if not binary.any():
        return np.zeros_like(mask_slice, dtype=np.uint8)
    edges = feature.canny(binary, sigma=0.0).astype(np.uint8) * 255
    return edges


def _rescale_image(
    array: np.ndarray,
    scale: float,
    *,
    order: int,
    channel_axis: int | None,
) -> np.ndarray:
    if math.isclose(scale, 1.0):
        return array
    target_h = max(1, int(round(array.shape[0] * scale)))
    target_w = max(1, int(round(array.shape[1] * scale)))
    return _resize_spatial(array, (target_h, target_w), order=order, channel_axis=channel_axis)


def _resize_spatial(
    array: np.ndarray,
    spatial_shape: Tuple[int, int],
    *,
    order: int,
    channel_axis: int | None,
) -> np.ndarray:
    if array.ndim == 2:
        resized = transform.resize(
            array,
            spatial_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=order != 0,
        )
        return resized.astype(array.dtype)

    if channel_axis is None:
        target_shape = spatial_shape + (array.shape[-1],)
        resized = transform.resize(
            array,
            target_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=order != 0,
        )
        return resized.astype(array.dtype)

    rearranged = np.moveaxis(array, channel_axis, -1)
    channels = []
    for c in range(rearranged.shape[-1]):
        channel_resized = transform.resize(
            rearranged[..., c],
            spatial_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=order != 0,
        )
        channels.append(channel_resized)

    stacked = np.stack(channels, axis=-1)
    restored = np.moveaxis(stacked, -1, channel_axis)
    return restored.astype(array.dtype)


def _ensure_shape(
    array: np.ndarray,
    target_shape: Tuple[int, int] | None,
    *,
    order: int,
    channel_axis: int | None,
) -> np.ndarray:
    if target_shape is None:
        return array
    target_h, target_w = target_shape
    if array.shape[0] == target_h and array.shape[1] == target_w:
        return array
    return _resize_spatial(array, target_shape, order=order, channel_axis=channel_axis)


def _write_lst(path: Path, pairs: Iterable[Tuple[Path, Path]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for img_path, gt_path in pairs:
            handle.write(f"{img_path} {gt_path}\n")


def _assign_splits(cases: Sequence[str], val_ratio: float, seed: int) -> Tuple[set[str], set[str]]:
    rng = random.Random(seed)
    case_ids = list(cases)
    rng.shuffle(case_ids)
    if val_ratio <= 0 or len(case_ids) < 2:
        return set(case_ids), set()
    val_count = max(1, math.floor(len(case_ids) * val_ratio))
    val_cases = set(case_ids[:val_count])
    train_cases = set(case_ids[val_count:])
    if not train_cases:
        train_cases, val_cases = val_cases, set()
    return train_cases, val_cases


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def prepare_dataset(args: argparse.Namespace) -> None:
    image_dirs = [Path(p).expanduser().resolve() for p in args.image_dirs]
    label_dirs = [Path(p).expanduser().resolve() for p in args.label_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _gather_volumes(image_dirs)
    labels = _gather_volumes(label_dirs)
    common_cases = sorted(set(images) & set(labels))
    if args.case_limit:
        common_cases = common_cases[: args.case_limit]
    if not common_cases:
        raise RuntimeError("没有找到匹配的影像/标注对，请检查输入目录。")

    train_cases, val_cases = _assign_splits(common_cases, args.val_ratio, args.seed)
    splits = {"train": train_cases, "val": val_cases}

    split_records: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": []}
    stats = {"cases": {}, "total_samples": 0}

    window = (args.window_min, args.window_max)
    scales = args.scales
    target_shape = None
    if args.target_height and args.target_width:
        target_shape = (args.target_height, args.target_width)

    for case_id in _progress(common_cases, desc="病例", unit="case"):
        image_path = images[case_id]
        label_path = labels[case_id]
        split = "train" if case_id in train_cases else "val"
        if split == "val" and not val_cases:
            split = "train"

        img_nifti = nib.load(str(image_path))
        lbl_nifti = nib.load(str(label_path))
        img_data = np.asarray(img_nifti.get_fdata(), dtype=np.float32)
        lbl_data = np.asarray(lbl_nifti.get_fdata(), dtype=np.float32)
        if img_data.shape != lbl_data.shape:
            raise ValueError(f"{case_id} 图像与标注尺寸不一致: {img_data.shape} vs {lbl_data.shape}")

        per_case_count = 0
        slice_iter = _progress(
            range(img_data.shape[0]),
            desc=f"{case_id} 切片",
            unit="slice",
            leave=False,
            total=img_data.shape[0],
        )
        for slice_idx in slice_iter:
            mask_slice = lbl_data[slice_idx]
            if mask_slice.sum() < args.min_positive_pixels:
                continue

            img_slice = img_data[slice_idx]
            img_uint8 = _window_to_uint8(img_slice, window)
            img_rgb = color.gray2rgb(img_uint8)

            edges = _edge_from_mask(mask_slice)
            if edges.sum() == 0:
                continue

            for scale_idx, scale in enumerate(scales):
                img_scaled = _rescale_image(img_rgb, scale, order=1, channel_axis=-1)
                edge_scaled = _rescale_image(edges, scale, order=0, channel_axis=None)
                img_scaled = _ensure_shape(img_scaled, target_shape, order=1, channel_axis=-1)
                edge_scaled = _ensure_shape(edge_scaled, target_shape, order=0, channel_axis=None)

                split_dir = output_dir / split
                img_dir = split_dir / "img"
                gt_dir = split_dir / "gt"
                img_dir.mkdir(parents=True, exist_ok=True)
                gt_dir.mkdir(parents=True, exist_ok=True)

                slug = f"{case_id}_{slice_idx:03d}_s{scale_idx}"
                img_file = img_dir / f"{slug}.png"
                gt_file = gt_dir / f"{slug}.png"
                imageio.imwrite(str(img_file), img_scaled.astype(np.uint8))
                imageio.imwrite(str(gt_file), edge_scaled.astype(np.uint8))
                split_records[split].append((img_file.resolve(), gt_file.resolve()))
                per_case_count += 1
                stats["total_samples"] += 1

        stats["cases"][case_id] = per_case_count

    for split, records in split_records.items():
        if not records:
            continue
        _write_lst(output_dir / f"{split}.lst", records)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output_dir": str(output_dir),
                "cases": stats["cases"],
                "total_samples": stats["total_samples"],
                "splits": {split: len(records) for split, records in split_records.items()},
                "scales": scales,
                "window": window,
                "min_positive_pixels": args.min_positive_pixels,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"完成数据集构建，摘要已写入 {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ASOCA2020 slices for RCF fine-tuning.")
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        required=True,
        help="One or more directories containing CTA volumes (.nii/.nii.gz).",
    )
    parser.add_argument(
        "--label-dirs",
        nargs="+",
        required=True,
        help="Directories containing the corresponding binary vessel masks.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the 2D dataset (creates train/ and val/).",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=(0.5, 1.0, 1.5),
        help="Scale factors to augment each slice.",
    )
    parser.add_argument(
        "--window-min",
        type=float,
        default=-200.0,
        help="CT intensity window minimum (HU).",
    )
    parser.add_argument(
        "--window-max",
        type=float,
        default=500.0,
        help="CT intensity window maximum (HU).",
    )
    parser.add_argument(
        "--min-positive-pixels",
        type=int,
        default=200,
        help="Skip slices whose mask foreground is below this pixel count.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of cases reserved for validation (case-level split).",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optionally only process the first N matching cases (useful for smoke tests).",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=None,
        help="Optional fixed output height (pixels). Requires --target-width.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help="Optional fixed output width (pixels). Requires --target-height.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for case split.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if len(args.image_dirs) != len(args.label_dirs):
        raise SystemExit("image_dirs 与 label_dirs 数量必须一致。")
    prepare_dataset(args)


if __name__ == "__main__":
    main()
