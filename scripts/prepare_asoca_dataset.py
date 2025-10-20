"""Prepare ASOCA dataset in nnUNetv2 format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def copy_case(image: Path, label: Path, image_dst: Path, label_dst: Path) -> None:
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dst)
    shutil.copy2(label, label_dst)


def gather_cases(root: Path) -> Dict[str, Dict[str, Path]]:
    cases: Dict[str, Dict[str, Path]] = {}
    for cohort in ("Normal", "Diseased"):
        img_dir = root / cohort / "CTCA_niigz"
        lbl_dir = root / cohort / "Annotations_niigz"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.nii.gz")):
            case_id = img_path.name.replace(".nii.gz", "")
            lbl_path = lbl_dir / f"{case_id}.nii.gz"
            if not lbl_path.exists():
                raise FileNotFoundError(f"Missing label for {case_id} at {lbl_path}")
            cases[case_id] = {"image": img_path, "label": lbl_path}
    return cases


def write_dataset_json(dataset_dir: Path, cases: List[str]) -> None:
    content = {
        "name": "ASOCA",
        "description": "Automated Segmentation of Coronary Arteries (ASOCA) challenge dataset",
        "reference": "https://asoca.grand-challenge.org/",
        "licence": "Challenge dataset (research use only)",
        "release": "2020",
        "modality": {"0": "CT"},
        "labels": {"background": 0, "coronary_arteries": 1},
        "numTraining": len(cases),
        "numTest": 0,
        "training": [
            {
                "image": f"./imagesTr/{case_id}_0000.nii.gz",
                "label": f"./labelsTr/{case_id}.nii.gz",
            }
            for case_id in cases
        ],
        "test": [],
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2)


def prepare_dataset(source: Path, target_root: Path, dataset_name: str) -> None:
    cases = gather_cases(source)
    if not cases:
        raise RuntimeError("No cases found – ensure NIfTI conversions exist.")

    dataset_dir = target_root / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for case_id, paths in cases.items():
        image_dst = images_tr / f"{case_id}_0000.nii.gz"
        label_dst = labels_tr / f"{case_id}.nii.gz"
        copy_case(paths["image"], paths["label"], image_dst, label_dst)

    write_dataset_json(dataset_dir, sorted(cases.keys()))


def main(asoca_root: str, nnunet_raw: str, dataset_name: str) -> None:
    prepare_dataset(Path(asoca_root), Path(nnunet_raw), dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ASOCA data to nnUNetv2 layout.")
    parser.add_argument("--asoca-root", default="ASOCA2020", help="Path to ASOCA dataset root.")
    parser.add_argument(
        "--nnunet-raw",
        default="nnUNet_raw",
        help="Destination root for nnUNet raw data (will create DatasetXXX folders inside).",
    )
    parser.add_argument(
        "--dataset-name",
        default="Dataset103_ASOCA",
        help="Name of the dataset folder to create.",
    )
    args = parser.parse_args()
    main(args.asoca_root, args.nnunet_raw, args.dataset_name)
