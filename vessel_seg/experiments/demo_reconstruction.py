"""Toy reconstruction demo using placeholder branch models."""

from __future__ import annotations

import argparse
from pathlib import Path

from vessel_seg.branch_model import SimpleLengthRadiusModel
from vessel_seg.centerline import extract_centerlines_from_mask
from vessel_seg.io import load_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo placeholder branch reconstruction.")
    parser.add_argument("--mask", required=True, help="Path to binary coronary mask (NIfTI).")
    args = parser.parse_args()

    mask_vol = load_mask(Path(args.mask))
    tree = extract_centerlines_from_mask(mask_vol.data, mask_vol.spacing)
    branches = list(tree.iter_branches())

    model = SimpleLengthRadiusModel()
    model.fit(branches)
    print(f"Loaded {len(branches)} branches; model '{model.name}' is ready for sampling.")


if __name__ == "__main__":
    main()
