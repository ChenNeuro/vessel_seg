"""Convert RCF edge prediction PNG slices into a NIfTI volume aligned to a reference scan."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import imageio.v3 as imageio
import nibabel as nib
import numpy as np


def load_png_stack(png_files: List[Path]) -> np.ndarray:
    slices = []
    for png in png_files:
        arr = imageio.imread(png)
        if arr.ndim == 3:
            arr = arr[..., 0]
        slices.append(arr.astype(np.float32))
    return np.stack(slices, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack RCF PNG slices into a NIfTI volume.")
    parser.add_argument("--rcf-dir", required=True, help="Directory containing RCF edge PNGs (sorted by filename).")
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference NIfTI (CTA or mask) to copy affine/shape for alignment.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output NIfTI path (e.g., outputs/rcf_edges_case001.nii.gz).",
    )
    args = parser.parse_args()

    png_files = sorted(Path(args.rcf_dir).glob("*.png"))
    if not png_files:
        raise SystemExit(f"No PNG files found in {args.rcf_dir}")

    ref_img = nib.load(args.reference)
    ref_shape = ref_img.shape

    volume = load_png_stack(png_files)
    if volume.shape[:3] != ref_shape[:3]:
        # If slice count or in-plane size mismatch, fail fast to avoid misalignment.
        raise SystemExit(
            f"Shape mismatch: PNG stack {volume.shape} vs reference {ref_shape}. "
            "Ensure slice order/count matches the reference volume."
        )

    # Normalize to [0,1] and save as float32
    volume = (volume - volume.min()) / max(volume.max() - volume.min(), 1e-6)
    out_img = nib.Nifti1Image(volume.astype(np.float32), ref_img.affine, ref_img.header)
    out_img.set_data_dtype(np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, out_path)
    print(f"Saved RCF edge volume to {out_path}")


if __name__ == "__main__":
    main()
