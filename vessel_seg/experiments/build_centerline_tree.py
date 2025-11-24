"""Extract centerlines from a coronary mask and export a tree JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vessel_seg.centerline import extract_centerlines_from_mask, extract_centerlines_from_vtp
from vessel_seg.io import load_pair
from vessel_seg.preprocessing import crop_to_mask, resample_isotropic
from vessel_seg.visualization import plot_centerlines_3d, show_slice_with_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Build coronary centerline tree from mask.")
    parser.add_argument("--volume", required=True, help="Path to CT volume (NIfTI).")
    parser.add_argument("--mask", help="Path to coronary binary mask (NIfTI).")
    parser.add_argument("--vtp", help="Path to VMTK centerline .vtp (if provided, mask is not required).")
    parser.add_argument("--output-json", required=True, help="Path to save tree JSON.")
    parser.add_argument("--new-spacing", type=float, default=None, help="Optional isotropic resampling spacing (mm).")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    args = parser.parse_args()

    if args.vtp:
        tree = extract_centerlines_from_vtp(Path(args.vtp))
        vol_data = None
        mask_data = None
    else:
        if not args.mask:
            raise SystemExit("Either --vtp or --mask must be provided.")
        volume, mask = load_pair(args.volume, args.mask)
        vol_data, mask_data = volume.data, mask.data
        if args.new_spacing:
            vol_data, new_sp = resample_isotropic(vol_data, mask.spacing, args.new_spacing, order=1)
            mask_data, _ = resample_isotropic(mask_data, mask.spacing, args.new_spacing, order=0)
            mask.spacing = new_sp
        vol_data, mask_data = crop_to_mask(vol_data, mask_data, margin=8)
        tree = extract_centerlines_from_mask(mask_data, mask.spacing)

    print(f"Extracted {tree.num_branches()} branches, total length {tree.total_length():.2f} mm")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tree.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved tree JSON to {out_path}")

    if not args.no_plot and vol_data is not None:
        show_slice_with_mask(vol_data, mask_data)
    if not args.no_plot:
        plot_centerlines_3d(tree)
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
