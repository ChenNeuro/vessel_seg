"""
Batch pipeline: build tree, branch dataset, similarity for multiple cases.

Defaults target ASOCA2020/Normal_*:
  Centerlines: ASOCA2020/Normal/Centerlines/Normal_<id>.vtp
  Masks:       ASOCA2020/Normal/Annotations_nii/Normal_<id>.nii.gz
Outputs:
  outputs/<case>/tree.json
  outputs/<case>/branch_dataset.npz
  outputs/<case>/branches/
  outputs/<case>/similarity/

Usage:
  conda activate vessel_seg
  python scripts/run_pipeline.py --pattern ASOCA2020/Normal/Centerlines/Normal_*.vtp
"""

from pathlib import Path
import argparse
import subprocess
import sys


def run(cmd, cwd):
    full_cmd = f"{sys.executable} {cmd}"
    print(">>>", full_cmd)
    subprocess.run(full_cmd, shell=True, check=True, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description="Batch run tree + branch dataset + similarity.")
    parser.add_argument("--pattern", type=str, default="ASOCA2020/Normal/Centerlines/Normal_*.vtp", help="Glob pattern for centerlines (used when no vtps are passed)")
    parser.add_argument("vtps", nargs="*", help="Optional explicit list of VTP files")
    parser.add_argument("--mask_dir", type=Path, default=Path("ASOCA2020/Normal/Annotations_nii"), help="Dir containing masks Normal_<id>.nii.gz")
    parser.add_argument("--start_offset", type=float, default=3.0, help="Branch start trim (mm)")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument(
        "--start_offset_percent",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of percents to trim branch starts (e.g., 0 2.5 5). Each percent produces its own output npz/dirs.",
    )
    parser.add_argument("--align", dest="align", action="store_true", help="Enable distance-map alignment (default).")
    parser.add_argument("--no_align", dest="align", action="store_false")
    parser.set_defaults(align=True)
    args = parser.parse_args()

    root = Path(".").resolve()
    if args.vtps:
        vtps = sorted([root / Path(p) for p in args.vtps])
    else:
        vtps = sorted(root.glob(args.pattern))
    if not vtps:
        print("No centerline files provided or matched pattern.")
        return

    for vtp in vtps:
        case = vtp.stem  # e.g., Normal_1
        mask = args.mask_dir / f"{case}.nii.gz"
        if not mask.exists():
            print(f"Mask not found for {case}: {mask}")
            continue
        out_root = root / "outputs" / case
        tree_json = out_root / "tree.json"
        branch_npz = out_root / "branch_dataset.npz"
        branch_dir = out_root / "branches"
        sim_dir = out_root / "similarity"
        out_root.mkdir(parents=True, exist_ok=True)

        # 1) tree
        run(f"scripts/build_centerline_tree.py --vtp {vtp} --out {tree_json} --case {case}", cwd=root)
        percents = args.start_offset_percent if args.start_offset_percent is not None else [None]
        for pct in percents:
            suffix = "" if pct is None else f"_pct{str(pct).replace('.', 'p')}"
            npz_path = branch_npz if pct is None else out_root / f"branch_dataset{suffix}.npz"
            bdir = branch_dir if pct is None else out_root / f"branches{suffix}"
            sdir = sim_dir if pct is None else out_root / f"similarity{suffix}"
            pct_arg = f"--start_offset_percent {pct}" if pct is not None else ""
            align_flag = "" if args.align else "--no_align"
            # 2) branch dataset
            run(
                f"scripts/build_branch_dataset.py --vtp {vtp} --mask {mask} --tree {tree_json} "
                f"--out {npz_path} --branch_dir {bdir} --case {case} "
                f"--start_offset {args.start_offset} {pct_arg} --K {args.K} --M {args.M} {align_flag}",
                cwd=root,
            )
            # 3) similarity
            run(f"scripts/branch_similarity.py --npz {npz_path} --out_dir {sdir} --pca_dim 8 --heatmap", cwd=root)

    print("Batch pipeline done.")


if __name__ == "__main__":
    main()
