"""
Compare provided centerlines (VTP) with centerlines extracted from ground-truth mask.

Steps:
- Read centerline polylines from VTP.
- Read segmentation mask (NRRD), skeletonize_3d to get a voxel skeleton.
- Convert skeleton voxels to physical coordinates (scatter) for visual comparison.
- Render both in a Matplotlib 3D figure (headless-friendly).

Usage:
  conda activate vessel_seg
  python scripts/compare_centerline_vs_gt_centerline.py \
      --vtp ASOCA2020/Normal/Centerlines/Normal_1.vtp \
      --mask ASOCA2020/Normal/Annotations/Normal_1.nrrd \
      --out outputs/normal1_centerline_vs_gt_centerline.png
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import vtk
import SimpleITK as sitk
try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    # Fallback for versions where skeletonize_3d is not exported
    try:
        from skimage.morphology._skeletonize_3d import skeletonize_3d  # type: ignore
    except ImportError as e:
        raise ImportError("skeletonize_3d not available in this scikit-image version.") from e

TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def read_centerlines(vtp_path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    lines = poly.GetLines()
    lines.InitTraversal()
    branches = {}
    cid = 0
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        if ids.GetNumberOfIds() < 2:
            cid += 1
            continue
        coords = np.array([pts.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
        branches[cid] = coords
        cid += 1
    return branches


def skeleton_from_mask(mask_path: Path):
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)  # z,y,x, indices in image grid
    mask = arr > 0
    if mask.max() == 0:
        raise ValueError("Mask is empty.")
    skel = skeletonize_3d(mask)
    coords_idx = np.argwhere(skel)  # (N, 3) in z,y,x
    if coords_idx.size == 0:
        raise ValueError("Skeleton is empty after skeletonize_3d.")
    # convert index -> physical using ITK direction/origin/spacing to honor orientation
    pts = [
        img.TransformIndexToPhysicalPoint(
            (int(idx[2]), int(idx[1]), int(idx[0]))
        )
        for idx in coords_idx
    ]
    return np.array(pts)


def plot_comparison(branches, skel_pts, out_path: Path, show: bool = False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # plot provided centerlines
    for bid, coords in branches.items():
        color = TAB10[bid % len(TAB10)]
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=2, label=f"VTP {bid}")
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, s=15)

    # plot skeleton points
    ax.scatter(skel_pts[:, 0], skel_pts[:, 1], skel_pts[:, 2], color="black", s=2, alpha=0.4, label="GT skeleton")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Centerline vs GT skeleton")
    ax.view_init(elev=20, azim=45)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved comparison to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare provided centerlines vs skeletonized GT mask.")
    parser.add_argument("--vtp", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Output PNG (default outputs/<case>/vis/centerline_vs_gt_centerline.png)")
    parser.add_argument("--case", type=str, default="Normal_1", help="Case name for default paths")
    parser.add_argument("--show", action="store_true", help="Show interactive Matplotlib window.")
    args = parser.parse_args()

    branches = read_centerlines(args.vtp)
    skel_pts = skeleton_from_mask(args.mask)

    out_path = args.out or Path(f"outputs/{args.case}/vis/centerline_vs_gt_centerline.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(branches, skel_pts, out_path, show=args.show)


if __name__ == "__main__":
    main()
