"""
Plot Normal_1 centerlines over ground-truth segmentation in a single 3D Matplotlib figure.

- Reads centerlines from VTP (physical coordinates).
- Reads segmentation mask (NRRD), extracts an isosurface via marching cubes (step decimated).
- Renders both with Matplotlib 3D (headless-friendly).

Usage:
  python scripts/plot_centerline_vs_gt.py \
      --vtp ASOCA2020/Normal/Centerlines/Normal_1.vtp \
      --mask ASOCA2020/Normal/Annotations/Normal_1.nrrd \
      --out outputs/normal1_centerline_vs_gt.png
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import SimpleITK as sitk
import vtk


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


def read_surface(mask_path: Path, step_size: int = 2):
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    spacing = img.GetSpacing()  # (sx, sy, sz)
    origin = img.GetOrigin()  # (ox, oy, oz)

    if arr.max() == 0:
        raise ValueError("Mask is empty.")
    # marching cubes expects spacing per axis order of array (z,y,x)
    verts, faces, _, _ = measure.marching_cubes(arr > 0, level=0.5, spacing=(spacing[2], spacing[1], spacing[0]), step_size=step_size)
    # convert to physical coordinates (x,y,z)
    x = origin[0] + verts[:, 2]
    y = origin[1] + verts[:, 1]
    z = origin[2] + verts[:, 0]
    verts_phys = np.column_stack([x, y, z])
    return verts_phys, faces


def plot_overlay(branches, verts, faces, out_path: Path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # plot segmentation surface
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        faces,
        verts[:, 2],
        color="lightgray",
        alpha=0.3,
        linewidth=0,
    )

    # plot centerlines
    for bid, coords in branches.items():
        color = TAB10[bid % len(TAB10)]
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=2)
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, s=15)
        ax.text(coords[0, 0], coords[0, 1], coords[0, 2], str(bid), color=color, fontsize=8)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Normal_1 centerlines vs ground-truth")
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved overlay to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Overlay centerlines and ground-truth segmentation.")
    parser.add_argument("--vtp", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Output PNG (default outputs/<case>/vis/centerline_vs_gt.png)")
    parser.add_argument("--case", type=str, default="Normal_1", help="Case name for default paths")
    parser.add_argument("--step", type=int, default=2, help="marching cubes step_size to decimate surface")
    args = parser.parse_args()

    branches = read_centerlines(args.vtp)
    verts, faces = read_surface(args.mask, step_size=args.step)
    out_path = args.out or Path(f"outputs/{args.case}/vis/centerline_vs_gt.png")
    plot_overlay(branches, verts, faces, out_path)


if __name__ == "__main__":
    main()
