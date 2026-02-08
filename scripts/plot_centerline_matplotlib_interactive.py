"""
Interactive Matplotlib 3D viewer for centerlines (optionally with segmentation surface).

Controls (Matplotlib 3D backend):
- 鼠标左键拖拽：旋转
- 鼠标滚轮：缩放
- 按键 s：保存当前视角截图到 --screenshot
- 按键 r：重置视角

Usage:
  conda activate vessel_seg
  python scripts/plot_centerline_matplotlib_interactive.py \
      --vtp ASOCA2020/Normal/Centerlines/Normal_1.vtp \
      --mask ASOCA2020/Normal/Annotations/Normal_1.nrrd \
      --screenshot outputs/normal1_centerline_interactive.png

If you omit --mask,只显示中心线。需要有图形界面后端（本地桌面或远程X/VSCode LiveShare）。
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import vtk

try:
    import SimpleITK as sitk
    from skimage import measure
    HAS_MASK_DEPS = True
except ImportError:
    HAS_MASK_DEPS = False


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


def read_mask_surface(mask_path: Path, step_size: int = 2):
    if not HAS_MASK_DEPS:
        raise RuntimeError("SimpleITK / skimage 未安装，无法读取 mask 表面")
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = img.GetSpacing()  # (sx, sy, sz)
    origin = img.GetOrigin()
    if arr.max() == 0:
        raise ValueError("Mask is empty.")
    verts, faces, _, _ = measure.marching_cubes(arr > 0, level=0.5, spacing=(spacing[2], spacing[1], spacing[0]), step_size=step_size)
    # convert to physical (x,y,z)
    x = origin[0] + verts[:, 2]
    y = origin[1] + verts[:, 1]
    z = origin[2] + verts[:, 0]
    verts_phys = np.column_stack([x, y, z])
    return verts_phys, faces


def main():
    parser = argparse.ArgumentParser(description="Interactive Matplotlib 3D for centerlines (+ optional mask surface).")
    parser.add_argument("--vtp", type=Path, required=True)
    parser.add_argument("--mask", type=Path, default=None, help="Optional segmentation mask (.nrrd) to overlay surface.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Save screenshot on key 's'.")
    parser.add_argument("--step", type=int, default=2, help="Marching cubes step_size (lower=denser surface).")
    args = parser.parse_args()

    branches = read_centerlines(args.vtp)
    verts, faces = None, None
    if args.mask:
        verts, faces = read_mask_surface(args.mask, step_size=args.step)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if verts is not None and faces is not None:
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
            color="lightgray",
            alpha=0.25,
            linewidth=0,
        )

    for bid, coords in branches.items():
        color = TAB10[bid % len(TAB10)]
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=2)
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, s=15)
        ax.text(coords[0, 0], coords[0, 1], coords[0, 2], str(bid), color=color, fontsize=8)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Interactive centerlines (Matplotlib 3D)")
    ax.view_init(elev=20, azim=45)
    fig.tight_layout()

    default_view = (ax.elev, ax.azim)

    def on_key(event):
        if event.key == "s" and args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.screenshot, dpi=200)
            print(f"Saved screenshot to {args.screenshot}")
        if event.key == "r":
            ax.view_init(elev=default_view[0], azim=default_view[1])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
