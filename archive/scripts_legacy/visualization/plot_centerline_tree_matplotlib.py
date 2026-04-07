"""
Headless 3D plot of centerline tree using Matplotlib (no OpenGL).

Reads:
- VTP centerlines (default: ASOCA2020/Normal/Centerlines/Normal_1.vtp)
- Tree JSON (default: outputs/normal1_tree.json) to draw parent-child connectors.

Outputs:
- PNG saved to outputs/normal1_tree_matplotlib.png
"""

from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
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


def read_branches(vtp_path: Path):
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


def main():
    parser = argparse.ArgumentParser(description="Matplotlib 3D plot of centerline tree.")
    parser.add_argument("--vtp", type=Path, default=Path("ASOCA2020/Normal/Centerlines/Normal_1.vtp"))
    parser.add_argument("--tree", type=Path, default=Path("outputs/normal1_tree.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/normal1_tree_matplotlib.png"))
    args = parser.parse_args()

    branches = read_branches(args.vtp)
    parent = {}
    if args.tree.exists():
        data = json.loads(args.tree.read_text())
        for b in data["branches"]:
            parent[b["branch_id"]] = b["attachment"]["parent"]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for bid, coords in branches.items():
        color = TAB10[bid % len(TAB10)]
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=1.8)
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, s=20)
        ax.text(coords[0, 0], coords[0, 1], coords[0, 2], str(bid), color=color, fontsize=8)

    # Draw parent-child connectors (straight line between starts)
    for child, p in parent.items():
        if p is None or child not in branches or p not in branches:
            continue
        c0 = branches[child][0]
        p0 = branches[p][0]
        ax.plot([p0[0], c0[0]], [p0[1], c0[1]], [p0[2], c0[2]], color="gray", linewidth=1, alpha=0.6, linestyle="--")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Centerline tree (Matplotlib 3D)")
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
