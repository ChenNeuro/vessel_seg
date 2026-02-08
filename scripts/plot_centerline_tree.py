"""
Visualize centerline tree with branch ids and parent-child relations.

Reads:
- VTP centerlines (default: ASOCA2020/Normal/Centerlines/Normal_1.vtp)
- Tree JSON produced by build_centerline_tree.py (default: outputs/normal1_tree.json)

Outputs:
- 3D plot saved to outputs/normal1_tree_plot.png
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import vtk


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
    vtp_path = Path("ASOCA2020/Normal/Centerlines/Normal_1.vtp")
    tree_path = Path("outputs/normal1_tree.json")
    out_path = Path("outputs/normal1_tree_plot.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)

    branches = read_branches(vtp_path)
    tree = json.loads(tree_path.read_text())
    attachments = {b["branch_id"]: b["attachment"] for b in tree["branches"]}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.get_cmap("tab10")

    # Plot each branch
    for bid, coords in branches.items():
        c = colors(bid % 10)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=c, linewidth=2, label=f"{bid}")
        # mark start
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=c, s=20, marker="o")
        # label near start
        ax.text(coords[0, 0], coords[0, 1], coords[0, 2], f"{bid}", color=c, fontsize=8)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Normal_1 centerline tree\n(color=branch id; labels at starts)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved tree plot to {out_path}")


if __name__ == "__main__":
    main()
