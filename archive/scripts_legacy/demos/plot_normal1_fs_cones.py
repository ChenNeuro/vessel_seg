"""
Plot coronary branches of ASOCA2020 Normal_1 in local Frenet-Serret (FS) coordinates.

For each polyline in the provided VTK centerline file, we:
1) Compute arc-length s along the line (tangent direction of FS).
2) Read per-point MaximumInscribedSphereRadius as local vessel radius.
3) Project the vessel to a 2D FS view (s on x-axis, ±radius on y-axis) so each branch
   appears like a cone/frustum. Branches are sorted by length and stacked vertically.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import vtk


def read_centerlines(path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def extract_branches(polydata: vtk.vtkPolyData):
    points = polydata.GetPoints()
    radius_arr = polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")

    branches = []
    lines = polydata.GetLines()
    lines.InitTraversal()
    cell_id = 0

    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        if ids.GetNumberOfIds() < 2:
            cell_id += 1
            continue

        coords = np.array([points.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
        radii = (
            np.array([radius_arr.GetTuple1(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
            if radius_arr is not None
            else np.ones(len(coords))
        )

        # Arc-length parameter s
        diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(diffs)])
        total_len = float(s[-1])
        centroid = coords.mean(axis=0)

        branches.append(
            {
                "cell_id": cell_id,
                "s": s,
                "r": radii,
                "length": total_len,
                "centroid": centroid,
            }
        )
        cell_id += 1

    return branches


def plot_cones(branches, out_path: Path):
    # Sort long → short for a chromosome-like layout
    branches = sorted(branches, key=lambda b: b["length"], reverse=True)
    y_gap = 6.0  # vertical separation between branches (mm), increase spacing for clarity

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for i, b in enumerate(branches):
        s = b["s"]
        r = b["r"]
        y_offset = i * y_gap

        upper = y_offset + r
        lower = y_offset - r
        plt.fill_between(s, lower, upper, alpha=0.6, label=f"B{i+1} (cell {b['cell_id']})")
        plt.text(
            s[-1] + 5,
            y_offset,
            f"{b['length']:.1f} mm",
            va="center",
            fontsize=8,
        )

    plt.xlabel("Arc length s (mm) along FS tangent")
    plt.ylabel("± radius in FS normal-binormal plane (offset per branch)")
    plt.title("Normal_1 branches in FS coordinates (cone-like projection)")
    plt.yticks([])  # offsets carry the separation; ticks add clutter
    plt.legend(fontsize=7, loc="upper right", ncol=2)
    # Use equal aspect so x (s) and y (radius/offset) share the same physical scale (mm).
    ax.set_aspect("equal", adjustable="box")
    # Ensure a bit of margin on x so labels fit after equal aspect
    x_max = max(max(b["s"]) for b in branches) if branches else 0
    ax.set_xlim(left=0, right=x_max * 1.1 if x_max else 1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def split_branches_by_x(branches, threshold):
    """Split branches into left/right by centroid x vs threshold."""
    left, right = [], []
    for b in branches:
        if b["centroid"][0] <= threshold:
            left.append(b)
        else:
            right.append(b)
    return left, right


def main():
    vtp_path = Path("ASOCA2020/Normal/Centerlines/Normal_1.vtp")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path_all = out_dir / "normal1_fs_cones.png"
    out_path_left = out_dir / "normal1_left_fs_cones.png"
    out_path_right = out_dir / "normal1_right_fs_cones.png"

    poly = read_centerlines(vtp_path)
    branches = extract_branches(poly)
    if not branches:
        raise RuntimeError("No valid polylines found in centerline file.")

    # Global x-threshold: mean of all branch centroids
    x_thresh = np.mean([b["centroid"][0] for b in branches])

    plot_cones(branches, out_path_all)
    left, right = split_branches_by_x(branches, x_thresh)
    if left:
        plot_cones(left, out_path_left)
    if right:
        plot_cones(right, out_path_right)

    print(f"Saved FS cone plot (all) to {out_path_all}")
    print(f"Split by centroid x <= {x_thresh:.2f} mm -> left: {len(left)} branches, right: {len(right)} branches")
    if left:
        print(f"Left plot: {out_path_left}")
    if right:
        print(f"Right plot: {out_path_right}")


if __name__ == "__main__":
    main()
