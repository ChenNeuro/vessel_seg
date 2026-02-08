"""
Interactive 3D viewer for coronary centerline trees.

Defaults:
- Centerlines VTP: ASOCA2020/Normal/Centerlines/Normal_1.vtp
- Tree JSON: outputs/normal1_tree.json (built by build_centerline_tree.py)

Controls (PyVista):
- Left drag: rotate; right drag: pan; scroll: zoom; R: reset; S: save screenshot.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pyvista as pv


def read_branches(vtp_path: Path):
    mesh = pv.read(vtp_path)
    pts = np.array(mesh.points)
    raw = mesh.lines  # [n_pts, id0, id1, ..., n_pts, ...]
    branches = {}
    idx = 0
    i = 0
    while i < len(raw):
        n = raw[i]
        ids = raw[i + 1 : i + 1 + n]
        coords = pts[ids]
        branches[idx] = coords
        idx += 1
        i += n + 1
    return branches


def build_parent_lookup(tree_json: Path):
    data = json.loads(tree_json.read_text())
    parent = {}
    for b in data["branches"]:
        parent[b["branch_id"]] = b["attachment"]["parent"]
    return parent


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D view of centerline tree.")
    parser.add_argument("--vtp", type=Path, default=Path("ASOCA2020/Normal/Centerlines/Normal_1.vtp"))
    parser.add_argument("--tree", type=Path, default=Path("outputs/normal1_tree.json"))
    parser.add_argument("--tube_radius", type=float, default=0.8, help="Tube radius in mm")
    parser.add_argument("--label", action="store_true", help="Show branch id labels at starts")
    parser.add_argument("--offscreen", action="store_true", help="Use off-screen rendering (for headless/WSL).")
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="If set, save a screenshot to this path and exit.",
    )
    args = parser.parse_args()

    branches = read_branches(args.vtp)
    parent = build_parent_lookup(args.tree) if args.tree.exists() else {}

    colors = [
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
    plotter = pv.Plotter(off_screen=args.offscreen)

    for bid, coords in branches.items():
        color = colors[bid % len(colors)]
        poly = pv.PolyData(coords)
        n = coords.shape[0]
        # build polyline connectivity
        lines = np.hstack([[n], np.arange(n)])
        poly.lines = lines
        tube = poly.tube(radius=args.tube_radius)
        plotter.add_mesh(tube, color=color, name=f"branch_{bid}")
        if args.label:
            plotter.add_point_labels(
                coords[:1],
                [f"{bid}"],
                point_size=12,
                text_color=color,
                name=f"label_{bid}",
                shape_opacity=0.4,
            )

    # Optionally draw parent-child connectors
    for child, p in parent.items():
        if p is None or child not in branches or p not in branches:
            continue
        start = branches[child][0]
        parent_start = branches[p][0]
        line = pv.Line(parent_start, start)
        plotter.add_mesh(line, color="gray", line_width=2, opacity=0.5, name=f"edge_{p}_{child}")

    plotter.add_axes()
    plotter.add_bounding_box(color="white", opacity=0.2)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(title="Centerline Tree 3D Viewer", screenshot=str(args.screenshot), auto_close=True)
        print(f"Saved screenshot to {args.screenshot}")
    else:
        plotter.show(title="Centerline Tree 3D Viewer")


if __name__ == "__main__":
    main()
