"""
Plot centerlines for each folder in a branch tree (recursively), saving a PNG per folder.

Usage:
  python scripts/plot_branch_tree_centerlines.py --root outputs/Normal_1/branch_tree
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_centerlines_in_dir(folder: Path):
    paths = sorted(folder.glob("**/centerline.npy"))
    if not paths:
        return None
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, len(paths)))
    for p, c in zip(paths, colors):
        pts = np.load(p)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=c, label=str(p.parent.relative_to(folder)))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(folder.name)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=6)
    plt.tight_layout()
    out_png = folder / "centerlines.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Plot centerlines for each folder in branch tree.")
    parser.add_argument("--root", type=Path, required=True, help="Root folder of branch tree (containing branch_*).")
    args = parser.parse_args()

    for dirpath, dirnames, filenames in os.walk(args.root):
        folder = Path(dirpath)
        png = plot_centerlines_in_dir(folder)
        if png:
            print("Saved", png)


if __name__ == "__main__":
    main()
