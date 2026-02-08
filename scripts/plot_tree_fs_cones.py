"""
Plot cone-style FS projections for the trimmed centerline tree (with lambda/parent info).

This recomputes trimming and tree attachments (same logic as build_centerline_tree.py),
then renders each branch as a (s, ±r) band, sorted by length. Labels include branch id
and parent/lambda for quick inspection.

Usage:
  python scripts/plot_tree_fs_cones.py \
      --vtp ASOCA2020/Normal/Centerlines/Normal_1.vtp \
      --out outputs/normal1_tree_fs_cones.png
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Local imports (assumes running from repo root)
import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from build_centerline_tree import (
    read_branches,
    cluster_starts,
    trim_overlap,
    build_tree,
    START_CLUSTER_EPS,
    OVERLAP_DIST_EPS,
)


def trim_branches(branches):
    """Trim overlapping prefixes within each start-cluster."""
    clusters = cluster_starts(branches, eps=START_CLUSTER_EPS)
    trimmed = []
    for cluster in clusters:
        cbranches = [b for b in branches if b.branch_id in cluster]
        root_branch = max(cbranches, key=lambda b: b.length)
        trimmed.append(root_branch)
        for br in cbranches:
            if br.branch_id == root_branch.branch_id:
                continue
            new_br = trim_overlap(br, root_branch, eps=OVERLAP_DIST_EPS)
            if new_br is not None:
                trimmed.append(new_br)
            # fully overlapping branches are dropped
    return trimmed


def plot_cones(branches, attachments, out_path: Path, y_gap: float = 6.0):
    # Sort by length descending for chromosome-like layout
    branches_sorted = sorted(branches, key=lambda b: b.length, reverse=True)
    plt.figure(figsize=(11, 8))
    ax = plt.gca()

    for i, b in enumerate(branches_sorted):
        coords = b.coords
        r = b.radius if b.radius is not None else np.ones(len(coords))
        # arc-length s
        diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(diffs)])
        y_offset = i * y_gap
        upper = y_offset + r
        lower = y_offset - r
        ax.fill_between(s, lower, upper, alpha=0.7, label=str(b.branch_id))

        att = attachments.get(b.branch_id)
        parent = att.parent if att else None
        lam = att.lambda_pos if att else None
        label = f"id={b.branch_id}"
        if parent is not None:
            label += f" p={parent}"
            if lam is not None:
                label += f" λ={lam:.2f}"
        ax.text(s[-1] + 5, y_offset, label, va="center", fontsize=8)

    ax.set_xlabel("Arc length s (mm) along branch")
    ax.set_ylabel("± radius (mm), stacked per branch")
    ax.set_title("Centerline tree FS cones (trimmed, with parent/λ)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks([])
    # Keep legend minimal (only ids)
    ax.legend(title="Branch id", fontsize=7, ncol=4, loc="upper right")
    x_max = max(max(np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(b.coords, axis=0), axis=1))])) for b in branches_sorted)
    ax.set_xlim(left=0, right=x_max * 1.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved FS cones to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot FS cones for centerline tree with lambda/parent.")
    parser.add_argument("--vtp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Output PNG (default outputs/<case>/vis/tree_fs_cones.png)")
    parser.add_argument("--case", type=str, default="Normal_1", help="Case name for default paths")
    parser.add_argument("--gap", type=float, default=6.0, help="Vertical gap between branches (mm)")
    args = parser.parse_args()

    branches = read_branches(args.vtp)
    branches = trim_branches(branches)
    roots, attachments = build_tree(branches)

    out_path = args.out or Path(f"outputs/{args.case}/vis/tree_fs_cones.png")
    print(f"Roots: {roots}")
    plot_cones(branches, {k: v for k, v in attachments.items()}, out_path, y_gap=args.gap)


if __name__ == "__main__":
    main()
