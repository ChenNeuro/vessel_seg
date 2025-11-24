"""Simple analysis over saved coronary tree JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from vessel_seg.graph_structure import CoronaryTree


def load_tree(path: Path) -> CoronaryTree:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CoronaryTree.from_dict(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze coronary tree statistics.")
    parser.add_argument("--trees", nargs="+", required=True, help="Paths to tree JSON files.")
    args = parser.parse_args()

    trees: List[CoronaryTree] = [load_tree(Path(p)) for p in args.trees]
    if not trees:
        print("No trees provided.")
        return

    all_lengths = []
    all_degrees = []
    all_radii = []
    for tree in trees:
        for br in tree.iter_branches():
            all_lengths.append(br.length())
            all_degrees.append(len(br.child_ids))
            if br.radii is not None and br.radii.size:
                all_radii.extend(br.radii.tolist())

    def summarize(name: str, values: List[float]) -> None:
        if not values:
            print(f"{name}: no data")
            return
        arr = np.asarray(values, dtype=np.float32)
        print(f"{name}: mean={arr.mean():.2f}, std={arr.std():.2f}, min={arr.min():.2f}, max={arr.max():.2f}, n={arr.size}")

    summarize("Branch length (mm)", all_lengths)
    summarize("Branch degree", all_degrees)
    summarize("Radius (mm)", all_radii)


if __name__ == "__main__":
    main()
