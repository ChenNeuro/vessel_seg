"""
Organize branches into a tree-structured folder layout based on tree.json.

Usage:
  python scripts/organize_branch_tree.py \
    --tree outputs/Normal_1/tree.json \
    --branches_dir outputs/Normal_1/branches \
    --out_dir outputs/Normal_1/branch_tree \
    [--name_map mapping.json]

The script reads tree.json (branch_id + parent) and creates a directory per branch:
  out_dir/
    branch_2/
      centerline.npy
      radii.npz (if exists)
      mesh.vtp (if exists)
      branch_0/
        ...

If a name_map is provided (JSON mapping branch_id -> custom name, e.g. "LAD"),
the folder name becomes that label (prefixed with branch_id to keep uniqueness).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def load_tree(tree_path: Path) -> Dict[int, int | None]:
    data = json.loads(tree_path.read_text())
    parent_map = {}
    for b in data.get("branches", []):
        parent = b.get("attachment", {}).get("parent", None)
        parent_map[int(b["branch_id"])] = None if parent is None else int(parent)
    return parent_map


def build_children_map(parent_map: Dict[int, int | None]) -> Dict[int | None, List[int]]:
    children: Dict[int | None, List[int]] = {}
    for bid, parent in parent_map.items():
        children.setdefault(parent, []).append(bid)
    return children


def make_branch_folder_name(bid: int, name_map: Dict[str, str] | None) -> str:
    label = name_map.get(str(bid)) if name_map else None
    return f"branch_{bid}" if not label else f"{bid}_{label}"


def copy_branch_files(src_dir: Path, dst_dir: Path):
    for fname in ["centerline.npy", "radii.npz", "mesh.vtp"]:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def organize(tree_path: Path, branches_dir: Path, out_dir: Path, name_map: Dict[str, str] | None = None):
    parent_map = load_tree(tree_path)
    children_map = build_children_map(parent_map)

    def dfs(bid: int, parent_dst: Path):
        folder_name = make_branch_folder_name(bid, name_map)
        dst = parent_dst / folder_name
        dst.mkdir(parents=True, exist_ok=True)
        src_branch = branches_dir / f"branch_{bid}"
        if src_branch.exists():
            copy_branch_files(src_branch, dst)
        # recurse
        for child in children_map.get(bid, []):
            dfs(child, dst)

    out_dir.mkdir(parents=True, exist_ok=True)
    roots = children_map.get(None, [])
    for r in roots:
        dfs(r, out_dir)
    print(f"Organized {len(parent_map)} branches into {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Organize branch files into a tree-structured folder layout.")
    parser.add_argument("--tree", type=Path, required=True, help="tree.json with branch attachments")
    parser.add_argument("--branches_dir", type=Path, required=True, help="Directory containing branch_<id>/centerline.npy etc.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output root directory for organized tree")
    parser.add_argument("--name_map", type=Path, default=None, help="Optional JSON mapping branch_id -> label (e.g., {\"2\": \"LAD\"})")
    args = parser.parse_args()

    name_map = json.loads(args.name_map.read_text()) if args.name_map else None
    organize(args.tree, args.branches_dir, args.out_dir, name_map)


if __name__ == "__main__":
    main()
