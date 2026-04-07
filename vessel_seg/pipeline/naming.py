"""Canonical cross-case branch naming for repaired centerline trees."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class CanonicalBranchName:
    """单个分支的稳定命名结果。"""

    branch_id: int
    canonical_name: str
    canonical_parent_name: str | None
    system_name: str
    system_guess: str | None
    root_id: int
    depth: int
    sibling_index: int | None
    subtree_branch_count: int
    subtree_total_length_mm: float


@dataclass(frozen=True)
class NamingSystem:
    """单个根系统的命名摘要。"""

    root_id: int
    system_name: str
    system_guess: str | None
    branch_count: int
    subtree_total_length_mm: float


@dataclass(frozen=True)
class CanonicalNamingResult:
    """整棵树的命名结果。"""

    version: str
    systems: tuple[NamingSystem, ...]
    branches: tuple[CanonicalBranchName, ...]

    def branch_map(self) -> dict[int, CanonicalBranchName]:
        return {branch.branch_id: branch for branch in self.branches}

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "systems": [asdict(system) for system in self.systems],
            "branches": [asdict(branch) for branch in self.branches],
            "branch_name_map": {str(branch.branch_id): branch.canonical_name for branch in self.branches},
        }


def _attachment_parent(branch: dict) -> int | None:
    attachment = branch.get("attachment") or {}
    parent = attachment.get("parent")
    return None if parent is None else int(parent)


def _branch_length(branch: dict) -> float:
    return float(branch.get("length_mm") or 0.0)


def _branch_centroid(branch: dict) -> tuple[float, float, float]:
    centroid = branch.get("centroid") or [0.0, 0.0, 0.0]
    return float(centroid[0]), float(centroid[1]), float(centroid[2])


def _compute_children(branches_by_id: dict[int, dict]) -> dict[int, list[int]]:
    children = {branch_id: [] for branch_id in branches_by_id}
    for branch_id, branch in branches_by_id.items():
        parent = _attachment_parent(branch)
        if parent is not None and parent in children:
            children[parent].append(branch_id)
    return children


def _compute_subtree_stats(
    branch_id: int,
    branches_by_id: dict[int, dict],
    children: dict[int, list[int]],
    cache: dict[int, tuple[int, float]],
) -> tuple[int, float]:
    cached = cache.get(branch_id)
    if cached is not None:
        return cached

    count = 1
    total_length = _branch_length(branches_by_id[branch_id])
    for child_id in children.get(branch_id, []):
        child_count, child_total_length = _compute_subtree_stats(child_id, branches_by_id, children, cache)
        count += child_count
        total_length += child_total_length
    result = (count, total_length)
    cache[branch_id] = result
    return result


def _root_sort_key(
    branch_id: int,
    branches_by_id: dict[int, dict],
    subtree_stats: dict[int, tuple[int, float]],
) -> tuple[float, ...]:
    branch_count, total_length = subtree_stats[branch_id]
    cx, cy, cz = _branch_centroid(branches_by_id[branch_id])
    return (-branch_count, -total_length, -cx, -cy, -cz, float(branch_id))


def _child_sort_key(
    parent_id: int,
    child_id: int,
    branches_by_id: dict[int, dict],
    subtree_stats: dict[int, tuple[int, float]],
) -> tuple[float, ...]:
    child = branches_by_id[child_id]
    parent = branches_by_id[parent_id]
    child_count, child_total_length = subtree_stats[child_id]
    cx, cy, cz = _branch_centroid(child)
    px, py, pz = _branch_centroid(parent)
    rel_x = cx - px
    rel_y = cy - py
    rel_z = cz - pz
    azimuth = math.atan2(rel_y, rel_x)
    elevation = math.atan2(rel_z, math.hypot(rel_x, rel_y) + 1e-8)
    return (
        -child_count,
        -child_total_length,
        -_branch_length(child),
        azimuth,
        elevation,
        -cx,
        -cy,
        -cz,
        float(child_id),
    )


def assign_canonical_branch_names(tree_payload: dict[str, object]) -> CanonicalNamingResult:
    """为 tree.json 生成跨病例可复用的 canonical names。"""
    branches = list(tree_payload.get("branches", []))
    branches_by_id = {int(branch["branch_id"]): branch for branch in branches}
    children = _compute_children(branches_by_id)
    roots = [int(root_id) for root_id in tree_payload.get("roots", [])]
    if not roots:
        roots = [branch_id for branch_id, branch in branches_by_id.items() if _attachment_parent(branch) is None]

    subtree_stats: dict[int, tuple[int, float]] = {}
    for branch_id in branches_by_id:
        _compute_subtree_stats(branch_id, branches_by_id, children, subtree_stats)

    ordered_roots = sorted(roots, key=lambda branch_id: _root_sort_key(branch_id, branches_by_id, subtree_stats))
    systems: list[NamingSystem] = []
    root_system_names: dict[int, tuple[str, str | None]] = {}
    for index, root_id in enumerate(ordered_roots):
        branch_count, total_length = subtree_stats[root_id]
        if len(ordered_roots) == 1:
            system_name = "SYS_A"
            system_guess = "MAIN"
        elif len(ordered_roots) == 2:
            system_name = "SYS_A" if index == 0 else "SYS_B"
            system_guess = "LCA" if index == 0 else "RCA"
        else:
            if index == 0:
                system_name = "SYS_A"
                system_guess = "LCA"
            elif index == 1:
                system_name = "SYS_B"
                system_guess = "RCA"
            else:
                system_name = f"AUX_{index - 1:02d}"
                system_guess = None
        root_system_names[root_id] = (system_name, system_guess)
        systems.append(
            NamingSystem(
                root_id=root_id,
                system_name=system_name,
                system_guess=system_guess,
                branch_count=branch_count,
                subtree_total_length_mm=total_length,
            )
        )

    named_branches: list[CanonicalBranchName] = []

    def visit(
        branch_id: int,
        *,
        root_id: int,
        system_name: str,
        system_guess: str | None,
        canonical_name: str,
        canonical_parent_name: str | None,
        depth: int,
        sibling_index: int | None,
    ) -> None:
        branch_count, total_length = subtree_stats[branch_id]
        named_branches.append(
            CanonicalBranchName(
                branch_id=branch_id,
                canonical_name=canonical_name,
                canonical_parent_name=canonical_parent_name,
                system_name=system_name,
                system_guess=system_guess,
                root_id=root_id,
                depth=depth,
                sibling_index=sibling_index,
                subtree_branch_count=branch_count,
                subtree_total_length_mm=total_length,
            )
        )
        ordered_children = sorted(
            children.get(branch_id, []),
            key=lambda child_id: _child_sort_key(branch_id, child_id, branches_by_id, subtree_stats),
        )
        for child_index, child_id in enumerate(ordered_children, start=1):
            visit(
                child_id,
                root_id=root_id,
                system_name=system_name,
                system_guess=system_guess,
                canonical_name=f"{canonical_name}.{child_index:02d}",
                canonical_parent_name=canonical_name,
                depth=depth + 1,
                sibling_index=child_index,
            )

    for root_id in ordered_roots:
        system_name, system_guess = root_system_names[root_id]
        visit(
            root_id,
            root_id=root_id,
            system_name=system_name,
            system_guess=system_guess,
            canonical_name=system_name,
            canonical_parent_name=None,
            depth=0,
            sibling_index=None,
        )

    named_branches.sort(key=lambda branch: branch.branch_id)
    return CanonicalNamingResult(
        version="canonical_branch_naming_v1",
        systems=tuple(systems),
        branches=tuple(named_branches),
    )


def enrich_tree_with_canonical_names(tree_payload: dict[str, object]) -> dict[str, object]:
    """将 canonical naming 结果写回 tree payload。"""
    result = assign_canonical_branch_names(tree_payload)
    branch_map = result.branch_map()
    enriched = dict(tree_payload)
    enriched_branches = []
    for branch in list(tree_payload.get("branches", [])):
        branch_id = int(branch["branch_id"])
        naming = branch_map[branch_id]
        enriched_branch = dict(branch)
        enriched_branch["naming"] = asdict(naming)
        enriched_branches.append(enriched_branch)
    enriched["branches"] = enriched_branches
    enriched["canonical_naming"] = result.to_dict()
    return enriched


def apply_canonical_naming(tree_json_path: Path, branch_names_json_path: Path | None = None) -> CanonicalNamingResult:
    """读取 tree.json，写回命名增强后的 tree，并导出 name map。"""
    payload = json.loads(tree_json_path.read_text(encoding="utf-8"))
    result = assign_canonical_branch_names(payload)
    enriched = enrich_tree_with_canonical_names(payload)
    tree_json_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    if branch_names_json_path is not None:
        branch_names_json_path.parent.mkdir(parents=True, exist_ok=True)
        branch_names_json_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return result
