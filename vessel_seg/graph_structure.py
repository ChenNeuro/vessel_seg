"""Core data models for coronary centerline graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class Branch:
    """Single vessel segment between two topological points."""

    id: int
    centerline: np.ndarray  # (N, 3) in world coordinates (mm)
    radii: Optional[np.ndarray] = None  # (N,) mean radius per point (mm)
    parent_id: Optional[int] = None
    child_ids: List[int] = field(default_factory=list)
    start_coord: Optional[tuple[int, int, int]] = None  # voxel coordinate of start node
    end_coord: Optional[tuple[int, int, int]] = None  # voxel coordinate of end node

    def length(self) -> float:
        """Return total arclength in mm."""
        pts = np.asarray(self.centerline, dtype=np.float32)
        if pts.shape[0] < 2:
            return 0.0
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(diffs.sum())

    def num_points(self) -> int:
        return int(np.asarray(self.centerline).shape[0])

    def sample_along(self, num_samples: int) -> np.ndarray:
        """Resample centerline to a fixed number of points using linear interpolation."""
        num_samples = max(int(num_samples), 2)
        pts = np.asarray(self.centerline, dtype=np.float32)
        if pts.shape[0] == 0:
            return np.zeros((num_samples, 3), dtype=np.float32)
        if pts.shape[0] == 1:
            return np.repeat(pts, num_samples, axis=0)

        s = np.zeros(pts.shape[0], dtype=np.float32)
        s[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        target = np.linspace(0.0, s[-1], num_samples, dtype=np.float32)
        resampled = np.vstack([np.interp(target, s, pts[:, dim]) for dim in range(3)]).T
        return resampled

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": int(self.id),
            "centerline": np.asarray(self.centerline, dtype=float).tolist(),
            "radii": None if self.radii is None else np.asarray(self.radii, dtype=float).tolist(),
            "parent_id": self.parent_id,
            "child_ids": [int(c) for c in self.child_ids],
            "start_coord": self.start_coord,
            "end_coord": self.end_coord,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Branch":
        return cls(
            id=int(data["id"]),
            centerline=np.asarray(data["centerline"], dtype=np.float32),
            radii=None if data.get("radii") is None else np.asarray(data["radii"], dtype=np.float32),
            parent_id=data.get("parent_id"),
            child_ids=[int(c) for c in data.get("child_ids", [])],
            start_coord=tuple(data["start_coord"]) if data.get("start_coord") is not None else None,
            end_coord=tuple(data["end_coord"]) if data.get("end_coord") is not None else None,
        )


class CoronaryTree:
    """Container for a single coronary tree (e.g., LCA or RCA)."""

    def __init__(self, root_id: Optional[int] = None) -> None:
        self.branches: Dict[int, Branch] = {}
        self.root_id: Optional[int] = root_id

    def add_branch(self, branch: Branch) -> None:
        if branch.id in self.branches:
            raise ValueError(f"Branch id {branch.id} already exists.")
        self.branches[branch.id] = branch

    def get_branch(self, branch_id: int) -> Branch:
        return self.branches[branch_id]

    def iter_branches(self) -> Iterable[Branch]:
        return self.branches.values()

    def num_branches(self) -> int:
        return len(self.branches)

    def total_length(self) -> float:
        return sum(br.length() for br in self.branches.values())

    def branching_degrees(self) -> Dict[int, int]:
        """Return a mapping branch_id -> number of children."""
        return {bid: len(branch.child_ids) for bid, branch in self.branches.items()}

    def to_dict(self) -> Dict[str, object]:
        return {
            "root_id": self.root_id,
            "branches": [br.to_dict() for br in self.branches.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CoronaryTree":
        tree = cls(root_id=data.get("root_id"))
        for br_data in data.get("branches", []):
            tree.add_branch(Branch.from_dict(br_data))
        return tree

    def to_networkx(self):
        """Convert to a networkx graph (optional dependency)."""
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("networkx is required for graph export.") from exc
        g = nx.DiGraph()
        for branch in self.branches.values():
            g.add_node(branch.id, length=branch.length())
            for child in branch.child_ids:
                g.add_edge(branch.id, child)
        return g
