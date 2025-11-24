"""Coronary tree prior placeholders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .graph_structure import CoronaryTree


@dataclass
class TreePriorStats:
    mean_branch_count: float = 0.0
    std_branch_count: float = 1.0
    mean_total_length: float = 0.0
    std_total_length: float = 1.0


class CoronaryTreePrior:
    """Trivial prior assuming independent Gaussians over branch count and total length."""

    def __init__(self, stats: Optional[TreePriorStats] = None, name: str = "independent_gaussian") -> None:
        self.name = name
        self.stats = stats or TreePriorStats()

    def fit(self, trees: Iterable[CoronaryTree]) -> None:
        trees = list(trees)
        if not trees:
            return
        counts = [tree.num_branches() for tree in trees]
        lengths = [tree.total_length() for tree in trees]
        self.stats.mean_branch_count = float(sum(counts) / len(counts))
        self.stats.mean_total_length = float(sum(lengths) / len(lengths))
        # simple std estimates
        self.stats.std_branch_count = float((sum((c - self.stats.mean_branch_count) ** 2 for c in counts) / len(counts)) ** 0.5 + 1e-3)
        self.stats.std_total_length = float((sum((l - self.stats.mean_total_length) ** 2 for l in lengths) / len(lengths)) ** 0.5 + 1e-3)

    def sample(self) -> CoronaryTree:
        # TODO: implement generative sampling of topology and branch shapes.
        return CoronaryTree()

    def log_prob(self, tree: CoronaryTree) -> float:
        """Return a placeholder compatibility score."""
        # TODO: plug in real likelihood once distributions are defined.
        return 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stats": {
                "mean_branch_count": self.stats.mean_branch_count,
                "std_branch_count": self.stats.std_branch_count,
                "mean_total_length": self.stats.mean_total_length,
                "std_total_length": self.stats.std_total_length,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CoronaryTreePrior":
        stats_data = payload.get("stats", {})
        stats = TreePriorStats(
            mean_branch_count=float(stats_data.get("mean_branch_count", 0.0)),
            std_branch_count=float(stats_data.get("std_branch_count", 1.0)),
            mean_total_length=float(stats_data.get("mean_total_length", 0.0)),
            std_total_length=float(stats_data.get("std_total_length", 1.0)),
        )
        return cls(stats=stats, name=payload.get("name", "independent_gaussian"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CoronaryTreePrior":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
