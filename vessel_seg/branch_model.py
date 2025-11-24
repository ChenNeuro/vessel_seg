"""Branch-level statistical modeling interfaces."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from .graph_structure import Branch


class BranchShapeModel(ABC):
    """Abstract interface for branch shape priors."""

    @abstractmethod
    def fit(self, branches: Iterable[Branch]) -> None:
        """Learn parameters from observed branches."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, num_points: int) -> Branch:
        """Generate a synthetic branch centerline + radius profile."""
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, payload: dict) -> "BranchShapeModel":
        raise NotImplementedError


@dataclass
class LengthRadiusStats:
    length_mean: float = 0.0
    length_std: float = 1.0
    radius_mean: float = 1.0
    radius_std: float = 0.5


class SimpleLengthRadiusModel(BranchShapeModel):
    """Trivial model assuming Gaussian length and radius distributions."""

    def __init__(self, stats: Optional[LengthRadiusStats] = None, name: str = "simple_gaussian") -> None:
        self.name = name
        self.stats = stats or LengthRadiusStats()

    def fit(self, branches: Iterable[Branch]) -> None:
        lengths = []
        radii = []
        for br in branches:
            lengths.append(br.length())
            if br.radii is not None and br.radii.size:
                radii.extend(br.radii.tolist())
        if lengths:
            self.stats.length_mean = float(np.mean(lengths))
            self.stats.length_std = float(np.std(lengths) + 1e-3)
        if radii:
            self.stats.radius_mean = float(np.mean(radii))
            self.stats.radius_std = float(np.std(radii) + 1e-3)

    def sample(self, num_points: int) -> Branch:
        rng = np.random.default_rng()
        length = max(rng.normal(self.stats.length_mean, self.stats.length_std), 1e-2)
        radius = max(rng.normal(self.stats.radius_mean, self.stats.radius_std), 1e-2)
        positions = np.linspace(0.0, length, num_points, dtype=np.float32)
        pts = np.stack([positions, np.zeros_like(positions), np.zeros_like(positions)], axis=1)
        radii = np.full(num_points, radius, dtype=np.float32)
        return Branch(id=-1, centerline=pts, radii=radii)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stats": {
                "length_mean": self.stats.length_mean,
                "length_std": self.stats.length_std,
                "radius_mean": self.stats.radius_mean,
                "radius_std": self.stats.radius_std,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SimpleLengthRadiusModel":
        stats_data = payload.get("stats", {})
        stats = LengthRadiusStats(
            length_mean=float(stats_data.get("length_mean", 0.0)),
            length_std=float(stats_data.get("length_std", 1.0)),
            radius_mean=float(stats_data.get("radius_mean", 1.0)),
            radius_std=float(stats_data.get("radius_std", 0.5)),
        )
        return cls(stats=stats, name=payload.get("name", "simple_gaussian"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleLengthRadiusModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
