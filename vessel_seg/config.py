"""Project-wide configuration helpers and default paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    """Container for frequently used repository paths."""

    root: Path
    data_dir: Path
    outputs_dir: Path
    models_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        """Instantiate paths relative to the repository root."""
        root_path = Path(root).expanduser().resolve()
        return cls(
            root=root_path,
            data_dir=root_path / "data",
            outputs_dir=root_path / "outputs",
            models_dir=root_path / "models",
        )
