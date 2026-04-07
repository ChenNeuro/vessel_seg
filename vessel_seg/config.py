"""Project-wide path resolution and repository layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _read_pointer_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical repository paths for the rebuilt engineering layout."""

    root: Path
    data_dir: Path
    external_data_dir: Path
    outputs_dir: Path
    legacy_outputs_dir: Path
    docs_dir: Path
    scripts_dir: Path
    package_dir: Path
    tests_dir: Path
    asoca_root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        root_path = Path(root).expanduser().resolve()
        data_dir = root_path / "data"
        external_data_dir = data_dir / "external"
        pointer_path = external_data_dir / "asoca2020.path"
        env_path = os.environ.get("VESSEL_SEG_ASOCA_ROOT")
        if env_path:
            asoca_root = Path(env_path).expanduser().resolve()
        else:
            asoca_root = _read_pointer_file(pointer_path) or (root_path.parent / "ASOCA2020").resolve()

        return cls(
            root=root_path,
            data_dir=data_dir,
            external_data_dir=external_data_dir,
            outputs_dir=root_path / "outputs_reorganized",
            legacy_outputs_dir=root_path / "outputs",
            docs_dir=root_path / "docs",
            scripts_dir=root_path / "scripts",
            package_dir=root_path / "vessel_seg",
            tests_dir=root_path / "tests",
            asoca_root=asoca_root,
        )

    def case_output_dir(self, case_id: str) -> Path:
        return self.outputs_dir / "cases" / case_id

