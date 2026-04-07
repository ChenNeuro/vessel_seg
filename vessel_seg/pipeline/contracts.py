"""Dataclasses for the modular five-stage pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


StageName = Literal[
    "ct_segmentation",
    "centerline_extraction",
    "centerline_repair",
    "wall_features",
    "rendering",
]


@dataclass
class CaseInputs:
    """Input bundle for a single case."""

    case_id: str
    ct_path: Path
    mask_path: Path | None = None
    centerline_vtp_path: Path | None = None
    probability_map_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: (str(value) if isinstance(value, Path) else value) for key, value in payload.items()}


@dataclass
class StageArtifact:
    """Standard output contract for one pipeline stage."""

    stage_name: StageName
    stage_dir: Path
    manifest_path: Path
    primary_path: Path | None = None
    secondary_paths: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "stage_dir": str(self.stage_dir),
            "manifest_path": str(self.manifest_path),
            "primary_path": None if self.primary_path is None else str(self.primary_path),
            "secondary_paths": {key: str(value) for key, value in self.secondary_paths.items()},
            "metadata": self.metadata,
        }


@dataclass
class PipelineConfig:
    """Runtime configuration for the rebuilt pipeline."""

    output_root: Path
    segmentation_backend: str = "existing_mask"
    centerline_backend: str = "mask_skeleton"
    repair_mode: str = "topology_only"
    dry_run: bool = False


@dataclass
class PipelineRunSummary:
    """Execution summary for one case."""

    case_id: str
    output_root: Path
    inputs: CaseInputs
    stages: list[StageArtifact] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "output_root": str(self.output_root),
            "inputs": self.inputs.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
        }
