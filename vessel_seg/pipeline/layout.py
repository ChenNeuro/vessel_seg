"""Filesystem layout helpers for the five-stage pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .contracts import StageArtifact, StageName


STAGE_ORDER: list[tuple[str, StageName]] = [
    ("01", "ct_segmentation"),
    ("02", "centerline_extraction"),
    ("03", "centerline_repair"),
    ("04", "wall_features"),
    ("05", "rendering"),
]


@dataclass(frozen=True)
class PipelineLayout:
    """Canonical stage folders under outputs_reorganized."""

    case_id: str
    case_dir: Path
    repo_root: Path

    def stage_dir(self, stage_name: StageName) -> Path:
        for prefix, candidate in STAGE_ORDER:
            if candidate == stage_name:
                return self.case_dir / "stages" / f"{prefix}_{stage_name}"
        raise ValueError(f"Unknown stage name: {stage_name}")

    def ensure(self) -> None:
        (self.case_dir / "stages").mkdir(parents=True, exist_ok=True)


def build_pipeline_layout(output_root: Path, case_id: str, repo_root: Path) -> PipelineLayout:
    return PipelineLayout(case_id=case_id, case_dir=output_root / "cases" / case_id, repo_root=repo_root)


def write_stage_manifest(artifact: StageArtifact) -> None:
    artifact.stage_dir.mkdir(parents=True, exist_ok=True)
    artifact.manifest_path.write_text(
        json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
