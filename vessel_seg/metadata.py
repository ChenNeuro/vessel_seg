"""Utilities for harmonising segmentation metadata across coronary artery studies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

SPECIAL_TOKEN_ALIASES: Dict[str, str] = {
    "lad": "LeftAnteriorDescending",
    "lcx": "LeftCircumflex",
    "lm": "LeftMain",
    "rca": "RightCoronaryArtery",
    "mra": "MedianRentropArtery",
    "ima": "InternalMammaryArtery",
}


def _pascal_case(name: str) -> str:
    """Convert snake/kebab/mixed case labels into PascalCase."""
    cleaned = (
        name.replace("-", " ")
        .replace("_", " ")
        .replace(".", " ")
        .replace("/", " ")
        .strip()
    )
    tokens = [tok for tok in cleaned.split() if tok]
    normalised: Iterable[str] = (
        SPECIAL_TOKEN_ALIASES.get(tok.lower(), tok.lower().capitalize()) for tok in tokens
    )
    return "".join(normalised)


def harmonise_label_names(labels: Mapping[str, int]) -> Dict[str, int]:
    """Return a dictionary with labels rewritten into academic PascalCase form."""
    harmonised: Dict[str, int] = {}
    for raw_name, index in labels.items():
        canonical = _pascal_case(raw_name)
        harmonised[canonical] = index
    return harmonised


@dataclass
class SegmentationMetadata:
    """Structured metadata record for cardiovascular segmentation volumes."""

    case_id: str
    dataset: str
    anatomy_region: str
    segmentation_method: str
    segmentation_version: str
    voxel_spacing_mm: tuple[float, float, float]
    structure_labels: Dict[str, int] = field(default_factory=dict)
    reference_volume: str | None = None
    source_institution: str | None = None
    notes: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the metadata into a JSON-friendly mapping."""
        payload = asdict(self)
        payload["voxel_spacing_mm"] = list(self.voxel_spacing_mm)
        return payload

    @classmethod
    def from_totalseg_json(
        cls,
        *,
        case_id: str,
        dataset: str,
        meta: Mapping[str, Any],
        anatomy_region: str = "CoronaryArteries",
        segmentation_method: str = "TotalSegmentator",
        segmentation_version: str = "unknown",
    ) -> "SegmentationMetadata":
        """Build a metadata record from a TotalSegmentator result descriptor."""
        spacing = tuple(meta.get("spacing", (1.0, 1.0, 1.0)))
        labels = meta.get("labels", {})
        harmonised = harmonise_label_names(labels)
        return cls(
            case_id=case_id,
            dataset=dataset,
            anatomy_region=anatomy_region,
            segmentation_method=segmentation_method,
            segmentation_version=meta.get("model_version", segmentation_version),
            voxel_spacing_mm=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            structure_labels=harmonised,
            reference_volume=meta.get("reference_image"),
            source_institution=meta.get("institution"),
            notes=meta.get("notes"),
        )


def load_metadata(path: str | Path) -> SegmentationMetadata:
    """Load a metadata record from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    payload["voxel_spacing_mm"] = tuple(payload["voxel_spacing_mm"])
    return SegmentationMetadata(**payload)


def save_metadata(metadata: SegmentationMetadata, path: str | Path) -> None:
    """Persist a metadata record to disk as UTF-8 encoded JSON."""
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(metadata.to_dict(), fh, indent=2, ensure_ascii=False)


def unify_metadata_file(source: str | Path, destination: str | Path) -> None:
    """Load metadata, harmonise label names, and save to a new location."""
    with Path(source).open("r", encoding="utf-8") as fh:
        payload: MutableMapping[str, Any] = json.load(fh)

    labels = payload.get("structure_labels") or payload.get("labels") or {}
    payload["structure_labels"] = harmonise_label_names(labels)

    metadata = SegmentationMetadata(
        case_id=payload["case_id"],
        dataset=payload["dataset"],
        anatomy_region=payload.get("anatomy_region", "CoronaryArteries"),
        segmentation_method=payload.get("segmentation_method", "Unknown"),
        segmentation_version=payload.get("segmentation_version", "unknown"),
        voxel_spacing_mm=tuple(payload.get("voxel_spacing_mm", (1.0, 1.0, 1.0))),
        structure_labels=payload["structure_labels"],
        reference_volume=payload.get("reference_volume"),
        source_institution=payload.get("source_institution"),
        notes=payload.get("notes"),
    )
    save_metadata(metadata, destination)
