"""Interfaces for coronary segmentation backends."""

from __future__ import annotations

import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import nibabel as nib
import numpy as np

from .io import VolumeData


class CoronarySegmentationBackend(ABC):
    """Abstract segmentation interface to allow swapping different models."""

    @abstractmethod
    def predict_mask(self, volume: VolumeData) -> np.ndarray:
        """Return a binary coronary mask for the given volume."""
        raise NotImplementedError


class DummySegmentationBackend(CoronarySegmentationBackend):
    """Placeholder backend that returns an empty mask."""

    def predict_mask(self, volume: VolumeData) -> np.ndarray:
        return np.zeros_like(volume.data, dtype=np.uint8)


class TotalSegmentationBackend(CoronarySegmentationBackend):
    """Run TotalSegmentator CLI and return a binary mask."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        prediction_file: str = "coronary_arteries.nii.gz",
        command: str = "TotalSegmentator",
        task: Optional[str] = None,
        fast: bool = False,
        extra_args: Optional[Sequence[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.prediction_file = prediction_file
        self.command = command
        self.task = task
        self.fast = fast
        self.extra_args = list(extra_args) if extra_args is not None else []

    def predict_mask(self, volume: VolumeData) -> np.ndarray:
        if volume.path is None:
            raise ValueError("TotalSegmentationBackend requires volume.path to run external CLI.")

        input_path = Path(volume.path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CT not found: {input_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.command,
            "-i",
            str(input_path),
            "-o",
            str(self.output_dir),
        ]
        if self.task:
            cmd.extend(["--task", self.task])
        if self.fast:
            cmd.append("--fast")
        cmd.extend(self.extra_args)
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "TotalSegmentator failed.\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout tail: {completed.stdout[-2000:]}\n"
                f"stderr tail: {completed.stderr[-2000:]}"
            )

        mask_path = self.output_dir / self.prediction_file
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Expected TotalSegmentator output missing: {mask_path}. "
                "Adjust `prediction_file` to match your TotalSeg output."
            )

        image = nib.load(str(mask_path))
        mask = image.get_fdata() > 0.5
        return mask.astype(np.uint8)


def run_segmentation(
    volume: VolumeData,
    backend: CoronarySegmentationBackend,
    postprocess: bool = False,
    min_size: Optional[int] = None,
) -> np.ndarray:
    """Run a backend and optionally postprocess small components."""
    mask = backend.predict_mask(volume)
    if not postprocess or min_size is None:
        return mask.astype(np.uint8)

    labeled, counts = np.unique(mask, return_counts=True)
    keep = {lbl for lbl, cnt in zip(labeled, counts) if cnt >= min_size and lbl != 0}
    cleaned = np.where(np.isin(mask, list(keep)), mask, 0).astype(np.uint8)
    return cleaned
