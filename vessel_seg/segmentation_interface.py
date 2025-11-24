"""Interfaces for coronary segmentation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
