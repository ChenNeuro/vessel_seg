"""Utilities for organising coronary segmentation results into an anatomical tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from .metadata import harmonise_label_names

_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        ",": " ",
        "&": " ",
        ":": " ",
    }
)


def _canonical_name(label: str) -> str:
    """Normalise raw label strings to PascalCase for consistent lookup."""
    if not label:
        return ""
    cleaned = label.translate(_PUNCTUATION_TRANSLATION)
    harmonised = harmonise_label_names({cleaned: 0})
    return next(iter(harmonised))


@dataclass
class SegmentAssignment:
    """Records a single segmentation payload associated with a tree node."""

    source_label: str
    payload: Any
    label_index: Optional[int] = None


@dataclass
class CoronaryTreeNode:
    """Node in the coronary artery anatomical tree."""

    name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    children: List["CoronaryTreeNode"] = field(default_factory=list)
    assignments: List[SegmentAssignment] = field(default_factory=list)

    @property
    def canonical_name(self) -> str:
        return _canonical_name(self.name)

    @property
    def all_names(self) -> Iterable[str]:
        yield self.name
        yield from self.aliases

    def add_child(self, child: "CoronaryTreeNode") -> None:
        self.children.append(child)

    def add_assignment(
        self,
        source_label: str,
        payload: Any,
        label_index: Optional[int] = None,
    ) -> None:
        self.assignments.append(
            SegmentAssignment(source_label=source_label, payload=payload, label_index=label_index)
        )

    def clear_assignments(self) -> None:
        self.assignments.clear()
        for child in self.children:
            child.clear_assignments()

    def iter_nodes(self) -> Iterable["CoronaryTreeNode"]:
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def to_dict(self, include_payload: bool = False) -> Dict[str, Any]:
        assignments: List[Any]
        if include_payload:
            assignments = [
                {
                    "source_label": assignment.source_label,
                    "label_index": assignment.label_index,
                    "payload": assignment.payload,
                }
                for assignment in self.assignments
            ]
        else:
            assignments = [assignment.source_label for assignment in self.assignments]
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "assignments": assignments,
            "children": [child.to_dict(include_payload=include_payload) for child in self.children],
        }


def _build_default_tree() -> CoronaryTreeNode:
    """Create the default coronary artery topology."""
    diagonal = CoronaryTreeNode(
        "DiagonalBranch",
        aliases=("Diagonal", "DiagonalBranches", "D1", "D2", "D3"),
    )
    septal = CoronaryTreeNode(
        "SeptalPerforator",
        aliases=("SeptalPerforators", "Septal", "SeptalBranch"),
    )
    lad = CoronaryTreeNode(
        "LeftAnteriorDescending",
        aliases=("LAD", "Left Anterior Descending"),
        children=[diagonal, septal],
    )

    obtuse = CoronaryTreeNode(
        "ObtuseMarginal",
        aliases=("ObtuseMarginalBranch", "OM", "OM1", "OM2", "OM3"),
    )
    posterolateral = CoronaryTreeNode(
        "PosterolateralBranch",
        aliases=("Posterolateral", "PL", "PLV"),
    )
    left_pda = CoronaryTreeNode(
        "LeftPosteriorDescending",
        aliases=("Left Posterior Descending", "LPDA", "Posterior Descending (LCx)"),
    )
    lcx = CoronaryTreeNode(
        "LeftCircumflex",
        aliases=("LCx", "Left Circumflex"),
        children=[obtuse, posterolateral, left_pda],
    )
    left_main = CoronaryTreeNode(
        "LeftMain",
        aliases=("Left Main", "LM", "LeftMainCoronaryArtery"),
        children=[lad, lcx],
    )
    ramus = CoronaryTreeNode(
        "RamusIntermedius",
        aliases=("Ramus", "Ramus Intermedius", "RI", "Intermediate Branch"),
    )
    left_coronary = CoronaryTreeNode(
        "LeftCoronaryArtery",
        aliases=("LCA", "Left Coronary Artery"),
        children=[left_main, ramus],
    )

    conus = CoronaryTreeNode(
        "ConusBranch",
        aliases=("Conus", "ConalBranch"),
    )
    sinoatrial = CoronaryTreeNode(
        "SinoatrialNodeArtery",
        aliases=("Sinoatrial Node Artery", "SAN", "SA Node Artery"),
    )
    marginal = CoronaryTreeNode(
        "RightMarginal",
        aliases=("Acute Marginal", "AcuteMarginal", "Right Marginal Branch"),
    )
    right_pda = CoronaryTreeNode(
        "RightPosteriorDescending",
        aliases=(
            "Right Posterior Descending",
            "Posterior Descending Artery",
            "RPDA",
            "Posterior Descending",
            "PDA",
        ),
    )
    av_node = CoronaryTreeNode(
        "AtrioventricularNodeArtery",
        aliases=("AV Node Artery", "AVN", "Atrioventricular Node Artery"),
    )
    right_coronary = CoronaryTreeNode(
        "RightCoronaryArtery",
        aliases=("RCA", "Right Coronary Artery"),
        children=[conus, sinoatrial, marginal, right_pda, av_node],
    )

    root = CoronaryTreeNode(
        "CoronaryArteries",
        aliases=("Coronary Artery Tree", "Coronary Tree", "Coronary"),
        children=[left_coronary, right_coronary],
    )
    return root


class CoronaryTree:
    """High-level interface for distributing segmentation results across the coronary tree."""

    def __init__(self, root: Optional[CoronaryTreeNode] = None) -> None:
        self.root = root if root is not None else _build_default_tree()
        self._lookup: Dict[str, CoronaryTreeNode] = {}
        self._build_index(self.root)

    def _build_index(self, node: CoronaryTreeNode) -> None:
        for alias in node.all_names:
            canonical = _canonical_name(alias)
            if canonical:
                self._lookup.setdefault(canonical, node)
        for child in node.children:
            self._build_index(child)

    def find(self, name: str) -> Optional[CoronaryTreeNode]:
        """Retrieve a node by canonical name or alias."""
        return self._lookup.get(_canonical_name(name))

    def clear(self) -> None:
        """Remove all segment assignments across the tree."""
        self.root.clear_assignments()

    def iter_nodes(self) -> Iterable[CoronaryTreeNode]:
        """Yield all nodes in depth-first order."""
        return self.root.iter_nodes()

    def assign_segments(
        self,
        segments: Mapping[str, Union[Any, Tuple[Any, Optional[int]], SegmentAssignment]],
    ) -> Dict[str, Any]:
        """Distribute segmentation payloads into the tree.

        Parameters
        ----------
        segments:
            Mapping of raw label name to an arbitrary payload (mask array, path, statistics, ...).

        Returns
        -------
        Dict[str, Any]:
            Entries that could not be matched to the tree.
        """
        unmatched: Dict[str, Any] = {}
        for raw_label, payload in segments.items():
            node = self.find(raw_label)
            if node is None:
                unmatched[raw_label] = payload
                continue
            if isinstance(payload, SegmentAssignment):
                assignment = payload
            elif isinstance(payload, tuple) and len(payload) == 2:
                assignment = SegmentAssignment(
                    source_label=raw_label,
                    payload=payload[0],
                    label_index=payload[1],
                )
            else:
                assignment = SegmentAssignment(source_label=raw_label, payload=payload)
            node.add_assignment(
                assignment.source_label,
                assignment.payload,
                assignment.label_index,
            )
        return unmatched

    def assign_from_labelmap(
        self,
        labelmap: np.ndarray,
        labels: Mapping[str, int],
        *,
        background_value: Optional[int] = 0,
        keep_empty: bool = False,
        dtype: Optional[np.dtype] = None,
    ) -> Dict[str, Any]:
        """Slice a label map into boolean masks and assign them into the tree.

        Parameters
        ----------
        labelmap:
            Dense array of label indices (e.g. nnU-Net or TotalSegmentator output).
        labels:
            Mapping from label names to integer indices in `labelmap`.
        background_value:
            Label value treated as background and skipped; set to ``None`` to disable.
        keep_empty:
            If ``True``, include masks even when no voxels match the label.
        dtype:
            Optionally down-cast the extracted masks (e.g. to ``np.uint8``). Defaults to boolean arrays.
        """
        label_array = np.asarray(labelmap)
        segments: Dict[str, Tuple[Any, Optional[int]]] = {}
        for raw_label, index in labels.items():
            if background_value is not None and index == background_value:
                continue
            mask = label_array == index
            if not keep_empty and not np.any(mask):
                continue
            if dtype is not None:
                mask = mask.astype(dtype, copy=False)
            segments[raw_label] = (mask, index)
        return self.assign_segments(segments)

    def to_dict(self, include_payload: bool = False) -> Dict[str, Any]:
        """Serialise the tree structure for reporting or debugging."""
        return self.root.to_dict(include_payload=include_payload)
