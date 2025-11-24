"""Split vessel branches into fixed-length segments and record relative angles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _arclength(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float | None:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def split_branch(points: np.ndarray, segment_len: float) -> List[Dict[str, object]]:
    s = _arclength(points)
    if s.size < 2 or s[-1] <= 0:
        return []
    segments: List[Dict[str, object]] = []
    edges = np.arange(0.0, s[-1] + segment_len, segment_len)
    for idx in range(len(edges) - 1):
        start, end = edges[idx], min(edges[idx + 1], s[-1])
        mask = (s >= start) & (s <= end + 1e-6)
        pts_seg = points[mask]
        seg_len = float(_arclength(pts_seg)[-1]) if pts_seg.shape[0] > 1 else 0.0
        if pts_seg.shape[0] < 2 or seg_len < 1e-3:
            continue
        direction = pts_seg[-1] - pts_seg[0]
        segments.append(
            {
                "points": pts_seg,
                "start_mm": float(start),
                "end_mm": float(end),
                "length_mm": seg_len,
                "direction": direction,
            }
        )
    return segments


def process_features(
    features_dir: Path, output_dir: Path, segment_len: float, min_seg_len: float
) -> Dict[str, object]:
    summary = json.loads((features_dir / "summary.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_meta: List[Dict[str, object]] = []

    for branch in summary.get("branches", []):
        branch_file = features_dir / branch["feature_file"]
        data = np.load(branch_file)
        pts = data["samples_world"]
        segs = split_branch(pts, segment_len)
        prev_dir = None
        for seg_idx, seg in enumerate(segs):
            if seg["length_mm"] < min_seg_len:
                continue
            angle = _angle_deg(prev_dir, seg["direction"]) if prev_dir is not None else None
            seg_path = output_dir / f"{branch['name']}_seg{seg_idx:03d}.npz"
            np.savez(
                seg_path,
                branch_name=branch["name"],
                segment_index=seg_idx,
                points_world=seg["points"],
                start_mm=seg["start_mm"],
                end_mm=seg["end_mm"],
                length_mm=seg["length_mm"],
                angle_to_parent_deg=angle if angle is not None else np.nan,
            )
            segments_meta.append(
                {
                    "branch": branch["name"],
                    "segment_index": seg_idx,
                    "file": seg_path.name,
                    "start_mm": seg["start_mm"],
                    "end_mm": seg["end_mm"],
                    "length_mm": seg["length_mm"],
                    "angle_to_parent_deg": angle,
                }
            )
            prev_dir = seg["direction"]

    summary_out = {
        "source_features": str(features_dir),
        "segment_length_mm": segment_len,
        "min_segment_length_mm": min_seg_len,
        "segment_count": len(segments_meta),
        "segments": segments_meta,
    }
    (output_dir / "segments_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    return summary_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split vessel branches into fixed-length segments and record relative angles."
    )
    parser.add_argument("--features", required=True, help="Directory produced by vessel_seg.shape extract.")
    parser.add_argument("--out", required=True, help="Output directory for per-segment npz files.")
    parser.add_argument("--segment-length", type=float, default=10.0, help="Target segment arclength (mm).")
    parser.add_argument("--min-segment-length", type=float, default=2.0, help="Drop segments shorter than this (mm).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    features_dir = Path(args.features).resolve()
    output_dir = Path(args.out).resolve()
    summary = process_features(features_dir, output_dir, args.segment_length, args.min_segment_length)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
