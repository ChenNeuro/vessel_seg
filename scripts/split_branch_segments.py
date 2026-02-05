"""Split per-branch centreline samples into fixed-length segments and measure bend angles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _discover_feature_files(features_dir: Path) -> List[Path]:
    summary_path = features_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            files = []
            for entry in summary.get("branches", []):
                feature_file = entry.get("feature_file")
                if not feature_file:
                    continue
                path = features_dir / feature_file
                if path.exists():
                    files.append(path)
            if files:
                return files
        except Exception:
            pass
    return sorted(p for p in features_dir.glob("*.npz") if p.is_file())


def _cumulative_arclength(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    s = np.zeros(points.shape[0], dtype=np.float64)
    if points.shape[0] > 1:
        s[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    return s


def _split_segments(
    points: np.ndarray, segment_length: float, min_segment_length: float
) -> List[Tuple[int, float, float, float, np.ndarray]]:
    s = _cumulative_arclength(points)
    if s.size == 0:
        return []

    total = float(s[-1])
    segments: List[Tuple[int, float, float, float, np.ndarray]] = []
    start = 0.0
    idx = 0
    eps = 1e-6
    while start < total - eps:
        end = min(start + segment_length, total)
        mask = (s >= start - eps) & (s <= end + eps)
        indices = np.where(mask)[0]
        if indices.size >= 2:
            length = float(s[indices[-1]] - s[indices[0]])
            if length >= min_segment_length:
                segments.append((idx, start, end, length, points[indices]))
                idx += 1
        start += segment_length
    return segments


def _segment_angle(parent_pts: np.ndarray, child_pts: np.ndarray) -> float:
    v1 = parent_pts[-1] - parent_pts[0]
    v2 = child_pts[-1] - child_pts[0]
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-8:
        return math.nan
    cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def _load_branch(points_file: Path) -> Tuple[str, np.ndarray]:
    data = np.load(points_file)
    branch_name = str(data.get("branch_name", points_file.stem))
    points = np.asarray(data["samples_world"], dtype=np.float64)
    return branch_name, points


def split_features(
    features_dir: Path, out_dir: Path, segment_length: float, min_segment_length: float
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_files = _discover_feature_files(features_dir)
    if not feature_files:
        raise FileNotFoundError(f"No branch feature files found in {features_dir}")

    segments_meta = []
    total_segments = 0

    for feat_file in feature_files:
        branch_name, points = _load_branch(feat_file)
        segments = _split_segments(points, segment_length, min_segment_length)
        if not segments:
            print(f"[skip] {branch_name}: no valid segments.")
            continue

        prev_pts: np.ndarray | None = None
        for seg_idx, start_mm, end_mm, length_mm, seg_pts in segments:
            angle = math.nan
            if prev_pts is not None:
                angle = _segment_angle(prev_pts, seg_pts)
            prev_pts = seg_pts

            filename = f"{branch_name}_seg{seg_idx:03d}.npz"
            np.savez(
                out_dir / filename,
                branch_name=branch_name,
                segment_index=seg_idx,
                points_world=seg_pts,
                start_mm=start_mm,
                end_mm=end_mm,
                length_mm=length_mm,
                angle_to_parent_deg=angle,
            )

            segments_meta.append(
                {
                    "branch": branch_name,
                    "segment_index": seg_idx,
                    "file": filename,
                    "start_mm": start_mm,
                    "end_mm": end_mm,
                    "length_mm": length_mm,
                    "angle_to_parent_deg": None if math.isnan(angle) else angle,
                }
            )
        total_segments += len(segments)
        print(f"[ok] {branch_name}: {len(segments)} segments.")

    summary = {
        "source_features": str(features_dir.resolve()),
        "segment_length_mm": float(segment_length),
        "min_segment_length_mm": float(min_segment_length),
        "segment_count": total_segments,
        "segments": segments_meta,
    }
    (out_dir / "segments_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split branch centrelines into fixed-length segments and compute inter-segment angles."
    )
    parser.add_argument("--features", required=True, help="Directory containing per-branch feature .npz files.")
    parser.add_argument("--out", required=True, help="Directory to save segment .npz files and summary JSON.")
    parser.add_argument("--segment-length", type=float, default=10.0, help="Target segment length in mm.")
    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=2.0,
        help="Drop segments shorter than this (mm).",
    )
    args = parser.parse_args(argv)

    summary = split_features(
        Path(args.features),
        Path(args.out),
        segment_length=float(args.segment_length),
        min_segment_length=float(args.min_segment_length),
    )
    print(f"Saved {summary['segment_count']} segments -> {args.out}")


if __name__ == "__main__":
    main()
