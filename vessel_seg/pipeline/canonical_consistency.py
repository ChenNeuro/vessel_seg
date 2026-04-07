"""Canonical branch naming consistency analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import csv
import json
import math
import numpy as np

from .semantic_topology import (
    _build_descriptors,
    _build_frame,
    _descriptor_feature_vector,
    _infer_root_systems,
)


@dataclass(frozen=True)
class CanonicalBranchSample:
    """单个病例中某个 canonical branch 的归一化几何样本。"""

    case_id: str
    cohort: str
    canonical_name: str
    branch_id: int
    parent_branch_id: int | None
    sibling_group_key: str
    orientation_axis: str
    orientation_signed_axis: str
    orientation_confidence: float
    system_name: str
    depth: int
    centroid_proj: tuple[float, float, float]
    direction_proj: tuple[float, float, float]
    start_tangent_proj: tuple[float, float, float]
    end_tangent_proj: tuple[float, float, float]
    length_norm: float
    lambda_pos: float | None
    subtree_length_norm: float
    tortuosity: float
    bend_angle_norm: float
    centerline_offset_norm: float


@dataclass(frozen=True)
class CanonicalLabelStats:
    """单个 canonical label 的跨病例稳定性统计。"""

    canonical_name: str
    case_count: int
    presence_rate: float
    is_root_label: bool
    mean_pairwise_similarity: float
    mean_shape_similarity: float
    min_pairwise_similarity: float
    max_pairwise_similarity: float
    mean_centroid_similarity: float
    mean_direction_similarity: float
    mean_length_similarity: float
    mean_subtree_similarity: float
    mean_depth_similarity: float
    mean_lambda_similarity: float
    worst_pair_case_left: str
    worst_pair_case_right: str
    best_pair_case_left: str
    best_pair_case_right: str


@dataclass(frozen=True)
class CanonicalCaseAnomaly:
    """某个病例在某个 canonical label 上偏离群体原型的记录。"""

    case_id: str
    cohort: str
    canonical_name: str
    overall_similarity_to_prototype: float
    shape_similarity_to_prototype: float
    centroid_similarity_to_prototype: float
    direction_similarity_to_prototype: float
    length_similarity_to_prototype: float
    subtree_similarity_to_prototype: float
    depth_similarity_to_prototype: float
    lambda_similarity_to_prototype: float


@dataclass(frozen=True)
class CanonicalConsistencyReport:
    """批量 canonical naming 一致性报告。"""

    summary_csv: Path
    report_json: Path
    label_stats_csv: Path
    anomalies_csv: Path
    total_cases: int
    total_labels: int
    mean_presence_rate: float
    mean_pairwise_similarity: float
    mean_shape_similarity: float
    label_stats: list[CanonicalLabelStats]
    anomalies: list[CanonicalCaseAnomaly]


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _gaussian_similarity(delta: float, sigma: float) -> float:
    safe_sigma = max(float(sigma), 1e-6)
    return float(math.exp(-0.5 * (float(delta) / safe_sigma) ** 2))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _direction_dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _normalize_vector(values: np.ndarray) -> tuple[float, float, float]:
    norm = float(np.linalg.norm(values))
    if norm <= 1e-8:
        return (0.0, 0.0, 1.0)
    values = values / norm
    return (float(values[0]), float(values[1]), float(values[2]))


def _project_vector(
    values: tuple[float, float, float],
    feature_vector: tuple[float, float, float],
    longitudinal_axis: tuple[float, float, float],
    normal_axis: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        _direction_dot(values, feature_vector),
        _direction_dot(values, longitudinal_axis),
        _direction_dot(values, normal_axis),
    )


def _orientation_signature(
    direction_proj: tuple[float, float, float],
    start_tangent_proj: tuple[float, float, float],
    end_tangent_proj: tuple[float, float, float],
) -> tuple[str, str, float]:
    axis_labels = ("lr", "long", "norm")
    axis_strengths = [
        0.50 * abs(float(direction_proj[index]))
        + 0.25 * abs(float(start_tangent_proj[index]))
        + 0.25 * abs(float(end_tangent_proj[index]))
        for index in range(3)
    ]
    order = sorted(range(3), key=lambda index: axis_strengths[index], reverse=True)
    dominant_index = order[0]
    secondary_index = order[1]
    confidence = max(0.0, axis_strengths[dominant_index] - axis_strengths[secondary_index])
    signed_value = (
        0.50 * float(direction_proj[dominant_index])
        + 0.25 * float(start_tangent_proj[dominant_index])
        + 0.25 * float(end_tangent_proj[dominant_index])
    )
    sign_label = "pos" if signed_value >= 0.0 else "neg"
    axis_label = axis_labels[dominant_index]
    return axis_label, f"{axis_label}:{sign_label}", float(confidence)


def _read_case_branch_polylines(row: dict[str, str]) -> list[np.ndarray]:
    overview_path = Path(row["overview_path"])
    vtp_path = overview_path.parents[2] / "stages" / "03_centerline_repair" / "centerline_repaired.vtp"
    if not vtp_path.exists():
        return []
    try:
        import vtk
    except Exception:
        return []
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    points = poly.GetPoints()
    lines = poly.GetLines()
    polylines: list[np.ndarray] = []
    lines.InitTraversal()
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        coords = np.asarray([points.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=np.float32)
        if coords.shape[0] >= 2:
            polylines.append(coords)
    return polylines


def _fallback_polyline(branch: dict[str, object]) -> np.ndarray:
    start = np.asarray(branch.get("start") or [0.0, 0.0, 0.0], dtype=np.float32)
    centroid = np.asarray(branch.get("centroid") or branch.get("end") or branch.get("start") or [0.0, 0.0, 0.0], dtype=np.float32)
    end = np.asarray(branch.get("end") or branch.get("start") or [0.0, 0.0, 0.0], dtype=np.float32)
    return np.vstack([start, centroid, end]).astype(np.float32, copy=False)


def _root_membership(
    branch_id: int,
    descriptors,
    root_systems: dict[int, str],
) -> str:
    current = descriptors[branch_id]
    parent_id = current.parent_id
    while parent_id is not None and parent_id in descriptors:
        current = descriptors[parent_id]
        parent_id = current.parent_id
    return str(root_systems.get(current.branch_id, "MAIN"))


def _branch_similarity(
    left: CanonicalBranchSample,
    right: CanonicalBranchSample,
) -> dict[str, float]:
    centroid_delta = math.sqrt(
        sum((a - b) * (a - b) for a, b in zip(left.centroid_proj, right.centroid_proj))
    )
    centroid_similarity = _gaussian_similarity(centroid_delta, 0.60)
    direction_similarity = _clamp01(0.5 * (1.0 + _direction_dot(left.direction_proj, right.direction_proj)))
    start_tangent_similarity = _clamp01(0.5 * (1.0 + _direction_dot(left.start_tangent_proj, right.start_tangent_proj)))
    end_tangent_similarity = _clamp01(0.5 * (1.0 + _direction_dot(left.end_tangent_proj, right.end_tangent_proj)))
    length_similarity = _gaussian_similarity(abs(left.length_norm - right.length_norm), 0.45)
    subtree_similarity = _gaussian_similarity(abs(left.subtree_length_norm - right.subtree_length_norm), 0.80)
    depth_similarity = _gaussian_similarity(abs(left.depth - right.depth) / 5.0, 0.25)
    tortuosity_similarity = _gaussian_similarity(abs(left.tortuosity - right.tortuosity), 0.18)
    bend_similarity = _gaussian_similarity(abs(left.bend_angle_norm - right.bend_angle_norm), 0.16)
    offset_similarity = _gaussian_similarity(abs(left.centerline_offset_norm - right.centerline_offset_norm), 0.18)

    if left.lambda_pos is None and right.lambda_pos is None:
        lambda_similarity = 1.0
    elif left.lambda_pos is None or right.lambda_pos is None:
        lambda_similarity = 0.5
    else:
        lambda_similarity = _gaussian_similarity(abs(left.lambda_pos - right.lambda_pos), 0.30)

    if left.system_name == right.system_name:
        system_penalty = 1.0
    elif "AUX" in {left.system_name, right.system_name}:
        system_penalty = 0.80
    else:
        system_penalty = 0.55

    min_orientation_confidence = min(left.orientation_confidence, right.orientation_confidence)
    if left.orientation_axis == right.orientation_axis:
        orientation_axis_gate = 1.0
    elif min_orientation_confidence >= 0.18:
        orientation_axis_gate = 0.60
    elif min_orientation_confidence >= 0.10:
        orientation_axis_gate = 0.78
    else:
        orientation_axis_gate = 0.90

    if left.orientation_signed_axis == right.orientation_signed_axis:
        orientation_sign_gate = 1.0
    elif left.orientation_axis == right.orientation_axis and min_orientation_confidence >= 0.18:
        orientation_sign_gate = 0.82
    elif left.orientation_axis == right.orientation_axis:
        orientation_sign_gate = 0.92
    else:
        orientation_sign_gate = 1.0

    if left.depth == 0 and right.depth == 0:
        depth_gate = 1.0
    elif (left.depth == 0) != (right.depth == 0):
        depth_gate = 0.30
    elif abs(left.depth - right.depth) >= 2:
        depth_gate = 0.55
    elif abs(left.depth - right.depth) == 1:
        depth_gate = 0.78
    else:
        depth_gate = 1.0

    overall_similarity = (
        0.10 * centroid_similarity
        + 0.16 * direction_similarity
        + 0.12 * start_tangent_similarity
        + 0.12 * end_tangent_similarity
        + 0.10 * length_similarity
        + 0.08 * subtree_similarity
        + 0.06 * depth_similarity
        + 0.04 * lambda_similarity
        + 0.10 * tortuosity_similarity
        + 0.07 * bend_similarity
        + 0.05 * offset_similarity
    ) * system_penalty * depth_gate * orientation_axis_gate * orientation_sign_gate
    shape_similarity = (
        0.18 * direction_similarity
        + 0.20 * start_tangent_similarity
        + 0.20 * end_tangent_similarity
        + 0.12 * length_similarity
        + 0.10 * tortuosity_similarity
        + 0.10 * bend_similarity
        + 0.10 * offset_similarity
    ) * system_penalty * depth_gate * orientation_axis_gate * orientation_sign_gate
    return {
        "overall": round(overall_similarity, 6),
        "shape": round(shape_similarity, 6),
        "centroid": round(centroid_similarity, 6),
        "direction": round(direction_similarity, 6),
        "start_tangent": round(start_tangent_similarity, 6),
        "end_tangent": round(end_tangent_similarity, 6),
        "length": round(length_similarity, 6),
        "subtree": round(subtree_similarity, 6),
        "depth": round(depth_similarity, 6),
        "lambda": round(lambda_similarity, 6),
        "tortuosity": round(tortuosity_similarity, 6),
        "bend": round(bend_similarity, 6),
        "offset": round(offset_similarity, 6),
        "system_penalty": round(system_penalty, 6),
        "depth_gate": round(depth_gate, 6),
        "orientation_axis_gate": round(orientation_axis_gate, 6),
        "orientation_sign_gate": round(orientation_sign_gate, 6),
    }


def _mean(values: list[float], default: float = 1.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _sample_branch_payloads(row: dict[str, str]) -> list[CanonicalBranchSample]:
    overview_path = Path(row["overview_path"])
    tree_path = overview_path.parents[2] / "stages" / "03_centerline_repair" / "tree.json"
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    descriptors, roots = _build_descriptors(payload)
    root_systems = _infer_root_systems(roots, descriptors)
    frame = _build_frame(root_systems, descriptors)
    feature_vectors = {
        branch_id: _descriptor_feature_vector(descriptor, frame)
        for branch_id, descriptor in descriptors.items()
    }
    polylines = _read_case_branch_polylines(row)

    samples: list[CanonicalBranchSample] = []
    for branch in list(payload.get("branches", [])):
        naming = dict(branch.get("naming") or {})
        canonical_name = naming.get("canonical_name")
        if canonical_name is None:
            continue
        branch_id = int(branch["branch_id"])
        descriptor = descriptors[branch_id]
        feature_vector = feature_vectors[branch_id]
        polyline = polylines[branch_id] if branch_id < len(polylines) else _fallback_polyline(branch)
        diffs = np.diff(polyline, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1) if diffs.size else np.asarray([0.0], dtype=np.float32)
        arc_length = float(segment_lengths.sum())
        chord = polyline[-1] - polyline[0]
        chord_length = float(np.linalg.norm(chord))
        start_tangent = _normalize_vector(diffs[0] if diffs.size else chord)
        end_tangent = _normalize_vector(diffs[-1] if diffs.size else chord)
        start_tangent_proj = _project_vector(
            start_tangent,
            frame.left_right_axis,
            frame.longitudinal_axis,
            frame.normal_axis,
        )
        end_tangent_proj = _project_vector(
            end_tangent,
            frame.left_right_axis,
            frame.longitudinal_axis,
            frame.normal_axis,
        )
        orientation_axis, orientation_signed_axis, orientation_confidence = _orientation_signature(
            tuple(float(value) for value in feature_vector[3:6]),
            start_tangent_proj,
            end_tangent_proj,
        )
        tortuosity = arc_length / max(chord_length, 1e-3)
        bend_angle_norm = 0.5 * (1.0 - _direction_dot(start_tangent, end_tangent))
        if chord_length <= 1e-6:
            centerline_offset_norm = 0.0
        else:
            chord_unit = chord / chord_length
            rel_points = polyline - polyline[0]
            projections = rel_points @ chord_unit
            closest = np.outer(projections, chord_unit)
            offsets = rel_points - closest
            centerline_offset_norm = float(np.linalg.norm(offsets, axis=1).max() / max(arc_length, 1.0))
        samples.append(
            CanonicalBranchSample(
                case_id=str(row["case_id"]),
                cohort=str(row.get("cohort", "")),
                canonical_name=str(canonical_name),
                branch_id=branch_id,
                parent_branch_id=descriptor.parent_id,
                sibling_group_key="ROOTS" if descriptor.parent_id is None else f"PARENT_{descriptor.parent_id}",
                orientation_axis=orientation_axis,
                orientation_signed_axis=orientation_signed_axis,
                orientation_confidence=float(orientation_confidence),
                system_name=_root_membership(branch_id, descriptors, root_systems),
                depth=descriptor.depth,
                centroid_proj=tuple(float(value) for value in feature_vector[0:3]),
                direction_proj=tuple(float(value) for value in feature_vector[3:6]),
                start_tangent_proj=start_tangent_proj,
                end_tangent_proj=end_tangent_proj,
                length_norm=float(feature_vector[6]),
                lambda_pos=None if descriptor.lambda_pos is None else float(descriptor.lambda_pos),
                subtree_length_norm=float(feature_vector[9]),
                tortuosity=float(tortuosity),
                bend_angle_norm=float(bend_angle_norm),
                centerline_offset_norm=float(centerline_offset_norm),
            )
        )
    return samples


def _prototype(samples: list[CanonicalBranchSample]) -> CanonicalBranchSample:
    if not samples:
        raise ValueError("Prototype requires at least one sample.")
    lambda_values = [sample.lambda_pos for sample in samples if sample.lambda_pos is not None]
    direction_mean = tuple(
        sum(sample.direction_proj[index] for sample in samples) / float(len(samples))
        for index in range(3)
    )
    direction_norm = math.sqrt(sum(value * value for value in direction_mean))
    if direction_norm > 1e-8:
        direction_mean = tuple(value / direction_norm for value in direction_mean)
    start_tangent_mean = tuple(
        sum(sample.start_tangent_proj[index] for sample in samples) / float(len(samples))
        for index in range(3)
    )
    start_tangent_norm = math.sqrt(sum(value * value for value in start_tangent_mean))
    if start_tangent_norm > 1e-8:
        start_tangent_mean = tuple(value / start_tangent_norm for value in start_tangent_mean)
    end_tangent_mean = tuple(
        sum(sample.end_tangent_proj[index] for sample in samples) / float(len(samples))
        for index in range(3)
    )
    end_tangent_norm = math.sqrt(sum(value * value for value in end_tangent_mean))
    if end_tangent_norm > 1e-8:
        end_tangent_mean = tuple(value / end_tangent_norm for value in end_tangent_mean)
    return CanonicalBranchSample(
        case_id="__prototype__",
        cohort="",
        canonical_name=samples[0].canonical_name,
        branch_id=-1,
        parent_branch_id=None,
        sibling_group_key="__prototype__",
        orientation_axis=max(
            {sample.orientation_axis for sample in samples},
            key=lambda axis: (
                sum(1 for sample in samples if sample.orientation_axis == axis),
                axis,
            ),
        ),
        orientation_signed_axis=max(
            {sample.orientation_signed_axis for sample in samples},
            key=lambda axis: (
                sum(1 for sample in samples if sample.orientation_signed_axis == axis),
                axis,
            ),
        ),
        orientation_confidence=sum(sample.orientation_confidence for sample in samples) / float(len(samples)),
        system_name=max(
            {sample.system_name for sample in samples},
            key=lambda system_name: (
                sum(1 for sample in samples if sample.system_name == system_name),
                system_name,
            ),
        ),
        depth=int(round(sum(sample.depth for sample in samples) / float(len(samples)))),
        centroid_proj=tuple(
            sum(sample.centroid_proj[index] for sample in samples) / float(len(samples))
            for index in range(3)
        ),
        direction_proj=tuple(float(value) for value in direction_mean),
        start_tangent_proj=tuple(float(value) for value in start_tangent_mean),
        end_tangent_proj=tuple(float(value) for value in end_tangent_mean),
        length_norm=sum(sample.length_norm for sample in samples) / float(len(samples)),
        lambda_pos=(sum(lambda_values) / float(len(lambda_values))) if lambda_values else None,
        subtree_length_norm=sum(sample.subtree_length_norm for sample in samples) / float(len(samples)),
        tortuosity=sum(sample.tortuosity for sample in samples) / float(len(samples)),
        bend_angle_norm=sum(sample.bend_angle_norm for sample in samples) / float(len(samples)),
        centerline_offset_norm=sum(sample.centerline_offset_norm for sample in samples) / float(len(samples)),
    )


def analyze_canonical_consistency(
    summary_csv: Path,
    *,
    output_dir: Path | None = None,
    min_case_count: int = 2,
    anomaly_similarity_threshold: float = 0.70,
    anomaly_shape_threshold: float = 0.50,
) -> CanonicalConsistencyReport:
    """读取 batch summary，输出 canonical naming 的跨病例一致性统计。"""
    rows = _load_summary_rows(summary_csv)
    output_dir = output_dir or summary_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_samples: dict[str, list[CanonicalBranchSample]] = {}
    for row in rows:
        for sample in _sample_branch_payloads(row):
            grouped_samples.setdefault(sample.canonical_name, []).append(sample)

    label_stats: list[CanonicalLabelStats] = []
    anomalies: list[CanonicalCaseAnomaly] = []
    for canonical_name, samples in sorted(grouped_samples.items()):
        if len(samples) < max(2, int(min_case_count)):
            continue

        pairwise_scores: list[dict[str, float]] = []
        worst_pair: tuple[str, str, float] = ("", "", 1.0)
        best_pair: tuple[str, str, float] = ("", "", 0.0)
        for left, right in combinations(samples, 2):
            score = _branch_similarity(left, right)
            pairwise_scores.append(score)
            if score["overall"] < worst_pair[2]:
                worst_pair = (left.case_id, right.case_id, score["overall"])
            if score["overall"] > best_pair[2]:
                best_pair = (left.case_id, right.case_id, score["overall"])

        label_stats.append(
            CanonicalLabelStats(
                canonical_name=canonical_name,
                case_count=len(samples),
                presence_rate=round(len(samples) / float(len(rows) or 1), 4),
                is_root_label="." not in canonical_name,
                mean_pairwise_similarity=round(_mean([item["overall"] for item in pairwise_scores]), 4),
                mean_shape_similarity=round(_mean([item["shape"] for item in pairwise_scores]), 4),
                min_pairwise_similarity=round(min((item["overall"] for item in pairwise_scores), default=1.0), 4),
                max_pairwise_similarity=round(max((item["overall"] for item in pairwise_scores), default=1.0), 4),
                mean_centroid_similarity=round(_mean([item["centroid"] for item in pairwise_scores]), 4),
                mean_direction_similarity=round(_mean([item["direction"] for item in pairwise_scores]), 4),
                mean_length_similarity=round(_mean([item["length"] for item in pairwise_scores]), 4),
                mean_subtree_similarity=round(_mean([item["subtree"] for item in pairwise_scores]), 4),
                mean_depth_similarity=round(_mean([item["depth"] for item in pairwise_scores]), 4),
                mean_lambda_similarity=round(_mean([item["lambda"] for item in pairwise_scores]), 4),
                worst_pair_case_left=worst_pair[0],
                worst_pair_case_right=worst_pair[1],
                best_pair_case_left=best_pair[0],
                best_pair_case_right=best_pair[1],
            )
        )

        prototype = _prototype(samples)
        for sample in samples:
            score = _branch_similarity(sample, prototype)
            if score["overall"] < float(anomaly_similarity_threshold) or score["shape"] < float(anomaly_shape_threshold):
                anomalies.append(
                    CanonicalCaseAnomaly(
                        case_id=sample.case_id,
                        cohort=sample.cohort,
                        canonical_name=canonical_name,
                        overall_similarity_to_prototype=round(score["overall"], 4),
                        shape_similarity_to_prototype=round(score["shape"], 4),
                        centroid_similarity_to_prototype=round(score["centroid"], 4),
                        direction_similarity_to_prototype=round(score["direction"], 4),
                        length_similarity_to_prototype=round(score["length"], 4),
                        subtree_similarity_to_prototype=round(score["subtree"], 4),
                        depth_similarity_to_prototype=round(score["depth"], 4),
                        lambda_similarity_to_prototype=round(score["lambda"], 4),
                    )
                )

    label_stats.sort(
        key=lambda item: (
            -item.case_count,
            -item.mean_pairwise_similarity,
            item.canonical_name,
        )
    )
    anomalies.sort(
        key=lambda item: (
            item.overall_similarity_to_prototype,
            item.shape_similarity_to_prototype,
            item.canonical_name,
            item.case_id,
        )
    )

    report_payload = {
        "summary_csv": str(summary_csv),
        "total_cases": len(rows),
        "total_labels": len(label_stats),
        "mean_presence_rate": round(_mean([item.presence_rate for item in label_stats], default=0.0), 4),
        "mean_pairwise_similarity": round(_mean([item.mean_pairwise_similarity for item in label_stats], default=0.0), 4),
        "mean_shape_similarity": round(_mean([item.mean_shape_similarity for item in label_stats], default=0.0), 4),
        "min_case_count": int(min_case_count),
        "anomaly_similarity_threshold": float(anomaly_similarity_threshold),
        "anomaly_shape_threshold": float(anomaly_shape_threshold),
        "top_stable_labels": [asdict(item) for item in label_stats[:10]],
        "anomaly_case_ids": sorted({item.case_id for item in anomalies}),
    }

    report_json = output_dir / "canonical_consistency_report.json"
    report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    label_stats_csv = output_dir / "canonical_label_stats.csv"
    with label_stats_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "canonical_name",
            "case_count",
            "presence_rate",
            "is_root_label",
            "mean_pairwise_similarity",
            "mean_shape_similarity",
            "min_pairwise_similarity",
            "max_pairwise_similarity",
            "mean_centroid_similarity",
            "mean_direction_similarity",
            "mean_length_similarity",
            "mean_subtree_similarity",
            "mean_depth_similarity",
            "mean_lambda_similarity",
            "worst_pair_case_left",
            "worst_pair_case_right",
            "best_pair_case_left",
            "best_pair_case_right",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in label_stats)

    anomalies_csv = output_dir / "canonical_case_anomalies.csv"
    with anomalies_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "cohort",
            "canonical_name",
            "overall_similarity_to_prototype",
            "shape_similarity_to_prototype",
            "centroid_similarity_to_prototype",
            "direction_similarity_to_prototype",
            "length_similarity_to_prototype",
            "subtree_similarity_to_prototype",
            "depth_similarity_to_prototype",
            "lambda_similarity_to_prototype",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in anomalies)

    return CanonicalConsistencyReport(
        summary_csv=summary_csv,
        report_json=report_json,
        label_stats_csv=label_stats_csv,
        anomalies_csv=anomalies_csv,
        total_cases=len(rows),
        total_labels=len(label_stats),
        mean_presence_rate=round(_mean([item.presence_rate for item in label_stats], default=0.0), 4),
        mean_pairwise_similarity=round(_mean([item.mean_pairwise_similarity for item in label_stats], default=0.0), 4),
        mean_shape_similarity=round(_mean([item.mean_shape_similarity for item in label_stats], default=0.0), 4),
        label_stats=label_stats,
        anomalies=anomalies,
    )
