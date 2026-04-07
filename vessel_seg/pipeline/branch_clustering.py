"""Branch-level clustering analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
import csv
import json

from .canonical_consistency import (
    CanonicalBranchSample,
    _branch_similarity,
    _load_summary_rows,
    _prototype,
    _sample_branch_payloads,
)


@dataclass(frozen=True)
class BranchClusteringConfig:
    """分支聚类输入契约。"""

    summary_csv: Path
    output_dir: Path
    clustering_mode: str = "side_first"
    side_assignments_csv: Path | None = None
    similarity_threshold: float = 0.80
    target_cluster_count: int | None = None
    match_max_case_branches: bool = False
    min_cluster_size: int = 3
    outlier_similarity_threshold: float = 0.70


@dataclass(frozen=True)
class BranchClusterAssignment:
    """单个分支样本的聚类结果。"""

    case_id: str
    cohort: str
    side_group: str
    canonical_name: str
    branch_id: int
    cluster_id: int
    prototype_case_id: str
    cluster_size: int
    overall_similarity_to_prototype: float
    shape_similarity_to_prototype: float
    centroid_similarity_to_prototype: float
    direction_similarity_to_prototype: float
    length_similarity_to_prototype: float
    subtree_similarity_to_prototype: float
    depth_similarity_to_prototype: float
    lambda_similarity_to_prototype: float
    is_outlier: bool


@dataclass(frozen=True)
class BranchClusterSummary:
    """单个 cluster 的摘要。"""

    cluster_id: int
    side_group: str
    cluster_size: int
    cluster_fraction: float
    prototype_case_id: str
    prototype_canonical_name: str
    mean_overall_similarity_to_prototype: float
    mean_shape_similarity_to_prototype: float
    mean_within_cluster_similarity: float
    mean_within_cluster_shape_similarity: float
    dominant_canonical_name: str
    dominant_canonical_ratio: float
    canonical_name_histogram: dict[str, int]
    member_case_ids: tuple[str, ...]
    outlier_count: int


@dataclass(frozen=True)
class BranchClusterGroupSummary:
    """左右系统优先分组后的组级摘要。"""

    side_group: str
    sample_count: int
    cluster_count: int
    target_cluster_count: int | None


@dataclass(frozen=True)
class BranchClusteringReport:
    """全局分支聚类报告。"""

    config: BranchClusteringConfig
    summary_csv: Path
    report_json: Path
    cluster_summary_csv: Path
    assignments_csv: Path
    outliers_csv: Path
    clustering_mode: str
    side_assignments_csv: Path | None
    total_cases: int
    total_samples: int
    cluster_count: int
    target_cluster_count: int | None
    singleton_cluster_count: int
    mean_cluster_size: float
    mean_assignment_similarity_to_prototype: float
    side_override_count: int
    group_summaries: list[BranchClusterGroupSummary]
    clusters: list[BranchClusterSummary]
    assignments: list[BranchClusterAssignment]


def _mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _pairwise_similarity_cache(samples: list[CanonicalBranchSample]) -> dict[tuple[int, int], dict[str, float]]:
    cache: dict[tuple[int, int], dict[str, float]] = {}
    for left_index, right_index in combinations(range(len(samples)), 2):
        score = _branch_similarity(samples[left_index], samples[right_index])
        cache[(left_index, right_index)] = score
        cache[(right_index, left_index)] = score
    for index in range(len(samples)):
        cache[(index, index)] = {
            "overall": 1.0,
            "shape": 1.0,
            "centroid": 1.0,
            "direction": 1.0,
            "length": 1.0,
            "subtree": 1.0,
            "depth": 1.0,
            "lambda": 1.0,
        }
    return cache


def _average_cross_similarity(
    left_cluster: list[int],
    right_cluster: list[int],
    cache: dict[tuple[int, int], dict[str, float]],
) -> float:
    scores = [
        0.75 * float(cache[(left_index, right_index)]["shape"]) + 0.25 * float(cache[(left_index, right_index)]["overall"])
        for left_index in left_cluster
        for right_index in right_cluster
    ]
    return _mean(scores, default=0.0)


def _group_sort_key(group_name: str) -> tuple[int, str]:
    if group_name == "LCA":
        return (0, group_name)
    if group_name == "RCA":
        return (1, group_name)
    if group_name.startswith("AUX"):
        return (2, group_name)
    if group_name == "MAIN":
        return (3, group_name)
    if group_name == "UNKNOWN":
        return (4, group_name)
    if group_name == "ALL":
        return (5, group_name)
    return (6, group_name)


def _sample_group_name(sample: CanonicalBranchSample, clustering_mode: str) -> str:
    if clustering_mode == "global":
        return "ALL"
    return str(sample.system_name or "UNKNOWN")


def _partition_samples(
    samples: list[CanonicalBranchSample],
    clustering_mode: str,
) -> dict[str, list[CanonicalBranchSample]]:
    grouped: dict[str, list[CanonicalBranchSample]] = {}
    for sample in samples:
        grouped.setdefault(_sample_group_name(sample, clustering_mode), []).append(sample)
    return dict(sorted(grouped.items(), key=lambda item: _group_sort_key(item[0])))


def _resolved_target_cluster_count(
    rows: list[dict[str, str]],
    *,
    target_cluster_count: int | None,
    match_max_case_branches: bool,
) -> int | None:
    resolved = target_cluster_count
    if match_max_case_branches:
        candidates = []
        for row in rows:
            try:
                candidates.append(int(row.get("num_branches", "")))
            except (TypeError, ValueError):
                continue
        if candidates:
            resolved = max(candidates)
    return resolved


def _allocate_target_cluster_counts(
    grouped_samples: dict[str, list[CanonicalBranchSample]],
    target_cluster_count: int | None,
    clustering_mode: str,
) -> dict[str, int | None]:
    if target_cluster_count is None:
        return {group_name: None for group_name in grouped_samples}
    if clustering_mode == "global":
        return {next(iter(grouped_samples.keys()), "ALL"): int(target_cluster_count)}

    non_empty = [(group_name, len(items)) for group_name, items in grouped_samples.items() if items]
    if not non_empty:
        return {group_name: None for group_name in grouped_samples}
    if int(target_cluster_count) <= len(non_empty):
        return {group_name: 1 for group_name, _ in non_empty}

    allocations = {group_name: 1 for group_name, _ in non_empty}
    remaining = int(target_cluster_count) - len(non_empty)
    total_samples = float(sum(count for _, count in non_empty))
    quotas: list[tuple[float, str]] = []
    for group_name, count in non_empty:
        raw_share = remaining * (float(count) / max(total_samples, 1.0))
        integer_part = int(raw_share)
        allocations[group_name] += integer_part
        quotas.append((raw_share - integer_part, group_name))
    assigned = sum(allocations.values())
    for _, group_name in sorted(quotas, key=lambda item: (-item[0], _group_sort_key(item[1]))):
        if assigned >= int(target_cluster_count):
            break
        allocations[group_name] += 1
        assigned += 1
    return allocations


def _load_side_group_overrides(path: Path) -> dict[tuple[str, int], str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    overrides: dict[tuple[str, int], str] = {}
    for row in rows:
        case_id = str(row.get("case_id", "") or "")
        side_group = str(row.get("side_group", "") or "")
        branch_id_raw = row.get("branch_id", "")
        if not case_id or not side_group:
            continue
        try:
            branch_id = int(branch_id_raw)
        except (TypeError, ValueError):
            continue
        overrides[(case_id, branch_id)] = side_group
    return overrides


def _apply_side_group_overrides(
    samples: list[CanonicalBranchSample],
    overrides: dict[tuple[str, int], str],
) -> tuple[list[CanonicalBranchSample], int]:
    overridden_samples: list[CanonicalBranchSample] = []
    override_count = 0
    for sample in samples:
        override = overrides.get((sample.case_id, sample.branch_id))
        if override is None or override == sample.system_name:
            overridden_samples.append(sample)
            continue
        overridden_samples.append(replace(sample, system_name=str(override)))
        override_count += 1
    return overridden_samples, override_count


def _merge_respects_case_sibling_exclusivity(
    left_cluster: list[int],
    right_cluster: list[int],
    samples: list[CanonicalBranchSample],
) -> bool:
    """同一病例、同一父节点下的兄弟分支不能被吞到同一类里。"""
    seen_keys: set[tuple[str, str]] = set()
    for index in list(left_cluster) + list(right_cluster):
        sample = samples[index]
        membership_key = (sample.case_id, sample.sibling_group_key)
        if membership_key in seen_keys:
            return False
        seen_keys.add(membership_key)
    return True


def _merge_respects_root_depth_purity(
    left_cluster: list[int],
    right_cluster: list[int],
    samples: list[CanonicalBranchSample],
) -> bool:
    """根分支不能和非根分支混簇。"""
    has_root = False
    has_non_root = False
    for index in list(left_cluster) + list(right_cluster):
        if samples[index].depth == 0:
            has_root = True
        else:
            has_non_root = True
        if has_root and has_non_root:
            return False
    return True


def _merge_respects_system_purity(
    left_cluster: list[int],
    right_cluster: list[int],
    samples: list[CanonicalBranchSample],
) -> bool:
    """左右冠系统不要混簇。"""
    systems = {
        samples[index].system_name
        for index in list(left_cluster) + list(right_cluster)
        if samples[index].system_name
    }
    return len(systems) <= 1


def _agglomerative_clusters(
    samples: list[CanonicalBranchSample],
    cache: dict[tuple[int, int], dict[str, float]],
    similarity_threshold: float,
    target_cluster_count: int | None = None,
) -> list[list[int]]:
    clusters = [[index] for index in range(len(samples))]
    while len(clusters) > 1:
        if target_cluster_count is not None and len(clusters) <= max(1, int(target_cluster_count)):
            break
        best_pair: tuple[int, int] | None = None
        best_score = -1.0
        for left_index, right_index in combinations(range(len(clusters)), 2):
            if not _merge_respects_case_sibling_exclusivity(
                clusters[left_index],
                clusters[right_index],
                samples,
            ):
                continue
            if not _merge_respects_root_depth_purity(
                clusters[left_index],
                clusters[right_index],
                samples,
            ):
                continue
            if not _merge_respects_system_purity(
                clusters[left_index],
                clusters[right_index],
                samples,
            ):
                continue
            score = _average_cross_similarity(
                clusters[left_index],
                clusters[right_index],
                cache,
            )
            if score > best_score:
                best_pair = (left_index, right_index)
                best_score = score
        if best_pair is None:
            break
        if target_cluster_count is None and best_score < float(similarity_threshold):
            break
        left_index, right_index = best_pair
        clusters[left_index] = sorted(clusters[left_index] + clusters[right_index])
        del clusters[right_index]
    clusters.sort(key=lambda cluster: (-len(cluster), tuple(cluster)))
    return clusters


def _cluster_medoid_index(
    cluster: list[int],
    cache: dict[tuple[int, int], dict[str, float]],
) -> int:
    return max(
        cluster,
        key=lambda index: (
            _mean(
                [
                    0.75 * float(cache[(index, other_index)]["shape"]) + 0.25 * float(cache[(index, other_index)]["overall"])
                    for other_index in cluster
                ],
                default=0.0,
            ),
            -index,
        ),
    )


def _within_cluster_similarity(
    cluster: list[int],
    cache: dict[tuple[int, int], dict[str, float]],
) -> tuple[float, float]:
    if len(cluster) <= 1:
        return (1.0, 1.0)
    overall = []
    shape = []
    for left_index, right_index in combinations(cluster, 2):
        similarity = cache[(left_index, right_index)]
        overall.append(float(similarity["overall"]))
        shape.append(float(similarity["shape"]))
    return (_mean(overall, default=1.0), _mean(shape, default=1.0))


def analyze_branch_clusters(
    summary_csv: Path,
    *,
    output_dir: Path | None = None,
    clustering_mode: str = "side_first",
    side_assignments_csv: Path | None = None,
    similarity_threshold: float = 0.80,
    target_cluster_count: int | None = None,
    match_max_case_branches: bool = False,
    min_cluster_size: int = 3,
    outlier_similarity_threshold: float = 0.70,
) -> BranchClusteringReport:
    """对所有病例分支做聚类分析。"""
    rows = _load_summary_rows(summary_csv)
    output_dir = output_dir or summary_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    clustering_mode = str(clustering_mode).strip().lower()
    if clustering_mode not in {"global", "side_first"}:
        raise ValueError(f"Unsupported clustering_mode: {clustering_mode}")

    samples: list[CanonicalBranchSample] = []
    for row in rows:
        samples.extend(_sample_branch_payloads(row))
    side_override_count = 0
    if side_assignments_csv is not None:
        overrides = _load_side_group_overrides(side_assignments_csv)
        samples, side_override_count = _apply_side_group_overrides(samples, overrides)

    resolved_target_cluster_count = _resolved_target_cluster_count(
        rows,
        target_cluster_count=target_cluster_count,
        match_max_case_branches=match_max_case_branches,
    )
    config = BranchClusteringConfig(
        summary_csv=summary_csv,
        output_dir=output_dir,
        clustering_mode=clustering_mode,
        side_assignments_csv=side_assignments_csv,
        similarity_threshold=float(similarity_threshold),
        target_cluster_count=resolved_target_cluster_count,
        match_max_case_branches=bool(match_max_case_branches),
        min_cluster_size=int(min_cluster_size),
        outlier_similarity_threshold=float(outlier_similarity_threshold),
    )
    grouped_samples = _partition_samples(samples, clustering_mode)
    group_target_counts = _allocate_target_cluster_counts(
        grouped_samples,
        resolved_target_cluster_count,
        clustering_mode,
    )

    assignments: list[BranchClusterAssignment] = []
    cluster_summaries: list[BranchClusterSummary] = []
    group_summaries: list[BranchClusterGroupSummary] = []
    assignment_similarity_values: list[float] = []
    next_cluster_id = 1
    for side_group, group_items in grouped_samples.items():
        if not group_items:
            continue
        group_cache = _pairwise_similarity_cache(group_items)
        local_clusters = _agglomerative_clusters(
            group_items,
            group_cache,
            similarity_threshold=float(similarity_threshold),
            target_cluster_count=group_target_counts.get(side_group),
        )
        group_summaries.append(
            BranchClusterGroupSummary(
                side_group=side_group,
                sample_count=len(group_items),
                cluster_count=len(local_clusters),
                target_cluster_count=group_target_counts.get(side_group),
            )
        )

        for cluster in local_clusters:
            cluster_id = next_cluster_id
            next_cluster_id += 1
            medoid_index = _cluster_medoid_index(cluster, group_cache)
            medoid_sample = group_items[medoid_index]
            prototype = _prototype([group_items[index] for index in cluster])
            within_cluster_similarity, within_cluster_shape_similarity = _within_cluster_similarity(cluster, group_cache)

            canonical_histogram: dict[str, int] = {}
            cluster_assignment_scores: list[float] = []
            cluster_assignment_shape_scores: list[float] = []
            member_case_ids: list[str] = []
            cluster_outlier_count = 0
            for index in cluster:
                sample = group_items[index]
                member_case_ids.append(sample.case_id)
                canonical_histogram[sample.canonical_name] = canonical_histogram.get(sample.canonical_name, 0) + 1
                similarity = _branch_similarity(sample, prototype)
                assignment_score = 0.75 * float(similarity["shape"]) + 0.25 * float(similarity["overall"])
                assignment_similarity_values.append(assignment_score)
                cluster_assignment_scores.append(assignment_score)
                cluster_assignment_shape_scores.append(float(similarity["shape"]))
                is_outlier = len(cluster) < int(min_cluster_size) or assignment_score < float(outlier_similarity_threshold)
                if is_outlier:
                    cluster_outlier_count += 1
                assignments.append(
                    BranchClusterAssignment(
                        case_id=sample.case_id,
                        cohort=sample.cohort,
                        side_group=side_group,
                        canonical_name=sample.canonical_name,
                        branch_id=sample.branch_id,
                        cluster_id=cluster_id,
                        prototype_case_id=medoid_sample.case_id,
                        cluster_size=len(cluster),
                        overall_similarity_to_prototype=round(float(similarity["overall"]), 4),
                        shape_similarity_to_prototype=round(float(similarity["shape"]), 4),
                        centroid_similarity_to_prototype=round(float(similarity["centroid"]), 4),
                        direction_similarity_to_prototype=round(float(similarity["direction"]), 4),
                        length_similarity_to_prototype=round(float(similarity["length"]), 4),
                        subtree_similarity_to_prototype=round(float(similarity["subtree"]), 4),
                        depth_similarity_to_prototype=round(float(similarity["depth"]), 4),
                        lambda_similarity_to_prototype=round(float(similarity["lambda"]), 4),
                        is_outlier=is_outlier,
                    )
                )

            dominant_name, dominant_count = max(canonical_histogram.items(), key=lambda item: (item[1], item[0]))
            cluster_summaries.append(
                BranchClusterSummary(
                    cluster_id=cluster_id,
                    side_group=side_group,
                    cluster_size=len(cluster),
                    cluster_fraction=round(len(cluster) / float(len(samples) or 1), 4),
                    prototype_case_id=medoid_sample.case_id,
                    prototype_canonical_name=medoid_sample.canonical_name,
                    mean_overall_similarity_to_prototype=round(_mean(cluster_assignment_scores, default=1.0), 4),
                    mean_shape_similarity_to_prototype=round(_mean(cluster_assignment_shape_scores, default=1.0), 4),
                    mean_within_cluster_similarity=round(within_cluster_similarity, 4),
                    mean_within_cluster_shape_similarity=round(within_cluster_shape_similarity, 4),
                    dominant_canonical_name=dominant_name,
                    dominant_canonical_ratio=round(dominant_count / float(len(cluster) or 1), 4),
                    canonical_name_histogram=dict(sorted(canonical_histogram.items(), key=lambda item: (-item[1], item[0]))),
                    member_case_ids=tuple(sorted(member_case_ids)),
                    outlier_count=cluster_outlier_count,
                )
            )

    cluster_summaries.sort(key=lambda item: (-item.cluster_size, item.cluster_id))
    group_summaries.sort(key=lambda item: _group_sort_key(item.side_group))
    assignments.sort(key=lambda item: (_group_sort_key(item.side_group), item.cluster_id, item.case_id, item.canonical_name))

    report_payload = {
        "summary_csv": str(summary_csv),
        "clustering_mode": clustering_mode,
        "side_assignments_csv": None if side_assignments_csv is None else str(side_assignments_csv),
        "side_override_count": side_override_count,
        "input_contract": {
            "summary_csv": "batch overview summary with case_id/cohort/overview_path",
            "per_case_dependencies": [
                "stages/03_centerline_repair/tree.json",
                "stages/03_centerline_repair/centerline_repaired.vtp",
            ],
        },
        "workflow": [
            "extract normalized branch samples",
            "assign left/right side group first",
            "cluster each side group independently",
            "write assignments, summaries, outliers",
        ],
        "total_cases": len(rows),
        "total_samples": len(samples),
        "cluster_count": len(cluster_summaries),
        "target_cluster_count": resolved_target_cluster_count,
        "singleton_cluster_count": sum(1 for cluster in cluster_summaries if cluster.cluster_size == 1),
        "mean_cluster_size": round(_mean([float(cluster.cluster_size) for cluster in cluster_summaries], default=0.0), 4),
        "mean_assignment_similarity_to_prototype": round(_mean(assignment_similarity_values, default=0.0), 4),
        "similarity_threshold": float(similarity_threshold),
        "match_max_case_branches": bool(match_max_case_branches),
        "min_cluster_size": int(min_cluster_size),
        "outlier_similarity_threshold": float(outlier_similarity_threshold),
        "group_summaries": [asdict(item) for item in group_summaries],
        "largest_clusters": [
            {
                "cluster_id": item.cluster_id,
                "side_group": item.side_group,
                "cluster_size": item.cluster_size,
                "dominant_canonical_name": item.dominant_canonical_name,
                "dominant_canonical_ratio": item.dominant_canonical_ratio,
                "prototype_case_id": item.prototype_case_id,
                "prototype_canonical_name": item.prototype_canonical_name,
            }
            for item in cluster_summaries[:10]
        ],
    }

    report_json = output_dir / "branch_clustering_report.json"
    report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    cluster_summary_csv = output_dir / "branch_cluster_summary.csv"
    with cluster_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cluster_id",
            "side_group",
            "cluster_size",
            "cluster_fraction",
            "prototype_case_id",
            "prototype_canonical_name",
            "mean_overall_similarity_to_prototype",
            "mean_shape_similarity_to_prototype",
            "mean_within_cluster_similarity",
            "mean_within_cluster_shape_similarity",
            "dominant_canonical_name",
            "dominant_canonical_ratio",
            "canonical_name_histogram",
            "member_case_ids",
            "outlier_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in cluster_summaries:
            payload = asdict(item)
            payload["canonical_name_histogram"] = json.dumps(item.canonical_name_histogram, ensure_ascii=False)
            payload["member_case_ids"] = json.dumps(item.member_case_ids, ensure_ascii=False)
            writer.writerow(payload)

    assignments_csv = output_dir / "branch_cluster_assignments.csv"
    with assignments_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "cohort",
            "side_group",
            "canonical_name",
            "branch_id",
            "cluster_id",
            "prototype_case_id",
            "cluster_size",
            "overall_similarity_to_prototype",
            "shape_similarity_to_prototype",
            "centroid_similarity_to_prototype",
            "direction_similarity_to_prototype",
            "length_similarity_to_prototype",
            "subtree_similarity_to_prototype",
            "depth_similarity_to_prototype",
            "lambda_similarity_to_prototype",
            "is_outlier",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in assignments)

    outliers_csv = output_dir / "branch_cluster_outliers.csv"
    with outliers_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "cohort",
            "side_group",
            "canonical_name",
            "branch_id",
            "cluster_id",
            "prototype_case_id",
            "cluster_size",
            "overall_similarity_to_prototype",
            "shape_similarity_to_prototype",
            "centroid_similarity_to_prototype",
            "direction_similarity_to_prototype",
            "length_similarity_to_prototype",
            "subtree_similarity_to_prototype",
            "depth_similarity_to_prototype",
            "lambda_similarity_to_prototype",
            "is_outlier",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in assignments if item.is_outlier)

    return BranchClusteringReport(
        config=config,
        summary_csv=summary_csv,
        report_json=report_json,
        cluster_summary_csv=cluster_summary_csv,
        assignments_csv=assignments_csv,
        outliers_csv=outliers_csv,
        clustering_mode=clustering_mode,
        side_assignments_csv=side_assignments_csv,
        total_cases=len(rows),
        total_samples=len(samples),
        cluster_count=len(cluster_summaries),
        target_cluster_count=resolved_target_cluster_count,
        singleton_cluster_count=sum(1 for cluster in cluster_summaries if cluster.cluster_size == 1),
        mean_cluster_size=round(_mean([float(cluster.cluster_size) for cluster in cluster_summaries], default=0.0), 4),
        mean_assignment_similarity_to_prototype=round(_mean(assignment_similarity_values, default=0.0), 4),
        side_override_count=side_override_count,
        group_summaries=group_summaries,
        clusters=cluster_summaries,
        assignments=assignments,
    )
