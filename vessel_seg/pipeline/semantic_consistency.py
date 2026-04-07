"""Semantic topology consistency analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
from itertools import combinations


@dataclass(frozen=True)
class SemanticConsistencyReport:
    """批量语义拓扑一致性报告。"""

    summary_csv: Path
    report_json: Path
    anomalies_csv: Path
    total_cases: int
    unique_semantic_signatures: int
    mean_semantic_score: float
    mean_semantic_jaccard: float
    cohort_mean_scores: dict[str, float]
    cohort_mean_jaccard: dict[str, float]
    anomalies: list[dict[str, object]]


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _family_set(signature: str) -> set[str]:
    families: set[str] = set()
    for item in signature.split("|"):
        token = item.strip()
        if not token:
            continue
        families.add(token.split(".")[0])
    return families


def _mean_jaccard(items: dict[str, set[str]]) -> float:
    if len(items) <= 1:
        return 1.0
    scores = []
    for left_key, right_key in combinations(items, 2):
        left = items[left_key]
        right = items[right_key]
        union = left | right
        scores.append(len(left & right) / float(len(union) or 1))
    return float(sum(scores) / len(scores)) if scores else 1.0


def analyze_semantic_consistency(
    summary_csv: Path,
    *,
    output_dir: Path | None = None,
    anomaly_score_threshold: float = 75.0,
    anomaly_branch_threshold: int = 12,
) -> SemanticConsistencyReport:
    """读取 batch summary，输出模板一致性统计和异常病例列表。"""
    rows = _load_summary_rows(summary_csv)
    output_dir = output_dir or summary_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    family_sets = {row["case_id"]: _family_set(row.get("semantic_signature", "")) for row in rows}
    mean_semantic_score = sum(float(row.get("semantic_score", 0.0)) for row in rows) / float(len(rows) or 1)
    unique_semantic_signatures = len({row.get("semantic_signature", "") for row in rows})
    mean_semantic_jaccard = _mean_jaccard(family_sets)

    cohort_mean_scores: dict[str, float] = {}
    cohort_mean_jaccard: dict[str, float] = {}
    for cohort in sorted({row.get("cohort", "") for row in rows}):
        subset = [row for row in rows if row.get("cohort", "") == cohort]
        cohort_mean_scores[cohort] = sum(float(row.get("semantic_score", 0.0)) for row in subset) / float(len(subset) or 1)
        cohort_sets = {row["case_id"]: family_sets[row["case_id"]] for row in subset}
        cohort_mean_jaccard[cohort] = _mean_jaccard(cohort_sets)

    anomalies: list[dict[str, object]] = []
    for row in rows:
        semantic_score = float(row.get("semantic_score", 0.0))
        num_roots = int(row.get("num_roots", 0))
        unmatched = int(row.get("semantic_unmatched_branches", 0))
        warning_count = int(row.get("semantic_warning_count", 0))
        num_branches = int(row.get("num_branches", 0))
        if (
            semantic_score < anomaly_score_threshold
            or num_roots != 2
            or unmatched > 0
            or warning_count > 0
            or num_branches > anomaly_branch_threshold
        ):
            anomalies.append(
                {
                    "case_id": row["case_id"],
                    "cohort": row.get("cohort", ""),
                    "semantic_score": semantic_score,
                    "num_roots": num_roots,
                    "num_branches": num_branches,
                    "dominance": row.get("dominance", "Unknown"),
                    "semantic_unmatched_branches": unmatched,
                    "semantic_warning_count": warning_count,
                    "semantic_signature": row.get("semantic_signature", ""),
                }
            )
    anomalies.sort(
        key=lambda item: (
            float(item["semantic_score"]),
            -int(item["semantic_warning_count"]),
            -int(item["semantic_unmatched_branches"]),
            -int(item["num_branches"]),
            str(item["case_id"]),
        )
    )

    report_payload = {
        "summary_csv": str(summary_csv),
        "total_cases": len(rows),
        "unique_semantic_signatures": unique_semantic_signatures,
        "mean_semantic_score": round(mean_semantic_score, 4),
        "mean_semantic_jaccard": round(mean_semantic_jaccard, 4),
        "cohort_mean_scores": {key: round(value, 4) for key, value in cohort_mean_scores.items()},
        "cohort_mean_jaccard": {key: round(value, 4) for key, value in cohort_mean_jaccard.items()},
        "anomaly_score_threshold": anomaly_score_threshold,
        "anomaly_branch_threshold": anomaly_branch_threshold,
        "anomaly_case_ids": [item["case_id"] for item in anomalies],
    }

    report_json = output_dir / "semantic_consistency_report.json"
    report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    anomalies_csv = output_dir / "semantic_anomalies.csv"
    with anomalies_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "cohort",
            "semantic_score",
            "num_roots",
            "num_branches",
            "dominance",
            "semantic_unmatched_branches",
            "semantic_warning_count",
            "semantic_signature",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(anomalies)

    return SemanticConsistencyReport(
        summary_csv=summary_csv,
        report_json=report_json,
        anomalies_csv=anomalies_csv,
        total_cases=len(rows),
        unique_semantic_signatures=unique_semantic_signatures,
        mean_semantic_score=round(mean_semantic_score, 4),
        mean_semantic_jaccard=round(mean_semantic_jaccard, 4),
        cohort_mean_scores={key: round(value, 4) for key, value in cohort_mean_scores.items()},
        cohort_mean_jaccard={key: round(value, 4) for key, value in cohort_mean_jaccard.items()},
        anomalies=anomalies,
    )
