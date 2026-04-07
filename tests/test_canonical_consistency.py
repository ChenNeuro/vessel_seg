import csv
import json
from pathlib import Path

from vessel_seg.pipeline.canonical_consistency import analyze_canonical_consistency
from vessel_seg.pipeline.naming import enrich_tree_with_canonical_names


def _branch(
    branch_id: int,
    *,
    length_mm: float,
    centroid: tuple[float, float, float],
    parent: int | None,
    end: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    start = centroid
    end = end if end is not None else centroid
    return {
        "branch_id": branch_id,
        "length_mm": length_mm,
        "start": list(start),
        "end": list(end),
        "centroid": list(centroid),
        "attachment": {
            "parent": parent,
            "lambda_pos": 0.2 if parent is not None else None,
            "theta_deg": 0.0 if parent is not None else None,
            "phi_deg": 0.0 if parent is not None else None,
        },
    }


def _tree_payload(left_child_end: tuple[float, float, float]) -> dict[str, object]:
    return enrich_tree_with_canonical_names(
        {
            "roots": [1, 10],
            "branches": [
                _branch(1, length_mm=100.0, centroid=(10.0, 0.0, 0.0), parent=None, end=(15.0, 0.0, 0.0)),
                _branch(2, length_mm=65.0, centroid=(14.0, 2.0, 0.0), parent=1, end=left_child_end),
                _branch(10, length_mm=90.0, centroid=(-10.0, 0.0, 0.0), parent=None, end=(-15.0, 0.0, 0.0)),
                _branch(11, length_mm=55.0, centroid=(-14.0, 0.0, 0.0), parent=10, end=(-18.0, 0.0, 0.0)),
            ],
        }
    )


def _write_case(tmp_path: Path, case_id: str, payload: dict[str, object]) -> str:
    case_dir = tmp_path / "cases" / case_id / "stages"
    tree_dir = case_dir / "03_centerline_repair"
    render_dir = case_dir / "05_rendering"
    tree_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    (tree_dir / "tree.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    overview = render_dir / "overview.png"
    overview.write_bytes(b"")
    return str(overview)


def test_analyze_canonical_consistency_reports_root_and_child_stability(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    overview_a = _write_case(tmp_path, "Case_A", _tree_payload((18.0, 3.0, 0.0)))
    overview_b = _write_case(tmp_path, "Case_B", _tree_payload((18.2, 2.8, 0.0)))
    overview_c = _write_case(tmp_path, "Case_C", _tree_payload((8.0, -8.0, 0.0)))

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "cohort", "overview_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "Case_A", "cohort": "Normal", "overview_path": overview_a},
                {"case_id": "Case_B", "cohort": "Normal", "overview_path": overview_b},
                {"case_id": "Case_C", "cohort": "Diseased", "overview_path": overview_c},
            ]
        )

    report = analyze_canonical_consistency(summary_csv, output_dir=tmp_path / "analysis", anomaly_similarity_threshold=0.8, anomaly_shape_threshold=0.7)

    stats_by_name = {item.canonical_name: item for item in report.label_stats}
    assert report.total_cases == 3
    assert "SYS_A" in stats_by_name
    assert "SYS_A.01" in stats_by_name
    assert stats_by_name["SYS_A"].presence_rate == 1.0
    assert stats_by_name["SYS_A.01"].mean_pairwise_similarity < stats_by_name["SYS_A"].mean_pairwise_similarity
    assert report.report_json.exists()
    assert report.label_stats_csv.exists()
    assert report.anomalies_csv.exists()
