import csv
import json
from pathlib import Path

from vessel_seg.pipeline.canonical_consistency import _branch_similarity, _sample_branch_payloads
from vessel_seg.pipeline.branch_clustering import analyze_branch_clusters
from vessel_seg.pipeline.naming import enrich_tree_with_canonical_names


def _branch(
    branch_id: int,
    *,
    length_mm: float,
    centroid: tuple[float, float, float],
    parent: int | None,
    end: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "branch_id": branch_id,
        "length_mm": length_mm,
        "start": list(centroid),
        "end": list(end),
        "centroid": list(centroid),
        "attachment": {
            "parent": parent,
            "lambda_pos": 0.2 if parent is not None else None,
            "theta_deg": 0.0 if parent is not None else None,
            "phi_deg": 0.0 if parent is not None else None,
        },
    }


def _tree_payload(sys_a_end: tuple[float, float, float]) -> dict[str, object]:
    return enrich_tree_with_canonical_names(
        {
            "roots": [1, 10],
            "branches": [
                _branch(1, length_mm=100.0, centroid=(10.0, 0.0, 0.0), parent=None, end=sys_a_end),
                _branch(10, length_mm=90.0, centroid=(-10.0, 0.0, 0.0), parent=None, end=(-16.0, 0.0, 0.0)),
            ],
        }
    )


def _tree_payload_with_sys_a_siblings() -> dict[str, object]:
    return enrich_tree_with_canonical_names(
        {
            "roots": [1, 10],
            "branches": [
                _branch(1, length_mm=100.0, centroid=(10.0, 0.0, 0.0), parent=None, end=(18.0, 0.0, 0.0)),
                _branch(2, length_mm=48.0, centroid=(16.0, 8.0, 0.0), parent=1, end=(24.0, 14.0, 0.0)),
                _branch(3, length_mm=46.0, centroid=(17.0, 0.0, 4.0), parent=1, end=(26.0, 0.0, 7.0)),
                _branch(4, length_mm=44.0, centroid=(16.0, -7.0, -1.0), parent=1, end=(24.0, -13.0, -2.0)),
                _branch(10, length_mm=92.0, centroid=(-10.0, 0.0, 0.0), parent=None, end=(-17.0, 0.0, 0.0)),
            ],
        }
    )


def _tree_payload_with_orthogonal_children() -> dict[str, object]:
    return enrich_tree_with_canonical_names(
        {
            "roots": [1, 10],
            "branches": [
                _branch(1, length_mm=100.0, centroid=(10.0, 0.0, 0.0), parent=None, end=(18.0, 0.0, 0.0)),
                _branch(2, length_mm=52.0, centroid=(18.0, 10.0, 0.0), parent=1, end=(26.0, 18.0, 0.0)),
                _branch(3, length_mm=50.0, centroid=(18.0, 0.0, 10.0), parent=1, end=(26.0, 0.0, 18.0)),
                _branch(10, length_mm=92.0, centroid=(-10.0, 0.0, 0.0), parent=None, end=(-17.0, 0.0, 0.0)),
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


def test_analyze_branch_clusters_supports_unlabeled_global_clustering(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    overview_a = _write_case(tmp_path, "Case_A", _tree_payload((16.0, 0.0, 0.0)))
    overview_b = _write_case(tmp_path, "Case_B", _tree_payload((16.2, 0.2, 0.0)))
    overview_c = _write_case(tmp_path, "Case_C", _tree_payload((9.0, 9.0, 0.0)))

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

    report = analyze_branch_clusters(
        summary_csv,
        output_dir=tmp_path / "analysis",
        similarity_threshold=0.82,
        min_cluster_size=2,
        outlier_similarity_threshold=0.78,
    )

    assert report.total_cases == 3
    assert report.total_samples == 6
    assert report.cluster_count >= 2
    assert len(report.assignments) == 6
    assert any(cluster.dominant_canonical_name == "SYS_A" for cluster in report.clusters)
    assert any(cluster.dominant_canonical_name == "SYS_B" for cluster in report.clusters)
    assert report.report_json.exists()
    assert report.cluster_summary_csv.exists()
    assert report.assignments_csv.exists()
    assert report.outliers_csv.exists()


def test_analyze_branch_clusters_can_match_target_cluster_count(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    overview_a = _write_case(tmp_path, "Case_A", _tree_payload((16.0, 0.0, 0.0)))
    overview_b = _write_case(tmp_path, "Case_B", _tree_payload((16.2, 0.2, 0.0)))
    overview_c = _write_case(tmp_path, "Case_C", _tree_payload((9.0, 9.0, 0.0)))

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "cohort", "num_branches", "overview_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "Case_A", "cohort": "Normal", "num_branches": 2, "overview_path": overview_a},
                {"case_id": "Case_B", "cohort": "Normal", "num_branches": 2, "overview_path": overview_b},
                {"case_id": "Case_C", "cohort": "Diseased", "num_branches": 2, "overview_path": overview_c},
            ]
        )

    report = analyze_branch_clusters(
        summary_csv,
        output_dir=tmp_path / "analysis_target",
        similarity_threshold=0.99,
        target_cluster_count=2,
        min_cluster_size=2,
        outlier_similarity_threshold=0.78,
    )
    assert report.cluster_count >= 2
    assert report.target_cluster_count == 2

    report_auto = analyze_branch_clusters(
        summary_csv,
        output_dir=tmp_path / "analysis_auto_target",
        similarity_threshold=0.99,
        match_max_case_branches=True,
        min_cluster_size=2,
        outlier_similarity_threshold=0.78,
    )
    assert report_auto.cluster_count >= 2
    assert report_auto.target_cluster_count == 2


def test_analyze_branch_clusters_keeps_same_parent_siblings_separate(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    overview_a = _write_case(tmp_path, "Case_A", _tree_payload_with_sys_a_siblings())
    overview_b = _write_case(tmp_path, "Case_B", _tree_payload_with_sys_a_siblings())

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "cohort", "num_branches", "overview_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "Case_A", "cohort": "Normal", "num_branches": 5, "overview_path": overview_a},
                {"case_id": "Case_B", "cohort": "Normal", "num_branches": 5, "overview_path": overview_b},
            ]
        )

    report = analyze_branch_clusters(
        summary_csv,
        output_dir=tmp_path / "analysis_sibling_exclusive",
        similarity_threshold=0.99,
        target_cluster_count=2,
        min_cluster_size=1,
        outlier_similarity_threshold=0.50,
    )

    by_cluster: dict[int, list[tuple[str, str]]] = {}
    for assignment in report.assignments:
        by_cluster.setdefault(assignment.cluster_id, []).append((assignment.case_id, assignment.canonical_name))

    assert report.cluster_count >= 3
    for members in by_cluster.values():
        seen: set[tuple[str, str]] = set()
        for case_id, canonical_name in members:
            parent_key = canonical_name.rsplit(".", 1)[0] if "." in canonical_name else "ROOTS"
            membership_key = (case_id, parent_key)
            assert membership_key not in seen
            seen.add(membership_key)


def test_branch_similarity_prefers_same_course_direction_over_orthogonal_direction(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    overview_a = _write_case(tmp_path, "Case_A", _tree_payload_with_orthogonal_children())
    overview_b = _write_case(tmp_path, "Case_B", _tree_payload_with_orthogonal_children())

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "cohort", "num_branches", "overview_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "Case_A", "cohort": "Normal", "num_branches": 4, "overview_path": overview_a},
                {"case_id": "Case_B", "cohort": "Normal", "num_branches": 4, "overview_path": overview_b},
            ]
        )

    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8", newline="")))
    samples: dict[tuple[str, int], object] = {}
    for row in rows:
        for sample in _sample_branch_payloads(row):
            samples[(sample.case_id, sample.branch_id)] = sample

    same_course = _branch_similarity(samples[("Case_A", 2)], samples[("Case_B", 2)])
    orthogonal_course = _branch_similarity(samples[("Case_A", 2)], samples[("Case_B", 3)])

    assert same_course["shape"] > orthogonal_course["shape"]
    assert same_course["overall"] > orthogonal_course["overall"]


def test_analyze_branch_clusters_keeps_left_right_systems_separate(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    overview_a = _write_case(tmp_path, "Case_A", _tree_payload((16.0, 0.0, 0.0)))
    overview_b = _write_case(tmp_path, "Case_B", _tree_payload((16.1, 0.1, 0.0)))

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "cohort", "num_branches", "overview_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "Case_A", "cohort": "Normal", "num_branches": 2, "overview_path": overview_a},
                {"case_id": "Case_B", "cohort": "Normal", "num_branches": 2, "overview_path": overview_b},
            ]
        )

    report = analyze_branch_clusters(
        summary_csv,
        output_dir=tmp_path / "analysis_lr_hard",
        similarity_threshold=0.99,
        target_cluster_count=1,
        min_cluster_size=1,
        outlier_similarity_threshold=0.50,
    )

    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8", newline="")))
    sample_by_branch = {}
    for row in rows:
        for sample in _sample_branch_payloads(row):
            sample_by_branch[(sample.case_id, sample.branch_id)] = sample

    systems_by_cluster: dict[int, set[str]] = {}
    for assignment in report.assignments:
        sample = sample_by_branch[(assignment.case_id, assignment.branch_id)]
        systems_by_cluster.setdefault(assignment.cluster_id, set()).add(sample.system_name)

    assert report.cluster_count >= 2
    assert all(len(systems) == 1 for systems in systems_by_cluster.values())


def test_analyze_branch_clusters_side_first_reports_lca_rca_groups(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    overview_a = _write_case(tmp_path, "Case_A", _tree_payload((16.0, 0.0, 0.0)))
    overview_b = _write_case(tmp_path, "Case_B", _tree_payload((16.1, 0.1, 0.0)))

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "cohort", "num_branches", "overview_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "Case_A", "cohort": "Normal", "num_branches": 2, "overview_path": overview_a},
                {"case_id": "Case_B", "cohort": "Normal", "num_branches": 2, "overview_path": overview_b},
            ]
        )

    report = analyze_branch_clusters(
        summary_csv,
        output_dir=tmp_path / "analysis_side_first",
        clustering_mode="side_first",
        similarity_threshold=0.99,
        target_cluster_count=2,
        min_cluster_size=1,
        outlier_similarity_threshold=0.50,
    )

    assert report.clustering_mode == "side_first"
    assert {item.side_group for item in report.group_summaries} == {"LCA", "RCA"}
    assert {item.side_group for item in report.assignments} == {"LCA", "RCA"}


def test_analyze_branch_clusters_can_override_side_groups_from_assignments_csv(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    side_csv = tmp_path / "side_assignments.csv"
    overview_a = _write_case(tmp_path, "Case_A", _tree_payload((16.0, 0.0, 0.0)))
    overview_b = _write_case(tmp_path, "Case_B", _tree_payload((16.1, 0.1, 0.0)))

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "cohort", "num_branches", "overview_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "Case_A", "cohort": "Normal", "num_branches": 2, "overview_path": overview_a},
                {"case_id": "Case_B", "cohort": "Normal", "num_branches": 2, "overview_path": overview_b},
            ]
        )

    with side_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "branch_id", "side_group"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "Case_A", "branch_id": 1, "side_group": "RCA"},
                {"case_id": "Case_A", "branch_id": 10, "side_group": "RCA"},
                {"case_id": "Case_B", "branch_id": 1, "side_group": "RCA"},
                {"case_id": "Case_B", "branch_id": 10, "side_group": "RCA"},
            ]
        )

    report = analyze_branch_clusters(
        summary_csv,
        output_dir=tmp_path / "analysis_side_override",
        clustering_mode="side_first",
        side_assignments_csv=side_csv,
        similarity_threshold=0.99,
        target_cluster_count=1,
        min_cluster_size=1,
        outlier_similarity_threshold=0.50,
    )

    assert report.side_override_count == 2
    assert {item.side_group for item in report.group_summaries} == {"RCA"}
    assert {item.side_group for item in report.assignments} == {"RCA"}
