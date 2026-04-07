import json
from pathlib import Path

from vessel_seg.pipeline.semantic_topology import (
    apply_semantic_topology,
    assign_semantic_topology,
    enrich_tree_with_semantic_topology,
    fit_semantic_geometry_prior_from_tree_payloads,
)


def _branch(
    branch_id: int,
    *,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    centroid: tuple[float, float, float],
    parent: int | None,
    length_mm: float,
    lambda_pos: float | None = None,
) -> dict[str, object]:
    return {
        "branch_id": branch_id,
        "length_mm": length_mm,
        "start": list(start),
        "end": list(end),
        "centroid": list(centroid),
        "attachment": {
            "parent": parent,
            "lambda_pos": lambda_pos,
            "theta_deg": 0.0 if parent is not None else None,
            "phi_deg": 0.0 if parent is not None else None,
        },
    }


def _toy_tree_payload() -> dict[str, object]:
    return {
        "roots": [10, 1],
        "branches": [
            _branch(1, start=(10.0, 0.0, 0.0), end=(11.0, 0.0, -25.0), centroid=(10.8, 0.2, -12.0), parent=None, length_mm=28.0),
            _branch(2, start=(11.0, 0.0, -25.0), end=(11.5, 1.0, -60.0), centroid=(11.2, 0.8, -43.0), parent=1, length_mm=36.0, lambda_pos=0.15),
            _branch(3, start=(11.0, 0.0, -25.0), end=(18.0, 6.5, -35.0), centroid=(15.0, 4.5, -30.0), parent=1, length_mm=22.0, lambda_pos=0.22),
            _branch(4, start=(11.0, 0.0, -25.0), end=(14.0, 2.5, -34.0), centroid=(13.0, 2.0, -29.0), parent=1, length_mm=12.0, lambda_pos=0.28),
            _branch(7, start=(11.5, 1.0, -60.0), end=(12.0, 1.2, -86.0), centroid=(11.8, 1.1, -74.0), parent=2, length_mm=26.0, lambda_pos=0.42),
            _branch(5, start=(11.5, 1.0, -60.0), end=(17.5, 6.5, -72.0), centroid=(15.0, 5.0, -67.0), parent=2, length_mm=16.0, lambda_pos=0.34),
            _branch(6, start=(11.5, 1.0, -60.0), end=(6.0, -2.5, -70.0), centroid=(8.5, -1.0, -66.0), parent=2, length_mm=12.0, lambda_pos=0.38),
            _branch(10, start=(-10.0, 0.0, 0.0), end=(-12.5, -0.4, -28.0), centroid=(-11.2, -0.2, -14.0), parent=None, length_mm=30.0),
            _branch(11, start=(-12.5, -0.4, -28.0), end=(-13.0, -0.5, -62.0), centroid=(-12.8, -0.4, -45.0), parent=10, length_mm=35.0, lambda_pos=0.18),
            _branch(12, start=(-13.0, -0.5, -62.0), end=(-16.0, -3.0, -77.0), centroid=(-14.5, -1.8, -70.0), parent=11, length_mm=18.0, lambda_pos=0.82),
        ],
    }


def test_semantic_topology_assigns_major_coronary_families() -> None:
    result = assign_semantic_topology(_toy_tree_payload())
    family_map = {assignment.branch_id: assignment.semantic_family for assignment in result.assignments}

    assert family_map[1] == "LeftMain"
    assert family_map[2] == "LeftAnteriorDescending"
    assert family_map[3] == "LeftCircumflex"
    assert family_map[4] == "RamusIntermedius"
    assert family_map[7] == "LeftAnteriorDescending"
    assert family_map[5] == "DiagonalBranch"
    assert family_map[6] == "SeptalPerforator"
    assert family_map[10] == "RightCoronaryArtery"
    assert family_map[11] == "RightCoronaryArtery"
    assert family_map[12] in {
        "RightCoronaryArtery",
        "RightPosteriorDescending",
        "PosterolateralBranch",
        "RightMarginal",
    }
    assert result.consistency.required_coverage >= 0.99
    assert result.consistency.unmatched_branches == 0
    assert result.consistency.score >= 80.0


def test_apply_semantic_topology_enriches_tree_and_writes_json(tmp_path: Path) -> None:
    tree_path = tmp_path / "tree.json"
    semantic_path = tmp_path / "semantic_topology.json"
    tree_path.write_text(json.dumps(_toy_tree_payload(), indent=2), encoding="utf-8")

    result = apply_semantic_topology(tree_path, semantic_path)

    enriched = json.loads(tree_path.read_text(encoding="utf-8"))
    assert semantic_path.exists()
    assert enriched["semantic_topology"]["version"] == result.version
    assert all("semantic" in branch for branch in enriched["branches"])


def test_fit_semantic_geometry_prior_from_semantic_payloads() -> None:
    payload = enrich_tree_with_semantic_topology(_toy_tree_payload())
    prior = fit_semantic_geometry_prior_from_tree_payloads([payload, payload], min_samples_per_family=1)
    families = {item.family for item in prior.families}

    assert "LeftMain" in families
    assert "LeftAnteriorDescending" in families
    assert "LeftCircumflex" in families
    assert "RightCoronaryArtery" in families
