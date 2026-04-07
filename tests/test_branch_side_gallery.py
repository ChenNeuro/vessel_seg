from vessel_seg.pipeline.branch_side_gallery import _group_case_hits


def test_group_case_hits_collects_lca_and_rca_rows_separately() -> None:
    rows = [
        {"case_id": "Normal_1", "side_group": "LCA", "branch_id": "1"},
        {"case_id": "Normal_1", "side_group": "RCA", "branch_id": "4"},
        {"case_id": "Normal_2", "side_group": "LCA", "branch_id": "2"},
        {"case_id": "Normal_2", "side_group": "AUX_01", "branch_id": "7"},
    ]

    grouped = _group_case_hits(rows, ("LCA", "RCA"))

    assert set(grouped) == {"LCA", "RCA"}
    assert set(grouped["LCA"]) == {"Normal_1", "Normal_2"}
    assert set(grouped["RCA"]) == {"Normal_1"}
    assert grouped["LCA"]["Normal_1"][0]["branch_id"] == "1"
