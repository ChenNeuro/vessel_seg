from vessel_seg.pipeline.naming import assign_canonical_branch_names


def _make_tree_payload(branches, roots):
    return {
        "roots": roots,
        "branches": branches,
    }


def _branch(branch_id, length_mm, centroid, parent):
    return {
        "branch_id": branch_id,
        "length_mm": length_mm,
        "start": list(centroid),
        "end": list(centroid),
        "centroid": list(centroid),
        "attachment": {
            "parent": parent,
            "lambda_pos": 0.0 if parent is not None else None,
            "theta_deg": 0.0 if parent is not None else None,
            "phi_deg": 0.0 if parent is not None else None,
        },
    }


def _centroid_name_map(result, payload):
    by_id = {int(branch["branch_id"]): tuple(branch["centroid"]) for branch in payload["branches"]}
    return {
        by_id[branch.branch_id]: branch.canonical_name
        for branch in result.branches
    }


def test_canonical_naming_is_stable_under_branch_id_permutation() -> None:
    payload_a = _make_tree_payload(
        [
            _branch(10, 120.0, (10.0, 0.0, 0.0), None),
            _branch(11, 70.0, (14.0, 3.0, 0.0), 10),
            _branch(12, 45.0, (14.0, -2.0, 0.0), 10),
            _branch(20, 80.0, (-10.0, 0.0, 0.0), None),
            _branch(21, 30.0, (-14.0, 0.0, 0.0), 20),
        ],
        roots=[20, 10],
    )
    payload_b = _make_tree_payload(
        [
            _branch(101, 30.0, (-14.0, 0.0, 0.0), 202),
            _branch(202, 80.0, (-10.0, 0.0, 0.0), None),
            _branch(303, 45.0, (14.0, -2.0, 0.0), 404),
            _branch(404, 120.0, (10.0, 0.0, 0.0), None),
            _branch(505, 70.0, (14.0, 3.0, 0.0), 404),
        ],
        roots=[404, 202],
    )

    result_a = assign_canonical_branch_names(payload_a)
    result_b = assign_canonical_branch_names(payload_b)

    assert _centroid_name_map(result_a, payload_a) == _centroid_name_map(result_b, payload_b)


def test_two_root_tree_receives_system_names_and_hierarchical_children() -> None:
    payload = _make_tree_payload(
        [
            _branch(1, 100.0, (10.0, 0.0, 0.0), None),
            _branch(2, 60.0, (14.0, 2.0, 0.0), 1),
            _branch(3, 40.0, (14.0, -1.0, 0.0), 1),
            _branch(4, 70.0, (-10.0, 0.0, 0.0), None),
        ],
        roots=[4, 1],
    )

    result = assign_canonical_branch_names(payload)
    name_map = {branch.branch_id: branch.canonical_name for branch in result.branches}

    assert name_map[1] == "SYS_A"
    assert name_map[2] == "SYS_A.01"
    assert name_map[3] == "SYS_A.02"
    assert name_map[4] == "SYS_B"
