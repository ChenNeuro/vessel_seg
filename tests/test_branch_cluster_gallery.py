import numpy as np

from vessel_seg.pipeline.branch_cluster_gallery import _align_polylines_to_anatomical_frame, _ordered_case_ids_for_gallery


def test_ordered_case_ids_for_gallery_uses_fixed_cohort_and_number_grid() -> None:
    rows = [
        {"case_id": "Diseased_10", "cohort": "Diseased"},
        {"case_id": "Normal_2", "cohort": "Normal"},
        {"case_id": "Diseased_2", "cohort": "Diseased"},
        {"case_id": "Normal_1", "cohort": "Normal"},
    ]

    ordered = _ordered_case_ids_for_gallery(rows, fixed_case_grid=True)

    assert ordered == ["Normal_1", "Normal_2", "Diseased_2", "Diseased_10"]


def test_ordered_case_ids_for_gallery_can_fallback_to_dynamic_order() -> None:
    rows = [
        {"case_id": "Normal_1", "cohort": "Normal"},
        {"case_id": "Normal_2", "cohort": "Normal"},
        {"case_id": "Diseased_1", "cohort": "Diseased"},
    ]
    cluster_cases = {
        "Diseased_1": [{"overall_similarity_to_prototype": "0.91"}],
        "Normal_2": [{"overall_similarity_to_prototype": "0.88"}],
    }

    ordered = _ordered_case_ids_for_gallery(rows, fixed_case_grid=False, cluster_cases=cluster_cases)

    assert ordered == ["Diseased_1", "Normal_2", "Normal_1"]


def test_align_polylines_to_anatomical_frame_projects_points_into_lr_long_norm_axes() -> None:
    tree_payload = {
        "roots": [1, 2],
        "branches": [
            {
                "branch_id": 1,
                "length_mm": 10.0,
                "start": [1.0, 0.0, 0.0],
                "end": [2.0, 0.0, 0.0],
                "centroid": [1.5, 0.0, 0.0],
                "attachment": {"parent": None, "lambda_pos": None, "theta_deg": None, "phi_deg": None},
            },
            {
                "branch_id": 2,
                "length_mm": 10.0,
                "start": [-1.0, 0.0, 0.0],
                "end": [-2.0, 0.0, 0.0],
                "centroid": [-1.5, 0.0, 0.0],
                "attachment": {"parent": None, "lambda_pos": None, "theta_deg": None, "phi_deg": None},
            },
        ],
    }
    polylines = [
        np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=np.float32),
    ]

    aligned = _align_polylines_to_anatomical_frame(polylines, tree_payload)

    assert np.allclose(aligned[0][:, 1], 0.0)
    assert np.allclose(aligned[0][:, 2], 0.0)
