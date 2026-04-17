import numpy as np

from vessel_seg.reconstruction_3d2d.heart_anatomy import build_heart_frame_from_landmarks
from vessel_seg.reconstruction_3d2d.contracts import HeartAnatomicalLandmarks


def test_build_heart_frame_from_landmarks_matches_anatomical_definition() -> None:
    landmarks = HeartAnatomicalLandmarks(
        apex_mm=(0.0, 0.0, 10.0),
        base_center_mm=(0.0, 0.0, 0.0),
        lm_ostium_mm=(1.0, 2.0, 3.0),
        lm_bifurcation_mm=(4.0, 6.0, 5.0),
        rca_ostium_mm=(-2.0, 1.0, 3.0),
    )

    transform = build_heart_frame_from_landmarks(landmarks)
    x_axis = transform[:3, 0]
    y_axis = transform[:3, 1]
    z_axis = transform[:3, 2]
    origin = transform[:3, 3]

    expected_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    coronary_direction = np.asarray(landmarks.lm_bifurcation_mm, dtype=np.float64) - np.asarray(landmarks.lm_ostium_mm, dtype=np.float64)
    expected_x = coronary_direction - float(coronary_direction @ expected_z) * expected_z
    expected_x = expected_x / np.linalg.norm(expected_x)

    assert np.allclose(origin, np.asarray(landmarks.lm_ostium_mm, dtype=np.float64))
    assert np.allclose(z_axis, expected_z)
    assert np.allclose(x_axis, expected_x)
    assert np.isclose(np.linalg.norm(x_axis), 1.0)
    assert np.isclose(np.linalg.norm(y_axis), 1.0)
    assert np.isclose(np.linalg.norm(z_axis), 1.0)
    assert np.isclose(float(x_axis @ y_axis), 0.0, atol=1e-8)
    assert np.isclose(float(x_axis @ z_axis), 0.0, atol=1e-8)
    assert np.isclose(float(y_axis @ z_axis), 0.0, atol=1e-8)
