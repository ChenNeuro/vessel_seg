from pathlib import Path

import numpy as np

from vessel_seg.reconstruction_3d2d.carm_geometry import (
    observer_points_to_detector_pixels,
    project_points_world_to_observer,
)
from vessel_seg.reconstruction_3d2d.contracts import (
    CArmConfig,
    DetectorConfig,
    HeartState,
    SyntheticProjectionResult,
)


def test_projection_maps_isocenter_to_detector_principal_point() -> None:
    config = CArmConfig(
        alpha_lao_rao_deg=0.0,
        beta_cra_cau_deg=0.0,
        source_to_detector_mm=1200.0,
        source_to_isocenter_mm=750.0,
        detector=DetectorConfig(width_px=1000, height_px=800, pixel_spacing_mm=0.5),
    )
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)

    observer_points, observer_valid = project_points_world_to_observer(points, config)
    pixels, pixel_valid = observer_points_to_detector_pixels(observer_points, config)

    assert bool(observer_valid[0])
    assert bool(pixel_valid[0])
    assert np.allclose(observer_points[0], np.asarray([0.0, 0.0, config.source_to_isocenter_mm], dtype=np.float64))
    assert np.allclose(pixels[0], config.detector.resolved_principal_point_px())


def test_projection_changes_when_carm_angle_changes() -> None:
    config_a = CArmConfig(detector=DetectorConfig(pixel_spacing_mm=0.4))
    config_b = CArmConfig(
        alpha_lao_rao_deg=35.0,
        beta_cra_cau_deg=15.0,
        detector=DetectorConfig(pixel_spacing_mm=0.4),
    )
    points = np.asarray([[40.0, 25.0, 10.0]], dtype=np.float64)

    observer_a, valid_observer_a = project_points_world_to_observer(points, config_a)
    observer_b, valid_observer_b = project_points_world_to_observer(points, config_b)
    pixels_a, valid_pixels_a = observer_points_to_detector_pixels(observer_a, config_a)
    pixels_b, valid_pixels_b = observer_points_to_detector_pixels(observer_b, config_b)

    assert bool(valid_observer_a[0])
    assert bool(valid_observer_b[0])
    assert bool(valid_pixels_a[0])
    assert bool(valid_pixels_b[0])
    assert not np.allclose(pixels_a[0], pixels_b[0])


def test_synthetic_projection_result_serialization_uses_new_schema() -> None:
    result = SyntheticProjectionResult(
        case_id="DemoCase",
        tree_json_path=Path("tree.json"),
        centerline_vtp_path=Path("centerline.vtp"),
        c_arm=CArmConfig(),
        heart_state=HeartState(),
        projected_branches=(),
        metadata={"observer_width_px": 1024, "heart_model_origin_mm": [0.0, 0.0, 0.0]},
    )

    payload = result.to_dict()

    assert "source_to_detector_mm" in payload["c_arm"]
    assert "source_to_isocenter_mm" in payload["c_arm"]
    assert "isocenter_to_detector_mm" in payload["c_arm"]
    assert "lao_rao_deg" not in payload["c_arm"]
    assert "sid_mm" not in payload["c_arm"]
    assert "bed_from_heart" in payload["heart_state"]
    assert "bed_config" in payload["heart_state"]
    assert "coronary_frame_rpy_deg" in payload["heart_state"]
    assert "world_pose" not in payload["heart_state"]
