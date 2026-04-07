import numpy as np

from vessel_seg.reconstruction_3d2d.carm_geometry import project_points_world_to_detector
from vessel_seg.reconstruction_3d2d.contracts import CArmConfig, DetectorConfig


def test_projection_maps_isocenter_to_detector_principal_point() -> None:
    config = CArmConfig(
        lao_rao_deg=0.0,
        cra_cau_deg=0.0,
        sid_mm=1200.0,
        sod_mm=750.0,
        detector=DetectorConfig(width_px=1000, height_px=800, pixel_spacing_mm=0.5),
    )
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)

    pixels, valid = project_points_world_to_detector(points, config)

    assert bool(valid[0])
    assert np.allclose(pixels[0], config.detector.resolved_principal_point_px())


def test_projection_changes_when_carm_angle_changes() -> None:
    config_a = CArmConfig(detector=DetectorConfig(pixel_spacing_mm=0.4))
    config_b = CArmConfig(lao_rao_deg=35.0, cra_cau_deg=15.0, detector=DetectorConfig(pixel_spacing_mm=0.4))
    points = np.asarray([[40.0, 25.0, 10.0]], dtype=np.float64)

    pixels_a, valid_a = project_points_world_to_detector(points, config_a)
    pixels_b, valid_b = project_points_world_to_detector(points, config_b)

    assert bool(valid_a[0])
    assert bool(valid_b[0])
    assert not np.allclose(pixels_a[0], pixels_b[0])
