from pathlib import Path

from vessel_seg.reconstruction_3d2d.coordinate_frames_figure import (
    CoordinateFramesFigureConfig,
    build_coordinate_frames_figure,
)


def test_build_coordinate_frames_figure_creates_png(tmp_path: Path) -> None:
    output = tmp_path / "frames.png"
    config = CoordinateFramesFigureConfig(output_path=output)

    path = build_coordinate_frames_figure(config)

    assert path.exists()
    assert path.suffix == ".png"
    assert "W0 / B / H / C / O" in config.title
