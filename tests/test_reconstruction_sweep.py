from pathlib import Path

from PIL import Image

from vessel_seg.reconstruction_3d2d.sweep import ProjectionViewSpec, save_projection_sweep_gallery


def test_save_projection_sweep_gallery_supports_custom_image_name(tmp_path: Path) -> None:
    view_specs = (
        ProjectionViewSpec("AP", 0.0, 0.0),
        ProjectionViewSpec("LAO25_CRA10", 25.0, 10.0),
    )
    for index, view in enumerate(view_specs):
        view_dir = tmp_path / view.name
        view_dir.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (120, 80), (20 + index * 20, 40, 60))
        image.save(view_dir / "projection_wall_preview.png")

    gallery_path = save_projection_sweep_gallery(
        output_dir=tmp_path,
        view_specs=view_specs,
        gallery_name="projection_wall_sweep_gallery.png",
        image_name="projection_wall_preview.png",
        tile_size_px=(200, 150),
        cols=2,
    )

    assert gallery_path.exists()
    gallery = Image.open(gallery_path)
    assert gallery.size == (400, 150)
