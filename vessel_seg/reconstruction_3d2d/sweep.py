"""Multi-view synthetic projection sweep helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .contracts import BranchLatentState, CArmConfig, DetectorConfig, HeartState, Pose3D
from .synthetic_projection import (
    project_case_centerlines,
    render_projection_preview,
    render_projection_wall_preview,
    save_projection_result_json,
)


@dataclass(frozen=True)
class ProjectionViewSpec:
    """One named synthetic C-arm view."""

    name: str
    lao_rao_deg: float
    cra_cau_deg: float


DEFAULT_VIEW_SPECS: tuple[ProjectionViewSpec, ...] = (
    ProjectionViewSpec("AP", 0.0, 0.0),
    ProjectionViewSpec("LAO25_CRA10", 25.0, 10.0),
    ProjectionViewSpec("RAO25_CRA10", -25.0, 10.0),
    ProjectionViewSpec("LAO40_CAU20", 40.0, -20.0),
    ProjectionViewSpec("RAO40_CAU20", -40.0, -20.0),
    ProjectionViewSpec("LAO15_CRA30", 15.0, 30.0),
)


def run_projection_sweep(
    *,
    tree_json_path: Path,
    centerline_vtp_path: Path,
    output_dir: Path,
    view_specs: tuple[ProjectionViewSpec, ...] = DEFAULT_VIEW_SPECS,
    heart_translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    heart_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ecg_phase: float = 0.0,
    sid_mm: float = 1200.0,
    sod_mm: float = 750.0,
    detector_width_px: int = 1024,
    detector_height_px: int = 1024,
    pixel_spacing_mm: float = 0.30,
) -> tuple[Path, ...]:
    """Generate one synthetic projection per view spec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for view in view_specs:
        view_dir = output_dir / view.name
        c_arm = CArmConfig(
            lao_rao_deg=float(view.lao_rao_deg),
            cra_cau_deg=float(view.cra_cau_deg),
            sid_mm=float(sid_mm),
            sod_mm=float(sod_mm),
            detector=DetectorConfig(
                width_px=int(detector_width_px),
                height_px=int(detector_height_px),
                pixel_spacing_mm=float(pixel_spacing_mm),
            ),
        )
        heart_state = HeartState(
            world_pose=Pose3D(
                translation_mm=tuple(float(value) for value in heart_translation_mm),
                rpy_deg=tuple(float(value) for value in heart_rpy_deg),
            ),
            ecg_phase=float(ecg_phase),
            left_state=BranchLatentState(side_group="LCA"),
            right_state=BranchLatentState(side_group="RCA"),
        )
        result = project_case_centerlines(tree_json_path, centerline_vtp_path, c_arm, heart_state=heart_state)
        result_json = view_dir / "synthetic_projection.json"
        preview_png = view_dir / "projection_preview.png"
        wall_preview_png = view_dir / "projection_wall_preview.png"
        save_projection_result_json(result, result_json)
        render_projection_preview(result, preview_png)
        render_projection_wall_preview(result, wall_preview_png)
        image_paths.append(preview_png)
    return tuple(image_paths)


def save_projection_sweep_gallery(
    *,
    output_dir: Path,
    view_specs: tuple[ProjectionViewSpec, ...] = DEFAULT_VIEW_SPECS,
    gallery_name: str = "projection_sweep_gallery.png",
    image_name: str = "projection_preview.png",
    tile_size_px: tuple[int, int] = (900, 900),
    cols: int = 2,
) -> Path:
    """Build a fixed-grid gallery from per-view preview images."""
    width_px, height_px = int(tile_size_px[0]), int(tile_size_px[1])
    images: list[Image.Image] = []
    for view in view_specs:
        image_path = output_dir / view.name / image_name
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.pad(image, (width_px, height_px), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width_px, 54), fill=(18, 18, 18))
        draw.text((20, 14), view.name.replace("_", " / "), fill=(240, 240, 240))
        images.append(image)

    cols = max(1, int(cols))
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * width_px, rows * height_px), (10, 10, 10))
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        canvas.paste(image, (col * width_px, row * height_px))

    gallery_path = output_dir / gallery_name
    canvas.save(gallery_path)
    return gallery_path
