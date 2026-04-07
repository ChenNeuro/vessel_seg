"""Synthetic C-arm projection utilities built on existing repaired centerlines."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .carm_geometry import pose_matrix_mm, project_points_world_to_detector, rotation_matrix_from_rpy_deg, transform_points_mm
from .contracts import CArmConfig, HeartState, ProjectedBranch2D, SyntheticProjectionResult


def read_branch_polylines(vtp_path: Path) -> list[np.ndarray]:
    """Read branch polylines from a repaired VTP file."""
    try:
        import vtk
    except Exception as exc:  # pragma: no cover
        raise ImportError("vtk is required to load centerline VTP files.") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    points = poly.GetPoints()
    lines = poly.GetLines()
    polylines: list[np.ndarray] = []
    lines.InitTraversal()
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        coords = np.asarray([points.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=np.float64)
        if coords.shape[0] >= 2:
            polylines.append(coords)
    return polylines


def _branch_metadata(tree_payload: dict[str, object]) -> dict[int, dict[str, object]]:
    metadata: dict[int, dict[str, object]] = {}
    for branch in list(tree_payload.get("branches", [])):
        branch_id = int(branch["branch_id"])
        naming = dict(branch.get("naming") or {})
        semantic = dict(branch.get("semantic") or {})
        side_group = str(semantic.get("system_name") or naming.get("system_guess") or naming.get("system_name") or "UNKNOWN")
        metadata[branch_id] = {
            "canonical_name": str(naming.get("canonical_name") or f"branch_{branch_id}"),
            "semantic_name": semantic.get("semantic_name"),
            "side_group": side_group,
            "depth": int(naming.get("depth") or 0),
        }
    return metadata


def _heart_center_mm(polylines: list[np.ndarray]) -> np.ndarray:
    """Estimate a stable heart-local origin from all branch polylines."""
    if not polylines:
        return np.zeros((3,), dtype=np.float64)
    coords = np.concatenate(polylines, axis=0)
    return 0.5 * (coords.min(axis=0) + coords.max(axis=0))


def _load_branch_radius_profiles(tree_json_path: Path) -> dict[int, np.ndarray]:
    """Load per-branch mean radius profiles from stage4 if available."""
    wall_npz = tree_json_path.parents[1] / "04_wall_features" / "branch_dataset.npz"
    if not wall_npz.exists():
        return {}
    payload = np.load(wall_npz, allow_pickle=True)
    if "branch_ids" not in payload or "radii" not in payload:
        return {}
    branch_ids = np.asarray(payload["branch_ids"], dtype=np.int64)
    radii = np.asarray(payload["radii"], dtype=np.float64)
    if radii.ndim != 3:
        return {}
    profiles: dict[int, np.ndarray] = {}
    for index, branch_id in enumerate(branch_ids.tolist()):
        profiles[int(branch_id)] = radii[index].mean(axis=1).astype(np.float64, copy=False)
    return profiles


def _resample_radius_profile(profile_mm: np.ndarray, count: int) -> np.ndarray:
    """Resample stage4 longitudinal mean radius profile to polyline point count."""
    if count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if profile_mm.size == 0:
        return np.zeros((count,), dtype=np.float64)
    if count == 1:
        return np.asarray([float(profile_mm.mean())], dtype=np.float64)
    source_x = np.linspace(0.0, 1.0, int(profile_mm.shape[0]), dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, int(count), dtype=np.float64)
    return np.interp(target_x, source_x, profile_mm).astype(np.float64, copy=False)


def _heart_world_matrix(heart_state: HeartState) -> np.ndarray:
    world_from_heart = pose_matrix_mm(heart_state.world_pose)
    cav_rotation = np.eye(4, dtype=np.float64)
    cav_rotation[:3, :3] = rotation_matrix_from_rpy_deg(heart_state.cav_rpy_deg)
    return world_from_heart @ cav_rotation


def project_case_centerlines(
    tree_json_path: Path,
    centerline_vtp_path: Path,
    c_arm: CArmConfig,
    heart_state: HeartState | None = None,
) -> SyntheticProjectionResult:
    """Project repaired centerlines into a synthetic detector image."""
    heart_state = heart_state or HeartState()
    tree_payload = json.loads(tree_json_path.read_text(encoding="utf-8"))
    polylines = read_branch_polylines(centerline_vtp_path)
    branch_metadata = _branch_metadata(tree_payload)
    radius_profiles = _load_branch_radius_profiles(tree_json_path)
    heart_matrix = _heart_world_matrix(heart_state)
    heart_center = _heart_center_mm(polylines)

    projected_branches: list[ProjectedBranch2D] = []
    visible_count = 0
    for branch_id, polyline in enumerate(polylines):
        heart_polyline = polyline - heart_center.reshape(1, 3)
        world_polyline = transform_points_mm(heart_polyline, heart_matrix)
        points_px, valid = project_points_world_to_detector(world_polyline, c_arm)
        rel = world_polyline
        from .carm_geometry import carm_rotation_world
        rotation_world = carm_rotation_world(c_arm)
        z_axis_world = rotation_world @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        source_world = rotation_world @ np.asarray([0.0, 0.0, -float(c_arm.sod_mm)], dtype=np.float64)
        z_cam = (rel - source_world.reshape(1, 3)) @ z_axis_world
        radius_profile = _resample_radius_profile(radius_profiles.get(branch_id, np.zeros((0,), dtype=np.float64)), polyline.shape[0])
        projected_radii_px = np.zeros_like(z_cam, dtype=np.float64)
        valid_depth = z_cam > 1e-6
        projected_radii_px[valid_depth] = (
            radius_profile[valid_depth] * float(c_arm.sid_mm) / z_cam[valid_depth] / max(float(c_arm.detector.pixel_spacing_mm), 1e-6)
        )
        visible_points = tuple(
            (float(point[0]), float(point[1]))
            for point, is_valid in zip(points_px, valid)
            if bool(is_valid)
        )
        visible_radii = tuple(
            float(radius_px)
            for radius_px, is_valid in zip(projected_radii_px, valid)
            if bool(is_valid)
        )
        visible = len(visible_points) >= 2
        visible_count += int(visible)
        meta = branch_metadata.get(
            branch_id,
            {
                "canonical_name": f"branch_{branch_id}",
                "semantic_name": None,
                "side_group": "UNKNOWN",
                "depth": 0,
            },
        )
        projected_branches.append(
            ProjectedBranch2D(
                branch_id=branch_id,
                canonical_name=str(meta["canonical_name"]),
                semantic_name=None if meta["semantic_name"] is None else str(meta["semantic_name"]),
                side_group=str(meta["side_group"]),
                depth=int(meta["depth"]),
                visible=visible,
                points_px=visible_points,
                radii_px=visible_radii,
                mean_radius_mm=None if radius_profile.size == 0 else float(radius_profile.mean()),
            )
        )

    case_id = str(tree_payload.get("case_id") or tree_json_path.parent.parent.parent.name)
    return SyntheticProjectionResult(
        case_id=case_id,
        tree_json_path=tree_json_path,
        centerline_vtp_path=centerline_vtp_path,
        c_arm=c_arm,
        heart_state=heart_state,
        projected_branches=tuple(projected_branches),
        metadata={
            "branch_count": len(projected_branches),
            "visible_branch_count": visible_count,
            "detector_width_px": int(c_arm.detector.width_px),
            "detector_height_px": int(c_arm.detector.height_px),
            "heart_center_mm": [float(value) for value in heart_center],
        },
    )


def save_projection_result_json(result: SyntheticProjectionResult, output_path: Path) -> None:
    """Serialize the synthetic projection result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def render_projection_preview(result: SyntheticProjectionResult, output_path: Path) -> None:
    """Render a lightweight 2D preview image for the synthetic projection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width_px = int(result.metadata.get("detector_width_px", result.c_arm.detector.width_px))
    height_px = int(result.metadata.get("detector_height_px", result.c_arm.detector.height_px))
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    color_map = {"LCA": "#ff4d4f", "RCA": "#4fc3f7"}
    occupied_x: list[float] = []
    occupied_y: list[float] = []
    for branch in result.projected_branches:
        if not branch.visible or len(branch.points_px) < 2:
            continue
        coords = np.asarray(branch.points_px, dtype=np.float64)
        occupied_x.extend(float(value) for value in coords[:, 0])
        occupied_y.extend(float(value) for value in coords[:, 1])
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=color_map.get(branch.side_group, "#d9d9d9"),
            linewidth=1.8 if branch.depth == 0 else 1.1,
            alpha=0.95,
        )
    if occupied_x and occupied_y:
        min_x = min(occupied_x)
        max_x = max(occupied_x)
        min_y = min(occupied_y)
        max_y = max(occupied_y)
        pad_x = max(20.0, 0.08 * max(max_x - min_x, 1.0))
        pad_y = max(20.0, 0.08 * max(max_y - min_y, 1.0))
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(max_y + pad_y, min_y - pad_y)
    else:
        ax.set_xlim(0.0, float(width_px))
        ax.set_ylim(float(height_px), 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "\n".join(
            [
                f"Synthetic Projection: {result.case_id}",
                f"LAO/RAO={result.c_arm.lao_rao_deg:.1f} deg, CRA/CAU={result.c_arm.cra_cau_deg:.1f} deg",
                f"visible_branches={result.metadata.get('visible_branch_count', '?')}/{result.metadata.get('branch_count', '?')}",
                "Preview: auto-cropped to occupied projection bbox",
            ]
        ),
        color="white",
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    plt.close(fig)


def _auto_crop_axes(ax, occupied_x: list[float], occupied_y: list[float], width_px: int, height_px: int) -> None:
    """Apply detector fallback or occupied-bbox cropping."""
    if occupied_x and occupied_y:
        min_x = min(occupied_x)
        max_x = max(occupied_x)
        min_y = min(occupied_y)
        max_y = max(occupied_y)
        pad_x = max(20.0, 0.08 * max(max_x - min_x, 1.0))
        pad_y = max(20.0, 0.08 * max(max_y - min_y, 1.0))
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(max_y + pad_y, min_y - pad_y)
    else:
        ax.set_xlim(0.0, float(width_px))
        ax.set_ylim(float(height_px), 0.0)


def render_projection_wall_preview(result: SyntheticProjectionResult, output_path: Path) -> None:
    """Render a filled vessel silhouette style projection preview."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width_px = int(result.metadata.get("detector_width_px", result.c_arm.detector.width_px))
    height_px = int(result.metadata.get("detector_height_px", result.c_arm.detector.height_px))
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    fill_map = {"LCA": "#ff7875", "RCA": "#69c0ff"}
    edge_map = {"LCA": "#ffd6d6", "RCA": "#d5f1ff"}
    occupied_x: list[float] = []
    occupied_y: list[float] = []
    for branch in result.projected_branches:
        if not branch.visible or len(branch.points_px) < 2:
            continue
        coords = np.asarray(branch.points_px, dtype=np.float64)
        radii = np.asarray(branch.radii_px, dtype=np.float64)
        if radii.size != coords.shape[0]:
            fallback_radius = max(1.0, float(branch.mean_radius_mm or 1.0) / max(float(result.c_arm.detector.pixel_spacing_mm), 1e-6))
            radii = np.full((coords.shape[0],), fallback_radius, dtype=np.float64)
        radii = np.clip(radii, 1.0, 80.0)
        occupied_x.extend(float(value) for value in coords[:, 0])
        occupied_y.extend(float(value) for value in coords[:, 1])
        for index in range(coords.shape[0] - 1):
            p0 = coords[index]
            p1 = coords[index + 1]
            tangent = p1 - p0
            norm = float(np.linalg.norm(tangent))
            if norm <= 1e-6:
                continue
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64) / norm
            r0 = float(radii[index])
            r1 = float(radii[index + 1])
            quad = np.asarray(
                [
                    p0 + normal * r0,
                    p1 + normal * r1,
                    p1 - normal * r1,
                    p0 - normal * r0,
                ],
                dtype=np.float64,
            )
            polygon = Polygon(
                quad,
                closed=True,
                facecolor=fill_map.get(branch.side_group, "#bbbbbb"),
                edgecolor=fill_map.get(branch.side_group, "#bbbbbb"),
                linewidth=0.0,
                alpha=0.58,
            )
            ax.add_patch(polygon)
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=edge_map.get(branch.side_group, "#f5f5f5"),
            linewidth=0.8 if branch.depth > 0 else 1.2,
            alpha=0.95,
        )
    _auto_crop_axes(ax, occupied_x, occupied_y, width_px, height_px)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "\n".join(
            [
                f"Synthetic Wall Projection: {result.case_id}",
                f"LAO/RAO={result.c_arm.lao_rao_deg:.1f} deg, CRA/CAU={result.c_arm.cra_cau_deg:.1f} deg",
                f"visible_branches={result.metadata.get('visible_branch_count', '?')}/{result.metadata.get('branch_count', '?')}",
                "Wall preview: silhouette from stage4 mean radius profiles",
            ]
        ),
        color="white",
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    plt.close(fig)
