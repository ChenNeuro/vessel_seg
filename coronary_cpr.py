"""Coronary CPR generation from CCTA volume and centerline.

Implements straightened CPR and curved MIP CPR using a stable
rotation-minimizing frame (RMF) along the centerline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _ensure_centerline(centerline_mm: np.ndarray) -> np.ndarray:
    centerline_mm = np.asarray(centerline_mm, dtype=np.float64)
    if centerline_mm.ndim != 2 or centerline_mm.shape[1] != 3:
        raise ValueError("centerline_mm must have shape (N, 3)")
    if centerline_mm.shape[0] < 2:
        raise ValueError("centerline_mm must have at least 2 points")
    return centerline_mm


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, eps, None)


def resample_centerline_by_arclength(
    centerline_mm: np.ndarray, step_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample centerline to equal arc-length spacing.

    Parameters
    ----------
    centerline_mm : np.ndarray
        Input centerline points in physical space, shape (N, 3).
    step_mm : float
        Target spacing along arc length in mm.

    Returns
    -------
    resampled : np.ndarray
        Resampled centerline points, shape (M, 3).
    s_new : np.ndarray
        Arc-length positions of resampled points, shape (M,).
    """
    centerline_mm = _ensure_centerline(centerline_mm)
    if step_mm <= 0:
        raise ValueError("step_mm must be > 0")

    diffs = np.diff(centerline_mm, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = s[-1]
    if total_length <= 1e-6:
        raise ValueError("centerline length too small")

    n_samples = max(2, int(np.floor(total_length / step_mm)) + 1)
    s_new = np.linspace(0.0, total_length, n_samples)
    resampled = np.zeros((n_samples, 3), dtype=np.float64)
    for dim in range(3):
        resampled[:, dim] = np.interp(s_new, s, centerline_mm[:, dim])
    return resampled, s_new


def _smooth_vectors(vectors: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return _normalize(vectors)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(vectors, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.zeros_like(vectors, dtype=np.float64)
    for dim in range(3):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return _normalize(smoothed)


def compute_rmf_frames(
    centerline_mm: np.ndarray, smooth_window: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rotation-minimizing frames (t, n, b) along centerline.

    Parameters
    ----------
    centerline_mm : np.ndarray
        Resampled centerline points, shape (N, 3).
    smooth_window : int
        Window size for tangent smoothing. Use 1 to disable.

    Returns
    -------
    t : np.ndarray
        Tangent vectors, shape (N, 3).
    n : np.ndarray
        Normal vectors, shape (N, 3).
    b : np.ndarray
        Binormal vectors, shape (N, 3).
    """
    centerline_mm = _ensure_centerline(centerline_mm)
    if centerline_mm.shape[0] < 2:
        raise ValueError("centerline must have at least 2 points")

    diffs = np.diff(centerline_mm, axis=0)
    tangents = np.zeros_like(centerline_mm, dtype=np.float64)
    tangents[0] = diffs[0]
    tangents[-1] = diffs[-1]
    tangents[1:-1] = diffs[:-1] + diffs[1:]
    tangents = _normalize(tangents)
    tangents = _smooth_vectors(tangents, smooth_window)

    n = np.zeros_like(tangents)
    b = np.zeros_like(tangents)

    t0 = tangents[0]
    abs_t = np.abs(t0)
    if abs_t[0] <= abs_t[1] and abs_t[0] <= abs_t[2]:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif abs_t[1] <= abs_t[2]:
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n0 = v - np.dot(v, t0) * t0
    n0 = _normalize(n0)
    b0 = np.cross(t0, n0)
    n[0] = n0
    b[0] = b0

    for i in range(tangents.shape[0] - 1):
        ti = tangents[i]
        tj = tangents[i + 1]
        axis = np.cross(ti, tj)
        axis_norm = np.linalg.norm(axis)
        dot = float(np.clip(np.dot(ti, tj), -1.0, 1.0))
        if axis_norm < 1e-6:
            if dot < 0.0:
                n[i + 1] = -n[i]
            else:
                n[i + 1] = n[i]
            b[i + 1] = np.cross(tj, n[i + 1])
            b[i + 1] = _normalize(b[i + 1])
            continue

        axis_unit = axis / axis_norm
        angle = np.arctan2(axis_norm, dot)
        kx, ky, kz = axis_unit
        K = np.array(
            [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
            dtype=np.float64,
        )
        R = (
            np.eye(3, dtype=np.float64)
            + np.sin(angle) * K
            + (1.0 - np.cos(angle)) * (K @ K)
        )
        n_next = R @ n[i]
        n_next = n_next - np.dot(n_next, tj) * tj
        n_next = _normalize(n_next)
        b_next = np.cross(tj, n_next)
        b_next = _normalize(b_next)
        n[i + 1] = n_next
        b[i + 1] = b_next

    return tangents, n, b


def sample_volume_trilinear(
    volume: np.ndarray,
    coords_xyz_mm: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    fill_value: float = 0.0,
) -> np.ndarray:
    """Trilinear sampling of a 3D volume at given physical coordinates.

    Parameters
    ----------
    volume : np.ndarray
        Volume array with shape (Z, Y, X).
    coords_xyz_mm : np.ndarray
        Physical coordinates (x, y, z) in mm, shape (..., 3).
    spacing : tuple
        Voxel spacing (sz, sy, sx) in mm.
    origin : tuple
        Physical origin (oz, oy, ox) in mm.
    fill_value : float
        Value for out-of-bounds samples.

    Returns
    -------
    samples : np.ndarray
        Sampled values with shape coords_xyz_mm.shape[:-1].
    """
    if volume.ndim != 3:
        raise ValueError("volume must have shape (Z, Y, X)")
    coords = np.asarray(coords_xyz_mm, dtype=np.float64)
    if coords.shape[-1] != 3:
        raise ValueError("coords_xyz_mm must have shape (..., 3)")

    sz, sy, sx = spacing
    oz, oy, ox = origin
    ix = (coords[..., 0] - ox) / sx
    iy = (coords[..., 1] - oy) / sy
    iz = (coords[..., 2] - oz) / sz

    x0 = np.floor(ix).astype(np.int64)
    y0 = np.floor(iy).astype(np.int64)
    z0 = np.floor(iz).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    X = volume.shape[2]
    Y = volume.shape[1]
    Z = volume.shape[0]
    valid = (
        (ix >= 0)
        & (ix <= X - 1)
        & (iy >= 0)
        & (iy <= Y - 1)
        & (iz >= 0)
        & (iz <= Z - 1)
    )

    x0c = np.clip(x0, 0, X - 1)
    x1c = np.clip(x1, 0, X - 1)
    y0c = np.clip(y0, 0, Y - 1)
    y1c = np.clip(y1, 0, Y - 1)
    z0c = np.clip(z0, 0, Z - 1)
    z1c = np.clip(z1, 0, Z - 1)

    dx = ix - x0
    dy = iy - y0
    dz = iz - z0

    coords_shape = coords.shape[:-1]
    flat = np.ravel_multi_index(
        (z0c.ravel(), y0c.ravel(), x0c.ravel()), volume.shape
    )
    v000 = volume.ravel()[flat].reshape(coords_shape)
    flat = np.ravel_multi_index(
        (z0c.ravel(), y0c.ravel(), x1c.ravel()), volume.shape
    )
    v001 = volume.ravel()[flat].reshape(coords_shape)
    flat = np.ravel_multi_index(
        (z0c.ravel(), y1c.ravel(), x0c.ravel()), volume.shape
    )
    v010 = volume.ravel()[flat].reshape(coords_shape)
    flat = np.ravel_multi_index(
        (z0c.ravel(), y1c.ravel(), x1c.ravel()), volume.shape
    )
    v011 = volume.ravel()[flat].reshape(coords_shape)
    flat = np.ravel_multi_index(
        (z1c.ravel(), y0c.ravel(), x0c.ravel()), volume.shape
    )
    v100 = volume.ravel()[flat].reshape(coords_shape)
    flat = np.ravel_multi_index(
        (z1c.ravel(), y0c.ravel(), x1c.ravel()), volume.shape
    )
    v101 = volume.ravel()[flat].reshape(coords_shape)
    flat = np.ravel_multi_index(
        (z1c.ravel(), y1c.ravel(), x0c.ravel()), volume.shape
    )
    v110 = volume.ravel()[flat].reshape(coords_shape)
    flat = np.ravel_multi_index(
        (z1c.ravel(), y1c.ravel(), x1c.ravel()), volume.shape
    )
    v111 = volume.ravel()[flat].reshape(coords_shape)

    c00 = v000 * (1 - dx) + v001 * dx
    c01 = v010 * (1 - dx) + v011 * dx
    c10 = v100 * (1 - dx) + v101 * dx
    c11 = v110 * (1 - dx) + v111 * dx
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    c = c0 * (1 - dz) + c1 * dz

    return np.where(valid, c, fill_value)


def _sample_cpr_strip(
    volume: np.ndarray,
    centerline_mm: np.ndarray,
    n: np.ndarray,
    b: np.ndarray,
    width_mm: float,
    height_mm: float,
    out_width_px: int,
    out_height_px: int,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    slab_mm: float = 0.0,
    reduce_mode: str = "center",
) -> np.ndarray:
    if width_mm <= 0:
        raise ValueError("width_mm must be > 0")
    if height_mm <= 0:
        raise ValueError("height_mm must be > 0")
    if out_width_px <= 1:
        raise ValueError("out_width_px must be > 1")
    if out_height_px <= 1:
        raise ValueError("out_height_px must be > 1")
    if reduce_mode not in {"center", "mean", "max"}:
        raise ValueError("reduce_mode must be 'center', 'mean', or 'max'")

    u_vals = np.linspace(-width_mm / 2.0, width_mm / 2.0, out_width_px)
    v_vals = np.linspace(-height_mm / 2.0, height_mm / 2.0, out_height_px)
    if slab_mm > 0:
        slab_half = slab_mm / 2.0
        slab_mask = (v_vals >= -slab_half) & (v_vals <= slab_half)
        if not np.any(slab_mask):
            slab_mask = None
    else:
        slab_mask = None

    cpr = np.zeros((centerline_mm.shape[0], out_width_px), dtype=np.float32)
    for i in range(centerline_mm.shape[0]):
        c = centerline_mm[i]
        ni = n[i]
        bi = b[i]
        grid = (
            c[None, None, :]
            + u_vals[None, :, None] * ni[None, None, :]
            + v_vals[:, None, None] * bi[None, None, :]
        )
        samples = sample_volume_trilinear(
            volume, grid.reshape(-1, 3), spacing, origin
        ).reshape(out_height_px, out_width_px)
        if slab_mask is not None:
            samples = samples[slab_mask]
        if reduce_mode == "max":
            cpr[i] = np.max(samples, axis=0)
        elif reduce_mode == "mean":
            cpr[i] = np.mean(samples, axis=0)
        else:
            center_idx = samples.shape[0] // 2
            cpr[i] = samples[center_idx]
    return cpr


def generate_straightened_cpr(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    centerline_mm: np.ndarray,
    width_mm: float,
    height_mm: float,
    step_mm: float,
    out_px: Tuple[int, int],
    slab_mm: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate straightened CPR image.

    Output shape is (N_samples, W), where rows follow centerline arc length
    and columns follow the local normal direction.
    """
    if volume.ndim != 3:
        raise ValueError("volume must have shape (Z, Y, X)")
    if len(out_px) != 2:
        raise ValueError("out_px must be a tuple (H, W)")
    if height_mm <= 0:
        raise ValueError("height_mm must be > 0")

    resampled, s = resample_centerline_by_arclength(centerline_mm, step_mm)
    t, n, b = compute_rmf_frames(resampled, smooth_window=5)
    cpr = _sample_cpr_strip(
        volume,
        resampled,
        n,
        b,
        width_mm,
        height_mm,
        out_px[1],
        out_px[0],
        spacing,
        origin,
        slab_mm=slab_mm,
        reduce_mode="center",
    )
    debug = {"s": s, "centerline_mm": resampled, "t": t, "n": n, "b": b}
    return cpr, debug


def generate_curved_mip_cpr(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    centerline_mm: Optional[np.ndarray],
    width_mm: float,
    height_mm: float,
    step_mm: float,
    out_px: Tuple[int, int],
    slab_mm: float,
    debug: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate curved MIP CPR image using slab MIP along binormal."""
    if debug is None:
        if centerline_mm is None:
            raise ValueError("centerline_mm is required when debug is None")
        resampled, s = resample_centerline_by_arclength(centerline_mm, step_mm)
        t, n, b = compute_rmf_frames(resampled, smooth_window=5)
        debug = {"s": s, "centerline_mm": resampled, "t": t, "n": n, "b": b}
    else:
        resampled = debug["centerline_mm"]
        n = debug["n"]
        b = debug["b"]

    if len(out_px) != 2:
        raise ValueError("out_px must be a tuple (H, W)")
    if height_mm <= 0:
        raise ValueError("height_mm must be > 0")

    cpr = _sample_cpr_strip(
        volume,
        resampled,
        n,
        b,
        width_mm,
        height_mm,
        out_px[1],
        out_px[0],
        spacing,
        origin,
        slab_mm=slab_mm,
        reduce_mode="max",
    )
    return cpr, debug


def window_and_normalize(
    image: np.ndarray,
    window: Tuple[float, float] = (100.0, 700.0),
    gamma: Optional[float] = None,
) -> np.ndarray:
    """Window and normalize an image to [0, 1]."""
    lo, hi = window
    if hi <= lo:
        raise ValueError("window high must be > low")
    img = np.clip(image, lo, hi)
    img = (img - lo) / (hi - lo)
    if gamma is not None:
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        img = np.power(img, gamma)
    return img.astype(np.float32)


def _centerline_inbounds_ratio(
    idx_xyz: np.ndarray, volume_shape: Tuple[int, int, int]
) -> float:
    if idx_xyz.ndim != 2 or idx_xyz.shape[1] != 3:
        raise ValueError("idx_xyz must have shape (N, 3)")
    x_in = (idx_xyz[:, 0] >= 0) & (idx_xyz[:, 0] < volume_shape[2])
    y_in = (idx_xyz[:, 1] >= 0) & (idx_xyz[:, 1] < volume_shape[1])
    z_in = (idx_xyz[:, 2] >= 0) & (idx_xyz[:, 2] < volume_shape[0])
    return float(np.mean(x_in & y_in & z_in))


def convert_centerline_to_physical(
    centerline_xyz: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    volume_shape: Tuple[int, int, int],
    mode: str = "auto",
) -> Tuple[np.ndarray, str, float]:
    """Convert centerline coordinates to physical mm space.

    Parameters
    ----------
    centerline_xyz : np.ndarray
        Centerline points in (x, y, z) order.
    spacing : tuple
        (sz, sy, sx) in mm.
    origin : tuple
        (oz, oy, ox) in mm.
    volume_shape : tuple
        (Z, Y, X) voxel shape of the volume.
    mode : str
        'physical', 'index', 'index_zflip', or 'auto'.

    Returns
    -------
    centerline_mm : np.ndarray
        Centerline in physical mm coordinates.
    chosen_mode : str
        Mode selected (if auto).
    in_bounds_ratio : float
        Fraction of points that map inside the volume bounds.
    """
    centerline_xyz = _ensure_centerline(centerline_xyz)
    sz, sy, sx = spacing
    oz, oy, ox = origin

    def idx_to_mm(idx: np.ndarray) -> np.ndarray:
        mm = np.empty_like(idx, dtype=np.float64)
        mm[:, 0] = ox + idx[:, 0] * sx
        mm[:, 1] = oy + idx[:, 1] * sy
        mm[:, 2] = oz + idx[:, 2] * sz
        return mm

    if mode not in {"physical", "index", "index_zflip", "auto"}:
        raise ValueError("mode must be 'physical', 'index', 'index_zflip', or 'auto'")

    if mode == "physical":
        idx = np.empty_like(centerline_xyz, dtype=np.float64)
        idx[:, 0] = (centerline_xyz[:, 0] - ox) / sx
        idx[:, 1] = (centerline_xyz[:, 1] - oy) / sy
        idx[:, 2] = (centerline_xyz[:, 2] - oz) / sz
        return centerline_xyz, "physical", _centerline_inbounds_ratio(idx, volume_shape)

    if mode == "index":
        return idx_to_mm(centerline_xyz), "index", _centerline_inbounds_ratio(centerline_xyz, volume_shape)

    if mode == "index_zflip":
        idx = centerline_xyz * np.array([1.0, 1.0, -1.0], dtype=np.float64)
        return idx_to_mm(idx), "index_zflip", _centerline_inbounds_ratio(idx, volume_shape)

    # auto mode
    candidates = {
        "physical": centerline_xyz,
        "index": idx_to_mm(centerline_xyz),
        "index_zflip": idx_to_mm(centerline_xyz * np.array([1.0, 1.0, -1.0], dtype=np.float64)),
    }
    ratios = {}
    for name in ("physical", "index", "index_zflip"):
        if name == "physical":
            idx = np.empty_like(centerline_xyz, dtype=np.float64)
            idx[:, 0] = (centerline_xyz[:, 0] - ox) / sx
            idx[:, 1] = (centerline_xyz[:, 1] - oy) / sy
            idx[:, 2] = (centerline_xyz[:, 2] - oz) / sz
        elif name == "index":
            idx = centerline_xyz
        else:
            idx = centerline_xyz * np.array([1.0, 1.0, -1.0], dtype=np.float64)
        ratios[name] = _centerline_inbounds_ratio(idx, volume_shape)
    chosen = max(ratios, key=ratios.get)
    return candidates[chosen], chosen, ratios[chosen]


def overlay_mask_on_grayscale(
    gray: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.9, 0.2),
    alpha: float = 0.6,
) -> np.ndarray:
    """Overlay a binary mask on a grayscale image (both in [0, 1])."""
    if gray.ndim != 2:
        raise ValueError("gray must be 2D")
    if mask.shape != gray.shape:
        raise ValueError("mask shape must match gray shape")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    rgb = np.stack([gray, gray, gray], axis=-1)
    mask_bool = mask > 0.5
    if np.any(mask_bool):
        rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * np.array(color, dtype=np.float32)
    return rgb.astype(np.float32)


def load_nifti_sitk(
    path: Path,
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """Load NIfTI using SimpleITK and return (volume, spacing, origin).

    Volume is (Z, Y, X), spacing/origin are (sz, sy, sx)/(oz, oy, ox).
    """
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise ImportError("SimpleITK is required to load NIfTI files") from exc

    img = sitk.ReadImage(str(path))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    sx, sy, sz = img.GetSpacing()
    ox, oy, oz = img.GetOrigin()
    return vol, (sz, sy, sx), (oz, oy, ox)


def read_centerlines_vtp(vtp_path: Path) -> List[np.ndarray]:
    """Read polylines from a VTP file as a list of centerlines."""
    try:
        import vtk
    except Exception as exc:
        raise ImportError("vtk is required to read VTP centerlines") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    lines = poly.GetLines()
    lines.InitTraversal()
    branches: List[np.ndarray] = []
    while True:
        ids = vtk.vtkIdList()
        if not lines.GetNextCell(ids):
            break
        if ids.GetNumberOfIds() < 2:
            continue
        coords = np.array([pts.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
        branches.append(coords.astype(np.float64))
    if not branches:
        raise ValueError(f"No centerlines found in {vtp_path}")
    return branches


def select_centerline_branch(
    branches: List[np.ndarray],
    branch_id: Optional[int] = None,
    strategy: str = "longest",
) -> np.ndarray:
    """Select one branch by index or longest arc length."""
    if branch_id is not None:
        if branch_id < 0 or branch_id >= len(branches):
            raise ValueError(f"branch_id {branch_id} out of range (0..{len(branches) - 1})")
        return branches[branch_id]
    if strategy not in {"longest", "first"}:
        raise ValueError("strategy must be 'longest' or 'first'")
    if strategy == "first":
        return branches[0]
    lengths = []
    for coords in branches:
        diffs = np.diff(coords, axis=0)
        lengths.append(float(np.sum(np.linalg.norm(diffs, axis=1))))
    return branches[int(np.argmax(lengths))]


def _draw_sphere(
    volume: np.ndarray, center_xyz: Tuple[float, float, float], radius_vox: int, value: float
) -> None:
    cz, cy, cx = center_xyz
    Z, Y, X = volume.shape
    for dz in range(-radius_vox, radius_vox + 1):
        for dy in range(-radius_vox, radius_vox + 1):
            for dx in range(-radius_vox, radius_vox + 1):
                if dx * dx + dy * dy + dz * dz > radius_vox * radius_vox:
                    continue
                z = cz + dz
                y = cy + dy
                x = cx + dx
                if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                    volume[z, y, x] = value


def main_demo() -> None:
    """Minimal demo using a synthetic helix centerline."""
    spacing = (0.5, 0.5, 0.5)
    origin = (0.0, 0.0, 0.0)
    vol = np.zeros((128, 128, 128), dtype=np.float32)

    t = np.linspace(0.0, 1.0, 200)
    x = 32.0 + 12.0 * np.cos(2.0 * np.pi * 2.0 * t)
    y = 32.0 + 12.0 * np.sin(2.0 * np.pi * 2.0 * t)
    z = 8.0 + 48.0 * t
    centerline_mm = np.stack([x, y, z], axis=1)

    radius_mm = 1.5
    radius_vox = int(np.ceil(radius_mm / min(spacing)))
    for pt in centerline_mm[::2]:
        ix = int(round((pt[0] - origin[2]) / spacing[2]))
        iy = int(round((pt[1] - origin[1]) / spacing[1]))
        iz = int(round((pt[2] - origin[0]) / spacing[0]))
        _draw_sphere(vol, (iz, iy, ix), radius_vox, value=600.0)

    straight, debug = generate_straightened_cpr(
        vol,
        spacing,
        origin,
        centerline_mm,
        width_mm=20.0,
        height_mm=20.0,
        step_mm=0.5,
        out_px=(64, 128),
        slab_mm=0.0,
    )
    curved, _ = generate_curved_mip_cpr(
        vol,
        spacing,
        origin,
        centerline_mm=None,
        width_mm=20.0,
        height_mm=20.0,
        step_mm=0.5,
        out_px=(64, 128),
        slab_mm=10.0,
        debug=debug,
    )

    straight_img = window_and_normalize(straight, window=(100.0, 700.0))
    curved_img = window_and_normalize(curved, window=(100.0, 700.0))

    try:
        import matplotlib.pyplot as plt

        plt.imsave("straightened.png", straight_img, cmap="gray")
        plt.imsave("curved_mip.png", curved_img, cmap="gray")
        print("Saved straightened.png and curved_mip.png")
    except Exception as exc:
        print("Failed to save demo images:", exc)


def main_cli() -> None:
    """CLI for ASOCA Normal_1 or synthetic demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate straightened and curved MIP CPR.")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo instead of ASOCA data.")
    parser.add_argument(
        "--volume",
        type=Path,
        default=Path("ASOCA2020/Normal/CTCA_nii/Normal_1.nii.gz"),
        help="Path to CCTA volume (NIfTI).",
    )
    parser.add_argument(
        "--centerline",
        type=Path,
        default=Path("ASOCA2020/Normal/Centerlines/Normal_1.vtp"),
        help="Path to centerline VTP.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional segmentation mask (NIfTI) for overlay.",
    )
    parser.add_argument(
        "--centerline_space",
        choices=("auto", "physical", "index", "index_zflip"),
        default="auto",
        help="Interpretation of centerline coordinates.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/Normal_1/cpr"),
        help="Output directory for CPR images and arrays.",
    )
    parser.add_argument("--branch_id", type=int, default=None, help="Branch index in VTP lines.")
    parser.add_argument(
        "--branch_strategy",
        choices=("longest", "first"),
        default="longest",
        help="Strategy when branch_id is not provided.",
    )
    parser.add_argument("--width_mm", type=float, default=20.0)
    parser.add_argument("--height_mm", type=float, default=20.0)
    parser.add_argument("--step_mm", type=float, default=0.5)
    parser.add_argument("--slab_mm", type=float, default=10.0)
    parser.add_argument("--out_h", type=int, default=64)
    parser.add_argument("--out_w", type=int, default=128)
    parser.add_argument("--window_lo", type=float, default=100.0)
    parser.add_argument("--window_hi", type=float, default=700.0)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--overlay_alpha", type=float, default=0.6)
    args = parser.parse_args()

    if args.demo:
        main_demo()
        return

    volume, spacing, origin = load_nifti_sitk(args.volume)
    branches = read_centerlines_vtp(args.centerline)
    centerline_raw = select_centerline_branch(branches, args.branch_id, args.branch_strategy)
    centerline, mode_used, in_bounds = convert_centerline_to_physical(
        centerline_raw, spacing, origin, volume.shape, mode=args.centerline_space
    )
    print(f"Centerline space: {mode_used} | in-bounds ratio: {in_bounds:.3f}")

    straight, debug = generate_straightened_cpr(
        volume,
        spacing,
        origin,
        centerline,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        step_mm=args.step_mm,
        out_px=(args.out_h, args.out_w),
        slab_mm=0.0,
    )
    curved, _ = generate_curved_mip_cpr(
        volume,
        spacing,
        origin,
        centerline_mm=None,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        step_mm=args.step_mm,
        out_px=(args.out_h, args.out_w),
        slab_mm=args.slab_mm,
        debug=debug,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "straightened_cpr.npy", straight)
    np.save(args.out_dir / "curved_mip_cpr.npy", curved)
    np.savez(
        args.out_dir / "cpr_debug.npz",
        s=debug["s"],
        centerline_mm=debug["centerline_mm"],
        t=debug["t"],
        n=debug["n"],
        b=debug["b"],
    )

    straight_img = window_and_normalize(
        straight, window=(args.window_lo, args.window_hi), gamma=args.gamma
    )
    curved_img = window_and_normalize(
        curved, window=(args.window_lo, args.window_hi), gamma=args.gamma
    )
    try:
        import matplotlib.pyplot as plt

        plt.imsave(args.out_dir / "straightened.png", straight_img, cmap="gray")
        plt.imsave(args.out_dir / "curved_mip.png", curved_img, cmap="gray")
    except Exception as exc:
        print("Failed to save images:", exc)

    if args.mask is not None:
        mask_vol, mask_spacing, mask_origin = load_nifti_sitk(args.mask)
        if mask_spacing != spacing or mask_origin != origin:
            print("Warning: mask spacing/origin differ from volume; alignment may be off.")
        mask_cpr, _ = generate_curved_mip_cpr(
            mask_vol.astype(np.float32),
            spacing,
            origin,
            centerline_mm=None,
            width_mm=args.width_mm,
            height_mm=args.height_mm,
            step_mm=args.step_mm,
            out_px=(args.out_h, args.out_w),
            slab_mm=args.slab_mm,
            debug=debug,
        )
        mask_cpr_bin = mask_cpr > 0.5
        overlay = overlay_mask_on_grayscale(
            curved_img, mask_cpr_bin.astype(np.float32), alpha=args.overlay_alpha
        )
        try:
            import matplotlib.pyplot as plt

            plt.imsave(args.out_dir / "curved_mip_overlay.png", overlay)
        except Exception as exc:
            print("Failed to save overlay image:", exc)

    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main_cli()
