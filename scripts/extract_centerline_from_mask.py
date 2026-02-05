"""
Extract a simple centerline polydata from a binary mask (NRRD) via 3D skeletonization.

Outputs a VTP file with points at skeleton voxels and line connectivity using 6-neighbor links.
This is a lightweight baseline; for production-quality centerlines consider vmtk or itk filters.

Usage:
  conda activate vessel_seg
  python scripts/extract_centerline_from_mask.py \
      --mask ASOCA2020/Normal/Annotations/Normal_1.nrrd \
      --out ASOCA2020/Normal/Centerlines/Normal_1_extracted.vtp
"""

from pathlib import Path
import argparse
import numpy as np
import SimpleITK as sitk
try:
    from skimage.morphology import skeletonize_3d  # type: ignore
except Exception:  # fallback for newer scikit-image
    from skimage.morphology._skeletonize import skeletonize_3d  # type: ignore
import vtk


def mask_to_skeleton_points(mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    if arr.max() == 0:
        raise ValueError("Mask is empty.")
    skel = skeletonize_3d(arr > 0)
    idx = np.argwhere(skel)  # (N,3) z,y,x
    if idx.size == 0:
        raise ValueError("Skeleton is empty.")
    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    return idx, spacing, origin, direction


def skeleton_points_to_polydata(idx: np.ndarray, spacing: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> vtk.vtkPolyData:
    """
    idx: (N,3) voxel indices (z,y,x)
    connectivity built in 6-neighborhood on voxel grid.
    """
    # map voxel index -> point id
    index_map = {tuple(map(int, p)): i for i, p in enumerate(idx)}

    vtk_points = vtk.vtkPoints()
    for z, y, x in idx:
        phys = origin + direction @ (np.array([x, y, z]) * spacing)
        vtk_points.InsertNextPoint(float(phys[0]), float(phys[1]), float(phys[2]))

    vtk_points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    for key, pid in index_map.items():
        z, y, x = key
        for dz, dy, dx in directions:
            neigh = (z + dz, y + dy, x + dx)
            if neigh in index_map and index_map[neigh] > pid:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, pid)
                line.GetPointIds().SetId(1, index_map[neigh])
                lines.InsertNextCell(line)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetLines(lines)
    return poly


def write_vtp(poly: vtk.vtkPolyData, out_path: Path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(poly)
    writer.Write()


def main():
    parser = argparse.ArgumentParser(description="Extract skeleton centerline from mask.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask (.nrrd)")
    parser.add_argument("--out", type=Path, required=True, help="Output VTP path")
    args = parser.parse_args()

    idx, spacing, origin, direction = mask_to_skeleton_points(args.mask)
    poly = skeleton_points_to_polydata(idx, spacing, origin, direction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_vtp(poly, args.out)
    print(f"Saved extracted centerline to {args.out} with {poly.GetNumberOfPoints()} points and {poly.GetNumberOfLines()} lines")


if __name__ == "__main__":
    main()
