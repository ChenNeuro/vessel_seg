#!/usr/bin/env python3
"""Reconstruct polyline centerlines from a points-only VTP.

This script is used as a fallback when VMTK output contains points but no line cells.
It builds local kNN graphs per connected component, extracts an MST forest, then
splits paths at branching nodes to generate VTP polyline cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree

try:
    import vtk  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("vtk is required to read/write VTP.") from exc


def read_vtp_geometry(path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()

    pts = poly.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float), []
    points = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)

    polylines: List[np.ndarray] = []
    lines = poly.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    while lines.GetNextCell(ids):
        if ids.GetNumberOfIds() < 2:
            continue
        arr = np.array([pts.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
        polylines.append(arr)
    return points, polylines


def write_vtp_polylines(polylines: Sequence[np.ndarray], path: Path) -> Dict[str, int]:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    total_lines = 0
    total_points = 0
    for poly in polylines:
        if poly.shape[0] < 2:
            continue
        start = points.GetNumberOfPoints()
        for p in poly:
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(poly.shape[0])
        for i in range(poly.shape[0]):
            line.GetPointIds().SetId(i, start + i)
        lines.InsertNextCell(line)
        total_lines += 1
        total_points += int(poly.shape[0])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()
    return {"lines": total_lines, "points": total_points}


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _polyline_length_mm(poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def _densify_polyline(poly: np.ndarray, step_mm: float) -> np.ndarray:
    if step_mm <= 0 or poly.shape[0] < 2:
        return poly
    out = [poly[0]]
    for i in range(poly.shape[0] - 1):
        p0 = poly[i]
        p1 = poly[i + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= step_mm:
            out.append(p1)
            continue
        n = int(np.ceil(seg_len / step_mm))
        for k in range(1, n + 1):
            t = float(k) / float(n)
            out.append(p0 * (1.0 - t) + p1 * t)
    return np.asarray(out, dtype=np.float32)


def _union_find_components(n: int, pairs: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
    parent = np.arange(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(int(a), int(b))

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)
    return [np.asarray(v, dtype=int) for v in comps.values()]


def _extract_paths_from_tree(sub_points: np.ndarray, adj: List[Set[int]]) -> List[np.ndarray]:
    m = len(adj)
    if m == 0:
        return []
    special = [i for i in range(m) if len(adj[i]) != 2]
    if not special:
        special = [0]

    visited: Set[Tuple[int, int]] = set()
    paths: List[np.ndarray] = []

    for start in special:
        for nxt in list(adj[start]):
            e = _edge_key(start, nxt)
            if e in visited:
                continue
            visited.add(e)
            path_ids = [start]
            prev = start
            cur = nxt
            while True:
                path_ids.append(cur)
                nxts = [x for x in adj[cur] if x != prev]
                if len(adj[cur]) != 2 or not nxts:
                    break
                nx = nxts[0]
                e2 = _edge_key(cur, nx)
                if e2 in visited:
                    break
                visited.add(e2)
                prev, cur = cur, nx
            if len(path_ids) >= 2:
                paths.append(sub_points[np.asarray(path_ids, dtype=int)])

    # Residual cycle edges
    for a in range(m):
        for b in list(adj[a]):
            e = _edge_key(a, b)
            if e in visited:
                continue
            visited.add(e)
            path_ids = [a, b]
            prev = a
            cur = b
            while True:
                nxts = [x for x in adj[cur] if x != prev]
                if not nxts:
                    break
                nx = nxts[0]
                e2 = _edge_key(cur, nx)
                if e2 in visited:
                    break
                visited.add(e2)
                path_ids.append(nx)
                prev, cur = cur, nx
            if len(path_ids) >= 2:
                paths.append(sub_points[np.asarray(path_ids, dtype=int)])
    return paths


def reconstruct_polylines_from_points(
    points: np.ndarray,
    *,
    component_radius_mm: float,
    component_radius_factor: float,
    min_component_points: int,
    knn_k: int,
    edge_max_dist_mm: float,
    edge_max_dist_factor: float,
    min_path_length_mm: float,
    densify_step_mm: float,
    keep_top_lines: int,
) -> Tuple[List[np.ndarray], dict]:
    if points.shape[0] < 2:
        return [], {"reason": "not_enough_points"}

    kdt = cKDTree(points)
    nn_dist, _ = kdt.query(points, k=2)
    nn_median = float(np.median(nn_dist[:, 1]))

    comp_radius = float(component_radius_mm) if component_radius_mm > 0 else float(nn_median * component_radius_factor)
    comp_radius = float(np.clip(comp_radius, 0.45, 1.8))
    edge_radius = float(edge_max_dist_mm) if edge_max_dist_mm > 0 else float(nn_median * edge_max_dist_factor)
    edge_radius = float(np.clip(edge_radius, 0.6, 2.5))

    pairs = list(kdt.query_pairs(r=comp_radius, output_type="set"))
    comps = _union_find_components(points.shape[0], pairs)
    comps = [c for c in comps if c.size >= int(min_component_points)]

    all_polys: List[np.ndarray] = []
    comp_summaries = []
    for comp_ids in comps:
        sub_points = points[comp_ids]
        m = sub_points.shape[0]
        if m < 2:
            continue

        sub_tree = cKDTree(sub_points)
        qk = max(2, min(int(knn_k) + 1, m))
        d, nn = sub_tree.query(sub_points, k=qk)

        rows = []
        cols = []
        vals = []
        for i in range(m):
            for jpos in range(1, qk):
                j = int(nn[i, jpos])
                if j == i:
                    continue
                w = float(d[i, jpos])
                if w <= 0.0 or w > edge_radius:
                    continue
                rows.append(i)
                cols.append(j)
                vals.append(w)
                rows.append(j)
                cols.append(i)
                vals.append(w)

        if not rows:
            comp_summaries.append({"points": int(m), "edges": 0, "paths": 0})
            continue

        graph = coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
        mst = minimum_spanning_tree(graph).tocoo()
        adj: List[Set[int]] = [set() for _ in range(m)]
        for i, j, _ in zip(mst.row, mst.col, mst.data):
            ii = int(i)
            jj = int(j)
            adj[ii].add(jj)
            adj[jj].add(ii)

        raw_paths = _extract_paths_from_tree(sub_points, adj)
        kept = 0
        for poly in raw_paths:
            length = _polyline_length_mm(poly)
            if length < float(min_path_length_mm):
                continue
            dense = _densify_polyline(poly, float(densify_step_mm))
            if dense.shape[0] < 2:
                continue
            all_polys.append(dense)
            kept += 1
        comp_summaries.append({"points": int(m), "edges": int(len(mst.data)), "paths": int(kept)})

    all_polys.sort(key=_polyline_length_mm, reverse=True)
    if keep_top_lines and keep_top_lines > 0:
        all_polys = all_polys[: int(keep_top_lines)]

    report = {
        "input_points": int(points.shape[0]),
        "nn_median_mm": nn_median,
        "component_radius_mm": comp_radius,
        "edge_radius_mm": edge_radius,
        "components_kept": int(len(comps)),
        "component_summaries": comp_summaries,
        "output_lines": int(len(all_polys)),
        "output_points": int(sum(poly.shape[0] for poly in all_polys)),
    }
    return all_polys, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct VTP polyline cells from points-only centerline VTP.")
    parser.add_argument("--in-vtp", type=Path, required=True)
    parser.add_argument("--out-vtp", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--component-radius-mm", type=float, default=0.0)
    parser.add_argument("--component-radius-factor", type=float, default=2.0)
    parser.add_argument("--min-component-points", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--edge-max-dist-mm", type=float, default=0.0)
    parser.add_argument("--edge-max-dist-factor", type=float, default=2.8)
    parser.add_argument("--min-path-length-mm", type=float, default=2.5)
    parser.add_argument("--densify-step-mm", type=float, default=0.5)
    parser.add_argument("--keep-top-lines", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    points, lines = read_vtp_geometry(args.in_vtp)
    payload: Dict[str, object] = {
        "in_vtp": str(args.in_vtp),
        "out_vtp": str(args.out_vtp),
        "input_points": int(points.shape[0]),
        "input_lines": int(len(lines)),
    }

    if len(lines) > 0:
        dense_lines = [_densify_polyline(poly, float(args.densify_step_mm)) for poly in lines]
        stats = write_vtp_polylines(dense_lines, args.out_vtp)
        payload["mode"] = "existing_lines"
        payload["output_lines"] = int(stats["lines"])
        payload["output_points"] = int(stats["points"])
    else:
        polys, report = reconstruct_polylines_from_points(
            points,
            component_radius_mm=float(args.component_radius_mm),
            component_radius_factor=float(args.component_radius_factor),
            min_component_points=int(args.min_component_points),
            knn_k=int(args.knn_k),
            edge_max_dist_mm=float(args.edge_max_dist_mm),
            edge_max_dist_factor=float(args.edge_max_dist_factor),
            min_path_length_mm=float(args.min_path_length_mm),
            densify_step_mm=float(args.densify_step_mm),
            keep_top_lines=int(args.keep_top_lines),
        )
        stats = write_vtp_polylines(polys, args.out_vtp)
        payload["mode"] = "reconstructed_from_points"
        payload["reconstruction"] = report
        payload["output_lines"] = int(stats["lines"])
        payload["output_points"] = int(stats["points"])

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
