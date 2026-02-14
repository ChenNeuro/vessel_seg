#!/usr/bin/env python3
"""Compare centerline topology between two VTP files.

We build a graph from polyline connectivity, collapse degree!=2 points
into nodes (endpoints + bifurcations), then compare node/edge sets
after spatial matching within a distance threshold.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    import vtk  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("vtk is required to read VTP centerlines.") from exc


@dataclass
class CenterlineGraph:
    node_points: np.ndarray  # Nx3 cluster centers
    degrees: np.ndarray  # degree per node
    endpoints: np.ndarray
    bifurcations: np.ndarray
    edges: Set[Tuple[int, int]]  # node index pairs


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def extract_polyline_endpoints(poly: vtk.vtkPolyData) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    pts = poly.GetPoints()
    if pts is None or pts.GetNumberOfPoints() == 0:
        return np.zeros((0, 3), dtype=float), []
    endpoints = []
    line_end_indices: List[Tuple[int, int]] = []
    lines = poly.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    while lines.GetNextCell(ids):
        m = ids.GetNumberOfIds()
        if m < 2:
            continue
        start_id = ids.GetId(0)
        end_id = ids.GetId(m - 1)
        if start_id == end_id:
            continue
        endpoints.append(pts.GetPoint(start_id))
        endpoints.append(pts.GetPoint(end_id))
        line_end_indices.append((len(endpoints) - 2, len(endpoints) - 1))
    if not endpoints:
        return np.zeros((0, 3), dtype=float), []
    return np.array(endpoints, dtype=float), line_end_indices


def cluster_points(points: np.ndarray, thr: float) -> Tuple[np.ndarray, List[int]]:
    if points.size == 0:
        return np.zeros((0, 3), dtype=float), []
    tree = cKDTree(points)
    parent = list(range(points.shape[0]))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(points.shape[0]):
        neighbors = tree.query_ball_point(points[i], thr)
        for j in neighbors:
            if i != j:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(points.shape[0]):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    node_points = []
    mapping = [0] * points.shape[0]
    for node_id, idxs in enumerate(clusters.values()):
        coords = points[idxs]
        node_points.append(coords.mean(axis=0))
        for idx in idxs:
            mapping[idx] = node_id
    return np.array(node_points, dtype=float), mapping


def build_graph(path: Path, merge_thr: float) -> CenterlineGraph:
    poly = read_polydata(path)
    endpoint_pts, line_end_indices = extract_polyline_endpoints(poly)
    node_points, mapping = cluster_points(endpoint_pts, merge_thr)
    edges: Set[Tuple[int, int]] = set()
    for a, b in line_end_indices:
        u = mapping[a]
        v = mapping[b]
        if u != v:
            edges.add((min(u, v), max(u, v)))

    degrees = np.zeros((node_points.shape[0],), dtype=int)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    endpoints = np.where(degrees == 1)[0]
    bifurcations = np.where(degrees >= 3)[0]
    return CenterlineGraph(
        node_points=node_points,
        degrees=degrees,
        endpoints=endpoints,
        bifurcations=bifurcations,
        edges=edges,
    )


def match_nodes(pred_pts: np.ndarray, gt_pts: np.ndarray, thr: float) -> Dict[int, int]:
    if pred_pts.size == 0 or gt_pts.size == 0:
        return {}
    tree = cKDTree(gt_pts)
    dists, idxs = tree.query(pred_pts, k=1)
    pairs = [
        (p_idx, g_idx, dist)
        for p_idx, (g_idx, dist) in enumerate(zip(idxs, dists))
        if dist <= thr
    ]
    pairs.sort(key=lambda x: x[2])
    matched_pred = set()
    matched_gt = set()
    mapping: Dict[int, int] = {}
    for p_idx, g_idx, _ in pairs:
        if p_idx in matched_pred or g_idx in matched_gt:
            continue
        matched_pred.add(p_idx)
        matched_gt.add(g_idx)
        mapping[p_idx] = g_idx
    return mapping


def compare_graphs(pred: CenterlineGraph, gt: CenterlineGraph, thr: float) -> dict:
    pred_node_pts = pred.node_points
    gt_node_pts = gt.node_points
    mapping = match_nodes(pred_node_pts, gt_node_pts, thr)

    matched_nodes = len(mapping)
    pred_nodes = pred.node_points.shape[0]
    gt_nodes = gt.node_points.shape[0]
    node_precision = matched_nodes / pred_nodes if pred_nodes else None
    node_recall = matched_nodes / gt_nodes if gt_nodes else None

    gt_edge_set = gt.edges
    matched_edges = 0
    valid_pred_edges = 0
    for u, v in pred.edges:
        if u in mapping and v in mapping:
            valid_pred_edges += 1
            gu, gv = mapping[u], mapping[v]
            edge = (min(gu, gv), max(gu, gv))
            if edge in gt_edge_set:
                matched_edges += 1

    pred_edges = len(pred.edges)
    gt_edges = len(gt.edges)
    edge_precision = matched_edges / pred_edges if pred_edges else None
    edge_recall = matched_edges / gt_edges if gt_edges else None
    edge_match_rate = matched_edges / valid_pred_edges if valid_pred_edges else None

    return {
        "pred_nodes": int(pred_nodes),
        "gt_nodes": int(gt_nodes),
        "pred_endpoints": int(pred.endpoints.size),
        "gt_endpoints": int(gt.endpoints.size),
        "pred_bifurcations": int(pred.bifurcations.size),
        "gt_bifurcations": int(gt.bifurcations.size),
        "pred_edges": int(pred_edges),
        "gt_edges": int(gt_edges),
        "matched_nodes": int(matched_nodes),
        "node_precision": node_precision,
        "node_recall": node_recall,
        "matched_edges": int(matched_edges),
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_match_rate": edge_match_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare centerline topology for a case or batch.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thr", type=float, default=1.0, help="Node matching threshold (mm).")
    parser.add_argument("--merge-thr", type=float, default=1.0, help="Node merge threshold (mm).")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=20)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.start, args.end + 1):
        pred = args.pred_dir / f"Normal_{i}_vmtk.vtp"
        gt = args.gt_dir / f"Normal_{i}.vtp"
        pred_graph = build_graph(pred, args.merge_thr)
        gt_graph = build_graph(gt, args.merge_thr)
        metrics = compare_graphs(pred_graph, gt_graph, args.thr)
        metrics["case"] = f"Normal_{i}"
        rows.append(metrics)

    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
