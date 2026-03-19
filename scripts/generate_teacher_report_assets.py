#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import vtk
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def read_polyline_vtp(path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()

    points = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())], dtype=float)
    lines: List[np.ndarray] = []
    cells = poly.GetLines()
    cells.InitTraversal()
    ids = vtk.vtkIdList()
    while cells.GetNextCell(ids):
        if ids.GetNumberOfIds() < 2:
            continue
        lines.append(np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=int))
    return points, lines


def connected_components(n_points: int, lines: Sequence[np.ndarray]) -> int:
    if n_points == 0:
        return 0
    adj: List[set] = [set() for _ in range(n_points)]
    used = np.zeros(n_points, dtype=bool)
    for line in lines:
        if line.size < 2:
            continue
        used[line] = True
        for a, b in zip(line[:-1], line[1:]):
            a_i = int(a)
            b_i = int(b)
            if a_i == b_i:
                continue
            adj[a_i].add(b_i)
            adj[b_i].add(a_i)
    used_idx = np.where(used)[0]
    if used_idx.size == 0:
        return 0

    visited = np.zeros(n_points, dtype=bool)
    comps = 0
    for start in used_idx:
        if visited[start]:
            continue
        comps += 1
        stack = [int(start)]
        visited[start] = True
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
    return comps


def pca_projection(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros(3, dtype=float), np.eye(3, dtype=float)[:, :2]
    center = points.mean(axis=0)
    x = points - center
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    basis = vh[:2].T
    return center, basis


def project(points: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=float)
    return (points - center) @ basis


def draw_polylines(
    ax: plt.Axes,
    points_2d: np.ndarray,
    lines: Sequence[np.ndarray],
    color: str,
    label: str | None = None,
    alpha: float = 0.95,
    linewidth: float = 1.0,
) -> None:
    label_drawn = False
    for line in lines:
        if line.size < 2:
            continue
        xy = points_2d[line]
        this_label = label if (label and not label_drawn) else None
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=alpha, linewidth=linewidth, label=this_label)
        label_drawn = True


def make_pipeline_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")

    box_w = 2.2
    box_h = 1.0
    y = 1.6
    x0 = 0.5
    gap = 0.55

    labels = [
        "Step1\nSegmentation\n(CT -> mask)",
        "Step2\nCenterline\nExtraction",
        "Step3\nCenterline\nRepair",
        "Step4\nFeature\nExtraction",
        "Step5\nReconstruction\n& Compare",
    ]
    colors = ["#5b8ff9", "#61dDAA", "#f6bd16", "#e8684a", "#6dc8ec"]

    xs = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = x0 + i * (box_w + gap)
        xs.append(x)
        rect = Rectangle((x, y), box_w, box_h, facecolor=color, edgecolor="#222222", linewidth=1.2, alpha=0.95)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center", fontsize=11, color="#111111")

    for i in range(len(xs) - 1):
        x1 = xs[i] + box_w
        x2 = xs[i + 1]
        arrow = FancyArrowPatch((x1 + 0.04, y + box_h / 2), (x2 - 0.04, y + box_h / 2), arrowstyle="->", mutation_scale=16, linewidth=1.6, color="#444444")
        ax.add_patch(arrow)

    ax.text(0.5, 3.0, "Quantitative Coronary Pipeline (Each Step Measurable)", fontsize=16, weight="bold", color="#111111")
    ax.text(
        0.5,
        0.75,
        "Metrics: Step1 Dice/ASD/HD95 | Step2/3 centerline distance+coverage | "
        "Step4 branch/descriptor | Step5 reconstruction distance+coverage",
        fontsize=10.5,
        color="#222222",
    )
    ax.set_xlim(0, x0 + 5 * (box_w + gap))
    ax.set_ylim(0, 3.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def pick_center(mask: np.ndarray) -> Tuple[int, int, int]:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        z = mask.shape[0] // 2
        y = mask.shape[1] // 2
        x = mask.shape[2] // 2
        return z, y, x
    z, y, x = np.round(idx.mean(axis=0)).astype(int)
    z = int(np.clip(z, 0, mask.shape[0] - 1))
    y = int(np.clip(y, 0, mask.shape[1] - 1))
    x = int(np.clip(x, 0, mask.shape[2] - 1))
    return z, y, x


def orient_2d(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr)


def make_step1_overlay(
    case: str,
    ct_path: Path,
    gt_mask_path: Path,
    pred_mask_path: Path,
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    ct = load_nifti(ct_path).astype(float)
    gt = load_nifti(gt_mask_path) > 0.5
    pred = load_nifti(pred_mask_path) > 0.5

    z, y, x = pick_center(gt | pred)
    views = [
        ("Axial", ct[z, :, :], gt[z, :, :], pred[z, :, :]),
        ("Coronal", ct[:, y, :], gt[:, y, :], pred[:, y, :]),
        ("Sagittal", ct[:, :, x], gt[:, :, x], pred[:, :, x]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (name, img, gt2d, pred2d) in zip(axes, views):
        v = orient_2d(img)
        ax.imshow(v, cmap="gray", vmin=np.percentile(ct, 2), vmax=np.percentile(ct, 99.5))
        gt_o = orient_2d(gt2d.astype(float))
        pred_o = orient_2d(pred2d.astype(float))
        if np.any(gt_o > 0):
            ax.contour(gt_o, levels=[0.5], colors=["#00cc44"], linewidths=1.4)
        if np.any(pred_o > 0):
            ax.contour(pred_o, levels=[0.5], colors=["#ff3333"], linewidths=1.4)
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    dice = float(metrics.get("dice", np.nan))
    hd95 = float(metrics.get("hd95_mm", np.nan))
    asd = float(metrics.get("asd_mm", np.nan))
    fig.suptitle(f"{case} Step1 Segmentation (GT vs Pred) | Dice={dice:.4f} ASD={asd:.3f}mm HD95={hd95:.3f}mm", fontsize=13, weight="bold")
    legend_items = [
        Line2D([0], [0], color="#00cc44", lw=2, label="GT mask"),
        Line2D([0], [0], color="#ff3333", lw=2, label="Pred mask (TotalSegmentator)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_centerline_overlay(
    case: str,
    pred_vtp: Path,
    gt_vtp: Path,
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    pred_pts, pred_lines = read_polyline_vtp(pred_vtp)
    gt_pts, gt_lines = read_polyline_vtp(gt_vtp)
    center, basis = pca_projection(np.vstack([pred_pts, gt_pts]))
    pred_2d = project(pred_pts, center, basis)
    gt_2d = project(gt_pts, center, basis)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    draw_polylines(ax, gt_2d, gt_lines, color="#00aa44", label="GT", linewidth=1.2, alpha=0.85)
    draw_polylines(ax, pred_2d, pred_lines, color="#ff3333", label="Pred repaired", linewidth=1.2, alpha=0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    p95 = float(metrics.get("pred2gt_p95", np.nan))
    cov = float(metrics.get("coverage_pred@1mm", np.nan))
    ax.set_title(f"{case} Step3 Centerline (Pred vs GT)\np95={p95:.3f}mm, coverage@1mm={cov:.3f}", fontsize=12, weight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def load_branch_lengths(features_dir: Path) -> List[float]:
    summary = load_json(features_dir / "summary.json")
    return sorted([float(b.get("length_mm", 0.0)) for b in summary.get("branches", [])], reverse=True)


def make_step4_branch_plot(
    case: str,
    pred_features_dir: Path,
    gt_features_dir: Path,
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    pred_lengths = load_branch_lengths(pred_features_dir)
    gt_lengths = load_branch_lengths(gt_features_dir)
    n = max(len(pred_lengths), len(gt_lengths))
    x = np.arange(n)
    pred = np.array(pred_lengths + [np.nan] * (n - len(pred_lengths)))
    gt = np.array(gt_lengths + [np.nan] * (n - len(gt_lengths)))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x, pred, "-o", color="#ff3333", label="Pred")
    ax.plot(x, gt, "-o", color="#00aa44", label="GT")
    ax.set_xlabel("Branch rank")
    ax.set_ylabel("Length (mm)")
    diff = int(metrics.get("branch_count_abs_diff", 0))
    cos = float(metrics.get("descriptor_cosine", np.nan))
    ax.set_title(f"{case} Step4 Branch Length\nbranch diff={diff}, cosine={cos:.4f}", fontsize=12, weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_cross_case_summary(
    case_rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    fig, (ax_table, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax_table.axis("off")
    headers = ["Case", "S1 Dice", "S1 HD95", "S2 p95", "S3 p95", "S4 diff", "S4 cos", "S5 p95"]
    table_data = []
    for row in case_rows:
        table_data.append(
            [
                row["case"],
                f"{row['step1_dice']:.4f}",
                f"{row['step1_hd95']:.3f}",
                f"{row['step2_p95']:.3f}",
                f"{row['step3_p95']:.3f}",
                f"{int(row['step4_diff'])}",
                f"{row['step4_cos']:.4f}",
                f"{row['step5_p95']:.3f}",
            ]
        )
    table = ax_table.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.16, 1.55)
    ax_table.set_title("Cross-case Quantitative Summary", fontsize=13, weight="bold", pad=10)

    cases = [r["case"] for r in case_rows]
    x = np.arange(len(cases))
    w = 0.34
    s2 = np.array([r["step2_p95"] for r in case_rows], dtype=float)
    s3 = np.array([r["step3_p95"] for r in case_rows], dtype=float)
    ax_bar.bar(x - w / 2, s2, width=w, label="Step2 p95", color="#5b8ff9")
    ax_bar.bar(x + w / 2, s3, width=w, label="Step3 p95", color="#f6bd16")
    for i in range(len(x)):
        ax_bar.text(x[i] - w / 2, s2[i] + 0.03, f"{s2[i]:.3f}", ha="center", va="bottom", fontsize=9)
        ax_bar.text(x[i] + w / 2, s3[i] + 0.03, f"{s3[i]:.3f}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cases)
    ax_bar.set_ylabel("Distance (mm)")
    ax_bar.set_title("Step2 vs Step3 p95", fontsize=12, weight="bold")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_connectivity_plot(
    baseline_vtp: Path,
    repaired_vtp: Path,
    out_path: Path,
) -> None:
    b_pts, b_lines = read_polyline_vtp(baseline_vtp)
    r_pts, r_lines = read_polyline_vtp(repaired_vtp)
    center, basis = pca_projection(np.vstack([b_pts, r_pts]))
    b2 = project(b_pts, center, basis)
    r2 = project(r_pts, center, basis)

    b_comp = connected_components(len(b_pts), b_lines)
    r_comp = connected_components(len(r_pts), r_lines)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    draw_polylines(axes[0], b2, b_lines, color="#ff8c00", linewidth=1.0)
    axes[0].set_title("Before repair (baseline centerline)", fontsize=11, weight="bold")
    axes[0].axis("off")
    axes[0].set_aspect("equal")
    axes[0].text(
        0.02,
        0.02,
        f"points={len(b_pts)}  lines={len(b_lines)}  components={b_comp}",
        transform=axes[0].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    draw_polylines(axes[1], r2, r_lines, color="#d62728", linewidth=1.0)
    axes[1].set_title("After repair", fontsize=11, weight="bold")
    axes[1].axis("off")
    axes[1].set_aspect("equal")
    axes[1].text(
        0.02,
        0.02,
        f"points={len(r_pts)}  lines={len(r_lines)}  components={r_comp}",
        transform=axes[1].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    fig.suptitle("Normal_1 Connectivity Before/After Repair", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def case_metrics(case: str) -> Dict[str, float]:
    base = ROOT / "outputs" / "quant" / case
    s1 = load_json(base / "step1_segmentation" / "metrics.json")["metrics"]
    s2 = load_json(base / "step2_centerline" / "metrics.json")["metrics"]
    s3 = load_json(base / "step3_repair" / "metrics.json")["metrics"]
    s4 = load_json(base / "step4_features" / "metrics.json")["metrics"]
    s5 = load_json(base / "step5_render" / "metrics.json")["metrics"]
    return {
        "case": case,
        "step1_dice": float(s1["dice"]),
        "step1_hd95": float(s1["hd95_mm"]),
        "step2_p95": float(s2["pred2gt_p95"]),
        "step3_p95": float(s3["pred2gt_p95"]),
        "step4_diff": float(s4["branch_count_abs_diff"]),
        "step4_cos": float(s4["descriptor_cosine"]),
        "step5_p95": float(s5["pred2gt_p95"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate teacher report images from latest quant outputs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "assets" / "teacher_report_20260209",
        help="Output directory for generated assets.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_n1 = load_json(ROOT / "outputs" / "quant" / "Normal_1" / "step1_segmentation" / "metrics.json")["metrics"]
    metrics_n2 = load_json(ROOT / "outputs" / "quant" / "Normal_2" / "step1_segmentation" / "metrics.json")["metrics"]
    step3_n1 = load_json(ROOT / "outputs" / "quant" / "Normal_1" / "step3_repair" / "metrics.json")["metrics"]
    step3_n2 = load_json(ROOT / "outputs" / "quant" / "Normal_2" / "step3_repair" / "metrics.json")["metrics"]
    step4_n1 = load_json(ROOT / "outputs" / "quant" / "Normal_1" / "step4_features" / "metrics.json")["metrics"]
    step4_n2 = load_json(ROOT / "outputs" / "quant" / "Normal_2" / "step4_features" / "metrics.json")["metrics"]

    make_pipeline_overview(out_dir / "pipeline_overview.png")
    make_cross_case_summary([case_metrics("Normal_1"), case_metrics("Normal_2")], out_dir / "cross_case_summary.png")

    make_step1_overlay(
        case="Normal_1",
        ct_path=ROOT / "ASOCA2020" / "Normal" / "CTCA_nii" / "Normal_1.nii.gz",
        gt_mask_path=ROOT / "ASOCA2020" / "Normal" / "Annotations_nii" / "Normal_1.nii.gz",
        pred_mask_path=ROOT / "outputs" / "quant" / "Normal_1" / "step1_segmentation" / "totalseg_output" / "coronary_arteries.nii.gz",
        metrics=metrics_n1,
        out_path=out_dir / "normal_1_step1_overlay.png",
    )
    make_step1_overlay(
        case="Normal_2",
        ct_path=ROOT / "ASOCA2020" / "Normal" / "CTCA_nii" / "Normal_2.nii.gz",
        gt_mask_path=ROOT / "ASOCA2020" / "Normal" / "Annotations_nii" / "Normal_2.nii.gz",
        pred_mask_path=ROOT / "outputs" / "quant" / "Normal_2" / "step1_segmentation" / "totalseg_output" / "coronary_arteries.nii.gz",
        metrics=metrics_n2,
        out_path=out_dir / "normal_2_step1_overlay.png",
    )

    make_centerline_overlay(
        case="Normal_1",
        pred_vtp=ROOT / "outputs" / "quant" / "Normal_1" / "step3_repair" / "repaired.vtp",
        gt_vtp=ROOT / "ASOCA2020" / "Normal" / "Centerlines" / "Normal_1.vtp",
        metrics=step3_n1,
        out_path=out_dir / "normal_1_step3_centerline.png",
    )
    make_centerline_overlay(
        case="Normal_2",
        pred_vtp=ROOT / "outputs" / "quant" / "Normal_2" / "step3_repair" / "repaired.vtp",
        gt_vtp=ROOT / "ASOCA2020" / "Normal" / "Centerlines" / "Normal_2.vtp",
        metrics=step3_n2,
        out_path=out_dir / "normal_2_step3_centerline.png",
    )

    make_connectivity_plot(
        baseline_vtp=ROOT / "outputs" / "quant" / "Normal_1" / "step2_centerline" / "pred_centerline_poly.vtp",
        repaired_vtp=ROOT / "outputs" / "quant" / "Normal_1" / "step3_repair" / "repaired.vtp",
        out_path=out_dir / "normal1_connectivity_before_after.png",
    )

    make_step4_branch_plot(
        case="Normal_1",
        pred_features_dir=ROOT / "outputs" / "quant" / "Normal_1" / "features_pred",
        gt_features_dir=ROOT / "outputs" / "quant" / "Normal_1" / "features_gt",
        metrics=step4_n1,
        out_path=out_dir / "normal_1_step4_branch_length.png",
    )
    make_step4_branch_plot(
        case="Normal_2",
        pred_features_dir=ROOT / "outputs" / "quant" / "Normal_2" / "features_pred",
        gt_features_dir=ROOT / "outputs" / "quant" / "Normal_2" / "features_gt",
        metrics=step4_n2,
        out_path=out_dir / "normal_2_step4_branch_length.png",
    )

    print(f"[done] assets regenerated in {out_dir}")


if __name__ == "__main__":
    main()
