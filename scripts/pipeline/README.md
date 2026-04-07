# Pipeline Scripts

This directory contains canonical entry points for the rebuilt five-stage pipeline.

Current entry point:

- `run_case_pipeline.py`
- `generate_case_overviews.py`
- `bootstrap_semantic_prior.py`
- `analyze_semantic_consistency.py`
- `analyze_canonical_consistency.py`
- `analyze_branch_clusters.py`
- `project_case_synthetic_xray.py`
- `project_case_synthetic_xray_sweep.py`

Recommended usage:

```bash
python scripts/pipeline/run_case_pipeline.py \
  --case-id Normal_1 \
  --ct ../ASOCA2020/Normal/CTCA_nii/Normal_1.nii.gz \
  --mask ../ASOCA2020/Normal/Annotations_nii/Normal_1.nii.gz \
  --seg-backend existing_mask \
  --repair-mode topology_only
```

Batch overview usage:

```bash
python scripts/pipeline/generate_case_overviews.py \
  --cohort Normal \
  --report-tag normal_gallery \
  --limit 6
```

Semantic prior bootstrap:

```bash
python scripts/pipeline/bootstrap_semantic_prior.py \
  --cohort Normal \
  --cohort Diseased \
  --output-root outputs_reorganized_runs \
  --out outputs_reorganized_runs/analysis/semantic_priors/asoca40_prior.json
```

Semantic consistency analysis:

```bash
python scripts/pipeline/analyze_semantic_consistency.py \
  --summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_named_gallery_40/summary.csv
```

Canonical naming consistency analysis:

```bash
python scripts/pipeline/analyze_canonical_consistency.py \
  --summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/summary.csv
```

Global unlabeled branch clustering:

```bash
python scripts/pipeline/analyze_branch_clusters.py \
  --summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/summary.csv \
  --clustering-mode side_first
```

Compress clusters to the maximum branch count observed across cases:

```bash
python scripts/pipeline/analyze_branch_clusters.py \
  --summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/summary.csv \
  --clustering-mode side_first \
  --match-max-case-branches
```

Minimal synthetic 3D-2D projection:

```bash
python scripts/pipeline/project_case_synthetic_xray.py \
  --case-dir outputs_reorganized_runs/cases/Normal_2 \
  --out-dir outputs_reorganized_runs/analysis/reconstruction_3d2d/normal2_demo \
  --lao-rao-deg 25 \
  --cra-cau-deg 10
```

Multi-view synthetic projection sweep:

```bash
python scripts/pipeline/project_case_synthetic_xray_sweep.py \
  --case-dir outputs_reorganized_runs/cases/Normal_2 \
  --out-dir outputs_reorganized_runs/analysis/reconstruction_3d2d/normal2_angle_sweep
```

This sweep now writes both:

- `projection_sweep_gallery.png`：中心线投影总览
- `projection_wall_sweep_gallery.png`：带血管宽度的血管壁投影总览
