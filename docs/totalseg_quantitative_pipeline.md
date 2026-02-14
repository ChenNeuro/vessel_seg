# TotalSeg Quantitative Pipeline (Step1~Step5)

This document defines a clear, modular, and fully-quantitative workflow.

## Stage Contract

Each stage writes:

- `outputs/.../<stage>/metrics.json`
- `outputs/.../<stage>/metrics.csv`

So every stage can be CI-gated independently.

## Step1 Segmentation (CT -> mask)

Goal:

- Use TotalSegmentator prediction mask vs GT mask.

Quant metrics:

- `dice`
- `iou`
- `asd_mm`
- `hd95_mm`
- `hdmax_mm`
- `volume_abs_diff_mm3`

Command:

```bash
python scripts/quant_pipeline.py step1 \
  --pred-mask <totalseg_mask.nii.gz> \
  --gt-mask <gt_mask.nii.gz> \
  --out-dir outputs/quant/<case>
```

Run TotalSegmentator directly from CT inside this repo:

```bash
python scripts/step1_totalseg_segment_and_eval.py \
  --ct <ct.nii.gz> \
  --gt-mask <gt_mask.nii.gz> \
  --out-dir outputs/quant/<case> \
  --task coronary_arteries \
  --pred-file coronary_arteries.nii.gz \
  --totalseg-arg=--ml
```

Note: `coronary_arteries` currently does not support `--fast`.

## Step2 Centerline extraction (mask -> centerline)

Goal:

- Extract centerline points from TotalSeg mask and compare with GT centerline.

Quant metrics:

- `pred2gt_mean`, `pred2gt_p95`, `pred2gt_max`
- `gt2pred_mean`, `gt2pred_p95`, `gt2pred_max`
- `coverage_pred@1mm`, `coverage_gt@1mm`

Command:

```bash
python scripts/step2_centerline_totalseg.py \
  --totalseg-mask <totalseg_mask.nii.gz> \
  --gt-centerline <gt_centerline.vtp> \
  --out-dir outputs/quant/<case> \
  --backend skeleton \
  --thr 1.0
```

or from unified CLI:

```bash
python scripts/quant_pipeline.py step2 \
  --seg-mask <totalseg_mask.nii.gz> \
  --gt-centerline <gt_centerline.vtp> \
  --out-dir outputs/quant/<case> \
  --backend vmtk \
  --thr 1.0
```

## Step3 Centerline repair (repaired vs GT)

Goal:

- Increase repair importance and quantify repaired centerline quality.

Quant metrics:

- same centerline metrics as Step2
- optional delta vs baseline centerline

Command:

```bash
python scripts/quant_pipeline.py step3 \
  --repaired-centerline <repaired.vtp> \
  --gt-centerline <gt_centerline.vtp> \
  --baseline-centerline <before_repair.vtp> \
  --out-dir outputs/quant/<case> \
  --thr 1.0
```

## Step4 Vessel features (TotalSeg vs GT)

Goal:

- Compare feature extraction outputs from predicted mask and GT mask.

Required inputs:

- feature dirs produced by `vessel_seg.shape extract`

Quant metrics:

- `pred_branch_count`, `gt_branch_count`, `branch_count_abs_diff`
- `descriptor_l1`, `descriptor_l2`, `descriptor_cosine`

Command:

```bash
python scripts/quant_pipeline.py step4 \
  --pred-features <pred_features_dir> \
  --gt-features <gt_features_dir> \
  --out-dir outputs/quant/<case>
```

## Step5 Render reconstruction (rendered vs GT mask)

Goal:

- Quantify reconstruction rendering against GT mask boundary.

Input options:

- `--pred-mesh` directly, or
- `--pred-features` (script will reconstruct mesh first)

Quant metrics:

- symmetric boundary distance metrics (`pred2gt_*`, `gt2pred_*`)
- coverage under threshold (`coverage_pred@1mm`, `coverage_gt@1mm`)

Command:

```bash
python scripts/quant_pipeline.py step5 \
  --pred-features <pred_features_dir> \
  --gt-mask <gt_mask.nii.gz> \
  --out-dir outputs/quant/<case> \
  --thr 1.0
```

## Full 5-step run

```bash
python scripts/quant_pipeline.py all \
  --pred-mask <totalseg_mask.nii.gz> \
  --gt-mask <gt_mask.nii.gz> \
  --gt-centerline <gt_centerline.vtp> \
  --step2-backend skeleton \
  --repaired-centerline <repaired.vtp> \
  --pred-features <pred_features_dir> \
  --gt-features <gt_features_dir> \
  --pred-mesh <pred_mesh.vtp> \
  --out-dir outputs/quant/<case> \
  --thr 1.0
```

## One-click run for Normal_1

For fast end-to-end execution (Step1~Step5, with auto fallback for step2 VTP line issues):

```bash
python scripts/run_normal1_pipeline.py --skip-existing
```

Outputs:

- `outputs/quant/Normal_1/pipeline_run_summary.json`
- per-stage metrics under `outputs/quant/Normal_1/step*/metrics.json`

Useful options:

- `--step2-backend vmtk|skeleton` (default: `vmtk`)
- `--thr 1.0` (coverage threshold in mm)
- `--pred-extract-arg \"--num-samples 120\"` (repeatable)
- `--gt-extract-arg \"--num-samples 120\"` (repeatable)
- `--dry-run` (print commands only)

Notebook wrapper:

- `notebooks/normal1_full_pipeline.ipynb`

## Note on TotalSegmentator integration

This pipeline treats TotalSeg output mask as standard input contract (`.nii/.nii.gz`).

You can run TotalSegmentator externally first, then feed its mask path to `step1/step2`.

For 3D Slicer usage, see `docs/totalseg_slicer_guide.md`.
