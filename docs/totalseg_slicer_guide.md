# TotalSegmentator in 3D Slicer (Practical Guide)

This guide focuses on using official TotalSegmentator inside 3D Slicer and connecting outputs to this repo.

## 1) Install Slicer extension

1. Open 3D Slicer.
2. Go to `Extension manager`.
3. Search `TotalSegmentator`.
4. Install extension `SlicerTotalSegmentator`.
5. Restart Slicer.

## 2) First run setup

1. Open module `TotalSegmentator`.
2. On first execution, click install/confirm dependencies.
3. If setup does not complete in one shot, restart Slicer and run again.

## 3) Run coronary task

1. Load CT volume.
2. In `TotalSegmentator` module:
   - select your CT as input
   - task: `coronary arteries`
   - optional: enable `fast` mode
3. Run segmentation and wait for completion.

## 4) Export masks for this repo

1. Save predicted coronary mask as `NIfTI` (`.nii.gz`), e.g. `case_totalseg_coronary.nii.gz`.
2. Prepare GT mask (`.nii.gz`) and GT centerline (`.vtp`).

## 5) Quantitative evaluation in this repo

### Step1 (segmentation quality)

```bash
python scripts/step1_totalseg_segment_and_eval.py \
  --ct <ct.nii.gz> \
  --gt-mask <gt_mask.nii.gz> \
  --out-dir outputs/quant/<case> \
  --task coronary_arteries \
  --pred-file coronary_arteries.nii.gz
```

Note: `coronary_arteries` task does not support `--fast`.

### Step2 (centerline from TotalSeg mask)

```bash
python scripts/step2_centerline_totalseg.py \
  --totalseg-mask outputs/quant/<case>/step1_segmentation/totalseg_output/coronary_arteries.nii.gz \
  --gt-centerline <gt_centerline.vtp> \
  --out-dir outputs/quant/<case> \
  --backend skeleton \
  --thr 1.0
```

If VMTK is available:

```bash
python scripts/step2_centerline_totalseg.py \
  --totalseg-mask outputs/quant/<case>/step1_segmentation/totalseg_output/coronary_arteries.nii.gz \
  --gt-centerline <gt_centerline.vtp> \
  --out-dir outputs/quant/<case> \
  --backend vmtk \
  --thr 1.0
```

## 6) Outputs to check

- `outputs/quant/<case>/step1_segmentation/metrics.json`
- `outputs/quant/<case>/step1_segmentation/metrics.csv`
- `outputs/quant/<case>/step2_centerline/metrics.json`
- `outputs/quant/<case>/step2_centerline/metrics.csv`

These files are the quantitative contract for CI/perf gate.
