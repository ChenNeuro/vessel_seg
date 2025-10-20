## ASOCA Coronary Segmentation Roadmap

### 1. Model selection & environment

- **Leaderboard snapshot:** `data/asoca_leaderboard_top10.json` records the latest public standings (Dice≈0.89 for the top submission as of 2024‑03‑12).
- **Accessible codebase:** Public implementations matching the top leaderboard ranks are not available. The strongest openly maintained baseline is the official [`nnUNet`](https://github.com/MIC-DKFZ/nnUNet) (v2) repository (5th place `junma` submission is nnU-Net based).
- **Environment setup:**
  ```bash
  conda env create -f env/asoca_nnunet.yaml
  conda activate asoca_nnunet
  pip install -e third_party/nnUNet  # editable install for nnUNetv2
  ```
- **Pre-trained starting point (optional):**
  - Download Task503 TotalSegmentator coronary weights from Zenodo (`https://zenodo.org/records/7271576`) when GPU resources are limited; convert to nnUNetv2 format via `nnUNetv2_convert_totalseg_task`.
  - Otherwise perform full nnUNetv2 3d_fullres training on ASOCA CTA volumes.

### 2. Training & inference with ASOCA

1. **Data ingestion**
   - Place ASOCA dataset under `datasets/ASOCA/raw/`.
   - Convert `nhdr/raw` inputs with `vessel_seg.conversion.convert_nrrd_to_nii`.
   - Structure according to nnUNetv2 naming via:
     ```bash
     nnUNetv2_plan_and_preprocess -d 103 -c 3d_fullres --verify_dataset_integrity
     ```
     (assign `Dataset103_ASOCA` to avoid task collisions).
2. **Training**
   ```bash
   nnUNetv2_train 103 3d_fullres all \
     --use-compressed --npz --disable_checkpointing False
   ```
   - Expect ~24h on A100 (AMP enabled). For Mac/CPU you can fine-tune with `--epochs 500` and patch cropping to 128³ (see `nnunetv2/configuration.py`).
3. **Inference**
   ```bash
   nnUNetv2_predict -i <input_cta_dir> -o outputs/asoca_predictions \
     -d 103 -c 3d_fullres --save_probabilities
   ```
4. **Post-processing**
   - Use largest-component filtering and centreline continuity gap filling (see `vessel_seg.metadata.harmonise_label_names` for consistent naming).

### 3. Quantitative evaluation (Task 2)

| Metric | Implementation | Notes |
|--------|----------------|-------|
| Dice coefficient | `nnUNetv2_evaluate_folder` or `scripts/eval_asoca.py` (to implement) | Multi-label per coronary segment |
| HD95 / ASSD | `scipy.ndimage` + `surface_distance` | Evaluate per branch |
| Vessel centreline similarity | VMTK `vmtkcenterlinecomparison` | Combine with radius error stats |

Steps:
```bash
nnUNetv2_evaluate_folder -ref datasets/ASOCA/labelsTs \
  -pred outputs/asoca_predictions --metrics dice hd95 assd \
  --json outputs/asoca_metrics.json
```

Augment with:
```bash
python scripts/compare_asoca.py \
  --gt datasets/ASOCA/labelsTs \
  --pred outputs/asoca_predictions \
  --branch-map configs/asoca_branch_map.yaml
```
(Script to compute per-branch dice/precision and optional detection F1.)

### 4. Dimensionality reduction of vessels (Task 3)

1. **Branch decomposition**
   - Convert segmentation to labelled tree using VMTK: `vmtkcenterlines -ifile seg.nii.gz -ofile centerline.vtp -seedselector pointlist ...`.
   - Map segments to canonical branches (`LeftMain`, `LAD`, `LCx`, `RCA`, diagonals).
2. **Centreline parameterisation**
   - Resample each branch to fixed arclength `L=100` samples.
   - Store points `C_b ∈ ℝ^{100×3}` for PCA-ready features.
3. **Cross-sectional profiling**
   - Extract orthogonal planes along centreline.
   - Convert the binary mask intersection into polar coordinates `(r, θ)`.
   - Normalise radii by branch mean to achieve scale invariance.
4. **Gap handling**
   - Use morphological closing (`sitk.BinaryClosingByReconstruction`) then nearest-neighbour interpolation along the centreline.
   - Track masks of synthesized voxels for later uncertainty reporting.
5. **Feature vector**
   - Concatenate centreline PCA coefficients + Fourier descriptors of polar profiles.
   - Optionally append stenosis statistics (min radius ratio, lesion length).

### 5. 3D reconstruction (Task 4)

- Rebuild tubular surface from polar sections:
  1. Sweep each normalised profile along centreline (Frenet frames).
  2. Export to triangulated mesh using `vedo`/`pyvista`.
- Merge branch meshes via union to form coronary tree preview for Slicer.
- Provide `.vtp` surfaces and `.json` feature descriptors for downstream shape modelling.

### 6. Open questions / risks

- Top-performing leaderboard solutions (Dice ≥ 0.89) do **not** provide public code; replicating requires reverse engineering (likely nnUNet variants with tailored losses and data augmentation).
- CTA intensity variability across ASOCA vs. clinical cohorts → recommend HU window standardisation (`[-200, 800]`) and per-volume z-score normalisation.
- Coronary disconnections remain a challenge; incorporate topology-preserving post-processing (minimum spanning tree bridging).
- For shape models, ensure consistent branch ordering and orientation to avoid PCA mode ambiguities.

### 7. Immediate action items

1. Download ASOCA training/validation cohorts and run `nnUNetv2_plan_and_preprocess`.
2. Start baseline training (3d_fullres); log metrics via `wandb`/`tensorboard`.
3. Implement evaluation helper (`scripts/compare_asoca.py`) producing Dice/HD95 centreline stats + error maps.
4. Use `python -m vessel_seg.shape extract --seg <mask> --out features/<case_id>` to generate branch descriptors, then `python -m vessel_seg.shape reconstruct --features features/<case_id>` for quick mesh validation.
5. Prototype centreline extraction on segmented case to validate polar feature derivation.
