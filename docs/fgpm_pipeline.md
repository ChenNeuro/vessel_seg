## Fourier-Gaussian-Process Modelling Pipeline

Adapted from Wang et al., *An Efficient Muscle Segmentation Method via Bayesian Fusion of Probabilistic Shape Modeling and Deep Edge Detection* (IEEE TBME, 2024).

### 1. Radial modelling (Fourier descriptors)

- Cross-sectional profiles exported by `vessel_seg.shape extract` already store `raw_profiles` (radius per angle bin) and orthonormal frames.
- `vessel_seg.fgpm` converts each valid slice into Fourier coefficients (Eq. (2) in the paper) with order `N` (default 6).
- The resulting parameter vector per slice is:
  ```
  Ψ = [a0, a1, b1, …, aN, bN]
  ```
  where `a0` encodes mean radius and the higher-order terms describe protrusions/stenoses.

### 2. Axial modelling (Gaussian processes)

- For every coefficient `ψ_j(h)` we fit a polynomial mean `m_j(h) = Σ α_d h^d` (degree configurable) plus an RBF kernel GP.
- Hyperparameters maximise the log marginal likelihood (Eq. (10) from the paper) via L-BFGS-B.
- The trained `JSON` bundle stores polynomial coefficients, kernel variance/length-scale/noise, and valid height range.

Training command:
```bash
python -m vessel_seg.fgpm fit \
  --features-dirs features/case_* \
  --branch Branch_00 \
  --order 6 \
  --degree 2 \
  --output models/branch00_fgpm.json
```

### 3. Annotation fusion

- Sparse annotations: run `vessel_seg.shape extract` on a volume that only contains manually labelled slices (others can be zero). Filter by confidence via `--annotation-min-conf`.
- Annotation propagation: `python -m vessel_seg.fgpm propagate --image volume.nii.gz --mask sparse_mask.nii.gz --source 30 --target 31 --axis 2 --output-mask mask_aug.nii.gz` performs ROI-based diffeomorphic Demons registration to produce pseudo labels for adjacent slices (matching Section II-C2 in the paper).

### 4. Edge feature modelling

- `vessel_seg.edge` implements a UAED-style multi-scale edge detector. Prepare a manifest JSON with image/edge-mask pairs:
  ```json
  [
    {"image": "volumes/case001.nii.gz", "edge": "edges/case001_manual.nii.gz"},
    {"image": "volumes/case002.nii.gz", "edge": "edges/case002_manual.nii.gz"}
  ]
  ```
- Train and infer:
  ```bash
  python -m vessel_seg.edge train --manifest data/edge_manifest.json --output weights/edge_net.pth --epochs 40
  python -m vessel_seg.edge predict --weights weights/edge_net.pth --volume volumes/case010.nii.gz --output outputs/case010_edges.nii.gz
  ```
- The predicted probability map replaces the `Pβ(I_edge)` term from Eq. (12) in the paper.

### 5. Bayesian fusion and MAP inference

```bash
python -m vessel_seg.fgpm infer \
  --model models/branch00_fgpm.json \
  --branch-features features/case010/Branch_00.npz \
  --annotations features/case010_sparse/Branch_00.npz \
  --edge-map outputs/case010_edges.nii.gz \
  --annotation-noise 0.4 \
  --edge-weight 1.2 \
  --output outputs/case010_fgpm_post.npz
```

- Stage 1 (interactive posterior): compute Eq. (13–14) to fuse annotations with the GP prior.
- Stage 2 (edge-aware MAP): for each slice solve Eq. (15) by minimising `-ln P(Ψ|h) - ln P(I_edge | L_Ψ)` with L-BFGS. The energy couples the Gaussian prior with the deep edge confidence aggregated along the reconstructed contour.
- Outputs include per-slice coefficients and reconstructed radii arrays that can be swept back into meshes with `vessel_seg.shape reconstruct`.

### 6. Integration tips

- Always ensure the `angles` arrays are identical across training cases (default from `shape extract` is 72 bins).
- Normalise arclength `h` to `[0, 1]` before fitting so that models transfer to branches of different absolute lengths.
- When the edge detector is unavailable, set `--edge-weight 0` to fall back to the interactive posterior (equivalent to skipping Eq. (15)).
- Use the saved covariance diagonals to report uncertainty bands for each slice (low confidence when diagonal values remain large after MAP).
