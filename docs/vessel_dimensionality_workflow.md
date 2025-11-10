## Coronary Vessel Dimensionality Reduction & Reconstruction

This note spells out actionable steps (with recommended tooling and pseudo-code) to finish Tasks 3–4 using the ASOCA masks.

---

### 1. Preliminaries

**Inputs**
- Binary coronary mask (`*.nii.gz`) produced by the segmentation stage.
- Optional lumen labels (multi-class) and/or centreline seeds (if provided by ASOCA `Centerlines` folder).

**Outputs**
- Per-branch descriptors: centreline samples, cross-sectional polar curves, normalised feature vectors.
- A combined 3D mesh constructed by sweeping the cross-sections along each branch and taking the union.

**Dependencies**
- `SimpleITK`, `nibabel`, `numpy`, `networkx`, `scikit-image`, `scipy`, `pyvista` (or `vedo`) for surface reconstruction.

---

### 2. Centreline extraction and branch decomposition

> Goal: express the vessel tree as ordered polylines per anatomical branch.

1. **Load segmentation**
   ```python
   import nibabel as nib
   mask = nib.load(seg_path).get_fdata() > 0.5
   spacing = nib.load(seg_path).header.get_zooms()[:3]
   ```

2. **Binary cleanup (gap mitigation)**
   - Apply `scipy.ndimage.binary_closing` with an anisotropic structuring element (respecting voxel spacing).
   - Fill holes with `binary_fill_holes`.
   - Keep largest connected component (`scipy.ndimage.label`).

3. **Skeletonise**
   ```python
   from skimage.morphology import skeletonize_3d
   skeleton = skeletonize_3d(mask)
   ```
   - Optionally smooth by pruning spurs shorter than 2 mm (remove skeleton voxels whose geodesic distance to a branch point is below a threshold).

4. **Graph construction**
   - Convert skeleton voxels to nodes in a graph (`networkx.Graph`), edges connect 26-neighbours.
   - Classify nodes:
     - Degree 1 → endpoints
     - Degree ≥3 → bifurcations
     - Degree 2 → interior points

5. **Branch tracing**
   - Perform depth-first traversal between consecutive high-degree nodes to extract simple paths.
   - Each path becomes a branch polyline (ordered by cumulative path length).
   - Store the polyline as `{ "name": branch_id, "points": [ (x,y,z), … ], "spacing": spacing }`.

6. **Anatomical labelling**
   - Use heuristics on endpoint coordinates relative to heart axes (e.g., identify RCA vs LAD by right/left orientation and anterior positioning).
   - Alternatively import provided `Centerlines/*.vtp` if available (see `ASOCA2020/Normal/Centerlines`), map to segmentation indices, and skip steps 3–5.

7. **Gap completion**
   - For missing segments (identified by sudden jumps > 1.5 mm), interpolate using cubic splines between surrounding known points.
   - Record a `source` flag per point (`"observed"` vs `"interpolated"`) for later reporting.

---

### 3. Cross-sectional polar descriptors

For each branch polyline \( C(s) \), parametrised by arclength \( s \in [0, L] \):

1. **Resample**
   - Interpolate the polyline to a fixed number of points (e.g., 100 samples) using linear interpolation on cumulative arclength.

2. **Local frame**
   - Compute tangent \( \mathbf{t}_i = \frac{C_{i+1} - C_{i-1}}{\|C_{i+1} - C_{i-1}\|} \).
   - Choose a reference vector \( \mathbf{u}_0 \) (e.g., global z-axis) and derive an orthonormal basis via Gram–Schmidt:
     ```
     n_i = normalize(t_i × u0)
     b_i = t_i × n_i
     ```

3. **Slice extraction**
   - Take a cubic patch centred on \( C_i \) (radius ≈ 3 mm).
   - For each voxel within the patch, project relative coordinates \( \mathbf{p} = (x - C_i) \) onto the normal plane:
     ```
     r = sqrt((p·n_i)^2 + (p·b_i)^2)
     theta = atan2(p·b_i, p·n_i)
     ```
   - Keep voxels where \( |p·t_i| < h \) (h ≈ 1 voxel) and the segmentation is 1.

4. **Polar profile**
   - Bin angles into \( N_\theta \) slots (e.g., 72 bins @ 5°) and record the maximal radius per bin.
   - Normalise radius by mean radius for the branch (`r_norm = r / r_mean`).
   - Save vector \( \mathbf{f}_i = [r_1, r_2, …, r_{N_\theta}] \).

5. **Feature aggregation**
   - Concatenate:
     - PCA coefficients of centreline (first 8 components from centred coordinates).
     - Fourier descriptors of the polar profile (first 10 magnitudes).
     - Local curvature and torsion (finite differences on tangents).
   - Store as JSON/NumPy arrays per branch: `features/{branch}.npz`.

6. **Handling discontinuities**
   - If the slice yields <50% of bins populated (due to segmentation gaps), interpolate `r(theta)` using neighbours \( i-1 \) and \( i+1 \); tag the slice as low-confidence.

---

### 4. Dimensionality reduced representation

Use the per-branch feature matrices to build statistical summaries:

1. **Branch PCA**
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=10)
   coeffs = pca.fit_transform(branch_feature_matrix)
   ```

2. **Global descriptor**
   - Concatenate branch PCA scores in fixed order (LM, LAD, LCx, RCA, diagonal, marginal).
   - Append stenosis metrics (min radius ratio, lesion length) for pathological insight.

3. **Storage**
   - Define a new metadata structure (`vessel_seg/metadata.py` integration) to hold:
     ```json
     {
       "case_id": "Diseased_01",
       "branches": {
         "LeftMain": {"centerline": "...npz", "features": "...npz", "confidence": 0.92},
         ...
       },
       "global_descriptor": "features/Diseased_01_global.npy"
     }
     ```

---

### 5. 3D reconstruction from curves

> Once cross-sections are available, create a surface by sweeping and merging.

1. **Tube sweep**
   - For each branch sample \( C_i \), reconstruct the contour from \( r(\theta) \):
     ```
     contour_points = [
         C_i + r(theta_k) * cos(theta_k) * n_i + r(theta_k) * sin(theta_k) * b_i
         for theta_k in angles
     ]
     ```
   - Build a `pyvista.PolyData` or `vedo.Mesh` from stacked contours (loft/sweep operation).

2. **Branch union**
   - Use boolean union (`mesh1.boolean_union(mesh2)`) or simply merge vertices and run Poisson surface reconstruction for a smooth coronary tree.

3. **Export**
   - Save as `.vtp`/`.stl` for inspection in 3D Slicer.
   - Embed the origin spacing in metadata for accurate measurements.

4. **Quick visual check**
   ```python
   import pyvista as pv
   plotter = pv.Plotter()
   plotter.add_mesh(reconstructed_mesh, color="salmon", smooth_shading=True)
   plotter.add_points(np.vstack(branch_centrelines), color="yellow", point_size=5)
   plotter.show()
   ```

---

### 6. Suggested implementation roadmap

1. Implement `vessel_seg/shape.py` with routines:
   - `extract_branches(seg_path) -> dict`
   - `compute_polar_profiles(branch) -> np.ndarray`
   - `export_branch_features(branch_dict, output_dir)`
2. Create CLI scripts:
   - `python -m vessel_seg.shape extract --seg <mask> --out features/case_id`
   - `python -m vessel_seg.shape reconstruct --features features/case_id --mesh outputs/case_id.vtp`
3. Test on a single ASOCA case and validate in 3D Slicer.
4. Integrate descriptors into metadata (extend `SegmentationMetadata`).

This workflow maintains the conceptual requirements (branch-wise centreline, polar descriptors, scale normalisation) while providing concrete processing steps and fallback strategies for imperfect masks. Once implemented, the resulting descriptors enable statistical analysis, and the reconstructed mesh can be visualised or further processed (e.g., CFD simulations).

---

### 7. Worklist

- **check totalseg 的可行性**
  - [x] 评估间断是否不可避免
  - [x] 分析黑箱风险
    - [x] 调研配置环境的可能性
    - [x] 评估 finetune 可行性
    - [x] 评估重新训练可行性
- **分割**
  - [ ] 本地部署 totalseg 流水线
  - [ ] 在测试集上统计 Dice/HD95
  - [ ] 实现连续的解剖学分割
- **降维**
  - [ ] 构建三维中心线
  - [ ] 处理分支
- **FGPM + Bayesian 融合**
  - [ ] 使用 `python -m vessel_seg.fgpm fit` 为重点分支拟合 Fourier-GP 先验（order≈6, degree=2）。
  - [ ] 通过 `vessel_seg.edge train/predict` 获得深度边缘概率，供 `fgpm infer` 融合。
  - [ ] 将 `fgpm propagate` 生成的伪注释纳入交互后验，观察 Dice/ASSD 提升。
    - [ ] 完成连续的解剖学分割
    - [ ] 复用 VMTK 分支逻辑
  - [ ] 对齐学长论文的方法
- **重建**
  - [ ] 解决“拉丝效应”
    - [ ] 评估取消平扫的影响
