# vessel_seg workflow

基于 ASOCA 冠脉数据的中心线树与分支形状建模流水线。

## 环境
```bash
conda create -n vessel_seg -c conda-forge python=3.10 simpleitk scikit-image vtk matplotlib numpy scipy
conda activate vessel_seg
pip install meshio rich pyvista
```

## 流水线
1) **掩膜 → 中心线（基线骨架）**
```bash
python scripts/extract_centerline_from_mask.py \
  --mask ASOCA2020/Normal/Annotations/Normal_1.nrrd \
  --out ASOCA2020/Normal/Centerlines/Normal_1_extracted.vtp
```
或使用已有中心线：`ASOCA2020/Normal/Centerlines/Normal_1.vtp`

2) **建树（父子/λ/θ/φ）**
```bash
python scripts/build_centerline_tree.py \
  --vtp ASOCA2020/Normal/Centerlines/Normal_1.vtp \
  --out outputs/normal1_tree.json
```

3) **生成分支张量（截面半径，自动对齐+可调截断）**
```bash
python scripts/build_branch_dataset.py \
  --vtp ASOCA2020/Normal/Centerlines/Normal_1.vtp \
  --mask ASOCA2020/Normal/Annotations/Normal_1.nrrd \
  --tree outputs/normal1_tree.json \
  --out outputs/normal1_branch_dataset.npz \
  --branch_dir outputs/branches_normal1 \
  --start_offset 3 \
  --start_offset_percent 5     # 例如按分支长度百分比截断（可选，默认启用自动距离图对齐）
```

4) **分支相似性 & PCA**
```bash
python scripts/branch_similarity.py \
  --npz outputs/<case>/branch_dataset.npz \
  --out_dir outputs/<case>/similarity \
  --pca_dim 8 --heatmap
```

5) **可视化**
- 中心线 vs 分割表面：`python scripts/plot_centerline_vs_gt.py --vtp ... --mask ...`
- 中心线 vs 分割骨架：`python scripts/compare_centerline_vs_gt_centerline.py --vtp ... --mask ...`
- 树的锥子图（含 parent/λ）：`python scripts/plot_tree_fs_cones.py --vtp ... --out ...`
- Matplotlib 交互 3D：`python scripts/plot_centerline_matplotlib_interactive.py --vtp ... --mask ...`

6) **批量运行多病例**
```bash
python scripts/run_pipeline.py \
  --pattern ASOCA2020/Normal/Centerlines/Normal_*.vtp \
  --mask_dir ASOCA2020/Normal/Annotations_nii \
  --start_offset_percent 0 2.5 5 7.5 10  # 多种截断自动批跑（输出带 pct 后缀）
```

7) **跨病例聚合**
```bash
python scripts/aggregate_branch_datasets.py \
  --npz_glob 'outputs/Normal_*/branch_dataset*.npz' \
  --out outputs/aggregate/branch_dataset_normal_all.npz \
  --pca_dim 8 --heatmap
```

## 目录说明
- `ASOCA2020/Normal/`：示例 CTA、掩膜、官方中心线。
- `outputs/`：树 JSON、分支数据、相似度矩阵、可视化 PNG 等。
- `scripts/`：流水线各步骤脚本（提取中心线、建树、分支张量、相似性、可视化）。
- `notebooks/workflow_demo.ipynb`：运行以上步骤的示例 Notebook（需 GUI/交互环境）。

## 训练/建模思路（对应论文）
- 树先验：统计 {parent, λ, 分叉角度}。
- 分支形状先验：将 `radii(K×M)` 张量做降维（PCA/GP/Fourier），得到低维 latent，后续可训练生成/回归模型。

## 边缘检测（UAED 风格）
- 通过 `vessel_seg.edge` 训练 UAED 风格的多尺度边缘检测器：
  ```bash
  python -m vessel_seg.edge train \
    --manifest data/edge_manifest.json \
    --output weights/edge_net.pth \
    --epochs 40 --batch-size 12 --device cuda
  ```
  `manifest` 为 JSON 数组，每条记录包含 `{"image": "...nii.gz", "edge": "...nii.gz"}`。
- 推理阶段按轴切片生成整卷概率：
  ```bash
  python -m vessel_seg.edge predict \
    --weights weights/edge_net.pth \
    --volume volumes/case001.nii.gz \
    --output outputs/case001_edges.nii.gz
  ```
- 将生成的概率图传入 `vessel_seg.fgpm infer --edge-map`，即可完成类似论文中的边缘-形状 Bayesian 融合。
- 旧版边缘检测流程已归档至 `archive/edge_legacy/`（核心代码仍在 `vessel_seg/edge.py`），主线推荐使用 RCF 生成的边缘并通过转换脚本融入管线。

## RCF 边缘（third_party）融合

1. 使用 `scripts/prepare_rcf_asoca.py` 生成 2D PNG 训练对并在 `third_party/RCF-PyTorch` 下细调 RCF。
2. 得到的 RCF 预测 PNG（按切片顺序命名）可用转换脚本堆叠为 NIfTI，便于后续形状/分割：
   ```bash
   python scripts/rcf_to_nifti.py \
     --rcf-dir third_party/RCF-PyTorch/results/RCF/ASOCA2020/epoch20-test \
     --reference ASOCA2020/Normal/CTCA_nii/Normal_1.nii.gz \
     --output outputs/Normal_1_rcf_edges.nii.gz
   ```
   生成的 `outputs/Normal_1_rcf_edges.nii.gz` 可直接作为 `--edge-map` 提供给 `vessel_seg.fgpm infer` 或其他后处理。

## RCF fine-tuning on ASOCA2020

1. 生成 RCF 需要的 2D PNG 训练对：
   ```bash
   python scripts/prepare_rcf_asoca.py \
     --image-dirs ASOCA2020/Diseased/CTCA_nii ASOCA2020/Normal/CTCA_nii \
     --label-dirs ASOCA2020/Diseased/Annotations_nii ASOCA2020/Normal/Annotations_nii \
     --output-dir third_party/RCF-PyTorch/data/ASOCA2020_rcf \
     --val-ratio 0.1 \
     --target-height 256 --target-width 256
   ```
   该脚本会按病例划分训练/验证集，输出 `train.lst` 与 PNG 文件夹，并写入 `summary.json`。
2. 在 `vesselfm` 环境下启动 RCF 细调（默认使用仓库内 `bsds500_pascal_model.pth` 作为初始化）：
   ```bash
   cd third_party/RCF-PyTorch
   python Ktrain.py \
     --dataset data/ASOCA2020_rcf \
     --save-dir results/RCF/ASOCA2020 \
     --batch-size 2 \
     --gpu 0
   ```
   若已训练过可通过 `--resume results/RCF/ASOCA2020/checkpoint_epochX.pth` 继续。输出的日志、检查点与推理结果均保存在 `--save-dir`。
