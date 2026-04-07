# vessel_seg

这是一个正在重建中的冠状动脉工程仓库。

新的主线不再按“脚本集合”组织，而是按 5 个标准阶段组织：

1. CT 分割
2. 中心线提取
3. 中心线分割与修复
4. 血管壁特征提取
5. 渲染

新的规范入口：

- 设计文档：`docs/rebuild_repo_plan.md`
- 协作规范：`AGENTS.md`
- 路径配置：`vessel_seg/config.py`
- 新流水线包：`vessel_seg/pipeline/`

## 新主线运行示例

```bash
python -m vessel_seg pipeline-case \
  --case-id Normal_1 \
  --ct ../ASOCA2020/Normal/CTCA_nii/Normal_1.nii.gz \
  --mask ../ASOCA2020/Normal/Annotations_nii/Normal_1.nii.gz \
  --seg-backend existing_mask \
  --repair-mode topology_only
```

或者：

```bash
python scripts/pipeline/run_case_pipeline.py \
  --case-id Normal_1 \
  --ct ../ASOCA2020/Normal/CTCA_nii/Normal_1.nii.gz \
  --mask ../ASOCA2020/Normal/Annotations_nii/Normal_1.nii.gz
```

## 数据位置

`ASOCA2020` 已经移出仓库根目录，默认放在 GitHub 根目录下：

```bash
../ASOCA2020
```

仓库内使用路径登记文件：

```bash
data/external/asoca2020.path
```

所有新代码必须通过 `ProjectPaths` 解析数据根路径。

## 环境
```bash
conda create -n vessel_seg -c conda-forge python=3.10 simpleitk scikit-image vtk matplotlib numpy scipy
conda activate vessel_seg
pip install meshio rich pyvista
```

## 旧流水线

下面这些仍可运行，但现在视为兼容层，不再是新的主线组织方式。

1) **掩膜 → 中心线（基线骨架）**
```bash
python scripts/extract_centerline_from_mask.py \
  --mask ../ASOCA2020/Normal/Annotations/Normal_1.nrrd \
  --out ../ASOCA2020/Normal/Centerlines/Normal_1_extracted.vtp
```
或使用已有中心线：`../ASOCA2020/Normal/Centerlines/Normal_1.vtp`

2) **建树（父子/λ/θ/φ）**
```bash
python scripts/build_centerline_tree.py \
  --vtp ../ASOCA2020/Normal/Centerlines/Normal_1.vtp \
  --out outputs/normal1_tree.json
```

3) **生成分支张量（截面半径，自动对齐+可调截断）**
```bash
python scripts/build_branch_dataset.py \
  --vtp ../ASOCA2020/Normal/Centerlines/Normal_1.vtp \
  --mask ../ASOCA2020/Normal/Annotations/Normal_1.nrrd \
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
- 中心线 vs 分割表面：`python scripts/visualization/plot_centerline_vs_gt.py --vtp ... --mask ...`
- 中心线 vs 分割骨架：`python scripts/visualization/compare_centerline_vs_gt_centerline.py --vtp ... --mask ...`
- 树的锥子图（含 parent/λ）：`python scripts/visualization/plot_tree_fs_cones.py --vtp ... --out ...`
- Matplotlib 交互 3D：`python scripts/visualization/plot_centerline_matplotlib_interactive.py --vtp ... --mask ...`

6) **批量运行多病例**
```bash
python scripts/run_pipeline.py \
  --pattern ../ASOCA2020/Normal/Centerlines/Normal_*.vtp \
  --mask_dir ../ASOCA2020/Normal/Annotations_nii \
  --start_offset_percent 0 2.5 5 7.5 10  # 多种截断自动批跑（输出带 pct 后缀）
```

7) **跨病例聚合**
```bash
python scripts/aggregate_branch_datasets.py \
  --npz_glob 'outputs/Normal_*/branch_dataset*.npz' \
  --out outputs/aggregate/branch_dataset_normal_all.npz \
  --pca_dim 8 --heatmap
```

## 新目录说明

- `vessel_seg/pipeline/`：新的五阶段工程主线
- `vessel_seg/reconstruction_3d2d/`：术中重建研究层，承接 clean tree 后的状态建模、C 臂几何和 synthetic projection
- `scripts/pipeline/`：新流水线入口
- `scripts/visualization/`：集中管理绘图与对比脚本
- `data/external/`：外部数据路径登记
- `outputs_reorganized/`：新主线输出
- `scripts/`：旧脚本兼容层，当前保留范围见 `scripts/README.md`
- `outputs/`：历史输出，只读
- `archive/`：已经退出主线但需要保留追溯性的历史文件

## 当前 stage3 产物

`03_centerline_repair` 当前除了 `tree.json` 之外，还会输出：

- `branch_names.json`：跨病例稳定的 canonical branch ids
- `semantic_topology.json`：基于预设冠脉拓扑和几何规则的语义拓扑结果
- 可选 `semantic prior`：从多病例拟合得到的 trunk 几何分布先验，可在批跑时通过 `--semantic-prior` 重新应用

其中：

- canonical naming 解决“稳定编号”
- semantic topology 解决“模板一致性”和主干/侧支语义映射
- semantic prior 解决 `LAD / LCx / RCA` 等 trunk 家族不再只靠启发式排序
- `analyze_canonical_consistency.py` 可量化 `SYS_A / SYS_A.01 / SYS_B ...` 在跨病例下的几何稳定性
- `analyze_branch_clusters.py` 可在不预设 label 的情况下，对所有分支做全局聚类并输出 cluster 原型与异常样本
- `plot_branch_cluster_galleries.py` 可把每个 cluster 在各病例树上的位置单独高亮成大图 gallery，默认按 `Normal -> Diseased` 和病例编号固定格子排版，并默认对齐到病例内解剖坐标系后再绘图
- `plot_branch_side_galleries.py` 可直接把 `LCA / RCA` 两类在各病例树上的位置高亮成两张大图，便于先检查左右分发是否稳定；默认视角与 cluster gallery 保持一致，也可通过 `--elev/--azim` 显式锁定
- `analyze_branch_clusters.py --match-max-case-branches` 可把 cluster 数压缩到全体病例中的最大血管段数
- `analyze_branch_clusters.py --side-assignments-csv <csv>` 可直接使用外部 `case_id, branch_id, side_group` 结果作为左右分组来源，再在 `LCA/RCA` 内重跑聚类
- `project_case_synthetic_xray.py` 可把已有 `tree.json + centerline_repaired.vtp` 放到世界系并投影成一份 synthetic X-ray 观测，作为 3D-2D 重建研究的最小闭环
- `project_case_synthetic_xray_sweep.py` 可批量生成多角度 synthetic X-ray 视图，并同时输出中心线总览和带血管宽度的血管壁总览
- `docs/branch_clustering_lr_baseline.md` 记录了当前“左右硬约束 + 走形软约束 + 固定格子布局”的解释和推荐命令
- `docs/branch_clustering_side_first_contract.md` 记录了当前聚类分析的输入、输出和 side-first 工作原理

## 归档说明

仓库根目录中的历史演示脚本、一次性 notebook、示例图片、汇报材料脚本，正在逐步迁入：

```bash
archive/
```

归档内容默认不再维护，也不作为当前工程入口。

## Agent 协作

- 统一仓库协作规范：`AGENTS.md`
- 重建计划：`docs/rebuild_repo_plan.md`
- 项目总设计文档：`docs/coronary_analysis_master_blueprint.md`

## 训练/建模思路（对应论文）
- 树先验：统计 {parent, λ, 分叉角度}。
- 分支形状先验：将 `radii(K×M)` 张量做降维（PCA/GP/Fourier），得到低维 latent，后续可训练生成/回归模型。

## TotalSeg 量化流水线（Step1~Step5）
- 文档：`docs/totalseg_quantitative_pipeline.md`
- Slicer 实操：`docs/totalseg_slicer_guide.md`
- 统一入口：`scripts/quant_pipeline.py`
- 第 2 步（TotalSeg mask -> 中心线量化）：
```bash
python scripts/step2_centerline_totalseg.py \
  --totalseg-mask <totalseg_mask.nii.gz> \
  --gt-centerline <gt_centerline.vtp> \
  --out-dir outputs/quant/<case> \
  --backend skeleton \
  --thr 1.0
```
