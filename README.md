Vessel segmentation toolkit

# 核心思路（当前阶段）
- 从 ASOCA CTA + 冠脉 mask 出发，提取中心线树 → 统计/建模分支几何（后续接入形状先验）。
- 代码结构已模块化：`vessel_seg/centerline.py`（中心线提取，支持骨架占位和 VTP 读取）、`graph_structure.py`（Branch/CoronaryTree）、`branch_model.py`（占位形状模型）、`tree_prior.py`（占位拓扑先验），附简单可视化和实验脚本。
- 实验脚本：  
  - `python -m vessel_seg.experiments.build_centerline_tree --volume <ct.nii.gz> --mask <mask.nii.gz> --output-json <tree.json>` （骨架占位）  
  - `python -m vessel_seg.experiments.build_centerline_tree --volume <ct.nii.gz> --vtp <vmtk_centerline.vtp> --output-json <tree.json>` （读取已有 VMTK 中心线）  
  - `python -m vessel_seg.experiments.analyze_tree_statistics --trees <tree.json ...>` 查看分支长度/度数/半径统计。

# 目前的卡点
- **VMTK 提取**：当前环境缺少可用的 VMTK（`vmtkcenterlines`/`vtkvmtk`），pip/conda (osx-arm64) 未找到可安装包，无法在本地直接运行 VMTK 生成中心线。暂时只能：
  1) 使用数据集自带的 VMTK 中心线 `.vtp`（已支持读取）；或
  2) 使用占位的 `skimage` 骨架提取（分支过多，质量有限）。
- 若需要代码内直接跑 VMTK，请提供可用的 VMTK 环境/镜像（如 x86_64 或容器），再接入 VMTK 命令/API。

# 环境提示
- 轻量预处理：`conda create -n vessel_seg -c conda-forge python=3.10 simpleitk numpy scipy scikit-image nibabel matplotlib vtk`
- ASOCA nnUNet 训练：`conda env create -f env/asoca_nnunet.yaml` 后 `pip install -e third_party/nnUNet`

# 其他已有工具
- NRRD→NIfTI 转换：`python -m vessel_seg.conversion <nrrd_dir> <output_dir>`
- 元数据归一化：`vessel_seg/metadata.py` + `metadata_schema.json`
- 文档：`docs/asoca_pipeline_plan.md`, `docs/fgpm_pipeline.md`, `docs/vessel_dimensionality_workflow.md`
## Probabilistic shape modelling (FGPM)

- 设计细节参见 `docs/fgpm_pipeline.md`。
- 训练 Fourier-Gaussian-process 分支模型：
  ```bash
  python -m vessel_seg.fgpm fit \
    --features-dirs features/case_* \
    --branch Branch_00 \
    --order 6 \
    --output models/branch00_fgpm.json
  ```
  训练样本来自 `vessel_seg.shape extract` 导出的分支 `*.npz` 文件，脚本会自动收集高置信度切片并拟合多项式均值 + RBF 高斯过程。
- 生成随机样本或运行 Bayesian 推理：
  ```bash
  python -m vessel_seg.fgpm sample --model models/branch00_fgpm.json \
    --num-slices 80 --output outputs/branch00_samples.npz

  python -m vessel_seg.fgpm infer \
    --model models/branch00_fgpm.json \
    --branch-features features/test_case/Branch_00.npz \
    --annotations features/test_case_sparse/Branch_00.npz \
    --edge-map outputs/test_case_edges.nii.gz \
    --output outputs/branch00_fgpm_post.npz
  ```
  `infer` 会先根据稀疏注释（可用 `propagate` 将邻近切片自动配准扩增）获得交互后验，再结合深度边缘置信度进行 MAP 估计。
- 注释传播（2D Demons 配准）：
  ```bash
  python -m vessel_seg.fgpm propagate \
    --image volumes/case001.nii.gz \
    --mask annotations/case001_sparse.nii.gz \
    --source 30 --target 31 --axis 2 \
    --output-mask annotations/case001_aug.nii.gz
  ```

## Deep edge detection (DED)

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
