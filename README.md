Vessel segmentation toolkit

- Conda environments
  - Lightweight preprocessing env: `conda create -n vessel_seg -c conda-forge python=3.10 simpleitk numpy scipy scikit-image nibabel`.
  - Full ASOCA training env: `conda env create -f env/asoca_nnunet.yaml` then `pip install -e third_party/nnUNet`.
- Conversion utility
  - Code lives in `vessel_seg/conversion.py` and exposes `convert_nrrd_to_nii`.
  - Run from the command line: `python -m vessel_seg.conversion <nrrd_dir> <output_dir>`.
  - Alternatively, import the function and call it directly as shown in the inline example comments.
- Metadata normalisation
  - `vessel_seg/metadata.py` defines `SegmentationMetadata` and helpers to harmonise label names into PascalCase.
  - `vessel_seg/metadata_schema.json` captures the JSON schema used to keep segmentation metadata consistent across cases.
- Leaderboard + literature
  - `python scripts/fetch_asoca_leaderboard.py --limit 10` captures the current challenge standings into `data/asoca_leaderboard_top10.json`.
  - The highest public Dice (0.8946) is achieved by user `hongqq` (submission 2024‑03‑12); no code release is available, so `nnUNetv2` serves as the best open baseline (5th place `junma`).
- Training & inference (nnUNetv2)
  - Prepare dataset with `nnUNetv2_plan_and_preprocess -d 103 -c 3d_fullres`, then launch `nnUNetv2_train 103 3d_fullres all`.
  - Run predictions via `nnUNetv2_predict -d 103 -c 3d_fullres -i <CTA_dir> -o outputs/asoca_predictions`.
- Evaluation
  - Quick comparison script: `python scripts/compare_asoca.py --gt <gt_dir> --pred outputs/asoca_predictions --output outputs/asoca_metrics.json`.
  - For native tooling use `nnUNetv2_evaluate_folder` to compute Dice/HD95/ASSD.
- Shape-model roadmap
  - Detailed workflow lives in `docs/asoca_pipeline_plan.md` (branch decomposition, polar descriptors, 3D reconstruction union).
  - Near-term actions: validate TotalSegmentator online for coronary visibility, kick off nnUNetv2 training, and prototype centreline-based descriptors.
- Dimensionality / reconstruction toolkit
  - `python -m vessel_seg.shape extract --seg <mask.nii.gz> --out features/<case_id>` 提取中心线、极坐标截面与统计描述。
  - `python -m vessel_seg.shape reconstruct --features features/<case_id> --output outputs/<case_id>.vtp` 将分支曲线扫掠为三维血管网格。
  - 方法细节与实现提示参见 `docs/vessel_dimensionality_workflow.md`.

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
