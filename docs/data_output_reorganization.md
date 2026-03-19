# 数据与输出整理建议

这份文档用于整理当前仓库里的两类资产：

- `ASOCA2020/` 及后续扩展数据集：原始数据与中间派生数据
- `outputs/`：算法结果、评估产物、实验版本、汇报图件

当前问题不是“数据缺失”，而是“层级和语义混放”。

## 1. 当前观察到的主要问题

### 1.1 原始数据整体还算干净

当前 `ASOCA2020/` 已经按数据集和模态分层：

- `ASOCA2020/Normal/CTCA_nii`
- `ASOCA2020/Normal/Annotations_nii`
- `ASOCA2020/Normal/Centerlines`
- `ASOCA2020/Diseased/...`

这部分可以继续保留为 legacy 原始数据入口。

### 1.2 outputs 混入了四种不同语义

当前 `outputs/` 同时包含：

- 单病例正式结果，如 `outputs/Normal_1/`
- 多病例聚合分析，如 `outputs/aggregate/`
- 拓扑数据集与统计，如 `outputs/topology_dataset*`
- 单次临时文件，如 `outputs/normal1_tree.json`、`outputs/normal1_*.png`

### 1.3 单病例目录内部存在多参数版本并列

以 `outputs/Normal_1/` 为例，里面同时出现：

- `branches/`
- `branches_pct2p5/`
- `branches_official_trim15/`
- `prior_coronary_k64_caps_ext/`
- `similarity_pct5p0/`
- `cpr/`

这会导致两个问题：

- 同一类结果没有统一归档层
- 无法快速判断哪个目录是正式版、哪个是实验版

### 1.4 quant 输出夹杂正式运行和 debug 运行

`outputs/quant/` 下面既有：

- `Normal_1/`
- `Normal_2/`

也有：

- `Normal_1_debug2/`
- `Normal_1_graph/`
- `Normal_1_connectivity_test/`
- `Normal_1_from_openprofiles/`

这类结构适合拆成“正式 case 结果”和“实验 run 归档”。

## 2. 推荐的目标结构

建议以后把仓库中的数据与结果统一成下面两层：

```text
data/
  raw/
    asoca2020/
      Normal/
        ctca_nii/
        annotations_nii/
        centerlines_vtp/
        surface_meshes/
      Diseased/
        ...
  interim/
    asoca2020/
      <case>/
        extracted_centerline/
        aligned_mask/
        repaired_centerline/
  external/
    ...

outputs/
  cases/
    <case>/
      tree/
      branches/
        default/
        pct2p5/
        trim_overlap0/
        official_trim15/
      datasets/
        default/
        pct2p5/
      priors/
        k64/
        k64_caps/
        k64_caps_ext/
      similarity/
        default/
        pct2p5/
      cpr/
      topology/
      vis/
      manifest.json
  aggregate/
    branch_shape/
    topology/
    benchmark/
  quant/
    official/
      <case>/
        step1_segmentation/
        step2_centerline/
        step3_repair/
        step4_features/
        step5_render/
    experiments/
      <run_name>/
  reports/
    teacher/
    ppt_assets/
  archive/
    legacy_outputs_root/
```

## 3. 命名规范

### 3.1 case 目录

统一使用：

- `Normal_1`
- `Normal_2`
- `Diseased_1`

不要再混用：

- `normal1_*`
- `shape_diseased_1`
- `branches_normal1`

### 3.2 变体目录

统一把参数信息压到“变体名”里，不要直接散在一级目录名里。

推荐写法：

- `default`
- `pct2p5`
- `pct5p0`
- `trim_overlap0`
- `official_trim15`
- `aligned`
- `k64_caps_ext`

例如：

- `outputs/cases/Normal_1/branches/pct2p5/`
- `outputs/cases/Normal_1/priors/k64_caps_ext/`

### 3.3 manifest

每个病例目录建议保留一个 `manifest.json`，记录：

- case id
- 数据来源
- 生成时间
- 主要输入路径
- 正式版本目录
- 实验版本目录

这样后面做批量分析或迁移时，不需要靠人脑记忆目录含义。

## 4. 当前目录到目标目录的映射建议

### 4.1 原始数据

- `ASOCA2020/Normal/CTCA_nii` -> `data/raw/asoca2020/Normal/ctca_nii`
- `ASOCA2020/Normal/Annotations_nii` -> `data/raw/asoca2020/Normal/annotations_nii`
- `ASOCA2020/Normal/Centerlines` -> `data/raw/asoca2020/Normal/centerlines_vtp`

短期内不必立即移动；先在文档和代码中把它们视为 legacy raw source。

### 4.2 单病例结果

- `outputs/Normal_*` -> `outputs/cases/Normal_*/`
- `outputs/Diseased_*` -> `outputs/cases/Diseased_*/`

### 4.3 聚合结果

- `outputs/aggregate` 保留，但建议细分为：
  - `outputs/aggregate/branch_shape`
  - `outputs/aggregate/topology`
  - `outputs/aggregate/benchmark`

### 4.4 topology

- `outputs/topology_dataset`
- `outputs/topology_dataset_all_trim_overlap0_tree`

建议统一迁到：

- `outputs/aggregate/topology/...`

### 4.5 quant

建议拆分：

- 正式：`outputs/quant/official/<case>/...`
- 调试：`outputs/quant/experiments/<run_name>/...`

例如：

- `outputs/quant/Normal_1` -> `outputs/quant/official/Normal_1`
- `outputs/quant/Normal_1_debug2` -> `outputs/quant/experiments/Normal_1_debug2`

### 4.6 outputs 根目录下的散文件

这些文件不应该长期直接留在 `outputs/` 根目录：

- `outputs/normal1_tree.json`
- `outputs/normal1_tree_plot.png`
- `outputs/normal1_centerline_vs_gt.png`
- `outputs/shape_diseased_1.vtp`

建议迁到：

- `outputs/cases/Normal_1/vis/legacy/`
- `outputs/cases/Normal_1/tree/legacy/`
- `outputs/reports/...`

## 5. 建议的整理顺序

推荐按下面顺序执行，风险最低：

1. 先生成现状清单，不移动数据
2. 固化目标结构与命名规则
3. 只迁移 `outputs/` 根目录散文件
4. 再迁移 `outputs/quant/` 的 debug run
5. 最后迁移 `outputs/Normal_*` 里的实验性子目录

## 6. 不建议现在就做的事情

- 不建议直接大规模改 `ASOCA2020/` 原始目录
- 不建议一次性重命名所有 outputs 子目录
- 不建议删除旧目录而不留索引或迁移记录

## 7. 配套工具

配套脚本：

- `scripts/audit_data_outputs.py`

用途：

- 扫描 `ASOCA2020/` 和 `outputs/`
- 统计当前目录结构、大小、case/aggregate/quant/debug 分布
- 输出 `json` 和 `markdown` 清单，作为后续迁移基线
