# Branch Clustering Side-First Contract

这份文档定义当前聚类分析的输入、输出和工作原理。

## 目标

聚类分析不再从“全分支混合”直接开始，而是先判断左右系统，再在各系统内部做聚类。

当前默认模式：

- `clustering_mode = side_first`

## 输入

主输入：

- `summary.csv`

这个文件来自 batch overview，最少需要：

- `case_id`
- `cohort`
- `overview_path`

聚类时还会自动读取每个 case 的：

- `stages/03_centerline_repair/tree.json`
- `stages/03_centerline_repair/centerline_repaired.vtp`

因此，真正的聚类输入不是单个 csv，而是：

1. batch summary
2. 每例的 repair tree
3. 每例的 repaired centerline polyline

## 样本表示

每个 branch 会被转换成一个 `CanonicalBranchSample`，包含：

- case / cohort
- canonical_name / branch_id
- system_name
- parent_branch_id / sibling_group_key
- depth
- centroid / direction / tangent 投影
- length / subtree / lambda
- tortuosity / bend / centerline_offset
- orientation_axis / orientation_signed_axis / orientation_confidence

## 工作原理

`side_first` 流程：

1. 从 `tree.json` 和 `centerline_repaired.vtp` 提取 branch sample
2. 先按 `system_name` 分组：
   - `LCA`
   - `RCA`
   - `AUX_*`
   - `MAIN / UNKNOWN`
3. 在每个 side group 内单独做 agglomerative clustering
4. 聚类时继续施加结构约束：
   - 同病例同父节点兄弟分支不能进同一类
   - root 和非 root 不混
   - 左右系统不混
5. 输出全局连续编号的 cluster，同时保留 `side_group`

如果已经有更可信的左右分组结果，也可以额外提供一个外部 csv：

- `case_id`
- `branch_id`
- `side_group`

然后通过：

```bash
--side-assignments-csv <csv>
```

把 `LCA / RCA` 的分组来源切换到这个外部文件，再在各自 side group 内重跑聚类。

这意味着：

- “判断左右”优先于“判断形状”
- 走形只在同侧内部竞争
- 左右不会因为几何相似而互相吞并

## 输出

核心输出：

- `branch_clustering_report.json`
- `branch_cluster_summary.csv`
- `branch_cluster_assignments.csv`
- `branch_cluster_outliers.csv`

其中：

- `report.json`
  - 记录 `clustering_mode`
  - 记录 `input_contract`
  - 记录 `workflow`
  - 记录每个 `side_group` 分了多少类
- `cluster_summary.csv`
  - 每个 cluster 一行
  - 包含 `side_group`
- `assignments.csv`
  - 每个 branch 一行
  - 包含 `side_group`

## CLI

```bash
python scripts/pipeline/analyze_branch_clusters.py \
  --summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/summary.csv \
  --clustering-mode side_first \
  --match-max-case-branches
```

如果要强制使用外部左右分组：

```bash
python scripts/pipeline/analyze_branch_clusters.py \
  --summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/summary.csv \
  --clustering-mode side_first \
  --side-assignments-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/branch_clustering_side_first_lr_baseline_max_case_branches_17/branch_cluster_assignments.csv \
  --match-max-case-branches
```

如果需要旧式全局模式：

```bash
--clustering-mode global
```

## 当前建议

老师提出的“先从左右开始聚类分析”是合理的，当前仓库默认应该优先使用：

- `side_first`

而不是：

- `global`

`global` 只适合做对照实验，不适合做现在的主线 baseline。
