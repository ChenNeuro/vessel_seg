# Branch Clustering LR Baseline

这份说明文档用于固定当前分支聚类的诊断基线，重点是先把“左右系统分发”看清楚，再讨论走形细节。

## 当前基线

当前基线的核心约束是：

- 左右冠系统硬约束：`LCA / RCA` 不允许混到同一个 cluster
- 同病例同父节点兄弟分支不允许被吞到同一个 cluster
- root 和非 root 不允许混到同一个 cluster
- 走形方向仍然参与相似度，但只作为软约束，不再一票否决

对应代码：

- `vessel_seg/pipeline/canonical_consistency.py`
- `vessel_seg/pipeline/branch_clustering.py`
- `vessel_seg/pipeline/branch_cluster_gallery.py`

## 固定排版规则

为了稳定检查左右枝分发问题，cluster gallery 现在默认采用固定格子布局：

- 先按 `cohort` 排序：`Normal -> Diseased`
- 同一 `cohort` 内按病例编号升序
- 每张 cluster 图都使用同一套 case 位置

这样做的目的不是为了更好看，而是为了：

- 不让“命中病例排前面”的动态排序掩盖左右系统分发错误
- 让同一个 case 在所有 cluster 图里都出现在固定位置
- 便于人工比较 `SYS_A / SYS_B` 在 40 例中的分布

此外，gallery 现在默认会先把每个病例旋转到病例内解剖坐标系：

- x 轴：`left_right_axis`
- y 轴：`longitudinal_axis`
- z 轴：`normal_axis`

这样不同病例之间的“横向 / 纵向 / 垂直”才是可比的，不会被原始世界坐标方向误导。

如果仍然需要旧的动态排序，可在脚本里显式加：

```bash
--dynamic-order
```

如果仍然需要旧的原始世界坐标，可在脚本里显式加：

```bash
--raw-world-coords
```

## 推荐命令

先重跑聚类：

```bash
python scripts/pipeline/analyze_branch_clusters.py \
  --summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/summary.csv \
  --output-dir outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/branch_clustering_lr_hard_baseline_max_case_branches_17 \
  --match-max-case-branches
```

再生成固定位置 gallery：

```bash
python scripts/visualization/plot_branch_cluster_galleries.py \
  --summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/summary.csv \
  --assignments-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/branch_clustering_lr_hard_baseline_max_case_branches_17/branch_cluster_assignments.csv \
  --cluster-summary-csv outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/branch_clustering_lr_hard_baseline_max_case_branches_17/branch_cluster_summary.csv \
  --output-dir outputs_reorganized_runs/analysis/overview_batches/asoca_semantic_prior_40/branch_clustering_lr_hard_baseline_max_case_branches_17/branch_cluster_galleries
```

## 测试案例

当前仓库里保留了两个直接相关的测试：

- `tests/test_branch_clustering.py::test_analyze_branch_clusters_keeps_left_right_systems_separate`
- `tests/test_branch_cluster_gallery.py::test_ordered_case_ids_for_gallery_uses_fixed_cohort_and_number_grid`

第一个测试保证左右系统不会混簇，第二个测试保证 gallery 的病例位置固定。
