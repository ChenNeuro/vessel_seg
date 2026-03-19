# vessel_seg 暑研阶段汇报（10-15 分钟）

> 汇报对象：导师（University of Michigan）  
> 汇报时间：2026-02-10  
> 目标：展示一个“可量化、可复现、可迭代”的冠脉分析闭环，并说明当前问题与下一步计划。

## 1. 本阶段目标与达成情况

本阶段的核心目标是把流程打通为 5 个可量化步骤：

1. CT -> 分割（TotalSegmentator）
2. 分割 -> 中心线提取（统一输出为有 `lines` 的高密度 polyline）
3. 中心线修复（几何与拓扑约束）
4. 血管特征提取（分支级描述）
5. 渲染/重建并与 GT 对比

当前状态：`Normal_1`、`Normal_2` 已完成全流程并产出量化结果；`Normal_1` 的 repair 离散问题已定位并修复（见第 6 节）。

## 2. Pipeline 总览（每步可量化）

![Pipeline overview](assets/teacher_report_20260209/pipeline_overview.png)

量化口径：

1. Step1 分割：`Dice`、`ASD(mm)`、`HD95(mm)`
2. Step2/3 中心线：`pred2gt_mean(mm)`、`pred2gt_p95(mm)`、`coverage_pred@1mm`、`coverage_gt@1mm`
3. Step4 特征：`branch_count_abs_diff`、`descriptor_cosine`、`descriptor_l1/l2`
4. Step5 重建：重建点到 GT 边界距离分布与覆盖率（同 Step2/3 口径）

## 3. 核心结果总表（当前可复现结果）

| Case | Step1 Dice | Step1 HD95 (mm) | Step2 p95 (mm) | Step3 p95 (mm) | Step4 branch diff | Step4 cosine | Step5 p95 (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal_1 | 0.8450 | 0.6250 | 2.0229 | 0.5932 | 1 | 0.9678 | 0.6999 |
| Normal_2 | 0.7274 | 10.2556 | 0.6650 | 0.6282 | 0 | 0.9583 | 0.7538 |

![Cross-case summary](assets/teacher_report_20260209/cross_case_summary.png)

关键观察：

1. `Normal_1`：Step3 明显降低中心线误差尾部（p95 从 2.02 -> 0.59 mm），但 Step4 仍有“多一段分支”（branch diff=1）。
2. `Normal_2`：Step1 分割（HD95=10.26 mm）是主要瓶颈，下游质量受其影响明显。
3. Step5 两例都达到较高 `coverage_pred@1mm`（约 0.986），但 `coverage_gt@1mm` 仍有提升空间。

## 4. Step1 分割结果（Pred vs GT）

### Normal_1

![Normal_1 Step1 overlay](assets/teacher_report_20260209/normal_1_step1_overlay.png)

- Dice = 0.8450
- HD95 = 0.625 mm
- 结论：分割质量较稳定，适合支撑后续中心线与特征分析。

### Normal_2

![Normal_2 Step1 overlay](assets/teacher_report_20260209/normal_2_step1_overlay.png)

- Dice = 0.7274
- HD95 = 10.2556 mm
- 结论：边界误差偏大，是后续拓扑不稳定的重要来源。

## 5. Step2-4 中心线与特征结果

### Step3 中心线结果（含修复后）

#### Normal_1

![Normal_1 Step3 centerline](assets/teacher_report_20260209/normal_1_step3_centerline.png)

#### Normal_2

![Normal_2 Step3 centerline](assets/teacher_report_20260209/normal_2_step3_centerline.png)

### Step4 分支特征对比

#### Normal_1

![Normal_1 Step4 branch length](assets/teacher_report_20260209/normal_1_step4_branch_length.png)

#### Normal_2

![Normal_2 Step4 branch length](assets/teacher_report_20260209/normal_2_step4_branch_length.png)

## 6. 关键问题修复：为什么 repaired 中心线之前看起来离散？

### 6.1 根因

之前的 repair 输出虽然把端点坐标“拉到一起”，但写 VTP 时每条 polyline 仍各自新建点，导致：

1. 空间上看起来靠近
2. 拓扑上并没有共享节点
3. 可视化和连通性分析会显示成离散碎段

### 6.2 修复动作

在 `scripts/repair_centerline.py` 中对输出阶段做了拓扑修复：

1. 写 VTP 时增加点合并（`--merge_point_tol`，默认 `1e-6 mm`）
2. 共享重合端点的点 ID，而不是重复建点
3. 在 `repair_report.json` 输出 `output_topology`（points/lines/components）

### 6.3 修复结果

![Normal_1 connectivity before/after](assets/teacher_report_20260209/normal1_connectivity_before_after.png)

- 旧版（legacy）：`1586 points / 113 lines / 60 components`
- 当前：`1050 points / 8 lines / 2 components`

说明：`2 components` 对应左右冠脉两棵主树，是符合解剖的结果，不再是无意义碎片化离散段。

## 7. 当前限制与风险

1. Step4 在 `Normal_1` 仍出现一段冗余分支（`branch_count_abs_diff=1`）。
2. Step1 在不同 case 的稳定性仍不足（`Normal_2` 的 HD95 偏大）。
3. Step3 虽解决了离散拓扑问题，但“自动补线能力”仍需增强（当前以连接/平滑为主）。

## 8. 下一阶段计划（两周）

1. Week 1：强化 Step3 补线策略
- 目标：从“端点连接+平滑”升级为“结构化缺失段补全”。
- 方法：加入基于局部方向场与半径先验的候选路径生成，并对每条补线做可解释评分。

2. Week 2：稳定 Step4 分支治理
- 目标：消除“多一段分支”的系统性误差。
- 方法：统一分支裁剪规则（最小长度、端点合法性、与主干夹角约束），并加入回归测试。

3. 同步建设评估基准
- 扩展到 10-20 例批量评估，输出统计显著性与失败案例分层报告。

## 9. 希望导师给出的反馈

1. 当前指标优先级是否合理（Step1 稳定性优先于 Step3/4 精细优化）？
2. 论文方向更建议强调“方法创新”还是“工程闭环+可解释评估”？
3. 是否认可先用 `Normal_1/2` 小样本闭环，再扩展到批量统计的推进顺序？

---

## 附录：复现命令（Normal_1 repair 检查）

```bash
python scripts/repair_centerline.py \
  --prob outputs/quant/Normal_1/step1_segmentation/totalseg_output/coronary_arteries.nii.gz \
  --vtp outputs/quant/Normal_1/step2_centerline/pred_centerline_poly.vtp \
  --out outputs/quant/Normal_1/step3_repair/repaired.vtp \
  --report outputs/quant/Normal_1/step3_repair/repair_report.json

python scripts/quant_pipeline.py step3 \
  --baseline-centerline outputs/quant/Normal_1/step2_centerline/pred_centerline_poly.vtp \
  --repaired-centerline outputs/quant/Normal_1/step3_repair/repaired.vtp \
  --gt-centerline ASOCA2020/Normal/Centerlines/Normal_1.vtp \
  --out-dir outputs/quant/Normal_1 \
  --thr 1.0
```
