---
marp: true
theme: default
paginate: true
size: 16:9
headingDivider: 2
title: vessel_seg Summer Research Progress
---

# vessel_seg 暑研进展汇报

**CT -> Segmentation -> Centerline -> Repair -> Feature -> Reconstruction**

- 汇报对象：暑研导师（University of Michigan）
- 汇报时长：10-15 分钟
- 汇报人：Chen Yihao
- 日期：2026-02-10

---

## 1. 目标与当前状态

**目标**：建立一个“每一步都可量化”的冠脉分析闭环。  

**五步流程**：

1. Step1：TotalSegmentator 分割（CT -> mask）
2. Step2：中心线提取（统一为有 `lines` 的高密度 polyline）
3. Step3：中心线修复（几何 + 拓扑）
4. Step4：血管特征提取（分支级）
5. Step5：重建并与 GT 对比

**当前状态**：`Normal_1`、`Normal_2` 全流程可复现，且有完整指标输出。

---

## 2. Pipeline 架构（可复现 + 可量化）

![w:1050](assets/teacher_report_20260209/pipeline_overview.png)

---

## 3. 指标体系（每一步都有量化口径）

1. Step1 分割：`Dice`、`ASD(mm)`、`HD95(mm)`
2. Step2/3 中心线：`pred2gt_mean(mm)`、`pred2gt_p95(mm)`、`coverage_pred@1mm`、`coverage_gt@1mm`
3. Step4 特征：`branch_count_abs_diff`、`descriptor_cosine`、`descriptor_l1/l2`
4. Step5 重建：重建点到 GT 边界距离（同 Step2/3 口径）

**解释**：

- 距离类指标越小越好
- 覆盖率/余弦相似度越大越好

---

## 4. 核心结果总览

| Case | Step1 Dice | Step1 HD95 (mm) | Step2 p95 (mm) | Step3 p95 (mm) | Step4 branch diff | Step4 cosine | Step5 p95 (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal_1 | 0.8450 | 0.6250 | 2.0229 | 0.5932 | 1 | 0.9678 | 0.6999 |
| Normal_2 | 0.7274 | 10.2556 | 0.6650 | 0.6282 | 0 | 0.9583 | 0.7538 |

![w:820](assets/teacher_report_20260209/cross_case_summary.png)

---

## 5. Step1 分割对比（Pred vs GT）

![bg right:50% w:760](assets/teacher_report_20260209/normal_1_step1_overlay.png)

**Normal_1**

- Dice = 0.8450
- HD95 = 0.625 mm
- 结论：分割较稳定，可支撑下游分析

---

## 6. Step1 分割对比（困难样例）

![bg right:50% w:760](assets/teacher_report_20260209/normal_2_step1_overlay.png)

**Normal_2**

- Dice = 0.7274
- HD95 = 10.2556 mm
- 结论：Step1 是该例的主要瓶颈，误差会向下游传导

---

## 7. Step2/3：中心线提取与修复

**可视化结果**

- 左：Normal_1
- 右：Normal_2

![w:520](assets/teacher_report_20260209/normal_1_step3_centerline.png)
![w:520](assets/teacher_report_20260209/normal_2_step3_centerline.png)

**观察**：Step3 能明显改善尾部误差与连通性稳定性。

---

## 8. 关键技术问题与修复（体现工程能力）

**问题**：`repaired` 中心线视觉上“像连着”，拓扑上却是离散碎段。  

**根因**：写 VTP 时每条 polyline 独立建点，没有共享重合端点 ID。  

**修复动作**：

1. 在 `repair_centerline.py` 引入端点合并（`--merge_point_tol`）
2. 输出 `output_topology`（points/lines/components）用于持续监控
3. 回归验证：修复前后连通性 + 误差指标同时检查

![w:900](assets/teacher_report_20260209/normal1_connectivity_before_after.png)

---

## 9. Step4 特征提取结果

![w:520](assets/teacher_report_20260209/normal_1_step4_branch_length.png)
![w:520](assets/teacher_report_20260209/normal_2_step4_branch_length.png)

结论：

1. `Normal_2` 分支数与 GT 对齐（diff=0）
2. `Normal_1` 仍有一段冗余分支（diff=1），是下一步重点

---

## 10. 我的核心贡献（给导师看“你能独立推进”）

1. 把单点算法串成了 5-step 可量化 pipeline（端到端可复现）
2. 建立了统一评估口径（分割、中心线、特征、重建）
3. 定位并修复了拓扑层面的关键 bug（不是只调参数）
4. 把失败案例转成可回归测试的问题清单（工程化推进）

**能力关键词**：问题定位能力、系统化思维、可复现工程实现、结果解释能力。

---

## 11. 主要风险与下一步（两周）

1. Step3：从“连接+平滑”升级到“结构化补线”
2. Step4：消除冗余分支，稳定分支拓扑治理策略
3. 批量评估：扩展到 10-20 例，给出统计显著性和失败分层

**里程碑交付**：

- Week 1：Step3 补线策略 + 回归集
- Week 2：Step4 分支治理 + 多例统计报告

---

## 12. 希望导师反馈的问题

1. 指标优先级是否应继续以 Step1 稳定性为第一优先？
2. 论文方向更强调方法创新，还是工程闭环 + 可解释性？
3. 是否认可先做小样本闭环，再扩展至多中心批量验证？

---

## Backup: 复现命令（可选）

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
