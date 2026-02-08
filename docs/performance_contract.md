# Performance Contract For Refactor

本文件定义仓库重构期间的硬门槛和验收标准。目标是保证:

1. 效果不退化。
2. 运行时间不变或更优。
3. 资源占用不变或更优。
4. 回归可被自动检测并阻断合并。

## 1. 基线冻结

1. 在开始重构前创建基线标签，例如 `baseline-refactor-2026-02-07`。
2. 固定运行环境版本（Python、vtk、SimpleITK、scikit-image、torch、vmtk）。
3. 固定评测数据清单与命令。
4. 生成基线 benchmark artifact（JSON）并纳入版本管理或制品库。

## 2. 指标体系

### 2.1 效果指标（来自 centerline compare CSV）

默认关注:

1. `pred2gt_mean`（越低越好）
2. `pred2gt_p95`（越低越好）
3. `gt2pred_mean`（越低越好）
4. `coverage_pred@1mm`（越高越好）
5. `coverage_gt@1mm`（越高越好）

要求:

1. 聚合指标必须满足契约容差。
2. 每病例指标必须满足契约容差（建议开启 `per_case_rules`）。
3. 候选结果的病例集合必须与基线一致。

### 2.2 性能指标

1. `wall_time_sec`（越低越好）
2. `peak_rss_mb`（越低越好，best effort）

要求:

1. 默认至少比较 wall-time。
2. 内存指标允许按平台设置为可选（`required: false`）。

## 3. 容差规则

每个规则使用:

1. `direction`: `lower` 或 `higher`
2. `rel_tol`: 相对容差
3. `abs_tol`: 绝对容差

判定:

1. `lower`: `candidate <= baseline + abs_tol + rel_tol * |baseline|`
2. `higher`: `candidate >= baseline - abs_tol - rel_tol * |baseline|`

## 4. 阻断策略

1. 任一必需规则失败即判定为回归，阻断合并。
2. 仅在明确记录豁免理由时，允许临时放宽容差。
3. 放宽必须有时限，到期回收。

## 5. 执行命令

### 5.1 采集基线

```bash
python scripts/perf/collect_benchmark.py \
  --name baseline_normal20 \
  --centerline-csv outputs/vmtk_centerlines/compare_summary.csv \
  --output outputs/benchmark/baseline_normal20.json
```

可选: 采集命令耗时与峰值内存

```bash
python scripts/perf/collect_benchmark.py \
  --name baseline_pipeline_run \
  --cmd "python scripts/run_pipeline.py --pattern ASOCA2020/Normal/Centerlines/Normal_*.vtp" \
  --repeat 3 \
  --output outputs/benchmark/baseline_pipeline_run.json
```

### 5.2 采集候选版本

```bash
python scripts/perf/collect_benchmark.py \
  --name candidate_normal20 \
  --centerline-csv outputs/vmtk_centerlines/compare_summary.csv \
  --output outputs/benchmark/candidate_normal20.json
```

### 5.3 比较并守门

```bash
python scripts/perf/compare_benchmark.py \
  --baseline outputs/benchmark/baseline_normal20.json \
  --candidate outputs/benchmark/candidate_normal20.json \
  --contract configs/perf_contract.example.json \
  --output outputs/benchmark/compare_normal20.json
```

失败时脚本返回非 0，适合直接接入 CI。

## 6. 建议流程

1. PR 阶段跑快速集（少量病例）。
2. 合并前跑全量集（20 例或完整评测集）。
3. 每个 PR 必须附 benchmark 对比结果。
4. 若回归，先回滚或修复，再进入下一轮重构。
