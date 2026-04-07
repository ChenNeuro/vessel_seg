# GitHub Performance Gate Setup

本说明对应工作流: `.github/workflows/perf-gate.yml`。

## 1. 工作流结构

1. `Perf Tooling Smoke`  
在 `ubuntu-latest` 上运行，验证对比脚本可执行（baseline vs baseline）。

2. `Perf Contract Gate`  
在 `self-hosted` 上运行，读取真实 `compare_summary.csv`，执行性能契约对比并阻断回归。

触发方式:

1. 手动触发 `workflow_dispatch`。
2. PR 上加标签 `run-perf-gate`。

## 2. 为什么需要 self-hosted

真实 gate 依赖你本地数据与产物路径（例如 `outputs/vmtk_centerlines/compare_summary.csv`），
这些通常不在 GitHub 托管 runner 上。

## 3. 推荐配置

在 GitHub 仓库 Settings -> Variables -> Actions 配置:

1. `PERF_CANDIDATE_CSV`  
例如: `outputs/vmtk_centerlines/compare_summary.csv`
2. `PERF_BASELINE_JSON`  
例如: `benchmarks/baseline_normal20.json`
3. `PERF_CONTRACT_JSON`  
例如: `configs/perf_contract.example.json`

如果你用 `workflow_dispatch` 手动输入了路径，输入值优先。

## 4. baseline 更新流程

当你确认某次结果应成为新基线:

```bash
python scripts/perf/collect_benchmark.py \
  --name baseline_normal20 \
  --centerline-csv outputs/vmtk_centerlines/compare_summary.csv \
  --output benchmarks/baseline_normal20.json
```

然后提交 `benchmarks/baseline_normal20.json`。

## 5. 回归判定

`scripts/perf/compare_benchmark.py` 失败时返回非 0，GitHub Action 会标红并阻断。
详细失败项在 `artifacts/compare_report.json`。

## 6. 首次使用建议

1. 先手动运行一次 `workflow_dispatch`，确认 self-hosted runner 路径可访问。
2. 再在 PR 上加 `run-perf-gate` 标签验证自动触发。
