# Refactor Execution Plan (Performance-Gated)

## 目标

在不破坏现有效果与性能的前提下，完成架构重构和工程化升级。

## Phase 0: Guardrail Setup

1. 固化 baseline tag 和环境版本。
2. 引入 benchmark 采集与对比脚本。
3. 固化契约配置（`configs/perf_contract*.json`）。
4. 在 CI 中接入失败阻断。

交付物:

1. 基线 artifact（JSON）
2. compare report 模板
3. CI gate（非 0 退出阻断）

## Phase 1: Correctness Before Refactor

先修高风险 correctness 问题，再进入结构重构:

1. 入口脚本可执行性（`--help` 不崩）。
2. 中心线导出有效性（点/线拓扑一致）。
3. 指标实现正确性（避免伪 HD95 等）。
4. 明确 fallback 行为语义，避免“看起来成功，实际不可用”。

交付物:

1. Bugfix PR（仅 correctness）
2. 回归 benchmark 报告（必须不退化）

## Phase 2: Structural Refactor

1. 将 `scripts/` 变为 CLI 薄层。
2. 核心逻辑统一迁移到 `vessel_seg/` 包。
3. 统一 I/O schema 与异常语义。
4. 保持 CLI 参数向后兼容。

交付物:

1. 模块化后的 package API
2. 命令兼容性清单
3. benchmark 对比报告

## Phase 3: Performance Optimization

1. 仅做可证明收益的优化（向量化、减少重复 I/O、缓存）。
2. 优化必须伴随前后 benchmark。
3. 禁止在同一 PR 混入算法变更。

交付物:

1. 优化项列表（收益/风险）
2. 对比报告（效果与性能）

## Phase 4: Optional Algorithm Improvements

1. 算法增强使用 feature flag（默认关闭）。
2. 先 shadow-run，再切默认。
3. 任何退化立即回滚。

交付物:

1. Feature flag 设计与默认值说明
2. A/B 报告

## PR 拆分规则

1. 每个 PR 只做一类变化:
2. `type=correctness`
3. `type=refactor`
4. `type=performance`
5. `type=algorithm`
6. 每个 PR 必附:
7. benchmark artifact 路径
8. compare report 路径
9. 风险点与回滚点

## 里程碑退出条件

1. 所有 gate 通过。
2. 全量数据集指标不退化。
3. 关键命令保持可用。
4. 文档与脚本一致。
