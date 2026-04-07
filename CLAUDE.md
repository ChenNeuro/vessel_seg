# CLAUDE.md

先读 `AGENTS.md`，它是仓库唯一的主规范。

Claude 适配要求：

- 探索阶段保持高信息密度，先说结论，再说证据。
- 讨论架构时，优先锚定：
  - `docs/coronary_analysis_master_blueprint.md`
  - `docs/data_output_reorganization.md`
  - `docs/rebuild_repo_plan.md`
- 如果任务涉及路径、输出、schema、归档，必须明确说明：
  - 是否影响旧路径兼容
  - 是否写入 `outputs_reorganized/`
  - 是否更新了文档
  - 是否跑了当前基线测试

默认心智模型：

1. 干净中心线树
2. 稳定拓扑表示
3. 五阶段流水线
4. 可复用可视化与报告

不要再引入并行架构，也不要把归档内容重新扶正为主入口。
