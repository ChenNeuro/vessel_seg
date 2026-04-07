# GitHub Copilot Instructions

以仓库根目录的 `AGENTS.md` 作为最高优先级规范。

仓库特定要求：

- 这是冠脉分析仓库，不是单纯分割仓库。
- 稳定逻辑优先进入 `vessel_seg/`。
- 五阶段主线优先进入 `vessel_seg/pipeline/` 和 `scripts/pipeline/`。
- 绘图与对比脚本集中在 `scripts/visualization/`。
- 新标准化输出使用 `outputs_reorganized/`。
- 不允许无迁移方案地破坏 `outputs/` 旧路径。
- `../ASOCA2020` 视为只读原始输入。
- 历史 notebook、文档、演示脚本应继续保留在 `archive/` 或 `docs/archive/`。

开始较大重构前，优先检查：

- `README.md`
- `docs/rebuild_repo_plan.md`
- `docs/coronary_analysis_master_blueprint.md`
- `docs/data_output_reorganization.md`

当前推荐的开发主题：

1. 干净中心线树
2. 拓扑规范化
3. 分支形状先验
4. SCCT18 标注
5. CPR / 临床可视化
6. 评估契约与可复现性
7. 仓库去杂物化与归档
