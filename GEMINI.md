# GEMINI.md

先读 `AGENTS.md`。

Gemini 适配要求：

- 先整理和标准化，再扩展功能。
- 优先使用明确 schema、manifest、结构化输出。
- 不要在 `outputs/` 下再造新的历史风格路径。
- 外部原始数据 `../ASOCA2020` 只读。
- 历史 notebook、历史文档、历史脚本继续放在归档区，不要回流。

写代码时：

- 稳定逻辑优先进入 `vessel_seg/`
- 主线入口优先进入 `vessel_seg/pipeline/` 或 `scripts/pipeline/`
- 绘图脚本集中在 `scripts/visualization/`
- 只要涉及目录整理和归档，结束后要跑当前基线测试
