# AGENTS.md

本仓库的规范以“可维护、可继承、可协作、可重建”为最高优先级。

如果其他 agent 适配文件与本文件冲突，以本文件为准。

## 1. 仓库目标

这个仓库正在从研究沙盒重建为一个标准化的冠状动脉工程仓库。

新的主线只有 5 个阶段：

1. `ct_segmentation`
2. `centerline_extraction`
3. `centerline_repair`
4. `wall_features`
5. `rendering`

任何新工作都要明确属于这 5 个阶段中的哪一个。

## 2. 目录原则

### 新主线

- `vessel_seg/pipeline/`: 新的标准化流水线代码
- `scripts/pipeline/`: 新流水线的脚本入口
- `scripts/visualization/`: 仍保留、但不属于工程主线的绘图与对比脚本
- `outputs_reorganized/cases/<case>/stages/`: 新流水线的标准输出

### 旧逻辑

- `scripts/`: 旧脚本和兼容包装层
- `outputs/`: 历史输出，只读，不再扩展
- `notebooks/`: 不再承载主逻辑，历史 notebook 已移入 `archive/notebooks/`
- `docs/archive/`: 历史方案、汇报和阶段性计划文档
- `archive/`: 已退出主线的历史文件，只保留，不继续扩展

## 3. 数据原则

- `ASOCA2020` 已经移出仓库根目录
- 外部数据路径通过 `data/external/*.path` 记录
- 新代码必须通过 `vessel_seg.config.ProjectPaths` 解析数据路径
- 不允许在仓库内重新放置大体积原始数据

## 4. 开发原则

- 新逻辑优先写入 `vessel_seg/pipeline/`
- 能复用旧算法时，优先做标准化包装，不要再复制一份脚本
- 新增文件结构时，要同步更新：
  1. `README.md`
  2. `docs/rebuild_repo_plan.md`
  3. `.gitignore`
  4. 相关测试
- 每一轮目录整理、归档或路径变更后，都要至少跑一次当前基线测试

## 5. 输出原则

新流水线统一写入：

- `outputs_reorganized/cases/<case>/stages/01_ct_segmentation`
- `outputs_reorganized/cases/<case>/stages/02_centerline_extraction`
- `outputs_reorganized/cases/<case>/stages/03_centerline_repair`
- `outputs_reorganized/cases/<case>/stages/04_wall_features`
- `outputs_reorganized/cases/<case>/stages/05_rendering`

每个阶段必须至少产出：

- `manifest.json`
- 一个主产物文件

## 6. 代码风格

- 尽量少拆碎文件，但每个阶段要有清晰边界
- 注释若存在，使用简体中文
- 配置键、类名、函数名保持英文
- 新功能优先写 dataclass 契约，再写实现

## 7. 当前允许的重构方向

只优先做以下事情：

1. 数据和输出整理
2. 五阶段流水线标准化
3. 旧脚本向新包结构迁移
4. 统一 CLI 和 manifest
5. 为后续 3D-2D 适配保留清晰接口
6. 根目录、脚本目录、文档目录持续去杂物化

## 8. 不要继续做的事情

- 不要继续往仓库根目录堆数据和图片
- 不要新增新的松散输出目录
- 不要再写只适用于 `Normal_1` 的硬编码主流程
- 不要把 notebook 里的逻辑当成长期接口
- 不要把已经归档到 `archive/` 的文件重新当成主入口
- 不要把历史汇报文档重新放回 `docs/` 根目录
- 不要把新的绘图脚本散落回 `scripts/` 根目录

## 9. 基线验证

当前整理和重构阶段，默认基线测试命令是：

```bash
pytest -q tests/test_cli_smoke.py tests/test_project_layout.py tests/test_pipeline_contracts.py
```

如果改动影响 CLI、路径、输出契约、归档结构，完成后必须运行这组测试。
