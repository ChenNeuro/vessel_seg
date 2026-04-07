# scripts

这个目录现在只保留两类脚本：

1. 当前仍在使用的兼容入口  
2. 尚未完全迁入 `vessel_seg/` 包内的算法包装脚本

主线入口：

- `pipeline/run_case_pipeline.py`：新的五阶段工程主线
- `quant_pipeline.py`：量化流程兼容入口
- `run_pipeline.py`：旧版批处理兼容入口
- `run_clinical_completion_demo.py`：临床展示流程入口
- `visualization/`：集中放置绘图与对比脚本

仍保留在根目录的脚本类型：

- 数据与输出整理
- 中心线提取、修复、建树
- 分支特征、相似性、聚合分析
- TotalSeg 量化流程
- 仍有文档引用的旧兼容脚本

不再作为主线维护的历史脚本已经移入：

- `archive/scripts_legacy/analysis/`
- `archive/scripts_legacy/visualization/`
- `archive/scripts_legacy/prior_sampling/`
- `archive/scripts_legacy/reporting/`
- `archive/scripts_legacy/demos/`

使用原则：

- 新功能优先写入 `vessel_seg/` 或 `vessel_seg/pipeline/`
- `scripts/` 只保留薄包装和兼容入口
- 若某个脚本只服务于单次实验或展示，应直接进入 `archive/`
