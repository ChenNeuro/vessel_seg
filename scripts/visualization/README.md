# visualization scripts

这个目录集中放置仍然保留的绘图和对比脚本。

当前文件：

- `plot_centerline_vs_gt.py`
- `plot_segmentation_overlay.py`
- `compare_centerline_vs_gt_centerline.py`
- `plot_tree_fs_cones.py`
- `plot_centerline_matplotlib_interactive.py`
- `compare_asoca.py`
- `plot_branch_cluster_galleries.py`
- `plot_branch_side_galleries.py`
- `generate_reconstruction_flow_ppt.py`

这些脚本仍然可用，但它们属于辅助可视化层，不是新的五阶段工程主线。

长期方向：

- 能迁入 `vessel_seg/visualization.py` 或 `vessel_seg/pipeline/` 的逻辑，后续应逐步迁入包内
- 保留在这里的脚本应尽量只是薄包装
