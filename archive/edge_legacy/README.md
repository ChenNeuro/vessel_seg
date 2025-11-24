# Edge detection (archived)

本目录用于归档此前仓库内的 UAED/自研边缘检测实现。核心代码仍保留在 `vessel_seg/edge.py` 与相关脚本中，若需复现旧版流程：

1. 安装依赖（在 `vesselfm` 环境）：`pip install torch torchvision`
2. 参考原 README 的 “Deep edge detection (DED)” 部分，使用 `python -m vessel_seg.edge train/predict ...`。

注意：主线流程现改用第三方 RCF (third_party/RCF-PyTorch) 边缘结果，并提供转换脚本以融入后续形状/分割管线。
