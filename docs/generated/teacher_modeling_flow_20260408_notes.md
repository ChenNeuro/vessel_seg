# 冠脉建模流程与所需数据：讲稿提纲

## Slide 1 标题页

- 先定口径：我们不是从零开始做术中重建，而是在现有冠脉结构化平台上继续往前走。
- 这次汇报要回答三件事：现有仓库做到哪里、老师要的建模层怎么接、真正还缺什么数据。

## Slide 2 Repo Review

- 仓库已经不是单一分割脚本，而是五条线并行：工程主线、语义拓扑、聚类原型、概率形状、3D-2D 观测。
- 这说明老师提出的参考系、聚类、Bayesian 估计并不是空中楼阁，代码里已经有接口。

## Slide 3 当前工程主线

- 先讲清楚五阶段 pipeline 是“静态结构层”。
- 强调 Step3 / Step4 最重要，因为术中重建会直接消费 clean tree、semantic topology 和 radii。

## Slide 4 老师想法映射

- 按输入层、结构层、原型层、状态层、观测层去讲。
- 明确指出：当前仓库已经有结构层和部分观测层，中间的状态建模和实时 matching 还要补。

## Slide 5 参考系

- 建议统一为世界系 W、心脏系 H、观察系 C / detector D。
- 手术台参数和患者摆位放在世界系，冠脉形变参数放在心脏系，C 臂几何放在观察系。

## Slide 6 数据树 / Config vs Workspace

- config space 是低维参数，适合采样和估计。
- workspace 是真实 3D/2D 几何，适合投影和比较。
- 这样能把“结构变化”和“摆位变化”拆开。

## Slide 7 聚类与抽象结构体

- 回答老师：可以换成更抽象的数据结构。
- tree.json 是显式结构，prototype graph 是抽象结构，FGPM / soft coeffs 是软性隐变量。
- 接下来聚类的任务不应只是出图，而要输出可采样的原型先验。

## Slide 8 状态与贝叶斯估计

- 把问题写成 z_t 和 y_t。
- forward 是 synthetic X-ray，inverse 是真实 X 光匹配。
- 可以先讲 MAP matching，再说下一步推广到 Bayesian filtering。

## Slide 9 数据需求

- 术前 CT / mask / tree 这些已有。
- 真正缺的是术中同步数据：C 臂标定、世界系位姿、ECG、真实 X 光。
- 如果老师问“现在最大的缺口是什么”，答案就是术中数据和标定，而不是分割脚本。

## Slide 10 当前证据与路线

- 用现有量化结果证明平台不是概念验证。
- 再落到三步路线：模型采样、实验闭环、实时匹配。
- 最后一句可以收成：先做最小闭环 synthetic -> matching，再走到 online Bayesian 估计。

## 仓库依据

- `README.md`
- `docs/coronary_analysis_master_blueprint.md`
- `vessel_seg/pipeline/stages.py`
- `vessel_seg/pipeline/semantic_topology.py`
- `vessel_seg/pipeline/branch_clustering.py`
- `vessel_seg/fgpm.py`
- `vessel_seg/reconstruction_3d2d/contracts.py`
- `vessel_seg/reconstruction_3d2d/carm_geometry.py`
- `vessel_seg/reconstruction_3d2d/synthetic_projection.py`
