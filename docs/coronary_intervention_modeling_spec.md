# 冠脉术前-术中 3D/2D 建模规范

本文档整理当前冠脉 CTA/XA 建模思路，目标是把“术前 CCTA / 冠脉树模型”与“术中 C 臂 / 床位 / ECG / X-ray”统一到一套可实现的坐标变换与参数化模型中。

适用范围：

- 术前 CTA / CCTA 冠脉树到术中单平面 XA 的 2D/3D 配准
- 冠脉树 synthetic projection 仿真
- 术中视角优化、在线匹配、Bayesian / MAP 状态估计
- 后续引入呼吸与心动非刚性补偿

相关实现入口：

- [carm_geometry.py](/Users/chenyihao/mycode/vessel_seg/vessel_seg/reconstruction_3d2d/carm_geometry.py)
- [contracts.py](/Users/chenyihao/mycode/vessel_seg/vessel_seg/reconstruction_3d2d/contracts.py)
- [synthetic_projection.py](/Users/chenyihao/mycode/vessel_seg/vessel_seg/reconstruction_3d2d/synthetic_projection.py)
- [coordinate_frames_figure.py](/Users/chenyihao/mycode/vessel_seg/vessel_seg/reconstruction_3d2d/coordinate_frames_figure.py)
- 坐标系示意图：[coordinate_frames_transform_demo_20260408.png](/Users/chenyihao/mycode/vessel_seg/docs/generated/coordinate_frames_transform_demo_20260408.png)

## 1. 总体目标

我们希望把问题统一写成：

1. 术前给出结构先验：CTA、segmentation、centerline tree、semantic topology、branch radii
2. 术中给出观测流：C 臂角度、SID/SOD、床位 xyz、ECG、X-ray 图像
3. 模型通过一套坐标系变换把 3D 冠脉模型投影到 2D detector
4. 比较真实 X 光与 synthetic projection，得到配准 / 匹配 / 状态估计结果

核心形式：

```text
state z_t  --h(.)-->  synthetic projection  --compare-->  observation y_t
```

其中：

- `z_t`：设备、病人、心脏与冠脉的状态参数
- `h(.)`：由坐标变换和成像几何定义的前向观测模型
- `y_t`：真实 X 光、ECG、设备参数等观测量

## 2. 推荐坐标系

推荐使用如下 5 个坐标系：

### 2.1 `{W0}`：床零位世界系

- 原点：床台中性位时的等中心参考点
- 用途：作为全局统一基准

这层主要服务于工程落地。若术中需要显式建模床体移动，则 `{W0}` 不随床台运动而变化。

### 2.2 `{B}`：当前床台系

- 原点：当前床台工作位下的局部参考点
- 用途：显式表示床台平移、升降、倾斜

如果不单独建模床台，可直接令 `{B}` 与 `{W0}` 重合。

### 2.3 `{C}`：C 臂机械系

- 原点：机械等中心 isocenter
- 用途：表示 C 臂几何姿态，包括 LAO/RAO、CRA/CAU 等角度

这层是设备层的核心。

### 2.4 `{O}`：观测系 / X-ray 相机系

- 原点：X 射线焦点 source
- `z_O`：指向探测器
- 用途：构造 pinhole-style 投影模型

这里的 `{O}` 是“成像光路坐标系”，与 `{C}` 不同。`{C}` 表示机械位姿，`{O}` 表示成像几何。

### 2.5 `{H}`：心脏解剖系

- 原点：推荐使用左冠入口附近 landmark
- 用途：在解剖上稳定地表达冠脉树、心脏刚体位姿和后续非刚性形变

`{H}` 不建议定义成简单的几何中心或 PCA 主轴，而应定义成由稳定解剖 landmark 驱动的心脏解剖坐标系。

## 3. 变换链

### 3.1 最简变换链

如果暂时不显式区分床零位系和当前床台系，可写成：

```text
^O T_H = ^O T_C · ^C T_W · ^W T_H
```

对一个心脏系点 `p_H`，有：

```text
p_O = ^O T_C · ^C T_W · ^W T_H · p_H
```

这条链对应：

```text
心脏模型点 -> 床/世界系 -> C 臂机械系 -> 观测系
```

### 3.2 含床台运动的完整变换链

若显式建模床台运动，推荐写成：

```text
^O T_H = ^O T_C(d_SI) · ^C T_W0(alpha, beta) · ^W0 T_B(Δx_B, Δy_B, Δz_B, τ_B) · ^B T_H
```

或等价写成：

```text
^O T_H = ^O T_C · ^C T_W · ^W T_H
```

其中 `^W T_H` 已经包含床台对病人的影响。

### 3.3 工程层级划分

推荐按 3 层组织：

1. 设备层
   - `^W T_C(alpha, beta, Δx_B, Δy_B, Δz_B, τ_B)`
2. 病人层
   - `^W T_H(x_H, y_H, z_H, ψ_H, θ_H, φ_H)`
3. 成像层
   - `^O T_C(d_SI)` 与 `d_SD`

优点：

- 改 C 臂角度时，只动设备层
- 改病人摆位或配准时，只动病人层
- 改 SID / detector 参数时，只动成像层

## 4. 参数化设计

## 4.1 心脏在床上的位姿：`^W T_H`

建议使用 6 自由度刚体：

```text
^W T_H = T(x_H, y_H, z_H) · Rz(ψ_H) · Ry(θ_H) · Rx(φ_H)
```

参数向量：

```text
q_H = [x_H, y_H, z_H, ψ_H, θ_H, φ_H]
```

推荐作为“配准软约束 / 优化先验”的范围：

```text
x_H ∈ [-150, 150] mm
y_H ∈ [-120, 120] mm
z_H ∈ [50, 180] mm
ψ_H ∈ [-20°, 20°]
θ_H ∈ [-15°, 15°]
φ_H ∈ [-15°, 15°]
```

解释：

- `x_H`：头脚方向偏移
- `y_H`：左右方向偏移
- `z_H`：心脏中心相对床面的高度
- `ψ_H, θ_H, φ_H`：体位、轻度旋转、呼吸和局部姿态偏差

这组范围适合：

- 2D/3D 配准初值
- 术中小范围状态更新
- ROI 约束

## 4.2 C 臂相对床 / 世界的位姿：`^W T_C`

推荐拆成：

```text
^W T_C = T(x_iso, y_iso, z_iso) · R_LAO/RAO(alpha) · R_CRA/CAU(beta) · R0
```

其中：

- `(x_iso, y_iso, z_iso)`：等中心在床系中的位置
- `alpha`：LAO/RAO 角
- `beta`：CRA/CAU 角
- `R0`：设备零位到软件定义机械系之间的固定旋转

推荐符号约定：

- `alpha > 0`：LAO
- `alpha < 0`：RAO
- `beta > 0`：CRA
- `beta < 0`：CAU

典型工作范围：

```text
alpha ∈ [-120°, 120°]
beta ∈ [-45°, 45°]
```

这组范围适合 Azurion / 同类单平面冠脉工作位的抽象建模。

## 4.3 床台参数：`^W0 T_B`

若世界系原点就定义在中性等中心，可把中性位置写成零位。否则推荐显式引入：

```text
q_B = [Δx_B, Δy_B, Δz_B, τ_B]
```

建议范围：

```text
Δx_B ∈ [-600, 600] mm
Δy_B ∈ [-180, 180] mm
Δz_B ∈ [790, 1040] mm
τ_B  ∈ [-16.5°, 16.5°]
```

解释：

- `Δx_B`：床纵向移动
- `Δy_B`：床横向移动
- `Δz_B`：床台高度
- `τ_B`：床台倾角

## 4.4 成像层：`^O T_C`、`d_SI` 与 `d_SD`

若 `{C}` 原点在 isocenter，`{O}` 原点在焦点，且 `z_O` 指向 detector，则可写成：

```text
^C T_O = T(0, 0, -d_SI)
```

或等价地：

```text
^O T_C = T(0, 0, d_SI)
```

其中：

```text
d_SI ≈ 765 mm
```

典型 SID 范围：

```text
d_SD ∈ [890, 1235] mm
```

则等中心到探测器距离：

```text
d_ID = d_SD - d_SI
```

这部分直接决定投影模型的尺度和放大率。

## 4.5 推荐的完整参数向量

推荐统一定义：

```text
q = [
  alpha, beta, d_SD,
  Δx_B, Δy_B, Δz_B, τ_B,
  x_H, y_H, z_H, ψ_H, θ_H, φ_H
]^T
```

其中：

- 前 7 项是设备 / 床台硬约束
- 后 6 项是病人 / 心脏软约束

## 5. 推荐默认值

可作为仿真、单元测试、配准初值的默认值：

```text
alpha = 30°
beta  = 20°
d_SI  = 765 mm
d_SD  = 1000 mm

Δx_B = 0
Δy_B = 0
Δz_B = 900 mm
τ_B  = 0

x_H = 0
y_H = 0
z_H = 120 mm
ψ_H = 0
θ_H = 0
φ_H = 0
```

这组值适合作为第一版系统的标准启动参数。

## 6. 心脏系 `{H}` 的解剖定义

这是本文档最重要的部分之一。

结论先行：

`{H}` 不建议只定义为“心脏几何中心 + PCA 主轴”，而建议定义为：

```text
基于 apex-base 长轴 + coronary ostia / LM bifurcation 的解剖坐标系
```

也就是说，`{H}` 应由解剖 landmark 驱动，而不是仅由形状统计驱动。

## 6.1 推荐使用的 landmark

优先级最高的一组 landmark：

- `p_apex`：左室心尖
- `p_base`：二尖瓣环中心或主动脉根中心
- `p_LM`：左主干冠脉入口 / 左冠 ostium
- `p_bif`：LM -> LAD / LCX 分叉点

辅助 landmark：

- `p_RCA_ostium`
- 近端 LAD / LCX / RCA 方向
- bifurcation 集合
- outlets

不推荐用来定义全局 `{H}` 的对象：

- 心脏几何中心
- 表面 PCA 主轴
- 远端细小分支
- 钙化散点
- guidewire / catheter 痕迹

这些可以用于 refinement，但不适合作为全局心脏系的基准。

## 6.2 两种推荐定义

### 方案 A：以冠脉导航为主

- 原点：`O_H = p_LM`
- `z_H`：base -> apex 长轴
- `x_H`：LM ostium -> LM bifurcation，并在垂直于 `z_H` 的平面内正交化
- `y_H = z_H × x_H`

这套定义最适合 CTA 到 XA 的冠脉 2D/3D 配准。

### 方案 B：以全心脏导航为主

- 原点：`O_H = p_base` 或主动脉根中心
- `z_H`：base -> apex 长轴
- `x_H`：指向左冠 ostium 或室间隔方向
- `y_H = z_H × x_H`

这套定义更适合全心脏 + 冠脉联合导航。

## 6.3 推荐的数学构造

设：

- `p_apex`：左室心尖
- `p_base`：二尖瓣环中心或主动脉根中心
- `p_LM`：左主干 ostium
- `p_bif`：LM bifurcation

### 原点

若以冠脉为主：

```text
O_H = p_LM
```

若以全心脏为主：

```text
O_H = p_base
```

### 长轴

```text
z_hat = (p_apex - p_base) / ||p_apex - p_base||
```

### 冠脉主方向

```text
v = p_bif - p_LM
x_tilde = v - (v^T z_hat) z_hat
x_hat = x_tilde / ||x_tilde||
```

### 第三个轴

```text
y_hat = z_hat × x_hat
x_hat = y_hat × z_hat
```

### 刚体位姿

```text
^W R_H = [x_hat  y_hat  z_hat]
^W t_H = O_H
```

```text
^W T_H =
[ ^W R_H   ^W t_H ]
[   0        1    ]
```

这就是整条链路里最值得保留的心脏位姿定义。

## 7. 呼吸与心动建模建议

`{H}` 建议固定成参考相位下的解剖坐标系，后续时间相关形变不要揉进 `{H}` 的定义本身，而应单独表示：

```text
^W T_H(t) = ^W T_H^rigid · Φ_cardiac(t) · Φ_resp(t)
```

也就是说：

- `{H}`：参考解剖系
- `Φ_cardiac(t)`：心动形变
- `Φ_resp(t)`：呼吸形变

这样：

- 刚体配准更稳定
- landmark 不会被时间形变污染
- 后续 Bayesian / filtering 更自然

## 8. 与当前仓库代码的映射

当前仓库已经有一部分实现，但命名还偏简化版：

### 已有

- `CArmConfig`
  - 当前包含 `lao_rao_deg`, `cra_cau_deg`, `sid_mm`, `sod_mm`, detector 参数
- `HeartState`
  - 当前包含 `world_pose`, `ecg_phase`, `cav_rpy_deg`
- `project_points_world_to_detector`
  - 已能做简化的 3D -> 2D 投影

### 建议补齐

#### 设备层

建议在现有 `CArmConfig` 基础上，显式补一层床台参数：

```python
@dataclass(frozen=True)
class BedConfig:
    dx_mm: float
    dy_mm: float
    dz_mm: float
    tilt_deg: float
```

#### 成像层

`DetectorConfig` 已基本具备雏形，但建议后续更明确区分：

- `source_to_isocenter_mm`
- `source_to_detector_mm`
- `isocenter_to_detector_mm`
- `principal_point_px`
- `distortion_coeffs`

#### 病人层

建议把心脏解剖坐标系的 landmark 显式保存：

```python
@dataclass(frozen=True)
class HeartLandmarks:
    apex_mm: tuple[float, float, float]
    base_center_mm: tuple[float, float, float]
    lm_ostium_mm: tuple[float, float, float]
    lm_bifurcation_mm: tuple[float, float, float]
    rca_ostium_mm: tuple[float, float, float] | None = None
```

并提供：

```python
def build_anatomical_heart_frame(landmarks: HeartLandmarks) -> Pose3D:
    ...
```

## 9. 推荐的工程落地顺序

### 第一阶段：先把刚体链写对

- 固定 `{W0}`, `{B}`, `{C}`, `{O}`, `{H}`
- 明确角度正负号
- 写出 `^O T_H`
- 跑通 synthetic projection

### 第二阶段：把 `{H}` 换成解剖系

- 从 CTA tree / segmentation 中提取 apex、base、LM ostium、LM bifurcation
- 用 landmark 构造 `{H}`
- 替换目前“任意局部旋转”的 `cav_rpy_deg`

### 第三阶段：把床台参数纳入状态量

- 补 `Δx_B, Δy_B, Δz_B, τ_B`
- 让设备层与病人层显式解耦

### 第四阶段：引入时间项

- `ecg_phase`
- `Φ_cardiac(t)`
- `Φ_resp(t)`
- 单帧 MAP -> 多帧 Bayesian filtering

## 10. 实现提醒

### 10.1 零位定义

不同厂商、不同安装位、不同设备工作模式下：

- 角度正负号
- 零位旋转
- 旋转顺序
- DICOM 几何字段含义

都可能不同。

因此在落地时，应允许存在一层固定校正：

```text
R0
```

用于把厂家机械零位映射到本软件内部统一零位。

### 10.2 设备模式

建议把“头端工作位 / head-end working position”视为明确模式，因为不同模式下：

- 可用角度范围不同
- 可达姿态不同
- 某些几何限制不同

### 10.3 真实系统与演示系统

在可视化图中可使用“display-oriented exploded layout”以避免多个坐标系重叠，但在真正计算投影时：

- 必须始终使用真实几何参数
- 不应把 exploded layout 用到数值计算链里

## 11. 推荐保留的核心公式

最值得在代码、文档和 PPT 中统一保留的总式：

```text
^O T_H
=
^O T_C(d_SI)
· ^C T_W(alpha, beta, Δx_B, Δy_B, Δz_B, τ_B)
· ^W T_H(x_H, y_H, z_H, ψ_H, θ_H, φ_H)
```

如果强调心脏系的解剖构造，再补一个：

```text
O_H = p_LM ostium
z_H ∥ (p_apex - p_base)
x_H ∥ projection_of(p_LM bif - p_LM ostium, onto z_H^⊥)
y_H = z_H × x_H
```

这两条式子已经足够作为后续坐标系统一、前向投影和状态估计的建模核心。

## 12. 后续建议

建议下一步做 3 个具体工作：

1. 在 `contracts.py` 里补 `BedConfig` 与 `HeartLandmarks`
2. 实现 `build_anatomical_heart_frame()`，替换当前纯 `cav_rpy_deg` 方式
3. 在 synthetic projection 层把总变换链正式改写成 `W0 / B / C / O / H` 五层模型

如果后续继续推进，建议配套增加：

- 一张 `W0 -> B -> C -> O -> H` 的正式示意图
- 一份状态向量 `q` 的 dataclass 契约
- 一个默认几何参数 JSON / YAML 配置文件
