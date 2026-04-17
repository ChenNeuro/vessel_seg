# 术前-术中数据流与在线监测示意

## Slide 1 讲法

- 左侧用 CCTA 图片说明术前提供的是结构先验：CT volume、冠脉重建、分割、centerline tree、branch prior。
- 右侧用介入室照片、冠脉造影和 ECG 说明术中提供的是观测流：C 臂角度、床位坐标、X 光帧、心电相位。
- 中间强调三套参考系：世界系 W、心脏系 H、观察系 C。

## Slide 2 讲法

- 左侧说明“床位 xyz + C 臂角度”首先是一个坐标系问题。
- 右侧说明观测 y_t、状态 z_t、前向模型 h(z_t) 和匹配更新的关系。
- 结论是：术前给模型骨架，术中给实时观测，两者通过投影模型连接。

## Sources

- CCTA image: https://commons.wikimedia.org/wiki/File:CCTA_CAD-RADS_4a.png
- Interventional room: https://commons.wikimedia.org/wiki/File:Interventional_radiology_A.jpg
- ECG image: https://commons.wikimedia.org/wiki/File:Bigeminy.jpg
- Coronary angiography: https://commons.wikimedia.org/wiki/File:Coronary_angiography_of_a_STEMI_patient,_showing_partial_occlusion_of_left_circumflex_coronary_artery.jpg
