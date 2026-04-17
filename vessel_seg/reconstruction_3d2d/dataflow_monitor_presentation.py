"""Compact PPT helpers for pre-op / intra-op dataflow and monitoring slides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


BG = RGBColor(247, 247, 243)
TEXT = RGBColor(27, 33, 41)
MUTED = RGBColor(88, 98, 108)
LINE = RGBColor(124, 132, 142)
RED = RGBColor(205, 79, 64)
BLUE = RGBColor(66, 108, 173)
GREEN = RGBColor(81, 142, 113)
GOLD = RGBColor(188, 146, 56)
WHITE = RGBColor(255, 255, 255)
WARM = RGBColor(252, 245, 232)
COOL = RGBColor(236, 244, 252)
SOFT = RGBColor(235, 247, 239)


@dataclass(frozen=True)
class DataflowMonitorPptConfig:
    """Config for a compact 1-2 slide deck on the dataflow and monitoring process."""

    output_path: Path
    notes_path: Path | None = None
    title: str = "术前-术中数据流与在线监测示意"
    subtitle: str = "用公开图片素材快速刻画 CCTA、C 臂、床位坐标、ECG 与 X 光匹配闭环"
    ccta_png: Path | None = None
    interventional_room_jpg: Path | None = None
    ecg_jpg: Path | None = None
    angiography_jpg: Path | None = None


def _set_bg(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG


def _textbox(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str,
    *,
    size: int,
    bold: bool = False,
    color: RGBColor = TEXT,
    align=PP_ALIGN.LEFT,
) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = Pt(6)
    frame.margin_right = Pt(6)
    frame.margin_top = Pt(2)
    frame.margin_bottom = Pt(2)
    para = frame.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bool(bold)
    run.font.color.rgb = color


def _title(slide, title: str, subtitle: str) -> None:
    _set_bg(slide)
    _textbox(slide, 0.55, 0.34, 12.0, 0.5, title, size=23, bold=True)
    _textbox(slide, 0.57, 0.88, 12.0, 0.28, subtitle, size=10, color=MUTED)
    bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.55), Inches(1.2), Inches(12.1), Inches(0.03))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RED
    bar.line.fill.background()


def _panel(slide, left: float, top: float, width: float, height: float, *, fill: RGBColor, line: RGBColor) -> None:
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.0)


def _labeled_panel(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    fill: RGBColor,
    accent: RGBColor,
) -> None:
    _panel(slide, left, top, width, height, fill=fill, line=accent)
    _textbox(slide, left + 0.12, top + 0.06, width - 0.24, 0.28, title, size=13, bold=True, color=accent)
    _textbox(slide, left + 0.12, top + 0.34, width - 0.24, height - 0.4, body, size=10)


def _arrow(slide, x1: float, y1: float, x2: float, y2: float, *, color: RGBColor, width: float = 1.8) -> None:
    connector = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = color
    connector.line.width = Pt(width)
    try:
        connector.line.end_arrowhead = True
    except Exception:
        pass


def _footer(slide, text: str) -> None:
    _textbox(slide, 0.62, 7.05, 12.0, 0.18, text, size=7, color=MUTED)


def _exists(path: Path | None) -> bool:
    return bool(path is not None and path.exists())


def _slide_one(slide, config: DataflowMonitorPptConfig) -> None:
    _title(slide, config.title, config.subtitle)
    _textbox(slide, 0.72, 1.33, 3.6, 0.22, "Slide 1. 术前与术中的核心数据流", size=11, bold=True, color=RED)

    _panel(slide, 0.72, 1.62, 4.55, 3.05, fill=WHITE, line=RED)
    _textbox(slide, 0.88, 1.73, 2.4, 0.25, "术前：CT / CCTA 重建", size=14, bold=True, color=RED)
    if _exists(config.ccta_png):
        slide.shapes.add_picture(str(config.ccta_png), Inches(0.9), Inches(2.02), width=Inches(4.18), height=Inches(2.22))
    _textbox(slide, 0.92, 4.26, 4.1, 0.28, "CT volume -> 冠脉可视化 -> segmentation / clean tree / branch prior", size=9, color=MUTED)

    _panel(slide, 7.03, 1.62, 5.55, 3.05, fill=WHITE, line=BLUE)
    _textbox(slide, 7.22, 1.73, 3.8, 0.25, "术中：C 臂 + 手术台 / 床位 + 实时监测", size=14, bold=True, color=BLUE)
    if _exists(config.interventional_room_jpg):
        slide.shapes.add_picture(str(config.interventional_room_jpg), Inches(7.2), Inches(2.02), width=Inches(3.63), height=Inches(2.2))
    if _exists(config.angiography_jpg):
        slide.shapes.add_picture(str(config.angiography_jpg), Inches(10.95), Inches(2.02), width=Inches(1.35), height=Inches(1.05))
    if _exists(config.ecg_jpg):
        slide.shapes.add_picture(str(config.ecg_jpg), Inches(10.95), Inches(3.18), width=Inches(1.35), height=Inches(1.05))
    _textbox(slide, 7.22, 4.26, 5.1, 0.28, "C-arm angles + bed xyz + ECG_t + fluoroscopy/X-ray_t", size=9, color=MUTED)

    _labeled_panel(
        slide,
        5.52,
        2.22,
        1.26,
        1.85,
        "融合层",
        "世界系\nW\n+\n心脏系\nH\n+\n观察系\nC",
        fill=WARM,
        accent=GOLD,
    )
    _arrow(slide, 5.26, 3.0, 5.52, 3.0, color=RED)
    _arrow(slide, 6.78, 3.0, 7.03, 3.0, color=BLUE)

    _labeled_panel(
        slide,
        0.84,
        5.05,
        2.9,
        1.22,
        "建模输入",
        "术前提供结构先验：冠脉 mask、clean tree、branch radii、prototype prior。",
        fill=WARM,
        accent=RED,
    )
    _labeled_panel(
        slide,
        3.98,
        5.05,
        4.02,
        1.22,
        "在线状态",
        "z_t = {^W0T_B, ^B T_H, [x,theta]_cav, [x,theta]_L, [x,theta]_R, beta_soft, ecg_phase}",
        fill=COOL,
        accent=BLUE,
    )
    _labeled_panel(
        slide,
        8.24,
        5.05,
        4.02,
        1.22,
        "输出 / 比较",
        "synthetic projection + real X-ray matching + 误差反馈，形成在线监测闭环。",
        fill=SOFT,
        accent=GREEN,
    )
    _arrow(slide, 3.75, 5.65, 3.98, 5.65, color=RED)
    _arrow(slide, 8.0, 5.65, 8.24, 5.65, color=GREEN)

    _footer(
        slide,
        "Images: MBq/Wikimedia Commons (CCTA CAD-RADS 4a, CC BY-SA 4.0); Navy Medicine/Wikimedia Commons (Interventional radiology A, Public Domain); "
        "Todt et al./Wikimedia Commons (Coronary angiography..., CC BY 2.0); James Heilman, MD/Wikimedia Commons (Bigeminy, CC BY-SA 3.0).",
    )


def _slide_two(slide, config: DataflowMonitorPptConfig) -> None:
    _title(slide, "术中监测闭环示意", "重点突出 C 臂角度、床位 xyz、ECG 与真实 X 光如何进入状态估计")
    _textbox(slide, 0.72, 1.33, 3.9, 0.22, "Slide 2. 在线估计与监测过程", size=11, bold=True, color=RED)

    _panel(slide, 0.72, 1.65, 4.2, 4.9, fill=WHITE, line=BLUE)
    _textbox(slide, 0.9, 1.78, 2.4, 0.24, "术中场景感知", size=14, bold=True, color=BLUE)
    if _exists(config.interventional_room_jpg):
        slide.shapes.add_picture(str(config.interventional_room_jpg), Inches(0.9), Inches(2.08), width=Inches(3.82), height=Inches(2.78))
    _arrow(slide, 1.65, 4.95, 1.65, 4.35, color=RED, width=2.0)
    _arrow(slide, 1.65, 4.95, 2.25, 4.95, color=GREEN, width=2.0)
    _arrow(slide, 1.65, 4.95, 1.25, 5.33, color=GOLD, width=2.0)
    _textbox(slide, 2.28, 4.83, 0.6, 0.18, "x", size=11, bold=True, color=GREEN)
    _textbox(slide, 1.52, 4.18, 0.6, 0.18, "z", size=11, bold=True, color=RED)
    _textbox(slide, 0.98, 5.33, 0.7, 0.18, "y", size=11, bold=True, color=GOLD)
    _textbox(slide, 0.95, 5.55, 3.7, 0.45, "床位 / 患者位姿 -> 统一成 ^W0T_B 与 ^B T_H；C 臂与 observer 几何则写成 alpha / beta / d_SD / d_SI。", size=9, color=MUTED)

    _labeled_panel(
        slide,
        5.18,
        1.75,
        3.0,
        1.15,
        "观测 y_t",
        "X-ray_t + ECG_t + C-arm_t + table_xyz_t",
        fill=COOL,
        accent=BLUE,
    )
    _labeled_panel(
        slide,
        8.72,
        1.75,
        3.75,
        1.15,
        "状态 z_t",
        "^W0T_B + ^B T_H + [x,theta]_cav,L,R + beta_soft + phase",
        fill=WARM,
        accent=GOLD,
    )
    _arrow(slide, 8.18, 2.33, 8.72, 2.33, color=BLUE)

    _labeled_panel(
        slide,
        5.18,
        3.18,
        3.0,
        1.75,
        "前向模型 h(z_t)",
        "把心脏系冠脉放到世界系，再用 C 臂投影到 detector，生成 synthetic X-ray。",
        fill=SOFT,
        accent=GREEN,
    )
    _labeled_panel(
        slide,
        8.72,
        3.18,
        3.75,
        1.75,
        "匹配 / 更新",
        "比较真实 X 光与 synthetic projection，先做 MAP matching，再扩展到 Bayesian filtering。",
        fill=WHITE,
        accent=RED,
    )
    _arrow(slide, 6.68, 2.9, 6.68, 3.18, color=GREEN)
    _arrow(slide, 8.18, 4.05, 8.72, 4.05, color=RED)

    if _exists(config.angiography_jpg):
        slide.shapes.add_picture(str(config.angiography_jpg), Inches(5.28), Inches(5.3), width=Inches(1.45), height=Inches(1.35))
    if _exists(config.ecg_jpg):
        slide.shapes.add_picture(str(config.ecg_jpg), Inches(6.9), Inches(5.3), width=Inches(2.65), height=Inches(1.35))
    _labeled_panel(
        slide,
        9.7,
        5.25,
        2.7,
        1.35,
        "一句话",
        "术前给结构先验，术中给观测流，二者在坐标系和投影模型里汇合。",
        fill=SOFT,
        accent=GREEN,
    )

    _footer(
        slide,
        "Recommended annotation in talk: y_t comes from image/monitor streams; z_t lives in the model state; h(z_t) bridges pre-op structure and intra-op observation.",
    )


def build_dataflow_monitor_ppt(config: DataflowMonitorPptConfig) -> Path:
    """Generate a compact 2-slide PPT."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_one(slide, config)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_two(slide, config)

    prs.save(str(config.output_path))
    return config.output_path


def build_dataflow_monitor_notes(config: DataflowMonitorPptConfig) -> Path | None:
    """Write short notes and source links for the compact deck."""
    if config.notes_path is None:
        return None
    config.notes_path.parent.mkdir(parents=True, exist_ok=True)
    content = """# 术前-术中数据流与在线监测示意

## Slide 1 讲法

- 左侧用 CCTA 图片说明术前提供的是结构先验：CT volume、冠脉重建、分割、centerline tree、branch prior。
- 右侧用介入室照片、冠脉造影和 ECG 说明术中提供的是观测流：C 臂角度、床位坐标、X 光帧、心电相位。
- 中间强调五套参考系：床零位世界系 W0、床台系 B、心脏系 H、机械系 C、观测系 O。

## Slide 2 讲法

- 左侧说明“床位 xyz + C 臂角度”首先是一个坐标系问题。
- 右侧说明观测 y_t、状态 z_t、前向模型 h(z_t) 和匹配更新的关系。
- 结论是：术前给模型骨架，术中给实时观测，两者通过投影模型连接。

## Sources

- CCTA image: https://commons.wikimedia.org/wiki/File:CCTA_CAD-RADS_4a.png
- Interventional room: https://commons.wikimedia.org/wiki/File:Interventional_radiology_A.jpg
- ECG image: https://commons.wikimedia.org/wiki/File:Bigeminy.jpg
- Coronary angiography: https://commons.wikimedia.org/wiki/File:Coronary_angiography_of_a_STEMI_patient,_showing_partial_occlusion_of_left_circumflex_coronary_artery.jpg
"""
    config.notes_path.write_text(content, encoding="utf-8")
    return config.notes_path
