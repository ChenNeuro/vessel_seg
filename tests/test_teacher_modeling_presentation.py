from pathlib import Path

from vessel_seg.reconstruction_3d2d.teacher_presentation import (
    TeacherModelingPptConfig,
    build_teacher_modeling_notes,
    build_teacher_modeling_ppt,
)


def test_build_teacher_modeling_ppt_creates_outputs(tmp_path: Path) -> None:
    output = tmp_path / "teacher_modeling.pptx"
    notes = tmp_path / "teacher_modeling.md"
    config = TeacherModelingPptConfig(output_path=output, notes_path=notes)

    ppt_path = build_teacher_modeling_ppt(config)
    notes_path = build_teacher_modeling_notes(config)

    assert ppt_path.exists()
    assert ppt_path.suffix == ".pptx"
    assert notes_path is not None
    assert notes_path.exists()
    assert "冠脉建模流程与所需数据" in notes_path.read_text(encoding="utf-8")
