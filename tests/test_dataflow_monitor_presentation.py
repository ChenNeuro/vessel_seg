from pathlib import Path

from vessel_seg.reconstruction_3d2d.dataflow_monitor_presentation import (
    DataflowMonitorPptConfig,
    build_dataflow_monitor_notes,
    build_dataflow_monitor_ppt,
)


def test_build_dataflow_monitor_ppt_creates_outputs(tmp_path: Path) -> None:
    output = tmp_path / "dataflow_monitor.pptx"
    notes = tmp_path / "dataflow_monitor.md"
    config = DataflowMonitorPptConfig(output_path=output, notes_path=notes)

    ppt_path = build_dataflow_monitor_ppt(config)
    notes_path = build_dataflow_monitor_notes(config)

    assert ppt_path.exists()
    assert ppt_path.suffix == ".pptx"
    assert notes_path is not None
    assert notes_path.exists()
    assert "术前-术中数据流与在线监测示意" in notes_path.read_text(encoding="utf-8")
