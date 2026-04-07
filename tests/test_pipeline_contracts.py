from pathlib import Path

from vessel_seg.pipeline.contracts import CaseInputs, PipelineConfig
from vessel_seg.pipeline.layout import build_pipeline_layout


def test_pipeline_layout_stage_dir() -> None:
    layout = build_pipeline_layout(Path("/tmp/out"), "Normal_1", Path("/tmp/repo"))
    assert layout.stage_dir("ct_segmentation") == Path("/tmp/out/cases/Normal_1/stages/01_ct_segmentation")
    assert layout.repo_root == Path("/tmp/repo")


def test_case_inputs_to_dict() -> None:
    inputs = CaseInputs(case_id="Normal_1", ct_path=Path("/tmp/ct.nii.gz"))
    payload = inputs.to_dict()
    assert payload["case_id"] == "Normal_1"
    assert payload["ct_path"] == "/tmp/ct.nii.gz"


def test_pipeline_config_defaults(tmp_path: Path) -> None:
    config = PipelineConfig(output_root=tmp_path / "outputs")
    assert config.segmentation_backend == "existing_mask"
    assert config.output_root == tmp_path / "outputs"
