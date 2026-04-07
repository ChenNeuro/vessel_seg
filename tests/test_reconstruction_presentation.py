from pathlib import Path

from vessel_seg.reconstruction_3d2d.presentation import ReconstructionFlowPptConfig, build_reconstruction_flow_ppt


def test_build_reconstruction_flow_ppt_creates_pptx(tmp_path: Path) -> None:
    output = tmp_path / "demo.pptx"

    path = build_reconstruction_flow_ppt(
        ReconstructionFlowPptConfig(
            case_id="Case_A",
            output_path=output,
        )
    )

    assert path.exists()
    assert path.suffix == ".pptx"
