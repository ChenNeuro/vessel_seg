from __future__ import annotations

from pathlib import Path
import sys

from vessel_seg.reconstruction_3d2d.contracts import CArmConfig, HeartState, SyntheticProjectionResult


def test_project_case_synthetic_xray_accepts_new_cli_parameters(monkeypatch, tmp_path: Path) -> None:
    import scripts.pipeline.project_case_synthetic_xray as script

    tree_json = tmp_path / "tree.json"
    centerline_vtp = tmp_path / "centerline.vtp"
    tree_json.write_text("{}", encoding="utf-8")
    centerline_vtp.write_text("<vtp />", encoding="utf-8")
    out_dir = tmp_path / "out"

    captured: dict[str, object] = {}

    def fake_resolve_inputs(case_dir, tree_json_arg, centerline_vtp_arg):
        return tree_json, centerline_vtp

    def fake_project_case_centerlines(tree_json_path, centerline_vtp_path, c_arm, heart_state=None):
        captured["tree_json_path"] = tree_json_path
        captured["centerline_vtp_path"] = centerline_vtp_path
        captured["c_arm"] = c_arm
        captured["heart_state"] = heart_state
        return SyntheticProjectionResult(
            case_id="DemoCase",
            tree_json_path=tree_json_path,
            centerline_vtp_path=centerline_vtp_path,
            c_arm=c_arm,
            heart_state=heart_state or HeartState(),
            projected_branches=(),
            metadata={"branch_count": 0, "visible_branch_count": 0},
        )

    def fake_save_projection_result_json(result, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

    def fake_render(result, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")

    monkeypatch.setattr(script, "_resolve_inputs", fake_resolve_inputs)
    monkeypatch.setattr(script, "project_case_centerlines", fake_project_case_centerlines)
    monkeypatch.setattr(script, "save_projection_result_json", fake_save_projection_result_json)
    monkeypatch.setattr(script, "render_projection_preview", fake_render)
    monkeypatch.setattr(script, "render_projection_wall_preview", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project_case_synthetic_xray.py",
            "--tree-json",
            str(tree_json),
            "--centerline-vtp",
            str(centerline_vtp),
            "--out-dir",
            str(out_dir),
            "--alpha-lao-rao-deg",
            "32",
            "--beta-cra-cau-deg",
            "14",
            "--source-to-detector-mm",
            "980",
            "--source-to-isocenter-mm",
            "760",
            "--bed-longitudinal-mm",
            "12",
            "--bed-lateral-mm",
            "-6",
            "--bed-height-mm",
            "120",
            "--bed-tilt-deg",
            "4",
            "--bed-from-heart-translation-mm",
            "1",
            "2",
            "3",
            "--bed-from-heart-rpy-deg",
            "5",
            "6",
            "7",
            "--coronary-frame-rpy-deg",
            "8",
            "9",
            "10",
        ],
    )

    script.main()

    c_arm = captured["c_arm"]
    heart_state = captured["heart_state"]
    assert isinstance(c_arm, CArmConfig)
    assert c_arm.alpha_lao_rao_deg == 32.0
    assert c_arm.beta_cra_cau_deg == 14.0
    assert c_arm.source_to_detector_mm == 980.0
    assert c_arm.source_to_isocenter_mm == 760.0
    assert isinstance(heart_state, HeartState)
    assert heart_state.bed_config.longitudinal_mm == 12.0
    assert heart_state.bed_config.lateral_mm == -6.0
    assert heart_state.bed_config.height_mm == 120.0
    assert heart_state.bed_config.tilt_deg == 4.0
    assert heart_state.bed_from_heart.translation_mm == (1.0, 2.0, 3.0)
    assert heart_state.bed_from_heart.rpy_deg == (5.0, 6.0, 7.0)
    assert heart_state.coronary_frame_rpy_deg == (8.0, 9.0, 10.0)
    assert (out_dir / "synthetic_projection.json").exists()
    assert (out_dir / "projection_preview.png").exists()
    assert (out_dir / "projection_wall_preview.png").exists()


def test_project_case_synthetic_xray_sweep_accepts_new_cli_parameters(monkeypatch, tmp_path: Path) -> None:
    import scripts.pipeline.project_case_synthetic_xray_sweep as script

    tree_json = tmp_path / "tree.json"
    centerline_vtp = tmp_path / "centerline.vtp"
    tree_json.write_text("{}", encoding="utf-8")
    centerline_vtp.write_text("<vtp />", encoding="utf-8")
    out_dir = tmp_path / "sweep"

    captured: dict[str, object] = {}

    def fake_resolve_inputs(case_dir, tree_json_arg, centerline_vtp_arg):
        return tree_json, centerline_vtp

    def fake_run_projection_sweep(**kwargs):
        captured.update(kwargs)
        preview = out_dir / "DEMO" / "projection_preview.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_bytes(b"png")
        return (preview,)

    def fake_save_projection_sweep_gallery(*, output_dir, view_specs, gallery_name, image_name):
        path = output_dir / gallery_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(script, "_resolve_inputs", fake_resolve_inputs)
    monkeypatch.setattr(script, "run_projection_sweep", fake_run_projection_sweep)
    monkeypatch.setattr(script, "save_projection_sweep_gallery", fake_save_projection_sweep_gallery)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project_case_synthetic_xray_sweep.py",
            "--tree-json",
            str(tree_json),
            "--centerline-vtp",
            str(centerline_vtp),
            "--out-dir",
            str(out_dir),
            "--view",
            "DEMO:18:6",
            "--bed-longitudinal-mm",
            "20",
            "--bed-lateral-mm",
            "3",
            "--bed-height-mm",
            "100",
            "--bed-tilt-deg",
            "2",
            "--bed-from-heart-translation-mm",
            "4",
            "5",
            "6",
            "--bed-from-heart-rpy-deg",
            "1",
            "2",
            "3",
            "--coronary-frame-rpy-deg",
            "7",
            "8",
            "9",
            "--source-to-detector-mm",
            "990",
            "--source-to-isocenter-mm",
            "770",
        ],
    )

    script.main()

    assert captured["bed_longitudinal_mm"] == 20.0
    assert captured["bed_lateral_mm"] == 3.0
    assert captured["bed_height_mm"] == 100.0
    assert captured["bed_tilt_deg"] == 2.0
    assert captured["bed_from_heart_translation_mm"] == (4.0, 5.0, 6.0)
    assert captured["bed_from_heart_rpy_deg"] == (1.0, 2.0, 3.0)
    assert captured["coronary_frame_rpy_deg"] == (7.0, 8.0, 9.0)
    assert captured["source_to_detector_mm"] == 990.0
    assert captured["source_to_isocenter_mm"] == 770.0
    assert len(captured["view_specs"]) == 1
    assert captured["view_specs"][0].alpha_lao_rao_deg == 18.0
    assert captured["view_specs"][0].beta_cra_cau_deg == 6.0
    assert (out_dir / "projection_sweep_gallery.png").exists()
    assert (out_dir / "projection_wall_sweep_gallery.png").exists()
