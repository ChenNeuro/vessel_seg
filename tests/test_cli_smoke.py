from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_ok(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_package_cli_help() -> None:
    completed = _run_ok("-m", "vessel_seg", "--help")
    assert "Unified CLI for vessel_seg workflows." in completed.stdout


def test_package_quant_help() -> None:
    completed = _run_ok("-m", "vessel_seg", "quant", "--help")
    assert "step1" in completed.stdout
    assert "step5" in completed.stdout


def test_package_normal1_help() -> None:
    completed = _run_ok("-m", "vessel_seg", "normal1", "--help")
    assert "Run ASOCA Normal_1 full step1~step5 pipeline in one command." in completed.stdout


def test_package_clinical_demo_help() -> None:
    completed = _run_ok("-m", "vessel_seg", "clinical-demo", "--help")
    assert "clinical dashboard workflow" in completed.stdout


def test_package_pipeline_case_help() -> None:
    completed = _run_ok("-m", "vessel_seg", "pipeline-case", "--help")
    assert "--case-id" in completed.stdout


def test_legacy_quant_wrapper_help() -> None:
    completed = _run_ok("scripts/quant_pipeline.py", "--help")
    assert "Coronary 5-step quantitative pipeline." in completed.stdout


def test_batch_runner_help() -> None:
    completed = _run_ok("scripts/run_pipeline.py", "--help")
    assert "Batch run tree + branch dataset + similarity." in completed.stdout
