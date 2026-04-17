#!/usr/bin/env python3
"""Generate a compact 2-slide deck for dataflow and monitoring discussion."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vessel_seg.reconstruction_3d2d.dataflow_monitor_presentation import (  # noqa: E402
    DataflowMonitorPptConfig,
    build_dataflow_monitor_notes,
    build_dataflow_monitor_ppt,
)


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact dataflow/monitoring PPT with public web images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated/dataflow_monitor_slides_20260408.pptx"),
        help="Output PPTX path.",
    )
    parser.add_argument(
        "--notes-output",
        type=Path,
        default=Path("docs/generated/dataflow_monitor_slides_20260408_notes.md"),
        help="Optional notes markdown path.",
    )
    parser.add_argument(
        "--ccta-png",
        type=Path,
        default=Path("docs/generated/web_assets/ccta_cad_rads_4a.png"),
        help="Pre-op CCTA image.",
    )
    parser.add_argument(
        "--interventional-room-jpg",
        type=Path,
        default=Path("docs/generated/web_assets/interventional_radiology_room.jpg"),
        help="Intra-op C-arm room image.",
    )
    parser.add_argument(
        "--ecg-jpg",
        type=Path,
        default=Path("docs/generated/web_assets/ecg_bigeminy.jpg"),
        help="ECG image.",
    )
    parser.add_argument(
        "--angiography-jpg",
        type=Path,
        default=Path("docs/generated/web_assets/coronary_angiography.jpg"),
        help="Angiography image.",
    )
    args = parser.parse_args()

    config = DataflowMonitorPptConfig(
        output_path=_resolve(args.output),
        notes_path=_resolve(args.notes_output),
        ccta_png=_resolve(args.ccta_png),
        interventional_room_jpg=_resolve(args.interventional_room_jpg),
        ecg_jpg=_resolve(args.ecg_jpg),
        angiography_jpg=_resolve(args.angiography_jpg),
    )
    ppt_path = build_dataflow_monitor_ppt(config)
    notes_path = build_dataflow_monitor_notes(config)
    print(f"pptx: {ppt_path}")
    if notes_path is not None:
        print(f"notes: {notes_path}")


if __name__ == "__main__":
    main()
