#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from vessel_seg.clinical_completion import add_clinical_completion_arguments, run_clinical_completion_command

    parser = add_clinical_completion_arguments(
        argparse.ArgumentParser(
            description="Clinical coronary completion demo: centerline, wall features, modeling, longitudinal section, dashboard."
        )
    )
    args = parser.parse_args(sys.argv[1:])
    run_clinical_completion_command(args)


if __name__ == "__main__":
    main()
