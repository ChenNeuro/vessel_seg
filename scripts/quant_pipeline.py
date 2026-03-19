#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from vessel_seg.cli import build_quant_parser, run_quant_command

    args = build_quant_parser().parse_args(sys.argv[1:])
    run_quant_command(args)


if __name__ == "__main__":
    main()
