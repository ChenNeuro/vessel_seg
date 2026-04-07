# Data Layout

This repository no longer keeps the ASOCA dataset inside the git working tree.

- `raw/`: reserved for tiny reproducible fixtures only
- `interim/`: stage-local temporary artifacts
- `processed/`: standardized datasets ready for modeling
- `external/`: path pointers to large external datasets

Current external dataset pointers:

- `external/asoca2020.path`

The default ASOCA location is now:

- `../ASOCA2020`

All new pipeline code should resolve dataset roots through `vessel_seg.config.ProjectPaths`.
