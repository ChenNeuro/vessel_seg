# Next Debug And Performance Work Package

## Goal

After the current repo cleanup tranche, the next work session should focus on correctness hardening and benchmark-safe performance work.

## Priority Order

### 1. CLI and contract smoke coverage

- Add a lightweight smoke suite for the main entrypoints.
- Extend the newly added `tests/test_cli_smoke.py` and `tests/test_quant_pipeline_contracts.py` instead of creating a second testing style.
- At minimum, validate `--help` and dry-run behavior for:
  - `python -m vessel_seg`
  - `python -m vessel_seg quant`
  - `python -m vessel_seg normal1`
  - `python scripts/quant_pipeline.py`
  - `python scripts/run_pipeline.py`

### 2. Correctness targets

- Add focused regression tests for `vessel_seg/quant/pipeline.py` metrics paths.
- Verify the step2 VTP fallback chain:
  - direct VMTK output
  - fallback polyline reconstruction
  - graph reconstruction fallback
- Verify stage outputs always write the expected contract files.

### 3. Profiling targets

Profile these before changing algorithms:

- repeated mask/VTP reads in orchestration code
- `vessel_seg/shape.py` extraction path
- `scripts/repair_centerline.py`
- batch workflow overhead in `scripts/run_pipeline.py`

### 4. Performance-safe optimizations

Only attempt optimizations that can be benchmarked against the existing perf contract.

Good candidates:

- reduce repeated file reads
- cache intermediate metadata in long workflows
- remove unnecessary subprocess indirection where logic is already in-package
- reduce redundant geometry parsing

### 5. SOTA experiment lane

Do not make this the default path yet. First create a feature-flagged experimental lane for one modern direction, likely one of:

- topology-aware post-processing for centerline continuity
- CNN + Transformer hybrid segmentation backend
- continuity-aware sequence or state-space module for vessel topology preservation

## Exit Criteria

This next package is done when:

1. smoke tests exist for main entrypoints,
2. key quantitative and fallback logic has regression coverage,
3. at least one profiling report identifies real hotspots,
4. any optimization landed has perf-gate evidence,
5. SOTA exploration remains isolated from the stable baseline.
