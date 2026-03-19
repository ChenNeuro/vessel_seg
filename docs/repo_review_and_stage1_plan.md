# Repository Review and Stage-1 Cleanup Plan

## What This Repository Does

This repository is a coronary-vessel workflow centered on ASOCA CCTA data.

- `vessel_seg/` contains reusable package logic for I/O, centerline/tree handling, branch-shape extraction, reconstruction, and quantitative metrics.
- `scripts/` contains many thin-to-thick CLIs for one-off experiments, pipeline steps, visualization, benchmark collection, and report generation.
- `docs/` already holds math notes, workflow docs, a refactor plan, and a performance contract.
- The main operational tracks are:
  - ASOCA branch-shape modeling: centerline -> tree -> branch tensor -> similarity/PCA
  - TotalSegmentator quantitative evaluation: CT -> mask -> centerline -> repair -> features -> render
  - Performance gating: collect benchmark -> compare against contract -> block regressions

## File-Level Review Summary

### 1. Core package modules

- `vessel_seg/quant/`: quantitative evaluation primitives and stage executors.
- `vessel_seg/shape.py`: branch extraction, polar profiles, reconstruction, and an internal CLI.
- `vessel_seg/segmentation_interface.py`: backend abstraction for TotalSegmentator.
- `vessel_seg/coronary_tree.py`, `vessel_seg/graph_structure.py`, `vessel_seg/tree_prior.py`, `vessel_seg/fgpm.py`: tree prior and branch-shape modeling work.

### 2. Primary entry points already in use

- `scripts/run_normal1_pipeline.py`: one-command Step1-Step5 workflow for `Normal_1`.
- `scripts/quant_pipeline.py`: stage-by-stage quantitative CLI.
- `scripts/run_pipeline.py`: batch branch dataset workflow.
- `scripts/perf/*.py`: benchmark collection and regression gate.

### 3. Documentation already present

- `docs/totalseg_quantitative_pipeline.md`: current quantitative workflow spec.
- `docs/performance_contract.md`: non-regression criteria.
- `docs/refactor_execution_plan.md`: existing staged refactor strategy.

## Current Problems

The repository already has useful package code, but the execution surface is still fragmented.

1. There is no packaging metadata, so the repo behaves like a script collection instead of an installable tool.
2. Several scripts still rely on `sys.path.insert(...)`, which is a sign that entry points and package boundaries are not yet clean.
3. The main workflow exists in more than one place (`scripts/quant_pipeline.py`, `scripts/run_normal1_pipeline.py`, multiple docs), so users must remember script names rather than product-level commands.
4. `scripts/run_pipeline.py` still shells out in a string-oriented style, which makes debugging and structured reuse harder.
5. The repo is strong on experiments and notes, but weak on a single "start here" engineering document.

## Stage-1 Cleanup Completed In This Pass

This pass focuses on reducing workflow friction without changing algorithms.

1. Added `pyproject.toml` so the repo can be installed editably and exposed as a package CLI.
2. Added `vessel_seg/__main__.py` so the project can run as `python -m vessel_seg`.
3. Added `vessel_seg/cli.py` to provide stable package-level commands:
   - `python -m vessel_seg quant ...`
   - `python -m vessel_seg normal1 ...`
4. Added `vessel_seg/workflows.py` to hold the package-level Normal_1 orchestration logic, reducing dependence on nested script-to-script hops for quantitative stages.
5. Removed `shell=True` command assembly from `scripts/run_pipeline.py` so batch execution is safer and easier to debug.
6. Turned `scripts/quant_pipeline.py` into a thin compatibility wrapper around package CLI logic, so legacy usage still works without duplicating implementation.
7. Added this review document to turn repo knowledge into a durable artifact.

## Simplified Run Flow

### Recommended now

```bash
conda activate vessel
python -m vessel_seg normal1 --skip-existing
```

Stage-level evaluation still works through the package CLI:

```bash
conda activate vessel
python -m vessel_seg quant step2 \
  --seg-mask <mask.nii.gz> \
  --gt-centerline <gt.vtp> \
  --out-dir outputs/quant/<case> \
  --backend vmtk \
  --thr 1.0
```

If you want a shell command after editable install:

```bash
conda activate vessel
pip install -e . --no-deps
vessel-seg normal1 --skip-existing
```

Legacy compatibility paths still work:

```bash
python scripts/quant_pipeline.py step2 --help
python scripts/run_pipeline.py --help
```

## Performance-Safe Engineering Direction

The requirement is "performance up or no worse". For this repository, the safest first-step optimizations are engineering optimizations, not algorithm swaps.

Recommended before touching model behavior:

1. Remove avoidable subprocess chaining for package-local stages.
2. Eliminate `shell=True` command assembly in batch runners.
3. Cache repeated heavy I/O and repeated VTP/NIfTI reads inside long workflows.
4. Keep benchmark artifacts mandatory for every refactor PR.
5. Separate correctness/refactor/performance/algorithm changes exactly as `docs/refactor_execution_plan.md` already suggests.

## SOTA Snapshot

The repo is currently more workflow- and geometry-oriented than end-to-end deep-learning-oriented, so "find SOTA" should be interpreted as: what strong modern vessel-segmentation directions are worth tracking before future model upgrades.

### Coronary / CCTA

- ASOCA remains the canonical public benchmark for full coronary lumen segmentation in CCTA.
- Recent work trends emphasize topology preservation, distal-vessel continuity, and multi-stage extraction instead of pure voxel overlap optimization.
- Recent 2025 examples include topology-preserving multi-stage coronary extraction pipelines and transformer-based CCTA vessel/plaque models such as PlaqueViT.

### Broader vessel segmentation trends

- Hybrid CNN + Transformer remains a strong baseline family when accuracy matters.
- State-space / Mamba-like modules are showing up for long-range vascular continuity modeling.
- Foundation-model-assisted pipelines (for pseudo labels, prompting, or pretraining) are rising, but are not yet the safest default for a small engineering refactor.

### Practical recommendation for this repo

Do not jump to a brand-new SOTA model first. The better order is:

1. make execution reproducible,
2. harden the metrics and perf gate,
3. modularize the workflow,
4. then add model experiments behind feature flags.

## Plan A: Completed First Tranche

### Goal

Create one stable package entry surface and one durable review document.

### Done

- package metadata added
- package CLI added
- package workflow entry added
- legacy quant wrapper aligned to package CLI
- repo review document added

## Plan B: Next Execution Plan

### Phase 2 - Entry-point consolidation

1. Convert remaining high-value scripts into thin wrappers around package modules.
2. Move batch branch workflow logic out of `scripts/run_pipeline.py` into `vessel_seg/`.
3. Remove `sys.path.insert(...)` from production paths.

### Phase 3 - Debug and correctness hardening

1. Run `--help` smoke tests over all primary CLIs.
2. Add focused regression tests for:
   - centerline metrics
   - VTP line reconstruction fallback
   - shape extraction summaries
3. Standardize failure messages and stage contracts.

### Phase 4 - Performance optimization without metric regression

1. Benchmark current workflow end-to-end on a fixed case set.
2. Remove repeated file reads and repeated conversions.
3. Profile `shape.py`, centerline repair, and batch runners.
4. Land only optimizations that pass the existing perf contract.

### Phase 5 - Algorithm exploration behind flags

1. Evaluate topology-aware or transformer/hybrid vessel models as optional backends.
2. Keep existing workflow as the default baseline.
3. Require A/B reports before flipping any default.

## What Was Verified In This Tranche

- `python -m vessel_seg --help`
- `python -m vessel_seg quant --help`
- `python -m vessel_seg normal1 --help`
- `python scripts/quant_pipeline.py --help`
- `python scripts/quant_pipeline.py step2 --help`
- `python scripts/run_pipeline.py --help`
- `python scripts/run_normal1_pipeline.py --help`
- `python -m vessel_seg normal1 --dry-run`
- `py_compile` on the new and modified entrypoints
- `python -m pytest tests/test_cli_smoke.py tests/test_quant_pipeline_contracts.py`

The dry-run path proves command construction and flow wiring, but it is not a substitute for full benchmarked execution with real external tools and datasets.
