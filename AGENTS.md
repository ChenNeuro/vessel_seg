# AGENTS.md

Canonical repository instructions for coding agents working in this repo.

If there is any conflict between agent-specific helper files and this file, follow this file.

## 1. Mission

This repository is evolving from a research sandbox into a reusable coronary analysis platform.

The core product direction is:

`CCTA / mask / centerline tree -> topology prior -> branch shape prior -> SCCT18 labeling -> CPR / curved MIP / report visualization`

Do not treat this repo as a pure segmentation project. Segmentation is only one entry point.

## 2. Primary Working Language

- User-facing responses should be in Chinese unless explicitly requested otherwise.
- Code, docstrings, schemas, and config keys should stay in English unless there is a strong reason not to.

## 3. Source-of-Truth Documents

Before making broad architectural changes, read these files first:

1. `README.md`
2. `docs/coronary_analysis_master_blueprint.md`
3. `docs/data_output_reorganization.md`
4. `docs/repo_review_and_stage1_plan.md`

For current data/output layout, inspect:

- `docs/generated/data_output_inventory.md`
- `docs/generated/outputs_migration_report.md`

## 4. Current Project Priorities

Prioritize work that strengthens one of these tracks:

1. Data and output normalization
2. Stable clean centerline trees
3. Topology statistics and topology prior
4. Branch shape prior / FGPM / residual modeling
5. SCCT18 labeling
6. CPR / curved MIP / clinical-style visualization
7. Evaluation contracts and reproducibility

If a task does not clearly support one of these tracks, challenge it or isolate it as experimental.

## 5. Repository Map

### Core code

- `vessel_seg/`: reusable package code; preferred home for stable logic
- `vessel_seg/quant/`: quantitative pipeline primitives
- `scripts/`: wrappers, migration tools, one-off utilities, legacy entrypoints
- `tests/`: smoke tests, contract tests, regression tests

### Docs and design

- `docs/`: design docs, pipeline specs, reporting materials
- `docs/coronary_analysis_master_blueprint.md`: master architecture and backlog

### Data and outputs

- `ASOCA2020/`: legacy raw dataset root; treat as read-only
- `data/`: normalized future data root
- `outputs/`: legacy outputs; do not destructively reorganize in-place
- `outputs_reorganized/`: preferred target for normalized outputs

## 6. Output and Data Rules

### Raw data

- Do not rename, move, or rewrite files under `ASOCA2020/` unless explicitly asked.
- Assume `ASOCA2020/` is legacy raw input.

### New data

- Prefer writing new standardized metadata under `data/`.
- Prefer new results under `outputs_reorganized/`, not under loose `outputs/` root.

### Legacy compatibility

- Keep legacy paths working when possible.
- If adding a new default output path, do it in a backward-compatible way.
- Prefer adding `--out-root` / `--output-root` rather than hard-breaking old defaults.

### Case layout

Preferred case output layout:

- `outputs_reorganized/cases/<case>/tree/...`
- `outputs_reorganized/cases/<case>/branches/<variant>/...`
- `outputs_reorganized/cases/<case>/datasets/<variant>/...`
- `outputs_reorganized/cases/<case>/similarity/<variant>/...`
- `outputs_reorganized/cases/<case>/priors/<variant>/...`
- `outputs_reorganized/cases/<case>/cpr/<variant>/...`
- `outputs_reorganized/cases/<case>/topology/...`

### Quant layout

- Official quantitative runs: `outputs_reorganized/quant/official/<case>/...`
- Experimental quantitative runs: `outputs_reorganized/quant/experiments/<run_name>/...`

## 7. Engineering Rules

### Where code should go

- Stable shared logic belongs in `vessel_seg/`
- Thin wrappers and one-off scripts belong in `scripts/`
- Notebook-only logic should be migrated into package code before it becomes important

### Schema-first changes

If a change affects file structure or outputs:

1. update schema/dataclass or at least the documented structure
2. update the relevant docs
3. add or update tests if behavior is stable enough

### Do not silently fork conventions

Avoid adding new one-off naming patterns like:

- `normal1_*`
- `tmp_*`
- `final2_*`
- `debug_new_*`

Use variant folders instead.

### Avoid destructive cleanup

- Do not delete legacy outputs unless explicitly requested
- Prefer copy/migrate/index before delete
- Preserve provenance

## 8. Task Intake Checklist

When starting a task:

1. Identify which track it belongs to:
   - repo/data
   - centerline
   - topology
   - prior
   - labeling
   - visualization
   - evaluation
2. Identify whether the task changes:
   - schema
   - default outputs
   - CLI
   - docs
   - tests
3. Check whether the work should go into:
   - package code
   - script wrapper
   - docs only
4. Preserve compatibility if the task touches existing workflows.

## 9. Preferred Commands

### Environment / package

```bash
pip install -e .
python -m vessel_seg --help
pytest -q
```

### Data/output inventory

```bash
python scripts/audit_data_outputs.py
python scripts/migrate_outputs_layout.py
```

### Quantitative pipeline

```bash
python -m vessel_seg quant --help
python scripts/quant_pipeline.py --help
```

## 10. Architecture Bias

Prefer these decisions unless there is a strong reason otherwise:

- clean centerline tree first, then topology, then labeling
- rule-first SCCT18 labeling, ML only for ambiguous cases
- topology prior and branch shape prior separated, then fused
- RMF-based CPR instead of Frenet frame
- standardized evaluation reports and manifests

## 11. What Good Changes Look Like

A good change usually does at least one of:

- reduces path chaos
- moves logic from scripts into package
- makes centerline/tree outputs more canonical
- improves topology/label schema stability
- strengthens testability
- improves reproducibility or report generation

## 12. Agent Adapters

This repo may also include:

- `CLAUDE.md`
- `GEMINI.md`
- `.cursorrules`
- `.github/copilot-instructions.md`

Those files are thin adapters. This file remains canonical.
