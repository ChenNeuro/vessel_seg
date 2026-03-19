# vessel-seg-overnight-reorg

## Objective

Spend one focused evening reorganizing the `vessel_seg` repository so it becomes easier to understand, easier to run, and safer to debug, while preserving or improving performance.

This plan is intentionally constrained to engineering cleanup, workflow consolidation, documentation, verification, and performance-safe preparation. It does not include large algorithm rewrites or model replacement.

## Why This Plan

The repository already contains useful package logic in `vessel_seg/`, but its operational surface is fragmented across many scripts. The biggest leverage tonight is not a new segmentation algorithm. It is reducing friction:

1. make the repo understandable,
2. make the main workflows runnable from one stable interface,
3. make debugging and benchmarking easier,
4. keep all performance and metric guardrails intact.

## Scope

### In scope tonight

- Full repository review artifact
- Stable package entrypoints and thin wrappers
- Simplified run/debug workflow for primary paths
- Performance-safe cleanup and smoke validation
- SOTA review translated into practical next steps
- A second-stage follow-up roadmap after tonight's tranche

### Out of scope tonight

- Replacing TotalSegmentator
- Rewriting all scripts into package modules
- Model architecture experiments that change benchmark semantics
- Any unguarded performance or metric regressions
- Broad destructive reorganization of user data, outputs, or notebooks

## Guardrails

1. Preserve current output contracts: `metrics.json`, `metrics.csv`, directory conventions, and current key CLI flags.
2. Do not introduce algorithm changes unless they are behind a flag and benchmark-safe.
3. Do not remove legacy scripts unless a thin compatibility wrapper remains.
4. Any cleanup that touches runtime behavior must be validated with smoke checks first.
5. Any performance claim must be tied to the existing benchmark/perf-contract path.
6. Avoid dependency lock-in that breaks the user's current `conda` environment `vessel`.

## Assumptions To Validate During Execution

1. The user's working environment is `conda activate vessel`.
2. Existing local data paths may not all be present, so help/smoke tests matter more than full end-to-end runs.
3. Current modified files in the worktree must not be overwritten casually.
4. High-value workflows are:
   - `quant` step pipeline
   - `normal1` end-to-end workflow
   - ASOCA branch/tree/shape workflow

## Acceptance Criteria For Tonight

Tonight is successful if all of the following are true:

1. A newcomer can understand the repo from one review doc.
2. Primary workflows have one clearer recommended invocation path.
3. Entry points are more stable than before and easier to debug.
4. No benchmark gate assumptions are weakened.
5. Help/smoke validation passes for the main commands touched tonight.
6. A concrete follow-up roadmap exists for correctness, debugging, performance, and SOTA exploration.

## Work Plan

### Phase 1 - Inventory and write the repo review

Goal: turn repo knowledge into a durable engineering artifact.

Tasks:

1. Review top-level structure and classify files by role:
   - reusable package logic
   - operational scripts
   - documentation
   - benchmark/perf tooling
   - experiments and presentations
2. Write or refine a single markdown review document covering:
   - what the repo does
   - current main workflows
   - pain points in execution/debugging
   - current performance constraints
   - practical SOTA directions
3. Make the review explicit about what is production-like vs exploratory.

Deliverables:

- `docs/repo_review_and_stage1_plan.md`

Completion check:

- doc exists and explains repo purpose, current state, and next steps clearly

### Phase 2 - Consolidate primary entry points

Goal: reduce script sprawl for the workflows that already have reusable package logic.

Tasks:

1. Ensure package metadata exists so editable install is possible.
2. Provide a package CLI surface under `python -m vessel_seg ...`.
3. Keep legacy scripts working as compatibility paths where practical.
4. Focus on already-package-backed workflows first:
   - quantitative stages
   - Normal_1 end-to-end path
5. Avoid trying to package every experimental script tonight.

Deliverables:

- `pyproject.toml`
- `vessel_seg/__main__.py`
- `vessel_seg/cli.py`
- `vessel_seg/workflows.py`

Completion check:

- `python -m vessel_seg --help` works
- `python -m vessel_seg quant --help` works
- `python -m vessel_seg normal1 --help` works

### Phase 3 - Remove fragile execution patterns

Goal: make workflow execution easier to reason about and debug.

Tasks:

1. Replace `shell=True` subprocess patterns in primary batch runners.
2. Prefer argument lists over shell-assembled command strings.
3. Keep external-tool subprocesses only where the tool boundary is real, such as TotalSegmentator or VMTK.
4. Reduce package-local script chaining where direct in-process calls already exist.

Deliverables:

- safer batch execution in key runners

Completion check:

- touched batch runners still pass `--help`
- touched Python files compile cleanly

### Phase 4 - Debug-readiness and smoke verification

Goal: tonight's cleanup must improve confidence, not just aesthetics.

Tasks:

1. Run `--help` on all touched primary CLIs.
2. Run `py_compile` on touched Python files.
3. Run dry-run flow for the Normal_1 package workflow if possible.
4. Record what could not be fully executed because of environment/data/tool availability.
5. Do not claim full success on unexecuted heavy pipelines.

Deliverables:

- smoke validation notes included in the final summary

Completion check:

- all touched entrypoints show valid help
- touched files compile

### Phase 5 - Performance-safe preparation

Goal: prepare the repo for later optimization without mixing in risky behavior changes tonight.

Tasks:

1. Align all cleanup with the existing perf contract and benchmark docs.
2. Keep performance-sensitive changes limited to engineering improvements, such as:
   - removing avoidable shell invocation
   - reducing workflow indirection
   - clarifying benchmark usage
3. Avoid changing algorithms tonight unless fully isolated.
4. List concrete profiling targets for the next work session.

Deliverables:

- updated review doc with perf-safe guidance

Completion check:

- plan does not weaken `docs/performance_contract.md`

### Phase 6 - SOTA translation, not SOTA churn

Goal: identify modern directions worth exploring later without destabilizing the repo tonight.

Tasks:

1. Summarize modern coronary/vessel segmentation trends relevant to this repo:
   - topology-preserving multi-stage extraction
   - CNN + Transformer hybrids
   - state-space or Mamba-style continuity modeling
   - transformer-based CCTA vessel/plaque systems
2. Translate those into future experiment lanes instead of immediate rewrites.
3. Separate "baseline engineering" from "future algorithm work".

Deliverables:

- SOTA section in repo review doc

Completion check:

- future model ideas are explicit, but deferred behind roadmap phases

## Tonight's Execution Order

Follow this exact order:

1. Confirm review doc is complete and repo-facing.
2. Stabilize package CLI and install surface.
3. Clean dangerous execution patterns in the main batch path.
4. Run smoke validation for every touched entrypoint.
5. Summarize verified results and remaining gaps.
6. Produce a second-stage roadmap for the next session.

## Auto-Resolved Decisions

These choices are locked in unless new evidence appears:

1. Prefer engineering cleanup over algorithm replacement tonight.
2. Keep legacy scripts rather than deleting them.
3. Use the existing performance contract as the governing refactor rule.
4. Treat `python -m vessel_seg ...` as the new recommended entry surface.
5. Assume the user's conda environment is `vessel` and avoid dependency churn.

## Critical Risks To Watch

1. Worktree is already dirty, so unrelated user changes must be left alone.
2. Packaging may accidentally imply hard dependency requirements that are only optional at runtime.
3. Duplicate sources of truth can emerge if old scripts and new CLI diverge.
4. It is easy to over-expand scope into a full package rewrite. Do not.

## Gaps And Edge Cases

### Critical

1. Some heavy workflows require data or external binaries that may not be locally runnable tonight.
2. Not every script has a package-level API yet, so complete script unification is not realistic in one evening.

### Minor

1. Some docs may overlap and need later consolidation.
2. Experimental notebooks and presentation assets should remain low priority.

### Ambiguous But Safe To Defer

1. Which SOTA model family should be explored first after stabilization.
2. Whether to fully package branch/tree workflows in the next session or only their most-used paths.

## After Tonight: Next Plan

After the overnight tranche is complete, immediately create the next-stage plan with these phases:

### Next Plan A - Correctness and debug hardening

1. Add systematic smoke checks for key CLIs.
2. Add focused regression tests for centerline metrics and VTP fallback behavior.
3. Standardize error messages and stage contracts.

### Next Plan B - Package migration of high-value workflows

1. Move the branch/tree workflow into package APIs.
2. Turn `scripts/` into thin wrappers.
3. Remove import hacks from production entrypoints.

### Next Plan C - Profile-driven optimization

1. Benchmark representative cases using the existing perf contract.
2. Profile I/O repetition and hot geometry code.
3. Only land optimizations that are benchmark-safe.

### Next Plan D - Feature-flagged SOTA experiments

1. Evaluate topology-aware and hybrid modern backends.
2. Keep baseline default unchanged until A/B evidence is strong.

## Definition Of Done

Tonight is done when:

1. the review doc is usable,
2. the package CLI is usable,
3. the main runner cleanup is validated,
4. verification results are documented honestly,
5. the next-stage roadmap is written.

## Start Command

When ready to execute this plan, start with:

```text
/start-work vessel-seg-overnight-reorg
```
