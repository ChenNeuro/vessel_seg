# CLAUDE.md

Read `AGENTS.md` first. Treat it as the canonical repo contract.

Additional Claude-specific guidance:

- Prefer concise, high-signal summaries during exploration.
- When making architectural suggestions, anchor them to:
  - `docs/coronary_analysis_master_blueprint.md`
  - `docs/data_output_reorganization.md`
- If a task touches outputs or schemas, explicitly mention:
  - legacy compatibility
  - target path under `outputs_reorganized/`
  - whether tests/docs were updated

Default mental model:

1. stable centerline tree
2. stable topology representation
3. stable SCCT18 labeling
4. stable visualization and reports

Do not introduce a parallel architecture unless necessary.
