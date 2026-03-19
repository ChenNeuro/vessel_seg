# GitHub Copilot Instructions

Follow `AGENTS.md` in the repository root as the primary coding guide.

Important repository-specific expectations:

- This is a coronary analysis repository, not only a segmentation repo.
- Prefer extending reusable logic in `vessel_seg/`.
- Keep `scripts/` thin when possible.
- Use `outputs_reorganized/` for newly standardized output layouts.
- Do not break legacy paths in `outputs/` without an explicit migration path.
- Keep `ASOCA2020/` treated as read-only raw input.

Before larger refactors, inspect:

- `docs/coronary_analysis_master_blueprint.md`
- `docs/data_output_reorganization.md`
- `README.md`

Preferred development themes:

1. clean centerline tree
2. topology canonicalization
3. branch shape prior
4. SCCT18 labeling
5. CPR / clinical visualization
6. evaluation contracts
