# Repository Rebuild Plan

## Rebuild Goal

The repository is being rebuilt around a single maintainable engineering pipeline:

1. CT segmentation
2. Centerline extraction
3. Centerline split and repair
4. Vessel wall feature extraction
5. Rendering

## Canonical Layout

### Code

- `vessel_seg/pipeline/contracts.py`
- `vessel_seg/pipeline/layout.py`
- `vessel_seg/pipeline/stages.py`
- `vessel_seg/pipeline/orchestrator.py`
- `vessel_seg/pipeline/naming.py`
- `vessel_seg/pipeline/semantic_topology.py`
- `vessel_seg/pipeline/canonical_consistency.py`
- `vessel_seg/pipeline/branch_clustering.py`
- `vessel_seg/pipeline/branch_cluster_gallery.py`
- `vessel_seg/pipeline/branch_side_gallery.py`
- `vessel_seg/reconstruction_3d2d/contracts.py`
- `vessel_seg/reconstruction_3d2d/carm_geometry.py`
- `vessel_seg/reconstruction_3d2d/synthetic_projection.py`
- `vessel_seg/reconstruction_3d2d/sweep.py`
- `vessel_seg/reconstruction_3d2d/presentation.py`
- `docs/branch_clustering_lr_baseline.md`
- `docs/branch_clustering_side_first_contract.md`

### Data

- `data/external/asoca2020.path`

### Outputs

- `outputs_reorganized/cases/<case>/stages/01_ct_segmentation`
- `outputs_reorganized/cases/<case>/stages/02_centerline_extraction`
- `outputs_reorganized/cases/<case>/stages/03_centerline_repair`
- `outputs_reorganized/cases/<case>/stages/04_wall_features`
- `outputs_reorganized/cases/<case>/stages/05_rendering`

Stage 3 currently includes:

- `tree.json`
- `branch_names.json`
- `semantic_topology.json`
- optional semantic prior re-labeling via fitted geometry priors
- cross-case analysis tools can quantify canonical-name stability from batch summaries
- optional batch analysis tools can cluster unlabeled branches to expose recurrent geometric subtypes
- cluster gallery diagnostics now support a fixed case grid ordered by cohort and case number, so left/right distribution can be compared across clusters without dynamic reordering
- branch clustering now supports an explicit `side_first` mode: first assign `LCA/RCA/AUX`, then cluster within each side group
- branch clustering can optionally consume an external `side_group` assignments csv, so left/right grouping may come from a prior clustering or manual correction rather than only internal root membership
- side-first diagnostics now also support direct `LCA/RCA` galleries, so left/right grouping can be checked before reading finer cluster splits
- downstream research now has a minimal 3D-2D loop: existing repaired centerlines can be placed in world coordinates and projected with a simplified C-arm model to generate synthetic X-ray observations
- projection sweep utilities can now generate both centerline galleries and wall-style galleries based on stage4 radius information

## Migration Rules

### Legacy

- `scripts/` remains as compatibility and algorithm wrappers
- `outputs/` remains read-only
- notebooks remain exploratory only

### New Development

All new engineering work must target the five-stage pipeline.

## Practical Rule

If a new feature cannot be mapped to one of the five stages, it should not become part of the rebuilt mainline.
