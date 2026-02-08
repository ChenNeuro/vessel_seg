# Coronary Centerline Repair — One‑Page Experiment Plan

## Goal
Repair broken coronary centerlines by bridging gaps using a probability‑guided shortest path with physical constraints (curvature + Murray‑style radius consistency).

## Inputs / Outputs
- **Inputs**: probability map (`.nii/.nii.gz`), VMTK centerlines (`.vtp`)
- **Outputs**: repaired centerlines (`.vtp`) with `is_bridge` flag, JSON report

## Current Algorithm (baseline + physics)
1. **Endpoint detection** from VTP polylines (start/end, outward tangent).
2. **Candidate pairing**: distance < `max_dist` and tangent angle < `max_angle_deg`.
3. **Cost field**:  
   \[
   C = w_{prob}(1-P) + w_{dist}\frac{1}{D+\epsilon} + \text{outside\_penalty}
   \]
   where `P` is probability and `D` is distance transform on `P >= prob_thresh`.
4. **Shortest path** via `route_through_array` (A* on 3D grid).
5. **Physics filters**:
   - **Curvature limit**: max curvature < `max_curvature` (1/mm).
   - **Murray‑style consistency**:  
     \[
     \frac{|r_p^m - r_c^m|}{r_p^m} < \text{murray\_tol}
     \]
     where radii are from VTP (`MaximumInscribedSphereRadius`) or distance transform.
6. **Smoothing**: moving‑average window on bridge points.

## Key Parameters (priority)
1. **Pairing**: `max_dist`, `max_angle_deg`
2. **Cost**: `w_prob`, `w_dist`, `prob_thresh`, `outside_penalty`
3. **Path**: `max_bridge_len`
4. **Physics**: `max_curvature`, `murray_exp`, `murray_tol`
5. **Smoothing**: `smooth_window`

## Metrics
With GT centerlines:
- **centerline‑to‑GT distance** (mean/95% HD)
- **coverage/recall** of GT centerline
- **#components reduction**
Without GT:
- **#components reduction**
- **Murray deviation distribution**
- **path length distribution**

## Ablation Plan (minimal)
1. Baseline only: (`w_prob`, `w_dist`, no curvature/murray)
2. + curvature constraint
3. + Murray constraint
4. Parameter sweep:
   - `max_dist`: 6 / 8 / 10 / 12 mm  
   - `max_angle_deg`: 60 / 75 / 90  
   - `w_prob:w_dist`: (1.0,0.3) / (1.0,0.6) / (1.0,1.0)

## Example Command
```bash
python scripts/repair_centerline.py \
  --prob <prob.nii.gz> \
  --vtp <centerline.vtp> \
  --out outputs/<case>/centerline_repaired.vtp \
  --report outputs/<case>/centerline_repair_report.json \
  --prob_thresh 0.2 --max_dist 10 --max_bridge_len 25 \
  --max_angle_deg 75 --w_prob 1.0 --w_dist 0.6 \
  --max_curvature 0.4 --murray_exp 3.0 --murray_tol 0.5
```

## Notes / Risks
- If VTP lacks `MaximumInscribedSphereRadius`, radii are sampled from the distance transform.
- Murray constraint here is a **consistency heuristic**; tune `murray_tol` to avoid over‑rejecting.
