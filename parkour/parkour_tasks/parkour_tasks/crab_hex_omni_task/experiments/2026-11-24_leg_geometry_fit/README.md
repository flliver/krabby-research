# Leg geometry fit, 2026-11-24

This is a fit of the front-left leg model to ruler measurements on the real robot. The measurements, the fitted numbers and the open questions are in `docs/leg-geometry-measurements-2026-11-24.md`.

- `fitted_leg.py`: numpy only.
  - `leg_pose(hip_counts, knee_counts)` returns body-frame points in inches: pivot, knee, toe, toe tip, the linkage pins and anchors, and the femur and tibia outline polygons.
  - It also returns hip and knee angles, rod lengths, and toe side/height in the ruler convention.
  - The fitted parameters are in its `FITTED` dict. The knee is femur-mounted (uncoupled), with 5.33 counts per mm on both actuators.
- `fit_results.json`: the primary fit, the variant fits and the Monte Carlo output.
- `fit_poses.svg`: the measured vs fitted poses.

Status: an analysis script written during the 2026-11-24 robot session, not a reviewed library. Check it against new measurements before relying on it, especially for hip counts above about 1000, where the hip linkage is near dead centre.
