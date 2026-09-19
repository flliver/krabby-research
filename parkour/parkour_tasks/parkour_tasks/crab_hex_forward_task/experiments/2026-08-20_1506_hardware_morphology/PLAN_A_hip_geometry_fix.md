# PLAN A — hip geometry correction

## Context

The 2026-08-20 hardware-morphology rebuild passed all its verification gates, but the
checkpoint video exposed a geometry misreading the numeric checks couldn't catch: the hip
was modeled as a HORIZONTAL beam extending 22" outboard. The real hip (user-confirmed) is
a **vertical 25.7" × 5" × 1" plate** door-hinged to the body side face: it extends 5"
horizontally outboard and ~6.6" past the top and bottom of the 12.5" body, and the femur
pivot is only **2.5" outboard of the wall** (middle of the width), 3.25" below the body —
i.e. 22.3" down from the plate's top tip. The legs sit ~0.5 m closer to the body than
modeled (toe-to-toe ~3.1 m, not 4.1 m). Supporting evidence that this reading is right:
the plate's center lands exactly at the body's mid-height (top tip +0.326 m = half the
plate length), and the hip actuator's anchor hole 2.08" from the plate's top tip is
~20.3" above the femur pivot — precisely the near-vertical anchor position the linkage
solver had already been forced to infer from the 450–650 mm rod window.

Outcome: the sim robot matches the real machine's silhouette and mass layout; all
verification gates re-pass; the user signs off visually on a fresh video; a re-run smoke
train characterizes the wheelie on the corrected geometry (input to Plan B's lever
decision).

## Changes (numbers land ONLY in the dimensions module + generator + linkage anchors)

1. `parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/mdp/crab_hex_dimensions.py`:
   - Reinterpret the hip as a vertical plate: 25.667" long **vertically**, 5" wide
     **outboard**, 1" thick tangentially. New constants: `FEMUR_PIVOT_OUTBOARD_IN = 2.5`
     (mid-width), `FEMUR_PIVOT_FROM_PLATE_TOP_IN = 22.334`. Assert the derived invariant:
     plate center z == body center z (pivot 3.25" below a 12.5" body centered ⇒ top tip
     = +half-plate-length). Update the provenance docstring.
   - Hip CoM: recompute in the plate plane (y outboard, z vertical) — plate ply at plate
     center; hip actuator CG along the top-anchor→femur-attach line; knee actuator near
     the pivot. Authored `physics:centerOfMass` gains a z component.
2. `assets/scripts/generate_crab_simple.py`:
   - Hip box scale → (0.0254, 0.127, 0.6519418); translate (x_leg, ±0.6731, 0).
   - Yaw + CamShaft joints: anchor z → 0 (mid-plate, on the vertical hinge line at the
     wall, y = ±0.6096); hip-side localPos1 = (0, ∓0.5, 0) (inboard edge).
   - Hip_Femur joint: localPos0 on the hip → (0, 0, −0.37014) (pre-scale z).
   - Femur/tibia/footpad translates shift inboard: femur pivot y = ±0.6731, knee/toe
     y = ±1.2573. Everything else (z heights, femur/tibia dims, masses) unchanged.
   - Regenerate `assets/crab_simple.usda`.
3. `crab_hex_linkage.py`: hip anchor → u = 0 (mid-width, same as the pivot),
   d = −21.0" (clevis just above the plate's top tip — keeps the rod window feasible per
   the existing `test_crab_hex_linkage.py` envelope checks; still flagged an estimate).
   Knee anchor unchanged (u = +2.33", d = −2.5" — near the pivot, within the plate's 5"
   width). Mid-stroke defaults recompute automatically (small shift).
4. Height-scanner grid in `crab_hex_scene_cfg.py`: size [2.4, 4.2] → [2.4, 3.4] (toes
   now reach y ≈ ±1.54 m). Spawn 1.13 and `GROUND_OFFSET_FROM_ROOT_M` −1.10 unchanged
   (pivot height and leg lengths didn't move).
5. Re-save the split plan documents into
   `sim_fine_tuning/2026-08-20_1506_hardware_morphology/` (PLAN A and PLAN B as separate
   files), superseding the combined `PLAN_geometry_fix_and_wheelie_retrain.md`.

## Verification

1. `pytest tests/unit/` — generation-diff, limits, linkage envelope checks all green.
2. Checkpoint battery (existing scripts, same campaign dir): `measure_static_posture`
   (230.06 kg, level, stable, all feet capable — judged over multiple settles),
   `stance_current_probe` (root height; expect ~unchanged 1.11), `verify_crab_joint_drive`
   (18/18), `verify_crab_cam_mechanism` (hip under the ±28° hard stop at π rad/s).
3. **Record a fresh video** (zero-action settle + scripted cam sweep, no checkpoint) and
   send it for the user's visual sign-off of the geometry. **STOP POINT — push-notify;
   nothing trains until the user confirms the silhouette.**
4. After sign-off: re-run the 2k smoke train + rollout diagnosis (same protocol as
   `smoke_train.log` / `diag_policy_rollout.py`) to re-characterize the wheelie on the
   corrected robot — leg yaw inertia and drag lever both drop with the legs 0.5 m closer
   in, so the failure mode may shift. Results feed Plan B's lever decision.
5. Update RESULTS.md + project memory; commit Plan A as one unit.

---
