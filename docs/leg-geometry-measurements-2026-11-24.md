# Leg geometry measurements, 2026-11-24

These were measured on the real robot, on its stand, mostly on the front-left (FL) leg, with front-right (FR) checks. Use them to bring the sim leg model (`crab_hex_dimensions.py`, `crab_hex_linkage.py`, the generated `assets/crab.usda`) in line with the hardware.

**Status: provisional.** These are ruler measurements, good to roughly ±1 in, plus a model fit. Re-measure anything a decision hangs on; see "Open questions" below.

The fitted model and its raw output are in `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-11-24_leg_geometry_fit/`.

## Actuators

- **The hips and knees now use the same actuator motor.** The hips were switched to the smaller motor used on the knees, so the hip actuator spec in `crab_hex_dimensions.py` (SLA08 2000 N) is out of date. Hip force and speed limits in the sim need re-checking.
- **Scale: 5.33 hall counts per mm of rod on both actuators**, i.e. 0.188 mm per count. The firmware counts both edges of hall A, with direction from hall B.
  - Knee: 6 7/8 in down to 1 1/8 in of exposed rod over 779 counts, i.e. 146.0 mm.
  - Hip: 1 1/8 in to about 4 in over 403 counts, i.e. about 5.5 counts/mm; this was a rough reading.
  - The datasheet scales (16 and 14.1 counts/mm) do not match.
- **Exposed rod length** (end face of the actuator body to the end of the rod; the pin centre is about 0.5 in further in): 1 1/8 in (28.6 mm) at full retraction on both actuators.
- **Hip stroke:** count 0 to count 1069, switch to switch, about 200 mm.
  - Count 0 = rod fully retracted = curl end (the leg is folded up), a limit switch.
  - Count 1069 = rod fully extended = uncurl end, a limit switch. FR measured 1070.
- **Knee travel:** count 0 to count 779-790, i.e. 146 mm.
  - Count 0 = curl end = **the leg's own mechanical stop**, with the rod extended (6 7/8 in exposed, about 4 mm short of the actuator's full extension).
  - Count 779-790 = rod fully retracted = a limit switch.
- **Signatures at the ends**, from PWM 80/120/150 runs:
  - At a limit switch, the count stops dead and the motor current *drops*. Running current is 10-17 (raw units, about 7 at no current); the switch-open level is 7-11, depending on the joint.
  - At the knee's mechanical curl stop, the current *rises* 30-80%, the speed collapses over about 0.5 s, and harder pushes end deeper, by up to about 20 counts.
- **The knee actuator's rear end rides on the FEMUR.** It moves with the upper leg when the hip lifts, so the knee angle depends only on the knee rod. `crab_hex_linkage.py` anchored it on the hip plate (coupled); that was wrong. The change on this branch makes it femur-mounted.

## Stand (calibration) geometry

- The robot rests on its belly on a table **20.5 in wide** (across, between the left and right leg rows) and **34 in tall**, longer than the robot front to back. The robot is roughly centred.
- The body (28 x 48 in) overhangs the table by about **13.75 in** on each leg side. The legs hang below the belly beside the table.

## Measured FL toe positions

Measured with a ruler; cam yaw unknown but fixed:
- **side** = horizontal distance from the body's side wall (+ outward, - inward under the body);
- **height** = distance below the belly plane (- below).

| Pose | Hip counts | Knee counts | Side (in) | Height (in) |
|---|---|---|---|---|
| A | 0 | 0 | +5 | -10 |
| B | 0 | 157 | +16 | -13 |
| C | 475 | 157 | +5 | -19 |
| D | 1069 | 157 | -16.5 (16-17) | -6 |

Pose A is suspect: the stated link lengths can't bring the toe that close to the femur pivot. See "Open questions".

## Contact and clearance facts

- **Knee at its curl end (0-24 counts):** uncurling the hip makes the folded tibia press the belly at hip **941-1003 counts**.
  - The run with the knee curled about 24 counts tighter contacted *later* (1003). Physics says it should be earlier, so the hip count reference may have shifted between those runs.
- **Knee held at 157-175 counts out:** the hip swept 0 to 1069 with **no belly contact** on FL and FR, and Nick saw "lots" of clearance.
  - At full hip extension the toe was about 16.5 in under the body and 6 in below the belly. That is inside the table's footprint below its top, so the sweep only cleared because this table is hollow there.
- **With today's supply, PWM 80 is too weak near some ends.** The hip crawls near count 1030-1040, and the knees stall mid-stroke. PWM 128 reaches every end. **FRHL pulling back from 1070 at 128** with the knee 175 out stalled at 878; pulling back at 150 worked.

## Model fit, FL

The fit used the repo body, femur pivot (2.5 in outboard of the wall, 3.25 in below the belly), femur 23.0 in, tibia 32.5 in and CAD outlines. It fitted the hip linkage, the knee linkage (on the femur), the leg-plane yaw and the toe point to poses A-D plus the contact facts.

**Hip angle against counts** (degrees below horizontal; negative = raised):

| Hip counts | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1069 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Angle | -68.0 | -55.4 | -44.7 | -34.7 | -25.1 | -15.5 | -5.7 | +4.7 | +16.1 | +29.2 | +46.7 | +70.7 |

- The repo assumed about 45 deg up to 60 deg down; the fit says **68 deg up to 71 deg down**.
- Near full extension the hip linkage is close to dead centre (linkage angle 172.7 deg at 1069). The angle changes about 0.5 deg per count there, so this end is the least certain.
- Fitted hip linkage:
  - femur pin radius 4.43 in, offset 11 deg below the femur centreline;
  - plate anchor (-0.33, +20.73) in from the pivot;
  - pin-to-pin length 438 mm at count 0 and 638 mm at 1069.

**Knee interior angle (femur to tibia) against counts:**

| Knee counts | 0 | 100 | 157 | 200 | 300 | 400 | 500 | 600 | 700 | 779 |
|---|---|---|---|---|---|---|---|---|---|---|
| Interior angle (deg) | 8.0 | 24.5 | 32.6 | 38.4 | 51.3 | 63.8 | 76.5 | 90.3 | 107.2 | 133.7 |

- Only counts 0 and 157 are pinned by data; above about 175 counts the angles come mostly from the priors.
- Fitted knee linkage: lever pin radius 3.40 in at phase +41.4 deg; anchor distance about 18.3 in (unconstrained); pin-to-pin length 525 mm at count 0 and 379 mm at 779.

**Fit quality** (model minus ruler, inches):
- A: (-1.32, -2.38)
- B: (+0.92, +0.29)
- C: (-0.46, -1.46)
- D: (+1.93, +0.28)

RMS is 1.35 in per coordinate. D's side offset matters most: the model puts the toe at full hip extension about 2 in *less* inboard than measured.

**Predictions:**
- With the knee at 0, first belly contact comes at hip about 980.
- With the knee at 157, the minimum belly gap over the full sweep is about 4.6 in.
- With the knee at 175, it is about 5.7 in.

## Open questions: measure rather than guess

1. **Pose A.** Re-measure it (hip 0, knee 0). The ruler reading is about 2.5 in closer to the pivot than the stated links allow.
2. **The femur pivot's height below the table top** (belly plane), and its offset outboard of the side wall, measured directly.
3. **The toe at full hip extension** (hip 1069) with the knee at 157 and at about 300 counts. That is where the model is least certain and the table edge matters.
4. **Knee angle at several counts** (e.g. 300, 500, 779), measured with an angle gauge or a phone inclinometer. This pins the knee linkage above 175 counts.
5. **The actuator anchor and pin positions** on the hip plate, femur and tibia, measured with a ruler. That would replace the fitted values.
6. **The hip and knee actuators' force and speed** with the new motor, if the sim's actuator model needs them.

Measurements happen on the robot laptop with Nick present. Request them through the laptop session; see the multi-machine-builds skill.
