# Calibration fix round: brief and inputs (2026-11-24)

Base: `feat/hall-calibration-end-detection` @ b11cc9ef (hall-encoder calibration plus the current-based end detector).
Job branch: `feat/hall-calibration-end-detection-fixround`.
Everything in this folder was measured or produced on the robot laptop on 2026-11-23/24, with Nick at the robot.

**Hard rules:**
- Never flash and never open a serial port; the robot stays on the laptop.
- Keep `firmware/arduino/arduino.ino` (sha1 31a15639…) and `firmware/arduino/src/telemetry.cpp` (sha1 d59aa784…) byte-identical.

## 1. Must fix (from two independent reviews and a robot test)

1. **V1/R1 (high): a hip that stalls mid-stroke gets homed on a rerun or a short drive.**
   - Every retract home (Switch, plain Still, nudge return) must show the limit-switch signature: the current flat and at the switch-open level.
   - Measured on FL today: the running current is 10-17 raw units, and every hip and knee switch reads 7-8 after the count stops. The PWM-80 mid-stroke stall at hip 841 held at 11, with no drop.
   - An earlier FRHL switch read 11; a stall reads 17-31. Use a per-joint ceiling, or a relative "flat and low against the first full-PWM samples" test.
   - Keep the last proven home through `fail()`, so a rerun that ends far from it fails instead of re-homing.
   - In SETUP.md, never rerun a stuck hip at 128: jog it home or use 150.
   - Evidence: `findings/independent-review-of-fixed-tree.json` (V1, R1). Also `findings/endtest-fl-replay.json`: on today's trace, a rerun at 80 from the stall point homes the hip at 848, and an unhomed start within about 2-66 edges of the stall is homed there.
2. **R3 (high): Curl only can home a knee resting against the belly when its hip starts extended.**
   - Make SEEK per leg: knee curl, hip curl, then knee curl again. The second knee curl is proven and re-homes.
   - Or, at the least, warn.
3. **R2: the hip's EXTEND end is a limit switch.** Today it showed the switch signature 3 times out of 3: current 11-12 dropping to 7-8 at 1070.
   - The STOP rule only catches it through a 30-40% current bump just before the switch opens, which is fragile: a 5-15% smaller bump gives Stuck.
   - Handle it explicitly with the switch rule.
4. **V3 (medium): two PWM-120 switch ends sit one sample from Stuck.**
   - Don't let Stuck fire while a switch drop is pending or the current is still falling.
   - Measure arrival speed over the last few edges.
5. **Lows:**
   - R4: cap the hip nudge PWM.
   - R5/V4/V2: correct the "never taken as an end" claims in SETUP.md, the GLOSSARY, the GUI help and the header comments.

## 2. Sequence change: table-safe calibration (FL geometry measured today)

Firmware frame: hip limits are [0, N]; knee limits are [-S, 0]. Knee counts below are counts uncurled from the curl stop, so in firmware they are **negative**.

- **Knee first, at hip 0:**
  - curl to its mechanical stop;
  - uncurl to its switch (785 out, firmware -785);
  - park at **K = 264 out (firmware -264), allowed band 244-280**.
- **Hip:**
  - Home on its curl switch (0).
  - Sweep **by count to H = 840** with the knee held at K. The model's worst case allows 864, but a robot check at 866 put the toe about 5 in from the table side against about 8 in modelled, so 840 keeps margin.
  - Any stall before H is an obstacle: retract to the curl switch, with no automatic retry.
- **Return:**
  - The hip goes back to its curl switch with the knee still at K. Check the re-home: the count should be about 0.
  - Move the knee only once the hip is home (or below 560).
- **Hip extend limit = home + 1069 counts.**
  - That's switch to switch, measured 3 times out of 3 today, and it matches the 203.2 mm label at 5.26 counts/mm.
  - Soft limit about 10 counts short.
  - Mark it "derived, not switch-verified".
  - Flag lost counts if the extend switch trips below about 1049.
  - **Never seek the extend switch on a solid stand.** At 1069, no knee setting keeps the leg clear of the belly, the table and the floor together.
- **Optional, off by default: a "hollow stand" full sweep** to the extend switch at knee about 192 (band 157-232). It lets the detector verify the extend switch, and needs item 3 above.
- **Other legs:** only FL is modelled. Use FL's values for all six legs for now, flagged per leg.

## 3. Owner decisions

These are Nick's accepted recommendations from 2026-11-24:

- **F3:** keep the knee nudge cap at 128. The semantics review found the blind knee nudge unfolds away from the body.
- **F4:** raise CAL_PWM_MIN to about 110. PWM 80 stalls mid-stroke with no signature on today's supply.
- **F5:** the 3000 B RAM floor applies to the compile-time figure. The figure after the setup heap was already under 3000 before the detector.

## 4. Gates and acceptance

- **Test gates:** Docker `all` (native, coverage at least 90/88, py) and AVR build-only for Rev 3 and Rev 2.
- **New replay goldens**, from `traces/clearance-FL-173941.log.gz` (today's FL end test):
  - knee switch ×2;
  - knee stop ×2, plus one short drive;
  - hip curl switch ×4, including a short 109-edge drive;
  - **hip extend switch ×3, as Switch**;
  - **the PWM-80 stall at 841: Stuck, and no later-start cut may home**.
- **V1 goldens:** add the FRHL 878 stall's later-start cuts.
- **Review:** an independent review of the final tree, with a robot-readiness verdict, before the merge back. Merging into `feat/hall-calibration-end-detection` needs Nick's OK.
- **Deliver:**
  - the job branch;
  - the gate results;
  - the review verdict;
  - a build-only hex tagged `hallcal-1125`;
  - any decisions still open.

## 5. Inputs in this folder

- `traces/`: every robot run from 2026-11-23/24, gzipped. These are the FRONT-board telemetry and host commands, with the LEFT/RIGHT board lines removed. Telemetry segment: `name pos pot current enL enR pwmL pwmR hallEdges state`, at about 20 Hz, integer current. Parse with `tools/trace-replay/parse.py`.
- `findings/`:
  - `build-review-and-fix.json`: the first build's review findings F1-F8 and its fix report;
  - `design-spec.json`: the end detector's design;
  - `independent-review-of-fixed-tree.json`: V1-V4 and R1-R5, with verifier verdicts and the synthesis;
  - `endtest-fl-replay.json`: today's robot test replayed through the detector and LegCalibration;
  - `table-safe-schedule.json`: the schedule and its independent check.
- `tools/`:
  - the reviewers' replay harnesses (`trace-replay/`, `robot-run/`, including `legsim.cpp`);
  - the fix stage's Python reference, `ced-fix/detector.py`.
- `schedule/`:
  - the geom3 schedule results (`schedule_result.json`, `envelope.json`, …);
  - `leg_model_v3.py`;
  - the measured inputs (`geom3_inputs.md`, `k1_knee_angles.md`).
