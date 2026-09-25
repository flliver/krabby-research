# move_turn failure mechanism — W (wave 2) and the wave-1 clock-free arms (2026-09-25)

Copied from the CPU analysis at `parkour/outputs/omni_plan/clock_deprecation/w2_readout/mechanism.md` (scripts and the per-episode table `mt_fail_rows.json` stay there, untracked). Recorded as the follow-up on the baked head (`crab_hex_omni_task/policy/manifest.yaml` `followups:`).

**Why W fails move_turn: the same forward nose-dive as the wave-1 clock-free arms, and it builds up during the hold rather than hitting at the start**

Everything below was computed on CPU from the raw suite npz files; nothing was edited and Isaac was not started. The six eval stamps used:
- W s3 c6x: `…/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29`
- X0x s3 c6x: `…_w2/seed3_X01_c6x_A15pB/…/2026-09-25_17-01-55`
- A4500 c6: `…/2026-09-23_clock_deprecation/seed3_X04_c6_A15pB/…/2026-09-24_16-39-45`
- A3000 c4: `…/seed3_X08_c4_A15pB/…/2026-09-25_13-01-06`
- X0 s3 c6: `…/seed3_X01_c6_A15pB/…/2026-09-24_00-55-05`
- E10 (X0 s3 c6, clock hidden): `…/step0_E10_X0s3-c6_hidden/…/2026-09-25_15-07-17`

Each arm has n=200 move_turn episodes (100 `move_turn`, 100 `move_turn_r`) and n=200 turn episodes. My fall counts reproduce the ledger survivals exactly: 127, 27, 82, 162, 51 and 46 falls give 0.365, 0.865, 0.59, 0.19, 0.745 and 0.77.

How the eval defines things (from `eval_crab_hex_gait.py`, `gait_eval/report.py` and `schedule.py`):
- Each episode is a 1.5 s prologue, a 2 s settle, then a 12 s hold starting at step 175 (dt 0.02).
- move_turn commands world travel +x at 0.22 m/s with facing +90° (`_r` is the mirror, −90°). The robot spawns facing 0, so the yaw command saturates at 0.3 rad/s.
- The npz files record the state before each step. For a fall, the last row is the step just before termination.
- A fall is `crab_failure`: tilt over 0.5 rad or hip contact over 500 N (W `params/env.yaml`).
- Recorded actions are the policy output before clipping; the runner clips at `clip_actions=1.0`.

**1. What the failure is** (all six arms; `mt_fail_summary.py`, `mt_fail_timecourse.py`)
- **Tip-over, not hip contact.** Every move_turn fall terminates on tilt: the last recorded tilt is above 0.45 in 100% of falls (W 127/127, X0x 27/27, A4500 82/82, A3000 162/162, X0 51/51, E10 46/46). The largest hip-link force in the last second is 0 N in every fall.
- **Nose-down.** The harness classifies 125 of W's 127 falls as `pitch_fwd` (2 roll), and the other arms are similar (A4500 73/82 with 8 roll, X0x 27/27). Roll just before the fall is small (median |roll| 0.05) and leans slightly toward the sideways travel direction: roll positive in 49/65 `move_turn` falls vs 16/62 `move_turn_r`.
- **Both hands fail equally.** W has 65 falls on the left-hand hold and 62 on the right; A4500 37 vs 45; A3000 84 vs 78.
- **The same pre-fall sequence in every arm, and in straight-forward falls too.** Median over W's move_turn falls, in bins from 3 s before the fall:

| Signal | −3 to −2 s | −2 to −1 s | −1 to −0.5 s | last 0.5 s |
|---|---|---|---|---|
| Pitch (rad) | 0.13 | 0.16 | 0.22 | 0.36 |
| Share of steps with both front feet in the air | 0.32 | 0.58 | 1.00 | 1.00 |
| Middle-leg contact share | 0.61 | 0.60 | 0.55 | 0.24 |
| Feet in contact | 3.06 | 2.86 | 2.80 | 2.30 |
| Forward body speed (m/s) | 0.06 | 0.07 | 0.19 | 0.35 |

  In the last second the front feet (FL, FR) are down only 7% and 4% of the time, against 90% and 76% for the rear feet (RL, RR). The body then surges forward as it tips. X0x, A4500, A3000, X0 and E10 show the same numbers, and so do the straight-forward (fwd) falls in each arm (W n=5, A3000 n=26, E10 n=32).
- **Ruled out** (all read the same in every arm and for fallers vs survivors):
  - *Legs crossing:* the closest same-side neighbouring feet stay about 0.07 m apart (0.10 m in A3000).
  - *Foot slip:* sliding speed of feet in contact is 0.12–0.16 m/s.
  - *Body sagging:* height stays at 1.06–1.14 m and rises in the last step as the body pivots over the rear feet.
  - *Action saturation:* 97% of policy outputs sit at or beyond the ±1 clip in every arm. That is a standing property of all these policies, not a failure signal.
- **Nobody actually turns on move_turn.** Measured yaw rate is 0.02–0.07 rad/s against a command of 0.3. Remaining heading error at the end of the hold is 1.1–1.5 rad (`summarize` `heading_final`: W 1.134, X0x 1.095). So in practice this hold is 12 s of walking with a large yaw error that never closes, in every arm.

**2. When it fails: the risk keeps growing through the hold, not a burst at the start** (`mt_fail_hazard.py`; falls / episodes still standing, per 2 s of hold)

| Arm | 0–2 s | 2–4 s | 4–6 s | 6–8 s | 8–10 s | 10–12 s |
|---|---|---|---|---|---|---|
| W | 0/200 | 15/200 | 38/185 | 34/147 | 20/113 | 20/93 |
| A3000 | 0/200 | 17/200 | 49/183 | 45/134 | 30/89 | 21/59 |
| A4500 | 0/200 | 4/200 | 19/196 | 18/177 | 20/159 | 21/139 |
| X0x | 1/200 | 13/199 | 4/186 | 2/182 | 4/180 | 3/176 |
| X0 | 3/200 | 12/197 | 15/185 | 5/170 | 6/165 | 10/159 |

- For the three arms trained without the clock (W, A3000, A4500), the fall rate stays at 10–36% per 2 s from 2 s onward. X0x has a small burst at the start of the turn and then about 1–2% per 2 s.
- **Drift in the robots still standing** (medians over episodes not yet falling, last 1.5 s before any fall excluded), from the first 2 s to the last:

| Arm | Pitch (rad) | Both front feet in the air | Forward speed (m/s) |
|---|---|---|---|
| W | 0.046 → 0.174 | 0.09 → 0.42 | 0.11 → 0.04 |
| A3000 | 0.043 → 0.178 | 0.10 → 0.43 | not tabulated |
| A4500 | 0.035 → 0.133 | not tabulated | not tabulated |
| X0x | 0.046 → 0.093 | 0.16 → 0.21 | not tabulated |

  W's pitch and front-air share roughly double X0x's. The robot drifts into the fall.
- **Other holds.** Time-averaged pitch over the fwd hold is low and about the same for W and X0x (0.039 vs 0.033), and so is the front-air share (0.14 vs 0.15). Fwd survival is W 0.95 vs X0x 0.84, and turn survival 0.99 vs 0.97. The problem is specific to walking and turning at once.

**3. How W walks on move_turn, including the survivors** (`tools/run_phases.summarize(...)["cat"]`, n=200 per arm)

| Metric | W | X0x | A4500 | A3000 |
|---|---|---|---|---|
| Tripod score | 0.401 | 0.541 | 0.489 | 0.458 |
| Steps per second | 3.16 | 3.42 | 3.00 | — |
| Gait frequency (Hz) | 0.439 | 0.494 | — | — |
| Cadence ratio | 0.879 | 0.989 | — | — |
| Feet in contact | 3.03 | 3.105 | — | — |
| Speed (m/s) | 0.097 | 0.115 | — | — |
| Speed-tracking ratio | 0.442 | 0.521 | — | — |
| Facing closed | 0.287 | 0.298 | — | — |

- Survivors only (`mt_fail_summary.py`): W pitch 0.12 vs X0x 0.07, time with 2 or fewer feet down 0.18–0.21 vs 0.14.
- On fwd W's gait is as good or better: tripod 0.613 vs 0.581, speed 0.196 vs 0.18. So W has a different, looser gait only when translating and turning together. The front pair loses its alternation (both front feet up together more often).
- **Caveat on E10.** Its 0.77 move_turn survival came from barely walking: tracking ratio 0.168, speed 0.037 m/s, tripod 0.167, a rapid tapping shuffle (step rate 3.9 Hz). The front feet stayed down (both-in-air 0.15). "Move_turn was fine at the start of W" therefore overstates it. W's training got translation back (ratio 0.44), and the nose-dive came back with it.

**4. What the training logs show** (tfevents, 4000 iterations each run)
- W's training fall rate jumped after the cut (0.40 over iterations 0–100, 0.58 over 100–500), then recovered to 0.42 at 1500–2000 and 0.40 over the last 500. X0x s3 ended at 0.38 and X0x s2 at 0.47.
- move_turn made up only 13% of training time in W (`move_turn_frac_actual`, last 500 iterations: 0.133; X0x 0.145). A fragility limited to move_turn would not show up in the overall training numbers.
- `penalty_base_pitch_forward_linear` has weight 0.0 in c6x (W `params/env.yaml`; it only applies when the forward speed command is at least 0.12). The W and X0x configs differ only in the clock and apex weights (0.001 vs 1.0) and `clock_hide`. Nothing else in the reward pushes back on nose-down pitch.

**Most likely mechanism (inferred)**
- Under a sustained, unmet yaw command while translating, a policy trained without the clock gradually lets its front pair swing together. The nose pitches forward, both front feet end up in the air, the middle legs unload, and the body tips forward over the rear feet.
- The clock's alternating stance schedule seems to have been what held the front pair out of phase: FL and FR belong to opposite tripods. I have not tested this directly. The support for it is that X0x keeps the front-air share near 0.2 with flat pitch, and the fail rate scales with how much training happened without the clock: A3000 (cut at c4) is worst, then W, then A4500.
- W does not fail in a new way. It is the wave-1 clock-free failure, a little more pronounced than A4500's.

**Which lever fits**
- **A ramp: unlikely to help much on its own.** The evidence points away from a shock: the fall rate isn't concentrated early, and the deficit remains after the training fall rate recovered.
- **Reward terms: most direct.** Two candidates:
  - Turn `penalty_base_pitch_forward_linear` back on (weight is 0 now), and check it is active on move-turn commands. In the eval the body-frame forward command is about 0.2, above its 0.12 gate.
  - Add a front-pair support term that penalises both front feet (or both of a same-side pair) being unloaded at once, directly targeting the pre-fall state.
- **More move_turn weight or a targeted curriculum: second, and complementary.** The failing state (large heading error with a saturated yaw command while translating) looks rare in training. The overall training `facing_err_abs` is only about 0.08–0.10 rad. That is inferred from an average over all commands; I did not break it down by command type.
- **Before spending GPU:** replay the saved healthy and degenerate npz offline against both candidate terms, per the offline-reward-replay gate. The W and X0x npz above already contain matching healthy and failing move_turn episodes.

Scripts and the per-episode table (`mt_fail_rows.json`) are in `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/clock_deprecation/w2_readout/`:
- mt_fail_extract.py
- mt_fail_extract_lib.py
- mt_fail_rows.json
- mt_fail_summary.py
- mt_fail_timecourse.py
- mt_fail_hazard.py