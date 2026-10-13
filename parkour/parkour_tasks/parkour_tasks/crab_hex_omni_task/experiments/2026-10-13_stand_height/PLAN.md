# Plan — height-commanded stand network (campaign `2026-10-13_stand_height`)

> **Audit trail (cap 3, reached):**
> - Pass 1 (2 opus·xhigh, lens split; 11 blocking): restore timeout < restore → timeout from h_min, counts as K4
>   failure; known-bads that were not bad → per-gate control table; K3 floor unreachable / in noise → ≥ 4 feet,
>   n ≥ 100, 2·SE margin, h_cmd ≥ top + 0.264; K2 timed from the command; K5 non-regression; posture-only passes
>   K1–K4 → **K6 homing + stillness per height (user)**; force cap statistic → **six-leg mean, coupled conversion,
>   knee hold-only, step-4 re-check (user)**; h_min in R_h's dead zone → hip-room rule; partial smoke → step-4
>   launcher/table/ground smoke + step-5 assertions.
> - Pass 2 (opus·xhigh + fable·high; 7 blocking): restore rows untrained / see a walk command → **restore steps are
>   stand-net steps (user)**; K6 stillness in noise → **pooled per seed, window after arrival (user)**; R_h had no
>   stand-up floor → step-4 premise stop per start type; K4 known-bad passed → off-height hand-back clause; K1/K2
>   known-bads named on the old ±1 path; ground re-arm relative to spawn height; K4 onset aligned with K2.
> - Pass 3 (opus·xhigh: none; fable·high: 1 blocking): the restore trigger was terrain-referenced and would fire at a
>   zero command on ~45–57 % of field_hard exits (checked on existing raws) → trigger and completion moved to the
>   limited command + joint space; bridge runs with the restore armed. **This last fix was not re-audited.**
> - Left for implementation (check, not edited in): clamp q(h) and the limiter to [h_min, h_max] (ground starts
>   measure 0.31 / 0.52 m); force a first stand (or restore at spawn) on random-height starts; park the table only
>   once the belly is clear; drop training top_max with any h_max re-freeze; a stand arriving mid-restore is a
>   stand→stand merge; step-5 timeout assertion relative to the R_h slice; restore zeroes current-frame cols 5, 8–10
>   only; pin K2 / K4 aggregates (p90) and K3 poses per cell, report K3 per top and homed-at-end per K3 cell; write the
>   realised clip into run_meta and update copied clip asserts; copy / wrap the closed campaign's scripts, never edit
>   them; size the K3 eval length for ground rise + ramp + settle + 3 s; height-manifest walkstop passes for K1/K2/K6
>   and a zero-command pass for K5 (+ ≈ 0.15 GPU-h per seed); K6 window re-opens after in-hold command changes; K3
>   eval commands ≥ top + 0.31; pre-register the judged checkpoint, iterations and premise window before the pilot;
>   pin campaign-local reader shas at step 4; carry the CHARTER §2 running rules (one Isaac process, no SIGKILL, Kit
>   cache check, MemoryMax, streaming recorder, no `pgrep -f` self-match); note in the CHARTER that the
>   max(1, R_h's) fall clauses compound chance misses across seeds.
> - Size: 9 steps / ~2,800 words (first review: 9 / ~2,100); growth is all blocking fixes and user decisions.

## Context

The adopted stand network (A-mask s17, `policy/stand_net/model_4000.pt`, campaign `2026-10-11_stand_network`) stands
the omni crab still at ONE height (walking height, root 1.064 m) beside the walking head S1 s18. The user wants it to
**stand at commanded heights**, set by the robot's mechanics rather than the policy's action limit (and capped by the
hip lead-screw force rating), to **raise / lower itself** from walking height while standing, and to **stand up from a
variety of starts**: legs retracted on tables of various heights, and from the ground. Before S1 takes over again the
stand network must **restore walking height**.

What the exploration established (workflow wf_fac866bf-2ea; campaign records):
- **Authority is the binding constraint.** Hip/knee actions are rod targets at default + 0.25 rad/unit; the wrapper
  clips every routed action to ±1 (`clip_actions` 1.0; S1 is saturated, so its ±1 is load-bearing). The env action
  term itself already allows ±4.8 under preset 2a (`crab_hex_env_cfg.py:272`, `:472-478`), so the wrapper is the only
  ±1 layer. Within ±1 the feet-planted root height spans only ~0.92–1.22 m; mechanically the robot reaches ~0.67 m
  (hip stop) up to ~1.5 m. Height is almost entirely the hip channel (knee correction −0.14 … +0.43 at constant reach).
- **R anchors to walking height** (DOWN/UP thresholds and HOLD/SEAT rod targets are absolute; output clamped ±1), so
  it cannot teach other heights unchanged; it is a pure function and can be wrapped campaign-locally.
- **Starts.** On the 25-in table every controller (even zero action) rose to 1.064 m by actuator slew; the stand net
  fell 2/32 from rod_min; the traces suggest (an interpretation, env-0 per-leg data only) that it spun its cams while
  still belly-down and later re-stepped from a large cam offset. The real retracted pose is **knee folded
  inward** (user). knee_folded toes sit 0.52 m below the root, so on the floor it rests on its toes; rod_min / random
  folds rest on the hip plates, where the hip-contact clause (> 500 N, a scalar on all envs) would fire at spawn.
- **Forces (static model):** hip ≈ 1.78 kN at 1.064 m (44 % of it from the knee-coupling term), ≈ 2.0 kN near 1.2 m;
  knee ≈ 1.4 kN vs its 500 N rating at every height. User (2026-10-13): **cap on the six-leg mean hip-rod force,
  knee treated as hold-only (non-backdrivable), reported, not a cap.**
- **Channel.** Hidden clock column 15 is free on the stand-net input; written after the estimator on the clone it
  leaves S1 and the estimator untouched. Its leftover backbone weights (norm 1.74) must be zeroed at warm start.

## Design decisions (recommendation each)

**H1 — Action authority: posture-offset residual (stand net only).** Executed stand action =
`q(h_cmd) + clip(a_net, ±1)`, `q(h)` = stance posture (hip, knee) in action units by campaign-local IK at constant foot
reach, cams 0, clamped to ±4.8, **anchored so q(1.064) = 0.0 exactly** (the settled sim height, not the rigid 1.056 m).
The wrapper clip is raised to 4.8 only in the stand-net harnesses (runtime `env.clip_actions = 4.8`; trainer, eval
launcher, table/ground test — no config edit), and the router **clamps S1's rows to ±1 explicitly** before the select.
Rejected: a larger hip scale (breaks the warm start); a wider clip for both nets.

**H2 — Height command and range.** Absolute root height above the floor under the robot, `h_cmd`, through a
0.08 m/s rate limiter whose state starts at the command for walk-handover starts (so a zero command stays bitwise
inert) and at the measured height for table, ground and random-height starts. Encoded into stand-net input
column 15 as `(h_cmd − 1.064)/0.4`, written after the estimate (`sn_common.stand_net_input(height=…)`); backbone
`W[:,15]` zeroed at warm start (the known-bad A-mask s17 copy is zeroed too). `h_meas` = root z minus the mean
terrain height under the six feet (= p2s body height), one definition shared by router, trainer and readers.
- `h_min` = max(kinematic floor at constant reach + 2 cm, lowest h whose hip keeps ≥ 0.125 rad (R_h's lift) to the
  −45° stop) ≈ 0.72 m;
- `h_max` = highest h whose **six-leg mean static hip-rod force** in settled holds with ≥ 5 feet loaded is ≤ 2000 N,
  from joint torques converted with the coupled linkage (solve Jᵀ·[F_h, F_k] = [τ_h, τ_k] incl. `knee_hip_coupling`);
  worst leg and knee reported. Frozen in the CHARTER at step 1, **re-checked at step 4** on R_h's measured forces in
  the top bin (over 2000 N → lower h_max and re-freeze before any training).

**H3 — Teacher: height-aware R (`R_h`), campaign-local, validated as a controller first.** Wrapper around
`restance_target` (D3 constants): hip/knee obs re-referenced to `q(h_cmd)`; HOLD / WAIT / SEAT = `q(h_cmd)`;
LIFT / HOME = `q(h_cmd)` + the D3 lift; clamp ±4.8; cams unchanged. At h = 1.064 with ±1 it must equal R bitwise.
Retracted / table / ground starts: legs not DOWN → SEAT → slew to `q(h_cmd)` (checked in the probe).

**H4 — Objective: DAgger of R_h** (per-episode β decay, 10 % R_h control slice, DART, per-leg branch weights,
LIFT-onset oversampling, reservoir), residual labels `R_h − q(h_cmd)`, warm start A-mask s17 with `W[:,15]` zeroed,
mask 1, actor backbone only. One arm, three seeds. RL fine-tuning is a later phase.

**H5 — Training starts and commands** (campaign-local env composition between `parse_env_cfg` and `gym.make`):
- per episode, seeded: **walk handover** ≈ 50 %; **table start** ≈ 25 % (one tall kinematic cuboid per env, top set
  by z placement, top ~ U[top_min, top_max] from step 1(c), under the robot's spawn xy/yaw, parked away after the first
  exit); **ground start** ≈ 15 %; **standing at a random height** ≈ 10 %;
- retracted poses: knee_folded for most, plus rod_min and random folds;
- `h_cmd` ~ U[h_min, h_max] (35 % mass at 1.064), resampled on stand→stand merges; table starts draw
  `h_cmd ≥ top + 0.264 m`;
- the first command of a table / ground episode is a stand long enough for ramp + settle + 3 s (campaign-local wrapper
  on `_resample_command`); pushes suppressed until the robot is up;
- ground starts: the hip-contact termination is wrapped (`DoneTerm.func` → per-env flag), re-armed once the belly has
  risen ≥ 0.10 m above its **spawn** height; tilt > 0.5 rad and timeouts stay.

**H6 — Restore before hand-back.** Terrain-independent (audit pass 3): when the slot leaves stand while the
**limited h_cmd** is > 2 cm from 1.064, the router keeps the stand net with `h_cmd → 1.064` through the same limiter
until the limiter has reached 1.064 **and** the hip rods are within δ of rod(q(1.064)) for 0.2 s (joint space); a
stand commanded at 1.064 never triggers it, on any terrain. Timeout = (1.064 − h_min)/0.08 + 1 s (≈ 5.4 s); a
timeout hands back anyway. The `h_meas`-within-2-cm check at hand-back is a reported flat-ground (walkstop) statistic. **Restore steps are stand-network steps everywhere (user, 2026-10-13):**
the stand net's input shows a stand (command columns 5 and 8–10 zeroed, column 15 from the limited h_cmd), and the
trainer treats them like stand rows (same per-episode executor, R_h labels, reservoir). Same rule in trainer router and
eval launcher. A deployment keys on an explicit stand command (latched-slot finding).

**H7 — Pre-registered closed-loop gates** (es1 and es2; adoption-grade = all gates, three seeds; adoption is the
user's). Statistics:
- **K1 tracking:** per height bin (h_min … h_max, ≥ 5 bins), median |h_meas − h_cmd| over the last 2 s of holds
  (from command arrival + 1 s) ≤ 2 cm, p90 ≤ 4 cm, flat ground.
- **K2 raise / lower:** time to within 2 cm ≤ Δh/0.08 + 1 s from the command change; falls ≤ max(1, R_h's) per seed.
- **K3 stand-up:** table (≥ 3 tops incl. 0.635 m) and ground starts, eval commands h_cmd ≥ top + 0.30 m, n ≥ 100
  per cell: STOOD = belly risen ≥ 0.10 m above its spawn height, ≥ 4 feet loaded on ≥ 95 % of the last 3 s (5/6-down
  reported), |h − h_cmd| ≤ 4 cm, no fall; read **per start type** (tables pooled over tops; ground) per env seed:
  rate ≥ R_h's − max(5 pp, 2·SE of the difference); falls ≤ max(1, R_h's) per 100.
- **K4 restore + hand-back:** hand-backs that miss the H6 joint-space completion (timeouts included) ≤ max(1, R_h's)
  per seed; flat-ground hand-back |h_meas − 1.064| reported;
  exit falls counted from hand-back ≤ max(1, R_h's) per seed; walking onset ≤ REFS + Δh/0.08 + 1.5 s.
- **K5 non-regression at 1.064:** G1–G5 of the stand campaign, unchanged lines.
- **K6 homing + stillness per height (user, 2026-10-13):** per height bin, walk-handover entries: homed and settled
  share ≥ R_h's − max(5 pp, 2·SE). Stillness (user, 2026-10-13): post-settle steps / taps **pooled over all height bins
  per env seed** ≤ the G2 lines (0.5 / 1.0 per s), every bin reported; the stillness window starts only once the height
  has arrived (|h − h_cmd| ≤ 2 cm) and the cams have settled.
- Reported: forces per bin (H2), six-down, hip yaw, observe pass.

Controls (calibrated at step 4, before training):

| gate | known-good (must pass) | known-bad (must fail) | also reported |
|---|---|---|---|
| K1, K2 | R_h | R on the old ±1 path (no q); A-mask s17 (W15 zeroed) on the old ±1 path (no q) | posture map only (zero residual); A-mask s17 + q |
| K3 | R_h | S1 alone (marches) | zero residual, A-mask s17 |
| K4 | R_h | "hand back at height, no restore" (fails the H6 completion clause) | A-mask s17 |
| K5 | R_h, A-mask s17 (existing stamps) | — (non-regression) | — |
| K6 | R_h | posture map only (zero residual: no homing) | A-mask s17 |

## Steps

0. Campaign dir `experiments/2026-10-13_stand_height/` (CHARTER = pre-registration + log, PLAN = this plan); commit
   those paths only.
   → sonnet · medium — transcription
1. **Facts and premises (CPU + existing raws):** (a) IK `q(h)` anchored at q(1.064) = 0, h_min by the rule in H2,
   re-step clearance per height; (b) measured joint torques at 1.064 m from the eval stamps' `applied_torque`, coupled
   conversion to rod forces, model-vs-measured tolerance ≤ 10 %, then h_max; (c) table range `top_min … top_max`
   (legs clear, toes above the floor at spawn, h_cmd ≥ top + 0.264 m reachable) and ground-start geometry;
   (d) `facts.json`. **STOP and ask** if the model misses the measured torques by > 10 %, if h_max < 1.10 m, or if the
   knee-coupling sign is ambiguous.
   → opus · high — numbers every later gate rests on
   ✓ fable · high — independent check of the force / IK premises
2. **Build** (campaign-local; reuse `sn_common`, `sn_dagger*`, `sn_stand_net_eval_crab_hex_gait.py`, `sn_table_core`
   / `sn_table_test`, `sn_calib`, `sn_observe_read`): q-IK and `R_h`; router (S1 clamp ±1, stand residual, restore);
   `stand_net_input(height=…)`; trainer start modes, commands, limiter, push suppression, hip-contact wrapper; launcher
   flag deriving per-hold heights from a campaign-local schedule manifest (harness and scenario files untouched);
   table/ground test with table heights, ground starts and commanded heights; K1–K6 readers (raws record `a_net` and
   `q(h_cmd)` separately); CPU tests incl. bitwise identities (R_h at 1.064 ≡ R; zeroed-W net with a **nonzero**
   command at q = 0 vs unzeroed differs only via col 15; S1 rows unchanged by the clip path).
   → opus · xhigh — new routing / action semantics; a wrong clip or offset trains quietly on the wrong thing
3. Code review of step 2 against this plan.
   → opus · xhigh — fresh context
   ✓ fable · high — independent check
4. **Bridges, probe and calibration (≈ 2.5 GPU-h):** first a ≈ 0.1 GPU-h launcher + table/ground smoke asserting
   realised doses (per-hold h_cmd reaches q and R_h, hip targets beyond ±1 on stand rows, restores complete, spawned
   table tops, ground spawns survive); identity bridge (S1 as the stand ckpt, zero command, new clip path, **restore
   armed** → bitwise = REFS on suite / walkstop / field_hard es1, which is what lets K5 reuse the existing stamps); **R_h probe** over the height, raise/lower, table and ground
   schedules; K1–K6 read on every control in the table (K5 reuses the existing R / A-mask stamps); h_max re-check
   (re-bin and drop table tops above the new h_max − 0.30 m before training). **STOP and ask** if R_h fails any gate,
   a known-bad passes the gate it must fail, or **R_h's own stand-up floor** misses in any start type (each table top,
   ground knee_folded, ground rod_min): STOOD ≥ 0.90 and falls ≤ 1 per 100.
   → main (opus) — runs plus the premise decision
5. **Smoke** (≈ 0.3 GPU-h), ~100 its asserting: start-mode shares, table tops, ground spawns, h_cmd distribution and
   limiter, residual labels `|R_h − q| ≤ 1`, restores complete (0 timeouts), table first stand ≥ ramp + settle + 3 s
   with h_cmd ≥ top + 0.264 m, table parked at the slot change, pushes off until up, ground episodes last ≥ ramp +
   settle + 3 s with 0 hip-contact terminations before re-arm, restore rows carry zeroed command columns and enter the
   reservoir with R_h labels, S1 rows at ±1, realised hip targets < −1 unit on stand rows, trainer↔launcher forward
   parity.
   → sonnet · high — assertions fail loudly
6. **Pilot s18** (one job, ≤ 2.5 GPU-h by the smoke's s/it). Mid-run premise check vs the concurrent R_h slice:
   homed share and |h − h_cmd| for walk-handover / height holds, STOOD share for table / ground starts, restore
   timeouts on student episodes; **STOP and ask** if the student is < 0.5× R_h on homed or STOOD, its fall rate
   > 2× R_h's + 1 %, or its restore-timeout share > 2× R_h's + 1 %. Then evaluate K1–K6 + the
   table/ground test; decide-record-notify; **STOP and ask** before replicates if any gate fails.
   → main (opus) — monitoring and the stop decision
7. Replicates s16 / s17, same config and evals.
   → main (opus) — chained queue
8. Report (per-seed K1–K6 vs the controls, force table per height, videos via the streaming recorder: a raise/lower
   stand, table and ground get-ups), commit, **STOP for the user's adoption decision**.
   → opus · xhigh — the synthesis the user decides on
   ✓ fable · high — independent check

**Stop points (push-notify each):** step 1 premise stop; step 4 calibration / R_h failure / h_max re-freeze; step 6
premise stop or gate failure; step 8 adoption.

## Budget, seeds

≈ 15 GPU-h: probe + calibration 2.5, smoke 0.3, three training runs × ≤ 2.5, evals 3 × ≈ 1.5. Seeds 18 (pilot), 16,
17; env seeds es1 tune / es2 held out.

## Verification

- CPU identities (step 2); GPU identity bridge bitwise = REFS before any training; smokes assert every mode and dose.
- Gates calibrated on the control table before training; closed-loop reads only; observe pass per head.
- code_sha / eval_sha unchanged (all code campaign-local); a shared-code change is a stop-and-ask.

## Later phases (not run)

- PPO fine-tune past R_h (heading, six-down, trim); hardware rod-force modelling in the actuator layer; S1 conditioned
  on height (hand-back without restore); export of the S1 + stand-net pair.

Routing: strong · roster ok (opus = Opus 5.5) · planned on opus · ceiling: opus · 1× sonnet·medium, 1× sonnet·high, 1× opus·high, 3× opus·xhigh, 3× main, 3× fable·high check, fable retry on 2 fails
Effort enforcement: prompt-level
