# Fault-injection sweep of the omni policy of record — PLAN (approved 2026-11-15)

> **Status:** APPROVED 2026-11-15 as recommended, after a 3-pass audit (blocking 8 → 5 → 1; the last fix, §3.1 depth
> stage, not re-audited). Decisions in §8. Nothing implemented, run or committed yet. Branch `feat/fault-injection-sweep`,
> worktree `.claude/worktrees/fault-sweep`, cut from origin/main **78b8f9db** (the PR #14 rough-terrain merge).

## 1. Question

How does the omni pairing of record behave when a sensor, the compute and comms path, or one actuator fails
**part-way through an episode**? It is tested by an eval-only sweep with no training.

- **Subject:** the depth student **C1S** = `RT_ST_s19/model_34499.pt` (sha `6b48ef33`, MAIN
  `parkour/logs/rsl_rl/crab_hex_omni_student/2026-11-12_15-08-00_RT_ST_s19/`). Its ONNX export is
  `policy/rt_12_student/` (parity 1.16e-6, `2026-11-10_rough_terrain/p2/onnx_PICK.json`).
- **Report-only second subject:** the walker **c1** (`policy/rt_11_p3b_rough/model_29500.pt`, `6d1f29be`). It runs
  only on the actuator and comms faults, which act on the plant or the action, not on the student's inputs.
- **Outcome:** a per-fault verdict (§4.4) plus a ranked list sorted into three classes:
  1. what goes into the second distill's corruption riders (REPORT rod-speed DR §9 rung 2: depth dropout and
     blackout, obs noise, current noise, and the vxy decision);
  2. what needs a hardware-side mitigation (a velocity-estimator health flag, a watchdog behaviour, a sit-down
     fallback);
  3. what would need a trained fault regime (for example walking on five legs).

**Why now.**
- The 2026-11-07 sweep (`2026-11-06_dr_stress_sweep/REPORT.md`) held each fault at a constant level from step 0. That
  measures tolerance to a steady miscalibration. It does not measure a fault that starts while the robot is
  walking, which is how failures happen on hardware.
- That sweep tested F19's student, not C1S.
- Its two worst findings were a zeroed vxy slot (survival 0) and zeroed depth hiding a lat_r failure. Neither has
  been re-read on the rough-terrain student, whose depth has a real job on ledge_down.

**Not in scope:** training or distilling anything, loose ground (P5), the stand networks (only the student's own
stand slot runs), combined faults (one fault per cell, as before), and the real robot.

## 2. Ground rules

- **One fault per cell.** Every knob is default-off and set by no preset. With every knob unset, the cfg dump is
  byte-identical to 78b8f9db's, as in the stress-knob contract.
- **Paired episodes.** The eval is deterministic and paired on `env_index`. Before its onset step a faulted episode
  must equal the same-class nominal episode bitwise. This is the per-env dose check (§4.3), and it gives every
  post-fault metric a same-episode reference. It holds strictly on suite wave 0 and on field_light only: waves 1–9
  start from the previous wave's end state, and even a same-class payload pair differs at step 0 in 11–27 envs per
  wave (probe, C1S P2SEL L000 vs L090). On waves 1–9 the pre-onset divergences are counted and reported, and those
  waves are read on net counts (ΔF per wave), not on per-episode pairing.
- **Process-state classes** ([[eval-process-state-sensitivity]]): ±0.04 completion, 0.05–0.07 at stressed cells.
  - Fall shares move less than completion: across 4 cross-class C1S pairs on stressed rows the net fall count moved
    by ≤ 2 per 100 (while 10–19 episodes flipped individually), and nominal C1S suite falls are 0/1000 in all 6
    stamps. ΔF thresholds (§4.4) are set against that, not against the completion band; 0.03 is 3 episodes per 100.
  - Each launch's class is fixed once, from wave 0's step-0 action fingerprint.
  - Two nominal launches (A and B) do not guarantee both classes: two launches share a class about 2/3 of the time,
    and the old sweep's A and B did. A faulted row with no same-class nominal is flagged, read against both
    nominals, and the worse delta decides.
  - The multi-wave suite rows are judged on wave 0's class.
  - A class flip is identified by the amended rod-speed DR §4 rule (step-0 action deltas allowed; no 1e-5 bound).
    That exemption applies to classing a launch only; it never excuses a pre-onset divergence on wave 0.
- **Decide-record-notify** ([[pause-only-for-irreversible]]). The approved budget runs without stops. A stop rule
  (§6) records, push-notifies and stops. Push, merge and delete stay the user's.
- **Never `git add -A`.** Other sessions share the tree, so commits use curated file lists.

## 3. Faults (pre-registered levels)

**Onset.** Each env draws one onset time from a private generator seeded from the env seed, as `stress_obs` does,
so no other random stream moves. The onset falls inside every suite wave's own hold (the stand and turn waves are
not moving holds; every wave is read on its own, §4.4) and inside field_light's run:

| Row | Onset window |
|---|---|
| suite (12-s own hold) | U(4, 8) s into the own hold |
| field_light | U(10, 60) s |
| kind rows | the approach window, fixed at step 2 from each row's schedule before any faulted row runs |

The sidecar `fault_onsets.json` records each env's onset step. A **transient** fault clears after its duration d.
A **permanent** fault lasts to the episode's end.

The hardware-basis column says why a fault is plausible. It is filled from the hardware-stack survey (§8 item 1)
before the CHARTER freezes.

**What the real stack does on a failure** (hardware survey 2026-11-15, read-only; cites are in the CHARTER).
The policy's IMU, its body xy velocity and the front depth image all come from the **ZED**:
- the IMU is sampled on each successful `grab()`, at 30 fps (`hal/server/jetson/hal_server.py` 525–532,
  `zed_camera.py` 268–300);
- vxy is ZED visual odometry; there is no kinematic estimator.

So the failure modes are:
- **ZED grab fails:** depth, RGB and scan go to **zeros**, while the IMU and vxy **hold their last values**.
  Nothing flags it.
- **VO tracking fails to start:** vxy stays 0.
- **SEN0581 stalls:** its last frame is **held indefinitely**. The 3-s freshness check only logs.
- **The MCU link drops or ticks are missed:** the last command stays in effect, and there is **no watchdog**. The
  HAL sends open-loop jog **PWM**, so a held command keeps the rods **moving**; it does not hold position. The STM
  board holds its last duty.
- **No observation carries a validity mask, age or health bit.** There is no e-stop or sit-down anywhere.
- **The deployed observation contract differs from sim:**
  - joint position = the last commanded target, not a measurement (TODO at `hal_server.py` 750);
  - joint velocity = 0;
  - contact = zeros; in sim `contact_fill` is the current-sense proxy;
  - the omni observation mapping is not wired on hardware at all.
- **Rates differ:** the HAL loop runs at 100 Hz and the sim policy at 50 Hz; the IMU and vxy refresh at 30 Hz.

The fault list follows these modes: hold-last and zero-fill with no flag, not graceful degradation.

### 3.1 Sensor faults (the student's inputs; hook beside `stress_obs`, before the scales and the history push)

| id | fault | levels | hardware basis |
|---|---|---|---|
| S1 | vxy slot → 0 | d = 0.1, 0.5, 2 s; permanent | VO tracking lost or never enabled |
| S2 | vxy slot held at its last value | d = 0.5, 2 s; permanent | a ZED grab failure holds vxy |
| S3 | vxy bias, constant, at a random heading | 0.1, 0.2 m/s, permanent | VO scale or drift error |
| S4 | vxy random-walk drift | σ giving ≈ 0.2 m/s RMS at 10 s | VO drift |
| S5 | IMU frozen (ang_vel + roll/pitch held) | d = 0.1, 0.5, 2 s | a ZED grab failure holds the IMU |
| S6 | roll/pitch bias | 0.05, 0.1 rad, permanent | ZED mount or calibration |
| S8 | current-sense / `contact_fill` channels → 0, or held | permanent (2 cells) | 0 = today's HAL, which sends zeros; held = a stale Mega telemetry frame |
| S10 | depth image → 0 (blackout) | d = 0.5, 2 s; permanent | ZED grab failure (zero-fill) |
| S11 | depth image frozen at its last frame | d = 2 s; permanent | SEN0581 stale-frame hold |

**Depth injection stage.** S10 and Z1 inject a raw 0 m frame *before* the normaliser (`depth / clip − 0.5`,
`crab_hex_omni_observations.py` 778), as the HAL zero-fills the raw metric frame (`hal_server.py` 589–591), so the
student sees −0.5 everywhere (an obstacle at the lens). T12 stays the old normalised-zero anchor (everything at half
the clip range). The dose check asserts the normalised value (−0.5). The deployed normaliser does not exist yet, so
the report states which stimulus was used.

S7 (a joint encoder held) is **dropped**: the deployed stack does not feed encoders back at all, so it is
covered by D1 below.

### 3.1b The compound fault the hardware actually produces

| id | fault | levels | hardware basis |
|---|---|---|---|
| Z1 | **ZED loss:** depth → 0, IMU and vxy held, together | d = 0.5, 2 s; permanent | one `grab()` failure does all three (survey §2) |

Z1 breaks the one-fault-per-cell rule on purpose. It is the single most likely real failure, and it is read as
its own family, never pooled with S2, S5 or S10.

### 3.2 Compute and comms faults (action path; hook in the joint-action term)

| id | fault | levels | hardware basis |
|---|---|---|---|
| B1 | one burst of missed ticks, last action held | 2, 5, 10, 25 steps (40 ms – 0.5 s) | inference overrun, serial hiccup |
| B2 | random tick loss, last action held | p = 0.05, 0.2 | lossy link |
| B3 | action latency 20–60 ms, mean 40 ms (delay drawn per step from U{1, 2, 3}) | permanent | scheduling jitter |
| B4 | **runaway:** each rod keeps its last commanded slew through the outage (the target extrapolated, clipped at the joint limits) | d = 0.1, 0.5 s | the HAL sends open-loop jog PWM and there is no watchdog, so a held command keeps driving |
| B5 | constant action delay, 2 steps (40 ms) | permanent | B3's latency-only control (approved 2026-11-15) |
| B6 | **stop:** rod targets frozen at each rod's current length through the outage; cams stopped | d = 0.1, 0.5 s | what a host-link watchdog that holds position would do; B6 vs B4 is the watchdog read (approved 2026-11-15) |

B1 is **not** a position hold: an action is a rod target and the rod slews toward it at cap, so a held action
keeps the rods driving to a stale target (nominal C1S: rods at cap on 92 % of steps, 53–63 % of hip/knee actions at
|a| ≥ 0.95). B1 and B4 are two models of a stale command. Their gap is **not** read as what a host-link watchdog
would buy.

B3 raises the mean latency as well as jittering it, and a constant 2-step delay already limit-cycles (old sweep
REPORT:117). A B3 failure is reported as "latency 20–60 ms", never as jitter alone.

### 3.3 Actuator faults (plant; per-env, one leg, one side only)

One side is tested. Mirror equivalence is not shown and is not assumed: this student is hand-asymmetric under
faults (T10b lat_l 0.14 vs lat_r 0.80; T12 0.899 vs 0.644). Every A read is labelled "sim plant, force side not
enforced" (rod slew is load-blind, old sweep REPORT:77, 110).

| id | fault | levels | hardware basis |
|---|---|---|---|
| A1 | one leg's rods locked at their onset position | front, mid, rear; permanent | lead-screw stall or jam; also an unpowered rod, since the lead screws self-lock |
| A2 | one leg limp (rod stiffness and effort ≈ 0) | front, mid, rear; permanent | a broken linkage (not lost power: unpowered self-locking rods hold, as A1) |
| A3 | one leg's rods at ×0.3 slew | mid; permanent | a failing motor |
| A4 | one cam stuck | mid; permanent | cam motor fault |
| A5 | A1 at 90 kg | mid; permanent | stall under load |

### 3.4 Anchors (constant from step 0, as in the old sweep)

- **T10b** (vxy zero) and **T12** (zero depth) on C1S. They tie this sweep to the 2026-11-07 sweep and show whether
  the rough-terrain student changed either finding.
- **S12: 30 Hz sample-and-hold** of the IMU and vxy. Each value is refreshed only on the steps a 30 Hz clock ticks,
  so it is held for 1–2 policy steps. That is the ZED's normal cadence, not a fault.
- **D1: today's HAL contract, report only.** Joint position = the last commanded target, joint velocity = 0,
  current/contact = 0. It is not a fault; it shows how far the shipped observation path is from the trained one.
  The result is expected to be bad. It feeds the hardware list in §1, not a verdict.

**Count:** 22 sensor + 3 compound + 12 comms + 9 actuator + 4 anchor = 50 cells.

## 4. Rows and the read

### 4.1 Rows

- **Every cell:** `suite__omni` + `field_light__omni`. They are the old sweep's rows, re-pointed at C1S's eval
  environment (the RT P2 read's student rows), at es1.
- **Perception and estimator cells** (S1 permanent, S2 permanent, S3 0.2, S10 ×3, S11 ×2, Z1 ×3, T12): also the kind rows
  **ledge_down b2** and **pits b2**, the two where the scan and depth plausibly matter (RT REPORT §1). These need a
  fresh campaign pin, because build_check reads BAD at eval cd5ae167 (RT memory, merge prep).
- **Payload:** 0 kg, except A5 (L090).

### 4.2 Seeds

- es1 for every cell.
- es2 for every cell whose es1 post-onset fall share is below 0.5. Above that the verdict is FALLS on one seed, by
  rule. That is about 30 cells.
- Kind rows run es1 + es2 throughout.

### 4.3 Dose checks (per row, automatic)

- **Pre-onset identity:** on suite wave 0 and field_light, for each env, the first step at which the faulted
  episode differs from the same-class nominal must be at or after its onset step. An earlier divergence is a
  launch error. On waves 1–9 the divergences are counted per wave and reported (§2).
- **Fault inactive at each wave's start:** the fault is off at every wave's step 0, and its onset is counted from
  that wave's own hold. This catches a fault that survives the wave reset or counts from the row's start.
- **Realized dose:** a per-fault probe written into the stamp, for example the held channel's variance after onset
  = 0, the locked joint's velocity ≈ 0, or the depth tensor all zero. A dose mismatch is a launch error, never a
  reading.

### 4.4 Metrics and verdicts

**Metrics, per cell-seed:**
- post-onset fall share F (episodes terminated by a fall after onset ÷ episodes alive at onset), computed **per
  suite wave** (n = 100) and per field_light row; ΔF is the worst of these, never the pooled 1000;
- fall-within-2-s share F2;
- the Task 4 v1.2 stable90 lines, plus **per-hand lines** (lat_l, lat_r, turn_l90, turn_r90, move_turn,
  move_turn_r) at the same thresholds, since pooled hands hid T12's lat_r 0.644 in the old sweep. For the verdict,
  every line statistic (ratio, facing_closed, roll/pitch_rms, h_meas) is recomputed from the raw npz over each
  episode's **post-onset** part of the steady window, with the paired nominal over the same per-env windows. The
  whole-hold lines are diluted by 33–67 % pre-onset walking (a mid-hold T12 lat_r 0.644 would read ≈ 0.81 and pass);
  they are report-only, for the tie to the old sweep;
- field_light completion;
- **tracking share** R = the share of episodes alive at the window start whose tracking ratio over a 2-s window
  is within 0.25 of 1. For a transient the window starts 2 s after the fault clears; for a permanent fault it is
  onset + 2 s to onset + 4 s (always inside the hold, since onset ≤ 8 s). R is per episode, so it also catches a
  heading-random bias (S3, S4) whose signed errors cancel in the line medians. R is computed from the raw per-step
  npz (`cmd_applied` vs `root_lin_vel_b`) and only on translate waves (fwd, back, diag, lat_l, lat_r) and translate
  field_light holds, where nominal C1S R = 1.00 per suite wave and 0.97–1.00 per field_light hold (es1–2, es9
  probes); move_turn's nominal R is 0.20–0.66, so it is not used there. R is computed **per translate wave and per
  field_light row**, and the worst decides, as for ΔF. Only episodes whose whole window fits inside the hold count;
  a row with fewer qualifying episodes than the CHARTER's floor (about 25) is report-only. The excluded count and the
  paired nominal's R at the same windows are reported beside it.

The nominal reference is the same episode's own same-class nominal (paired; §2 for waves 1–9 and for rows with no
same-class nominal).

**Verdicts, per fault and level** (both seeds where both ran; pre-registered here, frozen in the CHARTER). The
rule is ordered; the first match decides:

1. **FALLS** if ΔF > 0.10.
2. Else **TOLERATED** if every post-onset line (pooled and per-hand) and completion passes, ΔF ≤ 0.03, and
   R ≥ 0.9.
3. Else **RECOVERS** if transient, ΔF ≤ 0.05 and R ≥ 0.9, even if a post-onset line fails.
4. Else **DEGRADES**.

- A seed split is reported as the worse verdict, marked split.
- **Tolerated duration**, per transient family: the longest d such that it and every shorter level are TOLERATED or
  RECOVERS. This number is what a hardware watchdog timeout gets sized against.
- **Kind rows** (ledge_down, pits): report only, Δsuccess vs the paired nominal on both seeds, no verdict. Whether
  depth matters there at all is a hypothesis they test (RT REPORT: pits never gained).
- **Teacher c1 rows:** report only, one seed, no verdict.

## 5. Budget and order

Driver time is about 10.7 min per suite + field_light cell-seed (old sweep) and about 4 min per kind row (RT sweep).

| Item | Rows | GPU-h |
|---|---|---|
| nominals: A + B at 0 kg es1–2, L090 A es1–2, closing es1; then extra launches until both classes appear at 0 kg and L090 (≤ 2 extra pairs) | 7 + ≤ 4 cell-seeds + 8 kind rows | 2.5 |
| es1 grid | 50 | 8.9 |
| es2 (≈ 35) | 35 | 6.2 |
| kind rows, 12 cells × 2 kinds × 2 seeds | 48 | 3.2 |
| teacher c1, B1 + B4 + A1–A5 + nominal, es1 | 16 | 2.9 |
| smokes and dose checks (incl. field_light, kind, L090, c1 smoke rows) | — | 1.5 |
| **total** | | **≈ 25.2** |

Approved: **27 GPU-h**, with a stop-and-ask if the projection passes 27 (es2 may cover more cells; kind rows ran ≈ 5.5 min in the old sweep). Wall-clock is about 21 h on the local GPU,
serial ([[parallel-gpu-training-runs]]), through one detached queue runner + log-sentinel watcher
([[gpu-queue-chaining]]).

## 6. Stop rules (record, push-notify, stop)

- The knob-off cfg dump is not byte-identical.
- A knob-off or armed-but-never-onset eval row is not bitwise equal to the pin's nominal, after the class-flip rule
  (§2).
- A dose smoke fails, or a grid row fails its per-row dose check or its wave-0 / field_light pre-onset identity
  against a same-class nominal (§4.3).
- The baseline gate fails: a nominal stable90 line or completion is more than 0.04 off the RT P2 read's same-seed
  C1S rows on every launch.
- The tree is dirty, or the code changed since the launch pin (checked before every row).
- The budget projection passes 27 GPU-h.
- DDC growth passes 15 GB, or less than 60 GB of disk is free ([[kit-derived-data-cache-corruption]]).

## 7. Steps

0. Pre-flight: GPU idle, disk and DDC size, C1S .pt located by sha, a C1S nominal suite row at 78b8f9db
   reproduces the RT P2 read's es9 numbers within the class window.
   → main (opus) — commands and judgement on the run state
1. Hardware fault map — **DONE 2026-11-15** (Explore agent, read-only). It changed the list: S7 and S9 were
   dropped, and Z1, B4, S12 and D1 were added. Its file:line cites get copied into the CHARTER.
   → Explore agent (done) — read-only survey with cites, so mistakes are checkable
2. Freeze the CHARTER: faults, levels, onset windows (incl. kind rows), metrics, verdict thresholds, budget, stop
   rules, with this PLAN as its source. Nothing faulted runs before it is committed.
   → main (opus) — the keystone the whole read rests on
   ✓ fable · high — independent check of thresholds vs the noise floor, of reachability (can a 12-s hold show
   recovery?) and of confounds
3. Implement: `config/crab_hex/fault_knobs.py` (`KRABBY_FAULT_*`, refuses unknown names and two faults at once),
   `forward_task/mdp/fault_inject.py` (per-env onset scheduler, private RNG, sidecar), and hooks in
   `observations.py` (sensor), the omni depth term (S10, S11, Z1; S12 and D1 in the obs path), the joint-action term (B1–B6) and the actuator
   (A1–A5, per-env gain, effort and target writes).
   → opus · high — crosses five modules, and a wrong dose fails quietly
4. Tests: knob-off cfg byte-identity, knob parsing and refusal, scheduler determinism, per-fault dose on fake
   tensors, sidecar format.
   → sonnet · high — behaviour pinned by §3–4; the tests fail loudly
5. Review the diff against §2–§4: neutrality, dose and pairing.
   → opus · xhigh — fresh context; the pairing assumption fails quietly
6. GPU smokes: byte-identity dump, one armed-but-never row bitwise vs nominal, a dose smoke **per fault id**
   asserting realized sim state (e.g. A2 torque ≈ 0 on the limp leg, A3 max |rod_speed| = 0.3 × cap, B4 target
   slope = the last slew, S11 depth = the onset frame), and pre-onset identity on one faulted **suite** row with a
   strong fault (wave 0 gated without the class exemption; waves 1–9 counted against the nominal 11–27 per wave),
   plus one faulted field_light row, one kind row, one L090 row and one c1 row. Then step 9's reader is built and run
   on all these smoke rows (F, F2, R and post-onset lines from stamps + sidecars + npz) before step 8 queues
   anything.
   → main (opus)
7. Commit the code + CHARTER (curated list) and write the campaign pin.
   → main (opus) — local only
8. Queue: nominals + baseline gate → es1 grid → es2 → kind rows → teacher → closing.
   → main (opus) — detached runner + watcher; decide-record-notify
9. Reader `scripts/fault_read.py` (built during step 6, run on the full grid here): F, F2, R, line deltas and
   verdicts, from stamps + sidecars + raw npz.
   → opus · high — the statistic the verdicts rest on; quiet failure
   ✓ opus · medium recompute of the verdict table from raw stamps (a fresh agent, independent code)
10. REPORT: verdict table, tolerated durations, the three ranked lists (§1), caveats, and decisions for the user.
    → opus · xhigh — the synthesis the user acts on
    ✓ fable · high — claim check against the stamps
11. Decisions: which riders go into the second distill, the hardware mitigations, whether a fault regime gets
    trained.
    → you — your calls

Routing: strong · roster ok (opus = Opus 5.5, Claude Code 2.1.322) · planned on opus · ceiling: opus · 1× haiku-class
Explore, 1× sonnet·high, 2× opus·high, 2× opus·xhigh, 1× opus·medium check, 6× main, 2× fable·high check, 1× you, fable
retry on 2 fails
Effort enforcement: per call

## 8. Open items before the CHARTER

1. **Hardware findings that need no GPU.** They go into REPORT §1's hardware list, whatever the sweep shows:
   - there is no host-link watchdog, and open-loop PWM keeps rods running on a link loss;
   - no observation carries a validity or health bit;
   - there is no e-stop or sit-down;
   - joint position, joint velocity and contact are placeholders in the HAL;
   - the omni contract is not wired;
   - the HAL runs at 100 Hz against the sim's 50 Hz.

   None of these is in this sweep's scope to fix. The omni rows run at dt 0.02 s (50 Hz; suite 775 steps / 15.5 s),
   confirmed in pass 2.
2. **Code pin — DECIDED:** the code stays on this branch for the whole campaign; Task 4 is relocked only when it
   merges (the hooks sit under Task 4's code pathspec, as PR #14's did).
3. **c1 teacher rows — DECIDED: kept** (2.9 GPU-h, §5), report only.
5. **Decisions of 2026-11-15 (approved as recommended):** budget 27 GPU-h; extra nominal launches until both
   classes appear at 0 kg and L090; B5 (constant 2-step delay) added as B3's control; B6 (rod-freeze stop) added as
   the watchdog read; mirrored A1/A2 cells **not** added (one side tested, §3.3); S12 and D1 kept.
4. Kind-row onset windows depend on where each row's crossing sits in its schedule. They are fixed at step 2.
