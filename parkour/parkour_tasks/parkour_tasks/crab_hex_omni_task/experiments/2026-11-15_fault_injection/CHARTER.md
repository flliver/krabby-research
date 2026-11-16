# Fault-injection sweep of the omni student of record — CHARTER

> **Status:** FROZEN at commit (step 7 of PLAN.md). Source: `PLAN.md` (approved 2026-11-15 as recommended, 3-pass
> audit). Where this file and PLAN.md differ, this file rules; an edit after the freeze is an amendment, logged in §10
> with its time and reason, and never moves a pre-registered threshold after a faulted row has run.
> Branch `feat/fault-injection-sweep`, worktree `.claude/worktrees/fault-sweep`, base origin/main 78b8f9db.

## 1. Question and outcome

How does C1S (`RT_ST_s19/model_34499.pt`, sha 6b48ef33) behave when a sensor, the action path or one actuator fails
part-way through an episode? Eval only. c1 (`policy/rt_11_p3b_rough/model_29500.pt`, sha 6d1f29be) runs report-only
on the B and A families. Outcome: a verdict per cell (§5), a tolerated duration per transient family, and three
ranked lists (distill riders; hardware mitigations; faults needing a trained regime). REPORT §1 also carries the
hardware findings that need no GPU (PLAN §8.1).

## 2. Ground rules

1. One fault per cell (Z1 is the one designed compound). Every fault knob is default-off and set by no preset; with
   no knob set, the cfg dump is byte-identical to 78b8f9db's (§7.1).
2. Decide-record-notify: the approved budget (27 GPU-h) runs without stops; a §8 stop records, push-notifies and
   stops the queue. Push, merge and delete stay the user's. No `git add -A`.
3. The code stays on this branch for the whole campaign. Task 4 is relocked only when it merges.
4. Every compared row runs at this campaign's pin (§7.3). No stamp from an earlier pin is a reference, except as the
   step-0 reproduction check (§7.2).

## 3. Faults

### 3.1 Knobs

One knob arms a cell: `KRABBY_FAULT=<id>:<level>` (e.g. `S1:0.5`, `S1:perm`, `A1:mid`, `B2:0.2`), plus
`KRABBY_FAULT_ONSET=<lo>:<hi>` (seconds from the episode's step 0). `fault_knobs.py` refuses an unknown id or level,
a second `KRABBY_FAULT*` cell, any `KRABBY_STRESS_*` knob together with a fault, and an armed fault with no onset
window. Anchors use their own mechanisms (§3.6).

### 3.2 Onset

- Onsets are a table indexed by (wave, env), drawn from U(lo, hi) on a private `torch.Generator` seeded
  `(env_cfg.seed + 0x6661756C74) % 2**63`, **independent of the cell id**, so every cell of a row/seed shares the
  same onsets. The table is never redrawn or re-armed on a mid-wave auto-reset (a fallen env stays done for the
  wave). Dose draws (S3/S6 headings, S4 walk, B2/B3) use a second private generator. No other random stream is
  touched.
- At every episode reset all fault state is cleared (held values, delay queues, frozen frames, gain/effort writes,
  rod-lock lengths) and the fault is inactive at step 0.
- A transient fault is active on steps [onset, onset + d); a permanent one on [onset, end).
- The sidecar `fault_onsets.json` (per stamp) records, per (wave, env): onset step, last active step, and the dose
  probe values (§6.2).

| Row | `KRABBY_FAULT_ONSET` (s from step 0) | Same as |
|---|---|---|
| suite__omni | 7.5:11.5 | U(4, 8) s into the 12-s own hold (own hold starts at 3.5 s) |
| field_light__omni | 10.0:60.0 | fwd → turn_r90 holds |
| rt_ledge_down_b2__omni | 9.6:11.0 | fwd hold, before the earliest edge contact (11.18 s in 600 C1S episodes) |
| rt_pits_b2__omni | 9.6:13.4 | fwd hold, before the earliest pit contact (13.58 s) |

Durations in steps (dt 0.02): 0.1 s = 5, 0.5 s = 25, 2 s = 100.

### 3.3 Sensor faults (obs path, beside `stress_obs`: before the scales and the history push)

| id | levels | dose |
|---|---|---|
| S1 | 0.1, 0.5, 2, perm | vxy slot (obs 13–14) = 0 |
| S2 | 0.5, 2, perm | vxy held at its value on the step before onset |
| S3 | 0.1, 0.2 (perm) | vxy += b·(cos θ, sin θ), θ ~ U(0, 2π) per episode (fault RNG) |
| S4 | 1 level (perm) | vxy += 2-D random walk, per-axis step σ = 0.2/√1000 = 0.00632 m/s per step (≈ 0.2 m/s RMS at 10 s); realized RMS reported per row |
| S5 | 0.1, 0.5, 2 | ang_vel (obs 0–2) and roll/pitch (3–4) held at the pre-onset values |
| S6 | 0.05, 0.1 (perm) | roll/pitch += b·(cos φ, sin φ), φ ~ U(0, 2π) per episode |
| S8 | zero, held (perm) | contact_fill (obs 83–88) = 0, or held at the pre-onset value |
| S10 | 0.5, 2, perm | **front (ZED) camera only:** raw depth frame = 0 m **before** `process_depth_batch`, so its image is −0.5 everywhere |
| S11 | 2, perm | **left, right and rear (SEN0581 ToF) cameras:** raw depth frames frozen at the last frames captured before onset |
| Z1 | 0.5, 2, perm | S10's front raw-0 depth + S5's IMU hold + S2's vxy hold, all at the same onset and d |

The depth term refreshes every 5 steps and returns the frame one refresh old, so a depth fault reaches the policy
0.1–0.2 s after onset; S10/S11/Z1 apply at the raw-frame stage and so inherit that latency, as the hardware would.

### 3.4 Action-path faults (`parkour_actions.py`: after the clip, at the delayed read)

| id | levels | dose |
|---|---|---|
| B1 | 2, 5, 10, 25 steps | the applied action is held at the last pre-onset applied action for n steps (a stale rod target: rods keep slewing to it) |
| B2 | 0.05, 0.2 (perm) | from onset, each step with probability p (fault RNG) the applied action is the previous applied action |
| B3 | perm | from onset, the delay is drawn per env per step from U{1, 2, 3} (20–60 ms, mean 40 ms) |
| B4 | 0.1, 0.5 s | runaway: each hip/knee rod keeps its last pre-onset slew per substep (length clipped at its limits); each cam keeps its last velocity command |
| B5 | perm | from onset, constant delay 2 steps (B3's latency-only control) |
| B6 | 0.1, 0.5 s | stop: each rod's target length is frozen at its current length; cams commanded to 0 velocity (the watchdog-hold read; B6 vs B4) |

### 3.5 Actuator faults (plant; left side only; mirror equivalence not assumed)

| id | levels | dose |
|---|---|---|
| A1 | front, mid, rear | that leg's hip and knee rod lengths are locked at their onset values (slew = 0) |
| A2 | front, mid, rear | that leg's hip_femur and femur_tibia joints are free: actuator stiffness 0, **damping 0** and max effort 0, written to PhysX, and the rod linkage no longer sets their targets (a broken linkage) |
| A3 | mid | that leg's rod slew cap × 0.3 |
| A4 | mid | that leg's cam velocity = 0 (held at its onset angle) |
| A5 | mid, at L090 | A1 mid at 90 kg |

Every A read is labelled "sim plant, force side not enforced".

### 3.6 Anchors (active from step 0)

Anchors are active from step 0 but are **read on the row's standard onset table** (§3.2; the same windows as every
other cell), so post-onset lines and R are measured where the hold is settled. They change step-0 actions, so they
are exempt from the class and pre-onset identity checks and always read against both nominals. S12 and D1 set
`KRABBY_FAULT_FROM_START=1` with the row's standard `KRABBY_FAULT_ONSET`, so the sidecar still records the table.

| id | mechanism |
|---|---|
| T10b | `KRABBY_STRESS_VXY_ZERO=1` (old sweep env) |
| T12 | checkpoint copy with `zero_depth: true` in `params/agent.yaml` (old sweep `cells.py` mechanism; normalised-zero image) |
| S12 | `KRABBY_FAULT=S12:30`: ang_vel, roll/pitch and vxy refresh only on steps k where (3·k)//5 changes (value age 0 or 20 ms) |
| D1 | `KRABBY_FAULT=D1:1`: joint_pos_delta from the last commanded target where a target exists, joint_vel = 0, contact_fill = 0. Report only. |

**50 cells:** S 22 + Z1 3 + B 12 + A 9 + anchors 4.

## 4. Rows and recipe

- Launcher: `tools/pipeline/gate_row.sh NAME CKPT ENV_FILE MANIFEST SCENARIO ES` (gate_row strips parent `KRABBY_*`,
  applies only the env file, sets PYTHONPATH to this checkout, runs under a systemd `--scope` with MemoryMax 48G).
  Never the `systemd-run --unit` launchers (PYTHONPATH hazard; fix 792c45fa is not on this branch).
- Env files: `envs/<cell>__<row>[_L090].env` = the rod-speed DR `T00_nominal_L000_A__{suite,field_light}.env` (kind
  rows: RT `manifests/envs/rt_kind_eval.env`) + the cell's `KRABBY_FAULT*` lines. L090 adds the F19 `L090.env`
  level file. Student: `--policy-role student`, task `Isaac-Crab-Hex-Omni-Student-Play-v0`, suite with
  `--trained-categories stand,fwd,side,back,diag,turn,move_turn`.
- Output root: MAIN `parkour/outputs/fault_sweep/lr/` (untracked). `--save-raw` stays on (per-episode npz).
- Rows per cell: suite + field_light at es1; es2 for every cell whose es1 worst-row F < 0.5 (a cell at F ≥ 0.5 is
  FALLS on one seed). Perception cells (S1 perm, S2 perm, S3 0.2, S10 ×3, S11 ×2, Z1 ×3, T12) also run ledge_down b2
  and pits b2 at es1 + es2, **permanent levels only** (S1 perm, S2 perm, S3 0.2, S10 perm, S11 perm, Z1 perm,
  T12: 7 cells, 28 rows): a kind-row transient ends before the obstacle in most envs. Payload 0 kg except A5 (L090).
- c1 (report only, es1, suite + field_light): nominal, B1 ×4, B4 ×2, A1–A4 ×8 and A5 with a fresh c1 L090 nominal.
- Nominals (C1S): launches A and B at 0 kg es1–2, L090 A es1–2; then more launches (at most 2 extra pairs) until
  both classes appear at 0 kg and at L090; a closing nominal at es1.

## 5. Statistics and verdicts

Per cell-seed; every statistic is computed from raw npz + sidecar by `scripts/fault_read.py`.

### 5.1 Classes and the reference

- The class is drawn per launch process, so it is grouped per slot (row, seed, payload): two launches of a slot are
  in one class iff they are bitwise equal before the earliest possible onset (suite wave 0 to step 374, field_light
  to 499, kind rows to 479), faulted launches included (they share the onset table). A launch equal to no other
  launch of its slot is relaunched once; still unmatched, it stops the queue (§8). Leaks are caught by the
  per-(wave, env) inactivity probe (§6.2), not by the class.
- Reference = the same-class nominal at the same row, seed and payload. A row with no same-class nominal is
  flagged, read against both nominals, and the worse delta decides. Anchors always take this path.
- Per-episode pairing is required on suite wave 0 and field_light/kind rows, **per env up to that env's own
  onset − 1** (onsets are staggered, so an env already faulted could couple to others through the batched
  simulator). Any such divergence against a same-class reference breaks the pairing premise: stop and amend this
  CHARTER, never relaunch. Suite waves
  1–9 are read on net counts; their step-0 divergences are counted and reported beside the nominal-vs-nominal count
  (11–27 envs per wave).
- Nominal statistics are computed on the reference npz **over the same per-env windows**, using the faulted row's
  sidecar onsets. The R floor (< 25) is counted on the reference, so falls cannot push a row to report-only.

### 5.2 Metrics

- **F** = falls after onset ÷ episodes alive at onset, per suite wave (n ≤ 100) and per field_light row. **ΔF** =
  F − F_ref; the cell-seed's ΔF is the worst over the 10 waves and field_light. F2 = falls within 2 s of onset,
  reported. Falls before onset are counted and reported separately.
- **Post-onset lines** (Task 4 v1.2 thresholds, `thresholds_v1_2.json`): per episode over steady steps with
  t ≥ onset —
  - survival (own-hold survival, unchanged) ≥ 0.90 fwd/side/back/diag, ≥ 0.885 turn and move_turn;
  - |ratio − 1| ≤ 0.25 per category median, ratio = mean actual / mean commanded along-command speed, for fwd,
    back, diag, side **and per hand** lat_l, lat_r;
  - facing_closed ≥ 0.80 for turn and **per hand** turn_l90, turn_r90, re-anchored at onset:
    clip(1 − |err_final| / max(|err_onset|, 0.15), −1, 1); episodes with |err_onset| < 0.15 rad are excluded and
    counted;
  - move_turn and move_turn_r: **survival only** (nominal fails every other line there);
  - h_meas ≥ 0.85 m, roll_rms and pitch_rms ≤ 0.10 rad: suite-pooled medians and field_light, post-onset steps;
  - field_light completion ≥ 0.85.
  Whole-hold lines are reported beside them, report-only.
- **R** (tracking share) = share of episodes alive at the window start whose along-command tracking ratio over a
  2-s window is within 0.25 of 1. Window: transient, [clear + 2 s, clear + 4 s); permanent, [onset + 2 s,
  onset + 4 s). Computed only on translate waves (fwd, back, diag, lat_l, lat_r) and on field_light episodes whose
  whole window lies inside one translate hold (fwd 9.5–17.5, lat_l 17.5–25.5, diag 25.5–33.5, back 33.5–41.5 s).
  Per wave and per field_light row; the worst decides. A wave/row with < 25 qualifying episodes is report-only.
  Excluded counts and the reference R are reported beside it. Calibration (CPU, nominal C1S): 0.98–1.00 per suite
  translate wave, field_light 5th percentile 0.96.
- **Kind rows:** success = schedule complete AND r_max ≥ 3.2 m (RT `readout.py`); Δsuccess vs the paired nominal,
  both seeds. Report only.
- Reported beside F: a hazard rate (falls per post-onset minute), since field_light exposes 13–63 s after onset
  against 4–8 s on the suite; terminations split by reason on A cells.

### 5.3 Verdicts (ordered; first match decides; per seed, then the worse, marked split if they differ)

1. **FALLS** if ΔF > 0.10.
2. Else **TOLERATED** if every post-onset line passes, ΔF ≤ 0.03 and R ≥ 0.9 (R report-only → R not required).
3. Else **RECOVERS** if transient, ΔF ≤ 0.05 and R ≥ 0.9.
4. Else **DEGRADES**.

- ΔF 0.03 = 3 episodes per 100. Noise basis: cross-class C1S pairs moved net falls by ≤ 2/100; nominal suite falls
  0/1000. A verdict within 2 episodes of a ΔF threshold is marked **marginal**.
- **Tolerated duration** per transient family = the longest d such that it and every shorter level are TOLERATED or
  RECOVERS on both seeds.
- Anchors T10b, T12, S12: verdict on the standard onset table against both nominals (§3.6). D1: report only.
- The turn facing_closed margin is thin (nominal per hand 0.86–0.94 at 0 kg, 0.82–0.90 at L090, vs 0.80); the
  REPORT gives the paired nominal beside every turn value.
- c1 rows and kind rows: report only.

## 6. Checks

### 6.1 Smokes (step 6, before anything is queued)

1. Knob-off cfg dump byte-identical to 78b8f9db (`dump_check.sh`, extended to `rt_kind_eval.env` and L090).
2. Armed-but-never row (`KRABBY_FAULT=S1:perm`, onset beyond the episode) bitwise equal to the nominal launch of the
   same process class.
3. A dose smoke **per fault id** (short rows), asserting realized state from the sidecar probes: e.g. A1 rod slew 0,
   A2 PhysX read-back of stiffness/damping/max effort = 0 and the joint leaving its target by > 0.2 rad, A3 max |rod_speed| = 0.3 × cap, A4 cam velocity 0, B4 rod-length slope = the last slew, B6
   rod length constant, S10 front raw frame = 0 (image −0.5) and other cameras unchanged, S11 ToF raw frames = the
   onset frames, S12 hold pattern, Z1 all three channels, each
   transient off again at onset + d.
4. Pre-onset identity on one faulted suite row with a strong fault (wave 0 gated with no class exemption; waves 1–9
   counted), and one faulted field_light, kind, L090 and c1 row each.
5. `fault_read.py` run on all smoke rows; and a **null-fault read**: a nominal launch read as if faulted (real
   sidecar onsets) against another nominal, same class and cross class, must give TOLERATED on every row and seed.

### 6.2 Per row (automatic)

- The sidecar's realized dose matches the cell (held channel variance 0 after onset, depth −0.5, rod slew 0, …).
- Per (wave, env), on every wave: the fault is inactive and gains/limits/buffers are nominal from step 0 to
  onset − 1 (sidecar probe), and off again from onset + d for a transient.
- Pre-onset identity against the same-class nominal on suite wave 0, field_light and kind rows.
- Sim-side doses are recomputed by `fault_read.py` from the npz, not taken from the injector's own masks: A1
  locked-leg `joint_pos`/rod length constant, A3/B4/B6 `rod_speed`, B1/B2/B3/B5 applied `actions`, A2 PhysX gain
  read-back before onset and after. Sidecar probes labelled `self_check` verify masks only; `model_estimate`
  probes (A2 applied torque) are not dose evidence.
- Depth faults reach the policy up to 10 steps after the raw-frame onset and clear up to 10 steps after
  `planned_last`; off-again checks for S10/S11/Z1 use the sidecar's policy-visible clear step.
- A missing sidecar on an armed cell is a launch error.
- Either failing is a launch error, never a reading (§8).

## 7. Pin and pre-flight

1. Knob-off byte identity (§6.1.1).
2. Step 0 pre-flight: GPU idle; disk ≥ 150 GB free (raw ≈ 1.04 GB per suite+field_light cell-seed, ≈ 120 GB total);
   DDC size recorded; C1S .pt located by sha; one C1S nominal suite row at 78b8f9db reproduces the P2SEL es1 numbers
   within ±0.04 on every line.
3. Campaign pin `results/pin.json`, written after the code + this CHARTER commit: code_sha, eval_sha, HEAD, and the
   sha256 of this CHARTER, `envs/*.env` and `cells.yaml`. Checked before every row.

## 8. Stop rules (record, push-notify, stop)

- The knob-off cfg dump is not byte-identical.
- An armed-but-never row is not bitwise equal to a nominal of its slot (§5.1 class rule).
- A dose smoke fails, or a grid row fails its per-row dose or pre-onset identity check (§6.2).
- A launch's pre-onset part matches no other launch of its slot after one relaunch (§5.1).
- The baseline gate fails: a nominal stable90 line or completion is > 0.04 from the P2SEL same-seed C1S value on
  every launch.
- The code tree (pin's code pathspec) is dirty, or the pin moved.
- The budget projection passes 27 GPU-h.
- DDC growth passes 15 GB, or less than 60 GB of disk is free.

## 9. Budget and order

Nominals 2.5 · es1 grid 50 cells 8.9 · es2 ≈ 35 cells 6.2 · kind rows 28 (+ 8 nominal) 2.4 · c1 17 rows + L090
nominal 3.1 · smokes 1.5 → **≈ 24.6 GPU-h**, cap 27. Order: nominals + baseline gate → es1 grid → es2 → kind rows → c1 → closing nominal.
One detached queue runner + log-sentinel watcher; serial.

## 10. Log

- 2026-11-15: drafted from PLAN.md (approved as recommended).
- 2026-11-15: fable·high keystone check, 7 blocking fixes applied before the freeze: anchors read on the standard
  onset table; class grouped per slot by pre-onset equality; A2 damping 0 + PhysX read-back; cameras named per
  depth cell; per-(wave, env) inactivity probe; kind rows permanent levels only; c1 A1–A4 = 8 cells.
- 2026-11-15: after the opus·xhigh code review (no dose/neutrality bug found): pairing checked per env up to its own
  onset − 1; reader recomputes sim doses from npz; self-check vs sim probes labelled; dose seed mixes the env seed
  (es1/es2 no longer share dose noise); sensor faults refused on the teacher role; depth clear latency recorded.
- 2026-11-15: reader resolutions accepted before the freeze (`scripts/fault_read.py` docstring has the full list):
  the npz does not record the applied (post-delay, post-fault) action, so B1/B2/B3/B5 doses are verified by the
  paired sim-divergence rule (first divergence at or after onset, inside the active window + 1 for B1/B4/B6, in
  ≥ 90 % of envs alive at onset) plus sidecar counts; A1 gates on rod_speed = 0 (joint_pos drift reported);
  survival is the harness's own-hold survival over all episodes; a fall inside an R window counts as not tracking;
  rule 3 drops R when R is report-only, as rule 2 does; marginal = |ΔF − t|·n ≤ 2.
- 2026-11-16 00:20: smoke stop at FS_C1S_SMK_never_L000.suite.es1 was a reader false error (onsets beyond the
  episode counted as reached; fixed: alive-at-onset needs onset < n_steps). Diagnosis (2 extra suite es1 launches,
  FSD_*): nominal B and both armed-but-never launches are bitwise equal on waves 0–1; nominal A is the other class.
  So arming without onset leaks nothing, and the class rule (§5.1) works as designed. Baseline gate: nominal A
  field_light es1 bitwise equal to P2SEL; suite lines within 0.003.
