# DR stress sweep (rung 1 of the DR review) — CHARTER

- **Pre-registered:** 2026-11-06, before any row of this campaign ran. Amended the same day, still before any row
  ran, after an independent fable check and an adversarial code review (§6).
- **Plan of record:** the approved plan (`~/.claude/plans/rosy-toasting-dongarra.md`, approved 2026-11-06 after a
  2-pass audit that hit its cap; pass 2's fixes were not re-audited). This CHARTER restates its design where it
  decides a verdict, and pins what the plan left to implementation (§4). Where they differ, this CHARTER wins, and
  every change is logged in §6 before the result it affects is read.
- **Source:** `../lit-review-domain-randomization.md` §10 and §13 rung 1.
- **Branch:** `feat/dr-stress-sweep`, worktree `.claude/worktrees/dr-stress-sweep`. Nothing is pushed.

## 1. Question and outcome

**Question.** Is the F19 depth student, frozen, robust to each plausible sim-to-real discrepancy that F19's DR does
not cover? The sweep moves one plant or sensor parameter at a time, out to beyond its plausible real-world band,
and records where the policy stops passing the Task 4 stability lines.

**Outcome.** A table with one row per term and base:
- the tolerated edges on each side;
- a verdict against the pre-registered band (FRAGILE / ROBUST / MARGINAL / INCOMPLETE);
- per-line deltas from nominal;
- a lenient feasibility edge;
- for FRAGILE terms, a recommended DR range.

Terms 13–15 (payload and push) have no band and are reported as feasibility bounds for rung 4. Rungs 2–4 of the
review are decided and sized from this table; adopting any DR term is the user's call.

**Policy under test.** The student, `parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/
model_28000.pt` (`policy/manifest.yaml`: sha 4fa81f0f). The eval harness cannot load the ONNX export, so it runs the
`.pt` through `OmniStudentStep`. The CPU parity check (`scripts/onnx_parity.py`) ties the export to it: the bundled
ONNX matches the `.pt` to 1.1e-6 on actions over 120 recorded steps (25 encodes). T* is an optional follow-up (§3.6).

## 2. Ground rules

- **Code:** the stress knobs are default-off: unset, they write nothing into the cfg. They are:
  - `crab_hex_omni_task/config/crab_hex/stress_knobs.py`;
  - `crab_hex_forward_task/mdp/stress_obs.py` and `crab_hex_forward_task/mdp/stress_push.py`;
  - hooks in `observations.py`, `parkour_actions.py` and `crab_hex_omni_observations.py`.

  The knob-off Student-Play cfg dump is byte-identical to main for both eval env files. code_sha moves once, at the
  knob commit; this is a milestone boundary (Task 4 closed 2026-10-26) and nothing compares across it. eval_sha stays
  9fe964d7, because no eval or gait_eval file changes.
- **Running:**
  - One Isaac process at a time, through `tools/pipeline/gate_row.sh` from THIS worktree. The script sets PYTHONPATH
    to its own checkout, so the worktree's code is the code that runs.
  - The queue is one driver, `scripts/run_sweep.py`, run with the Isaac venv python and holding a single-instance
    lock, plus a log-sentinel watcher. Never `pgrep -f` a string in the watcher's own command line.
  - Memory-capped scopes; stop with SIGTERM only, never SIGKILL Kit.
  - Every knob is in the row's env file before launch, never written after Kit starts.
  - No smoke or other Isaac job runs alongside the queue: gate_row's busy pattern does not see `stress_smoke.py`.
- **Git:** commit only this campaign's paths and the knob files with their tests. Never `git add -A`; never push. The
  driver refuses a dirty code / eval tree and pins the code and eval hashes at its first launch.
- **Authority:** decide-record-notify everywhere. The run stops only at the §3.5 stop rules, which push-notify.
  Adoption of any DR term, bake and push are the user's.

## 3. Design (pre-registered)

### 3.1 Cells and rows

- **Cell.** One term at one level, run as two rows: `suite__omni` (`eval/scenarios_omni.yaml`, 10 waves × 100 envs)
  and `field_light__omni` (`2026-10-07_omni_phase2/scenarios_p2.yaml`, 100 envs, 80 s), at one env seed.
- **Rows** reproduce the Task 4 student rows exactly, as their `tp_row.json` records them (stamps under
  `parkour/outputs/omni_plan/task4/F19/lr/T4F19_sel_c28000*`):
  - **suite:** env file `F19/envs/p3b.env`, args `--task Isaac-Crab-Hex-Omni-Student-Play-v0 --policy-role student
    --trained-categories stand,fwd,side,back,diag,turn,move_turn`;
  - **field_light:** env file `F19/envs/p3b.field_light.env`, args `--task Isaac-Crab-Hex-Omni-Student-Play-v0
    --policy-role student`;
  - no `TP_STRIP_KEYS` (the record stripped none). The driver removes every parent `KRABBY_*` and `TP_*` variable
    and sets `TP_OUT`, `TP_ROW_TIMEOUT` and `TP_LEVEL_FILE` itself.
- **Per-cell env file:** a copy of that env file plus the cell's one `KRABBY_STRESS_*` line. `gate_row.sh` drops every
  non-payload key from a level file without an error, so stress keys never go there.
- **Base:**
  - At 0 kg the base is unarmed (no level file), as the record was.
  - L090 cells add `F19/envs/levels/L090.env`.
  - Terms 13 and 14 use this campaign's own level files (`levels/`): the four `KRABBY_PAYLOAD_*` keys with the
    cell's values.
- **Output:** `parkour/outputs/omni_plan/dr_stress/` in the main checkout, which is untracked data.

### 3.2 Terms, levels, bands

Bands are inferences from datasheets and the review. Payload and push have no real-world spec, so they get no band
and are report-only. The knob semantics are in `stress_knobs.py`.

| # | Term (knob) | Nominal | Levels | Plausible band |
|---|---|---|---|---|
| 1 | Rod speed ×, hip + knee: the sim's loaded speed, 28 / 33 mm/s (`STRESS_ROD_SPEED`) | 1.0 | 0.5, 0.7, 0.8, 1.15, 1.35 | 0.7–1.35 (light load runs faster: ≈ ×1.37 by the 26 → 19 mm/s droop of the review's reference actuator [7]; heavy load or voltage sag runs slower) |
| 2 | Rod effort limit × (`STRESS_ROD_EFFORT`) | 1.0 | 0.5, 0.7, 0.85 | 0.85–1.0 |
| 3 | Cam motor × (`STRESS_CAM`) | 1.0 | 0.5, 0.8, 1.2 | 0.8–1.2 |
| 4 | Pitch/knee Kp and Kd × (`STRESS_KPKD`), compliance proxy | 1.0 | 0.5, 0.75, 1.25 | 0.75–1.25 |
| 5 | Leg-link mass ×, with inertia (`STRESS_LEG_MASS`) | 1.0 | 0.9, 1.15, 1.5 | 0.9–1.15 |
| 6 | Foot–ground μ (`STRESS_MU`) | U(0.6, 2.0) robot vs 1.0 ground | 0.3, 0.4, 0.8, 1.2 | 0.4–1.2 |
| 7 | Action delay, policy steps (`STRESS_ACT_DELAY`) | 1 (20 ms) | 2, 3, 4 | 1–2 |
| 8 | Proprio obs noise, k × reference table (`STRESS_OBS_NOISE`) | 0 | 0.5, 1, 2, 4 | k ≤ 1 |
| 9 | Current proxy: gain × (`STRESS_CUR_SCALE`), noise σ (`STRESS_CUR_NOISE`) | ×1, σ 0 | ×0.85, ×1.15, σ 0.1, σ 0.25 | ×0.85–1.15, σ ≤ 0.1 |
| 10 | Body xy-velocity slot: noise σ (`STRESS_VXY_NOISE`), zeroed (`STRESS_VXY_ZERO`) | clean | σ 0.05, 0.1, 0.2 m/s; zeroed | σ ≤ 0.1 (zeroed = reliance read, no verdict) |
| 11 | Depth latency, refreshes (`STRESS_DEPTH_LAG`) | 1 | 2, 3 | ≤ 2 |
| 12 | Zero depth (a copy of the checkpoint whose `params/agent.yaml` sets `depth_encoder.zero_depth: true`) | — | 1 cell | read only: ≈ nominal means the depth terms are inert |
| 13 | Payload above 135 kg (existing `PAYLOAD_KG`) | — | L150, L165 | report only |
| 14 | Payload CoM xy spread at 90 kg (existing `PAYLOAD_COM_XY`) | 0 | r = 0.2, 0.3 m | report only |
| 15 | Additive per-env push Δv (`STRESS_PUSH_DV`) | none | 0.1, 0.2, 0.35, 0.5 m/s (≈ 23–115 N·s at 230 kg) | report only |

Terms 1 and 2 also run at L090 (90 kg payload): 8 cells.

**Knob semantics that decide a reading:**
- **Term 1** scales the rod slew limit, which the sim holds constant at the loaded speed. The real speed depends on
  load. The band covers both readings of the datasheet value, slower under heavy load or voltage sag and faster when
  lightly loaded.
- **Term 2** scales the sim's holding-level joint torque cap (`effort_limit_sim` 1000 / 400 N·m). The rod drive force
  (2000 / 500 N rated) is not enforced in sim at all; Task 4 recorded hip rod p90 ≈ 9 kN at 135 kg. So term 2 tests
  the only force bound the sim has. It is not the screw's stall limit, and the report says so.
- **Term 3** scales the cam acceleration limit (2π rad/s²) and, below 1, caps the cam speed at s·π rad/s. The policy
  cannot command more than π rad/s, so at 1.2 only the acceleration rises. The cam effort and saturation are scaled
  too, but they are INERT in sim: the leg's yaw load never reaches the cam shaft, because the hip follows the shaft
  through a software target, not a joint. So term 3 reads the motor's speed and ramp, not its torque. A torque-limited
  cam is a modelling gap the sweep cannot test.
- **Term 4** scales the screw-like PD stiffness and damping. At 16000 / 6000 the sim's static pitch stiffness is about
  1.5× gravity's destabilizer (`crab_hex_scene_cfg.py`). At 0.75× it is about 1.1×, and at 0.5× it is below 1, which
  is open-loop tip-over. A real lead screw is more rigid, not less. A term-4 FRAGILE is therefore a compliance and
  plant-model read: it does NOT generate a Kp/Kd DR range by itself.
- **Term 6** pins the robot material at (μ, μ) and the ground and sim default materials at static = dynamic = μ. The
  plant USD authors no physics material, so PhysX's default average combine applies, and the combined coefficient is
  μ under average, min or max combine. Nominal is not a single μ: it is the randomized robot side averaged with the
  1.0 ground, ≈ 0.8–1.5. The level ordering takes 1.0 as nominal. `--freeze-friction` would null the robot side, and
  no sweep row passes it.
- **Term 11:** the depth buffer refreshes every 5 steps (0.1 s), and the term returns the image `lag` refreshes old.
  At the policy the frame age is a sawtooth: about 0.10–0.18 s at lag 1 and 0.1 s more per extra refresh. The band
  is in refreshes, not milliseconds.
- **Term 12** is checked, not assumed: every student cell's stamps must report `run_meta.student.zero_depth` equal
  to whether the cell is term 12 (§3.3, launch errors).
- **Term 14:** `KRABBY_PAYLOAD_COM_XY=r` draws the payload CoM's x and y EACH U[−r, r] per reset. The bound it finds
  is a DR half-width, which is what rung 4 needs, not a worst-case fixed offset. The mean |offset| per axis is r/2.
- **Term 15** is the action term's additive push (`stress_push.py`): +dv m/s at a uniform heading, every 10 s per
  env, with the timer restarted at each reset of that env. It keeps its own counters and draws nothing from any
  random stream; an interval event would have shifted the reset draws against the same-seed nominal. The heading is
  a counter-based hash of (env seed, reset epoch, env, push index within the epoch), where an epoch is one reset of
  every env (one suite wave). So env e's k-th push in wave w has the same heading at every push level, whatever fell
  in earlier waves. About one push per suite wave (18 s) and about 7 per field_light episode.

**Reference noise table (term 8)**, uniform half-widths in raw units, × k, i.i.d. per channel per step (the
legged_gym convention), applied to raw sensor values before the proprio scales and before the history push:

| channel | half-width |
|---|---|
| body angular velocity | 0.2 rad/s |
| roll, pitch | 0.05 rad |
| joint position (all joints) | 0.01 rad |
| joint velocity (all joints) | 0.1 rad/s (plant-scaled; legged_gym's 1.5 rad/s is for a ~30 rad/s joint) |

**Noise models for terms 9 and 10:**
- **Term 9:** Gaussian on the normalised current, after the gain; then the proxy's [0, 1.5] clip and −0.5 offset.
  The gain is applied to the already-clipped current, so a saturated channel stays saturated.
- **Term 10:** Gaussian in m/s on the raw velocity, before the ×2 scale.

The sensor draws come from a private generator seeded from the env seed; the push headings are a pure hash. The
leg-mass startup event draws from the CPU stream after every other startup term. No knob shifts a random stream that
a nominal row's draws depend on.

### 3.3 Pass/fail and classification

- **Lines:** Task 4's `p3a.stable90.*` (thresholds v1.2): survival per category ≥ 0.90, |ratio − 1| ≤ 0.25,
  turn / move_turn survival ≥ 0.885, turn facing_closed ≥ 0.80, h_meas ≥ 0.85 m, roll / pitch RMS ≤ 0.10, and
  field_light completion ≥ 0.85.
- **Per-seed classification:** each seed is classified on its own 17 `.esK` lines plus its own field_light
  completion at n = 100. The pooled line is re-read per seed. `scripts/stress_read.py` calls `payload_read.py`'s line
  reader once per seed. Its plain `level` call needs both seeds and pools completion, so it would read every es1-only
  cell as unstable.
- **Launch errors are never readings.** A cell-seed is a launch error when either stamp:
  - misses the cell's knob or payload keys in `run_meta.env_vars`, or carries a stray `KRABBY_STRESS_*` or
    `KRABBY_PAYLOAD_*`;
  - reports a `run_meta.student.zero_depth` that does not match the cell;
  - has a dirty `code_state.json`, or one that names another code or eval hash than the launch pin (a missing pin is
    itself a launch error).

  An unknown `KRABBY_STRESS_*` name is refused at cfg build.
- **Noise rule:** the eval has two bitwise-repeatable outcome classes, worth about ±0.04 completion, and a single-seed
  n = 100 read has SE ≈ 0.02–0.03 near 0.9. es2 runs at:
  - every band-edge level;
  - any in-band level whose es1 fails;
  - for terms 13–15, the last passing and the first failing level, searching from the nominal outward. A launch error
    stops that search; it is not a failure.

  Such a level passes only if both seeds pass. A split is MARGINAL. For terms 13–15 the reported bound is
  `[last passing, first failing]` from the nominal outward, with split levels and one-seed levels listed, and an
  unread or launch-error level ending the search.
- **Verdicts** (per term and base):
  - FRAGILE = an in-band level fails on BOTH seeds → DR candidate.
  - MARGINAL = an in-band deciding level is split.
  - INCOMPLETE = a band edge, or a failing in-band level, is not resolved on both seeds, or an in-band level is
    unread or a launch error.
  - ROBUST = every band edge passes on both seeds and no in-band level fails → no DR needed, though cheap DR is
    optional.
  - The term's verdict is the worst of its levels (FRAGILE > MARGINAL > INCOMPLETE > ROBUST), reported per side too.
    Terms 1 and 2 combine their two bases the same way: FRAGILE on either base is FRAGILE.
- **Tolerated edges** are reported per side: the last passing level below nominal and the last above it, counting
  contiguous passing levels from nominal. Beyond-band levels run es1 only.
- **Recommended DR range (FRAGILE):** covers at least the plausible band edge on the failing side, never capped at
  the tolerated edge, because the failing region is what DR has to train over. Term 4 is the exception (§3.2).
- **Lenient feasibility edge** (RAPP-style, report only): per side, the last of the contiguous levels from the
  nominal outward at which every suite category's survival, `stand` included, is ≥ 0.5 on every seed read.
- **Deltas beside every verdict** (user decision 2026-11-06):
  - Every cell-seed reports each line's change from the same-day nominal of the same seed: launches A and B at
    0 kg, and A at L090 for the 90 kg cells and the payload terms. The number of nominal references used is
    reported; a nominal that is a launch error is dropped and shows as fewer references (`NO REF` at zero).
  - A survival or completion drop of more than 0.05 is flagged. With two nominal classes, the flag fires only if the
    drop exceeds 0.05 against both.
  - Term 13 (135+ kg) has no same-day reference: its deltas against the 90 kg nominal are reported but never
    flagged.
  - The flag does not change the verdict; the user judges flagged ROBUST terms at the report.

### 3.4 Order and size

1. **Nominal:** 0 kg es1 + es2 twice (separate launches A and B), and L090 es1 + es2 once (step 7). That is 6
   cell-seeds.
2. **es1 grid:** 55 cells (43 at 0 kg, the 2 term-13 cells at 150 / 165 kg, 10 at L090 including term 14).
3. **es2** at every band-edge level (21), at in-band es1 failures, and at the terms 13–15 transitions (≈ 6).
4. **Closing nominal:** 0 kg es1, 1 cell-seed.

Total ≈ 89 cell-seeds ≈ 15.9 GPU-h at ≈ 10.7 min per cell-seed, plus the optional T* follow-up of ≤ 2 GPU-h. The
smokes and checks already run (dumps, dose smoke, e2e rows) took ≈ 0.5 GPU-h.

### 3.5 Stop rules (record, push-notify, stop)

- **Byte-identical check fails.** With every knob unset, the cfg dump of the Student-Play env differs from main's.
  (Passed 2026-11-06.)
- **Dose smoke fails.** Any knob's realized dose differs from its spec. (Passed 2026-11-06; term 15 re-checked
  after its move to the action term.)
- **ONNX parity fails.** The ONNX export's actions differ from the `.pt`'s on recorded observations beyond 1e-4.
  The question goes to the user. (Passed 2026-11-06: 1.1e-6.)
- **The tree is dirty, or the code changed since the launch pin.** Checked at start and before EVERY row, so a
  change mid-queue stops it (`DRS STOP dirty_tree` / `code_changed_since_launch`) instead of producing rows the
  reader would reject.
- **Baseline gate fails (step 7):**
  - any per-seed stable90 line fails, or any launch error, on a nominal launch;
  - per-seed field_light completion deviates from the Task 4 record's same seed by more than 0.04 on EVERY launch
    of that base. The record is 0 kg es1 1.00, es2 0.99, and L090 es1 0.95, es2 0.95.
  - A split between the two 0 kg launches is the documented two-class flip. It is recorded and is not a stop.
- **Not a stop:** if the closing nominal's 18 line values do not equal launch A's or B's es1 (the bitwise classes),
  record it (`results/closing.json`, with status match / mismatch / unread / launch_error), push-notify, and flag
  every cell in the table (a per-row `closing_flag` and a banner in `table.md`). The queue is not stopped. A resume that runs rows after the closing runs a fresh closing
  (Z2, Z3 ...); a closing older than any sweep stamp flags the table as stale.
- **Crash or user stop:** the driver writes `DRS STOP crash`, `DRS STOP user` (STOP_QUEUE) or `DRS STOP signal`
  (SIGTERM / SIGINT / SIGHUP; the running row finishes first) and exits. A rerun resumes from the stamps: a row is
  skipped only when its newest stamp passes the launch checks, otherwise it is re-run (`DRS RERUN`). `--from` a
  later phase requires a passed baseline.

### 3.6 Optional T* follow-up (≤ 2 GPU-h)

**Trigger.** A term the student fails inside its band (FRAGILE).

**Run.** Rerun that term's deciding cells on T* (`policy/f19_8_p3b_payload180/model_25992.pt`, `--policy-role
teacher`, same Student-Play task and env files), at es1 + es2.

**Read.** Teacher fragility vs distillation loss.

**Caveat.** T*'s privileged inputs carry the true μ, body mass and gain ratios, and clean velocity in the privileged
block. On terms 6 and 10 the teacher therefore sees what the student cannot.

## 4. Pinned at implementation

- **Leg links (term 5):** `(FL|FR|ML|MR|RL|RR)_(Hip|Femur|Tibia|Footpad)`, 24 links, ≈ 11.6 of each leg's 11.9 kg.
  The CamShaft (0.3 kg, inertia pinned for actuator stability) is left alone.
- **Push (term 15):** the action term's `stress_push` hook, stepped once per env step in `process_actions` before
  physics and restarted in `reset`. The inherited velocity-replacement push stays nulled by the harness.
- **Action delay (term 7):** `action_delay_steps = [d]`, popped at step 0. `history_length` is 8, so d ≤ 7.
- **Depth lag (term 11):** sets `buffer_len = lag + 1` and `lag` on all four depth terms; without the knob the term
  is unchanged and returns `buffer[:, -2]`.
- **Zero depth (term 12):** a full copy (not a symlink) of the checkpoint and its `params/` under
  `parkour/outputs/omni_plan/dr_stress/ckpt_zero_depth/`, with the one `zero_depth: false` flipped. The harness reads
  the checkpoint's own `params/agent.yaml`. The driver verifies the copy before every run and rebuilds it if it does
  not match.

## 5. Files

- `CHARTER.md`: this file.
- `envs/`: the per-cell env files, generated by `scripts/run_sweep.py` (`prepare`).
- `levels/`: payload level files for terms 13 and 14.
- `scripts/`:
  - `cells.py` (the table);
  - `run_sweep.py` (the driver);
  - `stress_read.py` (the reader);
  - `stress_smoke.py` + `smoke.sh` (the dose smoke);
  - `dump_check.sh` (the knob-off proof);
  - `onnx_parity.py`.
- `results/`: the launch pin, baseline, es2 plan, closing record, the per-term table (`table.json`, `table.md`) and
  the driver log.
- `REPORT.md`: written at the end.

## 6. Log

- **2026-11-06, pre-registration.** CHARTER written; knobs implemented default-off on `feat/dr-stress-sweep`.
- **2026-11-06, checks before any row.**
  - Knob-off dumps byte-identical to main (both env files).
  - Dose smoke all checks pass: G0 hooks absent, G1 plant 25 checks, G2 sensor 23, G3 push 4.
  - ONNX parity 1.1e-6.
  - Three end-to-end gate_row rows (μ 0.4, obs noise 1, push 0.35, under the old event push) carried their keys.
- **2026-11-06, amendments before any row**, from an independent fable check of this CHARTER and a 31-agent
  adversarial code review (25 confirmed findings, none critical):
  1. **Term 1:** the sim's rod speed is the LOADED speed (`speed_loaded_m_s`), not no-load. The band moves from
     0.7–1.15 to 0.7–1.35 and a 1.35 level is added on both bases, +2 cells. The "datasheet" provenance is corrected:
     the 26 → 19 mm/s droop is the review's reference actuator [7], not the SLA08 / YH8.
  2. **Term 15:** the push moves from an IsaacLab interval event to the action term. The event manager drew its
     per-env timers from the global CUDA stream at construction, at every reset and at every fire, which shifted the
     reset draws of suite wave 0 and the whole field_light row against the nominal.
  3. **Term 3:** the cam effort / saturation half of the dose is inert in sim (no leg load on the shaft); the term
     reads speed and ramp only.
  4. **Term 4:** a FRAGILE is a compliance read that does not generate a Kp/Kd range.
  5. **Term 12:** checked through `run_meta.student.zero_depth`. The wording "`--zero-depth` flag" was wrong; no such
     flag exists.
  6. **Term 14:** stated as a U[−r, r] per-axis spread.
  7. **Term 13:** deltas are taken against the 90 kg nominal and never flagged.
  8. **Reader:**
     - FRAGILE needs both seeds;
     - INCOMPLETE is defined;
     - launch errors never count as failures, and they stop the terms 13–15 search;
     - the per-line deltas are now emitted in `table.json` / `table.md`;
     - the flag edge is strict (`> 0.05`);
     - the lenient edge includes `stand`;
     - terms 1–2 combine their bases (worst);
     - the closing mismatch reaches the table.
  9. **Baseline completion rule:** it now trips only when every launch of a base deviates. The old per-launch rule
     contradicted "a split is not a stop" whenever the two-class gap exceeded 0.04. The record values were also
     corrected: they had been swapped (es1 1.00, es2 0.99).
  10. **Driver:**
      - strips parent `TP_*` variables;
      - refuses a dirty tree and pins the code / eval hashes;
      - holds a single-instance lock;
      - writes crash and stop sentinels;
      - keeps per-attempt row logs;
      - verifies or rebuilds the zero-depth copy.
  11. **Unknown `KRABBY_STRESS_*` names** are refused at cfg build.
- **2026-11-06, second review round before any row** (a 25-agent verification review of the fixes, 15 confirmed
  findings, none critical; the knob-off dumps, smoke G0 and G3 re-passed after the push move):
  1. **Driver:** the tree and pin are checked before every row; a stamp that fails the launch checks is re-run
     instead of skipped; a fresh closing (Z2 ...) runs when rows ran after the last closing, and a stale closing
     flags the table, and a closing whose stamps fail the launch checks does not count (the next launch runs);
     SIGTERM / SIGINT / SIGHUP stop after the running row with a sentinel; `--from` later phases
     need a passed baseline; the zero-depth copy is verified by sha256; row logs carry the invocation time.
  2. **Reader:** a missing launch pin is a launch error, and so is a stray `KRABBY_PAYLOAD_*` key; dropped nominal
     references are counted and shown; terms 13–15 report an interval bound with split and one-seed levels; the
     closing flag marks every row.
  3. **Push headings** are counter-based (seed, epoch, env, index), so push levels share headings per wave.
