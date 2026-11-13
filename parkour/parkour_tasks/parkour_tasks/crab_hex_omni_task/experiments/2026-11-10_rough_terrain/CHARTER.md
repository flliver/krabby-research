# CHARTER — traversable rough-terrain teacher (2026-11-10)

Decision log for `PLAN.md` (audited: 3 passes plus one extra pass the user approved; the pass-4 fixes were not
re-audited). Literature review: `../lit-review-rough-terrain.md` (12 / 12 claims verified, 111 references, fable
final check applied). Predecessors:
- `2026-10-28_site_teacher`: closed not accepted, on the unmerged branch `worktree-sen0581-cameras`;
- `2026-11-07_rod_speed_dr`: TDR baked as teacher of record, merged as PR #13.

## 1. Settled decisions (user, 2026-11-10)

1. No walls, blockers or obstructions that require stopping. The terrain is slopes, hills, ledges and steps,
   potholes, gravel and rubble.
2. The literature review informs the design.
3. The DR on main (rod-speed DR, TDR as teacher of record) is the base and is kept in training.
4. G is judged per kind:
   - k = 2 gated kinds must gain, and none may regress;
   - bands above the blind envelope are report-only;
   - bands are chosen on es3 and G is read on es1–2.
5. Set aside to P3: the scan-ablation knob port, the `_flat` twins, the site recorder port and the new teacher's
   band-above rows.
6. After the pass-3 cap:
   - one more audit pass;
   - the plan size is kept;
   - the iteration-1500 early read is one log line;
   - TDR's three above-envelope baseline rows are kept.
7. Plan approved, execution started: "Approve, start step 0".

## 2. Standing rules

- Decide, record, notify. Stop and ask only at the pre-registered points: exactly one gated kind (step 7), and the
  HALT rows of the decision table. Push-notify at each of them.
- Only wall-clock time counts (local GPU): never idle the GPU on a pending decision.
- Pause for push, delete or bake; never push.
- One Isaac process at a time, through a detached queue and a log-sentinel watcher. Never `pgrep -f` a string in
  the watcher's own command line. Use memory caps; no RecordVideo.
- Workflow subagents never write report files (the parent writes them). No downloads or installs in any brief.
- Other sessions share these checkouts: never revert unexplained changes.

## 3. Log

### 2026-11-10 15:38 — step 0: worktree and campaign dir

- **Branch.** `feat/rough-terrain`, worktree `.claude/worktrees/rough-terrain`, cut from `origin/main` `3f944c4b`
  (the PR #13 merge: TDR as policy of record, rod-speed DR on main).
- **origin/main.** It has since moved to `b663af7d`, which adds STM leg-board firmware only (no `parkour/` or
  training-test change). The base stays `3f944c4b`, as the plan states.
- **Filed.** `../lit-review-rough-terrain.md`, `PLAN.md` (the audited plan) and this CHARTER.
- **Prior reviews.** The review cites `lit-review-construction-terrain-dr.md` and `lit-review-obstacle-avoidance.md`,
  which live on the unmerged site branch, not on this one.

### 2026-11-10 — step 1: trimmed site port (workflow `wf_bfd12127-97c`; opus·high porter, opus·xhigh reviewer)

- **Ported** from site head `2a3e3c51` by file extraction:
  - the site terrain library (`site_terrains.py`, `omni_site.py`, cfgs, generator 8-tuple and layers path);
  - `site_raster.py`;
  - a trimmed `site_task_math.py` (crossed and hazard radii only);
  - spawn pad level (`spawn_level_z`, site mode only);
  - per-kind telemetry in site mode, counting main's applied `promotion_mask`;
  - the knobs `KRABBY_SITE_MIX / DIFF / LADDER` (default-off, site geometry only);
  - 4 test files and the fixtures.
- **Excluded:** layout v2 and scan ablation, site DR, site rewards, the critic group, rsl_rl changes, camera code
  and the recorder or readout scripts. `parkour_actions.py` is untouched, so the rod-speed DR AST pins hold.
- **Loud refusals.** The 20 unported site knobs now raise loudly if set (`SITE_UNPORTED_VARS`). Step 2 removes any
  friction-contract name it reuses.
- **Neutrality (CPU).** All 232 recorded env files (rod-speed DR, DR sweep, F19) give identical cfg snapshots,
  stdout and exception types against a `git archive` of `3f944c4b`. TDR's recorded env.yaml gives an identical
  dump geometry and 23 / 23 dump checks. Only the error text for an unset geometry changes (it now lists 'site').
- **Unit suite** (same command as the baseline): 3 failed and 45 collection errors, identical node ids to main's
  baseline, with 369 more passes. The 3 failures are recorded-artifact tests; the collection errors are zmq / serial.
- **Review: READY, no blocking.** Minor items carried to step 2:
  - SITE's ladders, bands and cross_r are unchanged: pits and rubble band 2.0–3.6 m, cross_r 4.1;
  - `DEFAULT_SITE_MIX` still contains blockers: refuse a blank mix or change the default;
  - the 0.03 m roughness cap in site geometry blocks gravel's 0.05 m;
  - the telemetry's per-kind promote and demote means are not zero-filled; read `site_cnt_*`;
  - the two neutrality pins skip without `git show 3f944c4b` (CI shallow checkouts).
- **Not verifiable without Kit**, deferred to the step-5 smoke: the event manager accepting `spawn_level_z`, the
  importer attributes, and the site-mode extras under the real runner.
- **Pins.** code_sha moves with this commit (milestone boundary, as planned); eval_sha does not.

### 2026-11-10 — steps 2 and 4 (workflow `wf_72369b67-360`; two opus·high implementers in parallel against one fixed interface, two opus·xhigh reviewers: both READY, no blocking)

**Step 2 (library).**
- New ladder `KRABBY_SITE_LADDER=rt`, valid only under site geometry. Its own generator `OMNI_SITE_RT_TERRAINS_CFG`
  has columns: flat, gravel, field_sparse, field_dense, slopes, hills, ledge_up, ledge_down, pits, rubble. Gravel
  sits next to flat.
- `KRABBY_SITE_MIX` is required. Blockers and the unused site kinds are refused. A gravel column is refused unless
  `KRABBY_RT_FRICTION` is set. The default and b4 ladders are unchanged (264 layouts hashed identical).
- **Kinds:**
  - **slopes:** a constant-gradient frustum, 2 → 20°.
  - **hills:** 2 radial × 6 angular waves, flat out to the 1.7 m pad edge, then zero height and slope growing to
    full slope ≈ 0.34 m out. This differs in form from the "≤ 0.3 m blend": a separate blend window put ≈ 23 %
    gradient spikes in the rings, so it was dropped. Accepted.
  - **ledges:** up 0.02 → 0.30 m, down 0.02 → 0.35 m.
  - **pits:** one pit per angular slot in r 2.0–2.7 m, depth 0.03 → 0.35 m, side 0.15 → 0.60 m. Cover ladder
    0.10 → 0.30 (the site's 0.40 cannot be packed with 0.15 m pits).
  - **rubble:** 0.02 → 0.15 m in r 2.0–2.7 m.
  - **gravel:** its own roughness 0.02 → 0.05 m, which survives main's forced 0.02 m noise and the site's 0.03 m cap.
- **Crossing radii:** RT_CROSS_R are all ≤ 3.2 m.
- **Realised dose** (levels 0 / 4 / 9) is within ±10 % for every kind, and all three slope assertions hold. Hills
  read 2.0 / 9.3 / 18.6° on a 5 × 5 plane fit; the design maximum equals the table. Seam ring at 0, spawn at the
  centre level. Margin to the −0.25 m termination ≥ 0.535 m; the worst case is a −0.80 m sunk slope pad.
- **Friction contract** (`KRABBY_RT_FRICTION=gravel:LO:HI`, `mdp/rt_friction.py`):
  - multiply ground at μ 1.0;
  - every shape of an env gets one μ_eff, static = dynamic;
  - non-gravel envs get (b + 1)/2 of 64 buckets of U(0.6, 2.0); gravel envs get U(LO, HI);
  - draws are keyed by (seed, env_index), so pairing holds across processes;
  - the privileged friction obs reports 2·μ_eff − 1.
  - Note: the seed-19 bucket set is a different realisation than TDR's (mean μ_eff 1.189 against a law mean of
    1.15; KS 0.022 over 40 seeds).
  - Not measurable on CPU, deferred to the step-5 smoke: PhysX's combine priority (multiply over the robot's
    average) at contact level.
- **Item 17: no flat-ground reward term with material weight is live in TDR's recipe.** `reward_foot_clearance` and
  `penalty_swing_min_clearance` are weight 0. ClockSwingApex is 0.001 (ε). The orientation penalty is zeroed on
  non-flat tiles.
- **`envs/RT_train.env`** differs from TDR's train env exactly by:
  - geometry site, `SITE_LADDER=rt`, the 20-column mix;
  - `RT_FRICTION=gravel:0.2:0.5`;
  - `DR_ROD_SPEED=0.6:1.35` (no ramp);
  - PYTHONPATH pointed at this worktree; TDR's file pointed at dr-stress-sweep, which has no rt code.

**Step 4 (campaign scripts).**
- `manifests/` (spec → generator with `--check`): 21 kind rows `rt_<kind>_b<1|2|3>__omni` on field_hard's schedule.
  The header is byte-equal to scenarios_p2's. Gravel rows carry `KRABBY_RT_FRICTION`. No `KRABBY_DR_*` on any eval
  row.
- `rows_rt.json`: the retention rows.
- `readout.py` (module name `rt_readout`):
  - keys rows by checkpoint sha256;
  - computes crossed (xy, from the spawn), success, stuck, r_max and termination reasons;
  - G per kind (k = 2);
  - R rows, and R tracking-and-turning via the repo's `zero_kg_check` (v1.2, drift 0);
  - the decision rules.
- `make_thresholds.py` implements the step-7 rule (es3 only, n = 100, in-envelope bands, k = 2, 0 / 1 gated
  branches).
- **Validations on saved data (all PASS):**
  - e1: TDR's G1 / R, 481 numbers within 1e-12, R PASS at its_past 2999 and at 0;
  - e2: the site crossed rule on 27 T\* site rows, 2700 / 2700 episodes identical, plus 6 recorded values;
  - e3: share of TDR completers reaching the radius:

    | Row | 3.0 m | 3.2 m |
    |---|---|---|
    | field_hard | 1.000 | 1.000 |
    | field_light | 1.000 | 1.000 |
    | ×0.7 L090 field_light | 0.995 | 0.984 |

    No radius needs lowering.

**Unit suite.** After both steps: 3 failed + 45 collection errors, identical to main's baseline, with 75 more passes.

**Hardening in progress** (`wf_c3741d20-df5`), from the reviews' minor lists: read-out fail-closed checks, the slope
up / down variant alternating by column, and a rename of `RT_KINDS`. The step 5–9 run machinery is being built and
reviewed in the same workflow.

### 2026-11-10 — hardening and run machinery (`wf_c3741d20-df5`, `wf_01a4b94d-0f4`)

**Hardening** (from the step-2 / 4 review minors):
- e2 now asserts the campaign's own spawn-frame path (2700 / 2700) and reads T\* stamps only, checked by sha.
- The read-out radii and ladders are pinned to the library by a test.
- Kind rows must have n = 100 per seed.
- Duplicate-stamp resolution drops failed launches (rc ≠ 0) first.
- R tracking fails closed unless it reads track_err under v1.2. side22 is read relative to TDR **and** absolute
  (the new teacher must itself be FAST / FAST).
- `gates()` takes its rules from the frozen thresholds file and refuses if the recorded script sha256s differ.
- An explicit spec path must exist.
- **Slopes alternate up / down by column parity** (in the 20-column mix, columns 8 and 9), so every level holds both
  variants. The RNG stream is unchanged.
- `RT_KINDS` in `omni_terrain_knobs` is renamed `RT_COLUMN_KINDS`.

**Run machinery** (`scripts/`): `build_check.py` (pin), `rt_machinery.py`, the single detached queue `rt_queue.py`
(blocks smoke / baseline / train / gate), `rt_launch.sh`, `rt_wait.sh`, `rt_kit_run.sh`, two Kit smokes
(`rt_kit_train_smoke.py`, `rt_kit_eval_probe.py`), `rt_smoke_check.py` and the step-8 launcher `rt_train.py`.
- **Queue behaviour:**
  - one Isaac process at a time (flock plus a `ps` wait, no `pgrep -f`);
  - DerivedDataCache and disk guards;
  - an `RTQ <label> rc=<n>` sentinel per step and `RTQ_DONE <block> fail=<n>`, printed even on a crash;
  - stamps are reused only when they carry the pinned build (review blocking item, fixed).
- **Training launch (step 8):** resumes TDR (counter 30450, lr 1.5e-05) for 6000 iterations, final `model_34990.pt`.
  - Pre-flight PF1–PF8, including the build pin, the exact env key set, and the child's `/proc/<pid>/environ`.
  - In-run PF9–PF14, including a params diff against TDR's run with an explicit allow-list.
  - One report-only `RTQ NOTIFY early_read it=1500` line.
- **Smoke block (step 5):**
  - S.neutral: TDR's nominal L090 field_light es1 row, compared bitwise (or by the class rule) against the two saved
    stamps;
  - two Kit training smokes (rt, and a knobs-off reference) asserting kinds, column shares, all-shape μ, the friction
    obs = 2μ − 1, a contact slide-fit μ, telemetry, obs width 1149, spawn height and 2-s terminations;
  - S.train50: 50 real iterations, with the params diff;
  - 21 × 4-episode eval rows plus the Kit probe, then CPU checks. The spawn check also requires a raycast terrain
    mesh.
- G.ident requires `RT_s19.json` to show status ok, iteration 34990 and a matching sha.
- **Dry passes** fail only where expected (no pin yet, no rows).

**Unit suite:** 3 failed + 45 collection errors, identical to the baseline; 6562 passed.

**Disk:** 46 GB free. The baseline (≈ 15 GB) plus the gate block (≈ 18 GB) would cross the queue's 20 GB floor
during step 9. Asked the user whether to delete the closed site campaign's chain raw npz (≈ 39 GB) or lower the
floor. Pending; not needed before step 9.

### 2026-11-10 19:40–19:47 — step 5, smoke run 1 (pin code_sha 89c1916b, eval_sha 9fe964d7 = main)

- **S.neutral: BITWISE.** TDR at the new build, every new knob off, on its saved nominal L090 field_light es1 row
  equals both saved stamps (`results/step5/neutrality.json`).
- **S.train_ref and S.train_rt** (400 envs, TDR weights) pass:
  - every kind builds;
  - the column count is the mix (3 / 2 / 1 / 2 × 7) and the env share matches;
  - ≥ 20 envs per column, ≥ 40 resets per column;
  - every shape has one μ, static = dynamic, restitution 0;
  - μ equals the contract draw (err 6e-8); gravel 0.201–0.491; others 0.81–1.50 (mean 1.18), TDR's buckets;
  - friction obs = 2μ − 1 (err 3e-8);
  - obs width 1149;
  - spawn root height within 1.085 ± 0.05 m on every kind;
  - per-kind telemetry keys present.
- **Slide test (contact-level combine priority).**
  - **Gravel μ_eff 0.201 measures 0.177**, against 0.60 under "average" and 1.0 under "max". So the ground's
    multiply (or min) wins over the robot's average.
  - The high-μ env (1.498) read 0.11. Its trace falls for ≈ 0.3 s, then rebounds (0.085 → 0.27 m/s): the body
    pivots on stuck feet, so this is not a slide measurement. The fit also ran over the rebound.
  - **Decision (decide-record-notify): the high-μ slide is report-only.** The low-μ read is decisive. Multiply vs
    min is settled by the configured ground mode (multiply), PhysX's priority (multiply > min > average), and the
    PF14 params diff of the real run. `fit_decel` now ends the sliding phase at the first rebound (test added).
- **Module-origin check** read `parkour_isaaclab: null`. It is a namespace package with no `__file__`, so this was a
  check bug. The probe now reads `__path__`. The rt kinds that built exist only in this worktree.
- **Queue crash on the failure path.** `rt_queue.py` reused the name `st` (the failure tally) for a stamp list after
  every stamped row, so the first failure afterwards raised TypeError. Renamed; the sentinel still printed
  `RTQ_DONE fail=1 (queue crashed …)` as designed.
- Next: relaunch the smoke block from S.train_ref. The S.neutral stamp is at the pinned build and is reused.

### 2026-11-10 19:50–20:15 — step 5, smoke run 2

- **Training smokes pass** (rerun after the fixes): S.train_ref, S.train_rt and S.train.cmp all rc 0.
- **S.train50** stopped at iteration ≈ 2 on PF14 (params diff against TDR's run). The only disallowed difference was
  `events.reset_root_state.params.spawn_level_z: absent → True`, which is planned: the site spawn at the tile
  centre level. The allow-list rule is added (must equal True under site geometry). The recorded params now read
  `RT PARAMS OK`: every difference is a planned one (terrain generator keys, friction-contract keys, ground
  multiply, rod-DR ramp 0, run name and iterations). S.train50 reruns under the name `RT5_smoke50b`.
- **Eval smoke rows.** 7 rows plus their probes rc 0 (slopes b1–b3, hills b1–b3, ledge_up b1). The CPU checks run at
  the end of the block.
- **DerivedDataCache guard stop.** The cache was at 81 GB against the 80 GB cap. It grew at 75 GB (19:52) → 81 GB
  (20:14) over 13 Kit launches, ≈ 0.46 GB per launch: steady growth (new terrain meshes per row), not the
  2.4 GB/s corruption balloon. Disk went 45 → 39 GB free.
  - Projection: ≈ 40 GB more DDC plus ≈ 33 GB of baseline and gate stamps, against 39 GB free.
  - **Decision (reversible): cap raised to `RTQ_DDC_MAX_GB=100` for the rest of the smoke.** The 3 GB per-step
    growth guard still catches a balloon.
  - **Asked the user** to free space (delete the regenerable DDC, 81 GB, and/or the closed site campaign's chain
    raw npz, 39 GB) before step 6.
- **Launcher hazard fixed** (`rt_launch.sh` refuses before rotating while a queue holds the lock). A relaunch had
  rotated the live queue's log.

### 2026-11-10 20:16–20:58 — step 5 DONE (all smokes pass)

- **S.train50** (`RT5_smoke50b`, 50 real iterations from TDR) passes PF1–PF14:
  - the child environ and PYTHONPATH as composed;
  - checkpoint loaded and alg_state restored;
  - first title 28991 / 29041;
  - the rt knob lines and the `[rt_friction]` draw line;
  - telemetry for all 10 kinds;
  - **params diff vs TDR: 387 planned differences, 0 unplanned**;
  - final `model_29040.pt`.
- **Eval smoke: 21 / 21 rows pass** (4 episodes each; spawn height, no early termination, raycast mesh, realised
  dose and band, slope eval-track assertions, gravel μ_eff). The first read failed 9 rows (ledges and rubble) only
  on the probe-vs-harness surface tolerance: 0.0203–0.0242 m on single envs, with dpad ≤ 0.009 m and dxy 0. That
  is the pad's 2 cm roughness, so the tolerance is now 0.04 m. The fallback it guards is caught by the
  terrain-mesh check.
- **TDR's 4-episode fall shares** (report only, n = 4 per row):

  | Kind | b1 | b2 | b3 |
  |---|---|---|---|
  | slopes | 0 | 0.25 | 0.5 |
  | hills | 0 | 0 | 0 |
  | ledge_up | 0 | 0 | 0.5 |
  | ledge_down | 0 | 0.5 | 1.0 |
  | pits | 0 | 0.75 | 1.0 |
  | gravel | 0 | 0 | 0 |
  | rubble | 0 | 0 | 0 |

- **Disk:** 28 GB free; the DDC cap stays at 100 GB. Step 6 starts now and will stop at the queue's 20 GB disk floor
  (≈ 10 rows) unless the user frees space first. The rows are reversible, and the completed ones are reused.

### 2026-11-10 21:20–21:30 — disk freed (user: "Delete both")

- **Site chain raw episodes deleted.** All 86 `raw/` directories under `parkour/outputs/site_teacher/chain/`
  (`episode_*.npz` only), 39 GB. Every summary, metrics, run_meta, tp_row and code_state file is kept, as is
  `site_teacher/rows`.
- **Kit DerivedDataCache emptied** (95 GB; `isaac_venv/.../isaacsim/kit/cache/DerivedDataCache`). The baseline queue
  was first paused between rows with STOP_QUEUE, after 5 rows (RTQ_DONE fail=0), and no Isaac or Kit process was
  running. Kit regenerates the cache, so the first launches are slower.
- **Free disk 28 → 156 GB.** The baseline resumes; its 5 completed rows (at the pinned build) are reused. The DDC cap
  is back to the default 80 GB.

### 2026-11-10 21:25–23:39 — step 6 DONE (TDR baseline) and step 7 STOP 0 (gates frozen)

- **Step 6.** 43 steps, fail = 0. TDR (03c3e472) at the pinned build: 21 kind rows at es3 and 19 retention rows
  (`results/step6/baseline.json`).
- **TDR es3 success** (complete AND crossed; n = 100 per row):

  | Kind | b1 | b2 | b3 |
  |---|---|---|---|
  | slopes | 1.00 | 0.70 | 0.16 (stuck 0.09) |
  | hills | 1.00 | 1.00 | 1.00 |
  | ledge_up | 0.98 | 0.41 (stuck 0.46) | 0.00, complete 0.76, all stuck (above envelope) |
  | ledge_down | 1.00 | 0.37 | 0.00 (above envelope) |
  | pits | 0.99 | 0.55 | 0.01 (above envelope) |
  | gravel | 1.00 | 1.00 | 1.00 |
  | rubble | 1.00 | 1.00 | 0.79 |

- **TDR retention** (completion): field_hard es1–4 0.9475; field_light 0.995; ring_obst 1.00; field_turn 0.99;
  field_light L090 0.995; L135 0.96; ×0.7 L090 0.93; suite 1.00; side22 1.00.
- **Step 7, STOP 0.** `make_thresholds.py` wrote `thresholds_rough_v1.json` (sha ce45e01b…) by the frozen rule
  (decide-record-notify):
  - **gated (5):** slopes b2 (TDR 0.70), ledge_up b2 (0.41), ledge_down b2 (0.37), pits b2 (0.55), rubble b3
    (0.79, flagged borderline);
  - **solved (2):** hills and gravel. They enter R on b3 (TDR 1.00).
  - Decision **proceed** (5 ≥ k = 2).
  - The literature's prediction holds: TDR already handles hills and gravel blind.
- **Next:**
  - step 8 training launches now;
  - the plan's fable·high check of the gate bands and statistics runs in parallel on CPU (no new-teacher number
    exists before step 9). A thresholds problem it finds is fixed and re-frozen before step 9.

### 2026-11-11 00:00 — step 7 fable·high check: CONFIRMED (`wf_27ef0fed-687`, model claude-fable-5-2)

- **Re-derivation.** All 21 TDR es3 kind rows, recomputed from the raw stamps with independent code, equal
  baseline.json and thresholds_rough_v1.json exactly (success, completion, stuck, crossed, r_max). The 9 retention
  completions match. `make_thresholds.py --check` regenerates the file byte-identically (ce45e01b). Success is
  insensitive to ±0.2 m on the radii.
- **Rule application.**
  - Above-envelope = the three b3 bands, from the ladder bottoms (0.188 / 0.218 / 0.222 m). The closest
    in-envelope bottom is 0.126 m.
  - Gate bands and solved R bands follow the rule; k = 2; proceed. The file's G and R constants equal the PLAN's.
- **No blocking findings.** Six minor ones, all step-9 report items; no change to the frozen file.
- **Pre-registered step-9 report lines** (before any new-teacher number; none changes a gate or decision rule):
  1. **Completion beside success.** Per gated kind, the paired completion difference with its max(0.05, 2·SE)
     margin is printed beside the G line. Any kind that gains on success while completion drops past that margin
     is named in the verdict, e.g. ledge_up b2, where TDR is 41 success / 46 stuck / 13 fall.
  2. **Rubble b3** (TDR es3 79 / 100, borderline; Wilson 0.70–0.86): TDR's es1–2 rubble b3 success is reported
     beside the es3 pick. If G passes only because rubble is one of exactly two gaining kinds, the verdict says so.
  3. **Steps and potholes split at 0.14 m.** Success on the ledge_up, ledge_down and pits gate bands is split at
     0.14 m step height or depth, per episode from the pad level, for both teachers. 57–85 % of each band lies
     above 0.14 m, and TDR's failures are almost all there: ledge_up b2 34 / 40 succeed below 0.14 m against 7 / 60
     above. A gain there is read as above-0.14 m behaviour.
  4. **Falls are tilt-limit tips.** On these rows a fall is a 0.5 rad tilt-limit termination at walking height, not
     a collapse. The slopes up / down split is reported for both teachers: TDR's slope falls are mostly on the
     down (raised-pad) variant, 25 / 50 against 5 / 50 at b2.
  5. **Solved kinds' R rows sit at TDR 1.00.** hills b3 and gravel b3 are read as "≤ 10 failures in 200" (190 / 200
     passes, 189 fails). If TDR drops below 1.00 on es1–2, the line moves with it.
  6. **Hash checks**, confirmed by hand at step 9 and recorded in the verdict entry:
     - rows_rt.json sha256 = 0b038f03…;
     - each gated row's cross_r = the file's rule.cross_r;
     - step-9 stamps at code_sha 89c1916b / eval_sha 9fe964d7;
     - no change under tools/pipeline or the campaign manifests since c2054a8c.

### 2026-11-11 00:37 — step 8 early read at iteration 1500 (report only)

- **Training fall share per kind**, iterations 300–500 → 1300–1500:

  | Kind | 300–500 | 1300–1500 |
  |---|---|---|
  | flat | 0.172 | 0.196 |
  | field_sparse | 0.227 | 0.292 |
  | field_dense | 0.229 | 0.450 |
  | slopes | 0.604 | 0.568 |
  | hills | 0.292 | 0.232 |
  | ledge_up | 0.608 | 0.527 |
  | ledge_down | 0.799 | 0.788 |
  | pits | 0.716 | 0.792 |
  | gravel | 0.410 | 0.368 |
  | rubble | 0.373 | 0.343 |

- **Confounded by level.** Main's rule promotes falls, so columns climb to and wrap around their top levels within
  the first few hundred iterations (TDR's field levels 1.04 → 6 by ≈ 300 iterations). A rising fall share on the
  field columns therefore mixes harder levels with any forgetting. TDR's own training ran a 0.30–0.36
  obstacle-tile fall share. Not a stop, and no remedy, per settled 6.
- **Watch item for step 9:** field_dense 0.23 → 0.45 and pits 0.72 → 0.79. The retention rows (field_hard es1–4,
  field_light) and the pits gate band will answer them deterministically.

### 2026-11-11 03:17 — step 8 DONE; step 9 launched

- **Training `RT_s19`** ran 23:43 → 03:17 (12 859 s, 3.6 h, slower than the ≈ 2.5 h planned). Final
  `parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_34990.pt`, sha256 d58ceb34…, iteration
  34990. PF1–PF14 all PASS; `results/runs/RT_s19.json` status ok.
- **Step 9 gate block launched 03:18.** G.check, G.thresholds (frozen file regenerates) and G.ident (record ok, sha
  and iteration match) all rc 0. 47 rows:
  - both teachers at es1–2 on slopes b2, ledge_up b2, ledge_down b2, pits b2 and rubble b3 (G), and on hills b3 and
    gravel b3 (R solved);
  - the new teacher's 19 retention rows.
  - Then G.audit and G.read. Disk 120 GB free.

### 2026-11-11 06:25 — step 9 verdict: NOT ACCEPTED (G FAIL, R FAIL)

- **Gate block** 03:18 → 06:23, `RTQ_DONE gate fail=0`. 47 rows. Read-out in `results/step9/gates.json`:
  - new teacher RT_s19 `model_34990.pt`, sha d58ceb34;
  - TDR sha 03c3e472.
- **G per kind** (success, es1–2, n = 200 each; gain needs diff ≥ max(0.10, 3·SE) and CI lower bound ≥ 0.05;
  regress means diff < −max(0.05, 2·SE)):

  | Kind / band | New | TDR | Diff | 95 % CI | Read |
  |---|---|---|---|---|---|
  | slopes b2 | 0.895 | 0.745 | +0.150 | +0.091 … +0.209 | **GAIN** (falls 21 vs 51) |
  | ledge_up b2 | 0.445 | 0.430 | +0.015 | −0.055 … +0.085 | — (stuck 0.495 vs 0.48) |
  | ledge_down b2 | 0.320 | 0.405 | −0.085 | −0.152 … −0.018 | **REGRESS** (falls 136 vs 119) |
  | pits b2 | 0.515 | 0.475 | +0.040 | −0.039 … +0.119 | — |
  | rubble b3 | 0.840 | 0.825 | +0.015 | −0.052 … +0.082 | — |

  1 kind gains, and k = 2 → **G FAIL**.
- **R completion:** all 7 rows pass, every difference ≤ noise.

  | Row | Diff |
  |---|---|
  | field_hard | +0.0025 |
  | field_light | −0.010 |
  | ring | 0 |
  | field_turn | +0.010 |
  | L090 | −0.015 |
  | L135 | +0.005 |
  | ×0.7 L090 | −0.005 |

- **R solved:** both pass. hills b3 is 0.995 vs 0.98; gravel b3 is 1.00 vs 1.00.
- **R tracking (v1.2 suite, es1–2, drift 0): FAIL on 4 of 22 lines.**

  | Line | es1 head vs TDR | es2 head vs TDR | Margin |
  |---|---|---|---|
  | track_err back | 0.109 vs 0.018 | 0.109 vs 0.019 | 0.0617 |
  | yaw_progress turn | 0.147 vs 0.172 | 0.147 vs 0.171 | 0.0066 |

  - move_turn passes narrowly: worse by 0.034 / 0.033 against a margin of 0.0367.
  - fwd, side and diag tracking, survival and facing_closed pass.
  - side22 is FAST/FAST, not lost.
  - The drift term would not rescue these lines; the decisive read is drift 0.
- **Decision row:** "G fails: not accepted; CPU diagnosis; HALT, the user decides." TDR stays the teacher of record.
  Nothing is re-distilled, re-paired or baked. RT_s19 stays on disk as a run, not a policy.
- **Running:**
  - workflow `wf_107a6868-957`:
    - fable·high independent re-derivation of every G and R line from the raw stamps;
    - the hash checks (pre-registered line 6);
    - pre-registered report lines 1–5;
    - CPU mechanism diagnosis.
  - The results are recorded in the next entry, before the HALT question.

### 2026-11-11 06:50 — step 9 checks done; verdict stands NOT ACCEPTED; HALT for the user

Checks behind this entry: the fable·high independent re-derivation, the hash checks, pre-registered report lines 1–6 and the CPU diagnosis, all from workflow `wf_107a6868-957`. Nothing was edited or launched.

- **Re-derivation** (claude-fable-5-2) reproduces every G and R value, flag and status in `results/step9/gates.json` to 1e-12.
  - Method: its own code for G, R completion, R solved and the 22 suite lines. Its own bootstrap, paired on env_index (4000 draws, seed 20261016), agrees with the analytic SE. The repo's `zero_kg_check`, run on inputs it chose, gives 22 / 22 lines equal.
  - Not re-derived independently: the side22 class and the survival lines' binomial-guard R. Neither is near a decision.
  - Not binding: `readout.g_line` adds an "and not le_noise" clause (|diff| ≤ 0.04) that the frozen rule text does not have. It cannot bind here (gain floor 0.10, regress floor 0.05). Reconcile it before the readout is reused.

- **G: FAIL** (success, es1–2, n = 200 per teacher, identical spawns).

  | Kind / band | RT | TDR | Diff | Needed | Discordant (RT-only / TDR-only) | Read |
  |---|---|---|---|---|---|---|
  | slopes b2 | 0.895 | 0.745 | +0.150 | gain 0.100 | 36 / 6 | **GAIN** |
  | ledge_up b2 | 0.445 | 0.430 | +0.015 | gain 0.108 | 27 / 24 | — |
  | ledge_down b2 | 0.320 | 0.405 | −0.085 | regress margin 0.069 | 15 / 32 | **REGRESS** |
  | pits b2 | 0.515 | 0.475 | +0.040 | gain 0.120 | 36 / 28 | — |
  | rubble b3 | 0.840 | 0.825 | +0.015 | gain 0.102 | 24 / 21 | — |

  - n_gain = 1 < k = 2, so G fails even without the regression.
  - The ledge_down regression is the thinnest decisive call: 0.016 past its margin. Per seed it is es1 −0.12 and es2 −0.05; the exact sign test gives p ≈ 0.02.
  - Moving the radii by ±0.2 m changes no call.

- **R: FAIL.**
  - Completion: 7 / 7 rows pass, every difference ≤ 0.015.
  - Solved kinds: 2 / 2 pass.
  - Tracking: 4 of 22 lines fail.

    | Line | RT, es1 / es2 | TDR, es1 / es2 | Worse by, es1 / es2 | R |
    |---|---|---|---|---|
    | track_err back | 0.109 / 0.109 | 0.018 / 0.019 | 0.092 / 0.089 | 0.0617 |
    | yaw_progress turn | 0.147 / 0.147 | 0.172 / 0.171 | 0.024 / 0.024 | 0.0066 |

  - move_turn passes narrowly: worse by 0.034 / 0.033 against 0.0367.
  - The reading with the 6000-iteration drift term fails the same 4 lines.

- **Decision row:** "G fails: not accepted; CPU diagnosis; HALT, the user decides."
  - TDR (03c3e472) stays teacher of record.
  - RT_s19 `model_34990.pt` (d58ceb34) stays on disk as a run, not a policy.
  - Nothing is re-distilled, re-paired, baked or pushed.

- **Pre-registered report lines.**
  1. **Completion beside success** (paired; margin max(0.05, 2·SE)).
     - slopes: 0.895 vs 0.745 (+0.150).
     - ledge_up: 0.940 vs 0.910 (+0.030, within margin 0.050); stuck 0.495 vs 0.48.
     - ledge_down: 0.320 vs 0.405 (−0.085, past margin 0.069). Completion equals success here, since 0 episodes were stuck.
     - pits: 0.525 vs 0.475 (+0.050, within margin 0.080).
     - rubble: 0.920 vs 0.915 (+0.005).
     - **No kind gains on success while its completion drops past the margin.**
  2. **Rubble b3.**
     - TDR on es1–2 reads 0.825 (83 / 82 per seed), against its es3 pick of 0.79 (inside the Wilson interval 0.70–0.86).
     - RT reads 0.840 (80 / 88). The difference changes sign between seeds (−0.03 / +0.06).
     - G did not pass, so the "passes only because of rubble" caveat does not arise.
  3. **Split at 0.14 m** (success, RT / TDR).

     | Kind | Share above 0.14 m | Below 0.14 m | Above 0.14 m |
     |---|---|---|---|
     | ledge_up | 113 / 200 | 67 / 71 of 87 | 22 / 15 of 113 |
     | ledge_down | 162 / 200 | 23 / 38 of 38 | 41 / 43 of 162 |
     | pits | 175 / 200 | 24 / 20 of 25 | 79 / 75 of 175 |

     - ledge_up above 0.14 m: 83 stuck for each teacher; falls 8 vs 15.
     - ledge_down: RT has 15 falls below 0.14 m on drops TDR clears 38 / 38. These account for −15 of the −17 regression.
     - Pit depths are estimated per tile from the floor raycast samples.
     - The pattern holds for cuts from 0.13 to 0.15 m.
     - So no kind gained above 0.14 m, and the regression sits on easy drops.
  4. **Falls are tilt-limit tips.**
     - Every non-completer on the five gate rows (RT 280, TDR 310) ends as 'fall' at a last-sample tilt of 0.479–0.500 rad, with the root 0.85–1.31 m above the local terrain.
     - Slopes, down (raised pad): TDR 54 / 100 succeed with 46 falls (43 in the seg-2 side hold); RT 81 / 100 with 19 falls (15 in seg 2, 4 in seg 1).
     - Slopes, up: TDR 95 / 100, RT 98 / 100.
     - 27 of the +30 slope successes come from the descent, and both seeds agree.
  5. **Solved ceiling.**
     - hills b3: TDR reads 0.98 (196 / 200) on es1–2, so the line moves to 0.93 (≥ 186 / 200). RT reads 0.995: PASS.
     - gravel b3: TDR reads 1.00, line 0.95. RT reads 1.00: PASS.
  6. **Hash checks: all pass.**
     - rows_rt.json sha256 0b038f03… equals pin.json.
     - Each gated row's cross_r equals rule.cross_r: slopes 3.0, the other gated kinds 3.2, the solved kinds 3.0.
     - 47 / 47 RT9 stamps sit at code_sha 89c1916b and eval_sha 9fe964d7, with a clean tree and rc 0. The 40 RT6 stamps are at the same build.
     - Nothing changed under tools/pipeline, the manifests, scripts/, envs/ or the thresholds since c2054a8c. c2054a8c..HEAD touches only CHARTER.md, results/runs/*.json and results/step9/gates.json.
     - Stamp prefix equals the run_meta checkpoint sha on 47 / 47 (33 at d58ceb34, 14 at 03c3e472). Both checkpoints were re-hashed.
     - The thresholds file is ce45e01b, and its recorded script shas equal the current scripts.

- **Diagnosis** (CPU, report only; the mechanisms are supported, not proven).
  - **The slopes gain** is a lower, more level stance while traversing across the downslope. In the seg-2 hold on the descent:

    | Metric | RT | TDR |
    |---|---|---|
    | pitch (rad) | 0.119 | 0.146 |
    | body height (m) | 1.073 | 1.106 |
    | fall share | 9 % | 42 % |

  - **The ledge_down regression** is a nose-down lean carried into the edge. The body tips while crossing the rim (root at r 1.6–1.9 m; pad edge at 1.7 m).
    - Approach speed is equal: 0.311 vs 0.302 m/s.
    - Pitch with the first foot over the drop: 0.10–0.12 vs 0.06 rad. At r = 1.0 m: 0.165 vs 0.127.
    - The lean is general. On the flat suite, fwd pitch is +0.110 vs +0.063 and back −0.076 vs −0.033; it shows on the field rows too.
    - RT's drop-tolerance curve sits 1–2 cm lower in the 0.12–0.16 m range. At 0.16 m and above, RT reads 12 / 117 and TDR 10 / 117.
  - **Back overshoot.**
    - Back speed is 1.109 vs 1.018 of command: 0.244 m/s at a 0.22 command, against a training back semi-axis of 0.25.
    - Survival is 1.0, and it does not depend on load: 1.09–1.10 at 0, 90 and 135 kg.
    - Field rows are over command in every category (fwd 1.07–1.11).
    - RT's params confirm that the reward barely prices it. Progress (weight 0.6, absolute clamp 1.05 m/s) against L1 tracking (−2.0) nets about −1.4 per m/s.
    - It has happened before:

      | Run | Back speed / command |
      |---|---|
      | S1 s18 (phase 2) | 1.087 |
      | Task-4 pilot PP_p3a | 1.162 |
      | site p3b | 1.130 |
      | RT | 1.109 |

      The `back_overshoot` follow-up has been open and untested since 2026-10-08.
  - **Turning.**
    - Yaw rate at a saturated 0.3 rad/s command drops from 0.171 to 0.146; time to 90 % rises from 8.1 to 9.4 s.
    - The turning steps cost more per radian:

      | Per radian of turn | RT | TDR |
      |---|---|---|
      | swing apex | 0.143 | 0.125 |
      | taps | 19.5 | 16.2 |
      | work | 2290 | 2035 |

    - PLAN item 19 predicted about 0.012 of continuation drift on this line; the observed loss is 0.024.
  - **Exposure.**
    - Main's ratio rule saturated the levels in about 500 iterations. On ledge_down, 87 % of episodes promoted while 78 % fell. As a result:
      - about 30 % of each kind's ≈ 26 envs sat in b2, and about 40 % in b3, which is above the envelope for ledges and pits;
      - training crossings stayed flat (ledge_down 0.135; ledge_up fell from 0.19 to 0.03).
    - Flat and field shares each fell from 0.5 to 0.15. The orientation penalty applies only on `parkour_flat` tiles, so it now reaches 15 % of envs.
    - Other drifts: foot_idle −0.17 → −0.46 (TDR −0.09); noise std 0.807 → 0.845; flat-tile fall share 0.135 → 0.255.
    - Payload and friction match TDR.

- **Honest reading.**
  - The campaign bought one real terrain skill, slope descent (46 → 19 falls in 100), at the cost of an easy-terrain skill: 12–14 cm drops go from 38 / 38 to 23 / 38. It also failed two flat tracking lines.
  - ledge_up, pits and rubble did not move. In their bands, the above-0.14 m mass is unchanged, and below 0.14 m both teachers are already near the ceiling. This is consistent with the literature review's blind envelope and with a curriculum that kept only about 30 % of each kind in the band.
  - A blind fine-tune under these gates shows no route to a second gaining kind.
  - The lean and the back overshoot point to the recipe. The overshoot has now appeared in four fine-tunes.


- **Checked by the parent before recording.**
  - `reward_orientation` (parkour_isaaclab/envs/mdp/rewards.py:391) zeroes on every tile not named `parkour_flat`, and RT's params/env.yaml uses that func.
  - `KRABBY_ORIENT_ALL_TILES` / `KRABBY_ORIENT_W` exist in run_candidates.py.
  - 61 checkpoints, every 100 its (model_29000 … model_34990), are on disk in this worktree.
- **HALT: options put to the user** (recommendation: close as NOT ACCEPTED, plus a report-only checkpoint sweep while the close is written):
  - **Close as NOT ACCEPTED (step 10)** (≈ 0.5–1 (0 GPU) h wall-clock): Write the REPORT, CHANGELOG, LOG and memory from the CHARTER and gates.json, then make a local commit; no push. TDR stays teacher of record. RT_s19 (61 checkpoints, in this worktree's logs/) is kept on disk as a run. Record the lessons: the lean, back_overshoot now seen four times, curriculum saturation, and gate bands that sit mostly above 0.14 m.
  - **Checkpoint sweep, report-only appendix** (≈ 1.3 (≈ 1.0 GPU at ≈ 4 min/row, plus about 0.3 to set up and read on CPU) h wall-clock): Run model_30000, model_31500 and model_33000 (model_34990 is already done) on suite es1, slopes b2 es1–2 and ledge_down b2 es1–2: 15 rows at the pinned build, reusing TDR's existing stamps. It shows whether the lean, the back overshoot and the turning loss grow with iterations or appear early, and whether the slopes gain comes before the ledge_down loss. It does not change the verdict. Launch it without editing the three scripts whose hashes the thresholds file records (make_thresholds.py, readout.py, rt_spec.py).
  - **Env-only follow-on arm (tilt guard, retention share, turn dose)** (≈ 7–8 (plan and audit ≈ 1–1.5, CPU replay gate ≈ 0.5, smoke ≈ 0.3, training 3.6, 33 new-teacher rows ≈ 2.2) h wall-clock): A new pre-registered arm from TDR on the same frozen thresholds_rough_v1, with TDR's stamps reused. It changes three things, all through existing knobs: `KRABBY_ORIENT_ALL_TILES=1` with a replay-gated `KRABBY_ORIENT_W`; flat + field ≈ 0.4 of the mix; a turn dose (`KRABBY_YAW_W` or the CMD_POOL turn share). It targets the lean, ledge_down and both R lines, but no second gaining kind is in view, so under k = 2 it most likely ends NOT ACCEPTED again.
  - **Code follow-on: targeted curriculum and priced overshoot** (≈ 11–15 (code, tests and review 3–5; neutrality smoke 0.7; TDR's 33 rows at the new build ≈ 2.2, unless a bitwise neutrality row covers reuse; arm ≈ 6) h wall-clock): The option-3 levers, plus code: a per-kind ladder clamp (d ≤ 0.6 for ledges and pits) or promote on crossing and demote on hazard falls (`KRABBY_SITE_PROMOTE` is unported), and a relative progress clamp (≤ 1.0 × |cmd|) or a one-sided overshoot term. This is the best blind attempt at a second gaining kind; the code change moves the pin.
  - **P3 perception for the above-0.14 m mass** (≈ 20–30 over several days (port 4–8 agent hours, v2 surgery and training ≥ 6–10 GPU hours, eval and twin rows ≥ 3) h wall-clock): The set-aside P3 set: scan-ablation port, `_flat` twins and recorder, then a v2 scan by zero-init surgery or a v2 student, aimed at ledge_up, ledge_down and pits above 0.14 m. Cross-campaign hint, not isolated: the site teacher (v2 scan, edge and void terms) gained +0.30 on a similar ledge_down band where RT regressed.
  - **P4 lost-foothold levers** (≈ 11–15 (code and neutrality plus one arm) h wall-clock): A looser pitch termination in training only, ≥ 4 loaded feet, and slowing stance near drops. All 590 non-completions on the gate rows are tilt-limit tips at walking height. Eval keeps the 0.5 rad termination, so recovery past 0.5 rad earns no credit there.
  - **Partial use of the slopes skill** (0 for (a); as option 3 or 4 for (b) and (c), plus ≈ 1 for fresh TDR seeds in (c) h wall-clock): RT_s19 has no route to acceptance: it fails R as well as G. Realistic uses: (a) keep it as the slope-descent reference run; (b) warm-start option 3 or 4 from it; (c) a slopes-only teacher as a new campaign with G set before any number exists, read on fresh seeds.


### 2026-11-11 08:15 — checkpoint sweep (report only; table in `results/sweep/sweep.json`)

Report only. The step-9 verdict NOT ACCEPTED is frozen and this entry changes nothing. TDR stays teacher of record, and RT_s19 stays a run, not a policy. Nothing was trained, baked, re-distilled or pushed.

- **Sweep.** Three intermediate RT_s19 checkpoints were evaluated:
  - checkpoints: `model_29500` (6d1f29be), `model_31000` (6808f24c) and `model_33000` (c495e783), all from `parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19`;
  - rows per checkpoint: `rt_slopes_b2` es1–2, `rt_ledge_down_b2` es1–2 and `R_suite` es1 only;
  - driver: `results/sweep/rt_sweep.py` (run from scratch, outside the repo, using the campaign queue; copied in after the run). Log `outputs/omni_plan/rough_terrain/logs/queue_sweep.log`, 06:52 → ≈ 07:55, 15 rows, `RTQ_DONE sweep fail=0`;
  - stamps: `lr/RTSW_<sha8>_*`;
  - comparison rows: 34990 from `RT9_d58ceb34_*`, TDR kind rows from `RT9_03c3e472_*`, and the TDR suite from `RT6_03c3e472_R_suite` (seed001; TDR by run_meta sha);
  - build: every stamp is at code_sha 89c1916b and eval_sha 9fe964d7, clean, with the same composed env, args and level file as the TDR row. Spawn xy and pad level are bit-identical per (es, env_index).
- **Readers.** Two independent CPU readings were taken, A (claude-opus-5-5) and B (claude-fable-5-2), plus a reconciler re-count.
  - All three agree on every number, and all reproduce the ten step-9 values at 34990 and TDR.
  - The only differences are bootstrap seeds in the 4th decimal of the SEs. No call changes.
  - Wording settled from the counts: the ledge_down loss below 0.14 m is *mostly* late, not entirely. 31000 reads 33/38 (es2 14/18) before 33000 recovers to 37/38.

  | Checkpoint (its after TDR) | slopes b2 | down / up | ledge_down b2 | < 0.14 / ≥ 0.14 m | back ratio | back worse (R 0.0617) | yaw_prog turn | yaw worse (R 0.0066) | move_turn | pitch fwd / back |
  |---|---|---|---|---|---|---|---|---|---|---|
  | TDR 28991 | 0.745 | 54 / 95 | 0.405 | 38/38 / 43/162 | 1.0175 | — | 0.1715 | — | 0.7653 | +0.063 / −0.033 |
  | 29500 (+509) | 0.935 | 94 / 93 | 0.590 | 36/38 / 82/162 | 1.0791 | 0.0616 (in by 0.0001) | 0.1646 | 0.0069 (past by 0.0003) | 0.7513 | +0.061 / −0.022 |
  | 31000 (+2009) | 0.890 | 79 / 99 | 0.460 | 33/38 / 59/162 | 1.0481 | 0.0306 | 0.1650 | 0.0065 (in by 0.00006) | 0.7310 | +0.066 / −0.048 |
  | 33000 (+4009) | 0.845 | 74 / 95 | 0.495 | 37/38 / 62/162 | 1.1044 | 0.0869 PAST | 0.1634 | 0.0081 PAST | 0.8539 | +0.082 / −0.033 |
  | 34990 (+5999) | 0.895 | 81 / 98 | 0.320 | 23/38 / 41/162 | 1.1093 | 0.0918 PAST | 0.1474 | 0.0241 PAST | 0.7314 | +0.110 / −0.076 |

- **Paired against TDR** (n = 200, gain/regress rule as in step 9; descriptive only).

  | Checkpoint | slopes b2 diff (SE, lb) | ledge_down b2 diff (SE) |
  |---|---|---|
  | 29500 | +0.190 (0.033, +0.125) GAIN | +0.185 (0.035) reads as gain |
  | 31000 | +0.145 (0.032, +0.083) GAIN | +0.055 (0.033) — |
  | 33000 | +0.100 (0.033, +0.036) — (lb < 0.05) | +0.090 (0.030) — |
  | 34990 | +0.150 (0.031, +0.090) GAIN | −0.085 (0.034) REGRESS |

  - Other es1 lines (fwd, side and diag track_err, turn facing_closed, survival 1.0) are inside R at every checkpoint.
- **Reading.**
  - The slopes gain is an early step, complete by 29500 and all down-variant; it does not keep growing.
  - Ledge_down sits above TDR through 33000. The regression, the below-0.14 m loss, the pitch lean and the yaw_progress drop land together between 33000 and 34990.
  - The back overshoot is already +0.06 at 29500, and yaw sits on its floor from the start. An earlier stop would not obviously clear the tracking lines.
  - 31000 is the only checkpoint inside every es1 line, with a yaw margin of 0.00006.
  - Ledge_down swings by about 4 paired SE between checkpoints.
  - These calls rest on one suite seed, and choosing the best of four checkpoints on the same episodes is selection. A shorter fine-tune (about 500–2000 its) is a candidate for a fresh run on new eval seeds plus the es2 suite, not a result.
- **Open:** at 33000 only 73 fwd holds fall within ±20° of the commanded heading, which may mean heading drift. Not investigated.
- **Scratch** (outside the repo): reader A `step9diag/sweep_opus55/`, reader B `step9diag/sweep_reader2/`, reconciler `step9diag/sweep_reconcile/kinds.py`.
- **No verdict change.** The HALT for the user stands.

### 2026-11-11 11:05 — Amendment A (short arm): user decisions, audit, pre-registration

- **User decisions.**
  - At the step-9 HALT: "Plan short arm".
  - After the draft: "Audit, up to 2 passes". The two defaults offered with it stand: the 4-candidate pool, and no fifth mix-change candidate.
  - After the audit: "Approve + es2 suite". The selection screen reads `R_suite` on es1–2, not es1 only.
- **Design.** A judge panel (`wf_9b4e3a73-705`, 3 opus·high designers, opus·xhigh judge) produced it. Full text: `short_arm/ARM.md`, pre-registered by this commit.
  - Four candidates, same mix (`RT_train.env` unchanged):

    | id | checkpoint | sha |
    |---|---|---|
    | c1 | RT_s19 model_29500 | 6d1f29be |
    | c2 | RT_s19 model_30000 | a346185c |
    | c3 | new run RT_s20s (TDR resume, seed 20, 1010 its) model_29500 | — |
    | c4 | RT_s20s model_30000 | — |

  - A mechanical pick on es1–2 (slopes b2, ledge_down b2, suite).
  - Slack z = min over 22 suite lines of (margin − worse_by) / se_diff.
  - One confirm on fresh seeds es5–8 against TDR's RTF rows, G first.
  - Gates (thresholds_rough_v1, ce45e01b) and every frozen script are byte-unchanged; the build is unchanged.
- **Audit** (`wf_41c270b0-953`, `wf_03db3f66-e66`), 2 passes:
  - **Pass 1** found 4 blocking issues; all are fixed.
    1. The pick rule is now scaled by noise; the yaw line's per-seed noise is ≈ 0.0007–0.0009, not "seed-stable".
    2. The fable recompute now gates the selection commit.
    3. The HALT_R remedy now reads on es9+ with fresh TDR rows.
    4. The readout now fails closed on the R-tracking inputs.
  - **Pass 2** (opus·xhigh plus an independent fable·high; both found the same issue): the G FAIL branch skipped step 10. It now runs the fable re-derivation, hash checks and report lines on the confirm-1 rows. This fix was not re-audited.
  - Deferred findings are listed in ARM.md under "Left for implementation".
- **Pre-run.** TDR's 33 rows at es+4 (`RTF_03c3e472_*`, scratch driver `rt_tdr_fresh.py`) started 09:51 under the never-idle rule. They are unread, and stay so until `selection.json` is committed.
- **Odds** (stated before any number): ACCEPT ≈ 8 %, G pass ≈ 0.45, HALT_R ≈ 0.35. c1 sits on both the yaw and the back line.

### 2026-11-11 11:50 — Amendment A steps 2–3: scripts built, CPU checks pass

- **Workflow** `wf_72a409ad-c6b`:
  - Builders: sonnet·high (`rt_short_train.py`, `rt_short.py`, `rt_short_launch.sh`) and opus·high (`readout_fresh.py`, `select.py`).
  - Reviews: fable·high on the decisive reader, opus·high on the queue. Neither found a blocking issue.
  - Checks: sonnet·high.
- **Step-3 checks: 8 of 8 pass.**
  - Build check OK at 89c1916b / 9fe964d7, clean.
  - Thresholds regenerate byte-equal.
  - `readout_fresh --identity-check` reproduces step-9 gates.json: 1214 numbers, 0 differences, same verdict.
  - 62 / 62 fail-closed tests pass.
  - `select.py --check` reproduces the sweep's 29500 row (24 / 24) and step 9's 22 suite lines exactly. c1's es1-only z is −0.398, bound by yaw.
  - Dry passes give 1 train step and 19 / 12 / 21 slots, with no collisions.
  - Trainer preflight: RT_s20s, 1010 its, seed 20, final model_30000.pt, PF1–PF7 pass.
  - No code path opens an RTF outcome before the guard.
- **Naming.** Selection rows are `RTA_<sha8>_*` and confirm rows `RTC_<sha8>_*`. Readers key on the run_meta sha and seed, not on the name.
- **Parent edit after the checks:** `select.py` refuses a non-finite suite line (NaN would make `min()` order-dependent). `--check` was re-run and passes.
- **Minor review items not applied, handled by procedure:**
  - confirm-2 launches only after `g1.json` reads G PASS;
  - match the `G=PASS` sentinel, never the exit code;
  - keep `RT_TP_OUT` unset;
  - the step-10 hash line carries rows_rt 0b038f03 and the cross_r check.

### 2026-11-11 14:50 — Amendment A steps 4–6: RT_s20s trained, selection done, pick c1 (fable recompute AGREES)

- **Pre-run RTF.** TDR at es5–8, 33 rows, ran 09:51 → 12:07 with fail=0. Still unread.
- **Step 4: RT_s20s.**
  - TDR resume, seed 20, 1010 its, 256 envs, `RT_train.env` unchanged. Ran 12:08 → 12:47 (2327 s); PF1–PF14 PASS.
  - c3 = model_29500, sha 17fb0214. c4 = model_30000, sha 1245089a.
  - Record: `results/runs/RT_s20s.json`.
- **Step 5: selection rows.** 19 rows (`RTA_*`), 12:47 → 14:31, fail=0.
- **Step 6: pick.** Read on es1–2 against TDR, n = 200 per kind; z is the minimum over 22 suite lines.

  | id | slopes b2 diff | ledge_down b2 diff | z (binding line) | Q | E |
  |---|---|---|---|---|---|
  | c1, RT_s19 +509 | +0.190 GAIN | +0.185 GAIN | −0.40 (yaw es1) | yes | no |
  | c2, RT_s19 +1009 | +0.190 GAIN | −0.050 | −3.67 (yaw es1) | no | no |
  | c3, RT_s20s +509 | +0.065 | +0.145 GAIN | −23.9 (yaw es2) | no | no |
  | c4, RT_s20s +1009 | +0.130 GAIN | +0.040 | −33.8 (yaw es1) | no | no |

  - **Pick c1** (6d1f29be): tier 3, flagged "R tracking predicted marginal".
    - c1's es1 back line has z +0.035 of slack.
    - c1's es2 yaw line has z +0.40.
  - **fable·high independent recompute** (`wf_13ec06c2-187`, claude-fable-5-2): **AGREES**.
    - Same Q, E, tier and pick, max difference 0.0 on every number.
    - Its own stamp walk and success vectors; 30 stamps, es1–2 only, no RTF/RTC opened.
  - **Seed-20 replication: no.** RT_s20s gained one of the two kinds at each length, never both, and its turning is far worse (z −24 / −34).
- `selection.json` is committed with this entry. confirm-1 (`sconf1`) launches next: 12 rows, c1 at es5–6.

### 2026-11-11 15:40 — Amendment A steps 7–8: confirm-1 done; G PASS on fresh seeds; suite fails 3 lines

- **confirm-1** (`sconf1`): 12 rows (`RTC_6d1f29be_*`) for the pick c1, real seeds es5–6. Ran 14:38 → 15:36, fail=0.
- **G read** (`readout_fresh --g1`, `results/short_arm/g1.json`). Pick c1 against TDR's RTF rows, which are read now for the first time; n = 200 per kind; 24 lookups, all at real es5–6. **G PASS.**

  | Kind | c1 | TDR | Diff | CI lower | Result | Falls (c1 vs TDR) |
  |---|---|---|---|---|---|---|
  | slopes b2 | 0.920 | 0.790 | +0.130 | +0.065 | GAIN | 16 vs 42 |
  | ledge_up b2 | 0.330 | 0.360 | −0.030 | — | no change | stuck 0.535 vs 0.555 |
  | ledge_down b2 | 0.520 | 0.345 | +0.175 | +0.109 | GAIN | 96 vs 131 |
  | pits b2 | 0.570 | 0.495 | +0.075 | −0.013 | no gain | — |
  | rubble b3 | 0.825 | 0.735 | +0.090 | +0.013 | no gain | — |

  - 2 kinds gain, none regresses, k = 2.
- **Suite tracking** (v1.2, drift 0): **FAIL on 3 of 22 lines.**

  | Line | Seed | Worse by | Margin |
  |---|---|---|---|
  | track_err back | es5 | 0.070 | 0.0617 |
  | track_err back | es6 | 0.064 | 0.0617 |
  | yaw_progress turn | es6 | 0.0070 | 0.0066 |

  - yaw_progress turn passes on es5: worse by 0.0052.
  - Back track_err is 0.081 / 0.080, against TDR's 0.011 / 0.016.
  - move_turn passes, worse by 0.002 / 0.000. facing_closed and survival pass.
  - fwd and side tracking are better than TDR's.
- **Per step 8**, confirm-2 (21 rows) launched at once at 15:37. Its rows inform the decision, but R already fails on tracking, so the expected outcome is **G passes, R fails → HALT_R**, the user decides.
- **Remedy by failing line** (pre-registered):
  - back: priced overshoot (a code change; the pin moves);
  - yaw: the env-only retention arm.
  - Either is read on es9+ with fresh TDR rows.
- **Housekeeping.** At 15:37 a mistyped shell command created an empty stray file, `experiments/2026-11-11_rough_terrain_placeholder` (0 bytes). It was removed at 15:45. It sits under the `experiments/20*` path the code_sha excludes, so no stamp's build or dirty flag moved; confirm-2's stamps are checked at read time anyway.

### 2026-11-11 16:55 — Amendment A step 9 (confirm-2) done; full read: HALT_R

- **confirm-2** (`sconf2`): 21 rows, ran 15:37 → 16:52, fail=0.
- **Full read** (`readout_fresh --full`, `results/short_arm/gates_fresh.json`): 104 lookups, all at real es5–8.
- **Verdict: HALT_R.**
  - G PASS.
  - R completion 7 / 7 pass: field_hard +0.040, field_light +0.015, ring 0, field_turn −0.005, L090 +0.010, L135 +0.040, ×0.7 L090 +0.055.
  - Solved kinds 2 / 2 pass: hills +0.015, gravel 0.
  - side22 FAST/FAST.
  - R tracking FAIL: back es5 and es6, yaw es6.
- Step 10 is running: the fable·high independent re-derivation, hash checks and report lines (`wf_50c2a6c8-2b5`).

### 2026-11-11 17:10 — Amendment A steps 10–11: verdict HALT_R stands (G PASS, R FAIL on tracking only); checks done; HALT for the user

The checks behind this entry ran in workflow `wf_50c2a6c8-2b5`: the fable·high independent re-derivation, the hash checks, pre-registered report lines 1–7 and a CPU mechanism read. Nothing was edited or launched. **All seeds below are REAL seeds.** The readout's labels `suite es1` and `suite es2` mean real es5 and es6.

- **Verdict: HALT_R** (frozen). Source: `results/short_arm/gates_fresh.json`, sha 43849afb, commit 9da889b6.
  - The pick, c1 = RT_s19 `model_29500` (6d1f29be, +509 its), is read against TDR `model_28991` (03c3e472). TDR is read from its RTF rows.
  - Decision row: "G passes, R fails: HALT_R; the user decides."
  - TDR + student c31000 + SH_RSD_TDR / SN_RSD_TDR stay the omni policy of record.
  - Nothing has been re-distilled, re-paired, baked or pushed.
- **Re-derivation** (claude-fable-5-2): **reproduces.**
  - Method: its own stamp walk, success rule, pairing and margins on the 66 raw stamps (33 RTC, 33 RTF). Suite lines go through the repo `zero_kg_check` on inputs it assembled. It also computed its own point estimates from `metrics/episode_NN.json`.
  - Result: 382 values equal to 1e-12, and the same verdict.

- **G: PASS.** Success at es5–6, n = 200 per teacher, paired on (es, env_index). 2 kinds gain and none regresses, against k = 2.

  | Kind / band | c1 | TDR | Diff | SE | Needed | CI lower | Read | es5 / es6, c1 vs TDR |
  |---|---|---|---|---|---|---|---|---|
  | slopes b2 | 0.920 | 0.790 | +0.130 | 0.033 | gain 0.100 | +0.065 | **GAIN** | 0.92 / 0.92 vs 0.77 / 0.81 |
  | ledge_up b2 | 0.330 | 0.360 | −0.030 | 0.035 | regress 0.070 | — | — | 0.35 / 0.31 vs 0.38 / 0.34 |
  | ledge_down b2 | 0.520 | 0.345 | +0.175 | 0.033 | gain 0.100 | +0.109 | **GAIN** | 0.53 / 0.51 vs 0.35 / 0.34 |
  | pits b2 | 0.570 | 0.495 | +0.075 | 0.045 | gain 0.135 | −0.013 | — | |
  | rubble b3 | 0.825 | 0.735 | +0.090 | 0.039 | gain 0.118 | +0.013 | — | |

  - Some episodes end within 0.02 m of a crossing radius (slopes 3, rubble 5, ledge_up 2, pits 1). None of them can flip a flag.
  - A plain analytic paired SE agrees with the bootstrap SE within 1.1 % on every G line.

- **R: FAIL, on tracking only.**
  - **Completion: 7 / 7 rows pass** (rule: new ≥ TDR − max(0.05, 2·SE)).

    | Row | c1 | TDR |
    |---|---|---|
    | field_hard (es5–8) | 0.965 | 0.925 |
    | field_light | 1.000 | 0.985 |
    | ring | 1.0 | 1.0 |
    | field_turn | 0.995 | 1.000 |
    | L090 | 1.00 | 0.99 |
    | L135 | 0.99 | 0.95 |
    | ×0.7 L090 | 0.930 | 0.875 (SE binds; line 0.818) |

  - **Solved kinds: 2 / 2 pass.** hills b3 is 1.000 vs 0.985; gravel b3 is 1.000 vs 1.000.
  - **side22 (es5):** FAST/FAST, against TDR's FAST/FAST.
  - **Tracking: 3 lines fail.** This is the v1.2 decisive set, drift 0, 22 lines. The margin is max(R, 3·SE), and R binds on every line. R is the line's 95 % paired-bootstrap resolution.

    | Line | Seed | c1 | TDR | Worse by | Margin | se_diff | Read |
    |---|---|---|---|---|---|---|---|
    | track_err back | es5 | 0.0809 | 0.0107 | 0.0702 | 0.0617 | 0.0033 | **FAIL** (≈ 2.6 SE over) |
    | track_err back | es6 | 0.0796 | 0.0155 | 0.0641 | 0.0617 | 0.0040 | **FAIL** (≈ 0.6 SE over) |
    | yaw_progress turn | es5 | 0.1659 | 0.1710 | 0.0052 | 0.0066 | 0.0006 | pass |
    | yaw_progress turn | es6 | 0.1645 | 0.1715 | 0.0070 | 0.0066 | 0.0006 | **FAIL** (≈ 0.7 SE over) |

    - The other 18 lines pass:
      - fwd and side track_err improve (worse by −0.021 to −0.044);
      - diag is worse by 0.019 / 0.021, against 0.029;
      - move_turn is worse by 0.002 / 0.000, against 0.037;
      - survival is 1.00 in all four translate categories;
      - facing_closed is worse by 0.002 / 0.006, against 0.050.
    - The 6000-it drift re-read is not applicable (report only).

- **Report lines** (pre-registered; report only, never gated).
  1. **Success beside completion.** No kind gains on success while its completion drops past its margin.
     - slopes and ledge_down: completion moves with success (+0.130 and +0.175). ledge_down has 0 stuck episodes.
     - ledge_up: completion is 0.865 vs 0.915 (−0.050, inside the 0.065 margin). Falls are 27 vs 17, stuck 107 vs 111.
     - pits: completion 0.590 vs 0.520. rubble: 0.925 vs 0.860, a rise past its 0.058 margin.
     - Every non-completer on the five rows, for both teachers, is a 'fall' at a last-sample tilt of 0.48–0.50 rad.
     - ledge_down falls are 96 vs 131:
       - tips at the rim (seg 1, pitch_fwd): 68 vs 125;
       - c1 adds roll falls in the side hold (seg 2): 25 vs 6. es1–2 shows the same, 25 vs 10.
  2. **Split at 0.14 m** (step 9's height method; success counts, c1 / TDR).

     | Kind | Below 0.14 m | At or above 0.14 m | Where the change sits |
     |---|---|---|---|
     | ledge_down | 38 / 39 of 40 | 66 / 30 of 160 | +36 above, mostly 0.16–0.18 m (29 / 6) |
     | ledge_up | 57 / 59 of 79 | 9 / 13 of 121 | above: 10 fewer stuck, 14 more falls |
     | pits (depth estimated) | 22 / 25 of 28 | 92 / 74 of 172 | +18 above; no gain at the gate |

     - Easy drops are intact at +509; step 9's 34990 lost 15 of 38 of them.
     - The drop-tolerance curve moved up by about 2–3 cm.
     - The split does not depend on the cut: 0.13 m and 0.15 m give the same picture, and es1–2 tells the same story.
     - Where c1 is not better (ledge_up above the line, pits below it), it is level with TDR within noise.
  3. **Slopes up / down** (100 episodes per variant).
     - Down: c1 92 vs TDR 65. 30 of TDR's 35 falls are in the seg-2 side hold.
     - Up: 92 vs 93.
     - So +27 of the +26 net successes come from the descent.
     - c1 removes most of the lateral (roll) tips: 4 vs 30. This is the same mechanism as step 9: a lower, more level stance.
  4. **Back ratio, yaw rate, pitch** (flat suite, 0 kg; values are es5 / es6).
     - **Back hold** (command 0.22 m/s):
       - Ratio 1.081 / 1.080 vs 1.011 / 1.016, which is 0.238 vs 0.222–0.223 m/s.
       - c1's ratio is 1.076–1.081 on all four seeds (es1, 2, 5, 6).
       - Strides are longer: 10.8 vs 11.8 steps/m, at 2.50 vs 2.67 Hz.
       - Action saturation is 0.536 vs 0.498, body height 1.115 vs 1.127 m, and cross-track 0.018 vs 0.042.
     - **In-place turn** (saturated 0.3 rad/s command):
       - yaw_progress is 0.1659 / 0.1645 vs 0.1710 / 0.1715, which is −3.5 %.
       - The rate over the first 3 s is −7.5 %, and 90 % of net yaw comes at 8.3 vs 8.1 s.
       - Net yaw and facing_closed are unchanged.
       - The robot takes fewer steps per second with the same yaw per step.
     - **Pitch:** there is no lean into travel at +509. fwd is +0.061 vs +0.063, back −0.022 vs −0.033. The step-9 lean grows late along RT_s19: fwd +0.061, +0.066, +0.082 and +0.110 at 29500, 31000, 33000 and 34990.
     - **Survival** is 1.0 on every suite line.
  5. **Selection → confirm shrinkage** (es1–2 → es5–6).
     - **slopes:** +0.190 → +0.130 (−1.3 sd). c1 moved −0.015 and TDR +0.045, all on the descent (54 → 65 of 100), so about three quarters of the shrinkage is the reference row.
     - **ledge_down:** +0.185 → +0.175 (−0.2 sd), with CI lower +0.116 → +0.109. This is not a selection artefact.
     - **back worse_by:** 0.0616 / 0.0564 → 0.0702 / 0.0641, so z goes +0.04 / +1.80 → −2.55 / −0.63.
       - About 80 % (es5) and 50 % (es6) of the move is TDR's per-seed spread.
       - c1 itself is stable to ±0.003.
     - **yaw worse_by:** 0.0069 / 0.0064 → 0.0052 / 0.0070. The pattern flips from fail-pass to pass-fail, and c1 reads 0.1645–0.1659 on all four seeds.
     - The R lines were never selected on (Q = {c1}), so their drift is not a winner's curse.
     - The pre-registered "c1 sits on both lines" held.
     - Pooled over es1/2/5/6 (report only; ARM forbids pooling into the read): back 0.0631 vs 0.0617, yaw 0.0064 vs 0.0066.
  6. **Seed × length** (es1–2 screen; z = the minimum suite slack in SE units; back ratio and yaw are es1 / es2).

     | Cell | slopes b2 | ledge_down b2 | z (binding line) | back ratio | yaw_progress |
     |---|---|---|---|---|---|
     | c1, s19 +509 | +0.190 GAIN | +0.185 GAIN | −0.40 (yaw) | 1.079 / 1.076 | 0.1646 / 0.1649 |
     | c2, s19 +1009 | +0.190 GAIN | −0.050 | −3.67 (yaw) | 1.067 / 1.062 | 0.1627 / 0.1630 |
     | c3, s20 +509 | +0.065 | +0.145 GAIN | −23.9 (yaw) | 1.014 / 1.011 | 0.1508 / 0.1486 |
     | c4, s20 +1009 | +0.130 GAIN | +0.040 | −33.8 (yaw; move_turn also fails) | 1.045 / 1.042 | 0.1389 / 0.1389 |
     | s19 +2009 (sweep, es1 only) | +0.145 GAIN | +0.055 | +0.13 | 1.048 | 0.1650 |
     | s19 +4009 (sweep, es1 only) | +0.100 | +0.090 | −10.9 (back) | 1.104 | 0.1634 |
     | s19 +5999 (step 9) | +0.150 GAIN | −0.085 REGRESS | −22.5 (yaw) | 1.109 | 0.147 |
     | TDR | — | — | — | 1.018 / 1.019 | 0.1715 / 0.1712 |

     - Only s19 +509 gained on both kinds.
     - Yaw is below TDR in every cell. The loss is systematic and worst on seed 20.
     - Back overshoot varies with seed and iteration.
  7. **Hash line: all pass.**
     - **Thresholds:** thresholds_rough_v1.json is ce45e01b (decision 'proceed', frozen on TDR 03c3e472). Its recorded script and rows_rt hashes equal the current files.
     - **Frozen scripts:** readout.py 14e8717c, rt_spec.py 6716d999, make_thresholds.py 8c67532f.
     - **Rows:** rows_rt.json is 0b038f03. Every gated and solved row's cross_r equals rule.cross_r: slopes, hills and gravel 3.0; ledges, pits and rubble 3.2.
     - **R-tracking inputs** are unchanged since c2054a8c: tools/pipeline, thresholds_v1_2.json 969b2988, threshold_inputs.yaml, pipeline.yaml, resolution.36e5a8af7000.json 77a2b062 and side22_classify.py.
       - The diff is empty against both HEAD and the working tree, and nothing is untracked or ignored.
       - The modules actually executed resolve to these paths; thresholds version v1.2, statistic track_err.
     - **Stamps:** 66 / 66 (33 RTC, 33 RTF) are at code_sha 89c1916b / eval_sha 9fe964d7, which equals pin.json. All are clean with tp_row rc 0.
       - The real seed (5–8) agrees across the seed dir name, run_meta and tp_row.
       - env and args equal the composed row, the role is teacher, no KRABBY_DR_* key is set, and there is one stamp per seed dir.
       - The 66 are exactly the expected slots and the same paths gates_fresh.json served.
     - **Checkpoint identity:** the stamp name prefix equals run_meta checkpoint_sha256 on 66 / 66. model_29500.pt re-hashes to 6d1f29be and model_28991.pt to 03c3e472.
     - **Pairing:** env_index sets are equal and spawns agree within 1e-4 m. Each kind stamp has 100 episodes, and raw length equals n_steps.
     - **Records:** selection.json is tracked and clean (05f67c37, caab51ec), and its pick sha equals the RTC sha. gates_fresh.json (43849afb) is committed in 9da889b6.

- **Mechanism** (CPU, report only; supported, not proven).
  - **The back overshoot at +509 comes from stride length and action magnitude, not from the lean.**
    - RT_s19's median rod-action |μ| peaked at 1.094 around its 29291–29391 (TDR 1.006–1.020). It was 1.080 just before model_29500.
    - RT_s20s stayed near 1.00 through 29500, and c3 shows no overshoot.
    - Across TDR and c1–c4 (n = 5): corr(|μ|, back ratio) = 0.94 and corr(steps/m, back ratio) = −0.84.
    - The reward barely prices overshoot: the net pressure is about −1.4 per m/s.
    - Back crosses its line first because its command sits at 88 % of the 0.25 semi-axis.
  - **Possible coupling to the slopes gain** (n = 5, weak): corr(slopes diff, back ratio) = 0.93. c3, the only cell with neither a lowered stance nor overshoot, also has no slopes gain.
  - **Yaw slows because of the mix**, in both seeds and within about 400 its.
    - Flat and field fell to 0.15 each (TDR: 0.5 / 0.5), so clean flat-turn samples drop about 3×.
    - The orientation term reaches flat tiles only.
    - The turn gait becomes a hazard gait: cadence 1.16 vs 1.25, swing apex 0.128 vs 0.124.
  - **Not causes at +509:** noise std, KL, LR, priv_reg coefficient, payload and rod-speed DR.

- **Reconciled** (the report inputs disagreed; I opened the files):
  - the fwd pitch at 33000 is +0.082 (sweep.json), not +0.080;
  - about three quarters of the slopes shrinkage is the reference row, not all of it;
  - where c1 is not better on the split, it is "level within noise", not "better".

- **Honest reading.**
  - On fresh seeds, c1 is a better rough-terrain teacher than TDR:
    - slope-descent falls drop from 35 to 8 per 100;
    - ledge_down rises from 0.345 to 0.520, on 0.14–0.20 m drops, with the easy drops intact;
    - every completion row is level or better (field_hard +0.040, ×0.7 L090 +0.055, L135 +0.040);
    - survival is 1.0 and side22 is FAST/FAST.
  - ledge_up, pits and rubble did not gain.
  - c1 misses R on two flat-tracking lines:
    - a seed-stable 8 % back overshoot (1.08 vs 1.01–1.02), beyond the line's resolution on both fresh seeds;
    - an in-place turn 3–4 % slower, sitting on the line: it fails es6 by 0.0004 and passes es5, and 2 of 4 seeds fail.
  - The checkpoint replicated; the recipe did not.
    - Seed 20 never gained on both kinds.
    - Along RT_s19, ledge_down swings by about 4 paired SE per 500 its.
    - So c1 is one good checkpoint, not a reproducible recipe.
  - Back overshoot has now appeared in five fine-tunes, and the reward still does not price it.
  - The frozen verdict stands: HALT_R, and the user decides.

- **HALT: options put to the user.** The recommendation is (A) staged.
  - **(A) Accept c1 by a disclosed user overrule of R tracking, then P2** (≈ 19–21 h wall-clock to a bake proposal).
    - Steps: re-distil ≈ 9.6 h; a pre-registered student terrain and tracking read on es9+ against c31000, ≈ 1–2 h; stand re-pair ≈ 7 h; bake videos ≈ 0.5–1 h.
    - Staged: the user approves P2 now and makes the final overrule at the bake proposal, with the student's own numbers in hand.
    - The policy of record changes only on a bake yes.
    - Precedents: the S1 KEEP over its back 1.09, D12, D05, and the SH K4 accept-and-disclose.
  - **(B) Pre-registered remedy arm** (≈ 12–15 h to a verdict; ACCEPT ≈ 15 %).
    - Levers: yaw retention, which is env-only (flat + field ≈ 0.4, a turn dose, ORIENT_ALL_TILES after the replay gate), plus priced back overshoot, which is a code change (the pin moves; replay gate; neutrality row).
    - Read: selection on es1–2, confirm on es9–12 against fresh TDR rows.
    - If accepted, P2 follows.
  - **(C) Env-only remedy arm** (≈ 9–10 h; ACCEPT ≈ 10 %): retention plus a higher KRABBY_ACTION_BOUND_LOSS; from TDR, or warm-started from c1.
  - **(D) Close as NOT ACCEPTED** (≈ 0.5–1 h CPU): TDR stays of record, and c1 is kept as the slope- and drop-reference run.
  - **Not available:** re-reading c1 on es9+, pooling seeds, neighbouring checkpoints on es5–8, or the drift re-read. ARM's "Never" list forbids these, or they do not apply.

### 2026-11-11 17:25 — User decision at the HALT_R: "A" (accept c1 by a disclosed overrule of R tracking, then P2, staged)

- **Decision.** The user picked option A from the 17:10 HALT entry.
  - c1 (RT_s19 `model_29500`, 6d1f29be) goes forward to P2 by a **disclosed user overrule** of the R tracking lines.
  - The frozen verdict stays **HALT_R**. The overrule does not re-read any gate.
  - Disclosed costs:
    - back-hold overshoot 1.08× against TDR's 1.01–1.02× (0.238 vs 0.222 m/s at a 0.22 m/s command);
    - in-place yaw rate −3.5 % (0.165 vs 0.171 rad/s);
    - the recipe is not seed-replicated.
- **Staged.**
  - Steps: re-distil the depth student from c1, a new pre-registered student read on es9+ against c31000, stand re-pair, bake videos, then a bake proposal.
  - The final accept-or-overrule decision is made at the bake proposal, with the student's own numbers in hand.
  - TDR + c31000 + SH_RSD_TDR / SN_RSD_TDR stay the omni policy of record until a bake yes.
- **Next.** The P2 plan is designed, then shown to the user for review and audit before any GPU work.

### 2026-11-12 — User: loose ground noted as a follow-on (P5; not scoped, not in P2)

- **What the gravel is.** Rigid, not loose. `rt_gravel` (`parkour_isaaclab/terrains/omni/site_terrains.py`,
  `_layout_rt_gravel`) is a fixed height field: a smooth random field of ±0.02 → 0.05 m by ladder level, with
  μ_eff ~ U(0.2, 0.5) fixed per env for the episode (the friction contract). Rubble is rigid too: 0.32 m cells of
  U(0, h_max). Nothing sinks, shifts or rolls, and the slip is the same on every step.
  - So c1's gravel result means "slippery, uneven ground", not "loose ground". This is a likely sim-to-real gap.
  - Literature review §8 precedent: GM-Loco's rigid-trained humanoid scored 98 % on basalt and 0 % on sand in MPM
    evaluation; its RFT-trained policy scored 100 % on all three media [83].
- **User:** note loose-ground training as a follow-on task. Recorded as **P5. Loose ground** in PLAN.md's later
  phases. It has no plan or budget yet. Candidate levers, cheapest first:
  - per-foot support-failure / sinkage events: a stance foot's support gives way or sinks by a random depth
    (review §8 and rec. 8). No particles needed, and it also trains the lost-foothold response (P4).
  - footing that changes step to step and with load: per-step stochastic friction or a per-foot wrench perturbation;
  - randomized ground compliance (contact stiffness and damping per tile);
  - a per-foot granular force model (RFT computed in Warp, as in GM-Loco [83]), if deep loose fill matters;
  - free-moving rigid stones or particle (PBD) gravel. Costly: the review found PBD gravel CPU-bound and not
    scaling [84].
  - Measure with a per-foot slip ratio logged per terrain and friction bin (review rec. 8). The depth student's
    behaviour on these tiles is part of the read.

### 2026-11-12 — P2 pre-registered (`p2/P2.md`); audit passes 1–2 applied, pass 3 running; step-2 code built

- **User (2026-11-12):** "accept the defaults and audit", so the 7 P2 defaults are settled (P2.md settled 6). The
  answers to pass 1's four questions all took the recommendation (settled 7):
  - Q1: the selection terrain rows are WATCH only.
  - Q2: the read adds ledge_up, pits and rubble as decisive no-regress lines, and hills and gravel as report-only.
  - Q3: c31000 is read on slopes and ledge_down at es1–2 before the distill (a d' trigger); the c1 and TDR T-kind
    rows at es9–10 always run.
  - Q4: the T rows and the ×0.7 L090 rows launch twice up front.
  - At the 2-pass cap the user approved one more pass (pass 3).
- **Audit.** The trail is in P2.md.
  - Pass 1: 4 whole-document auditors (fable·high among them), a haiku inventory, 6 sonnet probes and an opus number
    check.
  - Pass 2: opus·xhigh and fable·high, which found 4 blocking issues, all fixed.
  - Pass 3 was running when this was committed. Any blocking fix it finds is committed as a dated amendment before
    the first selection row.
  - The plan grew from 1887 to ≈ 2980 words. Most of the growth comes from the user's defaults and Q1–Q4 answers;
    ≈ 30 % comes from blocking fixes.
  - Budget to the bake proposal: ≈ 34 h nominal, ≈ 39 h worst.
- **Built** (workflow `wf_d18bf0ba-4e8`: 2 opus·high builders, a fable·high reviewer for each, then fixers; new files
  only):
  - `envs/P2_distill.env` (c39971d5): its KRABBY_* set equals RT_train.env's (37 keys).
  - `p2/p2_distill.sh`, `p2/p2_distill_check.py`, `p2/p2_curriculum.py`, `p2/p2_step3.py` and `p2/p2_queue.sh`.
  - Self-tests:
    - D1–D7 reference on RT_s19: PASS. The predictor reproduces RT_s19's [rt_friction] line verbatim and predicts
      20 gravel envs of 192 for the student.
    - D5's windowed tail on its 29301–29500: ep_len 1401.9, crab_failure 0.431.
    - The curriculum self-test reads RT_s19 at +1000 at levels 6.45 / 4.76, and RSD's loss ratio at 0.643, so no
      trigger fires.
    - Every block's dry run passes, and the negative checks refuse as they should.
  - The fable reviewers found 4 blocking bugs, all fixed:
    - the curriculum code still used the superseded pass-1 rule;
    - D6 could kill after +250;
    - a step-3 preflight refusal still started the distill;
    - the ×0.7 bridge fingerprint was vacuous.
- **Notes.**
  - `envs/P2_distill.env` matches `rt_machinery.FROZEN_GLOBS` (`envs/*.env`). RT's own `build_check` against
    `results/pin.json` therefore now reads BAD, so any RT block would refuse; none is planned. P2 checks against
    `results/p2/pin.json`.
  - Step 2's hardlink of c1 into MAIN logs is deferred to the stand fork. The distill loads c1 from the worktree path,
    and D3 compares that path.
- **Next.** Write `results/p2/pin.json` on the clean tree. Then move the Kit DerivedDataCache aside: it is renamed,
  never deleted (settled 6d). Then launch the step-3 → distill chain.

### 2026-11-12 14:40 — P2 audit pass 3 done; amendment 1 (before any C1S number); step 3 under way

- **Pass 3** was the extra pass the user approved (workflow `wf_4dd5edcb-13d`: opus·xhigh, fable·high and an
  opus·medium number check). It read the plan while step 3 ran.
  - fable found nothing blocking.
  - opus found one blocking issue, in pass-2 text: the "a diag-only Tr fail is c1's predicted cost" label had no
    bound.
- **Amendment 1**, edited in `p2/P2.md`:
  - The diag label is bounded: worse_by ≤ 0.035, which is c1's ≈ 0.025 plus the largest recorded student offset,
    0.009. Beyond that the fail is presented as a student loss, beside selection's tracking_watch value.
  - The stand budget had double-counted REFS (0.6 h) and the lazy R_h (≈ 0.37 h): RSD's 6.8 h already includes both.
    Pass 1's premise was wrong. The stand line is back to ≈ 7.0 h, and the totals are ≈ 33 h nominal and ≈ 38 h
    worst.
  - The D7 line is 10.5 s/it in-loop, matching the built check.
  - Eight implementation checks were added (items 19 and 21–25).
  - The amendment's own fix was not re-audited. It touches only how a Tr fail is labelled at the proposal; no gate
    line moved.
- **Step 3.** The chain started 14:11:20.
  - smoke 14:11–14:22: rc 0, **P2_SMOKE PASS**. Both kinds' twin launches show the class signature (CLASS-FLIP),
    so the §4 pair check has been exercised before the read.
  - The d' rows started 14:22.

### 2026-11-12 — User: "Accept as is" (no fourth audit pass)

- The P2 audit ends after 3 passes. Amendment 1 stands unaudited, as disclosed in the 14:40 entry. `p2/P2.md` at
  2059235a is the plan of record for P2.

### 2026-11-12 14:41 — P2 step 3: d' rows read (Q3); trigger NONE

- **Rows.** `P2D_155b5981_rt_{slopes,ledge_down}_b2`: c31000 in the student role at es1–2, 14:22–14:40, rc 0. They were
  read on CPU (`p2_step3.py --read`) while the bridge ran. The read is paired on (es, env_index) against c1's RTSW rows
  and TDR's RT9 rows, using success (complete ∧ crossed), n = 200.

  | Kind | c1 | TDR | c31000 | c1 − c31000 (SE) | TDR − c31000 (SE) |
  |---|---|---|---|---|---|
  | slopes b2 | 0.935 | 0.745 | 0.785 | +0.150 (0.031) | −0.040 (0.030) |
  | ledge_down b2 | 0.590 | 0.405 | 0.360 | +0.230 (0.037) | +0.045 (0.030) |

- **Trigger: NONE.** c31000 is below c1 − 0.10 on both kinds, so neither kind is out of reach.
- **d' is small on both kinds.** c31000 sits within about 1.5 SE of TDR, so the Q3 confound is small and a T gain
  would mostly be c1's.
- **Projection, report only, not a gate.** c1's es1–2 values are inflated (it was picked on them): on fresh es5–6, c1
  read 0.920 / 0.520 and TDR read 0.790 / 0.345. So the fresh c1 − c31000 is roughly:
  - slopes: +0.09 to +0.135, against a hurdle of ≈ 0.105–0.118. That is marginal even for a lossless student.
  - ledge_down: ≈ +0.2, which leaves room for a distill loss of ≈ 0.1.
  - The likely T outcome is therefore PARTIAL (ledge_down), which routes to "you decide" at the proposal.

### 2026-11-12 15:08 — P2 step 3 DONE (smoke PASS, d' NONE, bridge PASS); step 4 distill RT_ST_s19 launched

- **Chain** `p2_queue.sh`, 14:11:20 → 15:07:48. Every block returned rc 0, and step 3 took 0.94 h.
  - **smoke** (es90, twin launches): PASS. Both kinds show the class signature.
  - **d'**: trigger NONE (see the 14:41 entry).
  - **bridge**, c31000 at es1 against the recorded RSD stamps, cross-pin, with path keys normalised:
    - L090 field_light and suite: PASS, a bitwise match on the first launch.
    - ×0.7 L090 field_light: RELAUNCH, then PASS on R1 (`P2B_155b5981_x07_L090_R1`). This is the case the pass-1
      fix was written for.
    - ×0.7 L090 suite: PASS.
  - The result is in `results/p2/step3.json`.
- **Step 4.** The distill RT_ST_s19 started at 15:07:48, through `p2_distill.sh` (train.py from this worktree).
  - The +10-it checks run from a background waiter. A D1–D5 miss → `p2_distill_check.py stop`.
  - The curriculum reads follow at +300 and +1000.

### 2026-11-12 15:10 — P2 step 4: the +10-it checks PASS (RT_ST_s19)

- Command: `p2_distill_check.py check --name RT_ST_s19 --teacher <c1>`, rc 0, log at +11.
  - **D1 PASS.** The 10 knob lines equal RT_s19's minus the stripped-key lines, and both site-geometry sync lines are
    present. The `[rt_friction]` line equals the prediction: 20 gravel envs of 192.
  - **D2 PASS.** The [DR] draw is as predicted.
  - **D3 PASS.** The teacher sha is c1's, 6d1f29be.
  - **D4 PASS.** The first iteration is 29500, and the log reached 29511.
  - **D5 PASS**, recorded only. The windowed c1 tail over its 29301–29500 reads ep_len 1401.9 and crab_failure 0.431.
  - **D6 N/A.** The live rules are not applied (P2.md step 4).
  - **D7 COST.** The pace is 9.77 s/it, so 5000 its take ≈ 13.6 h. That sits inside the pre-registered 10.2–14.7 h,
    under the 10.5 s/it re-notify line. The distill should end ≈ 2026-11-13 04:45.
- **Next.** The curriculum reads run at +300 and +1000. The step-5 CPU forks (selection, read, stand, video, ONNX) are
  built during the distill.

### 2026-11-12 — P2 step 5 code built (CPU, during the distill); +300 curriculum read healthy; selection queued

- **+300 curriculum read** (21-iteration window means, tb and log agree to 5e-5): slopes level 6.67, ledge_down
  4.89, terrain_levels 6.06, depth_actor_loss 0.728. That is already at the RT teacher's +1000 levels (6.45 / 4.76).
  No trigger is evaluated before +1000.
- **Built** (workflow `wf_24c1eb54-12b`: 3 opus·high builders, a fable·high reviewer and fixer per piece, then a
  fable·high integration review; new files only, no GPU):
  - **Selection + ONNX:** `p2/p2_select.py`, `p2_sel_queue.sh` and `p2_onnx.py`.
    - Fork check: with RSD's constants restored, it reproduces RSD's `sel.json` exactly (781 numbers, max |diff| 0,
      same pick c31000 155b5981).
    - The WATCH read on c31000's P2D rows reproduces step3.json's d'.
    - c1's suite es1/es2 stamps are proven reusable, so 8 teacher rows run.
    - An ONNX re-export of c31000 is byte-equal to the bundled graphs, with parity 1.2e-6.
    - The candidate grid projects to 30500 / 31500 / 32400 / 33400 / 34499. Saves are every 200 its after +2500, so
      the 32500 and 33500 targets tie and select_student's unchanged rule takes the earlier save. Disclosed.
    - Review: 0 blocking findings.
  - **es9+ read:** `p2/p2_read.py`, `p2_read_check.py` and `p2_read_queue.sh`.
    - The pooled statistic follows P2.md: per-key means, paired_completion_se on fractional dicts, and g_line's
      clauses, never routed through readout's bool casts.
    - The synthetic two-launch check and the gates_fresh identity reproduction both pass.
    - Review: 1 blocking finding, fixed: the diag label's tracking_watch lookup pointed at the wrong sel.json key.
  - **Stand + videos:** `p2/p2_stand.py`, `p2_stand_queue.sh` and `p2_bake_videos.sh`.
    - It is the STAND_REPAIR recipe with c1 as walker, K4 run as RSD's two-sided test, the body-height pre-flight
      line, and no MAIN hardlink needed for c1.
    - Review: 1 blocking finding, fixed: `--launch` could never pass its source check.
- **Integration review.** It found one blocking seam: the stand queue could take the GPU in a gap between the
  selection or read chains' blocks. It also found 5 minor ones. They are handled by orchestration, not code; these
  rules are binding for P2:
  1. Queue order is selection, then read, then stand, one chain at a time. The read queue is launched only after:
     - `P2SQ PICK`;
     - the independent recompute of the pick;
     - `sel.json` is committed;
     - `onnx_PICK.json` reads ok (parity, io_equal_c31000, the pick's sha).
  2. The stand queue is launched only after the read chain is DONE and `read.json`'s verdict has been read (not
     HALT, INCOMPLETE or PENDING), and after the step-9 DDC move.
  3. `p2_select.py --decide` is never re-run non-dry after the read starts, because it rewrites sel.json and would
     refuse the read.
  4. The live build is checked against `results/p2/pin.json` before the ONNX export and before the videos.
  5. A module-name shadow (`p2_read`, the stand's phase-2 reader versus the es9+ driver) resolves correctly today.
     The stand pre-flight is read for it.
- **Queued.** `p2_sel_queue.sh` was launched detached. It waits for `TPDISTILL END RT_ST_s19 rc=0` and the chain lock,
  then runs teacher → prescreen → rank → full → decide → onnx.

### 2026-11-12 ≈ 17:50 — P2 curriculum read at +1000: trigger NONE

- **Read** at iteration 30500, 21-iteration window means (tb and log agree to 5e-5):

  | Signal | RT_ST_s19 at +1000 | Comparison |
  |---|---|---|
  | slopes level | 6.93 | RT teacher at +1000: 6.45 |
  | ledge_down level | 4.70 | RT teacher at +1000: 4.76 |
  | terrain_levels | 5.66 | — |
  | depth_actor_loss | 0.665 | +50 value 1.010, ratio 0.659; field precedents 0.64–0.68 |

- **Trigger: NONE.** Both levels are ≥ 3.0, and the loss is ≤ 0.80× its +50 value.
- **What this shows.** The student keeps pace with the teacher's own curriculum on the hazard kinds, so DAgger is
  labelling b2-band states, which the design rests on. The loss falls like a healthy field distill.
- **Next.** No further mid-distill read is planned. The selection chain is queued for the distill's end (≈ 04:45).

### 2026-11-13 09:23 — P2 steps 4–6 done: distill END rc 0; selection PICK 6b48ef33 (model_34499); ONNX ok

- **Distill RT_ST_s19:** 2026-11-12 15:07 → 2026-11-13 ≈ 05:43, END rc 0. That is ≈ 14.6 h wall, at the top of the
  10.2–14.7 h estimate. The D7 pace of 9.77 s/it, measured at +10, under-read the overall rate.
- **Selection** (`p2_sel_queue.sh`, 05:43 → 09:22, fail 0):
  - c1 teacher rows: 8 run, plus the 2 reused suite stamps.
  - Prescreen: 25 rows. Every candidate has 0 floor misses.
  - Rank (floor misses, then field_hard, then mean |ratio−1|):

    | Rank | Save | sha | field_hard | mean \|ratio−1\| |
    |---|---|---|---|---|
    | 1 | it34499 | 6b48ef33 | 0.980 | 0.0265 |
    | 2 | it31500 | 75a85390 | 0.975 | 0.0256 |
    | 3 | it32400 | 5fdae7dc | 0.975 | 0.0260 |
    | 4 | it30500 | 343ab864 | 0.970 | 0.0336 |
    | 5 | it33400 | b7736eb5 | 0.950 | 0.0266 |

  - Full block 1 (19 rows) on 6b48ef33: **PASS → PICK** `6b48ef33_it34499`, `model_34499.pt`, sha 6b48ef33d26b….
- **WATCH terrain rows**, es1–2, student vs c1, report only:

  | Kind | Student | c1 | Diff | Line |
  |---|---|---|---|---|
  | slopes b2 | 0.920 | 0.935 | −0.015 | 0.885 |
  | ledge_down b2 | 0.555 | 0.590 | −0.035 | 0.518 |

  The student keeps most of c1's terrain skill on the seeds c1 was picked on.
- **ONNX:** `results/p2/onnx_PICK.json` reads ok. Parity act_err is 1.2e-6 and lat_err 2.2e-7, the I/O json equals
  c31000's, policy.onnx is 86d1c801 and depth_latest.onnx is cca6952d. The export sits in a P2 output dir, not
  policy/.
- `sel.json` and `onnx_PICK.json` are committed here, before any es9+ stamp. The independent recompute of the pick
  (step-6 check: sonnet·high plus fable·high, from raw stamps) is running. The read launches only after it agrees.
- DDC: 14 GB. Disk: 486 GB free.
