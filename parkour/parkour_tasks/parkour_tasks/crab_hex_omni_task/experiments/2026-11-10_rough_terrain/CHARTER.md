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
