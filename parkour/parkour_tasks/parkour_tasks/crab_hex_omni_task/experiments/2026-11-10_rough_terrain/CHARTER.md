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
