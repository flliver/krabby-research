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
