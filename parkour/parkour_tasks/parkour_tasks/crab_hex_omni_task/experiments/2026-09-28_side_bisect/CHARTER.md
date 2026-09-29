# Side-speed bisection: which omni-config factor forms the lean-and-shuffle attractor

- **Plan of record:** `/home/nickmagus/.claude/plans/side-speed-bisection.md` (AUDITED, cap-2 audit
  trail, 2026-09-28; STATUS block records the user's autonomy rule -- "run without my input,"
  decide-record-notify at every STOP point except push/publish, deleting run dirs or data, and
  baking a head).
- **The machine-readable spec:** [`candidates.yaml`](candidates.yaml) -- references `O`, `C`, `R`,
  `RC` (chains on base `1b`, from scratch, 3000 its, eval `[side22_25]`, `categories: [side,
  stand]`, `floor: {kind: none}`), no candidate queue.
- **The eval row:** `side22_25` in
  [`../eval/scenarios_omni.yaml`](../eval/scenarios_omni.yaml) (category_suite, lat_l at 0.22 and
  0.25 m/s, 2-s settle, 12-s holds) and its speed-matched sideways twin in
  [`../../crab_hex_sideways_task/experiments/eval/scenarios_side.yaml`](../../crab_hex_sideways_task/experiments/eval/scenarios_side.yaml)
  (plan step 3). The omni row also carries a `stand` wave that the gate never reads: the driver
  passes every stage's trained categories (1b: `stand,side`) to the eval, and the harness refuses a
  name the row lacks. GPU-0's wider sweep (0.15/0.22/0.30/0.35) is a separate file,
  [`../eval/scenarios_gpu0_sidesweep.yaml`](../eval/scenarios_gpu0_sidesweep.yaml) (plan item 8).
- **eval_sha moves for every omni campaign.** Adding `side22_25__omni` to `scenarios_omni.yaml` and
  `KRABBY_OMNI_CMD_OBS` to `EVAL_CARRY` changes the omni `eval_sha`. Other omni campaigns' next
  relaunch needs `--accept-eval-change` (and `--accept-code-change`, since `code_sha` moves too).
- **Gate statistics (noise measured, zero-GPU scoping):**
  `/home/nickmagus/.claude/plans/side-speed-bisection.evidence/scope_bisect/gate_stats.py`.

## Context

**Diagnosis.**
[`../2026-09-28_side_speed_investigation/REPORT.md`](../2026-09-28_side_speed_investigation/REPORT.md)
found that omni side walking is capped at about 0.10 m/s by a "lean-and-shuffle" gait attractor --
not the plant (the sideways head reaches 0.249 m/s) and not reward pricing.

**What CPU-1 narrowed it to.** Fwd-only 1a does not lean; side-only 1b leans (unidirectional
sampler, no co-training, forked from 1a); from-scratch co-trained 1c_004 leans. An omni-config
constant is the parsimonious suspect.

**Scoping (a 3-scout zero-GPU workflow, `wf_3cf88033-e34`).**
- R's reward economy is mostly already tested: 1b already trained under the sideways knobs (sum
  clock, track std 0.316, L1 -1, facing weight 0) and still leaned. R's prior is "slow".
- 1b was never truly one-direction: the 4-image mirror loss trains the -y branch through
  M_y / R_z(pi) even under a +y-only sampler. A fore-aft-only mirror plus a direction-blind
  observation (arm C) would give the first genuinely one-direction omni run.
- The sideways task's formation ran with RSI off and a wider lateral band (U[0.21, 0.35] vs omni's
  U[0.21, 0.25]).
- Not portable (the residual, shared by every arm): the ring vs corridor world (terrain, promotion,
  spawn, terminations) and the wz -> clock coupling.
- Primary gate statistic: feet_ss (the feet-centroid speed over hold-relative [4, 10) s of a side
  hold). Slow omni max 0.112 against fast sideways min 0.200; seed |Delta| <= 0.017 (omni),
  0.03-0.04 (sideways). Lean is noisy (seed spread 0.07-0.09) and does not discriminate at
  0.30-0.35 m/s. The early read at it 1000 (track_ratio_mean) tracks the it-3000 class in every run
  on record.

**User decisions.** (1) Investigate why side walking is slow, then plan and run GPU-0 followed by
the side-only bisection. (2) Keep R as its own arm (plan review, 2026-09-28), audit up to 2 passes.
(3) Carried over: one Isaac process at a time under a memory cap; references for arms (no
dead_walker kill); local commits only, never push; push-notify at stops.

**Audit trail (cap 2).** Pass 1: 2 premise probes (opus·xhigh auditor + opus·medium spot-check).
P1 holds (R reproduces the sideways reward economy); P2 partly, with 4 blocking fixes folded into
the "Left for implementation" rules below. Pass 2: no blocking findings, converged (opus·xhigh).

## Arms

All arms: omni task, base preset **1b** (side_l only: DIR_BAND pi/2:pi/2, SLOTS 0.2:0.8:0:0), from
scratch, 3000 its, seed 3, eval row `side22_25`.

| Arm | What changes from the 1b preset | Tests |
|---|---|---|
| **O** (control) | nothing (Revision-4 economy, RSI source 1, 4-image mirror) | Does side-only omni *from scratch* form the attractor? (1b was a 1000-it fork of fwd 1a.) |
| **C** | `KRABBY_OMNI_MIRROR_IMAGES=fa` (fore-aft-only mirror), `KRABBY_OMNI_CMD_OBS=scalar` (direction slots 8-9 zeroed, slot 10 keeps \|cmd\|, slots 5-7 kept), `KRABBY_RSI_FRAC=0` (RSI_SOURCE dropped -- inert at frac 0), `KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25` | The sideways task's command/obs/mirror/RSI/band bundle: the first genuinely one-direction omni run |
| **R** | `KRABBY_OMNI_CLOCK_COMBINE=sum`, `KRABBY_TRACK_SIGMA2=0.1`, `KRABBY_TRACK_L1_W=-1.0`, `KRABBY_YAW_PROGRESS_W=0` | The sideways reward economy, from scratch. Prior: slow (kept by user decision 2). |
| **RC** (conditional) | C's overlay plus R's | Runs only if both C and R are SLOW. Everything portable from the sideways task. Slow here -> the residual (world / code path). |

## Gate (pre-registered; read on lat_l only -- the trained hand)

**Per arm,** on `side22_25` (lat_l holds at 0.22 and 0.25 m/s, 2-s settle, then 12 s, 100 episodes):

| Class | Rule (feet_ss alone; both speeds must agree) |
|---|---|
| **FAST** | feet_ss >= 0.155 at 0.22 **and** >= 0.175 at 0.25 |
| **SLOW** | feet_ss <= 0.13 at 0.22 **and** <= 0.145 at 0.25 |
| **DEAD** | feet_ss < 0.05 |
| **INTERMEDIATE** | anything else -> add seed 2 for that arm |

- Lean and sweep are mechanism labels only (e.g. FAST-leaned, SLOW-nonleaning); they never trigger
  a run.
- Two seeds: the arm's class is the one both seeds share. If they differ, see the decision defaults.
- Per arm, report n (episodes that enter feet_ss) next to the class.
- O DEAD or INTERMEDIATE, or the 8.5 GPU-h cap: see the decision defaults.

**Early read.** At it ~1000, track_ratio_mean over its 800-1000: >= 0.58 means forming FAST,
<= 0.48 means forming SLOW. Uncalibrated triggers, used only for the O stop gate and progress
notes; no arm is killed on telemetry.

## Decision defaults (user's autonomy rule, 2026-09-28)

The user: "run without my input. Only pause for me if its an irreversible decision, don't pause for
me if it just costs GPU time." Every STOP-and-ask in the plan is a **decide-record-notify** point:
the decision and its reason go into the Log below (and GATE.md), with an FYI push-notify. Still
paused for: push/publish, deleting run dirs or data, baking a head into the default config.

| STOP point | Default action |
|---|---|
| W harness check off 0.095 +/- 0.01 | Diagnose the row or statistic. If the offset is explained and stable, shift every threshold by W's offset and record it before O launches. |
| Sideways FAST pin fails | Re-threshold FAST to (min sideways model_3000 stamp - 0.017) per speed, provided it stays >= SLOW + 0.01. Otherwise use the omni/sideways midpoint. Record before O launches. |
| O early read >= 0.58 | Let O finish (as planned). Hold C until O is classified. |
| O FAST | Seed-2 O. If confirmed FAST, the premise is broken: write it up, skip C/R/RC, and run the follow-up pair that splits the warm start from bidirectionality (O forked from fwd 1a, and side l+r from scratch, if the knobs allow). |
| An arm INTERMEDIATE | Seed 2 (gate rule). |
| Any DEAD, n = 0 or NaN | Diagnose the harness first. If the policy is genuinely dead, run seed 2. If both seeds are dead, record the arm as un-trainable at 3000 its and continue. |
| Seeds disagree | Seed 1 as the tie-break; majority class. With no majority, the arm is INCONCLUSIVE; continue. |
| 8.5 GPU-h cap | No stop. Notify FYI and report the overrun. |
| Report (step 10) | After GATE.md, continue one level into the pre-named split of whichever bundle is FAST (C: mirror+obs vs RSI vs band; R: yaw_progress vs fork vs the three knobs), with 1 seed each under the same gate. |

## Launching

Each arm is a reference run, launched from the main checkout once this branch is landed:

```
parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh \
  --campaign-dir parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect \
  --reference-chain <O|C|R|RC> --seed <3|2|1>
```

Never use the plain launcher (no `--reference-chain`): it would run the derived queue (O as a
baseline at s3 then s2), and C and R would never start.

Expected record-only values that are not failures:
- `phase_checks` records "omni RSI bank path in params/env.yaml" as False for C and RC (RSI off).
- **The "scalar requires fa" refusal lives in the campaign validator only**, not on the runner side
  as the plan said. The Play/eval task builds the runner cfg without the mirror knob, so a
  runner-side refusal would break every eval of C. A manual `train.py` launch with
  `KRABBY_OMNI_CMD_OBS=scalar` and the default mirror is therefore not guarded.

## Left for implementation (rules, carried from the plan audit)

1. **DEAD wins over SLOW.** A DEAD C or R, n = 0, or a NaN feet_ss is never an RC trigger; it takes
   the DEAD default above.
2. **feet_ss** = the per-episode median over hold-relative [4, 10) s, with u = the body's
   displacement over the hold (`gate_stats.py`).
3. **Sideways FAST pin (step 5).** Each sideways model_3000 stamp must read at least
   threshold + 0.017 at both speeds on the sideways `side22_25` row; otherwise re-threshold before
   O (decision default).
4. **Residual list.** Every arm keeps the facing and yaw-rate obs, the world-held relay (dead band
   0.133 vs 0.267) and a centred height scanner (sideways looks 0.6 m ahead). An all-SLOW result
   does not mean C tested the sideways observation.
5. **R's prior is weaker than stated.** 1b also had ang_vel std 0.15 and yaw_progress 1.0. An
   R-FAST split must include ang_vel std and 1000 vs 3000 its.
6. **The W harness check and GPU-0 run with W's own carry**, `KRABBY_OMNI_CLOCK_HIDE=1`.
7. **O's early read holds back C**; it does not interrupt O mid-training (resume-state hazard).
8. **GPU-0's 0.30 and 0.35 holds** need a separate manifest with a row `speed_ellipse` (built:
   `../eval/scenarios_gpu0_sidesweep.yaml`). Verify `cmd_applied`.

## Interpretation

| Outcome | Reading |
|---|---|
| O FAST | The premise is broken: side-only from-scratch omni does not form the attractor (decision default: seed-2 O, then the follow-up pair). |
| O SLOW, C FAST, R SLOW | The factor is in C's bundle. Next split (decision default): mirror+obs vs RSI vs band. |
| O SLOW, R FAST, C SLOW | The factor is in R's economy, from scratch. 1b (a fwd fork) leaned under most of it, so the split is: yaw_progress vs fork vs the three knobs. |
| O SLOW, C FAST, R FAST | Either bundle suffices on its own. Add a seed-2 O before reading it. |
| O, C and R SLOW, RC FAST | R x C interaction |
| O, C, R and RC all SLOW | The residual: the ring world, the wz -> clock coupling or the omni code path |

## Steps

**Phase A -- build (0 GPU; worktree `.claude/worktrees/stroke-posture`, branch
`feat/omni-side-bisect` rebased to `3860ddb7`)**

1. Knobs (`KRABBY_OMNI_MIRROR_IMAGES`, `KRABBY_OMNI_CMD_OBS`, driver dump checks/parsers) --
   sonnet·high. Done.
2. Tests (knob-off identity, mirror/obs unit coverage, refusals, EVAL_CARRY) -- sonnet·high, with an
   opus·high fresh-context review. Done.
3. Eval row `side22_25` (this campaign's manifest, the sideways twin, GPU-0's manifest) -- sonnet
   ·medium. Done (this record).
4. Campaign `experiments/2026-09-28_side_bisect/` (this CHARTER + `candidates.yaml`), `--dry-run`
   clean -- sonnet·medium. Done (this record).

**Phase B -- GPU (serial)**

5. Knob smoke and harness check (`--smoke-stage 1b:20:side22_25` with C's, R's and RC's overlays;
   `run_meta` assertions on C/RC's smoke and on C's real eval; W's harness cross-check at feet_ss
   0.095 +/- 0.01; the sideways FAST pin). A failing check takes its decision default --
   sonnet·high.
6. GPU-0 (W on side holds at 0.15/0.22/0.30/0.35, widened ellipse, verify `cmd_applied`; report
   only, gates nothing) -- sonnet·medium.
7. Arm O (early read at it 1000 holds C if forming FAST; O always runs to 3000 and is classified)
   -- main.
8. Arm C (launch, eval, classify; push-notify after O/C/R with the interpretation-table row) --
   main.
9. Arm R, then arm RC only if both C and R are SLOW (launch, eval, classify each) -- main.
10. Report: GATE.md (classes, mechanism checks, interpretation row), memory/plan status updated,
    records committed locally, push-notify -- main (opus synthesis), fable·high independent check.

**Budget**

| Item | GPU-h |
|---|---|
| Smoke and harness evals | ~0.7 |
| GPU-0 | ~0.3 |
| O, C and R | ~4.5 |
| RC, conditional | ~1.5 |
| Seed-2 top-ups, conditional | ~1.5 each |
| **Typical** | **~5.5** |
| **Cap** | **8.5** (FYI notify at the cap; no stop) |

## Out of scope -- not run

Porting the corridor world, spawn and promotion; the wz -> clock coupling fix; the heading dead-band
constants; the body-locked travel axis; any fix or bake. (Splitting the FAST bundle one level is in
scope under the decision defaults.)

## Log

- 2026-09-28: `side22_25` added to `../eval/scenarios_omni.yaml` (category_suite, lat_l 0.22/0.25,
  2-s settle, 12-s holds) and to the sideways manifest
  (`../../crab_hex_sideways_task/experiments/eval/scenarios_side.yaml`, a plain schedule row -- the
  sideways task is body-frame `vx`, so `category_suite`'s world-row-only holds do not apply there).
  GPU-0's wider sweep landed as its own file, `../eval/scenarios_gpu0_sidesweep.yaml` (adapted from
  scout 2's dry-run-validated draft). [Corrected 2026-09-29: the omni `eval_sha` does move, to
  `36e5a8af7000` before the stand-wave fix, because of the new row and the new EVAL_CARRY key.] All three
  manifests validated CPU-only (`schedule.load_manifest` + `validate_scenario`, no Isaac): clean.
  `candidates.yaml` written (references `O`, `C`, `R`, `RC` on base `1b`, `categories: [side,
  stand]`, `floor: {kind: none}`); `run_candidates.py --campaign-dir <this dir> --dry-run` clean --
  validator section reads `ok <campaign>` / `ok chain:O` / `ok chain:C` / `ok chain:R` / `ok
  chain:RC` / `ok <overlays>`, no `REFUSED` anywhere. (The two eval-scenario ids had to carry the
  `__omni` / `__side` suffix the driver's `_sid_for`/`run_phases` helpers append to a short row
  name -- first dry-run pass caught this as a "not in the omni manifest" refusal before the rename;
  a second pass came back clean.) No `resolution_inputs` are declared here -- unlike
  `2026-09-27_stroke_cat_t3`, this campaign reuses no old baseline pair and its `decision_rule` is
  carried only for validation parity (the real gate is `feet_ss`, read by hand), so there is
  nothing for D00s to resolve. Never launched from this worktree. GPU work (Phase B) has not
  started.
  Also: adding `side22_25__omni`/`side22_25__side` moved 3 pinned exact-set/exact-count assertions
  in `tests/unit/test_crab_hex_gait_schedule.py` (the committed-omni-manifest id set, the
  `uses_waves` suite-row list, and the compile-snapshot fixture's row count/membership); fixed by
  adding the two new rows' entries and updating the counts, never by touching any existing row's
  recorded bytes (`tests/unit/fixtures/gait_eval_compile_snapshot.json` diff is additive only, one
  new key). Verified with the campaign's own manifest CPU validator (`schedule.load_manifest` +
  `validate_scenario` on all three touched/new eval files, no Isaac) and the task's required unit
  command: 9 failed (the named pre-existing set) / 1738 passed / 5 skipped -- identical to step 2's
  reported baseline, no regressions.
- 2026-09-29: fresh-context review (opus·high) of the Phase A diff. **Blocking, fixed:** the gate row
  had no `stand` category, so every 1b stage's eval (trained `stand,side`) would have been refused by
  `suite_cli_args` and halted `eval_infra` after the 20-min retry wait. Fixed by adding a `stand`
  wave (`waves: 3`). The dry-run validator now refuses any suite eval row that cannot take the
  stage's `--trained-categories` (`_check_suite_rows_cover_trained`), with regression tests.
  **Minor, fixed:** GPU-0's recipe now runs with W's full recorded eval env; the eval_sha log line
  corrected; per-arm launch commands and the expected-False RSI-bank check recorded (Launching);
  the validator-only placement of the "scalar requires fa" refusal recorded.
- 2026-09-29 00:15–00:40: Phase B checks all PASS ([CHECKS.md](CHECKS.md)): knob smokes C/R/RC (overlay_checks 4/4, 4/4,
  8/8; C's eval path through the driver with every run_meta assertion); W harness 0.096 at 0.22 (target 0.095 ± 0.01);
  sideways FAST pin at model_3000 0.196/0.193 (s3) and 0.201/0.202 (s2), so no re-threshold; GPU-0: W flat at
  0.09–0.11 m/s from a 0.15 to a 0.35 command, lean grows 0.19 → 0.28.
- 2026-09-29 00:40: arm O launched (unit `cands-2026-09-28_side_bisect-1790656807`, run dir
  `crab_hex_omni_flat_walk/2026-09-29_00-40-18_seed3_X01_s1_A15pB`). Hazard: every reference's run tag is
  `seed3_X01_s1_A15pB`-shaped and the knob smokes left `seed3_X01_1b_A15pB` run dirs; never stop an arm mid-training.
- 2026-09-29 01:52: **O s3 = SLOW.** feet_ss 0.084 at 0.22 and 0.094 at 0.25 (n 98/96); lean_lvl 0.27/0.05; sweep 0.04.
  Early read 0.564 (C not held); final track_ratio 0.568. Side-only omni FROM SCRATCH forms the slow attractor, so the
  premise holds. The chain auto-launched C (unit `...-1790661165`).
- 2026-09-29 03:03: **C s3 = FAST.** feet_ss 0.215 at 0.22 and 0.214 at 0.25 (n 90/93); lean 0.05–0.075; sweep 0.085 (the
  sideways gait; the sideways model_3000 references read 0.196–0.202). Early read 0.671, final track_ratio 0.798. Eval
  run_meta assertions hold: CMD_OBS=scalar carried, no mirror knob, override warning 0.25, schedule deviation 0.
- **Process miss, 03:03–08:28: the GPU sat idle about 5.4 h.** Nothing chained C → R, and the watcher on the O chain
  never fired (`pgrep -f` matched its own command line). R was launched at 08:28 (unit `...-1790684899`). Since then a
  detached queue runner (`phaseB/queue_runner.sh`, log `parkour/outputs/omni_plan/side_bisect/queue.log`) owns the
  sequencing:
  - it classifies R, applies `phaseB/candidates_v2.yaml` (C_mo / C_band / C_rsi, plus seed-2 references) after R's
    driver exits, and runs the decision-default queue;
  - R SLOW → C_mo, C_band, C_rsi, then C_s2 (replicates the headline);
  - R FAST → O_s2, then a stop.
- 2026-09-29 09:39: **R s3 = SLOW.** feet_ss 0.075 at 0.22 and 0.078 at 0.25 (n 97/95); lean 0.14–0.18; sweep 0.023.
  **Interpretation row: O SLOW, C FAST, R SLOW → the factor is in C's bundle.** RC is not run (it needs C and R both
  SLOW). The runner applied `candidates_v2.yaml`: chains C_mo, C_band and C_rsi, plus references O_s2, C_s2 and R_s2
  (`--accept-candidates-change` on C_mo's launch). It launched C_mo (unit `...-1790689151`, run
  `seed3_X04_s1_A15pB`). The queue is C_mo → C_band → C_rsi → C_s2.
- 2026-09-29 10:52: **C_mo s3 = INTERMEDIATE** (fore-aft mirror + scalar command obs alone). feet_ss 0.162 at 0.22 and
  0.173 at 0.25 (n 100/100); FAST needs ≥ 0.175 at 0.25, so it misses by 0.002. The lean is gone (lean_lvl −0.04,
  lean04 ≈ 0) and the sweep is 0.056, between O's 0.04 and C's 0.085. So the one-direction mirror+obs pair alone
  removes the attractor and recovers most of the speed.
  - Gate rule (INTERMEDIATE → seed 2): reference `C_mo_s2` (`phaseB/candidates_v3.yaml`) goes to a follow-up runner
    (`phaseB/queue_more.sh`). It starts after the main queue (C_band running as X05, then C_rsi, then C_s2).
  - Mechanism note from the code read: 'fa' also drops the M_y (left-right) equivariance, which carries the
    tripod-set balance and left-right symmetric standing. It is untested whether the lean depends on that pair.
- 2026-09-29 12:07: **C_band s3 = SLOW** (wider band alone). feet_ss 0.093 at 0.22 and 0.100 at 0.25 (n 96/95); lean
  0.16–0.22; sweep 0.05. The band's cadence and tracking-economy changes alone do not break the attractor.
- 12:07: the C_rsi launch was refused ("another campaign driver is already running"). Nothing of ours was running;
  most likely another session briefly ran a driver command. The runner would have treated it as a missing run dir
  and stopped, and the follow-up would then have skipped C_rsi and C_s2. Both runners were killed and replaced by
  `phaseB/queue_now.sh`, which retries a refused launch every 60 s, up to 40 times. It applied `candidates_v3.yaml`
  and launched C_rsi at 12:09 (X06). Queue: C_rsi → C_s2 → C_mo_s2. GPU idle for about 2 min.
- 2026-09-29 13:22: **C_rsi s3 = SLOW** (RSI off alone). feet_ss 0.069 at 0.22 and 0.077 at 0.25 (n 96/99); lean_lvl
  0.05; sweep 0.035.
  - Bimodal: 34–35 % of episodes barely move (feet_ss < 0.03; p25 0.015–0.017). The rest walk at a median of
    0.079/0.086, i.e. the O-like slow speed.
  - So removing the fast-gait seeding alone gives no speed, and it adds a stall mode.
  - The low median lean is diluted by the stalled episodes; it does not mean a non-leaning walk.
  - C_s2 launched at 13:22.
- 2026-09-29 14:35: **C_s2 = FAST, so the headline replicates.** feet_ss 0.194 at 0.22 and 0.209 at 0.25 (n 90/95);
  lean 0.02–0.04; sweep 0.082–0.090. C_mo_s2 launched at 14:36 (X08).
- 14:37: the level-2 discriminators are queued behind C_mo_s2 (`phaseB/candidates_v4.yaml`, runner
  `phaseB/queue_after.sh`), under the autonomy rule, to avoid idle GPU while GATE.md is written:
  - **C_fa**: the fore-aft mirror alone, command direction visible. It separates the mirror's left-right coupling from
    direction blindness, and bears on portability, since full omni must see direction.
  - **C_nb**: C minus the band (mirror + scalar obs + RSI off).
  - **C_nr**: C minus RSI-off (mirror + scalar obs + band, RSI on).
  - C_nb and C_nr explain which knob completes C's speed with the pair (C_mo ≈ 0.17 against C ≈ 0.21).
