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
- 2026-09-29 14:44: **User: "stop after C_mo_s2 and write up."** The level-2 runner (C_fa, C_nb, C_nr) was killed
  before any launch; `candidates_v4.yaml` was not applied.
- 2026-09-29 15:48: **C_mo_s2 = FAST.** feet_ss 0.203 at 0.22 and 0.210 at 0.25 (n 95/98); sweep 0.086–0.087; lean
  in 20–33 % of episodes; early read 0.700, final track_ratio 0.870.
  - **C_mo's seeds disagree** (s3 INTERMEDIATE, s2 FAST). The decision-default seed-1 tie-break was not run because
    of the user's stop.
  - The queue ended (DONE). Total ≈ 9.9 GPU-h: 8 arms plus checks, over the 8.5 cap; reported, not stopped, per the
    autonomy rule. Write-up: [GATE.md](GATE.md).
- 2026-09-29 16:30: **GATE.md written** and checked independently (fable·high recompute plus an opus·high skeptic;
  every number and class reproduces).
  - **Correction adopted:** lean_lvl subtracts a hold-start offset of about ±0.15 m on these from-scratch arms, so the
    10:52 note "the lean is gone" (C_mo s3) was an artifact. By the absolute offset, C_mo s3 halves the lean (0.11 m).
    Every SLOW arm leans (0.16–0.21 m) and no FAST run does (≤ 0.056 m).
  - **Also adopted from the check:**
    - the C_mo → C gap is unresolved at n = 2;
    - skipping the seed-1 tie-break is a recorded deviation;
    - C_fa is low-information in 1b;
    - the portable test is a two-direction side run (fa vs 4-image mirror).

## Next steps (user, 2026-09-29 ~18:00: "run the next steps") — pre-registered before any result

**Step 1: C_mo seed-1 tie-break** (reference `C_mo_s1`). Launched 18:09 (X09).
- C_mo's class is the majority of its three seeds. With no majority, it is INCONCLUSIVE.
- If the class is INTERMEDIATE (or INCONCLUSIVE), queue C_nb and C_nr (one seed each) to attribute the C_mo → C gap.
  If FAST, skip them.

**Step 2: two-direction side test** (chains `LR4` and `LRfa`, seed 3, queued after C_mo_s1).
- Both arms are 1b from scratch with `KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4`, `KRABBY_OMNI_DIR_BAND=''`
  and `KRABBY_OMNI_SLOTS=0.2:0.8:0:0`, with the command direction visible.
- LR4 keeps the 4-image mirror. LRfa uses the fore-aft-only mirror.
- **Eval.**
  - The driver's `side22_25` covers lat_l only.
  - A by-hand run of `scenarios_side_lr.yaml` (`side22_25lr__omni`: lat_l and lat_r at 0.22 and 0.25) covers both
    hands. It uses the exact env_vars of the arm's own driver eval.
- **Class per hand**, with the same thresholds as the gate. An arm's class is the class both hands share. If the
  hands differ, or either is INTERMEDIATE, add seed 2 for that arm (`LR4_s2` / `LRfa_s2`).
- **Reading:**

  | LRfa | LR4 | Reading |
  |---|---|---|
  | FAST | SLOW | The left-right mirror tie itself forms the slow gait, even with −y really sampled → step 3 is warranted |
  | SLOW | SLOW | Two-direction training forms the attractor whatever the mirror; the one-direction benefit does not port → stop |
  | FAST | FAST | Sampling both directions for real removes it, so the 1b attractor came from phantom −y rows with no real −y data. Report; step 3 not needed on this evidence |
  | SLOW | FAST | Unexpected; report |

- **Known asymmetry, shared by both arms.** RSI stays at 0.2 from the source-1 bank, whose frames are all +y. So RSI
  seeds only left-walking starts. The per-hand read shows whether that matters.

**Step 3: full omni with the fa mirror.** Only if step 2 reads FAST/SLOW. It must gate fwd and turn as well, since M_y
carries the forward tripod-set handedness constraint. It is designed after step 2.
- 2026-09-29 19:18: **C_mo_s1 = FAST.** feet_ss 0.184 at 0.22 and 0.185 at 0.25 (n 96/98); lean_abs 0.065/0.074;
  sweep 0.069–0.076.
  - **C_mo's 3-seed majority is FAST** (s3 INTERMEDIATE by 0.002, s2 FAST, s1 FAST): the one-direction pair alone is
    sufficient.
  - Per the rule, C_nb and C_nr are skipped.
  - The runner applied `candidates_v7.yaml` and launched LRfa at 19:18 (X10).
- 2026-09-29 20:33: **LRfa = SLOW on both hands.** Both-hands eval:
  - left: 0.061 at 0.22, 0.065 at 0.25; lean_abs 0.25–0.26; sweep 0.036–0.038;
  - right: 0.087 at 0.22, 0.088 at 0.25; lean_abs 0.16–0.18; sweep 0.033–0.034.
  - The driver's lat_l eval agrees (0.061/0.064).
  - The hands agree, so no seed 2. With both side directions really sampled, the fore-aft mirror does not stop the
    slow gait; the left hand, the one the RSI bank seeds, is even slower than O.
  - LR4 launched at 20:33 (X11).
- 2026-09-29 21:48: **LR4 = SLOW on both hands.** Left 0.085 at 0.22, 0.091 at 0.25; right 0.082 and 0.096; lean
  0.11–0.13; sweep 0.033; cadence 1.0–1.1 Hz. The hands agree, so no seed 2.
  - **Pre-registered row LRfa SLOW + LR4 SLOW:** the one-direction benefit does not port → STOP. Step 3 is not run.
  - The queue ended (DONE lr-queue). Write-up: the GATE.md addendum.
  - The next discriminator (LR4_dwell, whole-episode command hold) is drafted in `scripts/candidates_v8.yaml`, not run.
    It needs a `KRABBY_RESAMPLE_S` dump check first.

## Reversal test (user, 2026-09-29 ~22:15: "run the dwell test") — pre-registered before any result

- **LR4_dwell** is LR4 plus `KRABBY_RESAMPLE_S=40:40`. The command is drawn once per 40-s episode, so there are no
  left↔right reversals inside an episode. 1b from scratch, 3000 its, seed 3; the 4-image mirror; direction visible.
- **Driver addition first:** an `OVERLAY_DUMP_CHECKS` entry for `KRABBY_RESAMPLE_S`, with a unit test against a real
  dump. `tools_sha` moves; `code_sha` is unchanged.
- **Eval:** the driver's `side22_25` (lat_l), plus the both-hands `scenarios_side_lr.yaml` with the arm's own eval env.
  Same per-hand classes as the gate.
- **Reading:**

  | LR4_dwell | Reading |
  |---|---|
  | FAST on both hands | Reversals inside an episode form the shuffle. Levers for full omni: longer side dwell, fewer or gentler reversals, a reversal-onset curriculum |
  | SLOW on both hands | Training two directions at all is the cause, not reversals. Lever: direction-specific policy structure |
  | Hands disagree or INTERMEDIATE | Seed 2 (`LR4_dwell_s2`) |

- **Caveat.** At 40:40 each episode is 20 % stand, 40 % left or 40 % right, so stand is now per episode rather than per
  segment. The RSI bank still seeds only +y starts.

## Seed-lottery check (user question, 2026-09-29 ~23:45: "is it possible all of this is caused by the seed lottery?")

The single-seed negatives rest on seed 3: O, R, C_band, C_rsi, LRfa, LR4 and LR4_dwell. The weakest link is the control
O. Queued behind LR4_dwell_s2 (runner `phaseB/queue_lottery.sh`, ≈ 2.4 GPU-h), using the existing references:
- **O_s2:** O at seed 2. FAST or INTERMEDIATE would mean the one-direction fast gait can form under O's config by
  luck. "The pair is the factor" would then weaken to "the pair makes it reliable", and more O seeds would be needed.
  SLOW gives O slow at 2 seeds, plus the historical 1b run.
- **LR4_s2:** LR4 at seed 2, with the both-hands eval. FAST on either hand would make two-direction slowness
  seed-dependent. SLOW is consistent with the 18+ earlier two-direction omni stamps.

RSI is on (0.2, source 1) in O, R, C_band, C_mo, LRfa, LR4 and LR4_dwell; it is off only in C, C_s2 and C_rsi.
- 2026-09-29 23:30: **LR4_dwell s3 = DEAD on both hands** (X12).
  - Left 0.047 at 0.22 and 0.058 at 0.25; right 0.055 and 0.028.
  - 42–51 % of the included episodes stall (< 0.03), and 6–14 % more are excluded as static.
  - The episodes that walk move at 0.067–0.070, leaning 0.16–0.21 m, with a 0.026–0.030 sweep.
  - The harness is clean: schedule deviation 0, no override warning, eval env consistent, the dump shows the 40-s
    resample. The track ratio sits at 0.47–0.51 throughout.
  - Provisional reading: removing within-episode reversals does not give the fast gait and adds a stall mode, so
    reversals are not the cause (pending seed 2).
  - The pre-registered table had no DEAD row. The campaign decision default for DEAD is "diagnose the harness, then
    seed 2", which was followed.
- 23:31: LR4_dwell_s2 launched (X13). 23:40: O_s2 and LR4_s2 queued behind it (seed-lottery check).
- **23:43: PAUSED by the user** ("Pause these tests for now. I'd like to work on a single test plan").
  - Both runners were killed. LR4_dwell_s2 was stopped gracefully at about iteration 400/3000; the driver parked at
    `await_user:stop`.
  - **Do not resume X13.** A resume restarts PPO counters, and the run tags collide. If it is needed, rerun from
    scratch.
  - O_s2 and LR4_s2 never launched. The campaign file is `candidates_v9.yaml`.

## Two-direction C (user, 2026-09-30: "Lets run two-direction C") — pre-registered before any result

- **C_lr** is 1b from scratch, 3000 its, seed 3. Overlay: `KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4`,
  `KRABBY_OMNI_DIR_BAND=''`, `KRABBY_OMNI_SLOTS=0.2:0.8:0:0`, `KRABBY_OMNI_MIRROR_IMAGES=fa`, `KRABBY_RSI_FRAC=0`,
  `KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25`.
- This is C with both side directions. The command direction must be visible, because C's scalar obs cannot tell left
  from right.
- It differs from LRfa only in RSI off (LRfa: 0.2, left-walking starts only) and the band (0.35 vs 0.25).
- **Eval:** the driver's `side22_25` (lat_l), plus the both-hands `scenarios_side_lr.yaml` with the arm's own eval env.
  Same per-hand classes as the gate.
- **Reading:**

  | C_lr | Reading |
  |---|---|
  | FAST on both hands | RSI-off and/or the wider band matter once both directions are trained. Next: split them (two arms) |
  | SLOW on both hands | Nothing in C's bundle carries over to two-direction training |
  | DEAD | Diagnose the harness, then seed 2 |
  | Hands disagree or INTERMEDIATE | Seed 2 (`C_lr_s2`) |

- **A single-seed caveat applies**, as in the seed-lottery discussion. The O_s2 and LR4_s2 checks are still unrun.

## Decision tree after C_lr (user, 2026-09-30 ~00:25) — pre-registered before C_lr's result

The user's words: "one more test to run if C_lr fails is a configuration of C_lr but with 10 second episodes with a
single command (resetting between episodes). If either C_lr or C_lr_10 succeed, then test adding back in the forward
and backward movement to the successful configuration and train through all parts of phase 1. If both C_lr and C_lr_10
fail, then run an analysis on all the sideways tests so far and propose next steps. Wait for me either after training
all parts of phase 1 or after proposing next steps."

**Success and failure** are judged on the both-hands side eval.
- **Success:** FAST on both hands.
- **Fail:** SLOW or DEAD on both hands.
- **Ambiguous:** anything else. That triggers a seed-2 run; if still ambiguous, a manual decision under the decision
  defaults.

**The branches** (`scripts/candidates_v11.yaml`, dry run clean; runner `scripts/queue_p1.sh`):
- **C_lr_10:** C_lr plus `KRABBY_EPISODE_S=10` and `KRABBY_RESAMPLE_S=20:20`, i.e. one command per 10-s episode with a
  reset between. 1b from scratch, 3000 its, seed 3. A dump check for `KRABBY_EPISODE_S` was added to the driver.
- **FB_\<cfg\>:** the successful config with fwd and back added. Pool `stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2`,
  1b from scratch, 3000 its.
  - The driver runs the `suite` eval (fwd and back readable there); the both-hands side eval is the side gate.
  - It is reported, not a stop: the user asked to go on through phase 1.
- **P1_\<cfg\>:** the phase-1 record trained with the config on every stage.
  - Stages: C1d t3 3000 → c4 1500 → c5 1500 → c6 2000 → c6x 4000, with the clock removal set K on c6x only;
    12000 its, seed 3.
  - Pools, slots, facing and yaw weight are verbatim from the record. RSI_SOURCE is dropped, since it is inert with
    RSI off.
  - Evals: `suite` at t3, c4, c6 and c6x (c5 none), matching the record's stage evals in the policy manifest. Then the
    both-hands side eval of the c6x head, and of W as the comparison.
- **Both fail:** analyse all the sideways tests so far and propose next steps, then wait for the user.

**Known design risks** (to report, not gates):
- The fa mirror drops M_y, which in full omni carries the forward gait's tripod-set handedness constraint. P1's suite
  evals show the fwd and turn cost.
- With 10-s training episodes, the side eval's 12-s holds after a 2-s settle run 4 s past any training episode.
- Terrain promotion is per-episode, so shorter episodes change the terrain curriculum pace.
- 2026-09-30 00:45: **Independent check of the decision-tree chains (opus·high).**
  - **No chain defects.** Every new stage composes to the record plus only the intended knobs.
  - **Runner bug fixed before any launch:** the c6x head was chosen by an alphabetical sort (it would have picked
    model_9900 over model_11995). Now `sort -V`.
  - **Rule change:** a DEAD hand in C_lr_10 now triggers seed 2 before a fail. This matches the campaign default for
    DEAD.
  - **Corrections to the tree's risk notes:**
    - **Terrain promotion is not a risk.** The light flat mode freezes terrain levels. What 10-s episodes do change is
      ring and obstacle exposure: 10-s walks barely reach the 2-m clear radius.
    - **The policy manifest does have a c5 suite stamp**; P1 has no c5 eval, as in the ladder record.
    - **The _10 variants never train a command switch**, while the evals switch from settle to hold at 3.5 s. This
      also confounds P1_C_lr_10's move_turn survival against W.
- 2026-09-30 01:15: **C_lr = SLOW on both hands.** Left 0.072 at 0.22 and 0.077 at 0.25; right 0.063 and 0.073;
  lean_abs 0.19–0.22; sweep 0.033–0.039; early read 0.524.
  - With both side directions trained, C's fa mirror + RSI off + band 0.35 still forms the slow shuffle.
  - Tree verdict FAIL → C_lr_10 launched automatically at 01:15 (unit `...-1790745322`).
- 2026-09-30 02:30: **C_lr_10 = SLOW on both hands.**
  - Left 0.085 at 0.22 and 0.085 at 0.25; lean_abs 0.15; sweep 0.051–0.054; 18 % of episodes short.
  - Right 0.067 and 0.073; lean_abs about 0 (!); sweep 0.039–0.041.
  - **Tree result: BOTH_FAIL.** The runner stopped (DONE p1-queue). Next, per the user: an analysis of all the sideways
    tests and proposed next steps, then wait for the user.
- 2026-09-30 03:20: **Analysis of all sideways tests: [ANALYSIS.md](ANALYSIS.md).**
  - Workflow: four analysts (evidence table, mechanisms, training dynamics, structural options), then a synthesis, then an
    independent fable·high critique. The critique found 1 blocking and 16 minor issues, all applied: step 3 reframed as
    validating the lever, the O→C_mo M_x-weight confound, step 5b promoted, a 5c no-mirror arm, and a penalty-replay
    side item.
  - Scripts: `scripts/analysis/`. Nothing launched. **Waiting for the user** (per their tree).

## Mirror canonicalization → two-direction RSI banks → C_lr + RSI (user, 2026-09-30) — pre-registered before any result

The user's words: "Let's try the Mirror canonicalization test. If that works in the eval use the same method to harvest
new RSI banks for both directions of walking, then test training with the same configuration as C but with RSI on and
using the new banks". At the question, the user chose **"C in both directions"** for the training arm.

**Step A: canonical eval.**
- The hook: `eval_crab_hex_gait.py --canon {wave,cmd}` (ANALYSIS.md step 1). For a −y command, the policy gets M_y(obs)
  and its action is mapped back with M_y.
- Heads (each with its own driver-eval env): C (X02), C_s2 (X07), C_mo_s2 (X08), C_mo_s1 (X09).
- Evals:
  - `scenarios_side_lr.yaml`, both hands, in both wave and cmd mode;
  - `scenarios_side_reversal.yaml` (rev_direct, rev_pause) in cmd mode.
- **"Works"** means, for each head, in wave mode: lat_r class FAST at both speeds, and |lat_r − lat_l| ≤ 0.03 at each
  speed.
  - It must hold on at least 3 of the 4 heads, and cmd mode must agree with wave on those heads.
  - Reversals are reported (falls, stalls, post-reversal speed from the raw series), not gated. The pause holds have no
    scored window.

**Step B: harvest, only if A works.**
- Run the canonical eval harness (wave mode) with the best canonical head whose training band covers 0.35, preferring C
  or C_s2.
- The manifest has lat_l and lat_r holds across the band.
- Convert the raw npz frames into a bank using the existing bank's filters (`2026-09-16_1805_omni_formation/RSI_BANK.md`):
  - upright > 0.95, speed along the travel axis > 0.15, |wz| < 0.3;
  - episode step ≥ 100, thinned every 3rd step;
  - travel angle within ±20° of ±y, |v| in 0.15–0.5;
  - the same array fields and meta.
- The bank covers +y and −y, balanced by direction.

**Step C: training.**
- **C_lr_rsi** is the C_lr overlay (both directions, direction visible, fa mirror, band 0.35) plus `KRABBY_RSI_FRAC=0.2`
  and `KRABBY_RSI_BANK=<new bank>`, source selection covering both directions. 1b from scratch, 3000 its, seed 3.
- Eval: the driver's side22_25, plus the both-hands side eval. Same per-hand gate.
- **Reading:**
  - FAST on both hands: two-direction seeding of the fast gait rescues two-direction training (C_lr was SLOW with RSI
    off).
  - SLOW: seeding does not change the formed gait.
  - Ambiguous: seed 2.
- **Note:** the harness change moves the omni `eval_sha`, so this campaign's next launch needs `--accept-eval-change`.
- 2026-09-30 09:00: **Step A (canonical eval) WORKS: 4 of 4 heads.** Wave mode, lat_l → lat_r at 0.22 / 0.25
  (`gate/canon_lr.json`):

  | Head | lat_l | lat_r (mirrored) | Class |
  |---|---|---|---|
  | C_mo_s2 | 0.203 / 0.211 | 0.199 / 0.209 | FAST / FAST |
  | C | 0.215 / 0.217 | 0.211 / 0.216 | FAST / FAST |
  | C_s2 | 0.194 / 0.208 | 0.190 / 0.203 | FAST / FAST |
  | C_mo_s1 | 0.184 / 0.185 | 0.182 / 0.191 | FAST / FAST |

  - |L−R| ≤ 0.006 at every speed, and cmd mode agrees (lat_r 0.180–0.212). The sweep is long on both hands
    (0.073–0.086).
  - **Reversals** (cmd mode, rev_direct / rev_pause): the post-reversal lat_r and the return lat_l hold the pre-reversal
    speed on every head, with no stall mode. Falls per 8-s hold track each head's baseline: C_mo_s1 1–2 %, C_mo_s2 3–9 %,
    C / C_s2 7–19 %.
  - This is the first omni policy that walks FAST in both side directions, with no training.
- 09:05: **Step B + C queued** (runner `scripts/queue_rsi.sh`):
  - harvest with C (s3; its band covers 0.35) on `scenarios_harvest_side.yaml` (lat_l / lat_r at 0.22, 0.28, 0.35,
    canon wave);
  - `scripts/harvest_canon_bank.py` → `rsi_bank_side_lr.npz`, 2000 frames per direction, tags 1 = +y and 2 = −y, with
    the 2026-09-16 filters;
  - then **C_lr_rsi** (`candidates_v12.yaml`): C_lr + RSI 0.2 from the new bank, source all, with the both-hands eval.
  - Driver additions: an `OVERLAY_DUMP_CHECKS` entry for `KRABBY_RSI_BANK`, plus a test fix.
- 2026-09-30 09:05: **Harvest + bank.**
  - C head canonicalized, wave mode (`gate/harvest_C.json`). feet_ss, left / right:
    - 0.22: 0.215 / 0.211;
    - 0.28: 0.207 / 0.199;
    - **0.35: 0.156 / 0.145, with lean 0.15 and sweep 0.05.** C degrades toward the slow gait at the top of its band.
  - Bank `rsi_bank_side_lr.npz` (sha256 `fd6704ae…`; `gate/harvest_bank.json`): 2000 frames of tag 1 (+y) and 2000 of
    tag 2 (−y). Speed p10/50/90 is 0.19 / 0.28 / 0.40, angle within ±14° of ±90, root_z 1.07.
    - About 32 % of the candidate pool came from the 0.35 holds, the partly leaned gait. This is a caveat for the
      reading; it was pre-registered as "across the band", so it was not re-harvested.
  - **C_lr_rsi launched 09:05 (X16).** Its env.yaml carries `bank_path` = the new bank, `fraction` 0.2 and `source` all.
- 2026-09-30 10:22: **C_lr_rsi = SLOW on both hands.**
  - Left 0.085 at 0.22 and 0.085 at 0.25; lean_abs 0.26–0.27 (the largest of any run); sweep 0.035.
  - Right 0.069 and 0.075; lean_abs 0.21; sweep 0.034–0.037.
  - Early read 0.512.
  - Seeding 20 % of starts from the fast canonical gait in BOTH directions does not stop two-direction training from
    forming the short-stroke shuffle (C_lr, with RSI off, was 0.063–0.077).
  - Caveat: about a third of the bank's candidate pool came from C's degraded 0.35 holds.
  - The user's instruction chain is complete: canonical eval WORKS; bank harvested; C_lr_rsi SLOW. **Waiting for the
    user.**

## C_lr_10_rsi (user, 2026-09-30: "lets try C_lr_10_rsi") — pre-registered before any result

- **The arm.** C_lr_10 (both directions, direction visible, fa mirror, band 0.35, `KRABBY_EPISODE_S=10`,
  `KRABBY_RESAMPLE_S=20:20`) plus RSI 0.2 from `rsi_bank_side_lr.npz`, source all (`scripts/candidates_v13.yaml`).
  - It differs from C_lr_10 only in the three RSI knobs, and from C_lr_rsi only in the episode structure.
  - With one command per 10-s episode, an RSI-seeded episode walks the frame's direction and speed for its whole
    length.
- **Eval, gate and rule** are the same as for C_lr_rsi:
  - FAST on both hands: single-command episodes plus fast-gait seeding together rescue two-direction training.
  - SLOW: they don't.
  - Ambiguous or DEAD: seed 2 (`C_lr_10_rsi_s2`).

## RSI retry (user, 2026-09-30: "if C_lr_10_rsi fails as well as C_lr_rsi, retry both with an rsi harvest at 0.22–0.28 only and a 33% RSI ratio") — pre-registered

- **Trigger:** C_lr_10_rsi fails, meaning SLOW or DEAD on both hands at s3, or ambiguous at s3 with a FAIL at s2.
  C_lr_rsi has already failed. If C_lr_10_rsi does not fail, nothing runs and the result is reported.
- **Bank B:** canonical C (wave mode) on `scenarios_harvest_side_b.yaml` (lat_l / lat_r at 0.22, 0.25, 0.28 only,
  where canonical C is FAST; its 0.35 gait degrades), converted with `scripts/harvest_canon_bank.py` →
  `rsi_bank_side_lr_b.npz`, 2000 frames per direction.
- **Arms** (`scripts/candidates_v14.yaml`), each differing from its original only in `KRABBY_RSI_BANK` (bank B) and
  `KRABBY_RSI_FRAC` 0.33:
  - **C_lr_rsi33** = C_lr_rsi, then **C_lr_10_rsi33** = C_lr_10_rsi. Seed 3 each, run sequentially.
- **Gate:** the same both-hands rule. FAST on both hands is a success; SLOW is a fail; ambiguous or DEAD gets seed 2.
- Runner: `scripts/queue_rsi33.sh`.
- 2026-09-30 11:50: **C_lr_10_rsi s3: left DEAD, right SLOW.** Left 0.042 at both speeds; right 0.078 and 0.081;
  sweep 0.043–0.046; early read 0.545.
  - The verdict is ambiguous, because a DEAD hand triggers seed 2. C_lr_10_rsi_s2 launched at 11:50 (X18), with early
    read 0.535.
- 12:22: **The user stopped seed 2** ("kill the seed 2 training and move on to the C_lr_rsi33").
  - X18 was stopped gracefully at about it 1300. Don't resume X13 or X18.
  - The retry runs directly: `scripts/queue_rsi33_now.sh`, the pre-registered retry without its trigger check. That is
    bank B harvest → C_lr_rsi33 → C_lr_10_rsi33, with the same gate and seed-2 rule.
- 12:27: **Bank B.**
  - Canonical C at 0.22 / 0.25 / 0.28 is FAST on both hands: feet_ss 0.207–0.217, lean 0.03–0.07, sweep 0.079–0.086
    (`gate/harvest_b_C.json`).
  - `rsi_bank_side_lr_b.npz` (sha256 `743f7dd6…`; `gate/harvest_bank_b.json`): 2000 frames of +y and 2000 of −y,
    median speed 0.28.
  - **C_lr_rsi33 launched 12:27 (X19).**
- 2026-09-30 13:46: **C_lr_rsi33 = SLOW on both hands.** Left 0.094 at 0.22 and 0.103 at 0.25 (lean 0.21–0.22, sweep
  0.027); right 0.069 and 0.072 (lean 0.18, sweep 0.05). Early read 0.522.
  - Bank B (0.22–0.28 only) at RSI 0.33 changes nothing: seeding a third of the starts in the fast gait, in both
    directions, still yields the shuffle.
  - **C_lr_10_rsi33 launched 13:46** (X20), per the user's retry instruction.
- 2026-09-30 14:10: **Root-cause analysis: [ROOT_CAUSE.md](ROOT_CAUSE.md)** (user: "aim for the root cause of the slow
  gaits"). Four probes (symmetry decomposition, representation, dynamics, actuation), a synthesis and a fable·high
  check, all at zero GPU. Scripts in `scripts/rootcause/`.
  - Established: every network trained on both hands lacks the consistent direction-flipped, phase-locked rod stroke
    (pooled odd RMS ≤ 0.93 vs ≥ 1.50 in every fast gait); the stepping is intact and the direction rides a static
    carrier. One-direction nets form the stroke by it ~600. Capacity, reward, input strength, RSI, band, episode
    structure and mirror type are ruled out.
  - Not established: why PPO misses it. Three candidates: the direction × phase interaction barrier; rate × early
    saturation lock-in; the loss/RL conflict (phantom arms only).
  - Controls named: C_fa and C_40 (zero code), T1 checkpoint series, T2 with its additive control. Nothing launched.
- 2026-09-30 15:05: **C_lr_10_rsi33 = SLOW on both hands.** Left 0.080 / 0.082, right 0.080 / 0.081 (lean 0.15–0.17,
  sweep 0.037–0.058, early read 0.540). The RSI retry queue is done: bank B at RSI 0.33 rescues neither the 40-s nor
  the 10-s single-command variant. **Every RSI-seeded two-direction arm (4 of 4) is SLOW.** Nothing running; waiting
  for the user on the ROOT_CAUSE.md controls.

### 2026-09-30 15:15 — the two ROOT_CAUSE.md controls (user: "Run those two controls")

- **Arms** (`scripts/candidates_v15.yaml`, dry run `scripts/candidates_v15_dryrun.txt`). Each is 1b (side_l only) from
  scratch, 3000 its, seed 3, scored on the driver's `side22_25` row (lat_l) with the usual gate. Both train only +y,
  so there is no both-hands eval. Run serially: C_fa, then C_40.
  - **C_fa** = O + `KRABBY_OMNI_MIRROR_IMAGES=fa` (so C_mo minus the scalar obs). RSI 0.2 and band 0.25 as in 1b.
  - **C_40** = C with `KRABBY_OMNI_CMD_POOL=stand:0.6,side_l:0.4`, `KRABBY_OMNI_DIR_BAND=''`,
    `KRABBY_OMNI_SLOTS=0.6:0.4:0:0`. That is one hand at C_lr's per-hand share; direction hidden (scalar obs), fa
    mirror, RSI off, band 0.35. The composed env differs from C in those three keys only.
- **Timing reference** (training `track_ratio_move_mean`, its 400 / 600 / 800 / 1000). Fast one-hand runs (C, C_s2,
  C_mo ×3) are 0.48–0.56 / 0.56–0.69 / 0.63–0.75 / 0.67–0.78. Slow runs (O, R, LRfa, C_lr) are 0.47–0.55 / 0.51–0.56
  / 0.49–0.57 / 0.47–0.56. `track_ratio_mean` equals the moving-only ratio at every one of these points, so C_40's
  larger stand share (0.6) does not dilute the read. The bands are a guide, not a gate.
- **Pre-registered reading, C_fa** (P6, predicted FAST ≈ C_mo):
  - FAST: a visible direction slot does not by itself reform the shuffle in a one-direction run. Two-direction arms
    with the direction visible are interpretable, and C_mo's scalar obs is not needed.
  - SLOW: the visible slot alone reforms it. C_lr's slowness is then confounded with the visible slot, and the next
    step is a two-direction arm with the direction hidden only from the actor's clock-coupled path (not designed).
- **Pre-registered reading, C_40** (P10):
  - FAST, moving ratio ≥ 0.58 by it 800: halving the +y rate does not delay formation. That rules out rate × lock-in
    (ii) as the sufficient cause and leaves (i) the interaction barrier (or (iii) in phantom arms).
  - FAST but late (≥ 0.58 only after it 800): rate slows formation, so (ii) contributes. (i) is still needed for
    C_lr never forming.
  - SLOW: the halved rate alone reproduces the shuffle with no second direction. (ii) is then sufficient, and the fix
    ranking changes: a per-direction rate lever (longer training, a higher side share, a slower early std/LR schedule)
    becomes a first test before any structural fix.
- **Rules:** INTERMEDIATE or DEAD → that arm's seed 2 (C_fa_s2, C_40_s2 are defined), run after both seed-3 arms.
  Cost about 2.5 GPU-h (5 GPU-h with both seed 2s). Runner: `scripts/queue_controls.sh`.
- 2026-09-30 16:26: **C_fa s3 (X21) = FAST.** feet_ss 0.211 at 0.22 and 0.206 at 0.25; lean_abs 0.028 / 0.068; sweep
  0.084 / 0.066 (`outputs/.../arms/C_fa.json`). Early read 0.794, the highest of the campaign (it 600 0.673, it 800
  0.771), on the fast schedule.
  - Reading (pre-registered): a visible direction slot does not reform the shuffle in a one-direction run. The scalar
    obs in C / C_mo is not needed, and the fa mirror alone moves O to FAST. Two-direction arms with the direction
    visible (C_lr and its variants) are interpretable as run.
  - lean_abs 0.068 at 0.25 is a little above the 0.056 ceiling seen in earlier fast runs; a label only.
  - C_40 launched 16:26 (X22); composed env checked against the design.
- 2026-09-30 16:55: **C_40 early read 0.325, written before its verdict.** It is below every earlier arm, slow ones
  included (it 400 / 600 / 800 / 1000: 0.171 / 0.241 / 0.295 / 0.355, still rising).
  - The stand share does not dilute this number. Each episode's ratio is along-command distance over commanded
    distance, and stand rows add to neither.
  - Other training signals: the progress reward is 0.009–0.018 against 0.05–0.09 for C and C_fa; tracking reward is
    about 0.6, carried by the stand rows; episode length falls from about 1960 at it 200 to about 1460 at it 600.
    The arm learns to stand first and walks late.
  - **Confound, noted before the verdict:** C_40 halves the total moving share (0.4 against 0.8), not only the +y
    share. A SLOW or DEAD verdict therefore cannot separate a per-direction rate effect (candidate ii) from a slower
    or stand-dominated start. The pre-registered reading "SLOW → (ii) sufficient" is weakened to "rate of lateral
    experience can reproduce the shuffle". A clean (ii) test keeps the moving share at 0.8 with +y at 0.4 and the other
    0.4 on a command that needs no lateral stroke flip. None of those is free of a second command, so none is designed.
- 2026-09-30 17:41: **C_40 s3 (X22) = DEAD.** feet_ss 0.035 at 0.22 and 0.031 at 0.25. 38 and 46 of 100 episodes die
  inside the hold ("short"); lean_abs 0.38–0.40 m; sweep 0.043–0.050 (`outputs/.../arms/C_40.json`).
  - Training: the ratio reaches only about 0.52 by it 2500–2950, and the progress reward stays at 0.016–0.019. Falls
    end 0.75 of episodes from it 2000 on, against 0.02–0.11 in O, C, C_fa and C_lr. Noise std reaches 2.0 as in every
    arm, so that is not the cause.
  - **Not the shuffle phenotype.** C_lr has the same +y share (0.4) but walks a stable slow shuffle with falls at about
    0.11. C_40 falls instead. So the lower lateral rate here produces a different failure, and, with the moving-share
    confound above, it neither supports nor excludes (ii) as C_lr's mechanism.
  - By rule, DEAD → **C_40_s2 launched 17:41**.
- 2026-09-30 18:58: **C_40_s2 (X23) = DEAD.** feet_ss 0.006 at both speeds, body_ss 0.012–0.022; only 2–4 episodes
  die in the hold; lean_abs 0.30 m; sweep 0.017–0.023 (`outputs/.../arms/C_40_s2.json`). Early read 0.180. The ratio
  peaks at 0.23 near it 1500 and falls back to 0.16 by it 2950. Falls stay at 0.03–0.09 of episode endings. This
  seed stands and leans instead of falling: a stand attractor.
- **Controls closed (runner "DONE controls").** Cost about 3.7 GPU-h (three arms).
  - **C_fa: FAST** (1 seed; the prediction held, and the early read was the campaign's highest). A visible direction
    slot is harmless in a one-direction run.
  - **C_40: DEAD on both seeds**, with two phenotypes: falls on seed 3, standing on seed 2. Neither is the stable
    shuffle C_lr walks at the same +y share. At a 0.4 moving share with RSI off, a one-hand arm forms no lateral gait
    in 3000 its.
  - **What it means for the root cause.** C_lr walks, just slowly: its stepping and carrier form from the pooled
    0.8 of moving experience, and only the direction-flipped stroke is missing. That matches ROOT_CAUSE.md 1 and 2.4.
    But C_40 does not separate candidate (i) from (ii). Its failure is upstream of the stroke, as expected under the
    moving-share confound recorded at 16:55. The mechanism question stays open. The fix (canonical training, a policy
    switch, or distillation) is the same under every candidate. Nothing queued; waiting for the user on the next step.

### 2026-09-30 19:09 — episode-length ladder on C_lr_10_rsi (user)

- **Request (verbatim):** "Lets try increasing the duration of the C_lr_rsi_10 configuration. 5 steps up of 10
  seconds each up to a max of 60 seconds. If any are fast confirm with a second seed. run all 5 steps regardless of
  the results". "C_lr_rsi_10" is read as the chain **C_lr_10_rsi** (bank A, RSI 0.2, 10-s single-command episodes).
- **Arms** (`scripts/candidates_v16.yaml`, dry run `scripts/candidates_v16_dryrun.txt`): **C_lr_20_rsi, C_lr_30_rsi,
  C_lr_40_rsi, C_lr_50_rsi, C_lr_60_rsi**, seed 3, 3000 its from scratch, run in that order.
  - Each sets `KRABBY_EPISODE_S` = E and `KRABBY_RESAMPLE_S` = E+10 : E+10. The resample time is longer than the
    episode, so each episode holds one command, as in C_lr_10_rsi (10 / 20:20).
  - The composed env of each differs from C_lr_10_rsi in those two keys only.
  - The 40-s rung is C_lr_rsi (40-s episodes, a new command every 10 s) with one command per episode.
  - RSI acts at reset, so seeded starts per unit time fall as 1/E. That is inherent to the request.
- **Gate:** the both-hands `side22_25lr__omni` eval (`scenarios_side_lr.yaml`) on model_2999, classed per hand as usual.
  A step **succeeds** if it is FAST on both hands.
- **Seed-2 rule (the user's; it replaces the campaign default):** a step that is FAST on either hand gets its `_s2`
  arm. The seed-2 arms run after all five steps, in step order. SLOW, INTERMEDIATE or DEAD steps get no seed 2 (a
  user-directed deviation from the "DEAD → seed 2" default). A success is confirmed only if the seed-2 arm is FAST on
  both hands too.
- **Prediction (ROOT_CAUSE.md):** SLOW or DEAD at every length.
  - Every network trained on both hands has lacked the direction-flipped stroke, whatever the episode structure: 10 s
    (C_lr_10, C_lr_10_rsi, C_lr_10_rsi33), 40 s with 10-s commands (C_lr, C_lr_rsi, C_lr_rsi33), and 40-s holds
    (LR4_dwell, 4-image mirror, RSI off).
  - A FAST step would put the episode structure back in play as a lever, and would contradict ROOT_CAUSE.md's
    "every two-hand network" claim for that configuration.
- **Reported per step:** class and feet_ss per hand, lean_abs, sweep, early read (its 800–1000), and falls.
- Runner: `scripts/queue_eplen.sh`. Cost is about 1.5 GPU-h per arm: about 7.5 GPU-h for the five, plus about 1.5 per
  seed-2 arm.
- 2026-09-30 19:52: **Ladder review** (read-only workflow: 4 reviewers, 2 refuters per medium+ finding, and a
  completeness critic; 11 agents; raw result in the session's workflow journal `wf_904a9186-b4e`). Nothing
  invalidates the ladder. Recorded before any rung's verdict:
  - **Eval is identical for every rung.** The both-hands eval runs its own 18-s episode (15.5-s scenario) and never
    receives `KRABBY_EPISODE_S` or `KRABBY_RESAMPLE_S`. The earlier 10-s arms were not truncated: their episodes end only
    at schedule completion or a fall. C_lr_10_rsi's left DEAD (0.042) is steady slow walking, not an artifact.
  - **Confound: obstacle exposure grows with E.** 1b is 80 % flat / 20 % ring tiles, with blocks 2.0–4.5 m from the
    spawn. At E = 10 the robots never reached the ring (r_max about 0.8 m). X24 (E = 20) already reaches blocks, with
    falls higher on ring spawns. Each rung is "longer episodes plus more obstacle contact". For each rung,
    crab_failure_obst / flat, reach_obst_frac, block_steps_mean and r_max_mean at its 2900–3000 go next to the class.
    From E ≈ 30, robots can leave the spawn tile, so those are lower bounds. A pure-flat control
    (`KRABBY_FLAT_TERRAIN_FLAT_FRAC=1.0`) was not run (the user's call).
  - **Correction to the 19:09 text:** the 40-s rung is not a single-factor contrast with C_lr_rsi. It also carries
    about 3× the staged-RSI-command share (0.20 against 0.07) and no in-episode reversals.
  - RSI 0.2 lowers the actual stand share to 0.16: staged rows are forced to move. The same holds on every rung. Staged
    commands are the frame's own bearing and speed, up to 20° off the lateral axis, and in single-command episodes they
    last the whole episode.
  - **Early reads are not comparable across rungs.** The ratio window spans more iterations, and the start-up transient
    weighs less, as E grows. Only the both-hands eval compares rungs.
  - **Direction hole in the gate (critic).** feet_ss is measured along each episode's own displacement, so a wrong-way
    walker scores fast. LR4_dwell, the closest precedent (40-s holds), had 36–44 wrong-way episodes per hold.
    **Added reading rule, pre-registered now:** a hand counts as FAST only if its along-command class is also FAST.
    That class uses `scripts/along_cmd.py`, with static episodes scored 0. The seed-2 trigger stays as registered:
    feet_ss FAST on either hand. `along_cmd.py` reproduces feet_ss within 0.004 on C_fa and C_mo_s2.
  - **Single-seed resolution:** hand-to-hand and seed-to-seed spread is 0.03–0.04 m/s. Only FAST vs not-FAST is read
    per rung; no trend inside the SLOW band is read as an episode-length effect.
  - **Runner handoff:** `queue_eplen.sh` (v1) was stopped at 19:52 while C_lr_20_rsi trained; the training unit was
    untouched. `queue_eplen2.sh` adopted the unit and continues the same plan. It adds the review's runner fixes: run
    dir by run name, head from state.json (follows an infra retry), unit death detected early with the journal logged,
    drivers_idle before each launch, the X17 eval-env fallback, and the along-command check logged per arm.
- 2026-09-30 20:23: **C_lr_20_rsi (X24, E = 20) = SLOW on both hands** (verdict FAIL). No seed 2 (not FAST).
  - feet_ss: L 0.071 / 0.095, R 0.054 / 0.052. Along the command: L 0.069 / 0.091 (SLOW), R 0.049 / 0.051 (DEAD by
    0.001); no wrong-way or static episodes. Excluded "short" 3–8 per hold. Early read 0.504.
  - Labels only (single seed): lean_abs 0.016–0.057, low for a slow arm (most leaned 0.12–0.22); sweep 0.079 / 0.089 on
    the left, in the range of fast gaits, 0.057 / 0.061 on the right. The left hand takes long strokes but moves
    slowly; cadence was not measured.
  - Obstacle telemetry at its 2900–3000 (`scripts/eplen_telemetry.py`), X17 (E = 10) → X24: r_max 0.82 → 1.41 m;
    reach_obst 0.001 → 0.225; block_steps 0.005 → 18.7; falls on ring spawns 0.033 → 0.112, on flat 0.041 → 0.078.
  - C_lr_30_rsi launched 20:23 (X25).
