# Plan — M18 Task 4 on the omni task: a gated, reproducible from-scratch pipeline (campaign `<date>_task4_pipeline`)

> **Audit trail** (cap 3, extended to 5 by you). Did not converge. Passes 3–5 kept returning to one root issue, the
> chance-HALT bound, which your pass-5 answer settled. The pass-5 fixes and your last answers were **not
> re-audited**. STOP-0 is the backstop: on CPU, before any GPU spend, it recomputes every line and its rate.
> - **Pass 1** (inventory, premise probes, 3 lens-split auditors): 19 blocking.
>   - Fixed: payload INVALID clause; tripod → WATCH; T\* fallback; student live, prescreen and paired rules; X
>     decisive only if bitwise; no-retry rationale and SIGTERM; preflight STOP-C triggers; per-camera grids; the
>     STOP-0 list; R18 split into two verdicts.
>   - Decided by you: 6.
> - **Pass 2** (opus + fable): 4 blocking, plus 1 scope question.
>   - Block z-test undefined → you chose the decisive list.
>   - Foot-force INVALID fired on falls → narrowed.
>   - K4 ≤ 1 failed 2 of 3 seeds → ≤ 3/800 (D9).
>   - p3a pilot read on the wrong checkpoint → fixed.
>   - ONNX → you kept it.
> - **Pass 3:** 5 blocking.
>   - A 3-seed line HALTs a healthy seed ≈ 6.7 % of the time (simulated) → you chose floors plus few lines.
>   - 0-kg check ambiguous → you chose paired.
>   - Floors halting p3b/p3c → measured ceiling.
>   - Sum models undefined → stated per class.
>   - C2 eval bridge → restored.
> - **Pass 4:** 6 blocking.
>   - Floors had no model (≈ 0.38) → you lowered them.
>   - The paired check failed S1's own continuation X12 → you added the drift margin.
>   - Row-list disagreement → one list.
>   - Stand rows missing from C2 → added.
>   - Untrained diag floor → trained categories only.
>   - Payload variance → 2σ̂² + SE².
> - **Pass 5:** 4 blocking.
>   - The lowered floors still cost ≈ 0.75, and no useful HALT set fits ≤ 0.20 → you chose **HALTs as prompts**.
>   - SN seeds missing from the sum → named.
>   - Drift applied at the wrong iteration count → you chose to scale it.
>   - Gate stamps lack `trained_categories` → now passed, and INVALID if missing.
> - **Gate check:**
>   - Walker HALTs are floors, plus p2a field_hard.
>   - p3a: stable-at-90 and the paired 0-kg check.
>   - Student: completion, paired against T\*. Stand: the G/K lines.
>   - Every chance rate is reported, not capped. The drift margin and live kills are `unvalidated`.
> - **Premises:**
>   - Bitwise to ≈ it 1905 → P-early.
>   - p3a reachable → the step-8 pilot, read on the full gate.
>   - 4 cameras: timing, parity and "depth helps" → step 8, with STOP-C.
> - **Corrections:** two of my estimates were wrong, the pass-3 "floors near-zero" and the pass-4 "≈ 1 % at
>   0.50/0.80". Your later answers replaced the choices built on them.
> - **Mechanisms for you:** none pending.
> - **Left for implementation (16 checks):**
>   1. STOP-A means the settle tips at ≤ 90 kg AND survival < 0.5, not "S1 unstable at 90".
>   2. The gate-0 carry env is stage-dependent (CLOCK_HIDE 0 through c6; no FACING_OBS_CLIP on P2-era stamps).
>   3. Lineage bridges: exclude the `trained_categories` metadata key, or compare with the ANC re-reads.
>   4. c5 needs a side22 row for its FAST floor.
>   5. A3: assert the realised mass from run_meta.
>   6. Skip the payload composite write at 0 kg.
>   7. 0-kg rows: either "no payload" or `0:0`, chosen once.
>   8. Pilot checkpoints every 500 its for N_p.
>   9. Replay every student live rule on the forward 3a logs.
>   10. Bridge the stand rows through the SH launcher, and compare aggregates, not files.
>   11. Budget-bound: define it on the pooled tail.
>   12. DESIGN.md also retires stable-at-L support/swing, the student grid and sweep lines, the §3.1 demotion order,
>       and the P-gates in REPRODUCED.
>   13. The freeze notice says commit or stash before the pin.
>   14. The report uses measured hours.
>   15. thresholds.json says whether the floors apply to the D1/D2 reads.
>   16. Stage T needs the paired 0-kg rows read on p3b/p3c first.
> - **Size:** 13 steps / ≈ 4,100 words (you reviewed 13 / ≈ 2,235). The growth is the blocking fixes plus your 15
>   answers.
> - **Audit routing:** pass-1 workflow; then 4× opus·xhigh + 4× fable·high auditors; main thread classified and
>   applied the fixes.

## Context

Grant M18 Task 4 (`patina-foundation-grants/.../Milestone18-Krab-Sim-Tuning/TASK-4-REPRODUCIBLE-CURRICULUM.md`) asks for
the milestone deliverable: one committed pipeline that trains the teacher from scratch through every stage, distills a
student, and gates every stage transition on harness metrics with committed thresholds, run once end to end, with the
final teacher + student bundled and tagged as the sim-to-real baseline and a baseline-vs-final report. It was written
for the old `crab_hexapod_task` (2b2 teacher 6300, student 9800, `runs/`, `parkour_mdp_cfg.py` stages). Today the
policy of record is the omni walker S1 s18 (`crab_hex_omni_task/policy/6_p2a_field/model_19994.pt`, lineage
`1_fb6k → 2_c4 → 3_c5 → 4_c6 → 5_c6x_clockfree → 6_p2a_field`) plus the stand network SH_T17.

**What exists and what does not (verified mapping, 2026-10-16).** The lineage is four hand-chained candidate chains
across two campaign dirs (`{reference:}` links and one literal run-dir checkpoint path), with no automatic gates; the
gates of record were campaign readers, relative to anchors. Task 0's harness exists (`crab_hex_forward_task/scripts/
eval_crab_hex_gait.py` + `gait_eval/`), but its 1 N contact metrics read the head as tippy-tap (0.47), while the 50 N
rigidity metrics show real steps, so gait is read (and reported) on the 50 N set. Teacher 6300 and student 9800 no longer load
(critic 1007 vs 1149 wide, pre-cam-migration). Task 1's gait terms are in. Task 2 is partly done: world facing
command; a robot-frame yaw-rate interface exists only as the harness `body_hold` adapter (S1 tracks 0.05–0.168 rad/s
at net ratio 0.91–1.03); no direction sweep. Task 3: the linkage speed side is modelled, the force side sits at
holding level by design, and the grant's actuator and mass figures are stale (as-built 2000 N hip / 500 N knee;
robot 230 kg, not 83). **There is no payload event or curriculum.** No omni student exists. The stand net is two
DAgger stages pinned to S1's sha.

**User decisions (2026-10-16):**
- **Student:** a depth student, with cameras matching the hardware: a ZED 2i at the front, and DFRobot SEN0583 ToF
  depth (320×240, 0.2–2 m) on the left, right and rear (M7 Task 2's chosen sensors).
- **Velocity:** obs cols 13–14 (true body velocity) stay privileged; a deploy-time velocity estimate goes on the M15
  list.
- **Payload:** built into this plan. The 90 kg level is a hard gate; 180 and 270 kg are ceilings.
- **Stand net:** both stand-net DAgger stages are included.
- **Runs:** R18, a full-pipeline seed-18 replay (reproducibility proof against the head of record, plus a replay of
  the stand tooling against SH_T17), and F19, the official fresh-seed gated run.
- **Ownership:** adoption, bake and push stay the user's.
- **Audit answers (2026-10-16):**
  - Gates HALT on the a-priori floors and a few capability lines (Rule); everything else is WATCH. Pass 2 dropped
    block z-tests, which at about 45 reads masked ANCW's field_hard miss. Pass 3 cut the list after a 3-seed line
    was found to HALT a healthy seed about 6.7 % of the time.
  - The ONNX export stays in the student step.
  - The 0-kg check is capability only, paired against the run's own p2a head, with the zero-lever continuation
    drift in its margin.
  - HALTs are prompts. A HALT you accept as chance lets the run continue, and it stays eligible as the deliverable
    (disclosed). The summed chance rate is reported, not capped, since no useful HALT set fits a hard ≤ 0.20 bound
    with 3 recorded seeds.
  - Floors at s1–c6x are lowered, with values fixed at STOP-0 next to their rates. Two earlier estimates were wrong:
    the pass-3 "near-zero" for the floors, and the pass-4 "≈ 1 % at 0.50 / 0.80".
  - Budget-bound is read from the training tail.
  - The force side is report-only (sim plant).
  - The direction sweep is report-only.
  - R18 stays full pipeline (a second sample after ≈ it 1905, and the fallback deliverable).
  - The stand nets are teacher-side (paired with T\*).

**Design detail of record:** the verified synthesis at
`/home/nickmagus/.claude/plans/validated-scribbling-kazoo-agent-aaedba0f25c1734d0.md`. It holds every file:line, the
gate tables and the code facts re-checked against three independent designs, and is copied to `PL/DESIGN.md` at
step 0. Where this plan differs, this plan wins; the difference is the camera layout, now four cameras.

## Shape (one spec, one entry point, no manual hand-off)

- `PL/pipeline.yaml` (spec, seeds, overlays verbatim from the lineage files with sha checks, gate rows, thresholds
  version) + `TP/pipeline.py gen | check | run | continue | status | gate | report` (TP = `crab_hex_omni_task/
  experiments/tools/pipeline/`, undated and reusable; PL = the dated campaign dir with run dirs `preflight/ R18/ F19/`).
- `gen` writes one candidates file per run in which every stage resumes the previous stage by `{reference:}` inside
  the SAME file (replacing the cross-campaign references and the literal p2a checkpoint path); the queue runs each
  stage through the unchanged `run_candidates.py --reference-chain`, then the gate rows (`bc_eval_row.sh`, raw npz
  kept), then a CPU gate read that writes `PL/<run>/gates/<stage>.json` and halts on FAIL (built from the body-cmd
  campaign's `bc_train_queue.sh` / `bc_after_stage.py` / `bc_gate_read.py` patterns).
- Stage table per run (seeds: R18 walker 18 / DAgger 17; F19 = 19 throughout):

| # | Stage | Base / task | Resume | Its | Overlay |
|---|---|---|---|---|---|
| W1–W5 | s1, c4, c5, c6, c6x | 1b, then 1a / Omni-Walk | scratch → previous | 6000/1500/1500/2000/4000 | the shipped FB6k__B / SF__B_c4 / SF__B_rest overlays |
| W6 | p2a | 2a / Omni-Teacher | previous | 5000 | P2__S1_s18 overlay |
| P1–P3 | p3a / p3b / p3c | 2a | previous | N_p each (pilot) | p2a + payload 0:90 / 0:180 / 0:270 kg (25 % zero, CoM shift); p3b/p3c only if the previous level holds |
| T | T\* | CPU | — | — | head with the highest stable payload that passes its 0-kg check |
| S | student + selection | Omni-Student (4 depth cameras, 192 envs) | T\* | 5000 | T\*'s env overlay; screen of +1k…+5k checkpoints |
| D1–D2 | stand SN → SH (teacher-side) | Omni-Teacher + T\* frozen | warm start chain | 4000 / 5000 | recipes of record, re-anchored on T\* |
| X | stand replay (R18 only) | beside the bundled S1 | — | 4000 / 5000 | recipe of record → compare with SH_T17 |
| B | bundle | CPU | — | — | keep-set + README + local tag |

## Gates (absolute, committed, read on CPU from the harness stamps)

- **Rule** (`TP/thresholds.py` → `PL/thresholds.json`). Each line comes from the three recorded seeds (s16/s17/s18)
  of that stage: higher-is-better lines are `min − max(R, range, G)`, lower-is-better and tracking lines are mirrored.
  R is the committed resolution floor and G the binomial guard.
  - **Decisive lines only.** Each is read per line. A healthy fresh seed falls below a 3-seed line about 6.7 % of
    the time, so there are few of them:
    - **The catastrophe floors**, a priori, at every stage except p3b/p3c (see Payload): translate survival ≥ 0.90,
      every translate ratio ≥ 0.70, side22 FAST/FAST, and live collapse. "Translate" means the stage's trained
      translate categories (the stamp's `trained_categories`; e.g. diag is untrained, so WATCH, at s1 and c4).
    - **A few capability lines:**
      - p2a field_hard (ANCW fails it);
      - p3a stable-at-90 and the paired 0-kg check;
      - student completion;
      - the stand lines.
    - s1–c6x HALT on the floors only, set lower than 0.90 / 0.70. Step 1 proposes each value from the recorded
      stamps with its chance rate, and you fix them at STOP-0, where any floor can also be kept or made WATCH. For
      reference, at survival ≥ 0.80 / ratio ≥ 0.50 the s1–c6x floors cost ≈ 0.75 per run, and side22 FAST/FAST at
      s1 costs .03–.06 per read. The stand lines get the same keep / relax / WATCH choice.
    - p2a, p3a and the student keep 0.90 / 0.70.
  - Every other line is WATCH and reported. That covers the s1–c6x tracking/turn lines, the track tail, p2a
    field_light and ring_obst, the Task 2 grid, turn levels, the 50 N gait set, tripod, sweep and heading drift.
  - **Self-checks:**
    - every recorded seed passes;
    - each decisive line names a known-bad that fails it (C0 arms, ANCW, P2 zero-shot, the Z/ST stand arms), or is
      tagged `unvalidated`;
    - every decisive read's chance-HALT rate, and their run sum, are reported, not enforced. With 3 recorded seeds
      no useful HALT set fits a hard bound; HALTs are prompts (On FAIL). Each read class has a stated model:
      - **3-seed lines:** the n = 3 predictive probability that a fresh seed falls below the line, simulated as a
        function of R/σ̂ with σ̂² = (range/1.693)² + (R/1.96)². That is ≈ 6.7 % for a range-set line and ≈ 1 % at
        R ≈ 2σ. The plug-in Φ((line − x̄)/σ̂) understates this about tenfold and is not used.
      - **Fixed lines** (the floors, the S1-era stand lines): the n = 3 predictive probability against the fixed
        line (t, 2 df, with the σ̂ above). Survival floors are the exception: they are binomial on the pooled
        recorded falls, because at n = 100 per category the R term overstates them. c5 has no world anchors, so it
        uses its STG_c5 body reads.
      - **Count lines** (K4, other stand counts): Poisson from the SH_T16/17/18 counts.
      - **Paired lines** (student, 0-kg check): the paired eval SE. The student's HALT needs 2 failed blocks, so it
        enters as the product of two block failures. Its rate is a lower bound, since no recorded run measures
        distillation-seed variance.
      - **Payload a-priori lines:** the step-8 pilot value as the mean. A fresh run is a second draw, so the
        variance is ≈ 2σ̂²(p2a) + SE².
      - **Live-kill reads:** shown as `unvalidated` and excluded from the sum, which STOP-0 states.
      - **The 0-kg drift margin:** `unvalidated`, since it comes from one continuation run.
  - The rates are shown at STOP-0 and re-reported at the v1 freeze (step 8). Lines are never widened or demoted
    silently.
- **Gate 0 INVALID** (rerun once): incomplete stamp, wrong pin, wrong carry env (CLOCK_HIDE, STD soft, INIT_STD 0.8,
  ellipse 0.35:0.35:0.25, facing clip), or the checkpoint does not match the stage record.
  - Payload. A level row is INVALID unless its run_meta per-env payload equals L. 0-kg, REFS and stand rows must
    carry no payload. Level files always set `KG=L:L, ZERO_FRAC=0, COM_XY=0, COM_Z=0.309`, with 0 kg as `0:0`. The
    Σ foot force check (within 3 % of (230 + L)·g) is hard only at the step-8 zero-action settle. On the level rows
    it is WATCH and feeds the limiting-factor read, since a robot tipping at its first unstable level sheds foot
    force. It never runs on field_hard (1.03–1.05 at 0 kg).
  - Golden known-bad tests read the line statistics directly (C0 and P2 stamps fail gate 0 on their carry env).
  - Gate rows pass `--trained-categories` from the stage's composed env, as `run_phases` does; `bc_eval_row.sh`
    does not today. A stamp without the list is INVALID, so null never silently drops or adds floors.
- **Walker** (s1…p2a): the floors, plus field_hard at p2a (Rule). All other walker lines are WATCH and reported,
  including the 50 N gait set (steps_per_m, swing_apex, support_feet) and the direction sweep through `body_hold`.
  The tripod score's contact test is the 1 N one.
- **Payload:**
  - p3a must be stable at 90 kg and pass its 0-kg check, or the run halts (3d).
  - **The 0-kg check (3f) is capability only, paired against the run's own p2a head** on the same rows and env
    seeds. It HALTs if survival, tracking, facing or field completion drops by more than the margin, or if
    FAST/FAST is lost.
    - Tracking means the trained translate ratios and move_turn ratio; facing means turn facing_closed and
      yaw_progress.
    - The margin is max(R, 3·SE_diff, the zero-lever continuation drift on that statistic, scaled by the head's
      iterations past p2a / 5000). The drift is X12 = P2__C0_s18, 5000 more its of the unchanged recipe, minus
      X08 = S1 s18 (one run). Continued training moves these statistics by more than eval noise; for example
      move_turn drops .048 and yaw_progress .012.
    - Each T\* candidate's notification logs how much of its margin it used.
    - X12 vs X08 is the check's known-good in the self-check; it passes by construction, so the class is
      `unvalidated` there.
    - It uses one row list wherever it is read (p3a, the step-8 pilot, T\*): suite es1–2, side22, field_hard,
      field_light, ring_obst and field_turn. p2a therefore reads all of these at es1–2.
    - WATCH: the body, grid, turn-level and sweep rows; the absolute lines; and the gait-shape lines, since carrying
      load is expected to change the gait.
  - The force side is not enforced, so every payload result is labelled "sim plant, force side not enforced".
  - "Stable at L" means survival, tracking, facing-closed, field_light completion, stance-height and orientation
    lines on suite + field_light at es1–2. The lines are a priori; S1 at 0 kg must pass them.
  - The curve runs 0–270 kg in 45-kg steps and stops at the first unstable level.
  - On p3b/p3c, a floor miss or a live kill ends that stage as a measured ceiling with its limiting factor (D7), and
    T\* is picked from the heads trained so far. It never HALTs the run.
  - Actuator load (rod force vs rated and self-lock, effort-cap share, sag) is reported only, read from the raw npz.
  - **T\*** = the trained head with the highest max_stable that passes the paired 0-kg check (its row list), tried
    in descending order. HALT only if none passes.
  - **Budget-bound.** A p3b/p3c level whose training-log survival and tracking are still rising at N_p is classed
    budget-bound (limiting-factor class 0, ahead of instability / actuator / stability / tracking). No extra eval is
    run. max_stable is reported "at N_p iterations per stage".
- **Student:**
  - Live checks:
    - `depth_actor_loss` rolling-50 at +250 ≤ 0.5× the first logged value (checked against the forward 3a
      tfevents and against the step-8 pilots; if either fails, the rule becomes WATCH);
    - episode length and crab_failure against T\*'s tail, from +250 on.
  - The prescreen ranks checkpoints, and the 90-kg level (suite es1 + field_light es1) is part of it. The first
    full-block pass is selected; the branch HALTs after 2 failed full blocks. "Not the last iteration" (4d) is met by
    this screen.
  - Pass means the floors, plus:
    - completion ≥ T\* − max(0.05, 2·SE_diff, the manifest-shape split measured at step 8). T\* is read with
      `--policy-role teacher` on Student-Play in the same manifest shape and env seeds, so the pair is real.
    - Field completion is read whole-schedule; every translate category counts (the four cameras cover back).
    - WATCH: tracking lines + 0.05, and stable at min(90, T\*'s max).
  - One headless clip goes to you for a veto, never as a gate.
- **Stand:** SN G1–G5 and SH K1–K6, re-anchored on REFS(T\*) (always re-measured). Lines derived from controls stay
  at their S1-era values; their controls run only on a failure. The exception is K4: its line is ≤ 3 misses per 800 per
  env seed (D9). The S1-era control value (≤ 1) fails SH_T16 and SH_T18. SN_Amask_s16/17/18 and SH_T16/T17/T18
  (es1, es2) join the "every recorded seed passes" self-check, the false-HALT sum and the C2 re-anchor set. SN's G2
  stillness lines (steps/s ≤ 0.5, taps/s ≤ 1) sit close to the recorded seeds (≈ 0.36 per run). **X** (the R18 stand replay) decides only on bitwise
  equality (file sha + loss curves), which step 8 tests. Otherwise X is report-only against the K lines; it never
  feeds STOP-R and never holds F19.
- **On FAIL, a HALT is a prompt:** halt the dependent subtree and notify with the line, the read and its chance
  rate. There is no automatic retry: a same-code rerun diverges after the priv_reg onset (≈ iteration 1905), so a
  rerun would be a best-of-N draw. You choose one of:
  - accept it as chance: the run continues from the halted stage and stays eligible as the deliverable, with the
    HALT and its rate disclosed in the CHARTER and the report;
  - abort;
  - amend the line in a CHARTER entry that discloses the number.
  Non-driver stages (distill, DAgger) are stopped by graceful SIGTERM, never SIGKILL.

## Steps

0. **Campaign dir** `OMNI/experiments/<date>_task4_pipeline/` (CHARTER, PLAN, DESIGN copy; isolation notice: other
   sessions stay off the code / eval / TP / PL paths from the pin until F19 ends). → sonnet · medium — transcription
1. **Thresholds v0 + self-checks** (CPU), from the recorded s16/17/18 stamps. **Camera geometry:**
   - ZED 2i front FOV/range; SEN0583 L/R/rear 0.2–2 m with FOV from the datasheet.
   - Mount poses from the chassis (placeholders flagged for M15 pose matching).
   - Raycast targets: field blocks must be in the camera's mesh set.
   - Per camera: the simulated ray grid (at the depth-encoder input resolution, not the sensors' native 320×240) and
     its update period; a 4b ground band per camera.
   - `PL/DESIGN.md` gets an annotation:
     - 3 → 4 cameras, back travel covered, K = 4;
     - the three wrong paths fixed;
     - the decisive list replaces its HALT set and its "≲ 0.04" figure; its tripod, sweep, grid and gait-shape HALT
       lines are retired;
     - the audit answers above.
   **STOP-0.** You approve thresholds v0 (the per-stage decisive lists and the false-HALT sum), the cameras and the
   budget, plus the design decisions the plan adopts:
   - D5 payload geometry;
   - D9 the K4 line (≤ 3/800 per env seed);
   - D14 the code freeze for other sessions;
   - D19 the untracked resolution file.
   → opus · high — the numbers every gate rests on
   ✓ fable · high — independent check of the threshold derivation
2. **Phase A at HEAD, no new code (≈ 1.3 GPU-h).**
   - 20-it stage smokes. P-config: the trainer-written params against the lineage run dirs. P-early: s1's first
     20 its against X43.
   - Payload feasibility probe: S1 with a fixed added mass at 0/45/90/135/180 kg (zero-action settle + suite).
   - **STOP-A** if parity or the 90-kg probe fails. → main (opus)
3. **M1a: payload knobs + reset event**, default-off. Lazy capture of the post-DR mass/CoM/inertia, partial resets,
   private generator. CPU tests; 13 knob-off dumps byte-identical. → opus · xhigh ✓ fable · high
4. **M1b: omni depth student**, default-off on every forward path. Pieces:
   - Scene with the 4 cameras.
   - Obs groups; env cfgs and the `Isaac-Crab-Hex-Omni-Student-v0 / -Play-v0` gym ids; runner (clip 1.0, soft std).
   - One shared student step used by the trainer, harness, `play.py` and `play_stream_video.py`. It never touches
     cols 6–7; it sets cols 11–12 to a constant field pair; it writes the estimator output into 207–215; it resets
     the GRU on done.
   - K-image depth backbone; ONNX export. No camera event; `STUDENT_PHASES` stays empty.
   → opus · xhigh ✓ fable · high
5. **E1: eval commit.**
   - Harness student branch via the shared step.
   - Payload/CoM recorded in run_meta.
   - Opt-in `metric_class` so sweep categories get the 50 N metrics.
   - The direction-sweep row file (16 bins at 0.22 m/s through `body_hold`).
   Existing rows byte-identical. → opus · xhigh ✓ fable · high
6. **Pipeline tooling (TP) + stand parameterisation.**
   - `pipeline.py`, `thresholds.py`, `gate_read.py`, `live_watch.py`, `distill.sh`, `select_student.py`,
     `payload_read.py`, `replay_read.py`, `report.py`.
   - SN/SH walker-spec and warm-start overrides (unset = today's behaviour).
   - Driver dump checks and whitelist for the new knobs.
   - `pipeline.lock.json` pins every out-of-hash file and the external stack.
   → opus · xhigh ✓ fable · high
7. **CPU test pass + review fixes.**
   - Golden gate reads: PASS on S1 s16/17/18, HALT on C0, FAIL on ANCW, INVALID on a flipped CLOCK_HIDE.
   - Generator byte equality.
   - Student column invariants.
   - Full `tests/unit`.
   → sonnet · high — failures are loud
8. **Phase C preflight (GPU, ≈ 20–26 h).**
   - Inertness bridges, bitwise (from-scratch smokes only; resumed stages carry the nondeterministic priv_reg term).
   - Eval bridge (C2, ≈ 1.5 GPU-h): every gate row family on S1 s18, plus SH_T17's `hwalkstop`/`hraise` es1 and
     `walkstop` rows, run through `bc_eval_row.sh` and compared bitwise with the recorded stamps. A family that
     differs is re-anchored before v1, at the pin, on the s16/17/18 heads of every stage that reads it
     (SN_Amask_s16/17/18 and SH_T16/17/18 for the stand rows).
   - **Payload:**
     - readback, a deck-CoM zero-action settle per level, and a pilot from S1 (sets N_p and calibrates the live
       collapse rule under load);
     - S1's 0–270 kg curve;
     - the full p3a gate (stable at 90 + its 0-kg check) read on the pilot's checkpoint at N_p, not its last. At the
       v1 freeze, the sum's p3a terms are recomputed from the pilot: stable-at-90 from its 90-kg values, and the
       paired 0-kg check from its paired SE.
   - **Camera and student:**
     - camera check per camera: axis error ≤ 1°, ground band;
     - teacher-on-Student-Play parity;
     - 4-camera timing (STOP if > 4 h per 1k its);
     - 2×50-it determinism;
     - the student's manifest-shape split;
     - a depth vs zeroed-depth pilot (depth must help);
     - `live_watch` replayed on the pilot logs;
     - one pilot distilled from the payload pilot head under its overlay, asserting the realised per-env mass/CoM and
       reading one 90-kg row.
   - Sweep anchors on s16/17/18 (report context only).
   - **Stand:** SN/SH smokes beside a new walker, asserting no `KRABBY_PAYLOAD_*` in the DAgger env; a 2×100-it
     same-seed SN/SH determinism smoke (decides X's rule).
   - **Full-graph dry pass:** 20–50-it stand-ins for EVERY stage kind, chained as `gen` chains them. It forces a FAIL,
     a student-branch HALT, an INVALID, a GUARD trip, and a live kill on a distill and on a DAgger stage, then resumes.
   **STOP-C** if any check fails: settle tips at ≤ 90 kg; the pilot fails the p3a gate; timing; parity; depth fails
   to help; live_watch fails on the pilots; the dry pass fails. Then freeze thresholds v1 and set the local
   annotated pin tag. → main (opus)
9. **R18 (full pipeline) + X stand replay (≈ 55–65 GPU-h).** Two separate verdicts:
   - **Process reproduced:** P-config identical; P-early ≥ 112/116 tags bitwise; P-live at +200; first divergence
     reported (expected near iteration 1905). **STOP-R** only if this fails.
   - **R18 gates:** after the divergence R18 is a second sample of the recipe, not a replay of the head files. Its
     gates are read like F19's.
   - X per the stand rule (decisive only if bitwise).
   → main (opus)
10. **F19 (official run, ≈ 46–60 GPU-h)**, seeds 19, same pin; HALTs per the gates. → main (opus)
11. **Baseline-vs-final report + keep-set + local tag.**
    - Columns: H = the v1 stand-ins (6300/9800 shown as not loadable), F = forward 3a student, T0 = S1 + SH_T17,
      T1/St = the deliverable run.
    - Contents: the full Task 0 set (50 N + 1 N with a caveat), the Task 2 grid / turn levels / sweep, the Task 3
      payload curves with max stable and limiting factor, K1–K6, wall-clock + hardware, and the grant
      reconciliation (230 kg; as-built actuators).
    - Generated `PL/README.md` with the rerun and play commands; keep-set via `bundle_experiment.py`.
    → opus · xhigh ✓ fable · high
12. **STOP-END: your decisions.** Which run is the deliverable: F19 if it is clean or every HALT was accepted as chance, else R18 on the same
    rule; the clip veto; the
    bake into `policy/` (needs `kind: distill` in `bundle_policy.py` and a test rewrite); the tag push. → you

**Stop points (push-notify each):** STOP-0, STOP-A, STOP-C, STOP-R, every HALT, STOP-END.

## Budget

≈ 130–170 GPU-h. Preflight is 21–27 h. Each run is ≈ 48–65 h:
- walker 7.7 (measured);
- payload ≈ 9–11;
- student 10–25, plus at most 2 screen blocks (the 4-camera cost is measured at step 8);
- stand ≈ 7.

X adds 5.2, plus contingency. That is ≈ 6–7 days of serial GPU time. Dev work is ≈ 8–11 days against the grant's 3; the growth comes from the payload build and the
omni depth student, which the old spec assumed already existed.

## Verification

- **CPU:** knob-off dumps and RNG bitwise; generator byte equality; overlay shas; gate-reader goldens (known-good
  passes, known-bads fail); payload mass/CoM/inertia math; student column invariants and forward call-identity;
  stand overrides unset = today; full `tests/unit` (pre-existing failures attributed).
- **GPU:** bitwise bridges (every gate row family on S1 s18 against its recorded stamp (C2), suite_body against ANC_S1, lineage heads, forward 3a
  slow row, payload armed at 0 kg against off). Plus the camera check, parity, timing and determinism at step 8, and
  R18's parity reads.
- **Isolation:** the pin guard refuses any code / eval / TP / PL / lock drift during a run; nothing is pushed.

## Later phases (not run)

- Bake the deliverable into `crab_hex_omni_task/policy/`, after your adoption.
- M15: the deploy-side yaw-rate mapper (`body_hold`), a velocity estimate for cols 13–14, camera pose matching, the
  prismatic retrain on this pipeline, and the force-side actuator model.
- Payload validation on hardware.
- A depth-student stand path (the baseline's stand nets pair with T\*).
- The walker + `body_hold` + stand-net combination evaluated together (the harness refuses it today).

Routing: strong · roster ok (opus = Opus 5.5) · planned on opus · ceiling: opus · 1× sonnet·medium, 1× sonnet·high, 1× opus·high, 5× opus·xhigh, 5× main, 6× fable·high check, fable retry on 2 fails
Effort enforcement: prompt-level
