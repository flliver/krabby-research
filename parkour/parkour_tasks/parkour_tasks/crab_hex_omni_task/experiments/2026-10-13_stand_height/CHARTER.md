# Height-commanded stand network for the omni crab — CHARTER

- **Pre-registered:** 2026-10-13, before any run of this campaign.
- **Plan of record:** [`PLAN.md`](PLAN.md) — approved by the user 2026-10-13 after a 3-pass audit (cap reached;
  11 → 7 → 1 blocking; the last fix — the terrain-independent restore trigger — was not re-audited, accepted by the
  user; trail at the top of PLAN.md). Its design decisions H1–H7, the control table, steps, stop points, budget and
  seeds ARE the pre-registration; they are not restated here.
- **Precedence:** this CHARTER is the record. Any change to a PLAN.md rule is logged in §3 before the result it affects
  is read.
- **Predecessor:** `../2026-10-11_stand_network/` (closed, adopted: A-mask s17 = `policy/stand_net/model_4000.pt`).
  Its scripts are reused by copy or wrap, never edited.

## 1. Question and outcome

**Question.** Can a stand network, trained by DAgger of a height-aware scripted controller `R_h` with a posture-offset
residual action, stand still and homed at commanded heights across the mechanically reachable, hip-force-capped range,
raise / lower between them, stand up from retracted starts on tables of various heights and from the ground, and
restore walking height before handing back to S1 — passing K1–K6 (PLAN H7) at three training seeds?

**Outcome.** Per-seed K1–K6 reads against the control table (R_h, R and A-mask s17 on the old path, posture map only,
S1 alone, hand-back-without-restore), forces per height, and an adoption-grade verdict. **Adoption, bake and push are
the user's.**

## 2. Ground rules

- **Code:** campaign-local only (`scripts/` here; excluded from code_sha and eval_sha). At pre-registration: HEAD
  `2f9e0a9b`, code_sha `14301cd2f2ba6080…`, eval_sha `1d021625844bb372…`. eval_sha must not move: never edit the eval
  harness, `gait_eval/`, scenarios files, `eval_categories.py`, `run_phases.py`. Never edit the predecessor campaign's
  or phase 2's scripts (copy or wrap). A shared-code change is a stop-and-ask.
- **Running:** one Isaac process at a time (detached queue runner + log-sentinel watcher; never `pgrep -f` a string in
  the watcher's own command line); stop with `systemctl --user stop`, never SIGKILL Kit; check the Kit DerivedDataCache
  before the first launch; memory-capped scopes; videos only through the streaming recorder; no GPU rendering while a
  queue can launch.
- **Git:** commit only this campaign's paths (and its `tests/unit/test_sh_*.py`); never `git add -A`; never push. Other
  sessions share the checkout.
- **Authority:** push-notify at every stop point (PLAN "Stop points"); decide-record-notify elsewhere.
- **Statistics note:** the `≤ max(1, R_h's)` fall clauses are read per env seed and per training seed; across 3 × 2
  reads a chance miss is possible — a miss is argued from the counts, not re-run.

## 3. Log

- 2026-10-13, **campaign opened.** PLAN.md approved (audit trail in its header). User decisions recorded in PLAN:
  mechanics-governed range; varied starts incl. tables of various heights and the ground; retracted = knee folded;
  restore before hand-back; force cap = six-leg mean hip-rod force ≤ 2000 N, knee hold-only; K6 homing + stillness
  per height with stillness pooled per seed; restore steps are stand-network steps. Next: step 1 (facts, CPU).
- 2026-10-13, **step 1 facts (CPU; `scripts/sh_kin.py`, `scripts/sh_facts.py`, `facts.json`, `facts.md`; opus·high,
  independently re-derived by fable·high — all numbers confirmed): STOP on two of three flags.**
  - Posture map: q(1.064) = (0, 0) exactly (tilted-pad rigid model, constant offset −0.0009 m); q monotone and smooth;
    kinematic floor 0.670 m; **h_min = 0.719 m** (hip-room term binds; q(h_min) = (−2.748, +0.371)); constant-reach
    top 1.552 m. Coupling sign unambiguous (finite differences match to 1e-10) — `stop_coupling_sign_ambiguous` false.
  - Forces (≈ 264k settled ≥ 5-feet stand rows, A-mask s17 and R walkstop es1/es2, coupled conversion): six-leg mean
    |F_hip| at walking height **2064 N measured** (worst leg up to 2580 N), |F_knee| 1161 N; the static model gives
    1785 / 1423 N. Model misses the measured joint torques by −32 % (hip), −39 % (knee) → **`stop_model_error` true**.
    With the model scaled to the measurement, h_max = 1.02 m (< walking height) → **`stop_h_max_low` true**.
  - Why they disagree: τ_hip − τ_knee = 246.8 N·m on every stamp, while the measured foot loads imply ≈ 216 N·m; the
    recorded `applied_torque` is the PD reading (16000·(target − q) − 600·v, residual ≈ 2 N·m); no leg contacts, leg
    weight reduces it, creep friction is ≈ 3 N·m. An unexplained ≈ 14 % excess in the sim's joint torque.
  - Unscaled curves: exploration model h_max 1.22 m; leg-weight model 1.342 m; measured-foot-load statics 1.346 m.
  - Tables: top_min 0.412 m (knee_folded toes clear the floor), training top_max 0.756 m at h_max 1.02 (0.805 m by the
    restore clearance otherwise); K3 tops 0.4572 / 0.635 / 0.6858 m (provisional). Ground: knee_folded rests on its toes
    (root 0.561 m), rod_min on the hip plates (root 0.311 m, hip-contact clause fires at spawn); 81 % of random folds
    start below h_min (q and limiter clamp).
  - **STOP and ask** (step-1 stop rule).
- 2026-10-13, **USER DECISION at the step-1 stop: cap from the measured-foot-load statics.** h_max = **1.346 m**
  (highest h whose six-leg mean |F_hip| from the measured foot loads through the linkage stays ≤ 2000 N); the sim's
  PD-torque reading (2064 N at walking height, the unexplained ≈ 14 % excess) is reported alongside every height as an
  open item, not used for the cap. Frozen (`facts.json` `frozen`): h_min 0.719 m, h_max 1.346 m, training table tops
  0.412–0.805 m (restore clearance binds), K3 eval tops 0.4572 / 0.635 / 0.762 m with h_cmd ≥ top + 0.31 m.
  - **Rule restated before its read (follows from the decision):** the step-4 h_max re-check uses the same statistic —
    six-leg mean |F_hip| from R_h's measured foot loads through the linkage in its top height bin — not the PD torques.
- 2026-10-13, **steps 2–3 build + review (CPU; workflow wf_27292acb-ff1).** Core `scripts/sh_core.py` (posture map,
  R_h, limiter, restore controller, router, stand-net input, zeroed-W warm start), then four parallel builds: eval
  launcher (`sh_eval_crab_hex_gait.py`, `sh_eval_common.py`, `sh_eval_rows.sh`, `sh_bridge_read.py`,
  `scenarios_sh.yaml`), DAgger trainer (`sh_dagger*.py/.sh`, `sh_starts.py`), table/ground test (`sh_table_*`),
  readers (`sh_read.py` K1–K6, `sh_observe_read.py`). Reviews opus·xhigh + fable·high: 2 blocking + 11 minor; fix pass;
  fable re-review clean (one minor residual, applied by me: the step-5 smoke also gates the R_h control slice's own
  σ0 restore-timeout share ≤ 0.05, ≤ 1 timeout when < 20 ended). 227 CPU tests pass
  (`tests/unit/test_sh_*.py`); code_sha / eval_sha unchanged (14301cd2 / 1d021625, no dirty paths).
  **Implementation statistics pinned before any read (logged here per §Precedence):**
  - **Restore completion rod tolerance δ = 4 mm** (PLAN's 2 mm example is below the PD sag); re-checked on R_h's
    step-4 probe (restore-timeout share near 0).
  - **K3 / train-time STOOD rise clause** = the ROOT rise above its spawn height ≥ 0.10 m on ≥ 95 % of the last-3-s
    window (was: chassis lowest corner on every step — biased by tilt and floor relief at the top + 0.31 command
    floor). Rise-only failures counted per cell. Hip-contact re-arm stays on the corner rise (trainer and K3 agree).
  - **Estimator off-height inputs: accepted and reported.** On stand rows away from 1.064 the frozen estimator sees
    q + residual beyond ±1 in its last-action block (and joint positions out of its walking range anyway); a ±1 clamp
    of the block alone would not restore its distribution and would split the shared input function. Estimate ranges
    are reported in the smoke and run_meta.
  - **Step-5 restore assertion** gates the R_h-executed slices only (control σ0 completes, absolute share above;
    β-R σ0 within the control line); the warm-start student's share is reported, gated at the step-6 premise.
  - **Ground first command** bounded below at max(h_min, unclamped spawn height + 0.146 m) (the table test's rule) so
    the "up" latch can fire on high random-fold spawns; never-up episodes counted. Table commands keep top + 0.264.
  - **Ground spawn height** = the highest floor under the pose's 23 probe points (table-test rule).
  - **h_max re-check:** the pre-registered top-bin [1.221, 1.346] foot-load statistic decides; the same statistic on
    rows within 2 cm of 1.346 is reported and flagged (the bin mean sits near 1.29 m).
  - **K4 completion** is decided by the launcher's recorded completion / timeout events; the reader's joint-space
    re-derivation is a cross-check, a disagreement beyond ±1e-6 m marks the stamp INVALID.
  - **K5 rows:** the rows script knows `field_turn__omni` on the shared manifest (zero command) so all nine K5 stamps
    come from the new launcher; `scenarios_sh.yaml` unchanged for it.
  - Table-test jobs: no `timeout -k` (no SIGKILL of Kit); the rows script waits on any Isaac process.
  - **Deferred:** `sh_record_video.py` (step-8 raise/lower video) — its own build + review + GPU check before step 8.
  GPU checks left for step 4: launcher/table/ground smoke, identity bridge bitwise with the restore armed, δ on the
  R_h probe, h_meas vs raycast, trainer↔launcher parity, h_max re-check, the K4 cross-check on real stamps.
- 2026-10-13, **step 4 launcher smoke (R_h hwalkstop / hraise es1; GPU) → restore band δ re-pinned 4 → 8 mm before
  any K4 read (decide-record-notify).** Every dose realised (each scheduled height arrives inside its hold,
  1000 / 1000 on hraise; hip targets beyond ±1 on stand rows; R_h branches re-derive; h_meas at settled 1.064 stands
  1.062 / 1.061 m). But **119 / 600 (hwalkstop) and 36 / 200 (hraise) restores timed out at δ = 4 mm**, every one
  settled in HOLD with one loaded leg at 4.1–4.9 mm of hip-rod PD sag — not motion. The 4-mm band sits inside R's own
  normal sag: on the predecessor's settled walking-height walkstop stands R (P2R_RSTg_s18) exceeds 4 mm on 20 % (es1)
  / 16 % (es2) of holds (best 0.2-s window, max-leg; p99 7.16 / 6.75 mm, max 8.02 mm); A-mask s17 p99 3.6 / 3.9 mm.
  The rod reading also cannot separate a lifting leg from a planted one (R_h LIFT rows p50 2.9 mm; HOME p50 14.6,
  SEAT p50 9.3 mm), so the band's job is "posture at the walking pose, no HOME / SEAT in progress".
  - **Rule (set from R's data, not R_h's restores):** δ = the smallest whole mm ≥ the p99 of R's settled walking-height
    stands' max-leg hip-rod deviation (es1 7.16 mm) → **δ = 8 mm** (`sh_core.RESTORE_ROD_TOL_M = 0.008`; tests
    updated; 224 pass + 3 skipped only because an Isaac process was running).
  - Offline replay of the completion rule on the recorded smokes reproduces the record exactly at 4 mm (481 / 119;
    164 / 36) and gives 600 / 600 and 200 / 200 completions at 8 mm (median 2.9 s / 4.0 s).
  - Consequence: every stamp recorded at 4 mm reads INVALID (controller record ≠ sh_core) — R_h's es1 smoke rows are
    re-run in the probe. SH_AMq hwalkstop es1 started after the edit (8 mm).
- 2026-10-13, **step 4 identity bridge: BITWISE.** S1 as the stand checkpoint, mask 0, zero command, the new 4.8
  clip path with S1 rows clamped ±1, restore ARMED (δ 8 mm): `sh_bridge_read.py --head SH_ID_s18 --identity` exit 0 —
  suite / walkstop / field_hard es1 every raw key of REFS_S1_s18 identical (max |Δa| = 0; 0 restores engaged; walkstop
  pre-entry actions checked on 100 episodes). K5 reuses the existing R / A-mask stamps as pre-registered.
- 2026-10-13, **step 4 calibration read (GPU ≈ 2.7 h; `step4_read.md/.json`, `step4_k3_summary.json`): control table
  14 / 14 land, 0 mismatches; h_max holds; STOP — R_h's own stand-up floor misses (step-4 premise rule).**
  - Height rows (es1 + es2, 100 episodes each): R_h PASSES K1 (median |h − h_cmd| 1.5–3 mm per bin), K2 (every
    command reached > 1.1 s inside its allowance, 0 falls), K4 (1598 / 1600 restores complete, 1 miss per seed, 0 exit
    falls), K5 (G1–G5, existing R stamps), K6 (homed / settled 93–100 % per bin; stillness 0.17 steps + 0.15 taps per
    s). Known-bads fail on their named clauses: R and A-mask on the old ±1 path K1 / K2 (median error up to 0.29 m,
    raises never reached); no-restore hand-back K4 (788 / 784 misses); posture-only K6 (homed 0). Reported: posture
    only passes K1 / K2; A-mask s17 + q fails K1 / K2 / K6 and falls in stands at 1.15–1.35 m (untrained for height).
  - **h_max re-check (pre-registered top-bin statistic): 1912 N ≤ 2000 N** — no re-freeze (within 2 cm of 1.346 m:
    1994 N, reported). PD reading 2457 N (the open ≈ 14 % excess).
  - **K3 control table lands** (S1 fails; R_h the reference), but **R_h's floor (STOOD ≥ 0.90, falls ≤ 1 / 100 per
    top / ground pose) MISSES**: 18-in table 98 / 300 and 87 / 300 STOOD (knee_folded 6 / 12 %); 25-in table STOOD
    93 / 97 % but 9 / 8 falls per 300 (rod_min 6 / 100 each seed, hip-contact clause; knee_folded 95 / 99 %, 0 falls);
    30-in table 100 / 97 %, 0 / 2 falls (passes); ground knee_folded 0 / 100 STOOD with 31 / 33 falls; ground rod_min
    17 / 21 %; ground random 58 / 68 %. **R_h ≡ posture-only on every K3 cell** (the re-stance never acts here).
  - **Cause (traces, verified against the linkage's joint-space targets):** hips track their targets (≈ 0.01 rad), but
    a LOADED folded knee does not extend — on the ground it sits 0.42 rad (≈ 16 mm of knee rod) short of its target for
    the whole 16 s (PD demand ≈ 2500 N·m vs the knee's 400 N·m effort cap), the robot pinned 13 cm low; on the 18-in
    table 0.15–0.42 rad short, 6–9 cm low. On the 25-in table the knees reach target in the air (≤ 2 mm rod error at
    1 s) before the toes touch at ≈ 2 s. Geometry (`sh_kin.toe_uz`): at the −45° hip stop (the retracted pose) an
    unfold drops the toe by up to 12 cm, so any top below ≈ 0.53 m loads the toes mid-unfold (the step-1 top_min 0.412
    only checked clearance at spawn); from the ground no hip angle lets the legs unfold in the air. A ground get-up
    needs a new motion (e.g. belly down, toes slid out under light load, hip push-up) whose hip forces vs the 2000 N
    cap are unknown. On hardware the knee is hold-only and rated far below this load, so the limit is not a sim
    artefact to tune away.
  - **STOP and ask** (push-notified). Nothing trained; budget used ≈ 3.0 GPU-h of the ≈ 15.
- 2026-10-13, **USER DECISION at the step-4 stop: narrow stand-up to tall tables.** Ground get-up becomes its own
  follow-up (it needs a scripted get-up motion and a hip-force check); this campaign trains and gates table starts
  where the knees unfold in the air. **Rules (set before any re-read or training):**
  - **Lowest table top = 0.635 m** (25 in), not the ≈ 0.55 m estimated at the stop: in R_h's 25-in trace the toes
    clear the floor by only 1–2 cm when the knees reach their targets (the hip slews down while the knees unfold, which
    the hip-stop-only estimate ignored), and the 18-in table fails. `facts.json` `frozen.top_min_m` 0.412 → 0.635;
    training tops U[0.635, 0.805] (top_max unchanged).
  - **K3 gated cells** = tops {0.635, 0.6985 (27.5 in, new), 0.762} × poses {knee_folded, random}; the R_h floor and
    the student's K3 line read tables pooled over these six cells per env seed (per top for the floor). **Reported
    only** (run, read, never gated): the 18-in table, every rod_min cell, every ground cell.
  - **Training starts:** ground starts removed; their 15 % goes to table starts (walk 0.50 / table 0.40 / ground 0 /
    random-height 0.10), so the stand-up dose is unchanged. Retracted poses on table starts: knee_folded 0.75, random
    fold 0.25, rod_min 0 (the plan's 0.60 / 0.20 / 0.20 renormalised without rod_min).
  - **R_h floor at 0.635 (knee_folded + random, existing stamps):** STOOD 95.5 % / 98.5 %, falls 3 / 200 (es1) and
    2 / 200 (es2). The es1 count is one over the 1 / 100 line. All five events are the hip-contact clause (> 500 N)
    with the robot UPRIGHT (tilt 2.5–3.3°) and still standing at the end, 0.3–5.8 cm from its command: hip-body
    contacts (source not identified; the table is the only object near), not tip-overs. Such events occur at every
    command margin (up to +0.69 m above the top). **Accepted and argued from the counts (CHARTER §2 statistics note), not
    re-run;** the student's K3 fall line is max(1, R_h's), so it is held to the same contacts.
  - Next: build + review of the re-scope (facts, `sh_table_core` cells / floor / pooling, `sh_read` K3, trainer start
    and pose shares, smoke checks, `sh_table_test.sh` surfaces, tests), then re-run the K3 calibration on the new
    surface list (R_h, S1, posture only, A-mask; es1 + es2), then step 5.
- 2026-10-13, **re-scope build + review (CPU; workflow wf_502100f0-7bd: opus·xhigh build, fable·high review —
  APPROVE, no blocking defect; 4 minor fixes applied).** `facts.json` frozen (old values under
  `previous_frozen_step4`); `sh_facts.py` now keeps an existing FROZEN block on a re-run (it would have overwritten
  both user decisions); `sh_table_core` 5 surfaces × 3 poses = 15 cells, `K3_GATED_CELLS` = the 6, the floor per gated
  top (INCOMPLETE if a gated cell is missing), reported cells never decide a line / floor / STOP (a contract problem in
  a reported cell is a note, not INVALID); `sh_read` K3 PENDING unless the 6 gated cells have n ≥ 100 per seed; trainer
  shares as pinned with 0-share kinds / poses asserted ABSENT in the smoke and skipped by the premise read. 236 CPU
  tests pass; code_sha / eval_sha unchanged.
  - **Floor read rule on the K3 re-run (pinned before it runs):** the R_h floor is read as coded (STOOD ≥ 0.90,
    falls ≤ 1 / 100 per gated top). A fall-clause miss made only of hip-contact events with the robot upright
    (tilt < 0.5 rad throughout) and standing at the end — the kind accepted above — is accepted under the same
    argument and recorded with its counts. Any tilt fall, any STOOD < 0.90 on a gated top, or a miss of any other kind
    → STOP and ask.
- 2026-10-13, **step 4b: K3 re-calibration on the re-scoped cells (GPU ≈ 0.35 h; stamp `sh_table/2026-10-13_21-59-02_full`;
  `step4b_read.md/.json`, `step4b_k3_summary.json`): control table 14 / 14 land, 0 mismatches.**
  - Gated tables (6 cells × 100 envs per seed): **R_h 590 / 600 and 587 / 600 STOOD, falls 0.83 / 0.33 per 100**
    (S1 0 / 600, falls 3.7 / 3.2 → FAIL as required; posture only 597 / 587, reported PASS; A-mask s17 13 / 16 %,
    reported FAIL). Reported cells as at step 4 (ground knee_folded 0 %, ground random 68 %, 18-in table low).
  - **R_h floor per gated top:** es2 all pass; es1 0.6985 (198 / 200, 1 fall) and 0.762 (199 / 200, 0) pass;
    **es1 0.635: 193 / 200 STOOD, 4 fall-clause events (line 2) → read as coded = MISS.** All four are random-fold
    starts, hip-contact only (peak hip force 568–11 932 N), maximum tilt 2.6–4.3° throughout, standing at the end
    0.4–5.9 cm from the command → **accepted under the pinned floor-read rule** (no tilt fall, STOOD ≥ 0.90 on every
    gated top). `sh_read` exits 4 on this alone.
  - Step 4 closes: R_h is the teacher for height, raise / lower, restore and the gated table starts. Next: step 5
    smoke.
