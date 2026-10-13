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
