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
