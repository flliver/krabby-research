# Stand network for the omni crab (separate from locomotion) — CHARTER

- **Pre-registered:** 2026-10-11, before any run of this campaign.
- **Plan of record:** [`PLAN.md`](PLAN.md) — approved by the user 2026-10-11 after a 3-pass audit (converged: opus·xhigh
  + fable·high on pass 3; trail at the top of PLAN.md). Its design decisions D1–D6 (architecture, training env,
  objective, code placement, switch contract, gates), steps, stop points, budget and seeds ARE the pre-registration;
  they are not restated here so the two cannot drift.
- **Precedence:** this CHARTER is the record. Any change to a PLAN.md rule is logged in §3 before the result it affects
  is read.
- **Predecessor:** stage p2r of `../2026-10-07_omni_phase2/` (closed 2026-10-11; handoff in `P2R_CLOSE.md`).

## 1. Question and outcome

**Question.** Can a stand network trained separately from omni locomotion (DAgger imitation of the scripted re-stance
controller R, S1 frozen in the loop) and routed at the latched stand slot next to the head of record S1 s18
(`policy/6_p2a_field/model_19994.pt`) pass the pre-registered closed-loop gates G1–G5 (PLAN.md D6) — STILL, homed /
settled, falls incl. exit handovers, walls unchanged, handover — at three training seeds?

**Outcome.** Per-arm and per-seed G1–G5 reads against REFS_S1_s18 / P2R_HOLD_s18 / P2R_RSTg_s18, an arm choice by the
pre-registered rule, and an adoption-grade verdict. **Adoption, bake and push are the user's.**

## 2. Ground rules

- **Code:** campaign-local only (`scripts/` here; excluded from code_sha and eval_sha). At pre-registration:
  HEAD `2ce304a4`, code_sha `14301cd2f2ba6080…`, eval_sha `1d021625844bb372…`. eval_sha must not move: never edit the
  eval harness, `gait_eval/`, scenarios files, `eval_categories.py`, `run_phases.py`. The phase-2 launcher
  `p2r_mode_eval_crab_hex_gait.py` is imported and extended, never edited. A shared-code change is a stop-and-ask.
- **Running:** one Isaac process at a time (detached queue runner + log-sentinel watcher); stop with
  `systemctl --user stop`, never SIGKILL Kit; check the Kit DerivedDataCache before the first launch; memory-capped
  scopes; no GPU rendering while the queue can launch.
- **Git:** commit only this campaign's paths; never `git add -A`; never push. Other sessions share the checkout.
- **Authority:** push-notify at every stop point (PLAN.md "Stop points"); the step-9 arm choice is decide-record-notify.

## 3. Log

- 2026-10-11, **campaign opened.** PLAN.md approved (audit trail in its header). Step 1 done (this dir, CHARTER,
  PLAN). Nothing run. Next: step 0 (lit review) and step 2 (gate calibration), on the user's go.
- 2026-10-11 ~23:00, **user go for steps 0 and 2.** Nothing else authorised (steps 3+ wait for the user).
- 2026-10-11 23:19–23:34, **step 2 GPU rows** (`scripts/sn_calib_rows.sh`, ≈ 0.25 GPU-h, one Isaac process, via the
  phase-2 `run_arm.sh --row-eval` path, guard anchored on REFH_S1_s18 = HEAD code_sha 14301cd2, eval_sha 1d021625):
  R (validated constants, `--p2r-mode restance`, STAND_OBS=clock) on S1 s18 as `SN_RSTg_s18`, field_hard es1–4.
  - **Identity: PASS.** SN_RSTg_s18 field_hard es1 is bitwise equal to P2R_RSTg_s18 es1 on every raw key, 100 / 100
    episodes, although the launcher (b5ba0c63 → 59539301), restance module (7006dff1 → 7a16b8d1) and code_sha
    (07e02578 → 14301cd2) moved since the probe. The stand banks differ only by the reservoir cap (100000 then, 20000
    now); no step of this campaign reads them.
  - R field_hard pooled n = 400: **0.825** (REFS 0.8125, HOLD 0.810, C0R 0.785).
- 2026-10-12 00:01, **step 2 calibration read: 15 / 15 expected cells land as listed, 0 mismatches, 0 pending — the
  step-2 STOP does not fire.** `scripts/sn_calib.py` (campaign-local, CPU, reuses the phase-2 readers read-only) →
  `calib.json` / `calib.md`. Built by an opus·xhigh agent, verified from three lenses (numbers recomputed from raw
  npz without the tool's code; gate logic vs D5/D6; expectations on the merits), 6 minor defects fixed, re-verified
  by fable·high (no remaining blocking defect).
  - G1: R, HOLD PASS; REFS FAIL (steps 2.83 / 2.83), ST2M FAIL (taps 16.08 / 15.33). G2: R PASS (homed 0.970 / 0.977,
    settle 2.92 / 2.96 s vs lines 3.92 / 3.96); HOLD, Z1, REFS homed 0, Z2d 0.017 / 0.023 FAIL. G3: 0 falls on every
    reference; HOLD, R, ST2M 0 / 200 exit-handover falls per seed. G4: every reference PASS (line 0.758). G5: R's
    walkstop onset (in no earlier read) **0.12 s** at es1, es2 and pooled (REFS 0.14, HOLD 0.12; line 0.44 s), 0 stalls.
  - **Caveat (recorded, not a mismatch):** G3, G4 and G5 have no known-bad reference in D6 and every reference passes
    them, so the calibration shows they compute and that R / HOLD pass, not that their lines can bind.
  - **Deviations from p2_read's rules** (each listed in calib.md): the D5 STAND_OBS=0 validity rule for REFS / HOLD /
    ST2M / C0R and for `stand_net` heads (which must also carry one stand-ckpt sha and one `stand_mask` per head — a
    field contract for the step-3 launcher: `p2r_mode.specialist.sha256`, `p2r_mode.stand_mask`); G5 adds a per-seed
    stall clause (censored onsets ≤ REFS + 1) because the onset median drops stalls; G3's exit clause audits the
    denominator (200 scheduled exits per seed, R's episode count); a head's complete stamp with no readable statistic
    FAILS (p2_read reads PENDING); Z1's suite es1 resolves to its driver stamp (lr row missing).
- 2026-10-12 00:03, **step 0 lit review done** (`../lit-review-stand-network.md`; 6 angles, 64 raw findings, 12 / 12
  claims held after one-verifier-per-claim, 20 gap findings, 84 references). **Its verdict: nothing contradicts D1–D3,
  so the step-0 STOP does not fire.** It recommends amendments that would change the pre-registered design; they are
  the user's call, not applied:
  0. a recoverability pre-check before step 7 (R taking over from student-like failure states, ≈ 0.3 GPU-h);
  2. β decayed within the first iterations (indicator or 0.5^i) and the R budget spent on discrepancy-gated takeovers
     with hysteresis, keeping the 10 % control slice;
  3. onset weighting by a last-action-only predictor's error, smoothed over (branch, |c| − eps_step);
  4. DART σ per channel from the measured closed-loop |μ − R| instead of {0, 0.1, 0.3};
  5. a branch-gated head (classifier on R's branch label + branch-masked regression) as a pre-registered third arm if
     an arm is STILL but misses G2 — the review's main disagreement with D1's flat MSE head (needs a loader adapter);
  6. exit only from R's SEAT state / post-exit crediting if G3's exit clause fails.
- **STOP (decide-record-notify): steps 0 and 2 complete; step 3 (launcher variant) waits for the user's go and their
  call on the lit-review amendments.**
- 2026-10-12, **USER DECISION at the step-0/2 stop:** adopt two lit-review amendments — the recoverability pre-check
  (A1, step 6b) and the branch-gated conditional third arm (A2) — and decline fast β, gated takeover, onset weighting
  and σ-from-residual. Go through step 6 (+ 6b); stop before step 7.
  - PLAN.md "Amendment 1" written, audited 1 pass (opus·xhigh): 6 blocking, all fixed (trail in PLAN.md); one needed
    the user's number: **control homed floor 0.75** for A1 (user decision 2026-10-12).
  - Step 9's arm rule and the stop points now route to A2 when an arm's only failing gate is G2; A1 PASS records and
    notifies, it does not launch step 7.
