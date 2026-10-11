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
