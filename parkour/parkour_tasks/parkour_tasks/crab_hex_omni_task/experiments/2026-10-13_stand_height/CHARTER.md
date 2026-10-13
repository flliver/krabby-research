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
