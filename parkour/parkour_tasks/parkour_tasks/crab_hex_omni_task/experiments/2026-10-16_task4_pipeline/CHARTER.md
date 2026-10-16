# M18 Task 4 — reproducible gated pipeline on the omni task — CHARTER

- **Pre-registered:** 2026-10-16, before any run of this campaign.
- **Plan of record:** [`PLAN.md`](PLAN.md), approved by the user on 2026-10-16. The audit took 5 passes (the cap of 3
  was extended to 5 by the user) and did not converge. Blocking findings per pass ran 19 → 4 → 5 → 6 → 4. The
  pass-5 fixes and the user's last answers were not re-audited; STOP-0 is the backstop. The trail is at the top of
  PLAN.md. Its stage table, gates, steps, stop points, budget and seeds ARE the pre-registration and are not
  restated here.
- **Design detail of record:** [`DESIGN.md`](DESIGN.md), the verified synthesis, annotated where PLAN differs.
  PLAN wins.
- **Precedence:** this CHARTER is the record. Every PLAN rule change, and every number a PLAN rule leaves to be
  pinned, is logged in §3 before the result it affects is read.

## 1. Question and outcome

**Question.** Can one committed pipeline train the omni teacher from scratch through every stage? The stages are
walker W1–W6, payload P1–P3, a depth student, and the stand SN → SH. Every stage transition must be gated on
harness metrics with committed thresholds, and the pipeline must reproduce the head of record's process (R18) and
then run a fresh seed (F19) end to end. This is grant M18 Task 4.

**Outcome.**
- A pipeline spec, an entry point and committed thresholds.
- Two gated runs (R18, F19), with per-stage gate records.
- A baseline-vs-final report.
- The deliverable teacher + student + stand nets bundled with a local tag.

**Adoption, bake, tag push and push are the user's.**

## 2. Ground rules

- **Code:** shared code changes are the plan's M1a (payload), M1b (omni depth student), E1 (eval) and the TP tools.
  Every one is default-off, and the forward paths stay bit-identical. code_sha and eval_sha move only at those
  commits, each recorded with its reason. Campaign scripts live in `scripts/` here; never edit another campaign's
  scripts (copy or wrap them).
- **Isolation (D14):** from the pin (end of step 8) until F19 ends, other sessions stay off the code / eval / TP /
  PL paths and the lock's pinned files. Any uncommitted edits there must be committed or stashed before the pin,
  because the driver refuses a dirty code- or eval-bearing tree.
- **Running:**
  - One Isaac process at a time: a detached queue runner plus a log-sentinel watcher. Never `pgrep -f` a string in
    the watcher's own command line.
  - Stop with `systemctl --user stop`, or SIGTERM for the non-driver stages; never SIGKILL Kit.
  - Check the Kit DerivedDataCache before the first launch.
  - Use memory-capped scopes, and record video only through the streaming recorder.
- **Git:** commit only this campaign's paths and the shared files the plan names, with their tests. Never
  `git add -A`; never push. Other sessions share the checkout: never revert changes this campaign did not make.
- **Authority:** push-notify at every stop point (STOP-0, STOP-A, STOP-C, STOP-R, every HALT, STOP-END). HALTs
  are prompts: the user accepts one as chance (the run continues and stays eligible, disclosed), aborts, or amends
  the line. Everywhere else, decide-record-notify. Bake, adoption, tag push and push are the user's.

## 3. Log

- **2026-10-16, campaign opened.** PLAN.md approved, with its audit trail in the header. User decisions are
  recorded in PLAN's Context, under "User decisions" and "Audit answers".
- **2026-10-16, step 0.** This directory created, with CHARTER, PLAN (copy of the approved plan) and DESIGN (the
  synthesis plus a supersession header).
  - Repo state: HEAD `5d00e2d7`, code_sha `a85193431a87f091…`, eval_sha `567d17040a181542…` (the body-cmd
    campaign's final values).
  - Processes: no Isaac or driver process running (`ps`), and no user units.
  - Uncommitted files: the tree carries other sessions' edits under `2026-09-19_facing_tracking_ladder/`,
    `2026-09-23_reward_simplification/`, `2026-09-28_side_bisect/` and `2026-10-07_omni_phase2/`, plus
    `lit-review-omni-reward-simplification.md`. All sit outside code_sha and eval_sha and are not touched.
  - D19: `2026-09-19_facing_tracking_ladder/resolution.74f93b91dc8b.json` is still untracked. If it is not
    committed before the pin, `thresholds.py` recomputes it with `tools/resolution.py` into this directory.
  - Step 0 ran in the main thread, not on its sonnet·medium tag: it was a transcription, smaller than its briefing.
