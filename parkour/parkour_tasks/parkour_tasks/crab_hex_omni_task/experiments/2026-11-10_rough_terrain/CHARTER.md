# CHARTER — traversable rough-terrain teacher (2026-11-10)

Decision log for `PLAN.md` (audited: 3 passes plus one extra pass the user approved; the pass-4 fixes were not
re-audited). Literature review: `../lit-review-rough-terrain.md` (12 / 12 claims verified, 111 references, fable
final check applied). Predecessors:
- `2026-10-28_site_teacher`: closed not accepted, on the unmerged branch `worktree-sen0581-cameras`;
- `2026-11-07_rod_speed_dr`: TDR baked as teacher of record, merged as PR #13.

## 1. Settled decisions (user, 2026-11-10)

1. No walls, blockers or obstructions that require stopping. The terrain is slopes, hills, ledges and steps,
   potholes, gravel and rubble.
2. The literature review informs the design.
3. The DR on main (rod-speed DR, TDR as teacher of record) is the base and is kept in training.
4. G is judged per kind:
   - k = 2 gated kinds must gain, and none may regress;
   - bands above the blind envelope are report-only;
   - bands are chosen on es3 and G is read on es1–2.
5. Set aside to P3: the scan-ablation knob port, the `_flat` twins, the site recorder port and the new teacher's
   band-above rows.
6. After the pass-3 cap:
   - one more audit pass;
   - the plan size is kept;
   - the iteration-1500 early read is one log line;
   - TDR's three above-envelope baseline rows are kept.
7. Plan approved, execution started: "Approve, start step 0".

## 2. Standing rules

- Decide, record, notify. Stop and ask only at the pre-registered points: exactly one gated kind (step 7), and the
  HALT rows of the decision table. Push-notify at each of them.
- Only wall-clock time counts (local GPU): never idle the GPU on a pending decision.
- Pause for push, delete or bake; never push.
- One Isaac process at a time, through a detached queue and a log-sentinel watcher. Never `pgrep -f` a string in
  the watcher's own command line. Use memory caps; no RecordVideo.
- Workflow subagents never write report files (the parent writes them). No downloads or installs in any brief.
- Other sessions share these checkouts: never revert unexplained changes.

## 3. Log

### 2026-11-10 15:38 — step 0: worktree and campaign dir

- **Branch.** `feat/rough-terrain`, worktree `.claude/worktrees/rough-terrain`, cut from `origin/main` `3f944c4b`
  (the PR #13 merge: TDR as policy of record, rod-speed DR on main).
- **origin/main.** It has since moved to `b663af7d`, which adds STM leg-board firmware only (no `parkour/` or
  training-test change). The base stays `3f944c4b`, as the plan states.
- **Filed.** `../lit-review-rough-terrain.md`, `PLAN.md` (the audited plan) and this CHARTER.
- **Prior reviews.** The review cites `lit-review-construction-terrain-dr.md` and `lit-review-obstacle-avoidance.md`,
  which live on the unmerged site branch, not on this one.
