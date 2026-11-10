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

### 2026-11-10 — step 1: trimmed site port (workflow `wf_bfd12127-97c`; opus·high porter, opus·xhigh reviewer)

- **Ported** from site head `2a3e3c51` by file extraction:
  - the site terrain library (`site_terrains.py`, `omni_site.py`, cfgs, generator 8-tuple and layers path);
  - `site_raster.py`;
  - a trimmed `site_task_math.py` (crossed and hazard radii only);
  - spawn pad level (`spawn_level_z`, site mode only);
  - per-kind telemetry in site mode, counting main's applied `promotion_mask`;
  - the knobs `KRABBY_SITE_MIX / DIFF / LADDER` (default-off, site geometry only);
  - 4 test files and the fixtures.
- **Excluded:** layout v2 and scan ablation, site DR, site rewards, the critic group, rsl_rl changes, camera code
  and the recorder or readout scripts. `parkour_actions.py` is untouched, so the rod-speed DR AST pins hold.
- **Loud refusals.** The 20 unported site knobs now raise loudly if set (`SITE_UNPORTED_VARS`). Step 2 removes any
  friction-contract name it reuses.
- **Neutrality (CPU).** All 232 recorded env files (rod-speed DR, DR sweep, F19) give identical cfg snapshots,
  stdout and exception types against a `git archive` of `3f944c4b`. TDR's recorded env.yaml gives an identical
  dump geometry and 23 / 23 dump checks. Only the error text for an unset geometry changes (it now lists 'site').
- **Unit suite** (same command as the baseline): 3 failed and 45 collection errors, identical node ids to main's
  baseline, with 369 more passes. The 3 failures are recorded-artifact tests; the collection errors are zmq / serial.
- **Review: READY, no blocking.** Minor items carried to step 2:
  - SITE's ladders, bands and cross_r are unchanged: pits and rubble band 2.0–3.6 m, cross_r 4.1;
  - `DEFAULT_SITE_MIX` still contains blockers: refuse a blank mix or change the default;
  - the 0.03 m roughness cap in site geometry blocks gravel's 0.05 m;
  - the telemetry's per-kind promote and demote means are not zero-filled; read `site_cnt_*`;
  - the two neutrality pins skip without `git show 3f944c4b` (CI shallow checkouts).
- **Not verifiable without Kit**, deferred to the step-5 smoke: the event manager accepting `spawn_level_z`, the
  importer attributes, and the site-mode extras under the real runner.
- **Pins.** code_sha moves with this commit (milestone boundary, as planned); eval_sha does not.
