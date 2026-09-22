<!-- The running log of deviations/ratifications relative to PLAN_2026-09-18_facing_tracking_ladder.md, kept during implementation
     (2026-09-18/19); every item is also folded into CHARTER.md's Caveats. Saved with the plan on 2026-09-19. -->

# Plan amendments discovered during implementation (fold into $LAD/CHARTER.md "Caveats"/"Log" at step 6)
1. Twin ruling (A1): a pool twin of a legacy window is bitwise on slot_id/speed_cmd/heading_target/vel_command_b for every row
   and on travel_dir_w for MOVING rows only (non-moving rows hold yaw + 0; unobservable). Bitwise twins also require the pool
   written in POOL_TYPES order.
2. `pool_<name>_draws` is a LIFETIME running fraction (overwritten each step, never reset), not "per resample"; `pool_<name>_frac`
   is step-weighted. Read `_draws` ≈ weights only on a fresh run.
3. speed_scale bound: a construction-time refusal keeps every scaled draw above the gait-clock stop threshold (0.2 m/s):
   `speed_scale > pool_speed_scale_floor(...)` — 0.88 for world/lateral/backward types, 0.6286 for fwd at the shipped ellipse.
   No candidate uses speed_scale. Compose-time check in the ladder validator.
4. The clock rule refuses CLOCK_TURN_R = 0 as well as radius × cap < 0.25 (inert on every preset: 1.2 × 0.3 = 0.36).
5. KRABBY_OMNI_FACING_OBS_ENCODING applies through omni_knobs._omni_obs_terms(cfg); omni_knobs.parse_cmd_pool(blank) → None.
6. The cfg-identity `dumps` fixture does not create KRABBY_CFG_DUMP_DIR — pre-create it (step 0 lesson).
7. The unit-suite command needs `--continue-on-collection-errors` (32 pre-existing collection errors in unrelated trees).
8. README (Part D) must document the six new knobs, cmd_pool, and CLOCK_TURN_R "> 0" (was ">= 0").
9. A4's "the ONLY deliberate behaviour change is the eval timeout's SIGTERM" is amended (2026-09-18 review).
   `_stop_child` (SIGTERM -> TERM_GRACE_S -> SIGKILL) is used on EVERY path that stops a live child, i.e. also
   train's stall / nan / dead_plant / dead_walker / stop / exception exits, which previously did
   `proc.kill(); proc.wait(60)`. Cost: up to +120 s per such death. Reasons: (a) those four exits kill a running
   Kit exactly like the timeout does (the DerivedDataCache hazard); (b) HEAD's bare `proc.wait(timeout=60)`
   raising TimeoutExpired fell into the generic handler and reclassified a `nan`/`dead_plant` death as `infra`,
   firing a spurious `_r2` relaunch. Reverting would reintroduce both.
   Knock-on (2026-09-18 second review, unverifiable without Isaac): on the `dead_walker` exit the head is
   resolved by `latest_run_ckpt(...)` AFTER `_stop_child`, i.e. after up to `TERM_GRACE_S` of SIGTERM grace,
   where HEAD resolved it immediately after `kill()`. If Kit's SIGTERM handling lets the trainer finish one
   more checkpoint save inside that window, the head `--continue` carries forward unretrained is a LATER
   checkpoint than HEAD would have filed. Bounded (one save) and probably unreachable; watch the first real
   dead_walker abort.
   `train_retry`'s retry pause is ticked for the same budget reason as `_run_eval`'s (below).
10. THREE further timing deltas in `_run_eval`, all deliberate: the 3600/5400-s timeout is enforced at tick
   granularity (worst case `timeout + EVAL_POLL_S + TERM_GRACE_S` per attempt, not exactly `timeout`); EVERY
   attempt now costs at least one `EVAL_POLL_S` tick, because the poll loop sleeps before re-testing `poll()`
   where `subprocess.run` returned at once (measured: a success path that took 2 s takes ~30 s -- ~30 s per
   eval row per campaign, accepted); and the `RETRY_WAIT_S` pause is ticked (a stop inside it ends the call
   `interrupted`) and is SKIPPED after the last attempt -- a `max_attempts=1` call (Part B's shape) no longer
   burns 120 s of dead time before the driver's infra detector runs.
11. `facing_closed_by_hold` has NO `heading_error_max_abs` fallback. An aggregate with no
   `heading_error_initial_abs` returns `None` for every hold and `facing_closed_source is None`. Measured on the
   recorded 1a slow row, the max-derived proxy sign-flips lat_l/diag/back (+0.30/+0.28/+0.29 vs the per-episode
   metric's -0.51/-0.39/-0.40) and overstates turn_r90 threefold, because `max > initial` whenever the body
   first turns the wrong way. Consequence for step 5: EVERY already-recorded row now yields
   `facing_closed_by_hold = {h: None}`; a milestone that needs the quantity must re-score the run_dir from its
   raws (`--save-raw` rows only). `None` is recoverable; a wrong-signed number is not.
12. `train` sets `PYTHONUNBUFFERED=1` in EVERY training child's environment, on the default path (HEAD's env
   was `{KRABBY_OMNI_PHASE, KRABBY_PLANT}` and nothing else). Reason: the poll loop reads the child's stdout
   FILE, so a block-buffered child looks stalled for as long as its buffer takes to fill -- which `STALL_S`
   calls an infra death and which `dead_walker` reads as "no track_ratio prints yet". Verdicts are unchanged
   (measured old-vs-new on ok / nan / no-checkpoint), but it DOES change the interleaving of the training log
   that the stall watchdog, the `Traceback` scan and `dead_walker` all read, and the `env {...}` line written
   into the campaign log and CHANGELOG -- so a later reader can tell a line-ordering difference caused by
   this from one caused by a code change. It is set AFTER the `extra_env` merge and is therefore not an
   overlay's to change (it is deliberately NOT in `TRAIN_ENV_RESERVED`, which is about records that lie, not
   about the watchdog).
13. Additive record changes on the default path, for step 5's reader: `state["current"]` gains
   `"extra_env": {}`, and `eval_morph` / `eval_obst`'s returned dicts gain `eval_status` / `eval_rc`, which
   `run_phase` files straight into `rec["evals"][<row>]`. Nothing enumerates an eval row's key set and
   `interrupted_phase` reads `current` only through `.get`, so both are inert -- but the two eval keys are
   the driver's, not the ladder's. `expo_lines`' terrain line also gained `(move rows only <x>)`, which
   re-renders an OLD campaign's line differently (as `n/a`); REPORT.md files already on disk are untouched.
9.  (step 3) `_stop_child` (SIGTERM → TERM_GRACE_S 120 s → SIGKILL) replaces `proc.kill()` on EVERY stop path of train/_run_eval
    (stall, nan, dead_plant, dead_walker, exception), not only the eval timeout; a dead_walker abort therefore hands back a head
    up to one grace later than before (watch the first real abort). Ctrl-C/SystemExit on an attached driver now stops the child
    explicitly (kept deliberately: an orphaned 256-env Kit holding the GPU is the worse failure).
    (2026-09-19, implemented) That Ctrl-C keep is now EXPLICIT: `train`'s handler is spelled `except (KeyboardInterrupt,
    SystemExit): _stop_child(proc); raise` instead of `except BaseException`, sitting beside the unchanged `except Exception`
    infra path, with the decision and its blast radius (attached terminal only — under systemd-run a SIGTERM to the driver
    raises no Python exception, so the stop arrives via `interrupt_check`) written at the handler and in train's docstring.
    Pinned by `test_train_ctrl_c_stops_the_training_child_before_it_propagates` and
    `test_train_system_exit_in_the_poll_loop_also_stops_the_child` (both fail if the handler is removed).
10. (step 3) `_run_eval` timing deltas beyond the named SIGTERM-first change: 30-s poll ticks, an interruptible retry wait, and
    (restored after review) `_wait_isaac_clear` still runs after the last failed attempt.
    (2026-09-19, implemented) Exact restored shape: `_wait_isaac_clear(interrupt_check)` runs after EVERY failed attempt, the
    LAST one included, and only the `RETRY_WAIT_S` `_interruptible_sleep` is skipped there — what waits on it is the NEXT ROW
    (slow → light → ring run back to back in `run_phase` and `_run_eval` never waits BEFORE an attempt, so three timeouts
    ending in SIGTERM + up to `TERM_GRACE_S` of grace would otherwise overlap the next row's Popen on the one GPU). The
    interrupted path is unchanged (no wait, no retry). Pinned by
    `test_run_eval_waits_for_isaac_after_every_failed_attempt_including_the_last` (3 calls for max_attempts=3, 1 for
    max_attempts=1) and `test_run_eval_does_not_wait_for_isaac_once_the_stop_has_fired` (0 calls either side of the launch).
11. (step 3) `facing_closed` for already-recorded rows is available only by re-scoring raw/*.npz; the aggregate JSON of a
    pre-step-3 eval yields None. `facing_closed_source` ∈ {metric, proxy, mixed, None}; proxy rows are never pooled against
    metric rows. Orchestrator decision: resolution.py re-scores every input row (ring included) from raws — the plan's "ring
    proxy on both sides" is superseded by the real metric.
12. (step 3) PYTHONUNBUFFERED=1 on every training child (set after the overlay merge) — the stall watchdog and the Traceback/
    dead_walker scans read the log live; verdicts unchanged.
13. (step 3) Additive record keys: `current.extra_env`, eval rows' `eval_status`/`eval_rc`, `expo_lines` prints
    track_ratio_move_mean (n/a when absent).
14. (step 2) Under `sincos` the facing_obs_clip is ignored (sin/cos are bounded by construction); `facing_sigma` is keyword-only
    with no default (Isaac's static argument check); `reward_facing_turn` carries `parkour_name` (ignored) so its dump block
    mirrors `reward_tracking_yaw`; the turn-slot gate is `slot_id >= SLOT_TURN`, not `cmd_wz != 0`; the obs param is declared
    in crab_hex_omni_env_cfg.py (load-bearing for every eval row now that EVAL_CARRY_DEFAULTS injects the encoding).
15. (step 4) `move_turn_r` (0.22, dir 0, face 0) is a right-handed counterpart, not a reflection of `move_turn`: its ellipse
    cap is the fore-aft 0.35 vs `move_turn`'s lateral 0.25, and 90° inside 8 s never closes — WATCH only. The replay's
    `--progress-w` alias arms the turn-hold table when supplied (post-review fix); a segment_id guard refuses raws that are not
    slow__omni's; `--co` is now an ambiguous abbreviation (no committed invocation abbreviates).
- Process floor pair (2026-09-19): parkour/logs/rsl_rl/gait_eval/omni_phases/campaign/seed3_2b_006_A15pB_floor/slow__omni/seed001/2026-09-19_06-17-18 vs the recorded 2b slow row (completion 0.50 both; tripod_dir 0.377 vs 0.38; yaw_progress_turn_min 0.0715 vs 0.071).
16. (step 5, RATIFIED by the orchestrator 2026-09-19) Ring completion at milestones follows decision 6's normative sentence
    — "WATCH unless the bootstrap resolves it": D00 resolves it (by_row ring__omni completion R_eval = 0.165), so from M2 the
    ring row's completion IS scored as a guard (a drop > 0.165 makes the ring row WORSE). Part C (ii)'s flat "ring completion
    is WATCH" cites decision 6 and is read as its conditional. The MILESTONE line names the branch taken
    (`ring_completion: scored (...)`). Both branches are pinned by tests.
17. (step 5) `turn_survival_r90`'s cross-head R_eval (0.028) is degenerate — the 2b and 2a-reeval heads sit at exactly 1.0 on
    it, so the paired difference carries only 1c-004's variation and the guard does not hold its nominal 99 % (null TIE
    ≈ 0.945; all-TIE 0.851–0.859 vs the 0.85 floor). No code change: the D00b seed term (B0's pair, window 2) is expected to
    widen it; the RESOLUTION line lists "guards still carried by the EVAL half-width" — a lone WORSE on that guard while it
    is so listed is to be read as uncertain and confirmed by the user, not as a clear REJECT.
18. (step 5) For C3a/C3b the absolute-units T block (achieved_speed) is mathematically identical to the relative T block
    (ratio = achieved / a manifest-constant command): identical z_T and R (rel. diff 1e-8). Only F_abs (net_yaw_rate_toward
    vs yaw_progress_rate) differs (R_F 0.550 vs R_F_abs 0.578). Faithful to the plan; not a bug.
19. (step 5) The D00 process floor is 0.0 on every metric by construction: the harness is bitwise deterministic at a fixed
    env_seed (raw/episode_00.npz md5-identical between the recorded 2b eval and the same-head re-eval; run_meta differs only
    by started_utc and the two defaulted carry keys). Recorded as a fact about the harness; no further eval — episode-sampling
    noise is what the independent-resample construction already measures.
20. (step 5, post-review fix) A user `decide <Cid> --choose skip` must never squash or interrupt the BASE recipe's twin booked
    for Cid's pair (kind `twin` exempt from derived-view cause (d) and from interrupt_reason's skip clause).
21. (step 5, post-review fix) `rp.gate_1c_verdict` / `rp.gate_facing_verdict` are printed as WATCH lines in the ladder's
    stage report and DECISION/MILESTONE fields (final chain stage; 2a/2b promotion stages) — context only, never a verdict.
22. (V3) Legacy twin smokes: 1c bitwise; 1a bitwise except one run-to-run nondeterministic priv_reg print digit (0.3693/0.3694), shown to be noise by a second run. 37 reward labels under the C4/C5/C6 overlay; env.yaml records the swapped func + facing_sigma + sincos.
23. (step 7 review, RATIFIED 2026-09-19) decide_pooled evaluates the two-seed sign DISAGREEMENT (→ disputed → incumbent) BEFORE the
    MISSING-member `uncertain` cap, in the plan's stated order (pooled clear → disagree → speculative_default → sign).
24. (step 7 review, RATIFIED) `overruled → confirmed` by the user is legal only for `overruled(driver|milestone)` entries (the plan's
    table); an `overruled(user)` entry takes a new `overrule`, never `confirm`.
25. (step 7 review, recorded) The driver traps SIGINT like SIGTERM (an attached Ctrl-C stops the child cleanly and cannot abort the
    driver mid-write); `CODE_DROP_GLOBS` carries a fourth pattern `tests/*` (inert on this tree); a single-seed decision whose
    candidate and base sit on different chains widens with max(D00c, D00b) (conservative); RETENTION is silent on a chain stage's
    first run (nothing to compare against — the trace it seeds is compared from the second run on).
26. (step 7 review) Milestone entries are not confirm/overrulable (their verb is `resume-milestone --accept|--drop`); `--accept` is
    refused unless the milestone's newest entry is `await`; resurrection after a revert is transitive (the squash view is recomputed
    to a fixpoint and reconciled with state at every launch); the drift anchor requires a CLEAR pooled level when a pooled level
    exists; the validator applies `ang_vel_max > clip` with the default clip 0.2 when no clip is overlaid; the 2a/2b pool is
    rescaled from the COMPOSED final-stage pool; `_wait_isaac_clear` precedes every train/eval launch; the replay child has a
    timeout and an exception guard; pooled DECISION lines cite levels and read `s3 + s2`.
27. (re-verify of the fix batch, RATIFIED 2026-09-19) Pool strings composed by the driver are rendered CANONICALLY: every weight and
    scale is rounded to 12 significant digits before repr(), so a pool that went through parse → normalise → rescale renders exactly
    like its yaml literal (`fwd:0.14`, never `fwd:0.14000000000000004`) and recipe_id — sha256 over the rendered strings — is stable
    under float noise. Still repr-precision, never fixed decimals; the projection/sum round-trip pins (1e-6 / 1e-9) hold.
28. (re-verify) Every decision write that can move a recipe (`_dispatch_candidate`, `_dispatch_floor_reject`) reconciles the record
    through `squash_and_report`, not `recompute()` alone — the plan's "recomputed after EVERY ledger write (decision or override)".
    The milestone verb recognises an already-applied answer by uuid (accept/drop), like the other two verbs; a second `--accept`
    on an `accepted_i` milestone is relayed as already applied. The drift anchor's s3 screen is resolved through `ok_screen` under
    the CURRENT code_sha (falls back to the baseline anchor when none exists), never the recorded old-code header.
29. (V8) Stop drill: systemctl --user stop returned in 46 s; phase await_user:stop; AWAIT line; no retry; no orphan. Resume half: see the CHARTER Log.
30. (2026-09-21, audit of PLAN_2026-09-21_per_category_eval.md) The per-category plan was audited in five further passes after
    its pass-1 review (passes 2-6; four fresh opus·xhigh auditors per pass — harness, driver state machine, statistics,
    assembly — every blocking finding adversarially verified), then ACCEPTED AS IS by the user at the 6-pass cap. Distinct
    blocking findings per pass 4 → 5 → 3 → 2 (passes 3-6), each pass's mostly fallout of the previous pass's fixes; pass 6's
    two fixes (V3 world-frame label identities; D00s/D00bs completeness) are un-re-audited. The plan file now carries the
    audited text and its Audit trail. User decisions taken during the audit (pass 2): (a) milestones keep the 2026-09-18
    three-row rule with the suite in place of slow (light/ring on the frozen v1 per-row rule, D00s' copied by_row_rule, D00b
    seed terms); (b) every PRESENT category keeps its survival guard, watch categories included; (c) the all-TIE speculative
    pick and the pooled signs use sign(Σ z_b / R_b). Surfaced, not reversed: the move_turn ratio guard vs a facing-for-speed
    trade (D3's example rewritten to the recorded data point). From pass 4 the audit used the revised plan-audit guide
    (blocking / deferred / nit; deferred findings are listed here, not edited into the plan).
    LEFT FOR IMPLEMENTATION (deferred findings — each one the implementer will hit early and plainly):
    - V6 needs per-wave `wave k/W start|end` and `write_run start|done` log lines on the suite path to time the stop.
    - The golden test FILE changes only in `test_body_rows_carry_no_omni_fields` (its key list); the fixture stays byte-identical.
    - Per-wave host arrays must be `.copy()`-ed before the buffers are zeroed if `--device cpu`.
    - Rename the pathspec-coverage test so TOOL_FILES and drop-glob matches are exempt.
    - D00s/D00bs/D00c ids need an eval_sha suffix at a second eval-only change.
    - V4's printed queue shows X03 resumed before X05's park line (find_inflight returns the first passing header).
    - Step-8 dispatch of a rescored header: `needs_s3` = the composed base (compose_recipe upto cid).
    - `make_ctx`/`_main_env` in the candidates tests write the LIVE yaml and patch `ensure_resolution`: repoint to the v1
      fixture path and `load_resolution` during the fixture split.
    - `_baseline_raw_dir` (:6450) reads `suite` before `slow` for the replay walker.
    - The all-N/A SKIP entry sets `alternative: ADOPT` so the DECISION line's overrule command is well-formed.
    - The floor band is a Wald bound (≈ 2 % false-FAIL at the threshold); switch to Wilson (same z) if it matters.
    - Tests asserting the literal `sign(z_F + z_T)` why-string move to `sign(Σ z_b/R_b)`.
    - `anchored_drift` passes an `anchor_s2_reused` flag so guard_rows uses the √3/2 binomial factor.
    - Part G placement: resolution.py + its tests and ledger_cli.py in commit 4, the compile-snapshot fixture in commit 1.
    - Step 0: inline the cfg-identity command from the 2026-09-18 plan.
    - `inspect.getclosurevars` must be applied per method for the `Campaign` class in the seam pin.
    - Write a `floor_rescored` field on a rescored rec beside its old `floor`.
    - Let the X01 rescreen seed `stage_trace_v2` for scratch_1c (otherwise the first v2 run, X05, seeds it).
    - A `resolution_missing` raised inside `_dispatch_candidate` should not orphan an ok header (unreachable today).
    - `_resume_field` (OVERRIDE-APPLIED line) consults `migration_item` before `next_item`.
    - `resolution_notify_fields`' guard loop (:5163) iterates the expanded rule's guards (display only).
    - Suggestion: the paired-category survival floor could use `n_env` instead of `n_c` (caveat 9's numbers would move).
31. (2026-09-21, per-category plan landing) The 2026-09-18 plan's decisions 4 (floors), 5 (blocks/guards by hold) and
    decision 6's curriculum-row borrow (`BORROW` / `--curriculum-runs`, its A6) are SUPERSEDED by
    `PLAN_2026-09-21_per_category_eval.md`'s D1-D10 (decision 6's resolution / seed-term / pooled-R machinery
    stays). Decision 7's sign rule becomes `sign(Σ z_b / R_b)` over the non-N/A blocks (D5); decision 9's
    pathspecs split into `code_sha` + `eval_sha` (D8); Part C (ii)'s milestone verdict keeps its three-row
    structure with the suite in place of `slow` (D7) — see the header comment of the per-category plan file for
    the full supersession list.
32. (execution deviations, 2026-09-21) Step 2 (the categories module, B1) ran on opus·high instead of the routed
    sonnet·high — the audit moved `expand_rule`/`composite_guards`, whose output shape the tools consume, into
    that module. Step 3's implementer B ran as three sequential opus·xhigh agents instead of one: B1
    (fixtures + `run_phases.py` + `ledger_cli.py` + the manifest row), B2 (hashes/rule v2/stages), B3
    (migration/milestones/rendering). Suite waves always run to full length — no early break when every env has
    fallen — because `common_step_counter` (the 5-step observation refresh, the action-delay update) is not
    reset by `env.reset()` and an early break would shift later waves' phase between heads. `resume_gate` PARKS a
    candidate header when the queue cannot be derived (it resumed before — a silent rank-gate bypass would
    otherwise be possible).
33. (implementation choices within the plan's text) Motion/direction entries are gated on the world frame only
    (they do not read `heading_err`); `run_meta.suite.hold_s` and `aggregate.suite.hold_s` are per-label maps;
    wave > 0 setup runs under `torch.inference_mode()`; the student's `depth_latent`/`yaw_pred` reset each wave;
    `complete.json` also records `raw_written`; the `[INFO] wave k/W` lines count from 1; direction metric
    definitions — `direction_error_rad` = |angle of the mean travel-frame velocity| (`None` below `min_speed`),
    `cross_track_ratio` = |mean cross-track velocity| / mean commanded speed, `travel_speed_m_s` = |mean
    velocity|, `yaw_drift_rad` signed (+ left). `resolve()` requires `decisive_at` under rule v2 (the CLI
    defaults it to all seven categories); the rank gate applies to rule-v2 specs only (a v1 spec is a record);
    v1 promotion stages keep `EVAL_FULL`; reeval stamps are recorded as full paths.
34. (review findings deferred at steps 1-3, left for implementation / a later fix) `null_construction` drops a
    scored block with no calibrated R, and `acceptance_d00s` checks completeness on the cross-head R only; at a
    milestone `_row_worse` never runs the `decisive_at` subset check (no `by_row` `decisive_at`);
    `_rescore_fail_complete` does not copy the new entry's floor fields onto the header after a crash between the
    ledger write and the reconcile save; the pooled DECISION line loses the watch reason; `_migrate_d00s` always
    writes `supersedes "D00"` (wrong after a second eval change); a rescreen does not run `squash_and_report`
    (matters only for a future rescreen of a candidate whose predicate reads `screen.<key>`); `dry_run_migration`
    lists a false pending D00s on an already-migrated campaign; `ledger_cli show`'s `trained:` omits watch
    categories; the validator does not check `screen.eval` or that checkpoint-shape inputs name a
    `category_suite` row; `tests/unit/fixtures/gen_gait_eval_compile_snapshot.py` would raise on the suite row if
    re-run (it compiles every row); the C1b/C7 `eval_categories` tests build the pool with
    `render_pool(apply_pool_edit(...))` rather than the driver's `compose_recipe`; `expand_rule`'s watch notes
    read "reduced demand" where D5 writes "watch (reduced demand)".
35. (step 5 review fixes, 2026-09-21) Two independent opus·xhigh fresh-eyes reviews of the whole diff (the plan's fable·high
    step, run within the opus ceiling with a second reviewer) found nothing blocking; three follow-ups were applied:
    `prepare_resume` resumes from the NEWEST run dir that saved a checkpoint (run-dir names sort chronologically) instead
    of the highest iteration pooled across every run with the tag — tags are not campaign-unique, so a smoke drill's X01
    would otherwise resume from the 2026-09-19 V8 drill's model_19 (test
    `test_prepare_resume_takes_the_newest_run_not_a_stranger_with_the_same_tag`; the infra-halt test's placeholder run
    dir renamed to a real older timestamp); `_baseline_raw_dir` prefers the legacy `slow` raws again because
    `replay_reward_economy.py` reads only that layout (suite raws are the fallback for a suite-only rec); two CHARTER
    wording fixes (a rescore PASS notifies MIGRATION; the 8-s proxies overstate the suite's survival). Deferred: the
    first resume of a rescore-reopened header logs "at None" (cosmetic).
36. (C1d, user request 2026-09-22) A turn-first variant of the curriculum joins the ranked list: candidate **C1d** on a new
    chain `curriculum_turnfirst` = `t3` (c3's pool and overlay verbatim -- stand, fwd, side, back, turn -- from scratch for
    3000 iterations, `eval [suite]`, categories `[back, fwd, side, stand, turn]`, floor `category_survival` 0.10/0.30) then
    c4, c5, c6 copied verbatim from `curriculum` (8000 iterations = B0's compute). Question: C1 and C1b lost in-place
    turning (facing_closed 0.42 -> 0.19 / 0.17 at their screens) after a curriculum whose first 2000 iterations carry no
    turn command -- does the ORDER, not the content, cost turning? User decisions: it composes on the ladder's recipe like
    every candidate (B0 + C3a's caps if C3a's pooled decision adopts, B0 otherwise), and it runs next after C3a's seed-2
    pair (X07) and C3a's pooled decision -- ranked between C3a and C3b (M1 stays `after: C3b`). The `curriculum` chain is
    untouched (C1/C1b's records name it). One edit beyond the request, required by the validator: C8's `overlay_by_chain`
    gains `curriculum_turnfirst: {pool_edit: {set_share: {stand: 0.05}, renormalise: true}}` (the same row C8 carries for
    `curriculum` and `scratch_c6`; inert unless C1d is adopted). If C1d is adopted, C7's predicate (`recipe.has_pool and
    decision(C1b) != ADOPT`) holds and C7 runs on C1d's chain, as its question intends. Validated on a scratch copy
    (validator ok, launch reconcile squashes nothing, queue simulated for both pooled outcomes: X07 -> C3a pooled -> C1d s3
    -> C3b [or SKIP] -> M1) and independently reviewed (no blocking findings). Applied live 2026-09-22 08:16: graceful
    `systemctl --user stop` (X07 at model_6100; at most 99 iterations lost), relaunch with `--accept-candidates-change`
    (candidates.yaml sha256 cc48245c1be5...); X07 resumed at model_6100 (1900 of 8000 left) as `seed2_X07_1c_A15pB_c` --
    a resumed window is not bit-identical to an uninterrupted one (restarted simulator/RNG state).
37. (C2r, user request 2026-09-22: "retry c2 after c1d") Candidate **C2r** = C2's overlay (`KRABBY_YAW_PROGRESS_W 3.0`,
    `stages: all`, same `replay_args`), unconditional, ranked right after C1d (before C3b). The ladder's own verb
    `decide C2 --choose run` was not used: it re-screens C2 at rank 4, always on B0 (its base composes only candidates ranked
    before it), voids D10 permanently (cause e), and if it adopted it would squash C3a/C1d (cause a); a relaunch while it is
    pending parks X08 behind it (rank gate). A new candidate is the only way to place a C2 run after a lower-ranked one.
    C2r composes on the post-C1d recipe: after a C1d ADOPT it tests C2 on the turn-first chain (control = C1d's screen; a
    speculative C2r then needs a twin, fa03ac033e7c has no s2 screen); after a C1d REJECT it is an uninterrupted seed-3
    redraw of C2 on B0 (recipe 27333979fd03 = X05's; X05 itself was resumed at model_400), most likely rejecting again on
    move_turn survival (D10 0.58 -> 0.12 against a seed spread of 0.06). `ledger_cli decide C2r --choose skip` before its
    dispatch drops that branch. D10 stays C2's record on B0. M1 now also waits for C2r. Validator ok (only the note that
    `KRABBY_YAW_PROGRESS_W` is set by C2 and C2r), launch reconcile squashes nothing, dry-run queue: resume X08 -> C2r s3.
    Applied by a stop inside X08's c4 suite eval (the stage is not yet in `recs`, so the relaunch re-runs only the eval and
    c5 starts from the c4 checkpoint in a fresh process, as at any stage boundary) and a relaunch with
    `--accept-candidates-change`. Tests: `test_milestone_i_cites_the_loaded_resolution_and_halts_without_it` hard-codes the
    live list of candidates decided before M1 and was broken by C1d (item 36, suite not re-run then); it now lists C1d and
    C2r. Two other candidates tests read the live campaign state and have failed since the 2026-09-21 migration, not
    because of C1d/C2r: `test_a_pre_suite_state_without_eval_sha_takes_the_alias_path` (asserts the ladder is pre-suite) and
    `test_dry_run_prints_checkpoint_shape_inputs` (expects the unfilled seed_pair inputs) -- left for a fix off the live dir.
    FINDING (2026-09-22, not fixed -- a training-code change would move code_sha): the PPO update counter that drives the
    `priv_reg` coefficient (`ppo_with_extractor.py:114/347/526`, schedule 0 -> 0.1 over updates 2000-5000) is NOT restored
    on resume (the runner restores `iter`, not `alg.counter`), and the adaptive learning rate restarts from its initial
    value. Consequences: (i) X07 (C3a s2, the run whose sign overruled D11) was stopped at 08:16 for item 36 and resumed at
    model_6100 -- its last 1900 iterations trained with `priv_reg_coef` 0.0 instead of 0.1 (logged `Mean priv_reg` rose
    0.0003 -> 0.0157); (ii) X05 (C2) resumed at model_400, its ramp shifted by 400 iterations; (iii) every chain stage is a
    fresh process, so curriculum chains (C1, C1b, C1d, and C2r on C1d's chain) train c4-c6 at coef 0 (t3 ends at 0.033),
    while a single-window 8000-it recipe (B0, C2, C3a) runs at 0.1 from update 5000 -- a confound in every chain-vs-B0
    comparison, beyond the curriculum itself.
38. (scope narrowed to the C1 question, user 2026-09-22: "We're getting too tangled up in the test ladder. Lets narrow
    the scope of this plan just to resolving the C1 question we are currently testing") The campaign now ends when C1d
    is terminal: its screen decision, plus its seed-2 pair (no twin: X02 is B0's s2 screen) and pooled decision if the
    screen is speculative. candidates.yaml keeps C1, C1b, C1c, C2, C3a (all recorded) and C1d; C3b, C4-C12 and the
    milestones M1-M3 are removed (`milestones: []`) and C2r (item 37) is withdrawn before it was ever loaded -- the stop
    job that would have applied it was cancelled while still waiting. With an empty queue the driver sets phase `done`,
    notifies DONE and exits. Chosen over the alternatives on a scratch copy and independently reviewed: `decide <cid>
    --choose skip` for the rest still runs milestone M1 (no verb skips a milestone); a timed stop after C1d's DECISION
    races a millisecond gap before C3b's header is written; a fake `deferred`/`user_decision` parks with a false message.
    Launch reconcile squashes nothing; no recorded recipe_id or base moves (C1d composes only ranks 1-5). Applied
    11:14:22 by a stop inside X08's c4 suite eval (not at c6: c4 carries a category_survival floor whose REJECT would have
    dispatched C3b under the old list before the c6 window) and a relaunch with `--accept-candidates-change` (yaml sha256
    137988470c43...): X08 resumed at model_4498 with "the window had already reached its final iteration -- continuing at
    its eval" (no training redone). The REPORT header table (written at campaign start) still lists C3b-C12 and M1-M3.
    Parked, not decided: the D10 (C2) overrule question, a redo of C3a's interrupted s2 run X07 (item 37 FINDING (i)).
    Tests: the candidates and eval_categories tests pinned the live yaml; they now load the full v2 ladder frozen at
    18948abe as `tests/unit/fixtures/candidates_v2_2026-09-22.yaml` (as the v1 spec was frozen); the two tests that read
    the live state.json stay red (item 37).
    Relevant to the C1 answer (scope check, 2026-09-22): the teacher is EVALUATED on the history latent
    (`eval_crab_hex_gait.py:804-807`, hist_encoding=True) while it trains on the privileged latent, and priv_reg is what
    pulls the two together; because every chain stage restarts the PPO counter (item 37), C1/C1b ended with a latent gap
    ~0.035 (~90x B0's 0.0004) and every head with an open gap turns 0.17-0.25 while every head with a closed gap turns
    0.29-0.75 -- but the partition is collinear with the curriculum, and translation and survival held, so the data cannot
    attribute the turning loss. A C1d REJECT on F therefore leaves "curriculum vs stage-boundary resets" open; the control
    that separates them is B0's 1c recipe split into processes at C1d's boundaries (3000/1500/1500/2000, ~3.6 h). It is
    NOT scheduled (optional); after `done` it can be added and relaunched with no timing constraint.
