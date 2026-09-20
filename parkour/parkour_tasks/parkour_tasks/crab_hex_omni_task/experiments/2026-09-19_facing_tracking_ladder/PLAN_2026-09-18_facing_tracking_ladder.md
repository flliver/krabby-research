<!-- Verbatim copy of the approved plan /home/nickmagus/.claude/plans/the-facing-and-tracking-fancy-breeze.md (approved 2026-09-18,
     audit-converged at pass 9), saved into the campaign directory on 2026-09-19 at the user's request when the campaign was paused.
     Deviations discovered during implementation are in PLAN_AMENDMENTS.md beside this file and in CHARTER.md's caveats. -->

# Plan: the facing/tracking candidate ladder — an autonomous, speculative campaign for `crab_hex_omni_task`

## Context

The omni task (`crab_hex_omni_task`, branch `feat/crab-hex-omni-task`, plan of record
`lets-move-on-to-composed-moth.md`) has a seed-3 lineage 1a → 1b → 1c-004 (from scratch) → 2a → 2b. Every head since
1c-004 walks in every commanded world direction but **sacrifices facing for translation**: on the `move_turn` hold the
2b head ends with a 1.218-rad heading error and never settles; `turn_l90` ends at 0.68 rad; the turning rate sits at
0.07–0.09 rad/s under every head; per-direction speed ratios are 0.47–0.60 (fwd/lat/diag/back 0.60/0.47/0.51/0.60);
completion 0.50 and moving-hold tripod 0.38 (down from 0.49 at 1c-004). The user's reading — the policy trades one
objective for the other and mostly drops facing — is what the evidence says, and five mechanisms are established
(`$CAMP/ANALYSIS_1c_failure_2026-09-18.md`, `replay_reward_economy.md`, the lit review, the three exploration reports of
2026-09-19, the design panel and its critics, and three audit passes):

1. **Price.** Under the shipped economy the whole facing sub-economy spans 0.423/s and the *reachable* gain from turning
   at the plant's 0.08 rad/s is 0.206/s — the price of a 20 % speed loss on a 1.638/s translation stake. The
   facing-*position* term (`reward_facing_exp`, `KRABBY_YAW_W` 0.2) is 0.121/s of that; 0.302/s is the even
   `track_ang_vel_z_exp` kernel, which pays a robot that never turns. Ignoring facing costs translation **nothing**: the
   command term re-projects a held world travel direction into the body frame every step (clamp-only re-cap), so
   world-frame tracking is heading-independent by construction.
2. **Kinematics.** `ang_vel_max` 0.3 is ~2× the plant's estimated 0.15–0.18 rad/s turn ceiling (lit review §3), and the
   `move_turn` slot (0.22 m/s + 0.3 rad/s ⇒ ≈ 0.58 m/s outer-foot speed at r_foot ≈ 1.2 m) is infeasible at ANY speed the
   sampler can command (walk floor 0.21 m/s ⇒ ω ≤ 0.01 rad/s while walking) — yet it is 20–25 % of every window's rows.
   A policy asked for the impossible learns to drop the half it cannot deliver. `move_turn` therefore measures a forced
   trade and is scored **relatively** only — and every training-log number that includes move+turn rows is watched, not gated
   (the one exception is the from-scratch `dead_walker` floor at its 800–1000, which is relative to B0's own early trace on
   the same number — decision 4, `D03`).
3. **Observation.** Slots 6/7 clip the facing error at 0.6 rad and slot 5 (yaw-rate command) saturates at 0.2 rad of
   error, so the policy is magnitude-blind above 0.6 rad and the 0.2-s history cannot integrate a 12-s turn.
4. **Curriculum.** The 1a → 1b → 1c hard switches lost translation at every handoff (fwd speed ratio 0.76 → 0.16), the
   soft gates were blind to it, and move+turn was never taught incrementally; the RSI bank holds no turn or backward frame.
5. **Economy.** The angular kernel pays ~1.0/s for standing still on the ~88 % of rows that command no turn; the staged
   `KRABBY_OMNI_ANG_TURN_ONLY` lever (replay margin +16.6 % → +37.5 %) is implemented but default-off — and the panel
   showed it also erases the turn-row settle bonus (finish vs dawdle becomes near-indifferent), so it is demoted here.

**What the user asked for (2026-09-19):** a plan that tests a *ranked list of candidates* to improve facing AND tracking
as a *series of sequential experiments*, each building on the previous successes, running *autonomously through a large
number of experiments*; when an outcome is uncertain the plan picks the best option and moves on while notifying the user
and asking to confirm; if the user overrules, the work done under the wrong choice is discarded and the campaign
continues from the new choice — a branch mispredict. The user's own first candidate is an **additive capability
curriculum**: enable command types one at a time (stand/forward → + sideways and backward → + turn in place → + turn while
moving forward/sideways → + omnidirectional travel → + turn while moving omnidirectionally), every enabled type keeping a
floor share of the command windows once enabled (backward from c2 onward — user amendment 2026-09-19).

**Decisions already taken with the user (AskUserQuestion, 2026-09-19):** screening depth = the phase-1 formation chain only
(8000 iterations ≈ 3.7–3.9 h with evals), scored on the flat canary `slow__omni`; adopted recipes are promoted through 2a/2b
only at milestones. Success rule = **joint, relative**: adopt when the composite (facing on the turn + `move_turn` holds
AND per-direction speed ratios) beats the current recipe outside the eval resolution with no regression on
completion/tripod; inside resolution = uncertain → speculative decision + notification. Seeds: one-seed (3) screens; seed 2
only on uncertain or adopted candidates. No wall-clock cap — the driver works through the whole ranked list.

Abbreviations: `$R` = `/home/nickmagus/krabby/krabby-research`; `$OMNI` = `$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task`;
`$FWD` = `$R/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task`; `$CAMP` = `$OMNI/experiments/2026-09-16_1805_omni_formation`
(the formation campaign, closed at its 2b decision point); `$LAD` = `$OMNI/experiments/2026-09-19_facing_tracking_ladder`
(this campaign, new — the ladder never writes into `$CAMP`); `$TOOLS` = `$OMNI/experiments/tools`; `$PY` =
`/home/nickmagus/krabby/isaac_venv/bin/python`; `$GE` = `$FWD/scripts/gait_eval`.
Standing constraints: one Isaac process at a time (256 envs saturate the GPU); never push (the user publishes); other
sessions share the checkout; `_subenv` strips every `KRABBY_*` from the parent — a knob reaches a run only through the
subprocess `env` dict; every omni process sets `KRABBY_FLAT_TERRAIN_GEOM=ring`; the eval manifest's existing rows are
shape-frozen (a new row may carry its own `env:`); no `__init__.py` under `experiments/`; keep-set rule for git; Kit
DerivedDataCache corruption (`isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache/`, quarantine =
`mv DerivedDataCache DerivedDataCache.broken_<date>`; Kit's own logs under `isaac_venv/.../isaacsim/kit/logs/Kit/Isaac-Sim/5.1/`)
and the debug-marker OOM are infra events, never verdicts; Isaac runs under a memory cap of at least 45 G on this 59 GB box;
a SIGKILL of a running Kit is itself a corruption risk, so every stop path is SIGTERM first.

## Vocabulary (one word, one meaning)
- **incumbent / control** — the effective recipe's own seed-3 screen (same `recipe_id`, same `code_sha`, 8000 it); `B0` at the start. **ok screen** = `status ok, seed as required, recipe_id == effective, code_sha == current`.
- **base recipe** of a candidate — the recipe composed of the EFFECTIVE choices of the candidates ranked BEFORE it (`base_recipe_id`; identical to the effective recipe while the ladder runs in rank order); an out-of-rank re-run (`decide --choose run`) uses that rank-composed recipe, and its control is that recipe's ok s3 screen (the last adoption before it, else `B0`/`B<k>`; if none exists `next_item` inserts the recipe-baseline rerun first). Its seed-2 screen is the **twin** a pooled decision pairs with. **Effective entry** of a candidate = its newest non-squashed ledger entry; TERMINAL, `after` and `compose_recipe` read effective entries only.
- **anchor** — the screen of the last ADOPT that was clear at its final level (s3 clear, or pooled clear), user-confirmed, or user-overruled to ADOPT (once its recipe has an ok s3 screen), else `B0`; recomputed by `compose_recipe` under the current `code_sha`.
- **recipe head** — a pure lookup: the final-stage checkpoint of the ok s3 screen of the effective recipe; used only by milestones.
- **FLOOR gate** — the non-final-stage survival floor or the from-scratch `dead_walker` floor (decision 4), meaning "nothing usable was produced" → candidate = clear REJECT (`rejected_by_floor`); on a seed-2/twin/batched run it is WATCH (`disputed`). **RETENTION** — a per-stage relative check, always WATCH + notify, never a verdict.
- **ok run** = `status ok, code_sha == current` (any recipe); **ok screen** adds `recipe_id == effective` (incumbent, twin and recipe-head lookups) and, when several exist, means the NEWEST such experiment (an older one — e.g. the screen behind a `decide run`-squashed entry — stays a record; the re-run's screen is the recipe head); the `D00c` lookup needs only an ok run of the same `recipe_id`.
- **INTERRUPT** — the mechanical stop of a running child (train or eval) after an override, a STOP or a SIGTERM (`interrupt_check`, status `interrupted`); never a decision.
- **halt** — the driver exits 1 with `phase = halt:<xid> <why>` (infra, nan, dead_plant, eval_infra, config/eval mismatch); never a decision.
- Ledger statuses: `speculative` (uncertain, machine-chosen, pair pending) · `decided` (machine, no pair pending — a clear screen, or a MISSING-guard REJECT recorded `confidence: uncertain`) · `final` (pair-resolved, or kind resolution/measurement/baseline, or milestone PASS/WATCH) · `confirmed` (user, frozen — by `confirm` or by a milestone `--accept`) · milestone-only `open` (between (i) and the close; `chosen: null`, exempt from the options pin, not TERMINAL) · `overruled` (`override.by ∈ {user — frozen —, driver, milestone}`; its effective choice stands) · `squashed` · `skipped` · `await` (milestone FAIL; sub-status `accepted_i` after a user `--accept` of a (i) FAIL, until the promotion lands). **TERMINAL** = `{decided, final, confirmed, overruled, skipped}`. **Undecided** candidate = no effective entry — `decide <Cid> --choose run` squashes the effective entry whatever its status (a SKIP or a decision; `override.by: user, why: decide run`) and runs the SQUASH algorithm with it as `D_k` (if that entry was `speculative`, the candidate's OWN seed-2 run is removed from `pending` and, if in flight, interrupted and squashed, `squashed_by: user_choice`; the base recipe's twin continues — SQUASH (2)), so the candidate is undecided by this one rule and no milestone's `after` holds until the re-run lands (`user_choice[Cid] = {choice, uuid, at}` is kept for the deferred/user_decision stops); a `user_choice` of `skip` newer than every experiment of `Cid` writes a SKIP entry at the current base (only a later `run` clears it); `skip` is refused when `Cid`'s newest non-squashed entry is a decision (the CLI names `confirm <Did>` / `overrule <Did>` instead).
- `confidence` = the seed-3 screen's classification, immutable (`clear|uncertain`); `pooled_confidence` = the pair's (`clear|uncertain|disputed`; evaluates to `none` in a predicate when no pair ran). `decision(X).chosen` in a predicate = the EFFECTIVE choice (override applied). `screen.<key>` = the effective recipe's ok s3 screen.
- Experiment kinds: `baseline`, `candidate`, `twin`, `seed2`, `batched_s2` (a milestone's seed-2 chain), `milestone` (the 2a→2b promotion), `milestone_s2` (M3's seed-2 promotion, 2a → 2b from `recipe_head_s2`). Only `candidate` arms `dead_walker`; every other kind runs its full window. The advisory replay subprocess is registered in `rp.CURRENT_CHILD` like a child (a terminated replay is `replay: interrupted`, not `error`).
- Ledger ids are assigned in write order: `D00` resolution, `D01` turn-rate measurement, `D02` B0 s3 baseline, `D03` the dead_walker floor measurement (from B0's log), `D04` B0 s2 baseline (→ `D00b`), candidates from `D05` (C1b/C1c always write an entry, so C2's is `D08` or later); per-chain seed resolutions are `D00c-<chain>`.

## Design decisions (main thread, after the panel, its critics and audit passes 1–3)

1. **The baseline is the incumbent recipe at equal compute, and the curriculum is the first CANDIDATE.** `B0` = the `1c`
   preset from scratch (Revision-4 economy, `SLOTS_OMNI`, facing 0.2:0.6, `KRABBY_YAW_W` 0.2 — the 1c-004 recipe) for
   **8000** iterations, seed 3; `B0` seed 2 is the **second** GPU window (it is the twin of every candidate screened on `B0`
   and fixes `D00b` before any candidate decision). The recorded 1c-004 (5000 it) is a zero-GPU prior row, not a control.
   The recipe identity includes its chain, so the curriculum is compared with the single-stage incumbent under the same rule.
2. **Stages and candidates are env OVERLAYS on existing presets, not new `PHASE_NAMES`, during the campaign.** The
   subprocess gets `KRABBY_OMNI_PHASE=1a|1c` (a real preset — `activate_phase` is *setdefault*, explicit `KRABBY_*` win)
   plus the overlay; `params/env.yaml` records the truth and is audited per run (`overlay_checks`). A closing **Revision 5**
   bakes the final adopted recipe into presets and test tables.
3. **Curriculum = a pool sampler (`KRABBY_OMNI_CMD_POOL`) projected onto the existing 2-bit slot**; legacy path bitwise
   unchanged when unset; five uniforms per row in the legacy order. World-held travel semantics kept (task of record; the
   body-locked variant is C12, gated on a user decision). `side_l`/`side_r` separate; backward from c2. Move+turn share ramps
   0.10 → 0.15 → 0.20 (never above the incumbent's 0.20). Pool candidates are **transforms** (`pool_edit`) rendered to one
   string per stage at dispatch (`repr`-precision floats, never fixed decimals), so share edits and scale edits compose.
4. **Stage gates on candidates:** FLOOR only — `survival through the stage's last enabled hold < 0.30` at a NON-final stage
   that has an eval row (final stages are covered by the screen's completion guard), and `dead_walker` on a from-scratch
   stage longer than 1000 it (the `scratch_1c` and `scratch_c6` chains) with a **relative** floor `max(0.35, 0.75 × B0's own
   its-800–1000 mean)` recorded in `D03`. RETENTION (each previously enabled hold vs the incumbent's stage trace) is WATCH +
   notify on every run. Arming: `early_abort = kind == candidate and stage.from_scratch and iters > 1000`; `False` for
   every other kind (baseline, twin, seed2, batched_s2, milestone, milestone_s2 — they run the full window); `soundness =
   stage.from_scratch and iters ≥ 2000`
   for every kind (dead_plant halts everywhere). Baselines, twins and milestones never stop on a gate; eval `None` is never a
   gate — it is `halt:<xid> eval_infra` after the infra path.
5. **Decision rule (pure function of `summarize()` + the resolution record):** F block = facing on `turn_l90`, `turn_r90`,
   `move_turn`: the per-episode *closed fraction* `clip(1 − final/max(initial, 0.15), −1, 1)` (`None` = uninformative when
   `initial < 0.15`) and the signed `yaw_progress_rate`; T block = `speed_ratio` on `fwd`, `lat_l`, `diag`, `back`;
   guards (four) = completion, `tripod_walking_dir` (min over the four direction holds), `turn_survival_l90` =
   `surv[turn_l90]/surv[back]` and `turn_survival_r90` = `surv[turn_r90]/surv[turn_l90]` (falls DURING the turns, not the
   cumulative survivals nested inside completion; a conditional survival is MISSING when its denominator is 0 or
   unavailable, and the bootstrap drops zero-denominator resamples). WATCH, never scored: `track_ratio_tail` (the training-log
   ratio includes the ~20 % move+turn rows, so a successful facing fix lowers it by design), the slot-gated
   `track_ratio_move_tail`, `dir_ratio[move_turn]`, `tripod_by_hold[move_turn]`, `settle`, right-handed holds, the cumulative
   survivals. **One standardiser everywhere:** `R_i = R_eval,i`, the metric's D00 paired-bootstrap half-width, never re-fitted.
   **Block statistic** `z_F = mean_i(Δ_i / R_i)` (a closed fraction and a rad/s rate weigh equally); **block resolution**
   `R_F = max(R_F^eval, R_F^seed)` where `R_F^eval` is the paired-bootstrap 95 % half-width of `z_F` itself and `R_F^seed` is the
   block-level seed term of decision 6 — likewise `R_T` and the two absolute-units blocks; guards one-sided at 99 % on their
   own per-metric `R` (`max(R_eval,i, R_seed,i)` single-seed). Null candidate all-TIE ≈ 0.95² × 0.99⁴ ≈ 0.87 under eval noise
   (pinned), and `B0_s2` submitted against `B0_s3` is TIE on both blocks and every guard by construction (pinned). Per block:
   BETTER if `z > R`, WORSE if `z < −R`, else TIE. **Clear ADOPT** = ≥ 1 block BETTER, no WORSE (blocks or guards), and
   anchored drift OK; **clear REJECT** = any WORSE, or a FLOOR gate (a MISSING metric never blocks a REJECT); **UNCERTAIN** =
   all TIE → speculative pick = the candidate's `speculative_default` when declared (C1c: `candidate`), else the sign of
   `z_F + z_T` (exact tie → incumbent), notify, pair queued. **Anchored drift** (`z` of candidate vs anchor ≥ −R on both
   blocks and every guard) is a precondition of ANY ADOPT, clear or speculative, s3 or pooled — a clear-ADOPT screen that
   fails it is recorded `confidence: uncertain` (`evidence.anchored_drift: FAIL`), status `speculative`, pick REJECT, and
   queues the pair exactly like an all-TIE UNCERTAIN; at the pooled level drift = the candidate's two-seed mean deltas vs the
   anchor's s3 (and its s2, paired seed-wise, when the anchor recipe has an ok s2 screen) at `R_F^pooled`/`R_pooled,i`, and a
   pooled drift pass with a pooled clear ADOPT is an ADOPT (a different pick → self-overrule as usual). MISSING in a guard →
   a `decided` REJECT (TERMINAL) with `confidence: uncertain`, `evidence.missing: [<guards>]` + WATCH and NO pair (a pair cannot
   supply the metric; `decide --choose run` re-runs the candidate later); MISSING in F/T → that metric leaves the block mean
   and the best outcome is a speculative ADOPT. **Absolute-units** candidates (C3a, C3b — any candidate that changes the
   command cap/clip or, if ever added, the speed ellipse): F = `facing_closed` + whole-hold `net_yaw_rate_toward`, T =
   `achieved_vx` m/s.
6. **Resolution is measured, cited, and grows with evidence.** `D00` = paired episode bootstrap (B = 2000, paired on
   `env_index`; standardised block means bootstrapped directly) on explicitly listed run dirs (`resolution_inputs:` in
   `candidates.yaml`, full paths — e.g. `outputs/omni_plan/reeval_2a/slow__omni/slow__omni/seed001/2026-09-18_16-42-00`):
   the corrected re-evals `outputs/omni_plan/reeval_corrected/{1a,1b,1c}` (listed `provenance_ok`, informational — degenerate n),
   the 1c-004 slow/light/ring, 2b slow/light/ring and the three `reeval_2a` rows (the half-widths; n ≥ 50 per hold). **Provenance
   rule:** a run dir is refused unless its `run_meta.env_vars` carries every `EVAL_CARRY` key that has NO entry in
   `EVAL_CARRY_DEFAULTS` (the five 2026-09-17 keys); a defaulted key absent from `env_vars` is taken at its default and recorded
   as `env_vars_defaulted`. **Pairing:** the candidate-vs-control `R_eval` per metric comes from CROSS-HEAD pairs on `env_index`
   (1c-004 vs 2b, 1c-004 vs 2a-reeval, per row); the same-head pair (one `slow__omni` re-eval of the 2b head, 4 min GPU, vs the
   recorded 2b eval) is reported separately as the process-noise floor, never as the candidate R. Floors from the sideways
   CHARTER (`2026-09-11_0414_sideways_formation/CHARTER.md:227`: completion ±0.10, walking tripod ±0.056; ring completion WATCH
   at milestones unless the bootstrap resolves it). `facing_closed` needs `initial`, which the recorded JSON lacks:
   `resolution.py` re-scores the saved `raw/episode_*.npz` through the new `facing_metrics` for slow/light rows; **every ring
   row** uses the documented proxy `1 − final/max(heading_error_max_abs, 0.15)` on both sides regardless of raw presence
   (`facing_closed_source` recorded). **Seed terms.** Per metric `R_seed,i = |s3 − s2|` of a control pair on the same chain;
   per block `R_F^seed = max(|z_F(control s2 as candidate vs control s3)|, rms_i(Δ_i^seed / R_eval,i) / √k)` (the second term
   is the independence-model spread of a k-metric block mean and cannot cancel across metrics, whereas the signed pair
   statistic alone can cancel to near zero when the seed deltas alternate in sign). Sources: **`D00b`** = B0's pair (window 2), cited by every
   decision on the `scratch_1c` chain AND by every decision on any other chain until that chain's own **`D00c-<chain>`**
   exists (the chain's first ok same-chain pair — formed by a twin, a seed-2 run or a batched milestone run, written as a
   lookup step BEFORE any pooled decision that would cite it, never re-fitted later); a pooled decision whose candidate and
   base are on different chains uses the `max` of the two. Single-seed decisions: blocks at `R_F = max(R_F^eval, R_F^seed)`,
   guards at `max(R_eval,i, R_seed,i)`. **Pooled decisions** keep `R_i = R_eval,i` in `z`, compute `z` on the two-seed mean
   deltas, and use `R_F^pooled = max(R_F^eval/√2, R_F^seed)` for blocks and `R_pooled,i = sqrt((R_eval,i/√2)² + R_seed,i²)` for
   guards. `lat_r`/`move_turn_r` borrow `lat_l`/`move_turn`'s R until `resolution.py --curriculum-runs` re-derives them at the
   first milestone. Levels are recorded and cited in every DECISION line. **Tests:** N = 1000 pairs of independent episode
   resamples of the 1c-004 run as candidate/control with R in the eval-only construction (seed terms 0) → all-TIE frequency in
   `[0.85, 0.98]` (the independence bound 0.87 is a floor — blocks and guards share fall events — not a target; fixed RNG seed)
   with the per-block marginal TIE frequency 0.95 ± 0.03 when `R_F^eval` is re-derived by the same paired bootstrap on the
   same-run resample distribution (a calibration check of the bootstrap; under D00's cross-head R the marginal is asserted
   only ≥ 0.92), and ≥ 0.85 with the full construction; `B0_s2` vs `B0_s3` → all-TIE on blocks and
   guards; `R_F(D00b) ≥ R_F(D00)`; six equal-magnitude alternating-sign seed deltas → `R_F^seed > 0`; `R_F^seed ≥ |z_F(pair)|`;
   a curriculum-chain decision before its `D00c` cites `D00b`'s block term.
7. **Speculation = proceed on the chosen branch, notify, resolve by evidence or by the user.** Uncertain → speculative
   choice, continue, and the confirming pair runs as the very next GPU windows: the candidate's **base recipe's** seed-2
   twin (`base_recipe_id`; run only if no ok s2 screen exists for it — `B0` s2 covers everything screened on `B0`; a
   clear-ADOPT recipe pays one twin when its first uncertain dependent arrives), then the candidate's seed 2 — run only if no
   non-squashed ok run of the same `recipe_id` at seed 2 exists (an s2 left by an earlier pair-resolved decision is reused;
   one squashed by derived-view cause (d)/(e) never is). A twin not yet started is never interrupted or squashed; it leaves
   the queue with its pending decision and is re-emitted by the next pair on that base or by `M<k>.s2`. **Pooled
   outcomes are all terminal.** The pooled pick is evaluated in this order: pooled clear → its pick; else the two seeds' own
   signs (`sign(z_F + z_T)` of candidate s_k vs base s_k) disagree → pick = incumbent, `pooled_confidence: disputed`; else the
   candidate's `speculative_default` when declared, else the pooled sign (exact tie → incumbent). Same pick as the s3 decision → `final`
   (`pooled_confidence` recorded); **a different pick** → driver self-overrule (`overruled`, `override.by: driver`, a
   `SELF-OVERRULE` line, SQUASH of dependents). A seed-2, twin or batched run that trips a FLOOR (the survival floor, or
   `dead_walker` read from its log on a from-scratch chain — never armed on those kinds) continues to its screen (decision 4)
   and is pooled normally; its pooled decision is tagged `disputed` + WATCH + RETENTION notify and the
   pick follows the pooled outcome. The user may **confirm** (freeze) or **overrule** ANY candidate/seed2 decision whatever
   its status (only resolution/measurement/baseline kinds are refused). A user `confirm` or `overrule` of a `speculative`
   decision drops the candidate's own seed-2 run from `pending` and, if it is in flight, interrupts and squashes it
   (`squashed_by: user_choice`); the base recipe's twin is never interrupted or squashed (it is the base's s2 screen, needed
   by later pairs and by `M<k>.s2`); a driver or milestone override never touches `D_k`'s own seed-2 run (by then the pooled
   run has finished).
   Overrule → **SQUASH**: every experiment whose base recipe changes is moved to `state.ladder.squashed` (records kept, run
   dirs on disk unreferenced, REPORT `SQUASHED` block); every SKIP entry whose recorded `base_recipe_id` changes or whose
   predicate now holds, and — transitively — every SKIP whose `conditional_refs` include a squashed candidate, is squashed
   too (candidate undecided again, re-evaluated at its rank; a decision that ran under a then-true predicate stays and is
   governed by its base hash); a child in flight is interrupted within one poll (SIGTERM, then
   SIGKILL after 120 s); the recipe is recomputed from the ledger's effective choices; the derived queue resumes at the
   overruled decision's parent. Squash is a DERIVED view recomputed on every launch (the cache in `state.ladder.squashed`);
   reverting an overrule resurrects an experiment squashed by derived-view causes (a)–(c) whose files still exist (user
   squashes, causes (d)/(e), never).
8. **No GPU probe for the turn ceiling** (`D01` records yaw smoke 3: 0.086 rad/s training / 0.063–0.094 eval; per-head yaw
   progress 0.09–0.10 at 5000 it; kinematic bound 0.15–0.18 rad/s). `ang_vel_max` is screened directly (C3a 0.21, C3b 0.18),
   each **cadence-preserving** (`CLOCK_TURN_R = 0.36/ang_vel_max` co-overlaid).
9. **Code freeze before the first GPU window** for C1–C9 and C11 (all knobs default-off, bitwise-identity tests). **`code_sha`**
   = sha256 over the blob ids of `git ls-tree -r HEAD -- <positive pathspecs>` (`$OMNI/{mdp,config}`, `$FWD/{config,mdp,scripts/gait_eval}`,
   `parkour/scripts`, `parkour/parkour_isaaclab`, `$TOOLS/run_phases.py`, `$OMNI/experiments/eval/scenarios_omni.yaml` —
   the manifest defines every screen's holds) after a Python `fnmatch` filter drops `*.md`,
   `*/experiments/20*` and `*/tests/*` (git's `:(exclude)` magic is not used); the dirty check is `git status --porcelain --
   <positive pathspecs>` filtered identically (untracked lines count). The ladder tools (`run_candidates.py`, `ledger_cli.py`,
   `resolution.py`) and the replay script are outside `code_sha` by design — they never touch a screen's training or eval
   output and decisions are pure functions of records; each ledger entry records an informational `tools_sha` over them
   (never a refusal). Recorded in `state.ladder.code_sha` and every experiment header; launch is refused while the tree is dirty.
   Record/docs commits (the plan's own `$LAD` commits, README, `parkour/scripts/README.md`) leave it unchanged. A changed `code_sha`
   at relaunch is refused unless `--accept-code-change <why>` (NOTIFY `RELAUNCH` with both hashes); then "ok screen" fails for
   every old screen, `next_item` inserts a recipe-baseline rerun `B<k>` (old screens stay as records, nothing is squashed),
   later uncertain candidates re-run their base twins (budgeted), `D00b`'s seed term carries over (eval half-widths are
   head-independent) and the anchor re-resolves under the new hash; a pair pending at the change keeps its recorded s3
   screens under the old `code_sha` (the landing is default-off, as for milestone pairs), its s2 and any missing twin run
   under the new hash, and the pooled entry records `base_code_sha`. C10 and C12 are **deferred**: accepted at load, refused at
   dispatch until their `requires_knob` is exported; reached with their predicate true (or user-forced) → `await_user:<Cid>
   code`; the user answers `decide <Cid> --choose run|skip`; `run` + the code landing → relaunch with `--accept-code-change` →
   `B<k>` → the deferred screen (billed to the candidate in the CHANGELOG).
10. **The ladder has its own campaign dir** (`$LAD`), its own launcher (no 4-day `timeout`; `MemoryMax=48G`,
    `MemorySwapMax=0`, `OOMPolicy=continue`, `KillMode=mixed`, `TimeoutStopSec=360`; the measured `MemoryPeak` after `B0` is
    recorded in the CHARTER and the cap lowered only with evidence), and reuses `run_phases.py` by import-by-path;
    `run_phases.py` gains only backward-compatible keyword arguments and refuses a `state.json` that carries a `ladder` key.

## Part A — code (all default-off; forward and sideways dumps byte-identical; omni legacy path bitwise unchanged)

### A1. Pool sampler and the command knobs — `$OMNI/mdp/crab_hex_omni_command_math.py`, `crab_hex_omni_command.py`, `config/crab_hex/omni_knobs.py`, `mdp/omni_parkour_event.py`, `mdp/omni_exposure.py`
- `POOL_TYPES` (name → slot, world flag, body band (lo, hi)): `stand`(0, (0,0)), `fwd`(1, (0,0)), `side_l`(1, (+π/2,+π/2)),
  `side_r`(1, (−π/2,−π/2)), `back`(1, (π,π)), `omni`(1, WORLD `travel_dir_range`), `turn`(2), `fwd_turn`(3, (0,0)),
  `side_l_turn`(3), `side_r_turn`(3), `back_turn`(3), `omni_turn`(3, WORLD). Per-type optional `ang_scale` (default 1.0;
  scales the row's P-control cap AND clip) and `speed_scale` (default 1.0; scales the ellipse cap at draw time).
  `CmdPool` NamedTuple (`names, weights, ang_scale, speed_scale`) with `slot_of()`, `projected_slot_probs()` and
  `rescale_to(slot_probs)`; `parse_cmd_pool(text)` grammar `name:weight[:ang_scale[:speed_scale]],...`; `render_cmd_pool(pool)`
  with `repr(float)` precision (render → parse round trip pinned: projection within 1e-6, sum within 1e-9);
  `pool_tables(pool, device, dtype)` builds K+1-entry tables whose trailing entry is the unit scale so that **`type_id = −1`
  (eval rows, RSI-staged rows) resolves to the global cap** (`torch.where(type_id < 0, 1.0, table[type_id.clamp_min(0)])`);
  `sample_pool_travel_dir(yaw, type_id, lo_t, hi_t, world_t, travel_dir_range, generator)` — one uniform, the legacy
  expressions with per-row gathered bands (zero-width band ⇒ `yaw + 0.0` bitwise), `torch.where(world_t[type_id], world, body)`.
- `CrabHexOmniCommandCfg.cmd_pool: str | None = None` (the canonical knob STRING, dumped verbatim). `_validate_omni_cfg`:
  when set — parse; refuse `dir_band is not None`; REQUIRE the parsed-and-normalised `slot_probs` to equal the
  parsed-and-normalised pool projection within 1e-6 (the ladder validator applies the same comparison at LOAD); legacy checks
  kept. `_resample_command`: legacy block VERBATIM under `if cfg.cmd_pool is None`; pool branch: `type_id = sample_slots(None,
  weights, n)` → `slot = slot_table[type_id]`, direction via `sample_pool_travel_dir`, speed with `cap0 · speed_scale`, facing
  as today; RSI-staged rows as today plus `type_id = −1`; `schedule_hook` sets `type_id[:] = −1`. `_write_command` uses the
  per-row cap/clip (unit for `−1`). Metrics `pool_<name>_frac` (step-weighted) AND `pool_<name>_draws` (per resample) exist
  ONLY when a pool is set (the nine-name metric test stays green).
- **Slot-gated tracking telemetry (WATCH):** `omni_parkour_event.py` accumulates `ep_track_dist_move`/`ep_cmd_dist_move` over
  rows whose slot is `SLOT_MOVE` only; `omni_exposure.py` gains `track_ratio_move_mean` in `OMNI_EXTRA_KEYS` (five names —
  deliberate edit of `test_emit_keys_add_exactly_the_four_omni_keys`), `push_episodes(..., track_ratio_move=None,
  track_ratio_move_valid=None)` (optional; `None` leaves the mean empty), validity `ep_cmd_dist_move ≥ min_cmd_dist_m`; the
  existing `track_ratio_mean` untouched (promotion keeps reading it).
- Knobs (all rows added in THIS step, default-off): `KRABBY_OMNI_CMD_POOL`, `KRABBY_OMNI_ANG_VEL_CLIP` (`clips.ang_vel_clip`),
  `KRABBY_OMNI_FACING_TURN_W`, `KRABBY_OMNI_TRACK_FACING_SIGMA`, `KRABBY_OMNI_PROGRESS_W`, `KRABBY_OMNI_FACING_OBS_ENCODING`
  (parsers + apply functions with the loud guards; the functions they bind are written in A2). **Clock rule** in
  `apply_omni_knobs` after `_apply_command_knobs` and the `CLOCK_TURN_R` parse: `omni_clock_turn_radius × ang_vel_max ≥ 0.25`
  else `RuntimeError` naming both knobs.
- Tests (`tests/unit/test_crab_hex_omni_command.py`, extend): pool table projects onto the 2-bit slot; grammar and
  rejections; render → parse round trip; **legacy bitwise twins** (`stand:0.2,fwd:0.8` == `(0.2,0.8,0,0)+dir_band(0,0)`;
  `side_l` == `dir_band (π/2,π/2)` with `math.pi/2` on both sides; `stand:0.15,omni:0.55,turn:0.10,omni_turn:0.20` == legacy
  1c) on `slot_id/travel_dir_w/speed_cmd/heading_target/vel_command_b`; default cfg takes the legacy path; five-uniform draw
  count independent of the pool; type frequencies and slot projection; exact body axes for `side_l/side_r/back`; `omni`
  world-uniform; per-type scales apply to their rows and **`type_id = −1` rows use exactly the global cap on a pool whose LAST
  type has scale ≠ 1**; metrics exist only with a pool and exclude staged rows; pool + `dir_band` refused; pool + mismatching
  `slot_probs` refused; `schedule_hook` clears `type_id`; `rescale_to(SLOTS_TEACHER)` round-trips. Deliberate edits:
  `test_shipped_cfg_defaults_are_the_plan_values`, `test_forward_command_class_and_cfg_have_none_of_the_omni_surface`,
  `test_init_rejects_other_unhonourable_cfgs`, `test_omni_term_replaces_both_base_debug_vis_hooks_and_leaves_the_cfg_alone`.
  `test_crab_hex_omni_phases.py`: `test_parse_cmd_pool_knob`, `test_pool_type_names_match_the_math_module`, one
  parse/apply/refuse test per new knob, `test_clock_radius_rule_refuses_a_stopped_clock`. `test_crab_hex_omni_exposure.py`: the
  five-key tuple, the optional move args, the validity rule.

### A2. Reward and observation candidates — `$OMNI/mdp/crab_hex_omni_rewards.py`, `crab_hex_omni_observations.py`, `crab_hex_omni_mirror.py`
- `reward_facing_exp_turn_slot` = `reward_facing_exp × slot_is_turning(cmd_term.slot_id)`; bound as a NEW term
  `reward_facing_turn` weight 0.0 — a 37th name; deliberate edits: `test_the_cfg_is_the_forward_35_plus_reward_yaw_progress`
  (36 → 37), the integration `test_reward_names_and_scan_pattern_unchanged`, the Appendix-A delta. Knob `KRABBY_OMNI_FACING_TURN_W`.
- `track_lin_vel_xy_exp_facing_coupled` = stock × `exp(−(wrap(ψ−yaw)/σ)²)`; `KRABBY_OMNI_TRACK_FACING_SIGMA` swaps the FUNC
  (the `ANG_TURN_ONLY` swap pattern, same three guards); unset = stock.
- `KRABBY_OMNI_PROGRESS_W` → `reward_forward_progress_along_command.weight`.
- `KRABBY_OMNI_FACING_OBS_ENCODING=raw|sincos` (obs param `facing_obs_encoding`, default `raw` bitwise): `sincos` writes slot
  6 = `sin(err)`, slot 7 = `cos(err)`. The module constants `HEAD_SIGNS_LR/FA/ROT_Z_PI` stay as the `raw` literals;
  `head_signs(encoding)` returns them for `raw` and slot-7-flipped copies for `sincos`, and `build_mirrors` resolves the
  encoding via `_facing_obs_encoding(env)` = `env.cfg.observations.policy.extreme_parkour_observations.params.get("facing_obs_encoding", "raw")`,
  raising a named `ValueError` when the obs cfg cannot be resolved (never a silent `raw`; the mirror test module's
  build-time tests gain the `sincos` twin and the unresolvable-cfg refusal): under `sincos` slot 6 keeps −1 in BOTH
  tables, **slot 7 = +1 in BOTH `HEAD_SIGNS_LR` and `HEAD_SIGNS_FA`** and `HEAD_SIGNS_ROT_Z_PI[7] = +1`; history unchanged.
  Tests: `test_head_tables_are_the_plan_literals` pins both encodings; `test_kinematic_head_signs_independent` gains a sincos
  variant for both planes; obs tests; reward tests; `tests/integration/test_crab_hex_omni_configs.py`: `EXPECTED_RL` gains
  `("rewards", "reward_facing_turn")` and the obs param `facing_obs_encoding`; `CMD_FIELDS_ADDED` gains `cmd_pool` (seven
  fields — `test_command_term_block`, `test_siblings_carry_no_omni_attribute`; the command subtree is stripped from the
  leaf delta, so `cmd_pool` is pinned there, not in `EXPECTED_RL`).

### A3. Eval harness additions — `$GE/metrics.py`, `$GE/report.py` (additive keys only)
- `facing_metrics` gains `heading_error_initial_abs` and `facing_closed` (definition in decision 5; `None` when uninformative,
  counted in `n_uninformative`; a hold containing a fall is scored on its truncated window). `report.aggregate` adds both to
  `facing_keys`. The golden fixture (body rows, no facing block) stays byte-identical — `test_crab_hex_gait_eval_golden.py` is
  the regression check; a small world-row unit test covers the two new keys.

### A4. Driver seams — `$TOOLS/run_phases.py` (backward compatible; every existing test stays green or is edited by name)
- `train(..., *, extra_env=None, interrupt_check=None, run_name=None, abort_track_ratio_min=None)`: merges `extra_env` after
  the two fixed keys (asserts none of `KRABBY_OMNI_PHASE/KRABBY_PLANT/KRABBY_PHASE/KRABBY_SIDE_PHASE`), always sets
  `PYTHONUNBUFFERED=1`, passes `--run_name`, records `current.extra_env`, re-checks `interrupt_check()` immediately before
  `Popen`, registers the child in a module-level slot (`CURRENT_CHILD`), calls `interrupt_check()` first in every 120-s poll
  tick → `proc.terminate()`, wait 120 s, then `kill()`, and — **after the poll loop exits for ANY reason** — checks the flag
  once more BEFORE the checkpoint classification so a child that exited between ticks is reported `(None, "interrupted", log)`,
  never `infra`; `abort_track_ratio_min` overrides the module constant for `dead_walker`. `train_retry` forwards the same
  kwargs and retries `infra` only while `not interrupt_check()`. `_wait_isaac_clear(interrupt_check=None)` returns early when
  the check fires. `latest_run_ckpt(root, t0, name_suffix=None)` resolves the head from `*_<run_name>*` dirs only when given;
  `train` passes `run_name`.
- **`_run_eval(..., max_attempts=3, interrupt_check=None) → (latest, a, info)`** switches from `subprocess.run` to `Popen` +
  a 30-s poll loop (behaviour identical to today when `interrupt_check is None`, with one deliberate change: on the 3600/5400-s
  timeout the child gets `terminate()` → 120 s → `kill()` instead of an immediate SIGKILL, then the existing freshness/retry
  logic), registers the child in `CURRENT_CHILD`, applies the same before-`Popen` / per-tick / after-loop interrupt checks, and returns `info = {status: ok|none|interrupted, rc, attempts}`
  (no retry on `interrupted`; `rc = −9` is visible to the infra detector). **`eval_morph`/`eval_obst` unpack the triple and
  KEEP returning `dict | None`** — the summary carries `eval_status` and `eval_rc`; on an interrupt they return
  `{"eval_status": "interrupted", "run_dir": None}` (distinguishable from `None` = no metrics); `run_phase`'s assignments are
  untouched. `eval_env_for(..., carry_overlay=None)`: after the preset `EVAL_CARRY` overlay, injects `EVAL_CARRY_DEFAULTS` for
  keys absent from the preset (so `run_meta.env_vars` records the effective value), then `carry_overlay` filtered to
  `EVAL_CARRY` keys (bitwise unchanged with `None` for today's five keys); `eval_morph`/`eval_obst` pass it through.
  **`EVAL_CARRY` gains `KRABBY_OMNI_FACING_OBS_ENCODING` and `KRABBY_OMNI_ANG_VEL_CLIP`** with
  `EVAL_CARRY_DEFAULTS = {ENCODING: "raw", ANG_VEL_CLIP: "0.2"}`. Deliberate edits in `test_crab_hex_omni_gate.py`:
  `test_eval_carry_names_the_obs_and_command_knobs` (seven names; `KRABBY_OMNI_CMD_POOL` BANNED),
  `test_every_eval_env_carries_every_eval_carry_key_from_the_preset` (`expected = pe.get(key, EVAL_CARRY_DEFAULTS.get(key));
  assert expected is not None and ev.get(key) == expected`), new: an overlay `ENCODING=sincos` appears in the canary env, a
  preset variable still never does.
- `summarize` hoists `achieved_speed_by_hold` (m/s), `facing_closed_by_hold` (the ring proxy when the row has no raws —
  `facing_closed_source` recorded), `tripod_walking_dir`, `yaw_progress_turn_min`, `turn_survival_l90/r90` (conditional
  survivals), `ratio_by_hold` (speed ratio over EVERY walking hold of the row — adds `lat_r`, `move_turn_r` on the curriculum
  row; WATCH only) — existing keys untouched; milestones run `summarize()` on each row's aggregate JSON (`run_dir`), never on
  `eval_obst`'s reduced dict. `track_ratio_series(log_path, key="track_ratio_mean")` reads any `Metrics/base_parkour/<key>`
  series (default = today's label, bitwise; `track_ratio_tail` unchanged; the driver's `track_ratio_move_tail` =
  `track_ratio_tail(track_ratio_series(log, key="track_ratio_move_mean"))`); `_run_eval` also records its last `info` in
  `rp.LAST_EVAL_INFO` so the infra detector reads `rc` even when `eval_morph` returned `None`; `EXPO_BASE` gains
  `track_ratio_move_mean`; `exposure_from_log` collects
  `pool_*_frac`/`pool_*_draws` too (a synthetic-log test + a cross-pin `set(OMNI_EXTRA_KEYS) ⊂ set(EXPO_KEYS)`). `main()`
  refuses a `state.json` carrying a `ladder` key. `Campaign.save` writes both files via temp + `os.replace`. A `set_seed(c,
  seed)` helper sets `seed` and `key`.

### A5. Replay screen — `$CAMP/replay_reward_economy.py` (extend, keep every existing flag)
- Flags `--yaw-w`, `--yaw-progress-w`, `--facing-turn-w` (gated on the manifest's turn holds), `--coupled-sigma`,
  `--progress-w` (alias of `--new-progress-w`); the existing `--walker` points at `B0`'s own `raw/` dir; creeper raws stay
  `reeval_corrected/1c`; `--new-ang-turn-only` exists. Output: G1/G2/G3 + a turn-hold income ordering table. Advisory only;
  a subprocess failure is `replay: error`. Path via `candidates.yaml` `replay.script`. The replay cannot price a cap change
  (C3a/C3b declare no replay).

### A6. Manifest — `$OMNI/experiments/eval/scenarios_omni.yaml` (append one row; the three legacy rows untouched)
- `curriculum__omni`: flat, 100 episodes, `env_seed 1`, `episode_length_s 92`, `spawn_yaw first_face`, prefix-ordered holds
  by enabling stage `stand, fwd` (c1) · `lat_l, lat_r(0.22, −π/2, 0), back(0.22, π, 0)` (c2) · `turn_l90, turn_r90` (c3) ·
  `move_turn, move_turn_r(0.22, 0, face 0)` (c4) · `diag` (c5) — every hold 8 s except `turn_l90`/`turn_r90` at 12 s, as in
  `slow__omni` (89.5 s incl. the 1.5-s prologue, inside the 92-s episode). WATCH telemetry at the curriculum stage evals;
  scoring stays on `slow__omni`. Deliberate edit: `tests/unit/test_crab_hex_gait_schedule.py::test_committed_omni_manifest_parses`.

## Part B — the ladder driver (`$TOOLS/run_candidates.py`, `ledger_cli.py`, `resolution.py`, `launch_candidates.sh`)

**Reuse by import-by-path** (the `run_morph_exposure.py` precedent): `rp.Campaign` + `rp.set_seed`, `_subenv`,
`_wait_isaac_clear`, `latest_run_ckpt`, `with_plant`, `eval_env_for`, `train`, `train_retry`, `_run_eval`, `eval_morph`,
`eval_obst`, `summarize`, `walking_tripod`, `survival_by_hold`, `_manifest_hold_s`, `track_ratio_series/tail`, `dead_walker`,
`exposure_from_log`, `smoke_fields`, `phase_checks` (minus the two stage-keyed entries → `overlay_checks`), `phase_lines`
(+ the driver's own curriculum-row lines), `gate_1c_verdict`/`gate_facing_verdict` (WATCH only), constants, `rp.ph`,
`rp.FWD_PKG`, `rp.CURRENT_CHILD`. NOT reused: `Campaign.ensure_header` — the ladder writes `ensure_ladder_header(c, cands)`.
Layout: all tools at `tools/` depth (`REPO = HERE.parents[5]`), no `experiments/2026-`/`rx.`/`rf.`/`STUDENT_TASK` literals
(docstrings avoid "perf."), added to `test_experiment_layout.PKGS`; every dated path is a CLI/yaml input; the unit name
arrives as `--unit "$UNIT"`.

**Files in `$LAD`:** `CHARTER.md`, `candidates.yaml` (sha256 pinned at first launch; a changed hash refused without
`--accept-candidates-change`), `ledger.json` (driver-owned, atomic), `ledger_overrides.jsonl` (CLI-owned, append-only, fsync,
uuid per line; lines are decision overrides `{uuid, decision, choice, why}` or candidate choices `{uuid, candidate, choice:
run|skip, why}` or milestone answers `{uuid, milestone, accept|drop}`), `state.json`/`heads.json` (the `run_phases` contract:
`phase` ∈ `run | done | await_user:<why> | halt:<xid> <why>`, `current.{phase,tag,iters,kind,...}` with `kind ∈ {rl, eval,
replay, bootstrap, wait}` so `heartbeat_phases.sh` works unchanged — a ⚠️ during a CPU-only window is expected; plus
`state.ladder.{candidates_sha256, code_sha, unit, resolution_ids, pending: [Dids], user_choice{}, experiments{}, squashed{}}`),
`resolution.json` (ingested if present, computed if absent), `REPORT.md`, `CHANGELOG.md`, `notify.log`, `orchestrator.log`, `STOP`.

**`candidates.yaml` schema (v1):** `plant`, `seeds {screen: 3, confirm: 2}`, `screen {chain, eval: [slow]}`, `chains{}`
(named lists of stages `{stage, base: 1a|1c, iters, from_scratch, overlay{}, eval: [...], floor: survival|dead_walker|none}`),
`decision_rule` (decision 5 verbatim, frozen into `D00`), `resolution_inputs: [{run_dir, provenance_ok}]`, `replay {script,
mode: advisory|none}`, `baseline {id: B0, chain: scratch_1c}`, `candidates[]` (`id`, `kind`, `question`, `chain`, `overlay{}`
per stage or `stages: all`, `overlay_by_chain{}` (C8), `pool_edit {set_share{}, renormalise, ang_scale{}, speed_scale{}}`,
`teacher_overlay{}`, `eval_carry: []` ⊂ `rp.EVAL_CARRY`, `absolute_units`, `replay_args`, `depends_on` (ordering only: every
dep TERMINAL), `requires_knob`, `conditional` (fixed vocabulary: `decision(<Cid>).chosen|confidence|pooled_confidence|status|
floor_stage`, `recipe.chain`, `recipe.has_pool`, `screen.<key> <op> <number>` over `summarize`/`exp.screen` keys),
`conditional_refs` (derived at load), `speculative_default: incumbent|candidate`, `deferred: true` (C10/C12), `user_decision:
true` (C12)), `milestones[] {id, after, promote: [2a, 2b]}` where `after` = "every candidate up to and including X has a
TERMINAL entry". Stage schema: `base ∈ {1a, 1c, 2a, 2b}`, `resume_from ∈ {previous_stage | recipe_head | recipe_head_s2 |
scratch}`; the promotion stage list is `[{stage 2a, base 2a, resume_from recipe_head, eval EVAL_FULL}, {stage 2b, base 2b,
resume_from previous_stage, eval EVAL_FULL}]` (the 2a rows are WATCH at every milestone, vs the previous milestone's 2a head;
only the 2b rows are scored), and
`M3.final_s2` is the same list at seed 2 with the first stage resuming from `recipe_head_s2` (the recipe's ok s2 screen's final
checkpoint — the seed-2 chain is never re-run). **Validator (load time, loud):** knob names ⊂ (union of keys over `rp.ph.PHASES[*].env` ∪ `rp.ph.FORMATION` ∪
`omni_knobs` `*_VAR` ∪ `rp.EVAL_CARRY` ∪ `"KRABBY_[A-Z_]+"` literals scanned from `rp.FWD_PKG/config/crab_hex/agents/parkour_mdp_cfg.py`;
pinned: `KRABBY_YAW_W` accepted, `KRABBY_YAW_WEIGHT` refused); values round-trip through `omni_knobs.parse_*`; the two
`_validate_omni_cfg` rules, the pool/SLOTS projection rule and the clock rule replicated (`r_turn × cap ≥ 0.25`; `≥ 0.35` for a
cap candidate unless its `question` records the cadence change); manifest 12-s-turn feasibility recomputed with the overlaid
`ang_vel_max` (refuse < 0.15); `eval_carry` ⊂ `EVAL_CARRY`; `requires_knob` exported (deferred candidates checked at dispatch);
`replay_args` flags present in the script's `--help`; no `depends_on` cycle; two adopted candidates that both set a raw pool
string for the same stage refused; overlay key collisions reported.

**Recipe** = `{chain, adopted: [ids in ledger order], overlay_by_stage{} (keyed by the chain's stages AND the promotion
stages `2a`, `2b` — `stages: all` covers all of them and a per-stage `overlay{}` may name 2a/2b; pool_edits applied in order and
rendered to one string + its SLOTS projection per stage; for 2a/2b the pool, when the recipe has one, is `rescale_to(SLOTS_TEACHER)`
of the rendered final-stage pool with every pool_edit applied — except adopted `set_share`/`renormalise` edits, which are
applied AFTER the rescale so they survive it (C8 on 2a/2b → `0.05:0.55:0.15:0.25`) — then the candidate's `teacher_overlay{}`
on top. A CHAIN stage's `overlay{}` is chain-stage-scoped: its preset-emulation keys `DIR_BAND`, `RSI_SOURCE`, `FACING`,
`YAW_W` NEVER reach 2a/2b (the 2a/2b preset facing ramp 0.2:1.2 / 0.2:2.4 stands), and its `KRABBY_OMNI_CMD_POOL`/`SLOTS`
reach them only in the rescaled form above (pool = `rescale_to(SLOTS_TEACHER)` + adopted share edits, SLOTS = that pool's
own projection — the `_validate_omni_cfg` equality rule requires it; for `B0+C1` it equals `SLOTS_TEACHER`); the
cap-candidate facing clamp (2a/2b `KRABBY_OMNI_FACING` upper bound = min(preset, 8 s × the recipe's `ang_vel_max`)) is
applied by `compose_recipe` after validation and is the one driver-written `FACING` key; the default `teacher_overlay` is the CANDIDATE's
own `stages: all` overlay (for a candidate with its own per-stage `overlay{}`, the final stage's overlay minus pool keys and
minus the preset-emulation keys); C1 and C1c declare `teacher_overlay: {}`; a raw-string chain overlay such as C8's
`KRABBY_OMNI_SLOTS` on `scratch_1c` is chain-stage-scoped unless a `teacher_overlay` is declared — C8 declares
`KRABBY_OMNI_SLOTS=0.05:0.55:0.15:0.25`; the validator refuses a candidate-declared 2a/2b overlay that sets `KRABBY_OMNI_FACING` unless the
candidate declares it in `teacher_overlay` (pinned: `compose_recipe(B0+C1).overlay_by_stage["2a"]` carries no `FACING`,
`RSI_SOURCE` or `DIR_BAND` key and its pool projects to `SLOTS_TEACHER`; same for `B0+C1c`); the validator refuses a 2a/2b pool or
SLOTS overlay that projects neither to `SLOTS_TEACHER` within 1e-6 nor to `SLOTS_TEACHER` with the adopted `set_share`/
`renormalise` edits applied (pinned for C8: `0.05:0.55:0.15:0.25` on both chains); `renormalise: true`
redistributes a freed share proportionally over the walking (slot-1) types only, so turn and move+turn shares are unchanged
and C8's c6 pool projects to `0.05:0.65:0.10:0.20` — pinned), eval_carry (union over adopted candidates; present in the 2a/2b
env and carried into their `EVAL_FULL` rows), conflicts[], anchor, recipe_id = sha256(canonical{chain, overlay_by_stage})[:12]}`;
`compose_recipe(ledger, cands, code_sha)` excludes overruled/squashed choices, applies overrides, and reports the anchor;
`stage_env(recipe, stage, extra)` is what `train(extra_env=...)` receives (test: `compose_recipe(B0+C2+C5).overlay_by_stage["2a"]`
carries `KRABBY_YAW_PROGRESS_W=3.0` and `KRABBY_OMNI_FACING_OBS_ENCODING=sincos`; the curriculum recipe's 2a pool projects to
`SLOTS_TEACHER`).
Every experiment records `base_recipe_id`, `recipe_id` and `code_sha`; every SKIP entry records the `base_recipe_id` at which
its predicate was evaluated.

**Main loop (one process, resumable, derived queue):**
```
load+validate candidates; ledger; ensure_ladder_header; phase=run; state.ladder.{unit, code_sha}; refuse a dirty code tree or a
    changed code_sha unless --accept-code-change; install the SIGTERM handler (sets the interrupt flag and terminates
    rp.CURRENT_CHILD; never exits from inside the handler); _wait_isaac_clear()
if resolution.json absent: resolution (Part C step 0) -> ledger D00 (final) + NOTIFY RESOLUTION; D01 measurement recorded
loop:
  ingest_overrides()  (complete lines only; rescan from 0 at launch, dedupe by applied uuids; decision overrides may SQUASH +
      recompute; candidate choices update state.ladder.user_choice[Cid] = {choice, uuid, at}: `run` squashes the candidate's
      effective entry — a SKIP or a decision — (override.by: user, why: decide run) and runs SQUASH with it as D_k,
      `skip` makes the driver write a SKIP entry (evidence: user) and interrupts that candidate's
      in-flight experiment; milestone answers below)
  if interrupt flag (SIGTERM) or STOP: save; phase=await_user:stop; NOTIFY AWAIT; exit 0
  resume an in-flight experiment if one exists, is not squashed (a candidate's `user_choice` of `skip` newer than the
      header's `started`, or a `confirm`/`overrule`/`decide run` of its decision newer than the header's `started` for a
      seed-2 run, squashes the running experiment on ingest and in the derived view — `squashed_by: user_choice`, never a
      decision) and `interrupt_reason(xid)` is falsy: rp.set_seed(c, header.seed); glob LOG_ROOT/<experiment>/*_<tag>*,
      newest ckpt by iteration; remaining = target_final - iter + 1 -> if remaining <= 0: head = that ckpt, skip training,
      continue at the eval; else train the remainder with the recorded overlay (tag <tag>_c; record keeps the ORIGINAL
      resume/iters); no ckpt -> rerun the stage; a half-written eval is rejected by _run_eval's freshness rule and re-run
  item = next_item():  B0 s3 -> B0 s2 -> recipe-baseline rerun B<k> if the next item's BASE recipe (the effective recipe in
                       rank order; the rank-composed recipe for an out-of-rank re-run) has no ok s3 screen
                       -> pending pairs (the base recipe's twin first when it has no ok s2 screen, then the candidate s2)
                       -> milestone whose newest entry is `await` WITHOUT the `accepted_i` sub-status -> await_user:M<k> again
                          (NOTIFY AWAIT; exit 0)
                       -> milestone whose `after` holds (every earlier candidate TERMINAL and not undecided) and which has no
                          final entry for this eligibility: (1) if R_k has no ok s2 screen -> emit M<k>.s2 (kind batched_s2; the
                          recipe chain at seed 2); (2) once R_k's s2 screen exists (from M<k>.s2 or a twin) run Part C (i) as a
                          zero-GPU step — the milestone entry is opened here (`status: open`); a contradiction overrules the
                          dropped adoption (`override.by: milestone`) and squashes M<k>.s2 and the block members screened after
                          it; the members re-screen first at their ranks, and M<k>.s2 is re-emitted only if the resulting
                          recipe has no ok s2 screen; (3) only when (i) is resolved (no `decided` adoption on R_k awaiting its
                          batched confirmation, or the entry is `accepted_i`) -> emit M<k>.promo (kind milestone; stages [2a,
                          2b] at seed 3, resume_from recipe_head, the 2a/2b overlays of the recipe, EVAL_FULL at both heads;
                          NEVER omitted at M1 — there is no previous milestone, and M1's heads are M2's rolling reference;
                          from M2 omitted when recipe_id equals the previous milestone's); (4) from M2, if both experiments are
                          omitted the milestone is a no-op: close the open entry at once (`chosen: PASS`, `evidence.carried_from:
                          M<k−1>`, `promoted_head {2a, 2b}` carried forward, status `final`), NOTIFY MILESTONE "carried
                          forward", continue — each experiment with its own xid/header/resume/interrupt/squash
                       -> M3 has a final entry (PASS/WATCH) and no ok milestone_s2 experiment of the effective recipe ->
                          M3.final_s2 (kind milestone_s2; [2a, 2b] at seed 2 from recipe_head_s2)
                       -> first ranked undecided candidate whose deps are TERMINAL: if its newest user_choice is `skip` ->
                          write a SKIP entry (evidence: user, base_recipe_id = current base) and move on; else dispatch it if
                          its `conditional` holds or its user_choice is `run` (else write chosen=SKIP, status skipped,
                          evidence=the predicate, base_recipe_id; NOTIFY DECISION ... chosen SKIP; move on), and if its base
                          recipe has no `speculative` decision;
                          a deferred candidate whose knob is missing -> phase=await_user:<Cid> code; NOTIFY AWAIT; exit 0
                       -> C12 (user_decision, no user_choice) -> await_user:C12; NOTIFY AWAIT; exit 0
  if none: phase=done; NOTIFY DONE; exit 0
  rp.set_seed(c, item.seed); replay (advisory) if declared; record in evidence
  xid = new id; header {kind, candidate, seed, chain, base_recipe_id, recipe_id, code_sha, overlay_by_stage, started, inflight}; save
  for stage in chain:
      interrupt_reason(xid) checked before launch; env = overlay_by_stage[stage]; tag = f"{seedkey}_{xid}_{stage}_{plant}"
      early_abort/soundness per decision 4; ckpt, status, log = rp.train_retry(..., extra_env=env,
            interrupt_check=lambda: interrupt_reason(xid), run_name=tag, abort_track_ratio_min=floor_from(D03))
      status interrupted -> exp.status=interrupted; break
      status infra|nan|dead_plant -> if the interrupt flag is set: exp.status=interrupted; break; else one experiment-level
            retry of the stage; then phase=halt:<xid> <status>; NOTIFY HALT (for a repeat on the same candidate the line
            quotes, per kind: candidate -> `decide <Cid> --choose skip`; seed2 -> `confirm <Did>` or `overrule <Did> --choose
            <incumbent pick>` (a user confirm/overrule of a speculative decision whose pair is in flight interrupts and
            squashes that pair run, `squashed_by: user_choice`, and removes it from pending); batched_s2|milestone|
            milestone_s2 -> "infra only: quarantine/relaunch; otherwise `resume-milestone <Mid> --drop <Cid>` (an ADOPT in the
            block) changes the recipe"; twin -> "infra only: quarantine/relaunch; otherwise overrule the
            adoption that forms this base (`overrule <Did of the last ADOPT in the base> --choose REJECT`) — a twin is the
            base's s2 and is not dropped by confirming the waiting decision"); exit 1
      status dead_walker (candidate only) -> exp.status=rejected_by_floor; floor_stage=stage; break
      rec = stage record (evals keyed by row name with run_dir inside, the run_phases shape) + smoke/exposure +
            phase_checks(filtered) + overlay_checks(params/env.yaml vs env)
      overlay_checks FAIL -> phase=halt:<xid> config_mismatch; NOTIFY; exit 1
      for row in stage.eval: rec.evals[row] = rp.eval_morph(..., carry_overlay=env, interrupt_check=..., max_attempts=1)
            (the driver owns the retry: one attempt per call, the infra detector between them);
            rec.evals[row].eval_status == interrupted -> exp.status=interrupted; break out;
            assert run_meta.env_vars[k] == env[k] for k in (recipe.eval_carry ∪ candidate.eval_carry ∪ every EVAL_CARRY key in env)
            (mismatch -> halt eval_mismatch); eval None after the infra path -> halt:<xid> eval_infra
      floor = stage_floor(stage, rec); retention = stage_retention(stage, rec, incumbent_trace) -> WATCH lines + NOTIFY RETENTION
      floor fail: kind candidate -> exp.status=rejected_by_floor; floor_stage=stage; break
                  kind baseline|seed2|twin|batched_s2|milestone|milestone_s2 -> evidence.floor + WATCH ('disputed'); the run continues to its screen
      head = ckpt; recs[f"{xid}:{stage}"] = rec; inflight=None; save; REPORT stage block; ingest_overrides()
  if exp.status == interrupted: continue
  if rejected_by_floor: ledger D (REJECT, clear, evidence=floor message, floor_stage); NOTIFY DECISION; continue
  exp.screen = summarize(final slow eval) + track_ratio_tail + track_ratio_move_tail (both WATCH) (+ absolute units); status ok
  D00c lookup: if exp.kind in (twin, seed2, batched_s2) and exp.chain != scratch_1c and the chain has no D00c-<chain> and an ok
      s3 RUN of the same recipe_id exists -> ledger D00c-<chain> (kind resolution) from this pair; NOTIFY RESOLUTION
  dispatch: baseline -> ledger D (final); if this is B0 s3: D03 (kind measurement) = max(0.35, 0.75 × the its-800–1000
                         track_ratio mean of this log via rp.track_ratio_series) — written once, never by a B<k> rerun; if this
                         is B0 s2: D00b + RESOLUTION re-emitted (the recipe head is a lookup, never set)
            candidate -> base = the ok s3 screen of the candidate's base recipe (Vocabulary: rank-composed; = the effective
                         recipe in rank order, the recipe before Cid for an out-of-rank re-run); decide(screen, base, resolution, anchor) -> ledger D
                         (status decided|speculative, confidence frozen); NOTIFY DECISION; uncertain -> pair queued;
                         clear ADOPT -> batched s2 at the milestone
            twin      -> record as the base recipe's s2 screen (never a decision)
            seed2     -> decide_pooled (decision 7 outcomes, all terminal); same pick -> final; different pick -> self-overrule -> SQUASH
            batched_s2 -> record as R_k's s2 screen; run Part C (i) NOW (next_item step 2)
            milestone  -> Part C (ii); the milestone entry is closed (PASS/WATCH/FAIL) after this dispatch
            milestone_s2 -> record M3's seed-2 2b rows; phase=done
  REPORT experiment block ending ">>> ENTRY <id>"; CHANGELOG row; save; if --max-wall-h exceeded: phase=await_user:relaunch; exit 0
```
`interrupt_reason(xid)` (called inside every poll tick of train AND eval, inside `_wait_isaac_clear`, and before each
stage/row/`Popen`): STOP exists, or the SIGTERM flag is set, or `ingest_overrides()` (run inside the tick) put `xid` in the
derived squash set, or `user_choice[xid.candidate].choice == skip` — the running experiment's overlay is frozen in its header,
so a mid-stage recipe recompute is harmless. The SIGTERM handler only sets the flag and sends SIGTERM to `rp.CURRENT_CHILD`;
the seams' after-loop check classifies the exit as `interrupted` (never `infra`, so no `_r2` relaunch during shutdown) and
the main loop performs save + `await_user:stop` + exit when control returns, inside the 360-s budget (tick ≤ 120 s + grace
≤ 120 s + save), so `systemctl --user stop` (`KillMode=mixed`, `TimeoutStopSec=360`) is a clean emergency path; the normal
stop is `ledger_cli stop`.
**Infra:** an eval that produced nothing → read (a) the unit cgroup's `memory.events` `oom_kill` delta (cgroup path derived
from `/proc/self/cgroup`), (b) the child's return code (`eval_rc == −9`), (c) the newest `kit_*.log` under the Kit logs dir
for `omni.datastore.*migrated`, and (d) the eval log tail for `Traceback`; on an OOM/cache signature → **stop retrying
immediately**, NOTIFY `INFRA` with the signature, `halt:<xid> eval_infra` (the assistant quarantines the cache and relaunches;
the experiment resumes at the eval); otherwise — if the interrupt flag is clear — `_wait_isaac_clear()` + 10 min in 30-s ticks
that each check `interrupt_reason(xid)` (a stop during the wait → `interrupted`, no retry) and ONE retry with `max_attempts=1`,
then the same halt. `_wait_isaac_clear()` before every train/eval; the launcher refuses to start while another `run_phases.py`/
`run_candidates.py` is alive.

**Ledger entry:** `{id, kind: resolution|measurement|baseline|candidate|seed2|twin|batched_s2|milestone|milestone_s2,
candidate, experiment, seed2_experiment, question, options ([ADOPT, REJECT, SKIP] for candidates; [PASS, WATCH, FAIL] for
milestones — WATCH legal only at M1 or on a squashed previous head; `chosen ∈ options` pinned except for the milestone `open`
status), chosen, alternative, promoted_head {2a, 2b: ckpt + eval run_dirs} (milestones), base_code_sha (milestone pairs use
`R_{k−1}`'s newest s3/s2 screens under any `code_sha`, recorded here),
confidence, pooled_confidence, status (Vocabulary), evidence {seed3 {deltas, z_F, z_T, classes, resolution_id, level}, seed2,
pooled, replay, floor, floor_stage, retention, anchored_drift}, depends_on, conditional_refs, base_recipe_id,
recipe_id_if_adopted, code_sha, tools_sha, decided_at, decided_by, override {id, by: user|driver|milestone, choice, why, at,
applied_at}, frozen, history[]}`. `overruled(driver)` below = status `overruled` with `override.by ∈ {driver, milestone}`.
**Transition table** (source → target by actor): `speculative` → `confirmed` (user) | `overruled` (user, frozen) | `final`
(pair, same pick) | `overruled(driver)` (pair, different pick) | `squashed`; `decided` → `confirmed` (user) | `overruled` (user)
| `final` (batched pair agrees) | `overruled(driver)` (batched pair contradicts, or milestone FAIL) | `squashed`; `final` →
`confirmed` (user) | `overruled` (user) | `overruled(driver)` (milestone FAIL) | `squashed`; `confirmed` → `overruled` (user
only) | `squashed`; `overruled(driver)` → `confirmed` (user `confirm` freezes the entry's `chosen`, i.e. reverts the driver's
override + resurrection) | `overruled(user)` (the user freezes the driver's pick or picks anew) | `squashed`; `overruled(user)`
→ `overruled` (user, new choice) | `squashed` (transitively only — its base hash changed by an earlier override or a
milestone drop; named `frozen:` in the SQUASH/MILESTONE line); `skipped` → `squashed` (referent flipped, or the user's `decide <Cid> --choose run`,
recorded `override.by: user`); `await` (milestone) → `accepted_i` (user `resume-milestone --accept` on a (i) FAIL: every
unfrozen adoption in the block, `decided` or `final`, → `confirmed` (user, frozen; `override.by: user, why: M<k> accept`),
then the milestone proceeds to (ii)) → `final` (after `M<k>.promo` PASS/WATCH) | `await` ((ii) FAIL after `accepted_i`: the
sub-status is cleared, `evidence.stage: ii` recorded, the sticky row applies) | `final` (user `--accept` on a (ii) FAIL: PASS,
the last promoted head is the next reference) | `squashed` (derived, after any recipe-changing override of a block member);
milestone `open` → `final` | `await` | `squashed`; `decided` → `confirmed` and `final` → `confirmed` (user `resume-milestone
--accept` on a (i) FAIL); any effective candidate entry → `squashed` (user `decide --choose run`); `squashed` → re-decided by
a new experiment. **The derived squash view is global, self-contained and recomputed after EVERY ledger write** (decision
or override): an experiment or SKIP entry is squashed when (a) its `base_recipe_id` differs from the recipe composed of the
effective choices of the candidates ranked before it, or (b) it is a SKIP entry written by the conditional rule whose
candidate's `conditional`, re-evaluated against the current effective ledger, now HOLDS (the referent flipped; a SKIP written
by a user `skip`, `evidence: user`, is never squashed by (b)), or (c) — closed transitively — it is a SKIP entry whose
`conditional_refs` include a candidate squashed in this pass (a decision that ran under a then-true predicate is a valid
screen of its base and is governed by (a) alone), or (d) its candidate's newest `user_choice` is `skip` with `at` after the
header's `started`, or (e) it is a USER squash: the entry carries `override {by: user, why: decide run}`, or it is the own
seed-2 run (`ledger[D].seed2_experiment == xid`, written with the s2 header) of a decision that — while it was `speculative`,
i.e. before any pooled evidence for it existed — was `confirmed`, `overruled` by the user, or squashed by a `decide run`; the
finished seed-2 run of a `final` decision is never squashed by (e), whatever the user later does to that decision — (d) and
(e) are `squashed_by: user_choice`, stored in the ledger and NEVER resurrected by a recompute (a later `decide run`
re-screens at s3, it never revives an old s2);
resurrection under SQUASH (2) applies only to entries squashed by (a)–(c). The `D_k`-relative SQUASH paragraph is the same
rule applied incrementally, so an out-of-rank re-run whose decision changes the recipe squashes later screens and
milestones exactly like an overrule (the accepted mispredict cost). A SQUASH or MILESTONE line that squashes a
`confirmed` or `overruled(user)` entry names it with `frozen:` so the user can re-confirm after the re-screen (the re-run's
DECISION line quotes the old frozen choice as `prior_user_choice`). Unknown id raises; uuid idempotent.

**SQUASH (decision 7), precisely:** trigger = an override changes decision `D_k`'s effective choice (user, driver pair,
milestone). (1) recompute the recipe and, for every later experiment and SKIP entry, the base recipe it would now have; (2)
squash set = experiments and SKIP entries whose `base_recipe_id` differs from the recomputed base, plus milestones built on a
changed recipe, plus every SKIP entry whose predicate now holds, plus — closed transitively until no change — every SKIP
whose `conditional_refs` include a candidate squashed in this pass, plus every running experiment whose candidate's newest `user_choice` is
`skip` (`squashed_by: user_choice`); never `D_k`'s own screen; `D_k`'s own seed-2 run only on a user `confirm`/`overrule`/
`decide run` of a `speculative` `D_k` (never on a driver or milestone override); never other candidates' seed-2/twin runs
whose base is unchanged; a squashed experiment whose base hash matches again is resurrected — unless it was squashed by a
user action, derived-view causes (d)/(e) — if its `ckpt`
and eval `run_dir` exist; (3) headers move to `state.ladder.squashed[xid]` with `squashed_by`/`reason`; stage records stay in
`keys`; run dirs and eval dirs stay on disk (tagged by `run_name`; the driver never deletes); dependent decisions → `status:
squashed`, candidate undecided again; REPORT `## SQUASHED — <override> — <stamp>` block; CHANGELOG row; NOTIFY `SQUASH`; (4)
resume = the derived queue under the new recipe; (5) late overrule with finished dependents: same algorithm (the accepted
mispredict cost), reruns get new ids with `supersedes`; overrule while a dependent trains: interrupted within 120 s; while
`D_k`'s own seed-2 trains: continues under a driver/milestone override, interrupted and squashed under a user
`confirm`/`overrule`/`decide run`; (6) crash-safe: the squash set is a derived view recomputed from the ledger on every
launch; `applied_at` is written after `state.json` is saved.

**Notification contract** (`notify.log` line = `%Y-%m-%d %H:%M TAG <id> | field | ...`; tags `RESOLUTION, DECISION, CONFIRM,
SELF-OVERRULE, OVERRIDE-APPLIED, SQUASH, MILESTONE, RETENTION, INFRA, HALT, DONE, RELAUNCH, AWAIT`). The decision line,
produced by `ledger_cli.format_notify` and parsed by `ledger_cli.parse_notify` (round-trip tested; one physical line, absolute paths):
```
DECISION D08 | X09 C2-yaw-progress {KRABBY_YAW_PROGRESS_W=3.0} on recipe 9f3a2c1e0b77 s3 | chosen ADOPT (clear, decided) alt REJECT
 | F z=+1.8 (R 0.9) BETTER: closed l90 0.55->0.72, r90 0.90->0.91, mt 0.20->0.24; progress l90 0.079->0.110, r90 0.071->0.090, mt 0.031->0.040
 | T z=+0.3 (R 0.8) TIE: fwd 0.603->0.58, lat_l 0.47->0.49, diag 0.51->0.50, back 0.603->0.61 | guards TIE: completion 0.50->0.52, tripod_dir 0.38->0.40, turn_surv l90 0.96->0.97, r90 0.99->0.98
 | anchored drift OK | resolution D00b @95%/99% | WATCH: track 0.645->0.60, track_move 0.71->0.72, move_turn dir_ratio 0.49->0.31, move_turn tripod 0.48->0.36, settle l90 None->9.8
 | next: C3a (s2 batched at M1) | confirm: $PY $TOOLS/ledger_cli.py --campaign-dir $LAD confirm D08 | overrule: $PY $TOOLS/ledger_cli.py --campaign-dir $LAD overrule D08 --choose REJECT --why "..."
```
A SKIP entry emits `DECISION <Did> | <Cid> | chosen SKIP (predicate: ...) | run it: ... decide <Cid> --choose run`. The
assistant's persistent Monitor tails `notify.log`, relays every new line (all thirteen tags) as a push notification + a chat
card, and on the user's word runs the quoted command verbatim; the driver answers `OVERRIDE-APPLIED <uuid> D08 -> REJECT |
squashed: X12 (2.1 h) | resume: C3a under recipe 3b9e…` within one poll. `ledger_cli` verbs: `confirm`, `overrule --choose
--why`, `decide <Cid> --choose run|skip --why` (the CLI only appends the choice line; on ingest the driver squashes the
candidate's effective entry — a SKIP or a decision — for `run` (`override.by: user, why: decide run`) and sets `user_choice`; `run` on an awaited `user_decision`/deferred candidate sets `user_choice` so the stop
is not re-raised — a deferred one still stops at `await_user:<Cid> code` until its knob is exported; `skip` makes the driver
write a SKIP entry with `evidence: user` and interrupt that candidate's in-flight experiment), `resume-milestone <Mid>
--accept|--drop <Cid>` (`--drop` is refused unless `Cid` is an ADOPT in `M<k>`'s block, so it always changes the recipe and
re-queues the milestone), `pending`, `show`, `stop`/`unstop`; it prints "driver not running — applied at next launch" when
`state.ladder.unit` is inactive. Every REPORT block ends with `>>> ENTRY <id>`.

**Launcher `launch_candidates.sh`:** `systemd-run --user --unit cands-<campaign>-<epoch> -p OOMPolicy=continue -p MemoryMax=48G
-p MemorySwapMax=0 -p KillMode=mixed -p TimeoutStopSec=360 --working-directory $R $PY $TOOLS/run_candidates.py --campaign-dir
$LAD --unit cands-<campaign>-<epoch> "$@"` — no `timeout` wrapper; the same 10-min Isaac-clear wait; prints the unit name.
`heartbeat_phases.sh <unit> $LAD 1800` unchanged.

**Tests (`tests/unit/test_crab_hex_omni_candidates.py`, pure, loads the tools by path; fixtures like `_agg`):** `decide`
(clear adopt / clear reject on a guard / on a block; all-TIE → uncertain with the speculative sign and `speculative_default`;
boundary = TIE; the null construction in `[0.85, 0.98]` with the per-block marginal 0.95 ± 0.03 (decision 6); `B0_s2 vs B0_s3`
all-TIE; `settle`/`track_ratio_tail` never scored;
MISSING rules incl. "a WORSE still rejects"; `absolute_units`; anchored drift forces REJECT on the speculative-ADOPT path;
`tripod_dir` and the conditional turn survivals), `decide_pooled` (the terminal outcomes incl. `disputed` → self-overrule,
`R_pooled`, the milestone pair, twin reuse, anchor unchanged by a pooled-uncertain final), `compose_recipe` (order, `pool_edit`
composition, `overlay_by_chain`, conflicts, overruled/squashed excluded, stable hash, chain in identity, anchor, `eval_carry`
union, `code_sha`), `squash_set` (late overrule; the overruled decision's finished seed-2 record kept; a pending/in-flight
seed-2 kept under a driver/milestone override and squashed under a user confirm/overrule/decide run; the base twin kept in
both; unchanged-base runs kept; milestone squashed; SKIP entries by base hash, by predicate re-evaluation and by transitive
refs — the two C1/C1b/C7 sequences end with C7 undecided; `C1 REJECT → C1b screened → C1 overruled to ADOPT → C1b squashed by
base hash, not by refs`; last decision → empty; revert → resurrection with files / rerun without, never for (d)/(e)
squashes; `decide C5 run` issued while the driver is stopped → next launch ingests it → C5's old entry stays squashed after
the recompute, C5 undecided and re-screened before the milestone; a further relaunch does not resurrect it; `confirm D(C5)`
while X(C5 s2) is in flight or halted with `inflight` set → X squashed_by user_choice, not resumed, next_item → C6; derived
view identical after a simulated crash), ledger transitions (the full table incl.
`await` rows and `final → confirmed`; `chosen ∈ options`), `user_choice` paths (`C3a clear REJECT → C3b SKIP → decide C3b run →
C3b next and not re-skipped`; `await_user:C12 → decide C12 run (knob absent) → await_user:C12 code; knob exported → B<k> → C12`),
recipe head lookup ('after B0 s2 the head is X01's ckpt'), `code_sha` (`$LAD`/README commit → unchanged, no `B<k>`; a commit
under `mdp/` → changed → `B<k>`; dirty tree refused), `stage_env` never emits a preset/plant/other-task variable, validator rules
(each refusal, deferred at dispatch, whitelist pins), `format_notify`/`parse_notify` (incl. the SKIP line), `resolution`
determinism, cross-head pairing, ring proxy, provenance rule with defaulted keys, `R_pooled`, D00c lookup ordering
(`C1 clear ADOPT → C2 uncertain → twin (B0+C1 s2) writes D00c → C2's pooled decision cites D00c`), `next_item` ordering
(`B0 s3 → B0 s2 → C1 → uncertain D(C1) → C1 s2 (B0's twin exists) → C1b/C1c by conditional → …`; `C4 clear ADOPT → C5 uncertain →
(R_1+C4) twin → C5 s2`; `C2 clear ADOPT → M1.s2 → C4 uncertain → no twin (R_1 s2 exists) → C4 s2`; `C3a clear REJECT → C3b SKIP →
M1 eligible`; `C2 overruled(user) → M1 eligible once C3b is TERMINAL`; `M1.s2 → (i) agree → M1.promo`; `M1.s2 → (i)
contradiction → C2 overruled(milestone), M1.s2 + C3a/C3b squashed → C3a/C3b re-screened → (M1.s2 only if the resulting R_1 has
no ok s2) → (i) → M1.promo`; `R_k s2 pre-existing → (i) runs with no M<k>.s2 before M<k>.promo`; `(i) FAIL twice → await →
--accept → every unfrozen block adoption (decided C3a AND pair-resolved final C2) confirmed → accepted_i → M1.promo; a (ii)
FAIL then → await, nothing dropped, no re-screen`; `C5 decided REJECT, C6 clear ADOPT, M2 final → decide C5 run → C5 re-runs
on R(≤ C4) with that recipe's ok s3 screen as control and is not self-squashed; ADOPT → C6's, C7's, M2's and later screens
squashed by the global view and re-screened (M2 not eligible until they land), the re-run's screen is the recipe head;
REJECT → nothing squashed`; `ring row with a WORSE turn_survival_r90 counts as a WORSE row; a ring row with surv[back] = 0 has
both turn survivals MISSING and is not WORSE on them`; `B<k> after a code change → D03 unchanged, candidates still
cite D03`; `_run_eval returning (None, None, info) leaves rp.LAST_EVAL_INFO == info with rc`; `SIGTERM during the replay →
replay: interrupted, no experiment header, top-of-loop STOP check`;
`C1–C3b all REJECT/SKIP → R_1 = B0 → M1: no M1.s2 (B0 s2 exists), (i) WATCH, M1.promo of B0 → chosen WATCH, final; M2 with C4
adopted compares vs M1's heads`; `C4–C7 all REJECT → M2 carried forward from M1, no GPU → C8 next; M3 unchanged → M3.final_s2
→ done`; `recipe_head_s2 resolves to X02's (B0 s2) final ckpt, after M1.s2 to that run's`; `repeat halt
on C5's s2 → HALT names D(C5); decide C5 skip refused; confirm D(C5) → pair removed from pending`; `decide C5 skip while X07 trains → X07 interrupted and
squashed → next_item C6; a relaunch does not resume X07`; `all-TIE, z_F + z_T < 0, speculative_default: candidate → speculative
ADOPT, pair queued`; `clear blocks + drift FAIL → speculative REJECT, confidence uncertain, pair queued; pooled drift OK +
pooled clear ADOPT → self-overrule to ADOPT`;
sticky milestone `await`; recipe-baseline rerun after an overrule and after a `code_sha` change; `await_user` on C12 and on a
deferred knob), arming per kind, the resume branch with a finished run dir and with a seed-2 header, `_run_eval(interrupt_check=…)`
returns interrupted with no retry, `run_phases` signature/carry/defaults-injection/`name_suffix`/`max_attempts` tests,
`exposure_from_log` synthetic-log test + `OMNI_EXTRA_KEYS ⊂ EXPO_KEYS`, layout PKGS + self-relative pins.

## Part C — the campaign: candidates, chains, milestones, budget

**Step 0 — zero-GPU + 4 min:** `resolution.py` → `resolution.json`/`D00` per decision 6. `D01` = the turn-rate evidence.
Replay (advisory) runs immediately before the dispatch of every candidate that declares `replay_args`.

**Chains.** `scratch_1c` = one stage `{base 1c, from_scratch, 8000 it, eval [slow], floor dead_walker (candidates only)}`.
`scratch_c6` = one stage `{base 1a, from_scratch, 8000 it, overlay = c6 pool + DIR_BAND="" + SLOTS_OMNI + RSI all + FACING
0.2:0.6 + YAW_W 0.2, eval [slow], floor dead_walker}` (C1c). `curriculum` (overlays on the `1a` preset: `KRABBY_OMNI_CMD_POOL`,
`KRABBY_OMNI_DIR_BAND=""` (⇒ None), `KRABBY_OMNI_SLOTS` = the pool's projection, `KRABBY_OMNI_RSI_SOURCE`,
`KRABBY_OMNI_FACING=0.2:0.6`, `KRABBY_YAW_W=0.2` in EVERY stage — the economy is uniform along the chain; consecutive stages
differ only in the pool (+ RSI source at c1→c2)):

| stage | it | RSI | pool (`name:weight`) | slot projection | eval | floor |
|---|---|---|---|---|---|---|
| c1 | 1000 | 0 | `stand:.20,fwd:.80` | 0.20:0.80:0:0 | curriculum | survival through fwd < 0.30 |
| c2 | 1000 | all | `stand:.20,fwd:.30,side_l:.20,side_r:.20,back:.10` | 0.20:0.80:0:0 | curriculum | survival through back < 0.30 |
| c3 | 1000 | all | `stand:.15,fwd:.25,side_l:.16,side_r:.16,back:.08,turn:.20` | 0.15:0.65:0.20:0 | — | — |
| c4 | 1500 | all | `stand:.15,fwd:.22,side_l:.14,side_r:.14,back:.10,turn:.15,fwd_turn:.05,side_l_turn:.025,side_r_turn:.025` | 0.15:0.60:0.15:0.10 | curriculum | survival through move_turn_r < 0.30 |
| c5 | 1500 | all | `stand:.15,fwd:.15,side_l:.10,side_r:.10,back:.10,omni:.15,turn:.10,fwd_turn:.07,side_l_turn:.04,side_r_turn:.04` | 0.15:0.60:0.10:0.15 | — | — |
| c6 | 2000 | all | `stand:.15,fwd:.14,side_l:.09,side_r:.09,back:.09,omni:.14,turn:.10,fwd_turn:.06,side_l_turn:.035,side_r_turn:.035,omni_turn:.07` | **0.15:0.55:0.10:0.20 = SLOTS_OMNI** | slow + curriculum | — (final stage) |

Total 8000 iterations (= B0); four curriculum rows ≈ 15 min. Backward is in the pool from c2 (user amendment): it has no RSI
seed and relies on the fore-aft mirror image of forward; its eval hold `back` sits right after `lat_r` in the prefix-ordered
row so c2's floor and RETENTION read it. Every enabled type keeps ≥ `POOL_FLOOR_SHARE` 0.025 once enabled (pinned by a test
on the yaml); move+turn ramps 0.10 → 0.15 → 0.20; c6's regime shares equal the incumbent's; every row sums to 1.00 and
projects to the stated slot mix (re-derived). RETENTION (WATCH + `RETENTION` notify) at c2/c4/c6 vs the incumbent's stage
trace. Teacher pool for milestones: `rescale_to(SLOTS_TEACHER)` of c6 (round-trip pinned). Confound recorded in C1's
`question`: c4–c6 use world-held move+turn at the 0.3 rad/s cap (infeasible, decision 2); `C1b` is the reduced-demand
re-screen and `C1c` the attribution arm.

**Ranked candidates** (each screened on the effective recipe, seed 3, 8000 it; control = the recipe's own screen):

| # | id | change | mechanism | code | replay | conditional / notes |
|---|---|---|---|---|---|---|
| B0 | baseline | `scratch_1c` 8000 it, seeds 3 then 2 | — | — | — | never stops on a gate; 1c-004 (5000 it) prior row; `D03` records B0's its-800–1000 mean → candidate `dead_walker` floor |
| C1 | curriculum | chain `curriculum` (table above); `teacher_overlay: {}` (only the rescaled pool reaches 2a/2b) | curriculum (4) | A1 | none | user's #1 |
| C1b | curriculum, reduced-demand turns | chain `curriculum` + `pool_edit {ang_scale: {"*_turn": 0.3}}` (0.09 rad/s cap on `*_turn` rows — still above the walking turn bound; tests whether a 70 % smaller demand changes the trade) | curriculum + kinematics | A1 | none | `conditional: decision(C1).chosen == REJECT and (decision(C1).floor_stage is None or decision(C1).floor_stage >= c4)` |
| C1c | pool at 8000, single stage | chain `scratch_c6`; `teacher_overlay: {}` | attribution | A1 | none | `conditional: decision(C1).chosen == ADOPT` (evaluated after C1's pair); `speculative_default: candidate` (Occam: on all-TIE the simpler recipe is the speculative pick; the pair still runs); `dead_walker` → REJECT of the arm; its chain's seed resolution is `D00c-scratch_c6` |
| C2 | yaw-progress 3.0 | `KRABBY_YAW_PROGRESS_W=3.0` | price (1) | none | `--yaw-progress-w 3` | reachable prize 0.206 → 0.366/s |
| C3a | cap 0.21 | `ANG_VEL_MAX=0.21, CLOCK_TURN_R=1.72` | kinematics (2) | none | none | absolute units; cadence preserved (0.361); cuts the kernel's marginal turn incentive ≈ 36 % (needs C2 first); clip stays 0.2 so the P-control's linear region is 0.133–0.14 rad (near bang-bang) — C3b is the paired variant |
| C3b | cap 0.18 paired | `ANG_VEL_MAX=0.18, ANG_VEL_CLIP=0.12, CLOCK_TURN_R=2.0` | kinematics (2) | A1 knob | none | `conditional: decision(C3a).chosen != REJECT or decision(C3a).pooled_confidence in (uncertain, disputed)` (a clear s3 REJECT of C3a — no pair — skips C3b); dead band 0.08 rad |
| C4 | turn-slot facing bonus | `KRABBY_OMNI_FACING_TURN_W=0.8` | price (1) | A2 | `--facing-turn-w 0.8` | settle bonus on turn rows 0.12 → 0.60/s (weight 0.2 → 1.0 on the settled-0.1-rad vs 1.2-rad span); 0 on translation rows |
| C5 | facing obs sincos | `KRABBY_OMNI_FACING_OBS_ENCODING=sincos` | observation (3) | A2 | none | `eval_carry`; from scratch (fits the screen) |
| C6 | coupled tracking | `KRABBY_OMNI_TRACK_FACING_SIGMA=1.0` | coupling (1) | A2 | `--coupled-sigma 1.0` | ignoring facing now costs translation; after the cheaper fixes |
| C7 | reduced-demand move+turn types | `pool_edit {ang_scale: {"*_turn": 0.3}}` (cap/clip = 0.3 × the recipe's) | kinematics (2) | A1 | none | `conditional: recipe.has_pool and decision(C1b).chosen != ADOPT`; training distribution only |
| C8 | stand share 0.05 | `overlay_by_chain`: curriculum/scratch_c6 → `pool_edit {set_share: {stand: 0.05}, renormalise: true}`; scratch_1c → `KRABBY_OMNI_SLOTS=0.05:0.65:0.10:0.20`; on 2a/2b (both chains) the same edit on `SLOTS_TEACHER` → `0.05:0.55:0.15:0.25` (`teacher_overlay` explicit) | economy (5) | none | none | |
| C9 | progress weight 1.5 | `KRABBY_OMNI_PROGRESS_W=1.5` | economy (lit rec 3) | A1 knob | `--progress-w 1.5` | raises the translation stake → late |
| C10 | RSI turn frames | bank + source tag 2 harvested from the recipe head's turn holds | seeding (lit rec 6) | **deferred** (harvester `--mode turn`, staged-turn tuple, staged-slot rule, tests) | none | `conditional: screen.yaw_progress_turn_min < 0.10`; reached → `await_user:C10 code` |
| C11 | ang-turn-only | `KRABBY_OMNI_ANG_TURN_ONLY=1` | economy (5) | none | `--new-ang-turn-only` | `conditional: screen.track_ratio_move_tail < 0.45` (the creeper's ≈ 0.25 vs a walker's ≈ 0.6 on move-only rows); erases the settle bonus otherwise |
| C12 | body-locked travel | `travel_frame=body` for `*_turn` types | task definition | **deferred** (cfg field + knob + `EVAL_CARRY` + recap tests) | none | `user_decision: true` → `await_user:C12`; answered by `ledger_cli decide` |

**Milestones** `M1` after C3b, `M2` after C7, `M3` (final) after C12 (`after` = every earlier candidate TERMINAL; a deferred
candidate ends `skipped` or decided like any other). A milestone is (i) a zero-GPU pooled decision preceded, when needed, by
one ordinary experiment `M<k>.s2`, then (ii) one ordinary experiment `M<k>.promo`, and for M3 (iii) `M3.final_s2` — each
experiment with its own resume/interrupt/squash (Part B next_item). One automatic rerun per eligibility, SHARED across (i)
and (ii) (counted by DECISION entries carrying `override {by: milestone, why: M<k>}` newer than the last user override on any
block member; a second milestone override of any kind within one eligibility → FAIL/await); eligibility restarts on a user
`--drop`. Procedure: (i) **batched seed-2 confirmation** — `M<k>.s2` runs whenever `R_k` has no ok s2 screen; the pooled
decision runs only when `decided` adoptions are pending since the last milestone (otherwise the pair is recorded as WATCH).
The pair is candidate = the effective recipe `R_k` (s3 = its ok s3 screen; s2 = `M<k>.s2`, or the recipe's existing ok s2
screen if a twin already exists — no duplicate) vs base = the previous milestone's recipe `R_{k−1}` (`B0` at M1; its s2 exists
by construction); `decide_pooled` on that pair (D00c lookup first) BEFORE any promotion: agree or pooled-uncertain-same-pick →
every `decided` adoption in the block → `final`; contradiction → driver overrule of the block's LAST unfrozen `decided`
adoption via SQUASH (`override.by: milestone, why: Mk`; `M<k>.s2` and every block member screened after the dropped adoption
go with it and are re-screened on the reduced recipe) → (i) re-runs ONCE on the reduced recipe `R_k'` against the same
`R_{k−1}` (a new `M<k>.s2`, 3.7–3.9 h, only when `R_k'` has no ok s2 screen — none when `R_k'` is `R_{k−1}` or already
twinned): agree → the remaining `decided` adoptions → `final` and `R_k'` now has its s2; a second
contradiction → `chosen: FAIL`, status `await` (below); frozen decisions → WATCH + notify. (ii) **promotion** `M<k>.promo` of
the recipe head through **2a then 2b** with the recipe's 2a/2b overlays (Recipe paragraph: the rescaled pool with adopted
share edits re-applied, every `stages: all` overlay, `eval_carry` keys, and each candidate's `teacher_overlay{}`; the 2a/2b
facing ramp capped at `8 s × ang_vel_max` if a cap candidate was adopted) — emitted only after (i) is resolved, and skipped when the `recipe_id`
equals the previous milestone's (the entry carries forward as `chosen: PASS`, `evidence.carried_from: M<k−1>`; the rolling
lookup follows `carried_from` to the last entry with a promotion); `EVAL_FULL` at both heads, every row re-summarised from
its aggregate JSON (the ring row with the proxy `facing_closed`) — `M1.promo` is NEVER omitted (there is no previous
milestone; whatever `R_1` is, `B0` included, its heads are M2's rolling reference); at M1 **WATCH-only** vs the seed-3 2b record
and vs `outputs/omni_plan/reeval_2a` for the 2a head (`chosen: WATCH`, status `final`); from M2 a rolling equal-compute
comparison vs the previous milestone's own heads (looked up by its milestone entry, following `carried_from`; if that head is
squashed → WATCH-only, the MILESTONE line names why): **FAIL = ≥ 2 of the three 2b rows WORSE vs the previous milestone's 2b
head** (a row is WORSE when either block or any guard is WORSE; per row, blocks at `R_F = max(R_F^eval(row), R_F^seed(chain))`
— the standardised block seed term is unitless and carried across rows — and guards at the row's own `R_eval,i` with the
slow-row `R_seed,i` borrowed; ring completion is WATCH (decision 6); every other guard — including the two conditional turn
survivals, which the ring row does define — is scored on every row; a guard that is MISSING on a row (zero or unavailable
denominator, decision 5) is never WORSE on that row); **the three 2a rows are WATCH at every milestone** (vs the previous milestone's 2a head, named in the
MILESTONE line); PASS otherwise (`chosen: PASS`, `final`); FAIL → the same driver
overrule of the block's last unfrozen adoption (SQUASH squashes `M<k>.promo` and the block members screened after the dropped
adoption) → the milestone re-runs once (4.3 h, + 3.9 h when the reduced recipe has no s2, + ≤ 3.9 h per re-screened member); a
second FAIL, or a FAIL with nothing unfrozen to drop → `chosen: FAIL`, status `await`, `await_user:M<k>` (sticky across
relaunches). **User answers:** `resume-milestone --accept` after a (i) FAIL moves every unfrozen adoption in the block —
`decided` or pair-resolved `final` — to `confirmed` (user, frozen; `override.by: user, why: M<k> accept` — the user has weighed
the s2 contradiction against the whole recipe `R_k`), sets the sub-status `accepted_i`, and next_item emits `M<k>.promo` for
the same eligibility (the entry becomes `final` after it); a later (ii) FAIL in that eligibility therefore has nothing
unfrozen to drop and goes straight to `await` (sub-status cleared); `--accept` after a
(ii) FAIL makes the entry PASS/`final` and the last promoted head the next reference; `--drop <Cid>` is an ordinary override
on `Cid` (the milestone re-queues). (iii) **M3 only:** after M3's `final` entry (PASS or WATCH) the
final recipe's seed-2 head runs 2a → 2b as `M3.final_s2` (from `recipe_head_s2`; ≈ 4.7 h with evals); `phase=done` follows
it; the user names the head of record from both seeds' 2b rows.

**Seed policy:** screens on seed 3. Uncertain → the base recipe's seed-2 twin (if none) then the candidate's seed 2, as the
next GPU windows; the pooled decision is terminal (decision 7). Clear ADOPT → batched at the milestone. Clear REJECT → none.
A contradicting batched pair on a frozen decision (Part C (i)) is WATCH, never a flip; a user `confirm`/`overrule` of a
`speculative` decision drops that candidate's own pending seed-2 run (decision 7).

**Budget** (0.43 h/1000 it; 2a 2.4 h, 2b 1.9 h; 3.3–3.9 min per eval row; a screen/twin/s2 run is 3.7 h on `scratch_1c` and
3.9 h on `curriculum`): B0 s3 + s2 7.4 h; C1 3.9 h; C2, C3a, C3b, C4–C9, C11 (10 screens, C3b/C7/C11 conditional) ≈ 37–39 h;
pairs (one twin per clear-ADOPT recipe that meets an uncertain candidate, ≈ 2–4, + one candidate s2 per uncertain decision,
≈ 4) ≈ 22–31 h; M1 (batched s2 3.7–3.9 + promotion 4.3 + evals 0.4) ≈ 8.6 h; M2 ≈ 8.6 h; M3 ≈ 8.6 h + `M3.final_s2` 4.7 h;
**core ≈ 101–112 h**; zero-incident extras: C1b 3.9 h, C1c 3.7 h, C10/C12 screens 3.7–3.9 h each + their control reruns
(≈ 101–135 h ≈ 4.2–5.6 GPU-days before incident reruns); incident reruns: a milestone (i) contradiction adds ≤ 3.9 h (only
when the reduced recipe lacks an s2) plus ≤ 3.9 h per block member screened after the dropped adoption (+ 3.7–3.9 h for its
seed 2 when the re-screen is uncertain), a (ii) FAIL adds 4.3 h (+ 3.9 h when the reduced recipe has no s2) plus the same
per re-screened member — ≤ 1 automatic rerun per eligibility; each user overrule discards and reruns ≤ 3.9 h per squashed run; an out-of-rank re-run whose rank-composed base has no ok s3 screen adds a 3.7–3.9 h `B<k>`; each uncertain
candidate screened after a code landing adds a 3.7–3.9 h twin; replay wall time ≈ 10 min each (CPU) — one long-lived unit, no
cap (user decision); the CHARTER carries that sentence.

## Part D — records, docs, memory

- `$LAD/CHARTER.md`: question, the five mechanisms with numbers, the vocabulary, the decision rule and resolution policy
  verbatim, the ranked list with rationale and confounds, the speculation/squash protocol, "how to confirm/overrule/decide/
  resume" (the commands; `systemctl stop` is the emergency path), held fixed (plant A15+B, Revision-4 economy unless a
  candidate says otherwise, seeds 3/2, 8000-it screens, frozen manifest rows, `code_sha`), caveats (below), Log (incl. the
  measured `MemoryPeak` after `B0`).
- `$OMNI/README.md`: the pool sampler (`KRABBY_OMNI_CMD_POOL` grammar, projection rule, `pool_*` metrics, the −1 sentinel,
  `track_ratio_move_mean`), the new knobs, the ladder driver (files, the notify/ledger protocol, launch/heartbeat lines), the
  curriculum manifest row.
- `$CAMP/CHARTER.md` Log: one entry closing the formation campaign at its 2b decision point and pointing at `$LAD`.
- Memory: `omni-task-campaign-active.md` → the ladder's state, unit, how to relay/overrule; `docs/FOLDER_LAYOUT.md` tree line.
- **Revision 5 (after the campaign, separate commit):** bake the final adopted recipe into `crab_hex_omni_phases.py`
  (c-stages as `PHASE_NAMES` if the curriculum was adopted, `OMNI_COMMON` values, the teacher pool) and update the
  `test_crab_hex_omni_phases.py` tables, `test_no_preset_arms_the_staged_lever` if C11 was adopted, the driver's name-keyed
  maps; then the bundle (`bundle.yaml` head pin) — the user's call after M3.

## Execution steps (routed per plan-model-routing; ultracode ON → Workflow dispatch; roster ok; session on fable)

0. Reference captures at the current HEAD: the `1a` 20-it smoke (64 envs, seed 3 → `outputs/omni_plan/smokes_ladder_base`)
   and the three cfg-identity suites (`KRABBY_CFG_DUMP_DIR=outputs/omni_plan/dumps_ladder_base`)
   → sonnet · low — exact commands; Verification 2/3 compare against these dirs
1. **Owns `crab_hex_omni_command_math.py`, `crab_hex_omni_command.py`, `omni_knobs.py`, `omni_parkour_event.py`, `omni_exposure.py`,
   `test_crab_hex_omni_command.py`, `test_crab_hex_omni_phases.py`, `test_crab_hex_omni_exposure.py`:** pool sampler, per-type
   scales with the −1 sentinel, `rescale_to`/render round trip, slot-gated tracking telemetry, ALL new knob rows (default-off;
   the three A2-bound apply functions import their targets lazily, the `ANG_TURN_ONLY` pattern) + the clock rule, and every
   A1 test except the apply tests of the three A2-bound knobs
   → opus · high — cross-module sampler with sign/RNG hazards; the bitwise twins are the catch
2. **Owns `crab_hex_omni_rewards.py`, `crab_hex_omni_observations.py`, `crab_hex_omni_mirror.py`, their three test modules,
   `tests/integration/test_crab_hex_omni_configs.py`, plus the apply tests of `FACING_TURN_W`/`TRACK_FACING_SIGMA`/
   `FACING_OBS_ENCODING` in `test_crab_hex_omni_phases.py`:** the reward twins, the encoding, `head_signs(encoding)`, the
   37-name edits, `CMD_FIELDS_ADDED`/`EXPECTED_RL` (binds to step 1's knob names)
   → opus · high — reward/obs/mirror semantics; a wrong sign is silent until a window is wasted
3. **Owns `run_phases.py`, `$GE/metrics.py`, `$GE/report.py`, `test_crab_hex_omni_gate.py`, the world-row facing test:**
   A3 + A4 (seams incl. the `_run_eval` Popen/interrupt seam and `CURRENT_CHILD`, `EVAL_CARRY` + defaults injection, hoisted
   keys, `EXPO_BASE`, atomic save, `set_seed`, `name_suffix`, `max_attempts`, `PYTHONUNBUFFERED`, the `ladder` refusal) + tests
   → opus · high — shared eval code and the driver's subprocess layer; the 0.00-vs-0.39 class of bug lives here
4. **Owns `replay_reward_economy.py`, `scenarios_omni.yaml`, `test_crab_hex_gait_schedule.py`:** A5 + A6
   → sonnet · high — exact specs; tests and a dry run catch mistakes
5. `run_candidates.py`, `ledger_cli.py`, `resolution.py`, `launch_candidates.sh`, `$LAD/candidates.yaml`, the candidates test
   module, layout PKGS entries (needs steps 1–4)
   → opus · xhigh — design settled but the failure modes are quiet (wrong control, dropped overlay, ledger races); reason through
     the squash/resume/next_item/pooled-outcome/milestone edge cases and check them before reporting
6. `$LAD/CHARTER.md`, README/FOLDER_LAYOUT/`$CAMP` log entry, memory note
   → sonnet · medium — written from the diff and this plan
7. Fresh-eyes review of the whole branch diff against this plan (bitwise identity of legacy paths, forward/sideways dumps,
   the decision rule vs decision 5, the pooled outcomes vs decision 7, squash semantics, notify contract, test coverage)
   → fable · high — design-bearing; protects two lineages of record and the campaign's honesty
8. Verification runs (below)
   → sonnet · medium — exact commands; failures loud
9. Commits (Part G)
   → haiku · low — mechanical; messages fixed
10. Step 0 (`resolution.json`/`D00`, `D01`), launch `B0` (s3 then s2), record `MemoryPeak`, arm heartbeat + the persistent
    `notify.log` Monitor, relay every new `notify.log` line (all thirteen tags) as push + chat card, run `ledger_cli` on the
    user's word, quarantine the Kit cache on `eval_infra` and relaunch
    → main (fable) — the stop-and-wait points and the user's overrules live here
11. Milestone stops, the C10/C12 user decisions, the final head naming, Revision 5
    → main (fable) — the decisions are the user's; Revision 5 → opus · high afterwards

Execution order: 0 → 1 → (2 ∥ 3 ∥ 4) → 5 → 6 → 7 → 8 → 9 → 10. Steps 2, 3, 4 touch disjoint files. A step that fails twice
reruns one notch up, never lower.

## Verification (in order; `export OMNI_KIT_ACCEPT_EULA=yes`; unset every `KRABBY_*`; one Isaac process at a time)

1. **Pure suites** after each part: `$PY -m pytest tests/unit -k "crab_hex or travel_frame or parkour_command or exposure or
   policy_of_record or experiment_layout or gait or candidates" -q` — all green, including the edited pins named in Parts A/B.
2. **Config identity:** `RUN_CRAB_HEX_CFG_IDENTITY=1 KRABBY_CFG_DUMP_DIR=$R/parkour/outputs/omni_plan/dumps_ladder $PY -m pytest
   tests/integration/test_crab_hex_phase_configs.py tests/integration/test_crab_hex_sideways_configs.py
   tests/integration/test_crab_hex_omni_configs.py -v`; `cmp` every forward and sideways `cfg_*.json` against
   `dumps_ladder_base` (step 0); the omni dumps differ from `dumps_ladder_base` ONLY by the new default-off fields
   (`cmd_pool: null`, `facing_obs_encoding: raw`, `reward_facing_turn` weight 0.0) — asserted by the updated delta test.
3. **Sampler smokes (20 it, 64 envs, seed 3):** (a) legacy `KRABBY_OMNI_PHASE=1a` twin — `params/env.yaml` identical to
   `smokes_ladder_base` modulo exactly three additions (`cmd_pool: null`, `facing_obs_encoding: raw`, the `reward_facing_turn`
   block at weight 0.0); the printed per-iteration series identical for every label present in the base, plus exactly two
   new labels — `Episode_Reward/reward_facing_turn` (0 on every line) and `Metrics/base_parkour/track_ratio_move_mean` (the
   ledger key is emitted on the legacy path too; printed as 0 until the first finished walking episode — the base event's
   `_update_metrics` writes an empty ledger group as 0.0, exactly like `track_ratio_mean` — then finite); (b) a c2 overlay (`CMD_POOL`, `DIR_BAND=""`, `SLOTS=0.2:0.8:0:0`)
   — `params/env.yaml` shows the pool string and `dir_band: null`; `pool_*_draws` ≈ the weights; the four `*_frac_actual`
   still sum to 1; `track_ratio_move_mean` printed; `error_vel_xy` shows body ±y and −x commands; (c) a `sincos` +
   `FACING_TURN_W` + `TRACK_FACING_SIGMA` + `ang_scale` overlay on `1c` — the 37 reward labels, the swapped func names in
   `params/env.yaml`, finite losses.
4. **Ladder dry-run:** `run_candidates.py --campaign-dir /tmp/…/ladder_dry --candidates $LAD/candidates.yaml --dry-run` prints
   the derived queue (B0 s3, B0 s2, C1, …), every stage env (no preset/plant/other-task variable in an overlay), the validator's
   verdict on each candidate (a deliberately bad one: `ANG_VEL_MAX=0.15` without `CLOCK_TURN_R` → refused; a pool overlay with a
   stale SLOTS → refused; a deferred one → accepted at load, refused at dispatch), the D00 metric list and levels, the
   `code_sha` and its pathspecs; `ledger_cli.py pending/show/confirm/overrule/decide/resume-milestone` on a fixture ledger;
   `heartbeat_phases.sh` against the dry state.
5. **D00:** `resolution.py` on the listed inputs → `resolution.json` with half-widths and levels for every scored metric and
   the standardised block means (plus the absolute-units blocks); a run dir lacking a non-defaulted `EVAL_CARRY` key is refused
   unless `provenance_ok`; the 2b same-head re-eval (4 min GPU) paired against the recorded 2b eval and reported as the process
   floor; the RESOLUTION line names the effective MDE per block/guard.
6. **B0 pre-flight (first two GPU windows):** after B0 s3, `overlay_checks` PASS, `run_meta.env_vars` equal the eval carry
   (defaults included), `$PY $CAMP/decision_table.py --campaign-dir $LAD --phases X01:1c` renders the row, `MemoryPeak`
   recorded, `D03` written from the log; the 1c-004 prior row and B0 within `D00` of each other on T is reported, not gated;
   after B0 s2, `D00b` (per metric and per block) exists, the RESOLUTION line is re-emitted with the block floors, `B0_s2 vs
   B0_s3` decides all-TIE, and the recipe head still resolves to X01's checkpoint.
7. **Speculation drill (before C1 trains far):** `ledger_cli overrule` of a resolution/measurement/baseline entry is refused
   (`show` picks the ids); a synthetic uncertain decision in a copied state dir exercises overrule → squash → resume, confirm →
   frozen, revert → resurrection, a SKIP referent flip, `decide --choose run` on a skipped candidate and on a decided one,
   `decide --choose skip` against a copied in-flight header (squashed, not resumed), `confirm <Did>` against a copied
   in-flight seed-2 header (the s2 squashed_by user_choice and not resumed, the base twin header untouched, next_item moves
   on), and a milestone (i) FAIL → `resume-milestone --accept` → `accepted_i` → promo queued, in `--dry-run`.
8. **Stop drill:** launch the unit on a 20-it `1a` smoke stage in a copied state dir and `systemctl --user stop <unit>` while
   the child runs; expect `phase=await_user:stop`, no `_r2`/retry log, the unit inactive in < 360 s, and the smoke's `inflight`
   header resumed by the next launch.

## Part G — commits (branch `feat/crab-hex-omni-task`; nothing pushed)
1. `feat(omni command): pool sampler KRABBY_OMNI_CMD_POOL with per-type yaw/speed scales; new omni knobs (default-off); clock-radius rule; slot-gated tracking telemetry -- legacy path bitwise`
2. `feat(omni rewards/obs): turn-slot facing bonus, facing-coupled tracking, progress-weight knob, sincos facing encoding with encoding-aware mirror signs`
3. `feat(gait_eval + omni driver): facing_closed metric; run_phases overlay seams (extra_env, interrupt_check for train and eval, run_name, carry_overlay, EVAL_CARRY defaults, atomic saves)`
4. `feat(omni eval): curriculum__omni manifest row; replay per-term weight flags`
5. `feat(omni experiments): the candidate ladder -- run_candidates.py, ledger_cli.py, resolution.py, launcher, candidates.yaml, charter`
6. `docs: omni README (pool, knobs, ladder), FOLDER_LAYOUT, formation charter closing entry`
Messages end with the session's attribution line. Record commits of `$LAD` at every milestone and at the end (they do not
change `code_sha`).

## Caveats carried into the CHARTER
1. `move_turn`/`move_turn_r` are infeasible by construction on this plant (walking at the 0.21 m/s floor leaves ω ≤ 0.01 rad/s
   at r_foot 1.2 m; and the manifest rule `1.0 s + 1.5708/cap ≤ 8 s` fails for any cap < 0.2244 rad/s); they are scored
   relatively (F) and their translation, tripod and the training-log tracking ratios are watched. A candidate that "fixes"
   them will turn more and walk less there.
2. Single-seed screens: seed variance is the dominant uncertainty; the per-block `R^seed` from `D00b`/`D00c` and the pairs are
   the mitigation, not a cure.
3. Milestone rows at M1 compare unequal compute and a different lineage (WATCH only); from M2 the comparison is rolling and equal.
4. RETENTION is never a verdict; the curriculum's first run seeds the stage trace that later runs are compared against.
5. World-held move+turn during c4–c6 is the same infeasible demand as the eval hold; C1b/C7 reduce the demand (0.09 rad/s on
   `*_turn` rows) without making it feasible; C3a/C3b lower the global cap; C12 is the redefinition.
6. `KRABBY_OMNI_ANG_TURN_ONLY` is demoted: it erases the turn-row settle bonus; it runs only if the creeper signature returns
   on move-only rows (`track_ratio_move_tail < 0.45`).
7. The replay screen is advisory and one-hold decidable until B0's raws exist; it never rejects; it cannot price a cap change.
8. A user overrule late in a block discards finished work (hours) — the accepted branch-mispredict cost; records are kept.
9. C10 and C12 are outside the code freeze; landing them changes `code_sha`, re-runs the control and the base twins.
10. Isaac under a 48 G cap: a cgroup kill is itself a cache-corruption risk; every stop path is SIGTERM first; the infra
    detector stops retrying on the first OOM/cache signature and hands the cache to the assistant.

Routing: strong · roster ok · planned on fable·high · 1× haiku·low, 1× sonnet·low, 2× sonnet·medium, 1× sonnet·high, 3× opus·high, 1× opus·xhigh, 1× fable·high, 2× main
Effort enforcement: prompt-level (tier agents not set up — say the word and I'll create `.claude/agents/effort-{low,medium,high,xhigh}.md` for the next session)

## Audit trail
- Pass 1 (3 blocking, ~30 substantial, ~27 minor). Per-metric multiplicity → block rule; `tripod_walking` guard → direction
  holds; milestone PASS/frozen drops → M1 WATCH, rolling from M2, FAIL via SQUASH; B0 s2 as window 2; c1 floor without an eval
  → curriculum row; eval `None` → halt only; retention → WATCH; C1b/C1c; queue order; twin keying; SKIP entries; `pool_edit`;
  `facing_closed` sourcing; `decided`; D00 inputs; −1 sentinel; sincos mirror; `EVAL_CARRY_DEFAULTS`; cadence co-overlay;
  teacher pool; arming; resume; SIGTERM; derived squash; pure whitelist; overrulable set; step ownership; C10/C12 deferred.
- Pass 2 (2 blocking, ~17 substantial, ~27 minor). `track_ratio_tail` guard → WATCH + slot-gated telemetry; `after` omitted
  `overruled` → TERMINAL; pooled outcomes; anchored drift on every ADOPT; twin keyed to `base_recipe_id`; SKIP referents;
  deferred/`decide --choose skip`; standardised deltas; milestone loop; recipe head lookup; `code_sha`; 48 G + unbuffered +
  cgroup/Kit signatures; render precision; null construction; all minors.
- Pass 3 (1 blocking, 8 substantial, ~25 minor; the backward amendment verified consistent). Block-level resolution ignored
  seed variance → `R_F = max(R_F^eval, R_F^seed)` from the B0 pair, `B0_s2 vs B0_s3` all-TIE pinned; C11 read the un-gated ratio
  → `track_ratio_move_tail < 0.45`; SKIP entries carried no base and refs were one level deep → base hash + transitive closure;
  provenance rule collided with the seven-name `EVAL_CARRY` → non-defaulted keys only; `code_sha = HEAD` caught record commits →
  code-bearing tree hash + dirty refusal; `decide --choose run` had no ledger effect → `user_choice`, SKIP squash, RUN option;
  milestone batched pair unnamed → `R_k` vs `R_{k−1}`, twin reuse; D00c written after the pooled decision → lookup first, on
  twins/seed2/batched, cross-chain `max`; SIGTERM during an eval SIGKILLed Kit → `_run_eval` Popen seam + `CURRENT_CHILD`,
  handler only flags, `TimeoutStopSec=360`. Minors (all applied): squashed-previous-head clause split; `await` rows and
  `final → confirmed`; WATCH in milestone options; `disputed` → self-overrule; C1c `speculative_default`; C8 `overlay_by_chain`;
  SKIP NOTIFY line; conditional turn survivals as guards; C3a/C3b replay none; C3b `pooled_confidence`; MISSING wording;
  defaults injection + test edit; five-key exposure test, optional push args, validity rule, `EXPO_BASE`; `R_eval`/`R_seed`
  defined; `ratio_by_hold` defined as WATCH; "reduced-demand" wording + C3a clip note; `skipped → squashed` via `run`; hold
  durations; `decision_table.py` path + `rec.evals` shape; 3.7/3.9 h per run; rerun allowance + `--drop` refusal; resume
  `set_seed`; tick-level ingest; `−9` return code + cgroup path; (iii) inside M3; `chosen ∈ options`; D numbering; post-landing
  twin reruns; `exposure_from_log` test + cross-pin.
- Pass 4 (0 blocking, 8 substantial, ~27 minor). One standardiser (`R_i = R_eval,i`) everywhere and a block seed term that
  cannot cancel (`max(|z_F(pair)|, rms/√k)`), `D00b` cited by every chain until its own `D00c-<chain>`, pooled blocks on
  two-seed mean deltas; FLOOR on a seed-2/twin/batched run pools normally (`disputed`); C3b's predicate drops `none`;
  milestone (i) contradiction re-runs once on the reduced recipe (no "re-check at M_{k+1}"); `_run_eval → (latest, a, info)`
  with `eval_morph`/`eval_obst` still `dict | None` (+ `eval_status`/`eval_rc`); the SIGTERM path classifies a terminated
  child as `interrupted` after the loop (no `_r2` during shutdown; `train_retry`/`_wait_isaac_clear`/infra retry consult the
  flag); milestones as up to three ordinary experiments (`M<k>.s2`, `M<k>.promo` resuming from the recipe head, `M3.final_s2`)
  with kinds `batched_s2`/`milestone_s2`. Minors (all applied): `RUN` option dropped; conditional-survival MISSING rule; pooled
  pick order incl. seed-sign disagreement; example ids `D08`; row-WORSE definition and (iii) after PASS or WATCH; milestone rows
  re-summarised with the ring proxy; MISSING-guard REJECT queues no pair; floor branch kinds; anchor incl. user-overruled ADOPT;
  `overruled(driver) → overruled(user)` row, `overruled(driver)` covers milestone overrides, `await → squashed` after any
  recipe-changing override; absolute-units wording; null test N = 1000 with eval-only R; Verification 3(a) modulo the three
  additions; `CMD_FIELDS_ADDED`; `batched_s2` kind defined; C1c 3.7 h and the budget sentence; step 10 relays every tag;
  `code_sha` via `ls-tree` + fnmatch, tools outside it (`tools_sha`); rerun-count referent; `D00c-<chain>` ids; `decide --choose
  skip` written by the driver and interrupting the in-flight candidate; `user_choice` with `at`; step 1/2 test ownership; V8
  stop drill.
- Pass 5 (0 blocking, ~7 substantial — all milestone/glue — ~23 minor). Milestone (i) is a zero-GPU step that runs BEFORE
  any promotion (batched_s2 dispatch runs it; promo emitted only once (i) is resolved); `--accept` after a (i) FAIL →
  `accepted_i` → promo → `final`; a milestone with both experiments omitted writes a carried-forward entry; `M3.final_s2` is
  its own next_item rule and runs 2a → 2b from `recipe_head_s2`; `speculative_default` is the all-TIE pick (not a tiebreak) in
  both decisions 5 and 7; a drift-forced REJECT of a clear screen is `confidence: uncertain` + pair, with the pooled drift
  defined; a user `skip` squashes the running experiment (no livelock) and is symmetric to `run`; MISSING-guard REJECT is
  `decided`; `ok run` vs `ok screen`; `D03` write point; header `started`; base ∈ {1a,1c,2a,2b} and the explicit promotion
  stage list; mirror constants stay literals; `_run_eval` timeout SIGTERM-first; arming list; rms wording; Context
  exception for `dead_walker`; FLOOR incl. `dead_walker`; shared rerun count; corrected next_item examples; budget
  (core 101–112 h, milestone incident costs itemised).
- Pass 6 (evidence lens CONVERGED; 0 blocking; goal 3 substantial, feasibility 3 substantial, ~18 minor — all milestone/CLI
  glue). `--accept` after a (i) FAIL freezes the block's adoptions (`confirmed`), so a following (ii) FAIL stops for the user;
  `decide --choose run` squashes the newest entry whatever its status and the derived squash view is global and recomputed
  after every ledger write; FAIL = ≥ 2 of the three 2b rows, 2a rows WATCH; `M1.promo` never omitted (M1's heads are M2's
  reference); 2a/2b overlays derived from the recipe (rescaled pool + eval_carry); `skip` refused on a candidate with a
  decision, HALT line per kind, user confirm/overrule interrupts an in-flight pair. Minors (all applied): sticky-await row
  excludes `accepted_i`; `decided` definition; `renormalise` semantics; FLOOR clause incl. `dead_walker`; step (2) wording
  (overrule, not squash, the dropped adoption); (ii) FAIL re-screen cost; C4 number; `track_ratio_series(key=)`; null band
  `[0.85, 0.98]`; milestone `open` status and `promoted_head`; `LAST_EVAL_INFO`; `D03` once; `base_code_sha`; tests and V7
  drills; base vocabularies; `milestone_s2` wording; replay registered in `CURRENT_CHILD`.
- Pass 7 (0 blocking; goal 1 substantial, evidence 2, feasibility 2 — overlapping; ~22 minor, most of them residue of
  pass-6 sentences left unchanged). `--accept` freezes every unfrozen block adoption (`decided` or `final`) and the `await`
  row says so; C8's promotion form pinned (share edits re-applied after the teacher rescale → `0.05:0.55:0.15:0.25`, explicit
  `teacher_overlay` on `scratch_1c`, validator accepts `SLOTS_TEACHER` with adopted share edits); a user `confirm`/`overrule`/
  `decide run` of a `speculative` decision drops and interrupts its own pair (never the base twin; never on a driver/
  milestone override) — SQUASH (2)/(5), the Seed policy and the HALT line agree; the base recipe is rank-composed (an
  out-of-rank re-run's control is the recipe before it) and the dispatch row says so. Minors (all applied): schema
  parenthetical; the two `run` sentences; "effective entry" defined once; the derived view self-contained (base hash, SKIP
  predicate re-evaluation, transitive refs, user skip); per-row milestone resolution; re-screen s2 cost; frozen entries
  named in SQUASH/MILESTONE lines; twin HALT verbs; null-test wording + marginal qualifier; V3(a) two new labels; (i) rerun
  cost conditional; Part C (ii) overlay wording + `teacher_overlay` default; step (4) closes the open entry; `max_attempts=1`
  on the loop's eval call; the 10-min wait in interrupt-checked ticks; `accepted_i → await` row; D03-once, `LAST_EVAL_INFO`
  and replay-interrupt tests.
- Pass 8 (evidence lens CONVERGED; 0 blocking; goal 2 substantial, feasibility 2 substantial — all in pass-7 sentences —
  ~13 minor). User squashes are sticky: derived-view cause (e) (`decide run` entries and a decision's own seed-2 run after a
  user confirm/overrule/decide run) is stored and never resurrected, resurrection applies to (a)–(c) only; the ring row's
  conditional turn survivals are scored like every guard (ring completion alone is WATCH); chain preset-emulation keys never
  reach 2a/2b (C1/C1c `teacher_overlay: {}`, validator refuses a stray 2a/2b `FACING`). Minors (all applied): clause (b)
  sense; the CLI `run` sentence; next_item consults a user `skip`; twin/batched/milestone HALT verbs split; line-75 pair
  wording; `B<k>` for an out-of-rank base; V3(a) prints 0 not NaN; the `decide run` and `squash_set` tests; SQUASH (2)
  transitive clause limited to SKIP entries; "ok screen = newest".
- Pass 9: **CONVERGED** — all three lenses report 0 blocking and 0 substantial findings; every pass-8 item verified in
  place. The 19 minors (all wording) applied: derived-view cause (e) reordered so the `speculative` qualifier binds every case
  and a `final` decision's finished s2 is never squashed; the HALT per-kind groups no longer overlap; the stale `decide C5
  run` test line removed; `overruled(user) → squashed (transitively)` row; seed-2 reuse rule and the untouched-twin rule in
  decision 7; pending pairs across a code change; `build_mirrors` encoding lookup named; the manifest under `code_sha`; the
  chain `CMD_POOL`/`SLOTS` reach 2a/2b only rescaled and the facing clamp is the one driver-written `FACING` key; the floor
  WATCH branch covers baseline/milestone kinds; the out-of-rank `B<k>` cost; the in-flight-seed-2 confirm drill in V7.
