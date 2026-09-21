<!-- APPROVED PLAN OF RECORD, 2026-09-21. Verbatim copy of /home/nickmagus/.claude/plans/the-facing-and-tracking-fancy-breeze.md
     as approved by the user, saved here so any later session can execute it without the approving session's scratchpad.
     It supersedes the evaluation half of PLAN_2026-09-18_facing_tracking_ladder.md (this directory): that plan's decisions 4
     (floors), 5 (blocks/guards by hold) and 6 (its curriculum row A6) are replaced by D1-D10 below; everything else in it -
     the ranked candidate list, the speculation/squash protocol, the milestone structure - still stands.
     START HERE: execution step 0, then (1 || 2) -> 3 -> 4 -> 5 -> 6 -> 7 -> 8. The campaign is PAUSED (a STOP file sits in
     this directory and `ledger_cli unstop` is step 8's first command); nothing may be relaunched until steps 1-7 have landed
     and the suite verification has passed. Deviations discovered while executing go in PLAN_AMENDMENTS.md beside this file. -->

# Plan: per-category evaluation for the omni task — "only the commands a model trained on decide"

> The previous plan (the candidate ladder, approved 2026-09-18) is preserved verbatim at
> `$LAD/PLAN_2026-09-18_facing_tracking_ladder.md` with its deviation log `$LAD/PLAN_AMENDMENTS.md` and the CHARTER.
> This plan narrows scope to the evaluation: it changes what a screen measures and what is decisive, migrates the paused
> campaign onto the new measurement without retraining, and resumes it. It changes no training code, preset, or reward.

## Context

The ladder campaign paused on 2026-09-19 at `await_user:stop` after C1 (the user's additive capability curriculum) and its
reduced-demand twin C1b were rejected by stage floors that read *prefix survival* on one ten-hold episode
(`curriculum__omni`): "survival through `move_turn_r`" 0.23 at C1's c4, "through `back`" 0.25 at C1b's c2. That number is
contaminated by construction — it counts every fall in every earlier hold, trained or not — and it is noisy at the
threshold: C1's and C1b's c2 stages ran an *identical* configuration (the turn scaling touches no c1/c2 command type; their
c1 results reproduce to the digit) and scored 0.36 and 0.25 against a 0.30 floor, while their per-hold speed ratios at
`back` were comparable (0.53 vs 0.61). The floor rejected the curriculum on noise in holds it was not about.

The same contamination sits in the ladder's decisive numbers: `completion` = survived the *whole* eight-hold row
(`slow__omni`); the F block's `move_turn` hold is a bearing sweep the plant cannot follow; the two "turn survival" guards
exist only because holds are nested in one episode. The user's request (2026-09-21): **separate evaluations for each command
category — standing, moving forward, moving sideways, backward, diagonally, turning, turning while moving — each with
metrics relevant to that motion; when a model is evaluated, only the categories it was trained on are decisive, and every
category is still run for information.**

**User decisions (2026-09-21, fixed):** (1) seven merged categories — `stand, fwd, side, back, diag, turn, move_turn`
(mirror pairs share a category, handedness reported inside it); **amended after the pass-1 review: `move_turn` is not
WATCH-only — it carries decisive and guard metrics (D5), because it is where the sacrifice the campaign exists to fix is
largest (the 2b head ends the move+turn hold at 1.218 rad)**; (2) one Kit launch per suite, one single-category episode per
env, W waves, n ≈ 100 per category; (3) an
eval-only change **re-evaluates** the recorded heads instead of retraining the baseline (an `eval_sha` split from
`code_sha`; D00 re-resolved on the suite); (4) "only trained categories are decisive" reaches floors, RETENTION, the screen
decision and milestones. Refinement adopted after the design critics: **a wave is one hold for all envs** (scalar commands),
which keeps the omni command term — inside `code_sha` — untouched while giving the same outcomes.

Abbreviations: `$R = /home/nickmagus/krabby/krabby-research`; `$OMNI = $R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task`;
`$FWD = …/crab_hex_forward_task`; `$GE = $FWD/scripts/gait_eval`; `$EVAL = $FWD/scripts/eval_crab_hex_gait.py`;
`$TOOLS = $OMNI/experiments/tools`; `$LAD = $OMNI/experiments/2026-09-19_facing_tracking_ladder`;
`$MANIFEST = $OMNI/experiments/eval/scenarios_omni.yaml`; `$PY = /home/nickmagus/krabby/isaac_venv/bin/python`.
Branch `feat/crab-hex-omni-task`, HEAD `9c2c1fba` (a record commit from another session sits on the ladder's `9f61eb2`; tree
clean). **Measured and load-bearing:** the code-bearing tree is identical at `81a1aa3`, `75dd943`, `9f61eb2` and `9c2c1fba`
under both the current pathspecs (`3175fc88c44a`, 127 rows) and the narrowed training-side pathspecs (`62f6f82a1a4a`, 121
rows), so D8's alias proof succeeds and no baseline retrain is needed. (`lad.code_sha` records `4c3f99c9…`, computed with the
driver's own row format; the proof compares like with like through `code_sha_at`.)

Standing constraints: forward and sideways tasks untouched (cfg dumps byte-identical; the gait_eval golden fixture is
body-rows-only and stays byte-identical — every new key is world-row-only, gated like `facing`); the three legacy omni rows
are shape-frozen; one Isaac process at a time; nothing is pushed; the keep-set rule; no `__init__.py` under `experiments/`;
the paused campaign's records are appended to, never rewritten; `$OMNI/mdp` and `$OMNI/config` do not change.

## Vocabulary
- **category** — one of `stand, fwd, side, back, diag, turn, move_turn`. **hold** — one command row of the suite; a category
  has one hold (`stand, fwd, back, diag`) or a left/right pair (`side: lat_l, lat_r`; `turn: turn_l90, turn_r90`;
  `move_turn: move_turn, move_turn_r`). **hand** — `l | r` of a paired hold. The **settle** hold is never an episode's own hold.
- **suite** — the manifest row `suite__omni`: ten waves in one launch; wave k = every env runs `[settle 2 s, hold_k 12 s]`.
- **trained categories** of a stage — what its command distribution covers (`eval_categories.trained_categories`, D4):
  `decisive` (share ≥ `POOL_FLOOR_SHARE` at the suite's full demand), `present` (share > 0), `watch` (present but trained at
  a reduced demand). **decisive set** of a comparison = the CANDIDATE's decisive categories.
- **N/A block** — a block with no decisive member on this comparison; never TIE, never MISSING.
- **eval_sha** — sha256 over the blob ids of the eval-side pathspecs (D8); **code_sha** — the training-side tree (today's
  list minus the eval-side files). **alias** — an older `code_sha` proven equal to the current one on the training-side tree.
- **rescreen** — run the suite on a recorded ok screen's checkpoint and file it into that record; **reeval** — run the suite
  on a recorded head that is a D00 input (no header); **rescore** — run the suite on a floor-rejected chain's failed-stage
  checkpoint and re-decide that floor. **migration** — the ordered set of those items plus D00s/D00bs (D9).

## Design decisions

**D1. Suite geometry (harness).** `suite__omni` = `kind: category_suite`, flat, `episodes: 100` (envs per wave),
`waves: 10`, `env_seed: 1`, `spawn_yaw: first_face` (= 0.0), `settle: {hold_s: 2.0, label: settle}` (speed 0, dir 0, face 0),
`hold_s: 12.0` for every category hold, `episode_length_s: 18.0` (≥ 1.5 + 2 + 12 = 15.5). Holds, in wave order, keep the
legacy label vocabulary wherever the command is the same, so every existing per-hold consumer reads them unchanged:

| wave | label | category / hand | speed | dir (world) | face | note |
|---|---|---|---|---|---|---|
| 0 | `stand` | stand | 0.00 | 0 | 0 | |
| 1 | `fwd` | fwd | 0.25 | 0 | 0 | = legacy |
| 2 | `lat_l` | side / l | 0.22 | +π/2 | 0 | = legacy |
| 3 | `lat_r` | side / r | 0.22 | −π/2 | 0 | = curriculum row |
| 4 | `back` | back | 0.22 | π | 0 | = legacy |
| 5 | `diag` | diag | 0.25 | +π/4 | 0 | = legacy (fixed hold, not a uniform sample: D00 pairs on `env_index`) |
| 6 | `turn_l90` | turn / l | 0.00 | 0 | +π/2 | = legacy (90° left from face 0) |
| 7 | `turn_r90` | turn / r | 0.00 | 0 | −π/2 | **new meaning**: 90° right from face 0 (legacy: +π/2 → 0) |
| 8 | `move_turn` | move_turn / l | 0.22 | 0 | +π/2 | = legacy (world-held travel while the body turns) |
| 9 | `move_turn_r` | move_turn / r | 0.22 | 0 | −π/2 | **mirror** of `move_turn` |

Every episode starts at the pinned spawn yaw 0 and its prologue (1.5 s) holds the settle command, so a turn hold's
`heading_error_initial_abs` is ≈ 1.5708 (only the standing yaw drift over the 3.5 s deviates) and `facing_closed` has a
near-constant denominator; translation holds start with facing error ≈ 0. Feasibility: 90° needs `1.0 + 1.5708/0.3 = 6.24 s
≤ 12 s`; every speed clears the ellipse cap at both the start and end body angle (0.35 / 0.25 / 0.28766 / 0.25).
**n:** 100 episodes per hold → **100 per hand**, 100 for a singleton category (`stand, fwd, back, diag`), **200** for a
paired one (`side, turn, move_turn`). `n_c` in `R_binom` and in the floor band is the category's own n.

**D2. Waves (harness).** The main loop runs W waves in one Kit process. Wave k compiles its own two-hold uniform schedule
(scalar `ScheduleRow`s — the omni `schedule_hook` is untouched, pinned by a source-hash test); between waves the harness
calls `env.seed(env_seed * 1_000_003 + wave)` (Isaac Lab's `set_seed`, covering torch/cuda/numpy/random; recorded as
`wave_seeds`) then `env.reset()` (all envs re-spawn at the pinned yaw), **re-installs the override with that wave's compiled
schedule**, applies row 0, re-reads observations, zeroes the per-wave buffers and bookkeeping, and runs the existing step
body verbatim; a fallen env stays frozen for the rest of its wave (as today). The re-seed is what makes wave k's initial
states independent of how many envs fell in wave k−1 (a head-dependent RNG draw), which is what `env_index` pairing across
heads requires. Episodes are indexed `env_index = wave × 100 + env` (0..999, unique; identical `(episodes, waves, suite)`
across heads is the pairing precondition, checked from `run_meta.env_index_map`). `segment_id` carries ROW-GLOBAL hold
indices into the union `hold_labels = [settle, stand, fwd, …, move_turn_r]` (11 entries, settle = 0), so every
position-keyed reader (`rescore_facing`, `_columns`) stays correct. GPU footprint per wave is 21 % of a `slow__omni` row;
host ≈ 1.5 GB; disk ≈ 510 MB raw + ≤ 240 MB JSON per suite run (≈ 2.5–3× a legacy row). Cost, at the 0.047 s/step + 26 s
fixed fit: 10 × (36 s sim + reset) + ~30 s fixed + the driver's 30-s tick ≈ **7–8 min per suite eval** (vs 3.8 min).

**D3. Metrics per category (harness, world-row-only, additive).** Existing per-hold blocks stay keyed by label
(`tracking_by_hold`, `facing_by_hold`, `tripod_score_by_hold`). New per-hold entries, gated exactly like `facing`:
`motion` = `{speed_abs_mean_m_s (mean ‖v_xy,body‖), pos_drift_m, path_len_m, yaw_drift_rad, foot_lift_count,
contact_churn_per_s, n_steps}` over the whole hold; `direction` = `{direction_error_rad, cross_track_ratio,
achieved_speed_m_s}` from the travel-frame velocity already computed for tracking, emitted **only when the hold's commanded
speed > 0** (so `settle`/`stand` carry none). Aggregates `motion_by_hold`, `direction_by_hold` (`_stats`).

On a suite row the aggregate gains **`suite`** = `{waves, wave_seeds, episodes_total, hold_labels (union),
hold_of_wave: [labels], category_of_hold, hand_of_hold, n_by_hold,
survival_by_hold (each hold's OWN completion rate), trained_categories,
by_category: {c: {key: _stats}}}`. `by_category` is computed in `report.aggregate` by POOLING the per-episode values of the
category's own-hold entries over both hands (never by averaging hand medians — the pooled median is what the stratified
bootstrap resamples and what `_reducer_check` reproduces): `ratio ← tracking.vx.ratio`, `achieved_speed ← tracking.vx.actual_mean`,
`tripod ← tripod.tripod_score` (excluding `low_confidence`), `facing_closed | yaw_progress | net_yaw | settle_s | heading_final
← facing.*`, `drift_speed ← motion.speed_abs_mean_m_s`, `direction_error | cross_track_ratio ← direction.*`, plus
`by_hand: {l|r: {key: _stats}}`. `aggregate(episodes, *, suite=None)` receives the scenario's suite block (and
`trained_categories` from `run_meta`) from `write_run`/`summary_text`; episodes carry `wave`, `hold`, `category`, `hand`.
Nothing else in `aggregate` changes; on a suite row the pooled keys (`schedule_completion_rate`, `tripod_score`,
`fall_classes`, `orientation`, `support_polygon`, `slip_ratio`, …) are means over all ten holds and are **WATCH-only**
downstream. Category metric roles:

| category | decisive members | guards | WATCH |
|---|---|---|---|
| stand | — | `survival` | `drift_speed`, `pos_drift_m`, `yaw_drift_rad`, `foot_lift_count`, `contact_churn_per_s` |
| fwd, side, back, diag | T: `ratio`; T_abs: `achieved_speed` | `survival`; `cat_tripod_min` (min of the category tripod medians over the decisive translation categories) | `direction_error_rad`, `cross_track_ratio`, `heading_final`, `duty_factor` |
| turn | F: `facing_closed`, `yaw_progress`; F_abs: `facing_closed`, `net_yaw` | `survival` | `settle_s`, `heading_final`, `pos_drift_m`, both hands |
| move_turn | F_mt: `facing_closed`, `yaw_progress`; F_mt_abs: `facing_closed`, `net_yaw` | `survival`; **`ratio`** (the anti-sacrifice guard) | `achieved_speed`, `tripod`, `direction_error`, `cross_track_ratio`, `settle_s`, both hands |

`drift_speed` stays WATCH this window: it is lower-is-better, and the guard machinery is one-sided higher-is-better, so
making it decisive needs a sign mechanism that only RETENTION requires (D6). A standing collapse is caught by
`cat.stand.survival`. A later window may promote it once D00s shows it resolves.

**`move_turn` is decisive, in its own block, paired with a translation guard.** The hold is infeasible by construction
(walking at the 0.21 m/s floor leaves ω ≤ 0.01 rad/s at `r_foot` 1.2 m), so both sides of every comparison face the same
impossible demand and the facing numbers are a **relative** measure — which is exactly the campaign's question, and where the
sacrifice is largest (the 2b head ends at 1.218 rad there versus 0.68 on the in-place turn). Two rules make it honest:
its facing sits in **its own block `F_mt`**, not inside F, so the noisiest facing member cannot drown the in-place turn signal
(and vice versa); and `cat.move_turn.ratio` is a **guard**, so a candidate cannot win `F_mt` by stopping to turn. A margin
trade still passes: the guard fires only on a drop beyond `max(R_eval, R_seed, …)`, about 0.05 to 0.10 in ratio units on the
recorded numbers, so losing a few percent of the commanded speed while closing 0.2 of facing is an ADOPT, while abandoning
the translation is a clear REJECT. `tripod` there stays WATCH (it is not a translation category and is excluded from
`cat_tripod_min`). For a candidate that declares `absolute_units` the guard is `cat.move_turn.achieved_speed` instead of
`ratio`, for the same reason T_abs swaps the block member.

**D4. Trained categories (omni package).** New Isaac-free module `$OMNI/eval_categories.py` (task root, outside `mdp/` and
`config/`, hence outside `code_sha`) which loads `crab_hex_omni_command_math.py` and `config/crab_hex/omni_knobs.py` **by
path** (`importlib.util.spec_from_file_location`, the `run_candidates.pool_math()` pattern — a dotted import would run
`parkour_tasks/__init__.py`'s Isaac `import_packages`): `CATEGORIES` (no category is WATCH-only by construction — a category
is decisive when the recipe trains it at the suite's demand, and watched otherwise),
`CATEGORY_HOLDS`, `HAND_OF_HOLD`, `TYPE_TO_CATEGORIES` (`stand→stand, fwd→fwd, side_l|side_r→side, back→back,
omni→{fwd, side, back, diag}, turn→turn, fwd_turn|side_*_turn|back_turn|omni_turn→move_turn`),
`TrainedCategories(decisive, present, watch, hands, demand, source, facing_band_hi, encoding, notes)`,
`trained_categories(env: Mapping) -> TrainedCategories`, `category_of_hold(label)`, `decisive_set(tc)`.
Pool mode: present at weight > 0, decisive at weight ≥ `POOL_FLOOR_SHARE`; `side` is decisive only when both `side_l` and
`side_r` (or `omni`) are; `omni` makes `fwd, side, back, diag` decisive (uniform-bearing support; note recorded). A scale
edit demotes **only the categories its own type maps to** through `TYPE_TO_CATEGORIES`: `ang_scale < 1` on `*_turn`/`omni_turn`
touches `move_turn` alone (so C1b's and C7's `{"*_turn": 0.3}` leave `turn` decisive — `fnmatch("turn", "*_turn")` is False);
`ang_scale < 1` on the `turn` type demotes `turn`; `speed_scale < 1` demotes that type's translation categories. Slots mode
(legacy presets): slot 0 → stand; slot 1 → `omni` if `dir_band` is None else the types whose band lies inside it (tolerance
1e-4); slot 2 → turn; slot 3 → `omni_turn`/band types; the slot share is the enabled share. So `move_turn` is decisive for
B0 and every `scratch_1c`/`scratch_c6` recipe (slot 3 at 0.20–0.25, `omni_turn` in the pool) and from curriculum stage c4;
it is **watch** for C1b and C7, whose `ang_scale 0.3` on the `*_turn` types trains it at 30 % of the suite's turn demand.
**`turn` is decisive for every
recipe**, including under `FACING_OBS_ENCODING=sincos`: the encoding changes what the policy observes, not which command
category it trained, and the comparison with the control is symmetric at the command. `facing_band_hi` and the encoding are
**annotations** printed on the RESOLUTION and DECISION lines (`turn: demand 1.571 rad > FACING band 0.600; encoding raw
clips at 0.6 / sincos sees the magnitude`), never a demotion. A stage whose `stand` share is 0 is a `halt:eval_mismatch`
(the suite's settle hold is part of every category's episode). Derived: B0/`scratch_1c` → decisive {stand, fwd, side, back,
diag, turn, move_turn}; curriculum c1 {stand, fwd}, c2 +{side, back}, c3 +{turn}, c4 +{move_turn}, c5 +{diag}.
Recorded in the experiment header per stage, passed to the harness as `--trained-categories` (recorded in `run_meta`,
mirrored into `aggregate.suite.trained_categories`; scoring never reads it), copied onto the screen dict by `_screen_of`,
and printed on the REPORT and DECISION lines.

**D5. Decision rule v2 (driver).** `summarize()` gains one additive key `cat` (`{}` when the aggregate has no `suite`):
`cat[c] = {n, survival, ratio, achieved_speed, tripod, facing_closed, yaw_progress, net_yaw, settle_s, heading_final,
drift_speed, direction_error, cross_track_ratio, by_hand: {l, r} | None}`, each read as
`aggregate.suite.by_category[c][key].median` (`survival` = the n-weighted mean of the category's holds' own survival;
`n` = the category's episode count). Every existing key is unchanged; on a suite row the legacy per-hold keys are the
per-hand readings and `turn_survival_l90/r90` are `None`.

`candidates.yaml decision_rule` becomes `rule_version: 2` with `per_category` templates: F `{turn: [facing_closed,
yaw_progress]}`, **F_mt `{move_turn: [facing_closed, yaw_progress]}`**, T `{fwd|side|back|diag: [ratio]}`,
F_abs `{turn: [facing_closed, net_yaw]}`, **F_mt_abs `{move_turn: [facing_closed, net_yaw]}`**,
T_abs `{…: [achieved_speed]}`; guards `{every category: [survival]}` + **`{move_turn: [ratio]}`** +
`composite: [cat_tripod_min]`; and the watch list (every `*.direction_error`/`cross_track_ratio`, `stand.drift_speed`,
`move_turn.achieved_speed`/`tripod`/`settle_s`, both hands of every paired category, `completion`,
`tripod_walking_dir`, `turn_survival_*`, `track_ratio_tail`, `track_ratio_move_tail`). A candidate scores either
{F, F_mt, T} or, when `absolute_units` is declared, {F_abs, F_mt_abs, T_abs} — three blocks, as today's two became three.
`expand_rule(rule, decisive)`
produces today's flat shape restricted to the decisive set (ids `cat.<c>.<key>`, `cat_tripod_min`) and moves every untrained
category's member into the WATCH dict with `note: untrained` — the informational half of the request, printed on the
DECISION line. A block with no member is **N/A** (`{class: "N/A", k: 0}`), never TIE.

Plumbing: `_screen_compare` gains `decisive: frozenset | None`, expands the rule internally, injects `composite_guards`
into shallow copies of both screens, and marks memberless blocks N/A; `decide`, `decide_pooled`, `anchored_drift`,
`_row_worse` and `_write_d00bs` derive `decisive` from the CANDIDATE screen's `trained_categories` (a category the base
never trained stays decisive with `note: base untrained`); `_speculative_pick`, `_any_worse` and `_better_blocks` skip N/A
blocks; `guard_rows` takes `n_c = min(cat.<c>.n of both sides)` for the binomial floor; `seed_terms_from_pair` receives the
rule already expanded at `decisive(B0)`.

`decide`: any WORSE → clear REJECT; any guard MISSING → REJECT `confidence: uncertain`, no pair; **all scored blocks N/A →
`status: undecided`, `chosen: WATCH`** (added to that branch's `options`; `is_terminal` False, so dependents wait) with
`why: "no decisive block (trained: …)"` and no pair — a narrow recipe is never rejected for being narrow; otherwise today's
branches over the non-N/A blocks. Guards are one-sided at 99 % on `R_i = max(R_eval,i, R_seed,i, R_binom,i)` with
`R_binom = z_0.99 · sqrt(2 p̄(1−p̄)/n_c)` and `p̄ = clip((surv_a + surv_b)/2, 0.05, 0.95)` — so a survival saturated on both
heads floors at 0.072 (n = 100) / 0.051 (n = 200) instead of 0. **MISSING is reserved for a missing VALUE:** a survival
guard whose `R_eval` is None or 0 is scored against `max(R_seed, R_binom)` and records `R_from_floor: binomial`, never
`missing: "resolution"`. `completion`, `tripod_walking_dir`, `turn_survival_*`, `track_ratio_*` → WATCH. Anchored drift,
pooled decisions, `absolute_units`, `speculative_default` and the MISSING-value semantics are otherwise unchanged, on the
expanded rule. Evidence records `decisive`, `untrained: {candidate, base}`, and every N/A block.

**D6. Floors and RETENTION (driver).** `floor: {kind: category_survival, row: suite, min_new: 0.10, min_prev: 0.30}` on
every non-final chain stage with an eval row (final stages `none`; the screen's guards cover them). For each
`c ∈ decisive(stage)`: `c ∈ decisive(stages[:i])` (any earlier stage, including one with `eval: []`) → threshold `min_prev`,
else `min_new`. The test is a **band test**: FAIL only when the one-sided 99 % upper binomial bound
`p̂ + z_0.99 · sqrt(p̂(1−p̂)/n_c)` is below the threshold; the FLOOR line prints point, bound, n and the failing categories.
Effective point thresholds: at n = 100 a stage passes down to p̂ ≈ 0.05 (new) / ≈ 0.21 (previous); at n = 200, ≈ 0.06 / ≈ 0.23.
A decisive category the suite row does not carry is a `halt:eval_mismatch` (not a silent pass); "not scored (no verdict)"
survives only for a stage whose eval list is empty. On the recorded conditional per-hold survivals — the proxy for the
per-category number (C1 c2 `back` 0.58, C1b c2 0.52, C1 c4 `fwd` 0.82 / `lat` 0.89–0.96 / `back` 0.83 / `turn` 0.88–0.90) —
neither D05 nor D06 would have fired. RETENTION compares `cat.c.survival` and the category's decisive metric (`ratio`;
`facing_closed` for turn; `drift_speed` for stand, **lower is better** — the per-metric sign lives here) for previously
enabled categories against the incumbent trace (`stage_trace_v2`, keyed by category; a hold-keyed trace is re-seeded with
the line "trace predates the suite"); thresholds unchanged (0.10 survival, 20 % metric), WATCH + notify.

**D7. Milestones (driver).** Promotion stages evaluate `[suite, light, ring]` (from `candidates.yaml`'s `promotion.eval`,
default `PROMOTION_EVAL`); `light`/`ring` are WATCH rows (terrain information, no per-category numbers). The 2b head versus
the previous milestone's 2b head is compared on the suite per category at `by_row[suite__omni]`, with the decisive set of
the **2b head's** recipe (slot-mode 2b preset → all seven): **FAIL iff ≥ 2 decisive members are WORSE** (block members and
the per-category `survival` and `move_turn.ratio` guards count; `cat_tripod_min` does not), so two members must corroborate;
nominal FAIL rate under the null ≈ 1 % (3 blocks + 8 counting guards), printed on the MILESTONE line. An unreadable suite row → WATCH. M1 stays WATCH-only
against the reeval'd 2b/2a heads; `milestone_reference` names the reeval **root** and resolves the newest stamp with the
same helper `load_row` uses (`resolve_eval_dir(root, sid)`). `_reference_watch`, `_close_final_s2` and `_stage_gate_watch`
read `suite`; the formation-era gate verdicts stay WATCH context on the suite's legacy per-hand keys.

**D8. `eval_sha` (driver).** `CODE_PATHSPECS` loses `scenarios_omni.yaml`, `run_phases.py` and `gait_eval`;
`EVAL_PATHSPECS = (gait_eval/, eval_crab_hex_gait.py, scenarios_omni.yaml, run_phases.py, $OMNI/eval_categories.py)`;
`eval_sha()` uses the same `CODE_DROP_GLOBS`; the dirty check covers both lists. `run_phases.py`'s training seam (`train`,
`train_retry`, `_subenv`, `with_plant`, the command line and env composition) is pinned by
`test_run_phases_training_seam_is_unchanged` (sha256 of those functions' source == a recorded constant): a training-relevant
edit there must bump the pin and be declared with `--accept-code-change` — a recorded caveat, the price of not splitting the
module. Launch records `lad.eval_sha`; every header and ledger entry carries `eval_sha`; `ok_screen` additionally requires
`h.eval_sha == lad.eval_sha`, so a screen measured on the old eval is never a control or an anchor until rescreened.
A changed `eval_sha` is refused without `--accept-eval-change WHY`. Because the pathspec change also moves the VALUE of
`code_sha`, the acceptance **proves the training tree did not move**: `code_sha_at(commit, pathspecs)` over
`git ls-tree -r <commit>` with the same drop globs; find the newest commit C in `git log -n 300` with
`code_sha_at(C, CODE_PATHSPECS_V1) == lad.code_sha`; require `code_sha_at(C, CODE_PATHSPECS) == code_sha()`, else refuse
("the training tree also moved: use `--accept-code-change`"); on success append `{sha, proved_at, why, at}` to
`lad.code_sha_aliases`. `ok_runs` reads that list itself and accepts `h.code_sha ∈ {current} ∪ aliases`; `baseline_label`
normalises through the same set (so no spurious `B1`). Records are never rewritten: a rescreen keeps the old screen under
`header.screens_superseded: [{eval_sha, screen}]`.

**D9. Migration of the paused campaign (driver, zero decisions by hand).** `loop()` checks a new `migration_item(ctx)`
**before `find_inflight` and before `_next_item_raw`** (whose baseline-rerun rule would otherwise book an 8000-iteration B0
rerun the moment `ok_screen` goes stale), and `find_inflight` returns None while `migration_item` is pending. Each item is
derived from the records, in this order:

1. **rescreen X01 then X02** — every non-squashed `status: ok` header of a `SCREEN_KINDS` kind whose `code_sha` is current or
   aliased and whose `eval_sha` is stale (today exactly X01 = B0 s3 and X02 = B0 s2). Each handler begins with
   `rp.set_seed(ctx.c, header.seed)` (X02 is a seed-2 record; filing it under seed 3 would leave it screenless and trigger a
   3.9 h rerun) and builds the item through `item_from_header`, so the tag, out root, overlay and `_carry_mismatch` are the
   recorded ones. Writes `rec["evals"]["suite"]` beside the kept `slow`, the new `header.screen` (old one under
   `screens_superseded`), `header.eval_sha`, `header.rescreens[]`. No ledger entry.
2. **reeval** the three D00 inputs (`1c-004`, `2b`, `2a-reeval`) on the suite — `resolution_inputs` entries carry
   `checkpoint + preset + eval_dir` and no `run_dir` (a validator branch accepts `run_dir` XOR `checkpoint+eval_dir+preset`
   and refuses neither/both); out root `EVAL_ROOT/<campaign>/reeval_<head>`, recorded in `lad.reevals`.
3. **`D00s`** (kind resolution, `supersedes: D00`, carries `eval_sha`) from the cross-head suite pairs (`1c-004` vs `2b`,
   vs `2a-reeval`; stratified paired bootstrap, B = 2000, `paired_on: env_index`, one-sided 99 % guards with the binomial
   floor). Startup never computes it: `load_resolution(ctx)` ingests `resolution.<eval_sha[:12]>.json` if present and
   otherwise leaves `ctx.resolution = None` (never the old slow-row record); `_dispatch_candidate`, `_dispatch_pooled`,
   `_row_worse` and `_write_d00bs` halt `resolution_missing` while it is None.
4. **`D00bs`** — the seed terms per block and guard, from the two rescreened B0 screens **against `D00s`** (the seed terms
   standardise through `D00s`'s per-metric `R_eval`, so this cannot precede step 3). Its RESOLUTION line carries the
   seed-dominance verdict: `R_F^seed` vs `R_F^eval`, the per-member MDE, and — the informative statement — B0 s2 vs s3
   classed at `D00s`'s **eval-only** R per block. (The all-TIE check under the full R is a construction check: the pair's own
   delta is inside its seed term. A non-TIE there is a bug.) Acceptance also records the null-candidate all-TIE inside
   `[nominal − 0.03, 0.98]` (nominal = 0.95^blocks × 0.99^guards = 0.95³ × 0.99⁹ ≈ **0.78** at B0's decisive set: three
   blocks, seven `survival` guards, the `move_turn.ratio` guard and `cat_tripod_min` — the null construction uses the
   stratified draw and the floored guard R, so the band is the one `decide` uses) and whether `R(D00bs) ≥ R(D00s)` per block.
5. **rescore** the floor-rejected chains: X03 (C1) at its c4 checkpoint and X04 (C1b) at its c2 checkpoint, each with that
   stage's recorded overlay and eval_carry, filed into the existing rec (`ctx.c.recs["X03:c4"]["evals"]["suite"]`) after
   `rp.set_seed`. The floor is re-decided under D6 on that stage's trained categories. **PASS** → the old entry (D05 / D06)
   is squashed by derived-view cause **(f)** (`kind == candidate` and `evidence.floor` and `eval_sha != lad.eval_sha`, a
   missing key counting as different), `header.recs.append("<xid>:<floor_stage>")` (the rec exists with its ckpt; without
   this the resumed chain would re-train the rescored stage), `floor`/`floor_stage` move into `rescores[-1]`, and the header
   re-opens for its NEXT stage (C1: c5, 3500 it; C1b: c3, 6000 it) with `inflight = None` (the next stage is fresh and takes
   `prev_ckpt` from the appended rec). **FAIL** → a new floor REJECT entry under the new eval, the old one squashed by (f)
   as history. D07 (C1c SKIP) follows by cause (c) automatically once D05 is squashed.
6. **the ordinary queue.** `find_inflight` resumes re-opened headers in candidate rank order, but **only** when the
   header's candidate is dispatchable: C1b (X04) carries `depends_on: [C1]` and a predicate on C1's decision, so its
   re-opened header is marked `rescored_awaiting_predicate` and is dispatched by `_next_item_raw` (which evaluates both)
   rather than resumed by `find_inflight`; cause (a) squashes it if C1 is adopted. X05 (C2, `model_400` of 8000) resumes
   after C1's chain. `item_from_header` takes each stage's `eval`/`floor`/`categories` from the CURRENT chain definition
   (the overlay stays frozen; the eval side is not a property of the window).

Every step is one state/ledger write plus one notify line; the SQUASH / RESOLUTION / DECISION contract is unchanged.

**D10. Resolution on the suite (`resolution.py`).** `SCREEN_ROW = "suite__omni"` (`LEGACY_SCREEN_ROW` kept for the recorded
rows' `by_row`). `load_row` accepts `checkpoint + eval_dir` entries (`run_dir = resolve_eval_dir(REPO/eval_dir, sid)` = the
newest stamp under `<root>/<sid>/seed001`, `ResolutionError` naming the reeval when absent), reads each episode's
category/hand/hold from `aggregate.suite` (never from the episode's hold dict) into `category_index`/`hand_index`, and
refuses a row whose populated hold is not the episode's own. `_columns` adds `cat:<c>:<key>` and `cat:<c>:hand:<h>:<key>`
columns (NaN outside the category), with `surv` from the episode's own hold end. `spec_for` learns `cat.<c>.<key>`,
`cat.<c>.by_hand.<h>.<key>` and `cat_tripod_min(decisive)`. `bootstrap_pair` draws **stratified by category** (each stratum
resampled to its own n; legacy rows bit-identical); `null_construction` uses the same stratified draw and the floored guard
R. `binomial_guard_floor(p̄, n)` with the clipped `p̄`. `role: seed_pair` inputs (the two rescreen run dirs, recorded so the
record documents D00bs' inputs) are excluded from cross-head pairing. `resolve` writes `screen_row`, `eval_sha`,
`supersedes`, `by_category_n`, `null_nominal`. `BORROW` / `--curriculum-runs` retire behind `rule_version == 2`.
`_reducer_check` reproduces every `cat.<c>.<key>` from the per-episode columns to 1e-9 — the one test that pins harness and
tools to a single definition of each number.

**Suite-aware manifest readers (both tools).** `rp._manifest_hold_s` and `resolution._manifest_scenario` currently do
`sc["schedule"]` on the raw YAML, which a suite row does not have; both instead load the manifest through
`gait_eval.schedule.load_manifest` (by path, as `resolution._facing_metrics()` already loads `gait_eval.metrics`) and expose
`kind`, the derived `hold_s` map and `settle_hold_s`. `_survival_end_steps` on a suite row returns the **own-hold** end
`round((prologue_s + settle_hold_s + hold_s[label]) / dt)` for every label — no cumulative walk. All **three** cumulative
injection sites (`rp._run_eval`, `resolution._summary_with_survival`, `run_candidates.summarize_run_dir`) skip when the
aggregate carries `suite`, taking `survival_by_hold` from `aggregate.suite.survival_by_hold` instead.

## Part A — the harness (eval-side; forward/sideways scoring byte-identical)

**A1. `$GE/schedule.py`.** `Scenario` gains `kind: str = "schedule"`, `suite: CategorySuite | None`, `waves: int = 1`;
dataclasses `CategoryHold(label, speed, dir, face, hand)`, `Category(name, kind: stand|translate|turn|move_turn, holds,
watch_only)`, `CategorySuite(settle_hold_s, settle_label, hold_s, categories)`. `load_manifest` parses `kind:
category_suite` rows (`waves`, `settle`, `hold_s`, `categories`; an explicit `schedule:` is a `ManifestError`) and DERIVES
`schedule = [settle] + [one Hold per category hold in manifest order]` (11 holds) as a **label union, not a timeline**:
on a suite row `schedule_duration_s = settle_hold_s + hold_s` and `total_duration_s = prologue_s + that` (= 15.5 s), so the
generic `episode_length_s ≥ total_duration_s` check and `$EVAL`'s `total_duration_s + 2.0` fallback both read one wave.
`_validate_suite`: kinds; unique names and labels (settle included); `hand ∈ {l, r, None}` with equal counts per category;
`waves == number of category holds` (one hold per wave; `Scenario.waves` is derived, and `--waves` is a consistency
assertion, not a configuration); `rotate_holds_by_env` False; `hold_s ≥ settle_s + min_window_s`;
`episode_length_s ≥ prologue + settle + hold_s`; the ellipse cap at both the start (`wrap(dir − 0)`) and end
(`wrap(dir − face)`) body angle; turn feasibility through `Scenario.transition_pairs()` — legacy rows return exactly today's
consecutive pairs (+ spawn pin, + wrap-around under rotation), a suite returns `[(spawn → settle.face)] +
[(settle.face → hold.face) per hold]`. `compile_wave(scenario, wave, *, dt, num_envs) -> CompiledSchedule` = the uniform
two-hold schedule of wave k with `segment_id ∈ {−1, 0, k+1}` (row-global), `hold_labels` = the union, `world_uniform` True;
`compile_schedule(suite row)` raises `ValueError("category_suite rows compile per wave: use compile_wave")` and is unchanged
for every existing row (`hold_order = range(n)`, rotated rows keep their rotation). `stack_waves(parts) -> CompiledSchedule`
concatenates `cmd/segment_id/steady_mask/world_*` along axis 1 and carries `wave_of_env` and `hold_of_env`
(`(W·E,)` int arrays) — the single `(T, W·E)` object `score_run`, `write_run` and `travel_yaw_by_hold` need.
`wave_plan(scenario, num_envs) -> list[EpisodeAssignment]` gives the `env_index → (wave, env, label, category, hand)` map.

**A2. `$EVAL`.** `--waves INT` (consistency assertion), `--trained-categories CSV` (recorded; names checked against the
suite). The main loop wraps the existing body in `for wave in range(W)`: `env.seed(env_seed*1_000_003 + wave)` →
`env.reset()` (wave > 0) → re-install the override with `compile_wave(scenario, wave, …)` → `apply(0)` →
`get_observations()` → zero buffers / `alive` / `n_steps_env` / `term_reason` → the body verbatim → per-wave
`schedule_complete` assignment → append the host arrays. `env.close()` after the loop; `max_cmd_dev` and
`max_sched_state_dev` accumulate across waves; `host = concatenate(parts, axis=1)`; `compiled = stack_waves(parts)` is what
scoring and `write_run` receive. `run_meta` (suite rows, or when `--waves`/`--trained-categories` is given — a legacy
invocation's key set is unchanged): `waves, wave_seeds, episodes_total, env_index_map, suite: {settle_hold_s, hold_s,
categories}, trained_categories, hold_labels_kind: "union"`; `termination_reason` and `n_steps_per_env` have length W·E.

**A3. `$GE/metrics.py`.** `travel_direction_metrics(vel_travel_xy, cmd_speed, *, min_speed=0.02)` and
`hold_motion_metrics(root_pos_w, root_quat_w, root_lin_vel_b, contact, dt)` (D3), pure numpy, `None` on empty windows.

**A4. `$GE/report.py`.** `score_episode`: `entry["direction"]` (commanded speed > 0 only) and `entry["motion"]` per world
hold; `out["wave"] = compiled.wave_of_env[env_idx]`, `out["hold"] = hold_labels[compiled.hold_of_env[env_idx]]`,
`out["category"]`, `out["hand"]` when the compiled schedule carries `wave_of_env`. `aggregate(episodes, *, suite=None,
trained_categories=None)`: `direction_by_hold`, `motion_by_hold`, and the `suite` block incl. `by_category` (D3) when
episodes carry a wave; `summary_text` gains a "by category" block; `write_run` passes the scenario's suite block and
`n_env = len(run_meta["n_steps_per_env"])`; episode files are `episode_{env_index:02d}` (three digits from 100 —
`resolution.rescore_facing` builds the same name).

**A5. `$MANIFEST`.** Append `suite__omni` (D1) with a header note (category-suite semantics, `episodes` = envs per wave, the
`turn_r90` / `move_turn_r` meanings). Legacy rows and `curriculum__omni` byte-identical (the curriculum row stays a record;
nothing runs it).

**A6. Tests (harness).** `tests/unit/test_crab_hex_gait_schedule.py`: `test_category_suite_row_derives_the_union_schedule`
(11 labels, settle first, per-wave durations 15.5 s), `test_wave_plan_gives_every_hold_100_episodes_and_unique_env_indices`
(E = 100, W = 10; hands 100/100; the settle hold is never an episode's own hold),
`test_compile_wave_is_uniform_and_row_global` (segments `{−1, 0, k+1}`, steady windows, `world_uniform` True),
`test_stack_waves_concatenates_and_labels_every_env`, `test_compile_schedule_refuses_a_suite_row`,
`test_suite_transition_pairs_are_from_the_settle_face`, `test_suite_validator_rejects` (duplicate label, explicit schedule,
unknown kind, unbalanced hands, speed over the start cap, `hold_s < 2`, `episode_length_s < 15.5`, `waves ≠ holds`,
`rotate_holds_by_env`), `test_legacy_rows_compile_byte_identically` (against the step-0 fixture
`tests/unit/fixtures/gait_eval_compile_snapshot.json`), `test_committed_omni_manifest_parses` (id set becomes exactly
`{slow, light, ring, curriculum, suite}__omni`; every existing assertion kept; the suite's ten labels, hands, speeds, dirs,
faces, `settle 2.0`, `hold_s 12.0`, `waves 10`, `episode_length_s 18.0`, `env_vars {}`, `total_duration_s 15.5`).
`tests/unit/test_crab_hex_gait_metrics.py`: the two metric functions; `test_world_row_holds_carry_direction_and_motion`
(and no `direction` on `settle`/`stand`); `test_suite_aggregate_pools_both_hands_and_scores_own_hold_survival` (synthetic
suite E = 4, W = 3: a fall in a turn episode leaves `fwd` survival 1.0; `by_category.side` is the pooled median of 2×n
episodes, not the mean of hand medians); `test_aggregate_has_no_suite_block_on_body_rows`;
`test_body_rows_carry_no_omni_fields` extended with `direction`, `motion`, `wave`, `hold`, `category`, `hand`.
`tests/unit/test_crab_hex_gait_eval_golden.py` unchanged and must pass byte-identically.
`tests/unit/test_crab_hex_omni_command.py`: `test_schedule_hook_is_untouched_by_the_suite` (source hash == recorded constant).

## Part B — the omni package and the ladder tools

**B1. `$OMNI/eval_categories.py`** (D4) + `tests/unit/test_crab_hex_omni_eval_categories.py`: decisive/present/watch at the
floor share; `omni` makes every bearing decisive with the note; one-handed `side` is watch; `test_c1b_and_c7_pool_edits_keep_turn_decisive`
(`{"*_turn": 0.3}` touches `move_turn` only); `test_an_ang_scale_on_the_turn_type_demotes_turn`;
`test_sincos_keeps_turn_decisive_and_records_the_encoding_note`;
`test_move_turn_is_decisive_at_full_demand_and_watch_under_a_reduced_one` (B0 and c4 decisive; C1b and C7 watch);
slots mode reproduces the legacy presets; the curriculum stages c1..c6 read from `$LAD/candidates.yaml`; `CATEGORY_HOLDS` covers the suite row's labels
exactly (reads `$MANIFEST`); `test_eval_categories_imports_without_isaac`; the file is absent from `code_rows(CODE_PATHSPECS)`.

**B2. `$TOOLS/run_phases.py`** (eval side only). `eval_morph(..., trained_categories=None)` appends
`--trained-categories`; the suite-aware `_manifest_hold_s`; the three-site survival rule and `summarize()`'s `cat` (D5, D10);
every existing key unchanged; `phase_lines`/`run_phase` untouched. Tests in `tests/unit/test_crab_hex_omni_gate.py`:
`test_summarize_hoists_cat_from_the_suite_block_and_is_empty_without_it`,
`test_summarize_legacy_keys_unchanged_on_a_suite_aggregate`, `test_turn_survivals_are_none_on_a_suite_row`,
`test_no_reader_injects_cumulative_survival_on_a_suite_row` (parametrised over the three sites),
`test_manifest_readers_derive_a_suite_rows_holds`, `test_eval_morph_forwards_trained_categories`,
`test_run_phases_training_seam_is_unchanged`; the existing pins unchanged.

**B3. `$TOOLS/run_candidates.py`.** Hashes and acceptance (D8): `CODE_PATHSPECS_V1`, the narrowed `CODE_PATHSPECS`,
`EVAL_PATHSPECS`, `eval_sha()`, `code_sha_at()`, `--accept-eval-change`, `lad.code_sha_aliases`, alias-aware `ok_runs` and
`baseline_label`, `eval_sha` on headers/entries and in `ok_screen`. Rule v2 (D5): `expand_rule`, `composite_guards`, the
`decisive` parameter through `_screen_compare`/`decide`/`decide_pooled`/`anchored_drift`/`_row_worse`/`_write_d00bs`, N/A
blocks, the binomial floor with `n_c`, the `undecided/WATCH` branch, `evidence.decisive/untrained`. Stages (D6):
`categories:` replaces `holds:` (validator: `== sorted(decisive ∪ watch)` of `trained_categories(stage env)`),
`stage_floor` kind `category_survival` with the band test, `stage_retention` per category with the per-metric sign and
`stage_trace_v2`, `previous_categories`. Migration (D9): `migration_item(ctx)` ahead of `find_inflight` and
`_next_item_raw`; `load_resolution` at startup and a `resolution` queue item (`res.main([... '--out',
'resolution.<eval_sha[:12]>.json'])`) with `resolution_missing` halts; the rescreen/reeval/rescore handlers (with
`rp.set_seed`, `item_from_header`, `_carry_mismatch`, `screens_superseded`, `recs.append`, `rescores[]`,
`rescored_awaiting_predicate`); cause (f); `D00bs`/`D00s` writes with `seed_terms_for` preferring
`D00c-<chain>` → `D00bs` → `D00b`; the `resolution_inputs` validator branch. Milestones (D7):
`promotion.eval` from the yaml (default `PROMOTION_EVAL`), `_row_worse` per category, the ≥ 2-member rule with its nominal
rate, `resolve_eval_dir` for `milestone_reference`. Rendering: `_stage_report` one line per category
(`- suite fwd: n 100 surv 0.93 ratio 0.62 tripod 0.51 [decisive]` / `[WATCH]`, hands in parentheses); DECISION line fields
`trained:`, the per-category block rows, the untrained WATCH rows, and the turn-magnitude/encoding annotation;
`HOLD_SHORT`/`WATCH_LABELS` for `cat.` ids; RESOLUTION and MILESTONE lines per D9/D7. `_turn_hold_s` points at the suite's
`turn_l90`. A suite eval's heartbeat `current.iters` is 1000. `--smoke-stage 1a:20:suite` (third field = eval rows) for V6.

**B4. `$TOOLS/resolution.py`** (D10). **B5. `$TOOLS/ledger_cli.py`**: `show` prints `eval_sha` (`— (predates the suite)`
for D05/D06) and `trained:`; no new verbs.

**B6. `$LAD/candidates.yaml`** (`--accept-candidates-change` at relaunch): `screen: {chain: scratch_1c, eval: [suite],
iters: 8000}`; `promotion: {eval: [suite, light, ring]}`; `decision_rule` v2 (D5; `text` restated per category, frozen into
`D00s`); `resolution_inputs` = the three suite reeval entries (`checkpoint + preset + eval_dir`) + the two rescreen dirs as
`role: seed_pair` + the legacy rows (informational); the 2b process-floor row dropped (0.0 by determinism, amendment 19);
`milestone_reference` reeval roots; chain stages: `eval: [suite]` where they had `[slow]`/`[curriculum]`, `categories:` per
stage, floors `category_survival` (c1, c2, c4) / `none` (c3, c5, c6; `scratch_*` keep `dead_walker`).

**B7. Tests (tools).** `tests/unit/test_crab_hex_omni_candidates.py`: fixtures `screen_cat()` / `resol_cat()` (legacy
fixtures kept for the v1-rule tests, which call `decide()` directly and bypass the campaign validator);
`test_expand_rule_restricts_blocks_and_guards_and_watches_the_untrained`, `test_a_block_with_no_decisive_member_is_na_not_tie`,
`test_all_blocks_na_is_undecided_watch_no_pair`, `test_T_only_recipe_adopts_on_T_better`,
`test_untrained_category_regression_is_watch_never_worse`,
`test_move_turn_facing_cannot_be_won_by_walking_less` (F_mt BETTER with a `move_turn.ratio` collapse → clear REJECT; the same
F_mt gain with a within-R ratio drop → ADOPT; `F_mt` and `F` carry their own R and neither dilutes the other),
`test_a_survival_saturated_on_both_heads_is_scored_not_missing` (1.00 vs 1.00, n = 100 → TIE at R ≥ 0.0717; n = 200 ≥ 0.0507),
`test_category_floor_is_a_band_test_and_both_c1_and_c1b_pass_at_back`, `test_category_floor_new_prev_and_missing_category_halts`,
`test_retention_keys_on_categories_with_per_metric_sign_and_reseeds_hold_keyed_traces`, `test_screen_of_carries_trained_categories`,
`test_eval_sha_change_is_refused_without_acceptance`, `test_accept_eval_change_proves_the_training_tree_and_records_an_alias`
(temp git repo: an eval-file edit → alias; a training-file edit → refusal),
`test_ok_runs_and_baseline_label_treat_an_aliased_code_sha_as_current`, `test_ok_screen_refuses_a_stale_eval_sha`,
`test_eval_and_code_pathspecs_are_disjoint_and_cover_every_touched_file`,
`test_the_migration_runs_before_any_inflight_resume_and_before_the_baseline_rerun_rule` (a state with X05 `running` and a
stale `eval_sha` yields `rescreen X01` first), `test_launch_with_no_suite_resolution_defers_d00s_and_never_decides_on_the_old_record`,
`test_rescreen_of_a_seed2_header_files_under_seed2`, `test_rescreens_write_suite_beside_slow_and_keep_the_old_screen`,
`test_reevals_run_for_checkpoint_inputs_and_the_validator_accepts_that_shape`,
`test_d00s_then_d00bs_are_written_in_order_with_supersedes_and_the_seed_dominance_line`,
`test_a_rescored_header_appends_its_rec_and_re_opens_at_the_next_stage` (PASS → D05 squashed by (f), X03's `recs` gains
`X03:c4`, next stage c5, c4 never retrained; FAIL → new REJECT, old squashed),
`test_a_rescored_conditional_header_waits_for_its_predicate` (X04 is not resumed by `find_inflight`),
`test_c1c_skip_follows_c1s_squash`, `test_milestone_fails_on_two_worse_members_and_names_the_nominal_rate`,
`test_milestone_watch_when_the_suite_row_is_unreadable`, `test_stage_report_renders_one_line_per_category_with_decisive_marks`,
`test_decision_line_carries_trained_untrained_and_per_category_rows` (format/parse round trip),
`test_validator_rule_v2_and_category_stages`; existing tests edited by name: the curriculum stage-order test (`["suite"]×4`),
`test_one_experiment_trains_evaluates…` (`["suite"]`), the floor/retention/C1b-predicate tests (category form), the
`code_drop`/pathspec pins (the manifest and `run_phases.py` now in `EVAL_PATHSPECS`), the null-construction band (nominal
from the expanded rule). `tests/unit/test_crab_hex_omni_resolution.py`: `make_run(suite=True)`; `RULE` v2;
`test_stratified_draw_keeps_every_category_count`, `test_suite_pairing_on_env_index_pairs_within_category_and_hand`,
`test_one_member_block_R_eval_is_exactly_one`, `test_cat_ids_reproduce_summarize_cat_keys` (extends the registry test),
`test_survival_guard_R_has_the_binomial_floor`, `test_load_row_refuses_a_populated_hold_that_is_not_the_episodes_own`,
`test_resolve_eval_dir_takes_the_newest_stamp_or_refuses`, `test_seed_pair_role_is_excluded_from_cross_head_pairs`,
`test_legacy_pairs_are_bit_identical_under_the_stratified_path`, `test_summary_with_survival_uses_the_suite_block`,
`test_the_screen_row_is_the_suite`, `test_null_band_uses_the_stratified_draw_and_the_floored_guard_R`. The B0-pair control
check lives with `_write_d00bs` in the candidates module, not in the resolution tests.

## Part C — records, docs
`$LAD/CHARTER.md`: a new section "Per-category evaluation (2026-09-21)" carrying D1–D10, the suite table, the rule-v2 text,
the migration and resume sequence, and the caveats below; `$LAD/PLAN_AMENDMENTS.md` items 30+ (the old plan's decisions 4/5/6
superseded); `$OMNI/README.md` (the suite row, the categories module, `eval_sha`, the new metrics); `docs/FOLDER_LAYOUT.md`;
the memory note.

## Execution steps (ultracode: Workflow dispatch; routing per plan-model-routing; roster ok)
0. **Reference capture at HEAD `9c2c1fba`** → sonnet · low: the three cfg-identity suites into
   `$R/parkour/outputs/omni_plan/dumps_cat_base` (dir pre-created; 18 JSONs); `code_sha_at(HEAD, CODE_PATHSPECS_V1)`
   recorded; and `tests/unit/fixtures/gait_eval_compile_snapshot.json` = sha256 of `cmd`, `segment_id`, `steady_mask`,
   `delta_yaw`, `world_speed/dir/face` for every committed row of `scenarios_{v1,v2,morph}.yaml`, `scenarios_side.yaml` and
   `scenarios_omni.yaml` compiled at dt 0.02.
1. **Harness (A1–A5)** → opus · high (shared scorer; byte-identity of forward/sideways scoring is the catch), then three
   fresh lenses (legacy compile/score byte-identity incl. the golden and the step-0 snapshot; spec conformance vs D1–D3;
   test strength with mutation checks) and a fix loop to convergence.
2. **Categories module (B1)** → sonnet · high. Parallel with 1 (disjoint files).
3. **Tools (B2–B6)** → opus · xhigh, two implementers on disjoint files (A: `resolution.py` + its tests; B: the rest)
   sharing the `aggregate.suite` / `cat.<c>.<key>` contracts, then four lenses (numerics vs D5/D10 with a by-hand
   recomputation from a synthetic suite; state machine vs D9 incl. the re-open and cause (f); integration with the real
   seams and a dry run against a COPY of the paused campaign dir; contract/yaml/notify) and a fix loop.
4. Records and docs (Part C) → sonnet · medium.  5. Fresh-eyes review of the whole diff against this plan → fable · high.
6. Verification (below) → sonnet · medium for the runbook, main for the GPU steps.  7. Commits (Part G) → haiku · low.
8. **Relaunch and migration (D9)** → main: `ledger_cli unstop`, then `launch_candidates.sh --campaign-dir $LAD
   --accept-eval-change "per-category suite (user 2026-09-21)" --accept-candidates-change "categories, floors, rule v2"`;
   heartbeat; the persistent `notify.log` monitor relaying every line as push + chat card; read the `D00s` then `D00bs`
   lines (the F seed-dominance sentence first), then the C1 rescore verdict, then the queue. Stop-and-wait points as before.

Order: 0 → (1 ∥ 2) → 3 → 4 → 5 → 6 → 7 → 8. A step that fails twice reruns one notch up, never lower.

## Verification
1. **Pure suites**: `env -i HOME=$HOME PATH=$PATH $PY -m pytest tests/unit -k "crab_hex or travel_frame or parkour_command
   or exposure or policy_of_record or experiment_layout or gait or candidates or resolution or eval_categories" -q
   --continue-on-collection-errors -p no:randomly` green (baseline 1495 passed + 32 pre-existing collection errors); the
   golden fixture byte-identical; the step-0 compile snapshot matches.
2. **Config identity**: the three cfg-identity suites into `dumps_cat` — all 18 dumps byte-identical to `dumps_cat_base`.
3. **Suite dry run** (one Isaac process, a REAL head — `--zero-actions` falls in a second and never exercises the wave
   path), run through the eval seam so the environment is the composed one (`rp.eval_morph(ph.PHASES["1c"], ckpt, "suite",
   tag)` under a 48 G scope, `KRABBY_FLAT_TERRAIN_GEOM=ring` included): `--episodes 7 --waves 10`,
   `--trained-categories stand,fwd,side,back,diag,turn`. Asserts: `run_meta.episodes_total == 70`, `env_index_map` unique
   0..69 with n = 7 per hold, `wave_seeds` recorded, `schedule_state_max_deviation == 0.0`,
   `command_override_max_deviation < 1e-4`, 70 raw npz + 70 episode JSONs, `aggregate.suite.survival_by_hold` has ten keys,
   `by_category` has seven, `summarize()` returns seven `cat` keys, `summary.md` prints the category block; wall time and
   disk recorded. Then **cross-head wave determinism**: run the same suite on a second head (X01's ckpt) and assert wave 1's
   step-0 state (`root_pos_w[0]`, `root_quat_w[0]`, joint positions of `episode_007.npz`) is identical across the two heads.
4. **Driver dry run** against a scratch COPY of the paused campaign dir (v2 `candidates.yaml` + the recorded
   `state.json`/`ledger.json`), with a new dry-run section that iterates the real `migration_item` derivation and
   `find_inflight` order on those records (not the synthetic ledger): prints the alias proof (`code_sha_at` under V1 at
   `9c2c1fba` == `lad.code_sha`; the narrowed pathspecs equal at `9c2c1fba` and at the launch commit), the new `eval_sha`,
   and the queue `rescreen X01 → rescreen X02 → reeval 1c-004 → reeval 2b → reeval 2a-reeval → D00s → D00bs → rescore
   X03:c4 → rescore X04:c2 → C1 (c5) → X05 …` — the first item is a rescreen although X05 is recorded `running`; the
   validator accepts the checkpoint-input shape and refuses `rule_version != 2`; `ledger_cli show D05` prints
   `eval_sha: — (predates the suite)`.
5. **`resolution.py` on the reeval outputs** (step 8): `D00s` with per-category R, the binomial floors named, the null band,
   `R(D00bs) ≥ R(D00s)`, the B0 s2-vs-s3 construction check and its eval-only-R classing; `_reducer_check` passes on the
   real rows.
6. **Stop drill** on the suite path: `--smoke-stage 1a:20:suite` on a copied dir, `systemctl --user stop` during wave 3 →
   `await_user:stop`, no orphan, and the relaunch re-runs the whole suite (a half-written suite is rejected by the freshness
   rule).

## Part G — commits (branch `feat/crab-hex-omni-task`; nothing pushed)
1. `feat(gait_eval): category suite -- one hold per wave, waves in one launch, own-hold survival, pooled by_category, motion/direction metrics (world rows; body scoring byte-identical)` — includes the manifest row and its pin edit.
2. `feat(omni eval): eval_categories -- the trained categories of a recipe`
3. `feat(omni driver): eval-side seams -- suite summarize, trained-categories flag, own-hold survival`
4. `feat(omni ladder): decision rule v2 per category, eval_sha with the training-tree alias proof, the rescreen/reeval/rescore migration, category floors and retention, suite milestones`
5. `docs/record(omni ladder): per-category evaluation -- charter section, amendments, README`

## Non-goals
No change to training code, presets, rewards, the RSI bank or the command term; no `suite_ring__omni` (light/ring stay WATCH
rows); no uniform-direction `diag`; no in-band `turn_band` hold; no new categories; no retraining of B0; no change to the
ranked candidate list or its conditionals; no per-env tensors in the schedule hook; `drift_speed` stays WATCH.

## Caveats carried into the CHARTER
1. `move_turn` is scored RELATIVELY: the hold is infeasible by construction, so its facing block `F_mt` compares two heads
   under the same impossible demand, and `cat.move_turn.ratio` guards against winning it by walking less. Its `tripod`,
   `achieved_speed` and direction numbers stay WATCH, because a facing fix is expected to cost speed there.
2. `turn` is decisive for every recipe although each trains a 0.6-rad facing band against a 1.57-rad demand; the comparison
   is symmetric and the band/encoding are printed. The recorded 1c head settles on 1 of 62 turn episodes.
3. The F block is expected to stay seed-dominated (`R_F^seed` ≈ 1.6 estimated); `D00bs` measures it before any decision and
   its line says whether a single-seed screen can clear F at all.
4. `run_phases.py` sits on the eval side of the hash; its training seam is pinned by a source hash, not by `code_sha`.
5. The stage floors are absolute (0.10 new / 0.30 previous) and band-tested at 99 %: at n = 100 a stage passes down to
   p̂ ≈ 0.05 / ≈ 0.21, at n = 200 down to ≈ 0.06 / ≈ 0.23. They are calibrated on conditional per-hold survivals, a proxy the
   C1/C1b rescores validate with the first real per-category numbers.
6. A suite eval costs ≈ 2× a legacy row in time and ≈ 2.5–3× in disk; raws are kept (D00 re-scoring needs them).
7. `turn_r90` and `move_turn_r` change meaning relative to the legacy rows (both now mirror their left-handed twin from face 0).
8. `by_category` pools both hands; a handedness asymmetry is visible only in the WATCH `by_hand` rows.
9. Three blocks and nine guards cost sensitivity: a null candidate reads some WORSE about 15 % of the time (11 % under the
   pass-1 rule), so a clear REJECT is worth reading before it is trusted. Every DECISION line quotes the overrule command,
   and `decide C<id> --choose run` re-screens. The count is the price of judging every trained category rather than a pooled
   number; the nominal is printed on the RESOLUTION line and asserted against the measured null.

## Open risks
- `env.reset()` plus a per-wave override re-install is a new harness path, verified by V3 and the unit tests around
  `stack_waves`, not by a full-loop unit test.
- If the recorded launch commit is not in `git log -n 300`, the alias proof fails → fall back to `--accept-code-change`
  (which retrains B0); the CHARTER records which path was taken.
- The C1/C1b rescores may pass the floors and the resumed chains may still lose at their screens: that is the honest outcome.
- X05 (C2) resumes after C1's chain; if C1 is adopted, X05 is squashed by cause (a) — an accepted mispredict cost.

## Audit trail
- **Pass 1** (three lenses: goal inversion/coverage, evidence/numbers/tests, feasibility). **9 blocking, 19 substantial,
  ~44 minor — all applied.** Blocking: the migration was ordered inside `next_item`, so the first relaunch would have
  resumed C2 for 3.5 GPU-h and then halted `control_missing` (→ `migration_item` ahead of `find_inflight` and of the
  baseline-rerun rule); the `sincos` exception demoted `turn` for C5, the one candidate built for 90° facing (→ deleted, the
  encoding is an annotation); the stand drift guard was sign-blind, so standing stiller read WORSE (→ `drift_speed` is
  WATCH; the sign lives in RETENTION); a survival saturated on both heads read MISSING and would have rejected every
  candidate (→ scored at `max(R_seed, R_binom)` with a clipped `p̄`); `cat` could not be computed from per-hold `_stats`
  (→ `aggregate.suite.by_category` pools the episodes); `ensure_resolution` at startup would have died before the reevals
  (→ `load_resolution` + a queue item); the scorer needed one `(T, W·E)` schedule (→ `stack_waves`, `wave_of_env`);
  the suite's derived 11-hold schedule made `total_duration_s` 123.5 s and the row invalid (→ per-wave durations); five
  manifest/survival readers would have crashed or emitted garbage (→ suite-aware readers, own-hold ends, three injection
  sites). Substantial: D00bs must follow D00s; X04 must wait for its predicate; the ang_scale demotion must map through
  `TYPE_TO_CATEGORIES` (C1b/C7 keep `turn`); n per hand is 100; the rescore must append its rec key or retrain the stage;
  rescreen/rescore must `set_seed` (X02 is seed 2); `decisive` and `n_c` need an explicit plumbing path; `ok_runs`/`baseline_label`
  must read the alias list; the code_sha unit test was replaced by the V4 printed proof plus the step-0 snapshot fixture;
  V3's determinism check must compare two heads, not two runs; the all-TIE acceptance is a construction check.
- **User amendment (2026-09-21, on reviewing the pass-1 plan): `move_turn` must carry decisive and guard metrics.** Applied:
  it is an ordinary category (decisive when trained at the suite's demand, watch under a reduced one), its facing sits in its
  own block `F_mt` / `F_mt_abs` so it neither dilutes nor is diluted by the in-place turn block, and `cat.move_turn.ratio`
  (`achieved_speed` for absolute-units candidates) becomes a guard so the block cannot be won by walking less. Knock-on:
  three blocks and nine guards, null all-TIE ≈ 0.78, milestone FAIL rate ≈ 1 %, spurious-REJECT rate ≈ 15 % (caveat 9).
- **Pass 2 onward: none, by user decision (2026-09-21)** after reviewing the pass-1 plan. The per-step review lenses and fix
  loops inside execution steps 1, 3 and 5 are therefore the remaining defence, and step 5's fresh-eyes review of the whole
  diff is the last gate before the commits and the relaunch.
