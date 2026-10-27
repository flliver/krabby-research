<!-- APPROVED PLAN OF RECORD, 2026-09-21 — AUDITED (passes 2-6 after the pass-1 review, same day; accepted by the user at the
     6-pass cap with pass 6's two fixes un-re-audited; see the Audit trail at the end and PLAN_AMENDMENTS item 30).
     Copy of /home/nickmagus/.claude/plans/the-facing-and-tracking-fancy-breeze.md as approved, saved here so any later session can execute it without the approving session's scratchpad.
     It supersedes the evaluation half of PLAN_2026-09-18_facing_tracking_ladder.md (this directory): that plan's decisions 4
     (floors) and 5 (blocks/guards by hold), and decision 6's curriculum-row borrow (BORROW / --curriculum-runs, its A6),
     are replaced by D1-D10 below (decision 6's resolution / seed-term / pooled-R machinery stays); decision 7's sign rule
     becomes sign(Σ z_b / R_b) (D5), decision 9's pathspecs split into code_sha + eval_sha (D8), the squash view gains cause
     (f) and a cause-(b) exemption (D5, D9); and Part C (ii)'s
     milestone verdict keeps its three-row structure with the suite in place of `slow` (D7); everything else in it -
     the ranked candidate list, the speculation/squash protocol, the milestone structure - still stands.
     START HERE: execution step 0, then (1 || 2) -> 3 -> 4 -> 5 -> 6 (V1-V4) -> 7 (commits) -> 6b (V6) -> 8. The campaign is PAUSED (a STOP file sits in
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
env, W waves, n ≈ 100 per hold (200 for a paired category); (3) an
eval-only change **re-evaluates** the recorded heads instead of retraining the baseline (an `eval_sha` split from
`code_sha`; D00 re-resolved on the suite); (4) "only trained categories are decisive" reaches floors, RETENTION, the screen
decision and milestones. Refinement adopted after the design critics: **a wave is one hold for all envs** (scalar commands),
which keeps the omni command term — inside `code_sha` — untouched while giving the same outcomes.

Abbreviations: `$R = /home/nickmagus/krabby/krabby-research`; `$OMNI = $R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task`;
`$FWD = …/crab_hex_forward_task`; `$GE = $FWD/scripts/gait_eval`; `$EVAL = $FWD/scripts/eval_crab_hex_gait.py`;
`$TOOLS = $OMNI/experiments/tools`; `$LAD = $OMNI/experiments/2026-09-19_facing_tracking_ladder`;
`$MANIFEST = $OMNI/experiments/eval/scenarios_omni.yaml`; `$PY = /home/nickmagus/krabby/isaac_venv/bin/python`.
Branch `feat/crab-hex-omni-task`, HEAD `464be672` (the record commit that saved this plan, on `9c2c1fba`, itself a record
commit on the ladder's `9f61eb2`; tree clean). **Measured and load-bearing (re-measured in audit pass 2 with the driver's own
`code_rows` format and drop globs):** the code-bearing tree is identical at `81a1aa3`, `75dd943`, `9f61eb2`, `9c2c1fba` and
`464be672` under both the current pathspecs (`4c3f99c9a2c5…` = `lad.code_sha`, 127 rows) and the narrowed training-side
pathspecs (`a617f5ee59d1…`, 121 rows), so D8's alias proof succeeds and no baseline retrain is needed. (`code_sha_at` reuses
`code_rows`' exact row format, so V1 at any of these commits reproduces `lad.code_sha`; the pass-1 values `3175fc88c44a` /
`62f6f82a1a4a` came from another row format and must not be compared with V4's printout.)

Standing constraints: forward and sideways tasks untouched (cfg dumps byte-identical; the gait_eval golden fixture is
body-rows-only and stays byte-identical — every new key is world-row-only, gated like `facing`); the three legacy omni rows
are shape-frozen; one Isaac process at a time; nothing is pushed; the keep-set rule; no `__init__.py` under `experiments/`;
the paused campaign's records are appended to, never rewritten; `$OMNI/mdp` and `$OMNI/config` do not change.

## Vocabulary
- **category** — one of `stand, fwd, side, back, diag, turn, move_turn`. **hold** — one command row of the suite; a category
  has one hold (`stand, fwd, back, diag`) or a left/right pair (`side: lat_l, lat_r`; `turn: turn_l90, turn_r90`;
  `move_turn: move_turn, move_turn_r`). **hand** — `l | r` of a paired hold. The **settle** hold (2 s, `settle_hold_s`) is
  never an episode's own hold; not to be confused with `defaults.settle_s` (1.0 s, the transient excluded after a command
  change, used by the validators) or the facing metric's settle time (`settle_s` in `cat`, = `facing.settle_time_s`, WATCH).
  **n_c** — a category's episode count (the guards' binomial floor); **n_env** — its distinct-env count (the floor band).
- **suite** — the manifest row `suite__omni`: ten waves in one launch; wave k = every env runs `[settle 2 s, hold_k 12 s]`.
- **trained categories** of a stage — what its command distribution covers (`eval_categories.trained_categories`, D4):
  `decisive` (share ≥ `POOL_FLOOR_SHARE` at the suite's full demand), `present` (share > 0), `watch` = present ∖ decisive,
  with a reason note: `reduced demand` (a scale edit) or `one hand` (a single-band `side`); share = weight / Σ weights; the
  **stage env** they are derived from is the composed training env `ph.phase_env(stage.base) ∪ stage_env(recipe, stage)`,
  never the overlay alone. **decisive set** of a comparison = the CANDIDATE's decisive categories;
  **guarded set** = the candidate's present categories (their `survival` guards stay, D5).
- **N/A block** — a block with no decisive member on this comparison; never TIE, never MISSING.
- **eval_sha** — sha256 over the blob ids of the eval-side pathspecs (D8); **code_sha** — the training-side tree (today's
  list minus the eval-side files). **alias** — an older `code_sha` proven equal to the current one on the training-side tree.
- **rescreen** — run the suite on a recorded ok screen's checkpoint and file it into that record; **reeval** — run the suite
  on a recorded head that is a D00 input (no header); **rescore** — run the suite on a floor-rejected chain's failed-stage
  checkpoint when that floor was measured under an older eval, and re-decide it. All three are eval-only — none trains; they are not the ladder's "re-screen" (`decide
  C<id> --choose run`), which retrains, and their notify lines say "eval-only". **migration** — the ordered set of those items
  plus D00s/D00bs (D9). **unit** (milestones, D7) — a block or a counting guard; a block member is never classed alone.

## Design decisions

**D1. Suite geometry (harness).** `suite__omni` = `kind: category_suite`, `task: Isaac-Crab-Hex-Omni-Walk-Play-v0` (the only
task this harness can build for the omni rows), `role: teacher`, `terrain_label: flat`, checkpoint supplied per run, no
`env:` block, `episodes: 100` (envs per wave),
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
| 5 | `diag` | diag | 0.25 | +π/4 | 0 | = legacy (fixed hold, not a uniform sample: D00s pairs on `env_index`) |
| 6 | `turn_l90` | turn / l | 0.00 | 0 | +π/2 | = legacy (90° left from face 0) |
| 7 | `turn_r90` | turn / r | 0.00 | 0 | −π/2 | **new meaning**: 90° right from face 0 (legacy: +π/2 → 0) |
| 8 | `move_turn` | move_turn / l | 0.22 | 0 | +π/2 | = legacy (world-held travel while the body turns) |
| 9 | `move_turn_r` | move_turn / r | 0.22 | 0 | −π/2 | **mirror** of `move_turn` |

Every episode starts at the pinned spawn yaw 0 and its prologue (1.5 s) holds the settle command, so a turn hold's
`heading_error_initial_abs` is ≈ 1.5708 (only the standing yaw drift over the 3.5 s deviates) and `facing_closed` has a
near-constant denominator; translation holds start with facing error ≈ 0. Feasibility: 90° needs `1.0 + 1.5708/0.3 = 6.24 s
≤ 12 s`; every speed clears the ellipse cap at both the start and end body angle (0.35 / 0.25 / 0.28766 / 0.25).
**n:** 100 episodes per hold → **100 per hand**, 100 for a singleton category (`stand, fwd, back, diag`), **200** for a
paired one (`side, turn, move_turn`). The ten waves reuse the same 100 envs, and startup-mode domain randomisation
(friction; mass/COM where a preset arms them) is drawn once at load, so env e is the same robot in every wave: a paired
category's 200 episodes are 100 robots × 2 hands, and every category shares those robots. Hence: `n_c` in the guards'
`R_binom` is the category's episode count — a floor where `R_eval` is informative, but the operative R at saturation, where
it treats the two hands as independent and no D00s null can test that approximation (both resample one run); the
env-clustered `R_eval` carries the real spread elsewhere; the floor band (D6) uses
the category's **distinct-env count `n_env` (100)**; the bootstrap is clustered on env (D10).

**D2. Waves (harness).** The main loop runs W waves in one Kit process. Wave k compiles its own two-hold uniform schedule
(scalar `ScheduleRow`s — the omni `schedule_hook` is untouched, pinned by a source-hash test); before every wave (wave 0
included; `wave_seeds` has W entries) the harness calls `env.seed(env_seed * 1_000_003 + wave)` (`ManagerBasedEnv.seed` → `configure_seed`, covering torch/cuda/numpy/random; recorded as
`wave_seeds`), then for wave > 0 `env.reset()` (all envs re-spawn at the pinned yaw), **re-installs the override with that wave's compiled
schedule**, applies row 0, re-reads observations, zeroes the per-wave buffers and bookkeeping, and runs the existing step
body verbatim; a fallen env stays frozen for the rest of its wave (as today). The re-seed is what makes wave k's initial
states independent of how many envs fell in wave k−1 (a head-dependent RNG draw), which is what `env_index` pairing across
heads requires. Episodes are indexed `env_index = wave × episodes + env` (0..999 at episodes 100, unique; identical
`(episodes, waves, suite)` across heads is the pairing precondition, checked from `run_meta.env_index_map`). **Only a
`category_suite` row takes this path:** every other row (W = 1: legacy omni, forward, sideways) runs today's loop verbatim —
no `env.seed` call, no extra reset, no re-install — because `configure_seed` writes process state (`PYTHONHASHSEED`,
torch/cudnn flags) and any env write after Kit startup can flip same-seed evals between two repeatable outcomes. `segment_id` carries ROW-GLOBAL hold
indices into the union `hold_labels = [settle, stand, fwd, …, move_turn_r]` (11 entries, settle = 0), so every
position-keyed reader (`rescore_facing`, `_columns`) stays correct. GPU footprint per wave ≈ one fifth of a `slow__omni`
row's sim time; host ≈ 1.5 GB; disk ≈ 0.6–0.7 GB raw (922 B per env-step on the recorded X01 row) + ≤ 240 MB JSON (≈ 10×
a legacy row's 25 MB) per suite run — ≈ 0.85–0.95 GB, ≈ 3.2–3.5× a legacy row's 0.27 GB. Cost, at a
0.047 s/step + 26 s fixed fit (a fit to the recorded legacy rows, 3.3–3.9 min each per the CHARTER budget — not a recorded
constant): 10 × (775 steps ≈ 36 s sim + reset) + ~30 s fixed + the driver's 30-s tick ≈ **7–8 min per suite eval**.

**D3. Metrics per category (harness, world-row-only, additive).** Existing per-hold blocks stay keyed by label
(`tracking_by_hold`, `facing_by_hold`, `tripod_score_by_hold`). New per-hold entries, gated exactly like `facing`:
`motion` = `{speed_abs_mean_m_s (mean ‖v_xy,body‖), pos_drift_m, path_len_m, yaw_drift_rad, foot_lift_count,
contact_churn_per_s, n_steps}` over the whole hold; `direction` = `{direction_error_rad, cross_track_ratio,
travel_speed_m_s}` from the travel-frame velocity already computed for tracking (`travel_speed_m_s` is the speed along the
travel frame, WATCH only — the category key `achieved_speed` is `tracking.vx.actual_mean` and is the only speed a block or
guard reads), emitted **only when the hold's commanded speed > 0** (so `settle`/`stand` carry none). Aggregates
`motion_by_hold`, `direction_by_hold` (`_stats`).

On a suite row the aggregate gains **`suite`** = `{waves, wave_seeds, episodes_total, dt, prologue_s, settle_hold_s,
hold_s, hold_labels (union), hold_of_wave: [labels], category_of_hold, hand_of_hold, n_by_hold,
survival_by_hold (each hold's OWN survival = the fraction of its episodes with n_steps ≥ round((prologue_s + settle_hold_s +
hold_s)/dt) — asserted equal to the wave's compiled step count — the legacy `survival_by_hold` convention: a fall on the
wave's last step counts as survived, exactly as the tools' `_columns` `surv` counts it — never `term_reason ==
schedule_complete`), trained_categories, by_category: {c: {key: _stats}}}`. `by_category` is computed in
`report.aggregate` by POOLING the per-episode values of the category's own-hold entries over both hands (never by averaging
hand medians — the pooled median is what the env-clustered bootstrap (D10) resamples and what `_reducer_check` reproduces): `ratio ← tracking.vx.ratio`, `achieved_speed ← tracking.vx.actual_mean`,
`tripod ← tripod.tripod_score` (excluding `low_confidence`), `facing_closed | yaw_progress | net_yaw | heading_final ←
facing.*`, `settle_s ← facing.settle_time_s`, `drift_speed ← motion.speed_abs_mean_m_s`, `direction_error |
cross_track_ratio ← direction.*`, plus `by_hand: {l|r: {key: _stats}}`. `aggregate(episodes, *, suite=None)` receives
`run_meta["suite"]` (dt, prologue_s, settle_hold_s, hold_s, categories) together with `run_meta`'s `waves`, `wave_seeds`,
`episodes_total` and `trained_categories` from `write_run`/`summary_text` — the manifest's `CategorySuite` alone carries no
run-time fields; episodes carry `wave`, `hold`, `category`, `hand`.
Nothing else in `aggregate` changes; on a suite row the pooled keys (`schedule_completion_rate`, `tripod_score`,
`fall_classes`, `orientation`, `support_polygon`, `slip_ratio`, …) are means over all ten holds and are **WATCH-only**
downstream. Category metric roles:

| category | decisive members | guards | WATCH |
|---|---|---|---|
| stand | — | `survival` | `drift_speed`; per-hold `motion_by_hold.stand` (`pos_drift_m`, `yaw_drift_rad`, `foot_lift_count`, `contact_churn_per_s`) |
| fwd, side, back, diag | T: `ratio`; T_abs: `achieved_speed` | `survival`; `cat_tripod_min` (min of the category tripod medians over the decisive translation categories) | `direction_error`, `cross_track_ratio`, `heading_final` |
| turn | F: `facing_closed`, `yaw_progress`; F_abs: `facing_closed`, `net_yaw` | `survival` | `settle_s`, `heading_final`, per-hold `motion_by_hold` `pos_drift_m`, both hands |
| move_turn | F_mt: `facing_closed`, `yaw_progress`; F_mt_abs: `facing_closed`, `net_yaw` | `survival`; **`ratio`** (the anti-sacrifice guard; `achieved_speed` under `absolute_units`) | `achieved_speed` (ratio-units candidates), `tripod`, `direction_error`, `cross_track_ratio`, `settle_s`, both hands |

The `survival` guard applies to every PRESENT category (decisive ∪ watch, D5); every other member and guard applies to
decisive categories only.

`drift_speed` stays WATCH this window: it is lower-is-better, and the guard machinery is one-sided higher-is-better, so
making it decisive needs a sign mechanism that only RETENTION requires (D6). A standing collapse is caught by
`cat.stand.survival`. A later window may promote it once D00s shows it resolves.

**`move_turn` is decisive, in its own block, paired with a translation guard.** The hold's commanded demand is infeasible by
construction (walking at the 0.21 m/s floor leaves ω ≤ 0.01 rad/s at `r_foot` 1.2 m — a statement about the command, not the
heads' operating point), so both sides of every comparison face the same impossible demand and the facing numbers are a
**relative** measure — which is exactly the campaign's question, and where the
sacrifice is largest (the 2b head ends at 1.218 rad there versus 0.68 on the in-place turn). Two rules make it honest:
its facing sits in **its own block `F_mt`**, not inside F, so the noisiest facing member cannot drown the in-place turn signal
(and vice versa); and `cat.move_turn.ratio` is a **guard**, so a candidate cannot win `F_mt` by stopping to turn. The guard
fires only on a drop beyond `max(R_eval, R_seed)`: about 0.05–0.09 in ratio units on the recorded legacy numbers (v1's WATCH
`R_eval` 0.043 at 95 % for `ratio_by_hold.move_turn`, ≈ 0.05 at 99 % by Gaussian extrapolation, not recorded; the B0 seed pair differs by 0.087; D00s/D00bs set the
suite value). A facing gain that costs less speed than that can ADOPT; abandoning the translation is a clear REJECT. The only
recorded trade is a seed difference, not a trade curve (B0 s2 vs s3: +0.18 `facing_closed` at −0.087 ratio, right at the
guard's R), and the recorded heads walk this hold at ratio 0.35–0.54 (≈ 0.08–0.12 m/s, well below the walk floor), so a
facing gain has room to come from turning better rather than walking less; a REJECT whose only WORSE unit is
`cat.move_turn.ratio` while F_mt is BETTER is the signature of a speed-for-facing trade and is read before it is trusted.
`tripod` there stays WATCH (it is not a translation category and is excluded from `cat_tripod_min`). For a candidate that
declares `absolute_units` the guard is `cat.move_turn.achieved_speed` instead of `ratio` (a guard-level spec with its own seed
term: `guards_abs`, D5), for the same reason T_abs swaps the block member.

**D4. Trained categories (omni package).** New Isaac-free module `$OMNI/eval_categories.py` (task root, outside `mdp/` and
`config/`, hence outside `code_sha`) which loads `crab_hex_omni_command_math.py` and `config/crab_hex/omni_knobs.py` **by
path** (`importlib.util.spec_from_file_location`, the `run_candidates.pool_math()` pattern — a dotted import would run
`parkour_tasks/__init__.py`'s Isaac `import_packages`): `CATEGORIES` (no category is WATCH-only by construction — a category
is decisive when the recipe trains it at the suite's demand, and watched otherwise),
`CATEGORY_HOLDS`, `HAND_OF_HOLD`, `TYPE_TO_CATEGORIES` (`stand→stand, fwd→fwd, side_l|side_r→side, back→back,
omni→{fwd, side, back, diag}, turn→turn, fwd_turn|side_*_turn|back_turn|omni_turn→move_turn`),
`TrainedCategories(decisive, present, watch, hands, demand, source, facing_band_hi, encoding, notes)`,
`trained_categories(env: Mapping) -> TrainedCategories`, `category_of_hold(label)`, `decisive_set(tc)`; and the two pure
helpers both tools must share instead of re-implementing: `expand_rule(rule, decisive, present, *, absolute_units)` (D5)
and `composite_guards(screen, decisive)` (`cat_tripod_min`). A key absent from the stage env (the 1c and 2a/2b presets carry
no `KRABBY_OMNI_DIR_BAND`; several omit FACING) takes the command cfg's class default, held as constants here and pinned
against the class body by a B1 test.
Pool mode: present at share > 0, decisive at share ≥ `POOL_FLOOR_SHARE` (share = weight / Σ weights — pool weights are raw
and a `set_share` edit need not renormalise them; slot probabilities are normalised the same way); `side` is decisive only when both `side_l` and
`side_r` (or `omni`) are; `omni` makes `fwd, side, back, diag` decisive (uniform-bearing support; note recorded). A scale
edit demotes **only the categories its own type maps to** through `TYPE_TO_CATEGORIES`: `ang_scale < 1` on `*_turn`/`omni_turn`
touches `move_turn` alone (so C1b's and C7's `{"*_turn": 0.3}` leave `turn` decisive — `fnmatch("turn", "*_turn")` is False);
`ang_scale < 1` on the `turn` type demotes `turn`; `speed_scale < 1` demotes that type's translation categories. Slots mode
(legacy presets): slot 0 → stand; slot 1 → `omni` if `dir_band` is None else the types whose band lies inside it (tolerance
1e-4); slot 2 → turn; slot 3 → `omni_turn`/band types; the slot share is the enabled share. So `move_turn` is decisive for
B0 and every `scratch_1c`/`scratch_c6` recipe (slot 3 at 0.20–0.25, `omni_turn` in the pool) and from curriculum stage c4;
it is **watch** for C1b and C7, whose `ang_scale 0.3` on the `*_turn` types trains it at 30 % of the suite's turn demand
(watch keeps its `survival` guard: D5, user decision 2026-09-21 in audit pass 2).
**`turn` is decisive for every
recipe**, including under `FACING_OBS_ENCODING=sincos`: the encoding changes what the policy observes, not which command
category it trained, and the comparison with the control is symmetric at the command. `facing_band_hi` and the encoding are
**annotations** printed on the RESOLUTION and DECISION lines (`turn: demand 1.571 rad > FACING band 0.600; encoding raw
clips at 0.6 / sincos sees the magnitude`), never a demotion. A stage whose `stand` share is 0 is a `halt:eval_mismatch`
(the suite's settle hold is part of every category's episode). Derived: B0/`scratch_1c` → decisive {stand, fwd, side, back,
diag, turn, move_turn}; curriculum c1 {stand, fwd}, c2 +{side, back}, c3 +{turn}, c4 +{move_turn}, c5 +{diag}.
Recorded in the experiment header per stage as `trained_categories_by_stage[stage]`. **Headers that predate the record**
(X01–X05 and every milestone header): any reader of a stage's sets computes `trained_categories(ph.phase_env(stage.base) ∪
header.overlay_by_stage[stage])` and appends it to the header (append-only: a new key, nothing else rewritten); the
rescreen and rescore handlers (D9 items 1, 5) write it BEFORE launching their suite eval, so the eval gets
`--trained-categories`, `_screen_of` copies it, `_write_d00bs`' construction check and the rescore floor read it, and
D6's min_prev/min_new split computes it for every earlier stage too (`eval: []` stages included). The sets are passed to
the harness as `--trained-categories` (recorded in `run_meta`,
mirrored into `aggregate.suite.trained_categories`; scoring never reads it), copied onto the screen dict by `_screen_of`
(which summarises the final stage's `suite` row and never `slow` when a suite row exists: B3; its source is
`header.trained_categories_by_stage[final stage]`, and an empty or missing set on a suite screen halts `halt:eval_mismatch`
rather than reaching decide's all-N/A SKIP), and printed on the REPORT and
DECISION lines.

**D5. Decision rule v2 (driver).** `summarize()` gains one additive key `cat` (`{}` when the aggregate has no `suite`):
`cat[c] = {n, n_env, survival, ratio, achieved_speed, tripod, facing_closed, yaw_progress, net_yaw, settle_s, heading_final,
drift_speed, direction_error, cross_track_ratio, by_hand: {l, r} | None}`, each read as
`aggregate.suite.by_category[c][key].median` (`survival` = the n-weighted mean of the category's holds' own survival;
`n` = the category's episode count; `n_env` = `episodes_total / waves`, the distinct-env count the floor band reads). Every existing key is unchanged; on a suite row the legacy per-hold keys are the
per-hand readings and `turn_survival_l90/r90` are `None`.

`candidates.yaml decision_rule` becomes `rule_version: 2` with `per_category` templates: F `{turn: [facing_closed,
yaw_progress]}`, **F_mt `{move_turn: [facing_closed, yaw_progress]}`**, T `{fwd|side|back|diag: [ratio]}`,
F_abs `{turn: [facing_closed, net_yaw]}`, **F_mt_abs `{move_turn: [facing_closed, net_yaw]}`**,
T_abs `{…: [achieved_speed]}`; guards `{every category: [survival]}` + **`{move_turn: [ratio]}`** +
`composite: [cat_tripod_min]`, and **`guards_abs: {move_turn: [achieved_speed]}`** in place of the ratio guard for an
`absolute_units` candidate (resolution computes both at guard level, 0.99 one-sided; D00bs records seed terms for both); and
the watch list: D3's WATCH column for each present category, restricted to `cat` keys (the per-hold `motion_by_hold` entries
there are summary.md only), plus a watch category's non-survival members and guards with their note (`direction_error`,
`cross_track_ratio`, `heading_final`, `settle_s`, `stand.drift_speed`, `move_turn.tripod`, `move_turn.achieved_speed` in
ratio-units expansions), both hands of every paired category, and the legacy `completion`, `tripod_walking_dir`,
`track_ratio_tail`, `track_ratio_move_tail`; `turn_survival_*` stay listed as print-only ids that read `N/A (suite)` —
undefined on a suite row, they land in `specs_from_rule`'s unresolved list and never reach `_reducer_check` (D10); the
per-hold `motion_by_hold` keys are `summary.md` only. A candidate scores either
{F, F_mt, T} or, when `absolute_units` is declared, {F_abs, F_mt_abs, T_abs} — three blocks, as today's two became three.
**The block set is derived from the expanded rule, never hard-wired:** `block_names(cand_spec, rule)` returns the blocks of
the rule actually passed — (F, T) / (F_abs, T_abs) for the frozen v1 `by_row_rule` on light/ring and the v1-rule tests — and (F, F_mt, T) or
(F_abs, F_mt_abs, T_abs), and every place that enumerates the literal (F, T, F_abs, T_abs) tuple or a two-block pair iterates
the expanded rule's blocks instead (the seams are listed in B3/B4; a two-block null would read all-TIE ≈ 0.82, inside the
acceptance band, so no band check can catch a missed seam — B7 pins each one).
`expand_rule(rule, decisive, present, *, absolute_units)` (hosted in `eval_categories.py`, D4)
produces today's flat shape restricted to the decisive set (ids `cat.<c>.<key>`, `cat_tripod_min`), keeps the `survival`
guard of every PRESENT category (a category trained at reduced demand must still not fall more at the suite's demand), and
moves every other member and guard of a non-decisive category into the WATCH dict with `note: untrained` or `note: watch
(reduced demand)` / `note: watch (one hand)` — the informational half of the request, printed on the DECISION line. A block
with no member is **N/A** (`{class: "N/A", k: 0}`), never TIE. **Resolution takes both unit systems:** `resolve()` (D00s)
and `seed_terms_from_pair` (D00bs) use the UNION of `expand_rule(rule, decisive(B0), present(B0), absolute_units=False)` and
`(…, absolute_units=True)` — blocks F, F_mt, T, F_abs, F_mt_abs, T_abs and their members; guards the seven `survival`,
`cat.move_turn.ratio`, `cat.move_turn.achieved_speed` and `cat_tripod_min` — so an `absolute_units` candidate (C3a/C3b) is
decidable and the both-systems control check has every term; `_screen_compare` still scores each candidate on its own
system's three blocks, and `null_construction` scores the ratio system only (the nominal 0.95³ × 0.99⁹). `decisive(B0)` =
`eval_categories.trained_categories` of the baseline chain's final-stage composed env, recorded in D00s as `decisive_at`;
a comparison in which a NON-EMPTY block, or a non-empty `cat_tripod_min`, has a member set that is a strict subset of its
`decisive_at` members (a narrower T or tripod-min) halts `resolution_missing` naming the set, because every published
block R and `cat_tripod_min` R was measured on B0's members; a memberless block is N/A and never halts — after a C1b or C7
adoption F_mt / F_mt_abs is N/A with its members on WATCH (no current screen hits the halt: every final stage trains all
four translation categories).

Plumbing: `_screen_compare` gains `decisive: frozenset | None` and `present`, expands the rule internally, injects the
shared `composite_guards` into shallow copies of every screen compared (four when pooled), and marks memberless blocks N/A;
`decide`, `decide_pooled`, `anchored_drift` and `_write_d00bs` derive `decisive`/`present` from the CANDIDATE screen's
`trained_categories` (a category the base never trained stays decisive with `note: base untrained`); at a milestone
`_row_worse` receives no screen (it gets `summarize_run_dir` output), so it takes `decisive`/`present` as explicit arguments
(D7);
`_speculative_pick`, `_any_worse` and `_better_blocks` skip N/A blocks; `guard_rows` takes `n_c = min(cat.<c>.n of both
sides)` for the binomial floor; `seed_terms_from_pair` receives the rule already expanded at `decisive(B0)` and applies the
same `composite_guards` to shallow copies of both screens before reading any guard, so `cat_tripod_min` has an `R_seed`.
**Speculative sign (user decision 2026-09-21, audit pass 2):** the all-TIE pick, `decide_pooled`'s per-seed signs and its
pooled-sign fallback use `sign(Σ_b z_b / R_b)` over the non-N/A blocks — each block in units of its own resolution; a raw
`Σ z_b` would follow F_mt, whose recorded seed spread is ≈ 3× F's (caveat 3); the `why` strings name the formula.

`decide`: any WORSE → clear REJECT; any guard MISSING → REJECT `confidence: uncertain`, no pair; **all scored blocks N/A → a
terminal entry `status: decided`, `chosen: SKIP`, `confidence: none`, `evidence.source: no_decisive_block`,**
`why: "no decisive block (trained: …)"`, no pair — SKIP is already a candidate option (no new status; `confidence: none` is
a new value); squash_view's cause (b) exempts it as it exempts a user SKIP (it answers the screen, not the predicate — a
conditional candidate is only screened when its predicate holds, so cause (b) would otherwise squash it at once and
re-screen it forever), while causes (a) and (c) still apply; dependents proceed; the DECISION line quotes the `overrule`
command; a narrow recipe is never rejected for being narrow (unreachable with the current ranked list: every final stage
trains a translation or turn category); otherwise today's branches over the non-N/A blocks. Guards are
one-sided at 99 % on `R_i = max(R_eval,i, R_seed,i, R_binom,i)`, where `R_binom,i` exists for the `survival` guards only (0
for every other guard) with `R_binom = z_0.99 · sqrt(2 p̄(1−p̄)/n_c)` and `p̄ = clip((surv_a + surv_b)/2, 0.05, 0.95)` — so a
survival saturated on both heads floors at 0.072 (n = 100) / 0.051 (n = 200) instead of 0. **MISSING is reserved for a
missing VALUE:** a survival guard whose `R_eval` is None or 0 is scored against `max(R_seed, R_binom)` and records
`R_from_floor: binomial`, never `missing: "resolution"` (for `survival` guards `guard_rows` applies this rescue whatever
`R_seed` is; every other guard keeps today's `r_seed > 0` condition and stays MISSING without an R — an unresolved
`move_turn.ratio` or `cat_tripod_min` must never read as a silent TIE). **Pooled** (`decide_pooled`, pooled anchored
drift): `guard_rows`' pooled branch gives a survival guard `max(sqrt((R_eval/√2)² + R_seed²), R_binom/√2)` with p̄ clipped
over the four screens and `n_c` the min over them; a pooled drift with no anchor s2 (`mean(a, a2) − b`) floors at
`R_binom·√3/2`; `resolution.pooled_guard_R` gains an optional `r_binom` argument (default None keeps today's behaviour and
its existing test) for resolution's own diagnostics — so a saturated survival is never MISSING on the pooled path either. `completion`, `tripod_walking_dir`, `turn_survival_*`, `track_ratio_*` → WATCH. Anchored drift,
pooled decisions, `absolute_units`, `speculative_default` and the MISSING-value semantics are otherwise unchanged, on the
expanded rule. Evidence records `decisive`, `untrained: {candidate, base}`, and every N/A block.

**D6. Floors and RETENTION (driver).** `floor: {kind: category_survival, row: suite, min_new: 0.10, min_prev: 0.30}` on
every non-final chain stage with an eval row (final stages `none`; the screen's guards cover them). For each
`c ∈ decisive(stage)`: `c ∈ decisive(stages[:i])` (any earlier stage, including one with `eval: []`) → threshold `min_prev`,
else `min_new`. The test is a **band test**: FAIL only when the one-sided 99 % upper binomial bound
`p̂ + z_0.99 · sqrt(p̂(1−p̂)/n_env)` is below the threshold, with `n_env` = `cat.<c>.n_env`, the category's **distinct-env
count** (100 for every category: a paired category's two hands share each env's startup-randomised physics, D1); the stage report's `- floor` line
and the DECISION line print point, bound, n and the failing categories (FLOOR is not a notify tag).
Effective point thresholds (n = 100): a stage passes down to p̂ ≈ 0.05 (new) / ≈ 0.21 (previous).
A decisive category the suite row does not carry is a `halt:eval_mismatch` (not a silent pass); "not scored (no verdict)"
survives only for a stage whose eval list is empty. On the recorded conditional per-hold survivals — the proxy for the
per-category number (C1 c2 `back` 0.58, C1b c2 0.52, C1 c4 `fwd` 0.82 / `lat` 0.89–0.96 / `back` 0.83 / `turn` 0.88–0.90 /
`move_turn` 0.66–0.76, new at c4 → `min_new`) — neither D05 nor D06 would have fired. RETENTION compares `cat.c.survival`
and the category's retention metric (`ratio` for a translation category; `facing_closed` for turn and move_turn, plus
`ratio` for move_turn; `drift_speed` for stand — WATCH in screens, tracked here — **lower is better**: the per-metric sign
lives here) for every
`c ∈ previous_categories = ∪ decisive(stages[:i])` against the incumbent trace (`stage_trace_v2`, keyed by category; a hold-keyed trace is re-seeded with
the line "trace predates the suite"); WATCH + notify. Thresholds per metric kind: survival 0.10 (unchanged); `ratio` a 20 %
relative drop (unchanged); `facing_closed` — a signed fraction that can sit at or below 0, where a relative drop flips sign or
fires on noise — an ABSOLUTE drop of `max(0.10, R_eval)`; `drift_speed` a rise of at least 0.02 m/s absolute and 20 %
relative (lower is better).

**D7. Milestones (driver).** Promotion stages evaluate `[suite, light, ring]` (from `candidates.yaml`'s `promotion.eval`,
default `PROMOTION_EVAL`, consumed in `promotion_stage_list`, which sets both promotion stages' `eval` from it;
`final_s2_stage_list` and `item_from_header`'s milestone branch inherit it through that function; `ph.EVAL_FULL` in
`$OMNI/config` — inside `code_sha` — is never edited). **The verdict keeps the 2026-09-18 three-row structure** (old Part C (ii), amendment 16; user
decision 2026-09-21 in audit pass 2) with the suite in place of `slow`: the 2b head versus the previous milestone's 2b head
**FAILs iff ≥ 2 of the three 2b rows {suite, light, ring} are WORSE**; unreadable rows keep today's handling (fewer than two
WORSE with a row missing → WATCH). The **suite row** is WORSE iff ≥ 2 decisive UNITS are WORSE at `by_row[suite__omni]` —
a unit is a block (F, F_mt, T or their `_abs` twins, as `_screen_compare` classes them; a block member is never classed on
its own) or a counting guard (the per-category `survival` guards of the 2b head's present categories, and the move_turn
translation guard: `ratio`, `achieved_speed` under `absolute_units`); `cat_tripod_min` does not count; nominal suite-row
WORSE rate under the null ≈ 1 % (3 blocks + 8 counting guards at B0's set), printed on the MILESTONE line. **light and
ring** keep the frozen v1 per-row rule byte-for-byte (their scoring does not change under this plan, so D00's
`by_row[light__omni|ring__omni]` R stays valid). Its three sources are named: **the rule** — D00s copies D00's frozen
`decision_rule` (`D00.evidence.decision_rule`, identical to the legacy `resolution.json` one) verbatim as `by_row_rule`
beside the two copied tables (`by_row_source: D00`); **the R** — those copied tables; **the seed terms** — D00b's v1 terms
(the v1 lineage, slow-row terms borrowed across rows as today; the one sanctioned D00b citation under rule v2). `_row_worse`
on a row whose table carries `by_row_source: D00` scores with `by_row_rule` — no `expand_rule`, no decisive/present — and
today's `_any_worse` (any WORSE block or guard makes the row WORSE); the ≥ 2-unit test applies to the suite row alone;
today's ring-completion clause is unchanged (ring completion scored from M2 per amendment 16). The suite row's
decisive/present sets are `trained_categories` of the promotion stage's composed env (all seven for a slot-mode 2b preset;
after an adopted `*_turn` scale edit move_turn is watch there, because `rescale_to` keeps ang_scale — its survival guard
stays, D5): `_close_milestone_promo` reads them from the promotion header's per-stage record (D4's rule, which also covers
headers that predate it) and passes them explicitly to
`_row_worse` and to the unit count; the 2a WATCH rows and `_reference_watch` use the 2a stage's sets; the candidate path's
`_run_row` passes `--trained-categories` to `eval_morph` for suite rows only. `_close_milestone_promo`'s
`("slow", "light", "ring")` literal iterates `promotion.eval`. M1 stays WATCH-only
against the reeval'd 2b/2a heads; `milestone_reference` names the reeval **root** and resolves the stamp with the same
helper `load_row` uses (`resolve_eval_dir`, D10). `_reference_watch`, `_close_final_s2` and `_stage_gate_watch`
read `suite`; the formation-era gate verdicts stay WATCH context on the suite's legacy per-hand keys.

**D8. `eval_sha` (driver).** `CODE_PATHSPECS` loses `scenarios_omni.yaml`, `run_phases.py` and `gait_eval`;
`EVAL_PATHSPECS = (gait_eval/, eval_crab_hex_gait.py, scenarios_omni.yaml, run_phases.py, $OMNI/eval_categories.py)`;
`eval_sha()` uses the same `CODE_DROP_GLOBS`; the dirty check covers both lists. `run_phases.py`'s training seam (`train`,
`train_retry`, `_subenv`, `with_plant`, the command line and env composition) is pinned by
`test_run_phases_training_seam_is_unchanged` (sha256 == a recorded constant, over the source of `train`, `train_retry`,
`_subenv`, `with_plant`, `latest_run_ckpt`, `_ckpt_iter`, `set_seed`, the training-log parsers the abort reads and the
`Campaign` class, PLUS a sorted repr of the module-level non-callable globals they read — `NUM_ENVS`, `PY`, `TRAIN`,
`SMOKE_ITERS`, `STALL_S`, `ABORT_*`, `TRAIN_ENV_RESERVED`, `LOG_ROOT`, collected via `inspect.getclosurevars`, runtime-state
names excluded; the command line is built from those constants): a training-relevant
edit there must bump the pin and be declared with `--accept-code-change` — a recorded caveat, the price of not splitting the
module. Launch records `lad.eval_sha`; every header and ledger entry carries `eval_sha`; `ok_screen` additionally requires
`h.eval_sha == lad.eval_sha`, so a screen measured on the old eval is never a control or an anchor until rescreened.
**Stamping:** each stage rec stores the `eval_sha` current at its eval, and `header.eval_sha` is (re)written together with
`header.screen` — it names the eval that produced the screen, not the header's birth — so a pre-suite header that resumes or
is re-opened after the relaunch (X05, X03, X04) qualifies once its suite screen is written and is never rescreened; a
rescore writes `header.eval_sha` with its `rescores[]` record; every eval-row floor verdict stamps `header.floor_eval_sha =
lad.eval_sha` in the same write as `floor`/`floor_stage` (absent on pre-suite headers such as X03/X04), so D9 item 5 can
tell a floor measured under an older eval from one measured under the current eval.
A changed `eval_sha` is refused without `--accept-eval-change WHY`. **A campaign with a recorded `lad.code_sha` and no
`lad.eval_sha` (every pre-suite campaign, this one included) counts as changed:** `main()` runs the `--accept-eval-change`
path and its alias proof BEFORE the existing `code_sha` comparison; a proven alias bypasses the `code_sha` refusal and its
"B<k> rerun is queued" RELAUNCH text, a launch without the flag is refused with a message naming `--accept-eval-change` (not
`--accept-code-change`), and only a FAILED proof falls through to the `code_sha` refusal. Because the pathspec change also moves the VALUE of
`code_sha`, the acceptance **proves the training tree did not move**: `code_sha_at(commit, pathspecs)` over
`git ls-tree -r <commit>` with the same drop globs; find the newest commit C in `git log -n 300` with
`code_sha_at(C, CODE_PATHSPECS_V1) == lad.code_sha`; require `code_sha_at(C, CODE_PATHSPECS) == code_sha()`, else refuse
("the training tree also moved: use `--accept-code-change`"); on success append `{sha, proved_at, why, at}` to
`lad.code_sha_aliases`. `ok_runs` reads that list itself and accepts `h.code_sha ∈ {current} ∪ aliases`; `baseline_label`
normalises through the same set (so no spurious `B1`). Records are never rewritten: a rescreen keeps the old screen under
`header.screens_superseded: [{eval_sha, screen}]`.

**D9. Migration of the paused campaign (driver, zero decisions by hand).** `loop()` checks a new `migration_item(ctx)`
**before `find_inflight` and before `_next_item_raw`** (whose baseline-rerun rule would otherwise book an 8000-iteration B0
rerun the moment `ok_screen` goes stale), and `find_inflight` returns None while `migration_item` is pending.
`migration_item` returns None under `--smoke-stage` (matching `main()`, which already skips `ensure_resolution` there), and
both startup branches, the smoke branch included, call `load_resolution(ctx)`. The migration's eval handlers reuse
`run_experiment`'s eval-row infra path (the OOM / DerivedDataCache signature read, the INFRA line with quarantine advice,
one ticked retry); a migration item that still cannot complete halts the CAMPAIGN (`phase = halt:migration <item> <why>`,
re-derived on relaunch); it never calls `_halt` on a recorded header or rewrites that header's status. Every halt the
migration and resolution can raise is campaign-level (`halt:migration …`, `halt:resolution D00s|D00bs …`,
`halt:resolution_missing …`) and never touches a header. Each item is derived from the records, in this order:

1. **rescreen X01 then X02** — every non-squashed `status: ok` header of a `SCREEN_KINDS` kind whose `code_sha` is current or
   aliased and whose `eval_sha` is stale (at the relaunch exactly X01 = B0 s3 and X02 = B0 s2; the rule is re-derived on
   every loop, and a header stamped per D8 never matches it again). Each handler begins with
   `rp.set_seed(ctx.c, header.seed)` (X02 is a seed-2 record; filing it under seed 3 would leave it screenless and trigger a
   3.9 h rerun) and builds the item through `item_from_header`, so the tag, out root, overlay and `_carry_mismatch` are the
   recorded ones. Writes `rec["evals"]["suite"]` beside the kept `slow`, then the new `header.screen` = `_screen_of` over
   that extended rec (the suite summary with `cat` and `trained_categories` — `_screen_of` never reads `slow` when a suite
   row exists; old screen under `screens_superseded`), `header.eval_sha`, `header.rescreens[]`. No ledger entry.
2. **reeval** the three D00 inputs (`1c-004`, `2b`, `2a-reeval` — the `role: pair` entries) on the suite —
   `resolution_inputs` entries carry `checkpoint + preset + eval_dir` and no `run_dir` (a validator branch accepts `run_dir`
   XOR `checkpoint+eval_dir+preset` and refuses neither/both); out root `EVAL_ROOT/<campaign>/reeval_<head>`. Pending per
   `role: pair` input while `lad.reevals` has no record for that head at the current `lad.eval_sha`; on completion
   `{head, eval_sha, stamp}` is appended (older records kept, never rewritten), and D00s reads exactly the recorded stamps
   (D10). The two `role: seed_pair` entries share the checkpoint shape (their stamped run dirs do not exist when
   candidates.yaml is written) but are filled by item 1's rescreens and are never reevaluated.
3. **`D00s`** (kind resolution, `supersedes: D00`, carries `eval_sha`) from the cross-head suite pairs (`1c-004` vs `2b`,
   vs `2a-reeval`; env-clustered paired bootstrap (D10), B = 2000, `paired_on: env_index`, one-sided 99 % guards with the
   binomial floor; `light`/`ring` `by_row` tables copied from D00, D7). **The queue item calls `resolve()` in-process**
   (`res.main` reads `resolution_inputs` only from the yaml on disk and has no channel for any of what follows): it builds
   the entries — each `role: pair` entry gets `stamp` from its `lad.reevals` record at `lad.eval_sha` (a missing record
   raises `ResolutionError` naming the head) — then `payload = res.resolve(dict(cands, resolution_inputs=entries),
   eval_sha=lad.eval_sha, decisive_at=sorted(decisive(B0)) (used for both decisive and present — B0 trains all seven), d00=<parsed $LAD/resolution.json: D00's by_row light/ring
   tables and decision_rule>, supersedes="D00", candidates_sha256=<sha256 of the unmodified live yaml>)`,
   `res.write_resolution(payload, ctx.dir / f'resolution.{eval_sha[:12]}.json')` (absolute: the driver's cwd is the repo
   root) and `load_resolution(ctx)` so the next item sees it in the same process; `eval_sha` is computed in run_candidates
   and passed in (resolution.py never imports it); a raised `ResolutionError` (a `_reducer_check` refusal included) halts
   `halt:resolution D00s refused: <text> — fix and relaunch` (no entry written); the resolution file is written before the
   D00s entry and item 3 is pending while either is missing at `lad.eval_sha`; `res.main` keeps a CLI mirror (`--eval-sha`, `--reeval-stamps JSON`, `--supersedes`,
   `--d00 PATH`) for manual runs. Item 3 is pending while no D00s entry carries `lad.eval_sha`. Startup never computes it:
   `load_resolution(ctx)` ingests `resolution.<eval_sha[:12]>.json` if present and otherwise leaves `ctx.resolution = None`
   (never the old slow-row record); `_dispatch_candidate`, `_dispatch_pooled`, `_row_worse` and `_write_d00bs` halt
   `halt:resolution_missing` while it is None, and the first three also while D00bs is missing (`_write_d00bs` is what
   writes it).
4. **`D00bs`** — the seed terms per block and guard, from the two rescreened B0 screens **against `D00s`** (the seed terms
   standardise through `D00s`'s per-metric `R_eval`, so this cannot precede step 3), written by `_write_d00bs`, a new
   function generalising today's `_write_d00b` (id `D00bs`, against D00s, iterating the expanded rule's blocks). Its
   RESOLUTION line carries the seed-dominance verdict per decisive block — `R_b^seed` vs `R_b^eval` for F, F_mt and T (or
   the `_abs` twins) with a seed-dominated yes/no per block (yes iff `R_b^seed > R_b^eval`; then a single-seed screen is judged at `R_b^seed`) — the per-member MDE, and — the informative statement — B0 s2 vs s3
   classed at `D00s`'s **eval-only** R per block. (The all-TIE check under the full R is a construction check: the pair's own
   delta is inside its seed term. A non-TIE there is a bug.) Acceptance also records the null-candidate all-TIE inside
   `[nominal − 0.03, 0.98]` (nominal = 0.95^blocks × 0.99^guards = 0.95³ × 0.99⁹ ≈ **0.78** at B0's decisive set in the
   ratio system: three blocks, seven `survival` guards, the `move_turn.ratio` guard and `cat_tripod_min` — the null
   construction uses the env-clustered draw and the floored guard R `max(R_eval, R_binom)` with no seed term, matching the
   eval-noise nominal) and whether `R(D00bs) ≥ R(D00s)` per block (a recorded note, not a gate); the construction check
   runs in both unit systems. **Which null gates:** the calibrated env-clustered null that `resolve()` computes (R
   re-derived on the same resample distribution, survival guards floored at `R_binom`) is the one held to the band; the
   null under D00s' cross-head R is recorded with its per-guard TIE rates. Each kind has its own acceptance, **computed
   BEFORE any write** — **D00s**: the null band AND completeness at `decisive_at` in both unit systems: a finite, positive
   `R_eval` for every non-empty block (F, F_mt, T and the `_abs` twins), every member of those blocks, and every
   non-survival guard (`cat.move_turn.ratio`, `cat.move_turn.achieved_speed`, `cat_tripod_min`) — `survival` guards are
   exempt (a None or 0 `R_eval` passes on the binomial floor); without this an unresolved block would be a permanent TIE
   (null_construction skips it and the band still passes); **D00bs**: the construction check AND empty `block_missing` /
   `guards.missing` on the B0 s2-vs-s3 comparison in both unit systems (a value missing on a rescreened B0 screen). If it passes, the entry (and for D00s the
   resolution file) is written as above with `evidence.acceptance = {ok: true}`. **If it fails and there is no
   `lad.resolution_acceptances` record `{kind, eval_sha: lad.eval_sha}`, NOTHING is written** except the diagnostic
   `$LAD/resolution.<eval_sha[:12]>.<kind>.rejected.json` (overwritten on each recomputation, never cited), and the campaign
   halts `halt:resolution <kind> acceptance: <failures> — fix the tools and relaunch, or relaunch with --accept-resolution
   <kind> WHY` (notify, stop-and-wait). The item stays pending, so every relaunch recomputes (deterministic: fixed bootstrap
   seed, recorded stamps, CPU only) — a tools fix in resolution.py or run_candidates.py (outside both hashes) re-runs it — a fix in eval_categories.py (the
   shared `expand_rule`/`composite_guards`) moves `eval_sha` and re-runs the migration evals —
   without another eval. `--accept-resolution <kind> WHY` appends `{kind, eval_sha, why, at}`; the next computation for
   that kind at the current `eval_sha` then writes the entry with `acceptance.ok = false` and cites the acceptance on its
   RESOLUTION line. A raised `ResolutionError` (item 3) likewise writes nothing and halts `halt:resolution D00s refused:
   <text> — fix and relaunch`. Item 4 is pending while no D00bs entry carries `lad.eval_sha`.
5. **rescore** the floor-rejected chains: X03 (C1) at its c4 checkpoint and X04 (C1b) at its c2 checkpoint, each with that
   stage's recorded overlay and eval_carry, filed into the existing rec (`ctx.c.recs["X03:c4"]["evals"]["suite"]`) after
   `rp.set_seed`. Rescore items are derived from HEADERS in `lad.experiments` (never `lad.squashed`), not from ledger
   entries, and only for a floor **measured under an older eval**: `status: rejected_by_floor`, a recorded `floor_stage`, an
   eval-row floor (the `floor.kind` of the header's RECORDED `stage_specs` entry whose `stage == floor_stage` — a list, not
   a mapping — ∈ {survival, category_survival} —
   `evidence.floor` is a free-text string; a `dead_walker` REJECT, a training-log verdict, is never rescored),
   `header.floor_eval_sha != lad.eval_sha` (absent = pre-suite, as on X03/X04; every floor verdict stamps it, the
   dead_walker abort at :4635 included, and a floor text starting `dead_walker at` is never rescored), a `code_sha` that is
   current or aliased (as item 1), and no `rescores[]` record at `lad.eval_sha` — so a floor REJECT written under the current eval (a re-screen, a later chain) is never a migration item
   and the floor is never a second-chance floor. The floor is re-decided under D6 on that stage's trained categories. Every
   `rescores[]` record names the entry it re-decides — `supersedes: <entry id>`, computed BEFORE any write as the newest
   `kind == candidate` entry with `experiment == xid`, `floor_stage == header.floor_stage` and `eval_sha != lad.eval_sha`
   (absent counts as stale: D05 for X03, D06 for X04), so it can never name the replacement. Derived-view cause **(f)**
   squashes exactly the entries named by some `rescores[].supersedes` — so the launch reconcile squashes nothing (no record
   yet), and the FAIL branch's new REJECT and any later floor REJECT of a re-opened header (X04 failing c4) are never named
   and stay effective. **PASS** → in one write: `rescores[]` gains `{eval_sha, floor, floor_stage, verdict, supersedes}`
   (the header's `floor`/`floor_stage`/`floor_eval_sha` move into that record),
   `header.recs.append("<xid>:<floor_stage>")` (the rec exists with its ckpt; without this the resumed chain would re-train
   the rescored stage), `header.eval_sha`, and the header re-opens for its NEXT stage (C1: c5, 3500 iterations left in the
   chain; C1b: c3, 6000 left) with `inflight = None` (the next stage is fresh and takes `prev_ckpt` from the appended rec)
   and `status: running` for a dispatchable header (X03) or `rescored_awaiting_predicate` for one whose CANDIDATE declares
   `depends_on`/`conditional` in candidates.yaml (X04 — the header's own fields are None); (f) then squashes D05 / D06 and D07 (C1c SKIP) follows by cause (c). **FAIL**, in this
   order → (1) the new floor fields (incl. `floor_eval_sha`) are set on the header (`_dispatch_floor_reject` reads
   `header.floor` / `floor_stage` and copies them into the entry; the save inside (2) persists them, so the old values
   for the `rescores[]` record are taken from the superseded entry's evidence); (2) the ledger REJECT is written, carrying `eval_sha = lad.eval_sha` — its
   reconcile runs while the old entry is still effective, so no dependant (D07) is archived; (3) one state write saves the
   `rescores[]` record `{eval_sha, verdict: FAIL, supersedes, old floor, old floor_stage, old floor_eval_sha}` (the old
   values kept as history, as on PASS), the new header `floor`/`floor_stage`/`floor_eval_sha` and `header.eval_sha`; (f)
   then squashes the old entry and C1 stays decided (D07 stands). A relaunch that finds a current-eval floor entry for
   `(xid, floor_stage)` AND a stale-eval floor entry for the same `(xid, floor_stage)` that no `rescores[].supersedes` names
   completes step (3) from them instead of re-running the eval (a completion item derived independently of
   `floor_eval_sha`, which the save inside (2) already moved; a re-opened header's later floor REJECT has no such stale twin).
   **Each rescore handler ends, after its state write (the PASS write, FAIL step (3), and the relaunch completion of step
   (3)), with `squash_and_report(ctx, f"rescore {xid}:{floor_stage} {verdict}")`** — cause (f) is the first squash cause
   keyed on STATE, and the driver recomputes `ctx.view` only after ledger writes, so without this the same process would
   still see D05/D06 effective and the rank gate would resume X05 (≈ 3.3 GPU-h) ahead of the re-opened chains.
   The PASS notifies under `MIGRATION` (B5), the FAIL under `DECISION`. A migration eval that returns `interrupted` writes
   nothing (never `_mark_interrupt`, which would rewrite a recorded header's status — X01 would turn `running`) and returns
   to the loop top, where the stop check parks; the next launch re-derives the same item.
6. **the ordinary queue, in rank order.** `find_inflight` resumes a `kind == candidate` header **only when
   `_next_item_raw` would dispatch that same candidate now** (steps 1–7 return nothing and step 8's first non-terminal
   ranked candidate is the header's cid; a speculative entry never reaches step 8 — step 3 returns its pair first — so the
   gate parks lower-ranked headers until the pair resolves); otherwise the header stays parked (logged `not resumed
   (waiting on <Cid> at rank k)`) and the loop falls through to `next_item`. Twin, baseline, milestone and pair headers
   resume unconditionally, as today. C1b (X04) carries `depends_on: [C1]` and a predicate on C1's decision, so its re-opened
   header is `rescored_awaiting_predicate`: step 8, reaching C1b after its dependency, predicate and user-choice checks
   pass, returns `item_from_header_raw(cands, <that header>)` — a ctx-free variant, since `_next_item_raw` is a function of
   `(cands, ledger, state)` also used by `simulate_queue` — **with its xid** and `needs_s3` = the header's base recipe
   (without the xid it would book a new C1b experiment from c1 and orphan X04); before training, the dispatch flips the
   header to `status: running` with a history row `{at, from: rescored_awaiting_predicate, why: step-8 dispatch}`
   (`run_experiment`'s RESUMABLE_HALTS flip at :4492 is extended to this status), so every later resume goes through
   `find_inflight` and `prepare_resume` and `_live_window` covers the window; cause (a) squashes it if C1 is adopted. X05
   (C2, `model_400` of 8000) therefore resumes only after C1 (and its pair, if speculative), C1b and C1c are terminal.
   `item_from_header` keeps the frozen overlay and stage list but overrides each frozen `stage_specs` entry's
   `eval`/`floor`/`categories`/`previous_categories` from the CURRENT chain definition when that chain exists in
   candidates.yaml (smoke headers keep their frozen specs; every recorded header carries `stage_specs`: X01/X02/X05
   `eval ['slow']` + dead_walker, X03/X04 `['curriculum']` + hold-keyed floors; the eval side is not a property of the
   window).

Every step's writes are those listed in its item (item 5's FAIL: the three ordered writes; items 3–4: the resolution file,
then the entry), each followed by the SQUASH reconcile when it can move the derived view (item 5), plus one notify line
(`MIGRATION` for eval items, lines marked "eval-only"; RESOLUTION / DECISION where stated); the SQUASH / RESOLUTION /
DECISION contract is unchanged.

**D10. Resolution on the suite (`resolution.py`).** `SCREEN_ROW = "suite__omni"` (`LEGACY_SCREEN_ROW` kept for the recorded
rows' `by_row`). `load_row` accepts `checkpoint + eval_dir` entries: a `role: pair` input reads the stamp recorded in
`lad.reevals` at the D00s `eval_sha` (the driver passes it in the input entry; `ResolutionError` naming the reeval when no
such record exists); a `seed_pair` input and `milestone_reference` use `resolve_eval_dir(REPO/eval_dir, sid)` = the newest
stamp under `<root>/<sid>/seed001` that passes the **freshness rule**, which every SUITE-row stamp read must pass (legacy
light/ring rows carry no marker and are read from their recorded run_dirs; a suite `milestone_reference` entry is
`{eval_dir, checkpoint}` so the checkpoint test has something to compare): its
`complete.json` exists (on suite rows `write_run` writes this marker LAST — after the ~1000 raw npz, which it writes after
the aggregate — holding `episodes_total` and the checkpoint sha256; body rows are unchanged), and the checkpoint matches the
entry (compared by sha256, or by `Path(REPO/entry.checkpoint).resolve()` vs `Path(run_meta.checkpoint).resolve()` — the
harness records absolute paths, candidates.yaml repo-relative ones); a missing or partial raw set raises `ResolutionError`
naming the stamp, never `FileNotFoundError`; `_run_eval` applies the same marker test before calling a suite row `ok`.
`load_row` reads each episode's category/hand/hold from `aggregate.suite` (never from the episode's hold dict) into
`category_index`/`hand_index`, and refuses a row in which any non-settle label other than the episode's own hold has
`n_steps > 0` (every union label carries an entry; the unvisited ones have `n_steps: 0`). `_columns` adds `cat:<c>:<key>` and
`cat:<c>:hand:<h>:<key>` columns (NaN outside the category), with `surv` from the episode's own hold end (the D3
convention) and, on a suite row, each `surv:<h>` column NaN outside hold h's own episodes, so `survival_by_hold.<h>`
reproduces `aggregate.suite.survival_by_hold`. `spec_for` learns `cat.<c>.<key>`, `cat.<c>.by_hand.<h>.<key>` and
`cat_tripod_min(decisive)`; every category id — `cat.*` and `cat_tripod_min` — on a row that is not a `category_suite`
row (the informational legacy slow/light/ring inputs included) goes to `specs_from_rule`'s unresolved list and never reaches
`_reducer_check`. **On a `category_suite` row** (the kind read from `run_meta` / `aggregate.suite`, never inferred from
labels; `load_row` stores it on the Row and passes `suite=True` into `spec_for`, `specs_from_rule`, `metric_specs`,
`watch_specs` and `why_unresolved`), `spec_for` returns None for `turn_survival_l90/r90` and any other conditional-survival
id (why: "undefined on a suite row: holds are not nested"), and `metric_specs` — the rule-less fallback that
`facing_closed_by_hold_from_raws` and `summarize_run_dir` use — omits them, so they land in the unresolved list and never
reach `_reducer_check` (summarize() returns None for them by design, D5; without this every suite row would raise). `resolve()` expands the v2
rule with the shared `expand_rule` at `decisive(B0)` — the union of both unit systems, D5 — before `specs_from_rule` (the raw v2 rule has no `blocks`/`guards` key,
and `specs_from_rule` would otherwise fall back silently to the legacy hold-keyed `metric_specs`). `bootstrap_pair` draws
**clustered on the within-wave env index**: each replicate resamples env ids 0..E−1 with replacement and takes every wave's
episode of each drawn env (both heads share the draw), so every hold keeps exactly its n, every paired category its
100/100 hand split, and the within-env correlation (the same startup-randomised robot in every wave, D1) is preserved —
across categories too, for `cat_tripod_min`; a legacy row (one wave) reduces to today's episode draw, bit-identical.
`null_construction` uses the same clustered draw and the floored guard R, and scores the ratio system's three blocks and
nine guards (D5).
`binomial_guard_floor(p̄, n)` with the clipped `p̄`. `role: seed_pair` inputs (the two rescreens, recorded so the record
documents D00bs' inputs) are excluded from cross-head pairing. `resolve` writes `screen_row`, `eval_sha`, `supersedes`,
`decisive_at`, `by_category_n`, `null_nominal`, per-survival-guard null TIE rates, and D00's `light`/`ring` `by_row`
tables with D00's frozen `decision_rule` as `by_row_rule` and `by_row_source: D00` (D7). `BORROW` /
`--curriculum-runs` retire behind `rule_version == 2`. `_reducer_check` reproduces every `cat.<c>.<key>` from the
per-episode columns to 1e-9, and every composite (`cat_tripod_min`) against the shared `composite_guards` applied to a copy
of the summary (the per-episode reducer is the min over the decisive translation categories' `cat:<c>:tripod` medians; a
composite id with no summary value raises a named error, never a NaN) — the one test that pins harness and tools to a
single definition of each number.

**Suite-aware manifest readers (both tools).** `rp._manifest_hold_s` and `resolution._manifest_scenario` currently do
`sc["schedule"]` on the raw YAML, which a suite row does not have; both instead load the manifest through
`gait_eval.schedule.load_manifest` (by path, as `resolution._facing_metrics()` already loads `gait_eval.metrics`) and expose
`kind`, the derived `hold_s` map and `settle_hold_s`. `_survival_end_steps` on a suite row returns the **own-hold** end
`round((prologue_s + settle_hold_s + hold_s[label]) / dt)` for every label — no cumulative walk. All **three** cumulative
injection sites (`rp._run_eval`, `resolution._summary_with_survival`, `run_candidates.summarize_run_dir`) skip when the
aggregate carries `suite`, taking `survival_by_hold` from `aggregate.suite.survival_by_hold` instead.

## Part A — the harness (eval-side; forward/sideways scoring byte-identical)

**A1. `$GE/schedule.py`.** `Scenario` gains `kind: str = "schedule"`, `suite: CategorySuite | None`, `waves: int = 1`;
dataclasses `CategoryHold(label, speed, dir, face, hand)`, `Category(name, kind: stand|translate|turn|move_turn, holds)`
(no `watch_only`: watch status comes only from what a recipe trains, D4), `CategorySuite(settle_hold_s, settle_label, hold_s,
categories)`. `load_manifest` parses `kind:
category_suite` rows (`waves`, `settle`, `hold_s`, `categories`; an explicit `schedule:` is a `ManifestError`) and DERIVES
`schedule = [settle] + [one Hold per category hold in manifest order]` (11 holds) as a **label union, not a timeline**:
on a suite row `schedule_duration_s = settle_hold_s + hold_s` and `total_duration_s = prologue_s + that` (= 15.5 s), so the
generic `episode_length_s ≥ total_duration_s` check and `$EVAL`'s `total_duration_s + 2.0` fallback both read one wave.
`_validate_suite`: kinds; a category mapping carries only `name`, `kind`, `holds` (any other key is a `ManifestError`);
unique names and labels (settle included); `hand ∈ {l, r, None}` with equal counts per category;
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

**A2. `$EVAL`.** `--waves INT` (consistency assertion), `--trained-categories CSV` (the driver passes only
`--trained-categories`, on suite rows only — `--waves` is for manual runs and V3's `extra_args` — and the harness refuses
either flag on any other row, so light/ring `run_meta` keeps its key set; recorded; names checked against the
suite). On a `category_suite` row only (every other row runs today's loop verbatim, D2), the main loop wraps the existing
body in `for wave in range(W)`: `env.seed(env_seed*1_000_003 + wave)` →
`env.reset()` (wave > 0) → re-install the override with `compile_wave(scenario, wave, …)` → `apply(0)` →
`get_observations()` → zero buffers / `alive` / `n_steps_env` / `term_reason` → the body verbatim → per-wave
`schedule_complete` assignment → append the host arrays. `env.close()` after the loop; `max_cmd_dev` and
`max_sched_state_dev` accumulate across waves; `host = concatenate(parts, axis=1)`; `compiled = stack_waves(parts)` is what
scoring and `write_run` receive. `run_meta` (suite rows only — a legacy invocation's key set is unchanged, and either flag
on a legacy row is an error): `waves, wave_seeds, episodes_total, env_index_map, suite: {dt, prologue_s, settle_label,
settle_hold_s, hold_s, categories}, trained_categories, hold_labels_kind: "union"`; `termination_reason` and
`n_steps_per_env` have length W·E.

**A3. `$GE/metrics.py`.** `travel_direction_metrics(vel_travel_xy, cmd_speed, *, min_speed=0.02)` and
`hold_motion_metrics(root_pos_w, root_quat_w, root_lin_vel_b, contact, dt)` (D3), pure numpy, `None` on empty windows.

**A4. `$GE/report.py`.** `score_episode`: `entry["direction"]` (commanded speed > 0 only) and `entry["motion"]` per world
hold; `out["wave"] = compiled.wave_of_env[env_idx]`, `out["hold"] = hold_labels[compiled.hold_of_env[env_idx]]`,
`out["category"]`, `out["hand"]` when the compiled schedule carries `wave_of_env`. `aggregate(episodes, *, suite=None)` (one signature, as D3:
`suite` = `run_meta.suite` ∪ `{waves, wave_seeds, episodes_total, trained_categories}`): `direction_by_hold`, `motion_by_hold` (both ABSENT — not empty — when no episode carries the
entry, as `facing_by_hold` is on body rows), and the `suite` block incl. `by_category` and the own-hold survival (D3) when
episodes carry a wave; `summary_text` gains a "by category" block; `write_run` passes `run_meta["suite"]` with
`run_meta`'s `waves`, `wave_seeds`, `episodes_total` and `trained_categories` (D3) and
`n_raw = len(run_meta["n_steps_per_env"])` (= W·E, the raw-writing loop bound — never called `n_env`, which is the
distinct-env count the floor band reads), and on a suite row writes `complete.json` (`episodes_total`, checkpoint
sha256) as its very LAST write, after the raws (D10's freshness marker; body rows byte-identical); episode files are
`episode_{env_index:02d}` (three digits from 100 — `resolution.rescore_facing` builds the same name).

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
`rotate_holds_by_env`, a category carrying `watch_only` or any other extra key),
`test_write_run_writes_the_completion_marker_last_on_suite_rows_only`, `test_a_legacy_row_refuses_waves_and_trained_categories`,
`test_only_a_suite_row_takes_the_wave_path` (the pure `uses_waves(scenario)` predicate the `$EVAL` loop branches on is False
for every committed legacy row of every manifest, so no legacy eval gains an `env.seed` call), `test_legacy_rows_compile_byte_identically` (against the step-0 fixture
`tests/unit/fixtures/gait_eval_compile_snapshot.json`), `test_committed_omni_manifest_parses` (id set becomes exactly
`{slow, light, ring, curriculum, suite}__omni`; every existing assertion kept; the suite's ten labels, hands, speeds, dirs,
faces, `settle 2.0`, `hold_s 12.0`, `waves 10`, `episode_length_s 18.0`, `env_vars {}`, `total_duration_s 15.5`, `task`
`Isaac-Crab-Hex-Omni-Walk-Play-v0`, `terrain_label` flat).
`tests/unit/test_crab_hex_gait_metrics.py`: the two metric functions; `test_world_row_holds_carry_direction_and_motion`
(and no `direction` on `settle`/`stand`); `test_suite_aggregate_pools_both_hands_and_scores_own_hold_survival` (synthetic
suite E = 4, W = 4 — fwd, lat_l, lat_r, turn_l90: a fall in a turn episode leaves `fwd` survival 1.0; a fall on the wave's last step, `n_steps` equal to
the end step, counts as survived; `by_category.side` is the pooled median of 2×n episodes, not the mean of hand medians);
`test_aggregate_has_no_suite_block_on_body_rows` (also asserts `direction_by_hold` and `motion_by_hold` are absent, not
empty);
`test_body_rows_carry_no_omni_fields` extended with `direction`, `motion`, `wave`, `hold`, `category`, `hand`.
`tests/unit/test_crab_hex_gait_eval_golden.py` unchanged and must pass byte-identically.
`tests/unit/test_crab_hex_omni_command.py`: `test_schedule_hook_is_untouched_by_the_suite` (source hash == recorded constant).

## Part B — the omni package and the ladder tools

**B1. `$OMNI/eval_categories.py`** (D4) + `tests/unit/test_crab_hex_omni_eval_categories.py`: decisive/present/watch at the
floor share, on normalised shares (an un-renormalised `set_share` edit gives the same sets); absent env keys take the
class defaults, pinned against the command cfg's class body; `expand_rule` keeps the survival guard of a watch category and
moves its other members to WATCH, and swaps the move_turn guard under `absolute_units`; `composite_guards` computes
`cat_tripod_min` over the decisive translation categories only; `omni` makes every bearing decisive with the note; one-handed `side` is watch; `test_c1b_and_c7_pool_edits_keep_turn_decisive`
(`{"*_turn": 0.3}` touches `move_turn` only); `test_an_ang_scale_on_the_turn_type_demotes_turn`;
`test_sincos_keeps_turn_decisive_and_records_the_encoding_note`;
`test_move_turn_is_decisive_at_full_demand_and_watch_under_a_reduced_one` (B0 and c4 decisive; C1b and C7 watch);
slots mode reproduces the legacy presets; the curriculum stages c1..c6 read from `$LAD/candidates.yaml`; `CATEGORY_HOLDS` covers the suite row's labels
exactly (reads `$MANIFEST`; it needs A5's row, so step 2 stubs it and it runs at the (1 ∥ 2) join);
`test_eval_categories_imports_without_isaac`; the file is absent from `code_rows(CODE_PATHSPECS)`.

**B2. `$TOOLS/run_phases.py`** (eval side only). `eval_morph(c, spec, ckpt, scenario, tag, *, ..., trained_categories=None,
extra_args=())` appends `--trained-categories <csv>` when given, then `*extra_args`, after the fixed harness tail (the
default call's command stays byte-identical); the suite-aware `_manifest_hold_s`; the three-site survival rule and `summarize()`'s `cat` (D5, D10);
every existing key unchanged; `phase_lines`/`run_phase` untouched. Tests in `tests/unit/test_crab_hex_omni_gate.py`:
`test_summarize_hoists_cat_from_the_suite_block_and_is_empty_without_it`,
`test_summarize_legacy_keys_unchanged_on_a_suite_aggregate`, `test_turn_survivals_are_none_on_a_suite_row`,
`test_no_reader_injects_cumulative_survival_on_a_suite_row` (parametrised over the three sites),
`test_manifest_readers_derive_a_suite_rows_holds`, `test_eval_morph_forwards_trained_categories_and_extra_args` (and the
default tail is unchanged),
`test_run_phases_training_seam_is_unchanged`; the existing pins unchanged.

**B3. `$TOOLS/run_candidates.py`.** Hashes and acceptance (D8): `CODE_PATHSPECS_V1`, the narrowed `CODE_PATHSPECS`,
`EVAL_PATHSPECS`, `eval_sha()`, `code_sha_at()`, `--accept-eval-change`, `lad.code_sha_aliases`, alias-aware `ok_runs` and
`baseline_label`, `eval_sha` on headers/entries and in `ok_screen`, the header/rec `eval_sha` stamping, and `main()`'s
pre-suite branch (no `lad.eval_sha` → the alias path runs before the `code_sha` comparison). Rule v2 (D5): `expand_rule`
and `composite_guards` imported from `eval_categories.py` (never re-implemented), the `decisive`/`present` parameters through
`_screen_compare`/`decide`/`decide_pooled`/`anchored_drift`/`_row_worse`/`_write_d00bs`, N/A blocks, the binomial floor
(survival guards only) with `n_c` and its pooled form, the all-N/A SKIP branch, `guards_abs`, `evidence.decisive/untrained`;
the block set from the expanded rule in every enumeration — the validator (:1497), `block_names` (:2419; the blocks of the rule passed — 2 for v1, 3 for v2),
`_screen_compare`'s block loop (:2426), the RESOLUTION fields (:5151), the DECISION line (:5320), `_write_d00b`'s control
check and R-floor loop (:5569–5584); `_speculative_pick` and `decide_pooled`'s per-seed/fallback signs on
`sign(Σ z_b / R_b)`; `seed_terms_from_pair` with `composite_guards`. **`_screen_of`** (:4831) — which keeps copying
`track_ratio_tail`, `track_ratio_move_tail` and `stage` from the rec (C11's predicate reads `screen.track_ratio_move_tail`;
without it C11 would be SKIPped silently as unevaluable) — summarises the final stage's
`evals["suite"]` when present and falls back to `evals["slow"]` only on a legacy rec with no suite row; it copies the stage's
`trained_categories` onto the screen. Every `resolution_id` label (`_dispatch_candidate` :5637, `_dispatch_pooled` :5814,
`milestone_i` :5948, the dry run) cites the loaded resolution's id (D00s), and `milestone_i` joins D9 item 3's
`resolution_missing` halt list. Stages (D6):
`categories:` replaces `holds:` (validator: `== sorted(decisive ∪ watch)` of `trained_categories(stage env)`),
`stage_floor` kind `category_survival` with the band test, `stage_retention` per category with the per-metric sign and
`stage_trace_v2`, `previous_categories`. Migration (D9): `migration_item(ctx)` ahead of `find_inflight` and
`_next_item_raw` (None under `--smoke-stage`); `load_resolution` at startup (both branches) and a `resolution` queue item
(the in-process `res.resolve(...)` + `res.write_resolution(...)` call of D9 item 3, with `resolve()`'s four new keyword
arguments) with `resolution_missing` halts; the
rescreen/reeval/rescore handlers (with `rp.set_seed`, `item_from_header`, `_carry_mismatch`, `screens_superseded`,
`recs.append`, `rescores[]` with `supersedes`, the re-open's `status`/`inflight` writes, `rescored_awaiting_predicate`),
the reuse of `run_experiment`'s eval-row infra path, and the campaign-level halts (`halt:migration`, `halt:resolution
D00s|D00bs`, `halt:resolution_missing`); `header.floor_eval_sha` stamped by `run_experiment`'s eval-floor path (:4730) and
the rescore derivation that requires it to differ from `lad.eval_sha`; cause (f) = exactly the entries named by some
`rescores[].supersedes`; `item_from_header` overriding each frozen `stage_specs` entry's
`eval`/`floor`/`categories`/`previous_categories` from the current chain (when the chain exists); `find_inflight`'s rank
gate (a `candidate` header resumes only when `_next_item_raw` would dispatch its cid now); `_next_item_raw` step 8
returning `item_from_header_raw(cands, <rescored_awaiting_predicate header>)` with its xid and `needs_s3`, and the flip of
that header to `running` (`run_experiment`'s flip at :4492 extended to the status) with a history row; `D00bs`/`D00s`
writes (`_write_d00bs` generalising `_write_d00b`; the D00s item calls `load_resolution` after writing) with
`seed_terms_for` preferring `D00c-<chain>` (only when it carries the current `eval_sha` and non-empty block terms) →
`D00bs`; **`_d00c_lookup` (:5380), which writes D00c, is a v2 seam** — today it passes the raw rule, which under v2 has no
`blocks`/`guards.metrics` and would write EMPTY seed terms that `seed_terms_for` then prefers (F_mt judged at `R_eval`
alone, ≈ 3.7× too tight): it expands the rule exactly as `_write_d00bs` does (the union of both unit systems at
`decisive(B0)`), applies `composite_guards` to both screens, stamps `eval_sha` and `decisive_at`, runs its existence
check at the current `eval_sha` (its early return at :5393 fires only then) and supersedes a stale D00c with a new entry,
never a rewrite; never `D00b` for a v2-scored comparison (a missing D00bs halts
`halt:resolution_missing`; the light/ring rows' v1 scoring is the one sanctioned D00b citation, D7); the D00bs acceptance
computed BEFORE any write for D00s (null band) and D00bs (construction check) — a failure writes nothing but the
`.rejected.json` diagnostic and halts `halt:resolution <kind>`, the item stays pending, and `--accept-resolution <kind> WHY`
(new relaunch flag, recorded in `lad.resolution_acceptances` as `{kind, eval_sha, why, at}`) lets the next computation
write the entry anyway; `lad.reevals` records `{head, eval_sha, stamp}`; the
`resolution_inputs` validator branch; `dry_run` printing checkpoint-shape inputs and the `resolution.<eval_sha[:12]>.json`
path (today it reads `item["run_dir"]`, a KeyError on the new shape). Milestones (D7): `promotion.eval` from the yaml
(default `PROMOTION_EVAL`) consumed in `promotion_stage_list` (:600–611; `final_s2_stage_list` and `item_from_header`'s
milestone branch inherit it; `ph.EVAL_FULL` never edited), `_close_milestone_promo`'s row loop (:6215) and gate line
(:6165) over `promotion.eval`, the suite row WORSE on ≥ 2 units with its nominal rate and with `decisive`/`present` passed
explicitly from the promotion stage's record, `_row_worse`'s legacy-row branch (a `by_row_source: D00` table → D00s'
`by_row_rule`, D00b's seed terms, today's `_any_worse`), `_run_row` passing `--trained-categories` for suite rows only,
`resolve_eval_dir` for `milestone_reference`. Rendering: `_stage_report` one line per category
(`- suite fwd: n 100 surv 0.93 ratio 0.62 tripod 0.51 [decisive]` / `[WATCH]`, hands in parentheses); DECISION line fields
`trained:`, the per-category block rows, the untrained WATCH rows, and the turn-magnitude/encoding annotation;
`HOLD_SHORT`/`WATCH_LABELS` for `cat.` ids; RESOLUTION and MILESTONE lines per D9/D7. `_turn_hold_s` points at the suite's
`turn_l90`. A suite eval's heartbeat `current.iters` is 1000. `--smoke-stage 1a:20:suite` (third field = eval rows) for V6.

**B4. `$TOOLS/resolution.py`** (D10), including the block set from the expanded rule at the spec construction (:471),
`bootstrap_pair` (:1149), `null_construction` (:1206 and its `scored` filter at :1234), `_standardised_blocks` (:1281) and
`_resolution_line` (:1493); the `suite` flag through `spec_for`/`specs_from_rule`/`metric_specs`/`watch_specs`/
`why_unresolved` (D10); the completion-marker freshness rule and recorded-stamp inputs; `pooled_guard_R`'s optional
`r_binom` (D5). **B5. `$TOOLS/ledger_cli.py`**: `show` prints `eval_sha`
(`— (predates the suite)` for D05/D06) and `trained:`; `NOTIFY_TAGS` gains `MIGRATION`; no new verbs.

**B6. `$LAD/candidates.yaml`** (`--accept-candidates-change` at relaunch): `screen: {chain: scratch_1c, eval: [suite],
iters: 8000}`; `promotion: {eval: [suite, light, ring]}`; `decision_rule` v2 (D5; `bootstrap {B 2000, paired_on env_index, rng_seed}`, `control_head: 1c-004` and
`charter_floors` carried over verbatim — `resolve()` reads them from the rule, and without the control-head pin the
control would default to the first accepted input, now possibly a seed_pair row; `text` restated per category, frozen into
`D00s`); `resolution_inputs` = the three suite reeval entries (`role: pair`, `checkpoint + preset + eval_dir`) + the two
rescreens as `role: seed_pair` entries in the same shape (`checkpoint` = the header's ckpt, `preset` 1c, `eval_dir` = the
X01/X02 tag roots, resolved to the newest suite stamp) + the legacy rows (informational); the 2b process-floor row dropped (0.0 by determinism, amendment 19);
`milestone_reference` per stage: `suite: {eval_dir: <reeval root>, checkpoint: <head ckpt>}`, light/ring keep their recorded
run_dirs; chain stages: `eval: [suite]` where they had `[slow]`/`[curriculum]`, `categories:` per
stage, floors `category_survival` (c1, c2, c4) / `none` (c3, c5, c6; `scratch_*` keep `dead_walker`). The file lives under
`experiments/20*` (outside both hashes); it is committed with the tools in Part G commit 4.

**B7. Tests (tools).** **Fixture split first:** the candidates test module loads the LIVE `$LAD/candidates.yaml` as its
`cands` fixture and `rule = cands['decision_rule']` (42 tests take `rule`, 244 of 308 take `cands`), and
`test_crab_hex_omni_resolution.py` reads it too, so B6's rewrite would silently re-target them. Freeze the pre-change spec
as `tests/unit/fixtures/candidates_v1_2026-09-19.yaml` (byte-for-byte `git show 464be672:<$LAD>/candidates.yaml`,
committed, not generated at test time), add `cands_v1`/`rule_v1` fixtures loaded through `rc.load_candidates` (no
validation, so they never meet the v2-only validator), and repoint the v1-rule and legacy state-machine tests at them;
`cands`/`rule` stay on the live v2 yaml for the new tests and for the few existing tests that genuinely pin the live
ladder (listed below as "edited by name"). Then `tests/unit/test_crab_hex_omni_candidates.py`: fixtures `screen_cat()` /
`resol_cat()` (legacy fixtures kept for the v1-rule tests, which call `decide()` directly);
`test_expand_rule_restricts_blocks_and_guards_and_watches_the_untrained`, `test_a_block_with_no_decisive_member_is_na_not_tie`,
`test_all_blocks_na_is_a_terminal_skip_with_no_pair_and_never_rescreens` (a CONDITIONAL candidate whose predicate holds:
the SKIP survives a recompute and is not re-dispatched; cause (a) still squashes it when the base moves),
`test_T_only_recipe_adopts_on_T_better`,
`test_every_block_enumeration_carries_f_mt` (validator, `block_names`, `_screen_compare`, RESOLUTION and DECISION lines,
`_write_d00bs`), `test_a_watch_category_keeps_its_survival_guard_and_watches_the_rest` (C1b: `cat.move_turn.survival`
guarded, F_mt and the ratio guard WATCH), `test_speculative_pick_weights_each_block_by_its_R` (a large raw `z_F_mt` inside
its R does not outvote F and T), `test_absolute_units_move_turn_guard_has_guard_level_R_and_a_seed_term`,
`test_seed_terms_cover_composite_guards`, `test_a_saturated_survival_is_scored_not_missing_in_pooled_decisions_and_pooled_drift`
(and a non-survival guard with no R stays MISSING, never a silent TIE),
`test_screen_of_reads_the_suite_row` (suite-only rec → non-empty `cat` with `trained_categories`; a rec with both rows → the
suite summary; slow-only → the legacy screen),
`test_untrained_category_regression_is_watch_never_worse`,
`test_move_turn_facing_cannot_be_won_by_walking_less` (F_mt BETTER with a `move_turn.ratio` collapse → clear REJECT; the same
F_mt gain with a within-R ratio drop → ADOPT; `F_mt` and `F` carry their own R and neither dilutes the other),
`test_a_survival_saturated_on_both_heads_is_scored_not_missing` (1.00 vs 1.00, n = 100 → TIE at R ≥ 0.0717; n = 200 ≥ 0.0507),
`test_category_floor_is_a_band_test_and_both_c1_and_c1b_pass_at_back`, `test_category_floor_new_prev_and_missing_category_halts`,
`test_retention_keys_on_categories_with_per_metric_sign_and_reseeds_hold_keyed_traces` (facing_closed on an absolute drop,
including a trace value ≤ 0; drift_speed on the absolute-plus-relative rise), `test_screen_of_carries_trained_categories`,
`test_eval_sha_change_is_refused_without_acceptance`, `test_accept_eval_change_proves_the_training_tree_and_records_an_alias`
(temp git repo: an eval-file edit → alias; a training-file edit → refusal),
`test_a_pre_suite_state_without_eval_sha_takes_the_alias_path` (fixture: the real ladder key set; no RELAUNCH notify, no
B<k> item, and the refusal without the flag names `--accept-eval-change`),
`test_a_resumed_pre_suite_header_is_stamped_and_never_rescreened` (X05 and a re-opened X03 run to `status: ok` under the new
eval: `ok_screen` accepts them and `migration_item` returns None),
`test_ok_runs_and_baseline_label_treat_an_aliased_code_sha_as_current`, `test_ok_screen_refuses_a_stale_eval_sha`,
`test_eval_and_code_pathspecs_are_disjoint_and_cover_every_touched_file`,
`test_the_migration_runs_before_any_inflight_resume_and_before_the_baseline_rerun_rule` (a state with X05 `running` and a
stale `eval_sha` yields `rescreen X01` first), `test_launch_with_no_suite_resolution_defers_d00s_and_never_decides_on_the_old_record`,
`test_rescreen_of_a_seed2_header_files_under_seed2`, `test_rescreens_write_suite_beside_slow_and_keep_the_old_screen`,
`test_reevals_run_for_checkpoint_inputs_and_the_validator_accepts_that_shape` (pending per `role: pair` head until
`lad.reevals` records it at the current `eval_sha`; D00s reads exactly the recorded stamps),
`test_d00s_then_d00bs_are_written_in_order_with_supersedes_and_the_seed_dominance_line`,
`test_a_rescored_header_appends_its_rec_and_re_opens_at_the_next_stage` (D05 is NOT squashed by the launch reconcile; PASS →
D05 squashed by (f) via `rescores[].supersedes`, X03's `recs` gains `X03:c4`, status `running`, next stage c5, c4 never
retrained; FAIL → the new REJECT carries the current `eval_sha` and stays effective, `supersedes` names D05 (computed
before the writes), old squashed, D07 stands, no fresh C1 experiment booked, and a crash between the ledger and state
writes is completed on relaunch without a second eval; a re-opened X04 that fails its c4 floor under the new eval keeps that REJECT effective),
`test_a_floor_reject_under_the_current_eval_is_never_a_migration_item` (`floor_eval_sha == lad.eval_sha` → no rescore),
`test_rescore_pass_in_process_resumes_x03_before_x05` (one process, no relaunch: after PASS/PASS and FAIL/PASS rescores
D05/D06 are stored `squashed`, a SQUASH block names them, and `find_inflight` returns X03 or parks X05 — never X05),
`test_dead_walker_rejects_are_never_rescored_or_squashed_by_f`,
`test_a_failed_migration_item_halts_the_campaign_and_never_rewrites_the_header`,
`test_a_rescored_conditional_header_waits_for_its_predicate` (X04 is not resumed by `find_inflight`) and
`test_step8_dispatches_the_rescored_header_with_its_xid` (no new C1b experiment; the header flips to `running` with a
history row, and a crash mid-c3 resumes from its own checkpoint through `find_inflight`/`prepare_resume`),
`test_a_running_lower_ranked_header_waits_for_an_undecided_higher_rank` (X05 parks while C1/C1b/C1c are undecided; a twin
or baseline header resumes unconditionally), `test_item_from_header_takes_the_current_chains_eval_side` (frozen overlay,
current `eval`/`floor`/`categories`), `test_seed_terms_for_never_cites_d00b_under_rule_v2`,
`test_d00c_is_written_under_the_expanded_v2_rule_with_every_block_and_guard_term` (and an empty-terms or stale D00c is
treated as absent), `test_d00s_is_resolved_in_process_with_the_recorded_stamps` (the payload's pair inputs are exactly the
`lad.reevals` stamps, its `eval_sha`/`decisive_at` equal the driver's, a missing stamp halts `halt:resolution D00s`),
`test_c1b_screen_decides_with_f_mt_na_and_no_resolution_missing_halt` (screen and M2 suite row),
`test_a_migration_eval_interrupt_writes_nothing`,
`test_migration_is_off_under_smoke_stage_and_both_startups_load_the_suite_resolution`,
`test_d00bs_acceptance_failures_halt_before_any_decision` (incl. completeness: F_mt with no `R_eval`, an F member with
`R_eval` 0, `cat_tripod_min` with `R_eval` None, and a MISSING value on a B0 screen each halt; a survival guard with
`R_eval` 0 passes on the binomial floor; a failing D00s or D00bs writes no entry, only the rejected
diagnostic; the item stays pending and a relaunch after a tools fix recomputes it without an eval; `--accept-resolution
<kind> WHY` makes the next computation write it with `acceptance.ok = false`; a D00s refusal halts `halt:resolution D00s`),
`test_dry_run_prints_checkpoint_shape_inputs`,
`test_c1c_skip_follows_c1s_squash`, `test_milestone_fails_on_two_of_three_rows_with_the_suite_worse_on_two_units` (light
and ring scored with D00s' copied `by_row_rule`, D00b's v1 seed terms and D00's `by_row` R — reproducing the v1 R, e.g. F
seed 2.04 — so a light + ring WORSE pair can FAIL; the suite nominal on the MILESTONE line),
`test_promotion_stages_evaluate_promotion_eval` (`promotion_stage_list`/`final_s2_stage_list` yield `[suite, light, ring]`,
fall back to `PROMOTION_EVAL`, and `item_from_header` on a milestone header gives the same list),
`test_milestone_suite_row_takes_decisive_present_from_the_promotion_stage` (after a C1b-style adoption M2 scores F_mt and
`move_turn.ratio` as WATCH and keeps `move_turn.survival`),
`test_milestone_keeps_todays_unreadable_row_handling`, `test_stage_report_renders_one_line_per_category_with_decisive_marks`,
`test_decision_line_carries_trained_untrained_and_per_category_rows` (format/parse round trip),
`test_validator_rule_v2_and_category_stages`; existing tests edited by name: the curriculum stage-order test (`["suite"]×4`),
`test_one_experiment_trains_evaluates…` (`["suite"]`), the floor/retention/C1b-predicate tests (category form), the
`code_drop`/pathspec pins (the manifest and `run_phases.py` now in `EVAL_PATHSPECS`), the null-construction band (nominal
from the expanded rule). `tests/unit/test_crab_hex_omni_resolution.py`: `make_run(suite=True)`; `RULE` v2;
`test_clustered_draw_keeps_every_hold_count_and_hand_split`, `test_suite_pairing_on_env_index_pairs_within_category_and_hand`,
`test_resolve_expands_the_v2_rule_and_publishes_F_mt_R` (R for all six blocks and both move_turn guards),
`test_null_all_tie_scores_exactly_the_ratio_systems_blocks_and_guards`, `test_reducer_check_covers_cat_tripod_min`,
`test_resolve_eval_dir_requires_the_completion_marker_and_matches_checkpoints_by_sha` (a stamp without `complete.json`, or
with missing raws, is refused with a `ResolutionError`; an absolute vs repo-relative checkpoint path still matches),
`test_load_row_verifies_a_suite_row_under_the_full_v2_rule_and_rule_less` (`turn_survival_*` unresolved, `surv:<h>` NaN
outside hold h, and `summarize_run_dir` on a suite row reports re-scored, not refused),
`test_d00s_copies_light_and_ring_by_row_from_d00`,
`test_one_member_block_R_eval_is_exactly_one`, `test_cat_ids_reproduce_summarize_cat_keys` (extends the registry test),
`test_survival_guard_R_has_the_binomial_floor`, `test_load_row_refuses_a_populated_hold_that_is_not_the_episodes_own`,
`test_resolve_eval_dir_takes_the_newest_stamp_or_refuses`, `test_seed_pair_role_is_excluded_from_cross_head_pairs`,
`test_legacy_pairs_are_bit_identical_under_the_clustered_path`, `test_summary_with_survival_uses_the_suite_block`,
`test_the_screen_row_is_the_suite`, `test_null_band_uses_the_clustered_draw_and_the_floored_guard_R`. The B0-pair control
check lives with `_write_d00bs` in the candidates module, not in the resolution tests.

## Part C — records, docs
`$LAD/CHARTER.md`: a new section "Per-category evaluation (2026-09-21)" carrying D1–D10, the suite table, the rule-v2 text,
the migration and resume sequence, and the caveats below; `$LAD/PLAN_AMENDMENTS.md` items 31+ (item 30 records audit passes
2–N and the user decisions of 2026-09-21; the old plan's decisions 4/5/6
superseded); `$OMNI/README.md` (the suite row, the categories module, `eval_sha`, the new metrics); `docs/FOLDER_LAYOUT.md`;
the memory note.

## Execution steps (ultracode: Workflow dispatch; routing per plan-model-routing; roster ok)
0. **Reference capture at the pre-change HEAD (the approval record commit on `464be672`; code-bearing tree identical to
   `464be672` and `9c2c1fba`)** → sonnet · low: the three cfg-identity suites into
   `$R/parkour/outputs/omni_plan/dumps_cat_base` (dir pre-created; 18 JSONs); the existing `rc.code_sha()` recorded
   (`code_sha_at` does not exist until step 3; it equals `code_sha_at(HEAD, CODE_PATHSPECS_V1)` by construction); and
   `tests/unit/fixtures/gait_eval_compile_snapshot.json` = sha256 of `cmd`, `segment_id`, `steady_mask`,
   `delta_yaw`, `world_speed/dir/face` for every committed row of `scenarios_{v1,v2,morph}.yaml`, `scenarios_side.yaml` and
   `scenarios_omni.yaml` compiled at dt 0.02.
1. **Harness (A1–A5)** → opus · high (shared scorer; byte-identity of forward/sideways scoring is the catch), then three
   fresh lenses (legacy compile/score byte-identity incl. the golden and the step-0 snapshot; spec conformance vs D1–D3;
   test strength with mutation checks) and a fix loop to convergence.
2. **Categories module (B1)** → sonnet · high. Parallel with 1 (disjoint files).
3. **Tools (B2–B6)** → opus · xhigh, two implementers on disjoint files (A: `resolution.py` + its tests; B: the rest,
   starting with B7's fixture split — the frozen v1 `candidates` fixture and the repointed v1 tests — before B6 rewrites the
   live yaml; the D8 training-seam pin constant is computed by running the test's own pin function over the UNEDITED
   `git show 464be672:<run_phases.py>` loaded by path — never over the edited file, which would prove nothing)
   sharing the `aggregate.suite` / `cat.<c>.<key>` contracts, then four lenses (numerics vs D5/D10 with a by-hand
   recomputation from a synthetic suite; state machine vs D9 incl. the re-open and cause (f); integration with the real
   seams and a dry run against a COPY of the paused campaign dir; contract/yaml/notify) and a fix loop.
4. Records and docs (Part C) → sonnet · medium.  5. Fresh-eyes review of the whole diff against this plan → fable · high.
6. Verification V1–V4 (below) → sonnet · medium for the runbook, main for the GPU step (V3).  7. Commits (Part G) → haiku ·
   low; then re-print V4's alias proof against the Part G tip.  6b. **V6 stop drill** → main, after the commits (`main()`
   refuses any non-dry-run launch on a dirty code- or eval-bearing tree, and `--smoke-stage` is not exempt).
8. **Relaunch and migration (D9)** → main: `ledger_cli unstop`, then `launch_candidates.sh --campaign-dir $LAD
   --accept-eval-change "per-category suite (user 2026-09-21)" --accept-candidates-change "categories, floors, rule v2"`;
   heartbeat; the persistent `notify.log` monitor relaying every line as push + chat card. **Budget before the first
   training window resumes:** 7 suite evals (2 rescreens, 3 reevals, 2 rescores) ≈ 50–60 GPU-min (7 × 7–8 min plus the
   driver's ticks) + the D00s bootstrap (CPU) and ≈ 6 GB of raws + JSON (≈ 0.85–0.95 GB per suite; disk ample); then C1 c5–c6
   (3500 it ≈ 1.5 h of training at the recorded 0.43 h / 1000 it, ≈ 1.7 h with its stage evals) before C1's screen; every
   later screen and promotion carries one suite (≈ 0.75 GB). Read, in order: the MIGRATION lines, the `D00s` then `D00bs`
   RESOLUTION lines (the F and F_mt seed-dominance sentences first; a `halt:resolution` stops here), then the C1 and C1b
   rescore verdicts, then the queue. No SQUASH line is expected before a rescore verdict (cause (f) needs the `rescores[]`
   record); a SQUASH line follows each rescore verdict (PASS or FAIL): it names D05 (plus D07 by cause (c) on a C1 PASS) or
   D06. Stop-and-wait points as before.

Order: 0 → (1 ∥ 2) → 3 → 4 → 5 → 6 (V1–V4) → 7 (commits) → 6b (V6) → 8 (V5 runs inside 8). A step that fails twice reruns
one notch up, never lower.

## Verification
1. **Pure suites**: `env -i HOME=$HOME PATH=$PATH $PY -m pytest tests/unit -k "crab_hex or travel_frame or parkour_command
   or exposure or policy_of_record or experiment_layout or gait or candidates or resolution or eval_categories" -q
   --continue-on-collection-errors -p no:randomly` green (baseline 1495 passed + 32 pre-existing collection errors); the
   golden fixture byte-identical; the step-0 compile snapshot matches.
2. **Config identity**: the three cfg-identity suites into `dumps_cat` — all 18 dumps byte-identical to `dumps_cat_base`.
3. **Suite dry run** (one Isaac process, a REAL head — `--zero-actions` falls in a second and never exercises the wave
   path), run through the eval seam so the environment is the composed one (`rp.eval_morph(c_scratch, ph.PHASES["1c"],
   ckpt, "suite", "v3_dryrun", trained_categories="stand,fwd,side,back,diag,turn,move_turn", extra_args=("--episodes",
   "7", "--waves", "10"))` with `c_scratch` a Campaign on a scratch dir, so nothing lands under a real rescreen/reeval root
   that `resolve_eval_dir` could later pick up; under a 48 G scope, `KRABBY_FLAT_TERRAIN_GEOM=ring` included). Asserts: `run_meta.episodes_total == 70`, `env_index_map` unique
   0..69 with n = 7 per hold, `wave_seeds` recorded, `schedule_state_max_deviation == 0.0`,
   `command_override_max_deviation < 1e-4`, 70 raw npz + 70 episode JSONs, `aggregate.suite.survival_by_hold` has ten keys,
   `by_category` has seven, `summarize()` returns seven `cat` keys, `summary.md` prints the category block; wall time and
   disk recorded; the first head is named: 1c-004's ckpt. Then **cross-head wave determinism**: run the same suite on a
   second head (X01's ckpt; same scratch Campaign, tag `v3_dryrun_x01`) and, for EVERY wave k ≥ 1, assert the step-0 state
   (`root_pos_w[0]`, `root_quat_w[0]`, `joint_pos[0]` of `episode_{7k:02d}.npz` — wave k, env 0 at episodes 7) is identical
   across the two heads. The check is only meaningful if the heads' per-wave fall COUNTS (`run_meta.termination_reason`
   sliced per wave) differ in at least one wave before a compared wave — wave 0 is `stand`, which the recorded heads never
   fall on; if they never differ, record "determinism not exercised" and rerun with a head pair that falls differently or
   more episodes per wave; record once whether the spawn has a real random draw (root-reset offset or joint-reset noise).
   **Label-vs-command check** on both dry-run heads (`schedule_state_max_deviation` compares the command with the INSTALLED
   schedule, so it cannot see a wave whose installed row differs from its label): from the raw npz, for each wave k and env
   e (`episode_{7k+e:02d}.npz`), over steps with `segment_id == k+1` and t < that env's `n_steps`, with
   `yaw[t] = gait_eval.metrics.yaw_from_quat_wxyz(root_quat_w[t])` and `dir_k`/`face_k` from the manifest LABEL: (a)
   ‖`cmd_applied[:, :2]`‖ equals label k's speed within 1e-3 (fwd/diag 0.25; lat_l/lat_r/back/move_turn/move_turn_r 0.22;
   stand/turn_* 0); (b) for every hold with speed > 0, max |wrap(`travel_yaw` + yaw − dir_k)| < 1e-3; (c) for every hold,
   max |wrap(`heading_err` + yaw − face_k)| < 1e-3; (d) |yaw(`root_quat_w[0]`)| < 1e-3 in every episode (the spawn pin
   survives each wave reset). `travel_yaw` and `heading_err` are body-relative and the facing dead band (0.2/1.5 = 0.133 rad)
   leaves yaw drift uncorrected, so only these world-frame identities are exact (they read 0.0 on X01's recorded raws); the
   first-step `heading_err` sign is printed for information only; `run_meta.travel_yaw_by_hold` is never the source.
4. **Driver dry run** against a scratch COPY of the paused campaign dir (v2 `candidates.yaml` + the recorded
   `state.json`/`ledger.json`), with a new dry-run section that iterates the real `migration_item` derivation and
   `find_inflight` order on those records (not the synthetic ledger): the pre-suite state (no `lad.eval_sha`) takes the
   alias path; prints the alias proof (`code_sha_at` under V1 at `464be672` == `lad.code_sha` = `4c3f99c9a2c5…`; the
   narrowed pathspecs at `464be672` = `a617f5ee59d1…`; `code_sha()` hashes `git ls-tree HEAD`, so before the commits the
   working-tree half is `dirty_paths()` over the narrowed `CODE_PATHSPECS` being empty, and the equality itself is re-proved
   against the Part G tip after step 7), the new `eval_sha`, that the launch reconcile squashes nothing (D05/D06 still decided), and the queue
   `rescreen X01 → rescreen X02 → reeval 1c-004 → reeval 2b → reeval 2a-reeval → D00s → D00bs → rescore X03:c4 → rescore
   X04:c2 → C1 (X03 resumed at c5) → [C1 pair, if speculative] → C1b (X04 at c3, if its rescore passed and its predicate
   holds) → C1c → X05 …` — the first item is a rescreen although X05 is recorded `running`, and X05 is logged `not resumed
   (waiting on C1 at rank 1)`; the validator accepts the checkpoint-input shape and refuses `rule_version != 2`;
   `ledger_cli show D05` prints `eval_sha: — (predates the suite)`.
5. **`resolution.py` on the reeval outputs** (step 8): `D00s` with per-category R and a published R for every block (F_mt
   included, both unit systems), the light/ring `by_row` tables and `by_row_rule` copied from D00, the binomial floors named,
   the per-survival-guard null TIE rates, the null band, `R(D00bs) ≥ R(D00s)` per block (recorded, not a gate), the B0
   s2-vs-s3 construction check in both unit systems and its eval-only-R classing, `evidence.acceptance`; `_reducer_check`
   passes on the real rows.
6. **Stop drill** on the suite path (step 6b, after the Part G commits): `--smoke-stage 1a:20:suite` on a FRESH scratch
   campaign dir holding only a copy of the v2 candidates.yaml (no state/ledger/resolution — a copy of `$LAD` would run the
   migration and then resume X05 before the smoke, even with today's code), `systemctl --user stop` during wave 3 →
   `await_user:stop`, no orphan, and the relaunch re-runs the whole suite (a stop mid-wave kills the harness before any
   output is written; the interrupted eval is re-run by `_run_eval`'s freshness check, and D10's completion-marker rule keeps
   any partially written stamp out of every later read).

## Part G — commits (branch `feat/crab-hex-omni-task`; nothing pushed)
1. `feat(gait_eval): category suite -- one hold per wave, waves in one launch, own-hold survival, pooled by_category, motion/direction metrics (world rows; body scoring byte-identical)` — includes the manifest row and its pin edit.
2. `feat(omni eval): eval_categories -- the trained categories of a recipe`
3. `feat(omni driver): eval-side seams -- suite summarize, trained-categories flag, own-hold survival`
4. `feat(omni ladder): decision rule v2 per category, eval_sha with the training-tree alias proof, the rescreen/reeval/rescore migration, category floors and retention, suite milestones` — includes `$LAD/candidates.yaml` (v2) and `tests/unit/fixtures/candidates_v1_2026-09-19.yaml`.
5. `docs/record(omni ladder): per-category evaluation -- charter section, amendments, README`

## Non-goals
No change to training code, presets, rewards, the RSI bank or the command term; no `suite_ring__omni` (light and ring stay
legacy rows, scored at milestones by the frozen v1 per-row rule, D7); no uniform-direction `diag`; no in-band `turn_band` hold; no new categories; no retraining of B0; no change to the
ranked candidate list or its conditionals; no per-env tensors in the schedule hook; `drift_speed` stays WATCH.

## Caveats carried into the CHARTER
1. `move_turn` is scored RELATIVELY: the hold is infeasible by construction, so its facing block `F_mt` compares two heads
   under the same impossible demand, and `cat.move_turn.ratio` guards against winning it by walking less. Its `tripod` and
   direction numbers stay WATCH, and `achieved_speed` stays WATCH except as the absolute-units twin of the ratio guard (on
   this hold ratio = achieved_speed / 0.22, so the speed is guarded once, through the ratio).
2. `turn` is decisive for every recipe although each trains a 0.6-rad facing band against a 1.57-rad demand; the comparison
   is symmetric and the band/encoding are printed. The recorded 1c head settles on 1 of 62 `turn_l90` episodes of the legacy
   row (31 of 60 on its legacy `turn_r90`, a return turn to face 0 — not the suite's mirror).
3. The facing blocks are expected to stay seed-dominated, F_mt far more than F: on the recorded B0 pair split along the v2
   blocks (legacy per-hand slow-row proxy, the D00b construction `max(|z̄|, rms/√k)`), `R_F^seed` ≈ 1.2 for the in-place turn
   members and `R_F_mt^seed` ≈ 3.7 (move_turn's two facing members differ between seeds by ≈ 3.7 `R_eval` each; the `_abs`
   twins ≈ 1.2 / ≈ 3.7). The suite's pooled hands and fresh n ≈ 200 move_turn episodes may shift both. `D00bs` measures every
   block before any decision, and its line says per block whether a single-seed screen can clear it at all.
4. `run_phases.py` sits on the eval side of the hash; its training seam is pinned by a source hash, not by `code_sha`.
5. The stage floors are absolute (0.10 new / 0.30 previous) and band-tested at 99 % on the category's distinct-env count
   (100 for every category, paired ones included): a stage passes down to p̂ ≈ 0.05 / ≈ 0.21. The translation and move_turn proxies are 8-s legacy (the turn proxies are already 12-s)
   holds; the suite's 12-s holds read ≈ p^1.5 under a constant hazard (still above the floors). They are calibrated on
   conditional per-hold survivals, a proxy the C1/C1b rescores validate with the first real per-category numbers.
6. A suite eval costs ≈ 2× a legacy row in time and ≈ 3.2–3.5× in disk; raws are kept (D00 re-scoring needs them). The
   migration adds ≈ 1 GPU-h and ≈ 6 GB before the first training window resumes (step 8).
7. `turn_r90` and `move_turn_r` change meaning relative to the legacy rows (both now mirror their left-handed twin from face 0).
   Predicates that read a screen's legacy per-hand keys through `predicate_context` (`screen.<key>`, e.g. C10's
   `screen.yaw_progress_turn_min < 0.10`, set on the legacy row) now read the suite screen; C10 is deferred, and its
   predicate is re-expressed on `cat.turn.yaw_progress` when it is undeferred.
8. `by_category` pools both hands; a handedness asymmetry is visible only in the WATCH `by_hand` rows.
9. Three blocks and nine guards cost sensitivity: a null candidate reads some WORSE about 15 % of the time (11 % under the
   pass-1 rule), so a clear REJECT is worth reading before it is trusted. Every DECISION line quotes the overrule command,
   and `decide C<id> --choose run` re-screens. The count is the price of judging every trained category rather than a pooled
   number; the nominal is printed on the RESOLUTION line and asserted against the measured null. The survival guards'
   binomial floor is unpaired, so at mid-range survival it dominates the paired `R_eval` (≈ 0.16 at p̄ = 0.5, n = 100): a
   category's survival must drop by ≈ 0.10–0.16 (n = 100, p̄ 0.9–0.5; ≈ 0.07–0.12 for side/turn/move_turn at n = 200) before
   it reads WORSE — fewer spurious REJECTs at the price of sensitivity to
   falls. The all-TIE speculative pick weights each block by its own R (D5), so F_mt's seed noise does not decide it.
10. A present-but-watch category keeps only its `survival` guard: after a C1b or C7 adoption (`{"*_turn": 0.3}`), move_turn is
   watch for every later screen and for the M2/M3 milestones (`rescale_to` keeps ang_scale), so its facing and speed are
   WATCH lines from then on and only a move_turn fall regression can REJECT or FAIL.

## Open risks
- `env.reset()` plus a per-wave override re-install is a new harness path, verified by V3 and the unit tests around
  `stack_waves`, not by a full-loop unit test.
- If the recorded launch commit is not in `git log -n 300`, the alias proof fails → fall back to `--accept-code-change`
  (which retrains B0); the CHARTER records which path was taken.
- The C1/C1b rescores may pass the floors and the resumed chains may still lose at their screens: that is the honest outcome.
- X05 (C2) resumes only after C1, C1b and C1c are terminal (D9 item 6); if any of them is adopted, X05 is squashed by
  cause (a) — an accepted mispredict cost.

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
  three blocks and nine guards, null all-TIE ≈ 0.78, milestone FAIL rate ≈ 1 % (the pass-1 rule; since pass 2 it is the
  suite row's WORSE rate inside the three-row rule, D7), spurious-REJECT rate ≈ 15 % (caveat 9).
- After pass 1 the user first declined further passes, then (same day) asked for an audit loop, cap 6, auditors on opus·xhigh.
- **Pass 2** (four fresh whole-document auditors — harness, driver state machine, statistics/resolution, assembly — three
  number spot-checks, and an adversarial verifier on every substantial finding, escalated when blocking or disputed): 44
  raw substantial findings → 1 blocking + ≈ 21 distinct substantial, ≈ 35 minor — all applied. Blocking: `_screen_of` reads
  only the legacy `slow` row, so every new screen would be empty (every candidate REJECT-uncertain) and every rescreen would
  re-file the old screen under the new `eval_sha` (→ it reads the suite row). Substantial: F_mt never reached the code's
  hard-coded two-block tuples and the acceptance band could not catch it (→ the block set derives from the expanded rule at
  every named seam; `expand_rule`/`composite_guards` shared from `eval_categories.py`); cause (f) would squash D05/D06 at the
  launch reconcile before any rescore (→ keyed on `rescores[]` at the current `eval_sha`; a failed migration item halts the
  campaign, never the header); `find_inflight` would resume X05 ahead of re-opened higher-ranked candidates and step 8
  would book a fresh C1b instead of resuming X04 (→ rank gate; xid-carrying dispatch; `item_from_header` takes the current
  eval side); a pre-suite state has no `eval_sha`, so the `code_sha` refusal would steer to `--accept-code-change` and a B0
  retrain (→ alias path first); headers stamped only at birth would be rescreened needlessly (→ `eval_sha` written with the
  screen); the all-N/A `undecided` branch would re-screen forever (→ terminal SKIP, confidence none); own-hold survival had
  two definitions that `_reducer_check` would trip on (→ the `n_steps ≥ end` convention); `cat_tripod_min` had no seed term
  and failed `_reducer_check` (→ the shared composite helper everywhere); the absolute-units move_turn guard had no
  guard-level R or seed term (→ `guards_abs`); a saturated survival was still MISSING on the pooled path (→ pooled binomial
  floor); the ten waves reuse the same 100 randomised robots (→ env-clustered bootstrap; distinct-env n in the floor band);
  `seed_terms_for` could fall back to the wrong-lineage D00b (→ never under v2); V3 could not run through `eval_morph`, and
  V6 could not run before the commits or on a copy of `$LAD` (→ `extra_args` + scratch Campaign; step 6b on a fresh dir);
  legacy rows would gain an `env.seed` call (process-state sensitivity) (→ the wave path on suite rows only); D00bs
  acceptance failures were only recorded (→ `halt:resolution`); caveat 3's `R_F^seed` ≈ 1.6 had no source (→ ≈ 1.2 for F,
  ≈ 3.7 for F_mt on the recorded pair); the Context hashes were another row format (→ driver-format values; HEAD
  `464be672`). **User decisions (2026-09-21, pass 2):** milestones keep the three-row rule with the suite in place of `slow`
  (light/ring on the frozen v1 rule); every PRESENT category keeps its `survival` guard; the speculative sign weights each
  block by its own R. **Surfaced, not reversed (settled decision 2):** the move_turn ratio guard versus a facing-for-speed
  trade — D3's unsupported "a few percent of speed for 0.2 of facing is an ADOPT" was replaced by the recorded data point;
  the rule stands.
- **Pass 3** (same structure; auditors fresh on the revised text): 70 raw findings → 4 blocking + ≈ 10 distinct
  substantial, ≈ 45 minor — all applied; no settled decision touched. Blocking: the legacy `turn_survival_*` ids resolve on
  a suite row (its union labels contain the nested holds) while summarize() returns None for them, so `_reducer_check`
  would refuse every suite row and D00s could never be written (→ undefined on a `category_suite` row: unresolved, omitted
  from the rule-less fallback, `surv:<h>` NaN outside its own hold); pass 2's cause (f) would also squash the REPLACEMENT
  REJECT a rescore FAIL writes and any later floor REJECT of a re-opened header (→ keyed on `rescores[].supersedes`); pass
  2's rescore derivation also matched floors rejected AFTER the relaunch, a second-chance floor (→ `header.floor_eval_sha`
  must differ from `lad.eval_sha`); pass 2's light/ring v1 rule had no named rule or seed source, so milestones could never
  FAIL (→ D00s copies D00's frozen rule as `by_row_rule`; D00b's v1 seed terms; today's `_any_worse`). Substantial: V3's
  determinism check was vacuous (→ every wave, differing fall counts, named heads); the freshness rule admitted a stamp
  killed during its raw writes (→ `complete.json` written last); the test module reads the live candidates.yaml (→ frozen
  v1 fixture); `halt:resolution` was neither sticky nor clearable (→ `evidence.acceptance`, re-derived, `--accept-resolution`);
  the step-8-dispatched X04 kept its non-running status (→ flipped to `running`); the all-N/A SKIP loops for a conditional
  candidate (→ exempt from cause (b)); one unit system in resolution left C3a/C3b undecidable (→ the union of both);
  promotions never ran the suite (`promotion_stage_list` hard-wires `EVAL_FULL`) (→ named seam, `EVAL_FULL` untouched);
  milestones had no source for trained categories (→ passed explicitly from the promotion stage); reevals had no
  derivation key (→ `lad.reevals` by `eval_sha`). Minor: stale "stratified" wording, wave-0 seeding, `n_env` on `cat`,
  `--trained-categories` on suite rows only, the watch list, RETENTION metric kinds, survival-only rescue, budget arithmetic
  (7 × 7–8 min = 49–56 min), the caveat-9 range, the settle vocabulary, `watch` reasons, task fields, fixture sizes.
- **Pass 4** (classified under the revised audit guide — blocking / deferred / nit; only blocking fixes and one-line nits are
  edited in): 52 raw findings → 5 distinct blocking, ≈ 10 one-line nits, ≈ 15 deferred. Blocking: pass 3's `decisive_at`
  subset halt also fired on C1b/C7's memberless F_mt, halting the campaign after an 8000-it screen (→ strict non-empty
  subsets only; N/A never halts); the D00s item called `res.main`, which cannot carry the reeval stamps, `eval_sha`,
  `decisive_at` or D00's tables — D00s would halt on every relaunch (→ in-process `resolve()` with explicit arguments, CLI
  mirror); on a rescore FAIL `supersedes` was computed after the new REJECT, naming it, so (f) would squash the current
  verdict and keep the contaminated one (→ computed before any write over stale-eval entries; ledger-then-state order;
  crash-completable); `_d00c_lookup` was unnamed and would write EMPTY v2 seed terms that `seed_terms_for` prefers, so
  post-C1 decisions would run on eval-only R (→ a named seam, expanded like D00bs); pre-suite headers have no per-stage
  trained-categories record, so rescreens, the D00bs check and the rescore floors had no sets (→ D4's general
  compute-and-append rule, written before the eval). Also blocking and fixed in a line each: an interrupted migration eval
  would `_mark_interrupt` X01 into `running`; which null construction gates the halt; `n_env` reused for the raw loop bound.
  Nits: the training-seam pin now covers module constants, `block_names(cand_spec, rule)`, `resolution_id` labels at every
  site, the watch list, flag scoping, RETENTION's `previous_categories`, "clearable", the 8-s proxy note, the header's list
  of amended 2026-09-18 decisions, acceptance records naming their entry. Deferred items are listed for implementation in
  PLAN_AMENDMENTS item 30 (see On approval), not in this plan.
- **Pass 5** (revised guide): 32 findings → 3 distinct blocking (all verified), 14 deferred, 15 nits. Blocking: cause
  (f) is keyed on state but the driver recomputes its view only after ledger writes, so in the same process the rank gate
  would resume X05 (≈ 3.3 GPU-h) ahead of the re-opened chains (→ `squash_and_report` after every rescore state write);
  pass 4's acceptance design wrote a failing D00s/D00bs entry, after which no item could ever re-open — a tools bug there
  had no clean exit (→ acceptance computed before any write; a failure writes only a rejected diagnostic and the item stays
  pending; `--accept-resolution <kind> WHY`); V3 could not see a wave whose installed command differs from its label, so a
  wave-loop slip would mislabel every migration eval silently (→ a label-vs-command check from the raw npz). One-line
  fixes: `_screen_of` keeps the training-log tails C11's predicate reads; the seam-pin constant is taken at step 0 before
  B2 edits; `decision_rule` v2 keeps `bootstrap`/`control_head`/`charter_floors`; the FAIL step (1) persistence note; disk
  ≈ 6 GB; turn proxies are 12-s; seed-dominated wording; `ManagerBasedEnv.seed`; block_names and milestone_reference shapes.
- **Pass 6** (the last within the cap): 30 findings → 2 distinct blocking (both verified), 14 deferred, 13 nits.
  Blocking: pass 5's V3 label checks compared body-frame `travel_yaw` / `heading_err` with world labels, so a correct
  harness would fail about half the translation episodes and stop step 6 (→ exact world-frame identities, `+ yaw(root_quat)`,
  residual 0.0 on X01's raws, plus a spawn-yaw check); the D00s/D00bs acceptance did not require every block and guard to
  be resolved, so a missing term would make that block a permanent TIE silently (→ completeness in both unit systems,
  survival guards exempt on the binomial floor). One-line fixes: the FAIL crash-completion predicate (independent of
  `floor_eval_sha`), resolution file before entry, one halt spelling, the stage_specs list lookup, category ids unresolved
  on every non-suite row, `_screen_of`'s trained-categories source with an eval_mismatch halt on an empty set, X04's
  predicate read from candidates.yaml, the seam-pin constant computed over the unedited blob, disk ≈ 3.2–3.5×, the SQUASH
  line after either verdict, the D9 closing sentence. **The cap was reached with 2 blocking findings fixed but not yet
  re-audited** (trend of distinct blocking: 4 → 5 → 3 → 2; each pass's blockers were fallout of the previous pass's fixes).
  **The user accepted the plan as is at the cap (2026-09-21)** rather than run further passes.
- The per-step review lenses and fix loops inside execution steps 1, 3 and 5 remain, and step 5's fresh-eyes review of the
  whole diff is still the last gate before the commits and the relaunch.
