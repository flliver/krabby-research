<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-09_0106_stride_length_v4/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0106_stride_length_v4/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`. Older records may call this directory `sim_fine_tuning/stride_length_v4`.

# Changelog: `sim_fine_tuning/2026-08-09_0106_stride_length_v4`

**Milestone 18, Task 1 item**: §2.2 "Stride length" (continued -- redesign of the term after
`stride_length_v3`'s own data showed neither v2 nor v3 actually beat `motor_reversal_on`'s
*incidental* stride-length improvement on the harness's own literal metric, despite v3 being the
strongest run in the series on every training-stability measure).

**Predecessor**: `sim_fine_tuning/2026-08-08_1701_stride_length_v3` (v3 design, kept, but flagged with two open
threads: worse-than-baseline 2b2 stability, and an unexplained non-zero tripod score at 2b2).

## Change (v4 design)

Redesigned `reward_stride_length` again (`crab_hex_stride_reward.py`, `RewardStrideLength` in
`parkour_isaaclab/envs/mdp/rewards.py`) after analysis showed v2 measured the *joint* (exploitable,
no floor cost) and v3 measured the *body* (redundant with existing tracking rewards, multi-counted
across simultaneously-planted legs, no cost on short swings).

**v4 semantics**: per-foot, world-frame, **touchdown-to-touchdown displacement**, rotated into the
base frame using the *stored* yaw at the previous touchdown (matching the Task 0 harness's own
`stride_metrics()` convention exactly), projected onto the commanded direction, paid **at
touchdown**:

- **Signed, not clipped.** A backward replant costs what a forward one pays (`sign(d) * |d|**power`),
  so an oscillating replant (forward then back) nets to roughly zero -- closes the v2-style
  exploit structurally instead of patching around it.
- **`min_swing_duration=0.1s`** replaces v3's `min_phase_duration`: a swing shorter than this is an
  *invalid* touchdown -- not paid, and critically the per-foot reference is left **untouched**, so
  a blip cannot reset the baseline a later genuine stride is measured from.
- `power=2.0` kept (convex -- one long stride still outscores several short ones covering the same
  net range).
- Replaces v3 entirely (same term name, new semantics) -- one clean definition, one-change
  attribution vs the v3 run, per Task 1 §3's discipline.

Weight unchanged at `0.5`. `penalty_motor_direction_reversal=-0.3` remains active (unchanged from
every prior run in this series).

Verified before training: 11 new/rewritten unit tests
(`tests/unit/test_crab_hex_stride_reward.py`) covering forward payout, an oscillation-nets-to-zero
regression test (the specific exploit this design targets), a real backward-replant penalty, blip
transparency (no pay, no reference update), yaw-rotation correctness matched against the harness's
own `stride_metrics()` formula, command-norm gating with reference-still-updates, convexity,
slip-shortens-payout semantics, first-touchdown seeding, per-foot/per-env batching, plus zero-agent
smoke on both teacher/2b2 and flat-walk configs and a 300-iteration flat-walk training smoke
(scratchpad, not committed to this run) confirming `Episode_Reward/reward_stride_length` stays
finite under the new signed semantics.

Full run: flat (20000 iter, from scratch) -> gait-eval -> bridge -> 2b1 -> 2b2 (4 batches, 500
iter, matching the series budget) -> gait-eval.

## Metric delta vs stride_length_v3 (direct predecessor) and the full series

**Training**:

| | v3 | v4 |
|---|---|---|
| flat crab_failure | 2.49% | 3.52% |
| bridge crab_failure | 3.91% | 5.75% |
| 2b1 crab_failure | 1.56% | 4.69% |
| 2b2 crab_failure (batch range) | 6.8-14% | 14.5-18.25% |
| 2b2 gates cleared (best single checkpoint) | 4/5 | 4/5 (`model_22000.pt`) |
| 2b2 forward_progress (peak) | 0.1496 (closest ever to the 0.15 gate) | 0.123 |

v4 is stable throughout (no v2-style escalating collapse), but is worse than v3 -- its own direct
predecessor -- at every stage tested. `forward_progress_along_command` remains the persistent
series-wide holdout, and v4's peak (0.123) is further from clearing it than v3's (0.1496).

**Gait-eval** (Task 0 harness):

| | flat: v3 / v4 | 2b2: v3 / v4 |
|---|---|---|
| schedule_completion_rate | 100% (0 falls) / **100% (0 falls)** | 80% (2 falls) / 80% (2 falls) |
| tripod_score (median) | 0.0 / **0.0242** | **0.0250** / 0.0050 |
| slip_ratio | **1.7%** / 1.78% (tied) | **3.0%** / 4.33% |
| tippy_tap_fraction | 19.1% / **39.2%** | 21.0% / **28.7%** |
| stride length (pooled mean magnitude, n_td-weighted) | **0.178 m** / 0.170 m | 0.162 m / **0.189 m** |

Full series stride length (pooled mean magnitude), for context:

| | flat | 2b2 |
|---|---|---|
| baseline | 0.120 m | 0.139 m |
| motor_reversal_on (no stride term) | 0.180 m | 0.171 m |
| v2 | 0.183 m | 0.156 m |
| v3 | 0.178 m | 0.162 m |
| **v4** | 0.170 m | **0.189 m** |

**The headline result is genuinely mixed, not a clean win.** v4's 2b2 stride length (0.189 m) is
the best of the entire series and is the first run to actually beat `motor_reversal_on`'s figure
that motivated this whole redesign -- confirming the reward now measures *something* that moves
the literal target metric. But this comes bundled with the **worst tippy-tap fraction in the
series at both stages** (39.2% flat, 28.7% 2b2 -- more than double v3's flat figure), worse
training-time `crab_failure` at every stage than v3, worse `slip_ratio` at 2b2 than v3, and a
*lower* forward-progress peak than v3. At the flat stage specifically, v4 does not even clear its
own predecessor on stride length (0.170 m vs v3's 0.178 m).

**Tripod score anomaly moved, not resolved.** v3 flagged an unexplained non-zero tripod score
(0.025) specific to its 2b2 checkpoint. v4's 2b2 tripod score is back to near-zero (0.005,
consistent with every other run's typical range) -- but v4's *flat* checkpoint now shows the same
magnitude of anomaly (0.024) that v3 showed at 2b2. The anomaly appears to be a generic side effect
of directly rewarding footpad displacement (in either the v3 body-progress or v4 per-foot form),
not something either design "caused" or "fixed" -- it just surfaces at a different stage of
training depending on the specific term. Still not investigated further; still not remotely close
to a clean tripod gait (~0.5-1.0).

**Working hypothesis for the stride-length/tippy-tap split.** `tippy_tap_fraction` is computed from
raw (non-debounced) air-time intervals pooled across all six legs; `stride_metrics()`'s pooled
magnitude is computed after debouncing (`min_contact_steps`/`min_air_steps`) filters out exactly
the short blips that drive tippy-tap. It's plausible v4 is producing *more* total touchdowns with a
wider spread of swing durations -- many short (undebounced, tippy) and some genuinely long (survive
debouncing, pull the pooled-magnitude average up) -- rather than a uniformly longer, calmer stride
across the board. That would mean v4's raw reward signal is being satisfied by a bimodal gait
rather than the intended "fewer, longer steps" outcome. Not confirmed against raw per-touchdown
duration data here; flagged as the first thing to check before attempting a v5.

## Verdict

**Not kept -- reverting to v3.** Per Task 1 §3's discipline ("revert changes that do not move
their target metric"), the letter of the target metric (2b2 stride length) did move, and moved to
the best value in the series. But v4 is compared against `stride_length_v3`, the run it explicitly
replaced, and on nearly every other axis this series has tracked -- tippy-tap (worst in the series,
both stages), training-time `crab_failure` (worse at every stage), `slip_ratio` at 2b2, and the
`forward_progress_along_command` peak -- v4 is a regression relative to v3, not an improvement. v3
also already cleared the literal Task-1-motivating comparison at its own 2b2 stage on a fully
apples-to-apples basis when weighed against its dramatically better stability. Restoring v3's
design is the recommended action; v4's code is kept in `sim_fine_tuning/` (not deleted) as the
"reverted attempt with reasons" this changelog series documents, and the bimodal-gait hypothesis
above is left as the concrete starting point for any future v5 attempt.
