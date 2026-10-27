# Gait motion quality -- a fine-tune ladder on X08:c6 (omni task)

**Status: specified, NOT launched (2026-09-23).** The driver code and this directory are step 4 of the
approved plan `/home/nickmagus/.claude/plans/there-is-a-lot-compressed-unicorn.md` ("New ladder
campaign"). Nothing has trained or been evaluated under this charter. The launch waits on the
pre-launch checklist below. Four reward weights are still placeholders, and the validator refuses the
file by name until the step-3 replay gate fills them.

## The question

The user reports a lot of extra motion in the gait across all modes. The visible symptom is stepping
in place at stand: stand tripod is 0.587, the same as the forward hold's 0.576, so the legs cycle
whether or not motion is needed. The goal is a quieter, more economical gait in every mode: fewer
wasted swings, less chatter, and feet that stay planted when they don't need to move.

**The ladder asks one thing per screen: does this lever make the incumbent quieter (block Q
BETTER) without costing facing, translation, speed, slip or survival?**

The plan's context section and the lit review (`../lit-review-gait-motion-quality.md`) give five
causes, all confirmed in the code and the checkpoint:
1. Exploration noise is saturated. The raw std sits at 2.0001-2.0004 on all 18 dimensions, at the
   2.0 clamp.
2. Motion regularisation is weak.
3. The reward economy pays for leg motion that doesn't move the body.
4. The action path has no low-pass filter.
5. Stand is not its own regime.

## The incumbent and the fork

- The head is **X08:c6**: C1d s3, `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_11-58-07_seed3_X08_c6_A15pB/model_7996.pt`,
  recipe `fa03ac033e7c` of the facing ladder.
- Every experiment forks it for **2000 iterations** on its own C1d c6 overlay, preset `1a`. The
  overlay is copied verbatim from `../2026-09-19_facing_tracking_ladder/candidates.yaml`,
  `curriculum_turnfirst`, stage `c6`.
- The fork is the new driver resume kind `resume_from: {checkpoint: <path>}`:
  - the path is hashed into every `recipe_id`;
  - every seed of a recipe forks the same file;
  - a missing file halts `resume_head_missing` before any GPU is booked.
- The resume restarts PPO's counters (memory `ppo-counter-reset-on-resume`). Every screen, X0
  included, takes the same reset.
- **X0** is the baseline: the fork with no knob. Its s3 screen is the control until the first
  ADOPT. Its s3/s2 pair gives D00bs, the seed term: the s2 is a second training run, not an
  eval-only repeat.
- After an ADOPT, the ladder's compose-on-adopt rule makes the incumbent recipe the base and the
  control, as in the facing ladder.

## Screens (rank order = `candidates.yaml`)

| id | lever | knobs | eval-carried |
|---|---|---|---|
| M1 | exploration (cause 1) | `KRABBY_ACTION_STD_RESET=0.4`, `KRABBY_ACTION_STD_MAX=0.5` | -- |
| M2 | energy (cause 2) | `KRABBY_POWER_W` = **TBD** | -- |
| M3 | smoothness (cause 2) | `KRABBY_ACTION_SMOOTH2_W` = **TBD**, `KRABBY_ACTION_RATE_SQ=1`, `KRABBY_JOINT_VEL_W` = **TBD** | -- |
| M4 | reward economy (cause 3) | `KRABBY_FOOT_SLIP_W` = **TBD**, `KRABBY_AIRTIME_W` = **TBD** (de-weight of the +0.8 stock) | -- |
| M5 | command / clock (cause 3) | `KRABBY_OMNI_RELAY_HYST=1`, `KRABBY_OMNI_STAND_HOLD=1` | both |
| M6 | stand regime (cause 5) | `KRABBY_OMNI_STAND_PARK=1`, `KRABBY_STAND_POSE_W` = **TBD** | STAND_PARK |
| M7 | action path (cause 4) | `KRABBY_ACTION_LPF_HZ=4` (unobserved filter; X08:c6 loads) | LPF_HZ |

- **TBD** = `TBD_replay_gate`: the step-3 offline reward-replay gate sets these weights.
- M1's values are the orchestrator's. The step-2 smoke measured that a std reset EQUAL to the clamp
  drifts straight back above it, where the clamp's gradient is zero again and the std is pinned at
  the ceiling. So the reset (0.4) sits below the clamp (0.5). The plan's table read 0.5 / 0.5.
- M7 always parks for the user (`user_decision`): it runs only if jitter survives M1-M3.
- Composition follows the ladder's compose-on-adopt rule, in rank order.

## The decision rule (frozen into D00s)

Rule v2, per category, all seven categories decisive for every fork. The campaign's own
`per_category` template is in the yaml; the shared `RULE_V2_PER_CATEGORY` stays pinned to the
facing ladder.

### Blocks (95 %, two-sided)

- **F**: turn -- `facing_closed`, `yaw_progress`.
- **F_mt**: move_turn -- `facing_closed`, `yaw_progress`.
- **T**: fwd, side, back, diag -- `ratio`.
- **Q (motion quality)**, 42 members:
  - **stand**: every stand-applicable quality metric (12 members) -- `action_jitter`,
    `action_fft_freq`, `joint_speed_mean`, `rod_speed_mean`, `contact_switch_hz`, `foot_slip`,
    `wz_active_frac`, `clock_restarts`, `six_down_frac`, `settle_six_down_s`, `foot_disp_end`,
    `work_per_s`.
  - **fwd / side / back / diag**: `action_jitter`, `swings_per_m`, `slip_per_m`, `work_per_m`,
    `cam_work_per_m`.
  - **turn / move_turn**: `action_jitter`, `swings_per_rad`, `slip_per_rad`, `work_per_rad`,
    `cam_work_per_rad`.

  Cam work is its own member so dither cannot just move from the rods to the cams. There are no
  `*_abs` blocks: no screen moves the command cap.

### Sign

Every Q member is **lower-is-better** except `six_down_frac`
(`eval_categories.metric_sign` / `LOWER_IS_BETTER_KEYS`).

- The sign is applied wherever a delta is formed:
  - the driver's `metric_rows` / `guard_rows`;
  - the resolution's bootstrap and its null.
- So `delta = sign x (candidate - control)`, and z > R reads BETTER in every block.
- Every pre-Q metric keeps +1, so no recorded number of an earlier campaign moves.
- The REPORT marks a signed row with "lower is better, signed delta".

### Guards (99 %, one-sided)

- **Survival** of every category. It is floored at `R_binom`, as before.
- **move_turn.ratio**.
- **The speed guard `cat.<c>.ratio`** for fwd, side, back, diag. D12 let side 0.521 -> 0.430
  through T as a TIE, so "shuffles slower" must not pass as "quieter".
- **The slip guard `cat.<c>.slip_per_m`** for fwd, side, back, diag. A gait that swings less but
  drags its feet must not win the pooled Q.

The speed and slip guards are also T / Q members. They are resolved under their own key
(`cat.<c>.ratio@guard`), a separate one-sided 99 % spec, and never read the block member's two-sided
half-width.

### WATCH (printed, never scored)

- **tripod** on fwd, side, back, diag, move_turn (user decision below). `cat_tripod_min` is no
  guard.
- The moving categories' **per-second rates**: `joint_speed_mean`, `rod_speed_mean`,
  `contact_switch_hz`, `action_fft_freq`, `wz_active_frac`, `clock_restarts`, `foot_slip`. A
  slower gait lowers them without being quieter.
- `direction_error`, `cross_track_ratio`, `heading_final`, `settle_s`, `stand.drift_speed`.
- Both hands of every paired category.
- Every `per_category.demoted` member, with its reason.

### Adoption (`decision_rule.adopt_requires: Q`)

- **Clear ADOPT**: Q BETTER, nothing WORSE, anchored drift OK, no MISSING member.
- **Clear REJECT**, decided, no pair:
  - any block or guard WORSE;
  - Q TIE (or WORSE) while another block is BETTER -- a win without quieting is not this
    campaign's goal;
  - **all-TIE, Q included** -- no seed-2 pair;
  - a stage floor.
- **Speculative** (pair queued): Q BETTER with a MISSING member, or with an anchored-drift FAIL.
  The pooled decision then again requires **pooled Q BETTER**. Otherwise it is a clear REJECT, a
  driver self-overrule when the s3 pick was ADOPT.
- A MISSING guard is a decided REJECT (uncertain, no pair), unchanged.
- Campaigns without the flag behave exactly as before.

### Stage floor

- The floor is `category_survival` with an explicit `min_by_category` for all seven categories.
  A fork has no previous stage, so `min_new` would read every category as new at a formation floor
  of 0.10.
- Each minimum is X08:c6's own suite survival minus **0.20**:

  | category | X08:c6 survival | minimum |
  |---|---|---|
  | stand | 0.990 | 0.79 |
  | fwd | 0.920 | 0.72 |
  | side | 0.975 | 0.77 |
  | back | 0.920 | 0.72 |
  | diag | 0.790 | 0.59 |
  | turn | 0.950 | 0.75 |
  | move_turn | 0.750 | 0.55 |

- Each is band-tested one-sided at 99 % on n_env 100: a collapse detector that trips near a 0.3
  drop (for example after a bad σ reset). Smaller losses are the survival guards' business.
- A FLOOR on a candidate is a clear REJECT; on X0 or a seed-2 run it is WATCH.

### Resolution

- **D00s** is computed by the migration before any training item runs, from two EXISTING heads
  re-evaluated at the current eval_sha:
  - `X08-c6`, the control (`control_head`);
  - `X08-c4`, `.../2026-09-22_10-32-02_seed3_X08_c4_A15pB/model_4498.pt`, on the same C1d chain.
  One cross-head pair, so R is that pair's half-width.
- **D00bs** comes from X0 s3/s2.
- There are no `seed_pair` rows.
- Every Q member and every guard key needs a finite R > 0 (D00s acceptance).

## Decisions recorded before launch

1. **Tripod is WATCH only** (user, 2026-09-22). `cat_tripod_min` is printed, never a guard.
   - Why: D12 rejected C1d itself on it (0.463 -> 0.396), and quieting may change swing timing.
   - Implemented by dropping the composite and adding `tripod` to the fwd/side/back/diag watch
     members. Dropping the composite alone would have removed tripod from the lines entirely.
2. **The step-6 confirm is judged by hand** (user, 2026-09-22), outside the ladder rule, against
   X08:c6 re-evaluated under the new eval_sha.
   - It runs outside the driver: a plain chain launch plus suite evals, so no driver DECISION
     competes with the user's.
   - The seed term comes by hand from the confirm and its seed-2 pair.
   - The table is written in the ladder's standardised form, with X0's D00s giving R_eval.
3. **The adoption rule is ADOPT requires Q BETTER** (plan step 4, approved), including the pooled
   decision.
4. **All-TIE (Q included) is a clear REJECT with no seed-2 pair** (orchestrator, 2026-09-23,
   resolving the plan's "left for implementation: all-TIE and pooled handling"). A pair could only
   turn an unresolved quieting into a marginal one; the budget goes to the next lever.
5. **M1 is `STD_RESET 0.4` + `STD_MAX 0.5`**, not 0.5 / 0.5 (orchestrator, from the step-2 smoke).
   A reset equal to the clamp drifts back above it.
6. **Branch after M1: stop and ask** (plan step 4). If M1 does not ADOPT, the driver stops before
   M2-M4 run at σ 2.0, where regularisers bite weakly.
   - Implemented with the driver's new conditional gate: M2, M3 and M4 carry
     `depends_on: [M1]` and `await_if: decision(M1).chosen != ADOPT`.
   - On M1 ADOPT they dispatch as usual. Otherwise the driver parks at `await_user:M2`, and the user
     answers `ledger_cli decide M2 --choose run|skip`.
   - M3 and M4 park the same way. Answering all three at the first stop passes them in one go.
7. **Zero-spread Q members are demoted, never `--accept-resolution`** (plan step 4).
   - Tool: `run_candidates.py --campaign-dir <this dir> --q-spread-check <X08-c6 stamp> [<X08-c4 stamp>] --q-spread-heads X08-c6,X08-c4`.
     It is CPU-only and writes nothing.
   - It prints every member with no finite positive spread, and the exact
     `decision_rule.per_category.demoted` lines to paste.
   - A member with NO value on a stamp (a pre-step-1 eval) is reported as absent (re-evaluate), never
     as a demotion.
8. **Promotions are disabled** (`promotion: {enabled: false}`, `milestones: []`). Recipes compose
   empty 2a/2b overlays, and the validator refuses any milestone.

## Pre-launch checklist (in order)

1. **Commit the eval-side changes before the step-1 re-eval.** `eval_categories.py` (the metric
   sign, the guard key, the demotion) is inside `EVAL_PATHSPECS`, so it moves `eval_sha`. The
   baseline and the replay npz must be recorded under the final `eval_sha` (plan step 1,
   "Ordering").
2. **Step 1**: re-evaluate X08:c6 under the new eval_sha (`systemd-run --user --scope -p
   MemoryMax=45G`). This gives the baseline table and the stamp for the spread check.
3. **Step 3**: the offline replay gate fills every `TBD_replay_gate` weight. Its report goes in this
   directory.
4. **Spread check** on the X08:c6 stamp. Demote any zero-spread Q member under
   `decision_rule.per_category.demoted`.
   - The check on X08:c6 alone covers the plan's rule.
   - The cross-head half of D00s (X08:c6 vs X08:c4) is first measured by the launch's own migration.
     If its acceptance still fails on a Q member, run the check on the two recorded reeval stamps
     (`reeval_X08-c6`, `reeval_X08-c4` under this campaign's eval root), demote, and relaunch with
     `--accept-candidates-change`.
5. **Validate and dry-run**: `run_candidates.py --campaign-dir <this dir> --dry-run`. The validator
   must be clean.
6. **Launch**: `launch_candidates.sh`, with `setsid nohup`, a MemoryMax cap and a `notify.log`
   Monitor. Push-notify at every decision, stop and wait for the user on each DECISION, and push
   nothing.

## Caveats

- **The speed guard's level.** A 99 % one-sided guard is looser per metric than D12's drop was
  large. On D12's numbers (side 0.521 -> 0.430, R_eval 0.014, R_seed 0.058) it trips on the seed
  term. A smaller slowdown inside max(R_eval, R_seed) passes as TIE. The T block still sees the
  mean.
- **M5 and M6 will likely TIE on Q outside stand.** They act on a small share of the steps. Under
  `adopt_requires` a Q TIE is a clear REJECT, so their stand-only gain must move the pooled Q mean
  (stand is 12 of Q's 42 members) to be adopted.
- **Q's R rests on one cross-head pair** (X08-c6 vs X08-c4) plus X0's seed term. A degenerate pair
  on a member (one side constant) under-resolves it. That is what the spread check and the
  seed-term max are for.
- **The reeval annotation.** The migration's reeval of X08-c6 / X08-c4 records the preset's trained
  categories (`1a`: fwd, stand) as the eval's `--trained-categories` annotation. The annotation is
  informational; the suite scores all seven categories, and the eval-carried knobs are the same as
  under the c6 overlay (none of its keys is carried).
- **No D00.** This campaign has no pre-suite `resolution.json`, so D00s carries no light/ring
  `by_row` copy. None is needed: there is no milestone.
- **The recipe identity includes the fork path.** Renaming or moving the checkpoint file changes
  every recipe_id, as it should: a different path is a different claim about what was forked.
- **Step 6 confounds two changes.** In the from-scratch confirm, σ-from-scratch and `STAND_OBS`
  (plus `ACTION_LPF_OBS` if M7 adopts) enter together. A failed confirm can't be attributed to
  either one, and the fine-tuned winner stays the fallback head. The σ initialisation of step 6
  follows M1's verdict.
- **The stochastic-rollout diagnostic** (plan step 1, `--sample-actions`) feeds no decision.

### From the code review (fable, 2026-09-23)

The review found nothing that blocks launch. Two fixes were applied before commit:

- **One-shot std reset.** `action_std_reset` now fires only when the loaded std exceeds the ceiling. The reset is
  therefore one-shot, and a later stage or crash resume keeps its learned σ.
- **Unmeasured `adopt_requires` block.** A required block with no measured member reads MISSING, and the REJECT is
  `uncertain`, not `clear`.

**M1 means "σ pinned at 0.5", not "σ free below 0.5".**
- The clamp has zero gradient at its ceiling, and the step-2 smoke saw σ drift up from a reset.
- So the reset of 0.4 is expected to climb to 0.5 within a short window and stay there.
- Read M1's verdict as "σ = 0.5 vs σ = 2.0".
- Whether the policy would choose a lower σ is not tested. That would need a gradient-passing clamp or a lower entropy
  coefficient.

**`cam_work_per_m`, `cam_work_per_rad` and `work_per_s` include the `_Body_Hip_` linkage-emulator joint.** Its τq̇ is
the emulator's holding torque, a sim artefact. It is consistent across candidates, so Q deltas are valid, but the
absolute values are not "cam motor power".

**`rod_speed_mean` samples the last physics substep of each env step.** It is aliased rather than averaged over the
decimation. It is a relative WATCH metric only.

**The obs-widening knobs (`STAND_OBS`, `ACTION_LPF_OBS`; step 6 only) shift the positional slices under the hard-coded
`num_prop = 75`.**
- Train and play stay consistent for a from-scratch net.
- Expect the eval's `obs_dim_warning` to fire.

**Under `RELAY_HYST`, turns release at 0.05 rad instead of the 0.133-rad dead band.** Net yaw grows slightly, so M5's
`*_per_rad` deltas are not purely "quieter". This only matters if M5 lands within R of TIE on turn.

## Budget (plan step 5)

| item | cost |
|---|---|
| X0 + X0 seed 2 | ~2 h |
| 7 screens | ~7 x (0.83 h + suite eval) ~ 9 h |
| speculative seed-2 pairs | up to ~1.2 h each |
| migration re-evals + D00s | ~1 h |
| **steps 1-5** | **~12-16 GPU-h** |
| step 6 (confirm + seed-2 pair) | ~8 GPU-h, outside the driver |

## Records

The driver writes `ledger.json`, `state.json`, `heads.json`, `REPORT.md`, `CHANGELOG.md`,
`notify.log` and `orchestrator.log` beside this file. `.gitignore` keeps the bulk out: logs, raw
data and checkpoints.
