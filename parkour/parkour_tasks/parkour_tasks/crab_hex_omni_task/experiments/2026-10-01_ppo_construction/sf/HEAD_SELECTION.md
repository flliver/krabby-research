# Head-of-record selection: side-first B lineage (2026-10-06)

## Decision

**Head of record: SF__B_rest_s18 (seed 18), c6x head `model_14995.pt`.**

Checkpoint: `/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-06_02-24-21_seed18_X60_c6x_A15pB/model_14995.pt`

It replaces W (`crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt`).

Runner-up: SF__B_rest_s16. Third: SF__B_rest_s17.

The choice of the head is unanimous: all three judges ranked s18 first. They split only on second place, and that split does not affect the decision.

## Selection rule (stated before judging)

1. **Floors.** Every candidate must have both side hands FAST at the head, hold falls <= 0.15, dose OK, the clock hidden on every eval stamp, and no NA in a gated read.
2. **Whole task.** Among the heads that pass, pick the best across the whole omni task: fwd, back, diag, side ratio, turn facing_closed, turn yaw_progress, move_turn ratio and move_turn survival. No single metric decides.
3. **Tie-breakers.** Motion quality (action jitter, executed action rate), then the turn band (in-band and 1.0 rad).

## Step 1: floors

All three B heads pass every floor (head_evidence.md sections 1, 2 and 9).

| floor | s16 | s17 | s18 | W |
|---|---|---|---|---|
| side hands at the c6x head (and at m13000, c6) | FAST/FAST | FAST/FAST | FAST/FAST | SLOW/SLOW |
| lr feet_ss l / r (m/s) at 0.22 m/s | 0.215 / 0.220 | 0.228 / 0.228 | 0.228 / 0.233 | 0.096 / 0.098 |
| lr feet_ss l / r (m/s) at 0.25 m/s | 0.243 / 0.227 | 0.244 / 0.245 | 0.252 / 0.256 | 0.102 / 0.102 |
| hold-fall share | 0.000 | 0.000 | 0.000 | 0.000 |
| dose mu_med cam / rod (<= 1.5) | 0.654 / 1.034 | 0.611 / 0.971 | 0.598 / 0.981 | NA |
| clock hidden on every c6x stamp | yes | yes | yes | yes |
| NA in a gated read | none | none | none | dose NA |

- The lr floor values are `labels.lat_{l,r}_{022,025}.feet_ss` from `outputs/omni_plan/ppo_construction/arms_p1/SF__B_rest_s1{6,7,8}_lr.json` and `W_p1b6_lr.json`, per commanded speed. (An earlier version of this table showed one unlabeled row per head, the mean of the 0.22 and 0.25 labels, e.g. s18 0.240 / 0.245 = (0.228+0.252)/2, (0.233+0.256)/2; review finding 4, 2026-10-06.)
- W fails the side floor and has no dose read, so W cannot be retained over a passing B head.

## Step 2: whole task (the deciding step)

Sources: suite medians in `aggregate.suite.by_category.<cat>.ratio` and turn reads in `scenario_metrics.json` for each of these stamps. I re-read them for this note:

- s16: `logs/rsl_rl/gait_eval/omni_phases/2026-10-01_ppo_construction/seed16_X58_c6x_A15pB/suite__omni/seed001/2026-10-06_01-25-00`
- s17: `.../seed17_X59_c6x_A15pB/suite__omni/seed001/2026-10-06_04-40-41`
- s18: `.../seed18_X60_c6x_A15pB/suite__omni/seed001/2026-10-06_07-56-23`
- W: `outputs/omni_plan/ppo_construction/lr/W_p1b6/suite__omni/seed001/2026-10-05_13-22-10`

move_turn survival is the mean of the two holds (L / R): s16 0.48 / 1.00, s17 1.00 / 0.97, s18 0.90 / 0.99, W 0.35 / 0.38.

Each B cell shows the value, then that seed's rank on the row.

| row | s16 | s17 | s18 | W | s18 shortfall vs row best |
|---|---|---|---|---|---|
| fwd ratio | 1.011 (3) | 1.029 (2) | **1.041 (1)** | 0.785 | 0 |
| back ratio | 1.025 (2) | 0.969 (3) | **1.030 (1)** | 0.784 | 0 |
| diag ratio | 0.966 (2) | **0.999 (1)** | 0.956 (3) | 0.581 | -0.043 |
| side ratio | 0.976 (3) | 1.033 (2) | **1.045 (1)** | 0.473 | 0 |
| turn(90) facing_closed | 0.763 (2) | 0.714 (3) | **0.797 (1)** | 0.787 | 0 |
| turn(90) yaw_progress | 0.100 (2) | 0.095 (3) | **0.104 (1)** | 0.102 | 0 |
| move_turn ratio | **0.650 (1)** | 0.597 (3) | 0.609 (2) | 0.442 | -0.041 |
| move_turn survival | 0.740 (3) | **0.985 (1)** | 0.945 (2) | 0.365 | -0.040 |
| **rank-sum (lower is better)** | **18** | **18** | **12** | | |

**Rank-sum.** s18 scores 12; s16 and s17 both score 18.

**Head-to-head over the 8 rows:**
- s18 beats s16 6-2. s16 is ahead only on diag and move_turn ratio.
- s18 beats s17 6-2. s17 is ahead only on diag and move_turn survival.
- s16 and s17 split 4-4.

**Worst shortfall of each seed against the best value on a row:**
- s18: -0.043 (diag).
- s16: -0.245 (move_turn survival). This comes from its left move_turn hold: survival 0.48, 52 pitch-forward falls in 1000 episodes, schedule completion 0.948.
- s17: -0.083 (turn facing_closed). Its left hand reaches only 0.598. It is the only seed below W on a gated row by more than 0.05, so p1_read marks it "not a candidate" (head_evidence.md section 5).

**s18 against W.** s18 is the only seed at or above W on every gated row: turn facing_closed +0.010 and yaw_progress +0.002. s16 is 0.024 below W on turn facing_closed. s17 is 0.074 below W there and 0.007 below on yaw_progress.

## Step 3: tie-breakers (second place only)

The tie-breakers do not decide the head. They are used only to separate s16 from s17, who tie in step 2 (rank-sum 18 each, head-to-head 4-4). Both tie-breakers favour s16, so the motion-quality tie-break decides second place.

**Motion quality** (s16 / s17):
- Mean suite action_jitter over 7 categories: 0.206 / 0.214. This is my mean of the section 4 medians.
- Side action_jitter: 0.158 / 0.186.
- Training exec_action_rate (last 200 its): 2.982 / 3.048.

**Turn band at c6x** (s16 / s17):
- In-band heading_final / facing_closed: 0.036 / 0.928 against 0.076 / 0.849.
- 1.0 rad: 0.059 / 0.942 against 0.088 / 0.912.

**s18 on the same tie-breakers.** Mean jitter is 0.206 (equal to s16), side jitter 0.160, and exec_action_rate 3.040. Turn band in-band is 0.045 / 0.910 and 1.0 rad is 0.084 / 0.915. If the tie-breakers were applied between s18 and s16, they would lean slightly to s16. They are not reached, because step 2 already separates the two.

## Per-judge rankings

| judge | ranking | lens | how second place was settled |
|---|---|---|---|
| 1 | s18 > s16 > s17 | rank-sum over the 8 rows | s16 and s17 level on rows; turn band + jitter favour s16 |
| 2 | s18 > s17 > s16 | robustness and hardware | weighted s16's left move_turn hole (52 falls) heavily, and ranked it last |
| 3 | s18 > s16 > s17 | motion quality and wear | 4-4 row split; jitter, exec rate and turn band favour s16 |

**Resolving the disagreement.** The judges disagree only on second place. Judge 2's order makes one metric (falls) decisive, and the rule forbids that. Under the rule, s16 and s17 tie on the whole-task step, and the stated tie-breakers put s16 second.

Judge 2's concern is still valid and goes into the bake record as a risk note: s16 has the largest single defect in the B set. If a fall-weighted (hardware) ordering is ever preferred, s17 would come second.

## Margin

**s18 over s16 and s17: modest, not a near-tie.**

- The per-row gaps are small, mostly 0.005 to 0.04 on the ratios.
- With one 1000-episode suite per head, any single row gap of a few hundredths could be eval resolution. Same-seed reruns replay bit-identically, so they cannot add samples.
- The selection does not depend on any one of those gaps. s18 leads on 5 of 8 rows, is second on 2 and last on 1. Its worst shortfall on any row is 0.043.
- Each rival has one material defect (shortfall >= 0.08 or a failed p1_read flag): s16 on left move_turn survival, s17 on 90-degree turning.

**Second place (s16 over s17): a near-tie.** The rank-sum is 18 against 18 and the head-to-head is 4-4. The motion-quality tie-break decides it, and the turn band agrees.

## What the runners-up do better than s18

**s16:**
- move_turn ratio 0.650 against 0.609.
- diag 0.966 against 0.956.
- Suite tripod 0.711 against 0.685.
- Best turn band: in-band 0.036 / 0.928 and 1.0 rad 0.059 / 0.942, against 0.045 / 0.910 and 0.084 / 0.915.
- Slightly lower exec_action_rate: 2.982 against 3.040.
- Highest training track_ratio_mean: 0.841 against 0.831.

**s17:**
- Most robust: 3 falls against 11, schedule completion 0.997 against 0.989, left move_turn survival 1.00 against 0.90.
- Best diag: 0.999 against 0.956.
- Lowest training crab_failure: 0.116 against 0.180.
- Lower pitch_max_abs: 0.075 against 0.099 (W 0.127).

## s18 weak spots to record with the bake

- **Diag.** diag ratio 0.956 is the lowest of the three. diag cross_track is 0.062, against 0.017 for s16 and 0.044 for s17.
- **Left move_turn hold.** Survival is 0.90 (s17 1.00), with 11 suite falls, all pitch_fwd. move_turn cross_track is 0.152, the highest of the three.
- **Training failures.** Training crab_failure is 0.180, against 0.116 for s17.
- **Turning came early; straight walking moved late.** s18's turning was already present before the clock cut: c6 move_turn survival 0.945 and c6 1.0-rad facing_closed 0.880, against 0.230 / 0.510 for s16 and 0.445 / 0.077 for s17. Its straight-walk ratios, however, moved the most across c6 to c6x: fwd +0.143, back +0.178 (section 7). So s18 is the most settled on turning, not on every row.
- **Jitter (recipe-wide).** All B heads run at about 1.5 to 2.5 times W's action jitter in every suite category: s18 0.160 to 0.255 against W 0.109 to 0.127, with s17 at stand reaching 0.308 against 0.122. This comes from B's action_rate weight of 0, so it applies to the recipe, not to the seed choice. It is a hardware note for the bake.

## Bake note: play and eval recipe

A B head must be played and evaluated with all four of these env vars. They are present in every B stamp's run_meta env_vars (head_evidence.md section 10), and the c6x run's `params/agent.yaml` and `params/env.yaml` match.

- `KRABBY_OMNI_CLOCK_HIDE=1`
- `KRABBY_STD_PARAM=soft`
- `KRABBY_INIT_STD=0.8`
- `KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25`

W ran at the preset ellipse 0.35:0.25:0.25 without the std knobs.
