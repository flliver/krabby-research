<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-08_0950_stride_length_on/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-08_0950_stride_length_on/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`. Older records may call this directory `sim_fine_tuning/stride_length_on`.

# Changelog: `sim_fine_tuning/2026-08-08_0950_stride_length_on`

**Milestone 18, Task 1 item**: §2.2 "Stride length" -- "New reward on per-step footpad
displacement at touchdown so fewer/longer steps beat shuffling."

**Predecessor**: `sim_fine_tuning/2026-08-07_2303_motor_reversal_on` (this run keeps
`penalty_motor_direction_reversal=-0.3` active and adds the stride-length term on top of it).

## Change (v2 design -- see Verdict; superseded by `stride_length_v3`)

New reward term `reward_stride_length` (`crab_hex_stride_reward.py`,
`RewardStrideLength` in `parkour_isaaclab/envs/mdp/rewards.py`), weight `0.5`, added to both
`CrabHexRewardsCfg` and `CrabHexFlatWalkRewardsCfg`. **v2 design**: rewards per-leg hip-yaw
(`*_Body_Hip_RevoluteJoint`) displacement across *every* touchdown<->liftoff transition (both
swing and stance phases), convex (`power=2.0`) so one long stride outscores several short ones
covering the same net range -- the mechanism this session used to close the "many small hops
score the same as one big hop" degeneracy the user flagged during design review.

Full run: flat (20000 iter, from scratch) -> bridge -> 2b1 -> 2b2 (4 batches, 2000 iter, matching
the other runs' budget), all on this v2 code.

## Metric delta vs motor_reversal_on

**Training**:

| | motor_reversal_on | this run (v2) |
|---|---|---|
| flat crab_failure | 0.78% | 2.86% |
| bridge crab_failure | 27.6% | **52.4%** (escalating pattern: 6.3% -> 27.6% -> 52.4% across the series) |
| 2b1 crab_failure | 14.3% | 51.2% (did not recover, unlike the prior run at this stage) |
| 2b2 gates cleared (best) | 4/5 | **3/5** (needed all 4 batches vs 3 for the other runs, still worse) |
| 2b2 crab_failure (best, `model_21300.pt`) | 7.8% | 24.2% |
| 2b2 forward_progress (best) | 0.126 | 0.080 |

**Gait-eval** (Task 0 harness):

| | flat: motor_reversal_on / this run | 2b2: motor_reversal_on / this run |
|---|---|---|
| schedule_completion_rate | 100% / 80% (2 falls) | 100% (0 falls) / **60% (4 falls)** |
| tripod_score (median) | 0.0 / 0.0 | 0.0 / 0.0012 |
| tippy_tap_fraction | 31.0% / **36.6%** | 22.2% / **24.5%** (worst of all three runs at both stages) |
| slip_ratio | 2.5% / 2.5% | 5.9% / 4.7% (best of the three at 2b2) |
| stride length (pooled mean magnitude) | 0.180 m / 0.183 m | 0.171 m / 0.156 m (worse than motor_reversal_on alone at 2b2) |

**The reward's own per-transition behavior is real and not the problem.** Direct inspection of raw
env data (`foot_force_norm`, `foot_pos_w`, `foot_lin_vel_w`) for MR's short air-time events showed
contact force cleanly zero throughout, foot horizontal displacement up to 195mm and vertical lift
up to 65mm within a single 20ms physics step -- implied foot speed up to 9.9 m/s. This is a real,
fast motion, **not sensor noise**: the leg genuinely covers that distance. But nothing in the v2
reward priced in *how fast* -- a "snap" through the joint's whole range in one step banks the same
convex reward as a slow, deliberate stride. FL/RL legs, by contrast, showed zero short-blip events
and clean 0.35s median swing durations -- the reward worked as intended there. The exploit is
concentrated on specific legs (MR worst, 57% single-step events), not uniform.

**Task 1 §1d (tripod score)**: still near-zero (0.0-0.0012) across every stage and speed hold in
this series, this run included -- see `motor_reversal_on`'s changelog for why (front/middle/rear
duty split, not left/right tripod, an established and consistent finding, not something this
change moved either direction).

## Verdict

**Reverted (v2 design).** Per Task 1 §3's discipline ("revert changes that do not move their
target metric"): despite some genuinely good per-leg behavior, the net effect across the full
curriculum was negative on every headline metric -- worst stability of the series (4 falls / 10 at
2b2, an escalating bridge-transition failure rate that never fully recovered even after 2000
iterations), and the highest tippy-tap fraction at every stage tested. The swing-phase reward
component is judged the root cause (a foot in the air cannot push the robot, so rewarding its
joint-space movement regardless of outcome was mis-specified from the start, not just
under-tuned). Superseded by `stride_length_v3`, which redefines the term to reward only
stance-phase body progress along the commanded direction -- see that run's changelog once
training completes.

This run is kept in `sim_fine_tuning/` (not deleted) specifically as the "reverted attempt with
reasons" Task 1 §Outputs asks the changelog to document.
