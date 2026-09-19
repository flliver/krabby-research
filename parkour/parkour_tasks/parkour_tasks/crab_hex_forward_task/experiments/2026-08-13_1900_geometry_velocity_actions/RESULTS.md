<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-13_1900_geometry_velocity_actions/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_1900_geometry_velocity_actions/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Geometry alignment + camshaft velocity actions — results

See CHANGELOG.md for the full change list and verification/debug history.

## Commit 1 (1b42d8d): measured joint limits — DONE, all gates green
- pytest 19/19; joint drive 18/18 (live soft limits match hardware table)
- statics x3 within basin scatter of perp baseline
- cam mechanism PASS under +-32 Body_Hip (max err 0.6 deg, no limit contact)
- FT canary from symmetric ref: 0.47 -> 27.1 (perp precedent 27.5), gait eval
  completion 1.0 / tripod 0.416 / slip 2.6% — statistically identical to perp canary

## Commit 2: camshaft velocity actions
Mechanism-level: PASS (spin check r3): 2.00 revs at +-3 and +-6 rad/s both directions,
pointwise hip err <= 0.046 rad, max |hip| 0.536 < 0.5585 hard stop, no env resets.
Statics unchanged. Unit tests 100/100.

### FT canary (2000 it from symmetric model_19999): PARTIAL
- reward 0.35 -> 11.4, monotonic climb throughout (1.4 @200 ... 11.3 @1800), no NaN
- gait eval: tripod 0.0, completion 0.5, slip 34% — the position-era gait does NOT
  transfer to velocity cam semantics in 2000 it (expected worst case in the plan)
- NOT a gate failure by itself: trajectory still climbing; mitigation options queued
  (longer FT, or reinit the 6 cam actor-output columns + action-std entries)

### From-scratch control (3000 it): RUNNING
- The MDP-health verdict: a fresh policy learning tripod + sustained one-direction
  shaft spin proves the env; then FT recovery is a schedule question, not a design one.

## Decision points for the user (pending from-scratch verdict)
1. If from-scratch shows gait formation: choose FT mitigation (long FT vs cam-column
   reinit) vs from-scratch 20k adoption path for the velocity-era reference.
2. CAM_VEL_SCALE=6.0, cam effort 5.0/sat 6.0, vel_limit 15, camshaft inertia 0.015,
   hip tracking 2000/40 are all placeholders pending hardware measurement — flagged in
   code comments (crab_hex_scene_cfg.py, parkour_mdp_cfg.py).

### From-scratch control (3000 it): MDP HEALTHY — verdict data
- reward 0.23 -> 21.0 (500: 9.6, 1000: 16.7, 2500: 21.3; symmetric ref hit 27 at 20k),
  no NaN; learns FASTER than the FT canary adapted (9.6@500 vs FT's 11@1800)
- gait eval: completion 0.9, slip 7.9%, tripod 0.0
- raw shaft stats (episode_00): mean |shaft v| = 5.61 rad/s on ALL SIX shafts (~93% of
  CAM_VEL_SCALE) — front/rear idle pathology GONE, every leg thrusts
- BUT one-direction ratio = 0.01 (signed mean ~0): the policy oscillates the shafts at
  ~1.4 Hz full speed instead of continuous rotation. Two candidate causes, both
  reward-side, neither config-side:
  1. penalty_motor_direction_reversal weight is 0.0 in flat-walk (-0.3 only in teacher
     stack) — nothing prices reversals, and position-era habits are not in play here
  2. tripod 0.0 may be partly a scoring artifact: 1.4 Hz sweep = 0.7 s period, outside
     RewardTripodSchedule's 0.10-0.60 s band (interaction flagged in the plan)

## VERDICT: config change verified and committed; training methodology forks to user
Mechanism: PASS. MDP: trainable, all-legs thrust at full speed. Remaining decisions are
reward-shaping/campaign-path, not configuration:
(a) weight penalty_motor_direction_reversal > 0 in flat-walk to elicit one-direction spin
    (offline replay gate on saved npz first, per campaign SOP)
(b) revisit RewardTripodSchedule.max_period for velocity-era gait cadence
(c) velocity-era reference path: long from-scratch (20k) vs FT mitigation (cam-column
    reinit); from-scratch strongly indicated by the canary pair
