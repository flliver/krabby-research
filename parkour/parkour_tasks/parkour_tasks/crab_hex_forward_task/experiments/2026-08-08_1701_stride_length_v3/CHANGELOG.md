<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-08_1701_stride_length_v3/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-08_1701_stride_length_v3/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`. Older records may call this directory `sim_fine_tuning/stride_length_v3`.

# Changelog: `sim_fine_tuning/2026-08-08_1701_stride_length_v3`

**Milestone 18, Task 1 item**: §2.2 "Stride length" (continued -- redesign of the term introduced
in `stride_length_on`, after that run's data showed a net-negative effect).

**Predecessor**: `sim_fine_tuning/2026-08-08_0950_stride_length_on` (v2 design, reverted).

## Change (v3 design)

Redesigned `reward_stride_length` (`crab_hex_stride_reward.py`, `RewardStrideLength` in
`parkour_isaaclab/envs/mdp/rewards.py`) after v2's per-leg investigation showed a foot (MR)
"snapping" through ~195mm of horizontal travel within a single 20ms physics step -- a real, fast
motion (contact force cleanly zero, not sensor noise) that banked the same reward as a slow,
deliberate stride because the v2 design measured joint-space displacement with no time cost.

**v3 semantics**:
- **Swing-phase reward dropped entirely.** A foot in the air cannot push the robot, so it earns
  nothing -- this structurally eliminates the v2 snap exploit rather than patching around it
  (no separate velocity-cost term needed).
- **Stance-phase reward now measures real body progress**, not joint angle: while a leg is
  planted, integrates `root_lin_vel_b` (body-frame) projected onto the commanded direction
  (same convention as the existing `reward_forward_progress_along_command`), clipped to >= 0 so
  motion in the wrong direction earns nothing (not a penalty), accumulated over the stance and
  paid out at liftoff.
- `power=2.0` kept (convex in accumulated stance progress -- one long productive stance still
  outscores several short ones covering the same net progress, same anti-tippy-tap rationale as
  v2).
- `min_phase_duration=0.1` kept, now guarding only against a spurious one-step contact reading
  being trusted as a real stance (not against exploit-by-speed, since fast *body* progress during
  a genuine stance is the desired outcome, not a pathology).
- `min_cmd_norm=0.12` added (no defined "desired direction" when the command is ~stopped).

Weight unchanged at `0.5`. `penalty_motor_direction_reversal=-0.3` remains active (unchanged from
`motor_reversal_on`/`stride_length_on`).

Verified before training: 11 new/rewritten unit tests (`tests/unit/test_crab_hex_stride_reward.py`)
covering stance-only reward, swing contributing nothing even at high body velocity, wrong-direction
motion scoring zero not negative, directional projection, one-long-stance-beats-several-short
convexity, duration-gate rejection with no state leakage into the next stance, command-norm gating,
per-leg independence, and batching -- plus two zero-agent smoke runs (teacher and flat-walk
configs) confirming the term resolves and steps cleanly against the real scene.

Full run: flat (20000 iter, from scratch) -> bridge -> 2b1 -> 2b2 (4 batches, 2000 iter, matching
the other runs' budget).

## Metric delta vs stride_length_on (v2)

**Training** -- the standout result of this series:

| | v2 | v3 |
|---|---|---|
| flat crab_failure | 2.86% | 2.49% |
| bridge crab_failure | **52.4%** (escalating, never recovered) | **3.91%** |
| 2b1 crab_failure | 51.2% | **1.56%** |
| 2b2 crab_failure (batch range) | 24-63% | **6.8-14%** |
| 2b2 gates cleared (best single checkpoint) | 3/5 | 4/5 (multiple checkpoints, batches 1-4 all produced 4/5s) |
| 2b2 forward_progress (peak) | 0.080 | **0.1496** (essentially at the 0.15 gate; never fully cleared across any run in this series) |

Bridge and 2b1 -- both fixed, un-gated stages -- show the clearest signal: v2 showed an
escalating, never-recovered instability (6.3% -> 27.6% -> 52.4% -> 51.2% across the whole series
through this point); v3 is dramatically stable at every stage (3.91% -> 1.56%), better than even
`baseline`'s own transition. This is the strongest evidence that the v3 redesign fixed the
underlying problem, not just the gait-eval symptoms.

**No checkpoint across all 4 batches (2000 iterations) cleared all 5 README §4.2b gates
simultaneously** -- `forward_progress_along_command` remains the persistent holdout across every
run in this entire series (baseline, motor_reversal_on, v2, and now v3), never once fully clearing
0.15. But v3 gets closer than any prior run (0.1496 vs v2's 0.126, motor_reversal_on's 0.126,
baseline's 0.121) while *also* clearing the other 4 gates far more reliably and with wider
margins. Accepted `model_21900.pt`.

**Gait-eval** (Task 0 harness):

| | flat: v2 / v3 | 2b2: v2 / v3 |
|---|---|---|
| schedule_completion_rate | 80% (2 falls) / **100% (0 falls)** | 60% (4 falls) / 80% (2 falls) |
| tripod_score (median) | 0.0 / 0.0 | 0.0012 / **0.025** (notably non-zero -- see below) |
| slip_ratio | 2.5% / **1.7%** (best of the series) | 4.7% / 3.0% |
| tippy_tap_fraction | **36.6%** / **19.1%** | 24.5% / 21.0% |
| stride length (pooled mean magnitude) | 0.183 m / 0.178 m | 0.156 m / 0.162 m |

Flat-walk stability fully recovered (0 falls, matching baseline/motor_reversal_on) and
tippy_tap_fraction dropped by nearly half versus v2 (36.6% -> 19.1%), though still above
baseline's 13.5% -- the redesign substantially closed the v2 regression without fully eliminating
it. 2b2 stability (2 falls / 10) is better than v2's 4 but worse than baseline's 1 and
motor_reversal_on's 0 -- somewhat surprising given how much stronger this run's training-time
crab_failure was throughout; the gait-eval command schedule (fixed forward speeds) may be
stressing a different part of the policy's competence than the training terrain distribution does.

**Task 1 §1d (tripod score) -- unexpected new signal.** Every other run in this series (including
this run's own flat-walk stage) scored tripod_score in the 0.0-0.0035 range, consistent with the
established front/middle/rear duty-split gait. This run's 2b2 checkpoint alone scored a median of
0.025 -- 7x higher than the next-highest result in the series (`baseline` 2b2's 0.0035), and rising
with commanded speed (low=0.008, mid=0.023, high=0.031). This is still far from a clean tripod
gait (which would score closer to 0.5-1.0), and no contact-schedule term (§2 item 3) was added, so
this reads as an incidental side effect of rewarding stance-phase body progress specifically,
rather than an intentional result. Not investigated further here -- flagged as a genuine, unplanned
finding worth a follow-up if §1d's tripod-score threshold becomes a priority.

## Verdict

**Kept.** By a wide margin the strongest run in this series: the v2 regression (escalating
training instability, worst gait-eval stability and tippy-tap of any run) is not just fixed but
overcorrected -- v3's bridge/2b1/2b2 crab_failure and gate-clearance reliability beat every other
run, including `baseline` and `motor_reversal_on`. `forward_progress_along_command` still never
clears its gate in this series, but v3 gets closer than any predecessor while being more robust
everywhere else. The 2b2 stability regression relative to `baseline`/`motor_reversal_on` (2 falls
vs 1/0) and the unexplained tripod-score jump are the two open threads worth a closer look before
calling this term finished, but neither outweighs the clear net improvement over v2 or the
underlying design fix (rewarding body progress instead of joint displacement) that produced it.
