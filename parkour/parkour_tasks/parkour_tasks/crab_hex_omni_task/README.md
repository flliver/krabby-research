# Crab hexapod omnidirectional-walk task (`crab_hex_omni_task`)

The **omnidirectional sibling** of [`crab_hex_forward_task`](../crab_hex_forward_task/README.md)
and [`crab_hex_sideways_task`](../crab_hex_sideways_task/README.md): the same robot, the same
plant (A15+B, `assets/crab.usda`), and the same MDP terms, but travel and facing are two
INDEPENDENT world-frame commands instead of one body-frame axis. Command 1 is a **world** travel
direction with a direction-scaled speed cap; command 2 is an **absolute world heading** to face.
Four command-regime slots are sampled: **stand / move / turn / move+turn**. The obstacle course is
a **ring** of randomly placed blocks around the spawn (a clear central disc, then an annulus of
bars and slabs) instead of a corridor. This package is a thin subclass of the forward package's
config classes, exactly like the sideways task: it reuses `crab_hex_forward_task/mdp/*` unchanged
and re-declares only the cfg-valued attributes that need world-frame commands. See
`docs/crab-hex-forward-policy-config.md` for the shared reference and the delta table below for
exactly what differs.

Nothing about the *policy* has changed on purpose beyond what the facing half requires — reward
weights, terrain/curriculum knobs (besides the ring geometry), PPO hyper-parameters, terminations
and the 18-channel action space are the forward recipe (verified by an Isaac config-delta test;
see `tests/integration/test_crab_hex_omni_configs.py`). Teacher-only in this plan — no depth
student (the shared camera-event bug and the +y mount decision stay open).

## Delta from the forward task

| Site | Forward | Omni |
|---|---|---|
| Command | one body-frame axis, heading-P-control steering | `CrabHexOmniCommand`: world travel direction + direction-scaled speed, plus an absolute world facing target; four slots (stand/move/turn/move+turn) |
| Observation slots 5, 8, 9, 10 | `0*delta_yaw`, `0*cmd`, `0*cmd`, `cmd[:,0]` | `cmd wz`, `cmd vx_b`, `cmd vy_b`, `‖cmd_xy‖` (`CrabHexOmniParkourObservations`, width 1149 unchanged) |
| Steering channels (6, 7) / history | forward goal-heading error, zeroed on flat tiles; blanked in history | `wrap(heading_target − heading_w)` on EVERY tile, both channels, **clipped to ±0.6 rad by default** (`facing_obs_clip`, knob `KRABBY_OMNI_FACING_OBS_CLIP`, armed in every preset — see Revision 3 below); kept in history (`ZERO_STEERING_HISTORY = False`) — it is a command the policy must integrate, not a look-ahead to ignore |
| Clock speed | `cmd[:, 0].abs()` | `max(‖cmd_xy‖, r_turn·\|cmd_wz\|)`, `r_turn` (`KRABBY_OMNI_CLOCK_TURN_R`) 1.2 m — an active turn never stalls the clock |
| Five `cmd[:,0]`-only gate sites + two clock-reward sites | the literal `cmd[:,0]` | dispatch through shared `_cmd_gate`/`_clock_cmd_speed` helpers; forward branch = each site's old literal, unchanged **Revision 2 (2026-09-17):** the omni branch of `_cmd_gate` reads the clock speed max(‖cmd_xy‖, r_turn·\|cmd_wz\|), not ‖cmd_xy‖ alone, so a pure turn switches the gait-shaping terms on like translation does — both earlier yaw smokes had stepped in place with them off. |
| Reward twins (same NAMES) | `penalty_lin_vel_y`, `reward_tracking_goal_vel`, `reward_tracking_yaw`, `reward_obstacle_clearance` | `penalty_cross_command_vel_l2` (velocity across the commanded direction, gated `‖cmd‖>0.12`, weight −3.0), `reward_tracking_command_vel_on_obstacle` (world progress along the commanded direction, obstacle-masked), `reward_facing_exp` = `exp(−\|wrap(heading_target−heading_w)\|)` on every tile no gate (`KRABBY_YAW_W` 0.2 **from 1c**, 0 in 1a/1b), `reward_obstacle_clearance_omni` (lift-and-cross bonus read along the commanded direction) |
| Reward ADDED (the 36th name) | — | `reward_yaw_progress` = `clip(wz_b·sign(cmd_wz), ±ang_vel_max)`, gated `cmd_wz ≠ 0`, weight 1.0 (`KRABBY_YAW_PROGRESS_W`, armed in every window) — the rotational twin of `reward_forward_progress_along_command`: dense, SIGNED income for rotating the way the turn was commanded. Added 2026-09-17 after the yaw-capability smoke stood, stepped in place and rotated the **wrong way** (see [The turning fix](#the-turning-fix)) |
| `track_ang_vel_z_exp` std | 0.5 | class body 0.15 (plan D4: gradient at 0.15–0.3 rad/s), but **every preset sets `KRABBY_OMNI_ANG_VEL_STD=0.5` — back to the forward value** (Revision 4, 2026-09-18; the tight kernel taxed gait wobble on the 88 % of steps that command no turn) |
| Termination | `terminate_episode` (course-goal cutoff) | `terminate_episode_no_goals` (roll/pitch/height/timeout only — the ring's sector anchors never advance) |
| Mirror-symmetry | 1 image (L/R only) | **4 images**: `identity, M_y (L/R), M_x (fore-aft), M_x M_y = R_z(π)` — both body mirror planes preserve the task now that any direction is commandable |
| Height-scanner offset | `pos (0.6, 0.0, 20.0)` | `pos (0.0, 0.0, 20.0)` — required to lie on BOTH mirror planes (108 rays, obs 1149 unchanged) |
| RSI bank | forward-only frames | provenance-tagged bank merging BOTH heads of record (forward `source=0`, sideways `source=1`); re-yawed to a sampled facing, world velocity rotated, and the **first command staged** so the frame starts on-command with zero facing error |
| Terrain | corridor (gaps/hurdles/stones) | **ring**: clear central disc, then an annulus of randomly placed bars/slabs by difficulty; 8 sector goal anchors that never advance; both-axes edge mask (`parkour_field_to_mesh_xy`) |
| Parkour term / promotion | downrange distance vs. `cmd[:,0]·T` (dead for lateral/backward) | `OmniParkourEvent`: promotion by **tracking ratio** `ep_track_dist / ep_cmd_dist` (works for every direction); tile-centre spawn, radial/sector exposure |
| Preset variable | `KRABBY_PHASE` | `KRABBY_OMNI_PHASE` |
| Gym ids | `Isaac-Crab-Hex-{Flat-Walk,Teacher,Student}-v0` | `Isaac-Crab-Hex-Omni-{Walk,Teacher}-v0` (+ `-Play-v0`; **no student id** — teacher-only) |
| RSL-RL experiment dirs | `crab_hex_{flat_walk,teacher,student}` | `crab_hex_omni_{flat_walk,teacher}` |
| Command telemetry | `error_vel_xy`, `error_vel_yaw`, `stand_frac_actual` | + `move`/`turn`/`move_turn_frac_actual`, `facing_err_abs`, `recap_frac`, and `yaw_progress` (the SIGNED mean `wz_b·sign(cmd_wz)` over the turning rows — the one number that separates "turns the wrong way" from "does not turn") |
| Eval harness | body-frame holds | same harness, **world-frame holds** (`speed, dir, face`) driven through the SAME `schedule_hook`/P-control training uses; `facing_metrics`, per-step travel-frame rotation |

Everything else — the forward 35 reward NAMES and their weights/ramps (the omni stack is those
35 plus the one added `reward_yaw_progress`), the clock/stride/tripod reward
math, the contact proxy, DR knobs, PPO hyper-parameters, seed and env counts, the scan grid
size/resolution/ordering (108 rays) — is the forward recipe unchanged; `travel_frame.py` and the
pinned seam-site set are untouched (the omni cfg carries no `travel_yaw` — the stock 2-D
`track_lin_vel_xy_exp`, `penalty_tracking_error_l1`, `reward_forward_progress_along_command` and
`RewardStrideLength` are already correct for a 2-D body-frame command).

### The mirror sign tables, as measured (not as first drafted)

`crab_hex_omni_mirror.py`'s `HEAD_SIGNS_LR` (`M_y`, y→−y) is **numerically identical** to the
forward L/R table; `HEAD_SIGNS_FA` (`M_x`, fore-aft) differs from the sideways table at **slot 8
alone** (the body-frame command's x-component, which flips under `x→−x`). This is not an error —
the forward head's slots 5, 8 and 9 were structurally zero (`0*delta_yaw`, `0*cmd`) so their signs
were free and happened to already match what the omni head's live values need; only slot 8 carries
a genuinely different sign because it is now a real number under both tasks. `differ_from_forward_LR
== set()` and `differ_from_side_FA == {8}`, pinned by `tests/unit/test_crab_hex_omni_mirror.py`.

## The turning fix

**Finding that drove this design:** no head of record, forward or sideways, has ever turned. The
only turning evals on disk achieve `|wz| ≈ 0.003 rad/s` against 0.6–1.0 rad/s commanded; an earlier
"turning@10k" gate had passed on a vx proxy with the yaw-rate aggregate never recorded. The
plant's kinematic turn rate is ≈0.15–0.2 rad/s, so the stock `±1 rad/s` yaw command demanded 5× what
the plant can do and left `track_ang_vel_z_exp` (std √0.25) with no usable gradient there.

The fix, all landing together: a yaw-rate **cap** `ang_vel_max` 0.3 rad/s (`KRABBY_OMNI_ANG_VEL_MAX`,
replacing the stock ±1) with a tighter dead band `ang_vel_clip` 0.2 (a facing dead band of 0.133 rad
at stiffness 1.5, vs. the forward task's 0.267 rad); `track_ang_vel_z_exp` std tightened 0.5 → 0.15
(gradient across 0.15–0.3 rad/s — **reverted to 0.5 by Revision 4**, see below); the gait clock driven by turning (`r_turn` 1.2 m,
`KRABBY_OMNI_CLOCK_TURN_R`, so the smallest non-zero yaw command 0.2 gives clock speed 0.24, above
the 0.2 m/s stall floor); and `reward_facing_exp`, a facing reward with no tile mask and no command
gate, armed from stage **1c** (`KRABBY_YAW_W` 0.2 — inert at 0 in 1a/1b, which hold facing and have
no turning slots). Because the whole facing half rests on this fix and has never been validated
on this plant, the campaign runs a **500-iteration yaw-capability smoke** (turn-only slots, from
scratch) before any 5000-iteration window: it must show a median **signed yaw-progress rate**
`mean(wz · sign(cmd_wz)) ≥ 0.05 rad/s` (10× the record; the wobble-blind quantity — a plain `|wz|`
or `yaw_rate_ratio` reads ~0.28 rad/s of gait wobble on a head that does not rotate at all) and
median final heading error `≤ 0.6 rad` on the harness's 90° turn holds, or the mechanism is revised
(the first smoke failed and produced Revision 1 below) before training continues.

### Revision 1 — the yaw-progress reward (2026-09-17)

**The smoke failed.** `KRABBY_OMNI_PHASE=1c KRABBY_OMNI_SLOTS=0:0:1:0
KRABBY_FLAT_TERRAIN_FLAT_FRAC=1.0`, 500 iterations from scratch
(`parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-16_20-35-23`): the policy learned to **stand**
(flat failure 0.50 → 0.12) and to **step in place** under the clock (tripod 0.47 on the turn holds),
but not to **rotate**. On the harness's `turn_l90` hold the yaw-rate command sat saturated at
**+0.30 rad/s for 8 s** while the median net body rotation was **−0.025 rad/s — the wrong sign** —
inside a mean `|wz|` of **0.28 rad/s of pure gait wobble**; training `facing_err_abs` never trended
down (0.27–0.43 throughout).

**Why the existing terms could not fix it.** Both facing terms are even functions of a signed
quantity read through wobble an order of magnitude larger than the drift they price:
`track_ang_vel_z_exp` (std 0.15 at the time; Revision 4 set it back to 0.5) and `reward_facing_exp` (weight 0.2) both pay about the same for
+0.28 and −0.28 rad/s, so one step of wobble swamps their gradient and neither can tell "rotating
towards the target" from "rotating away". And **nothing in the 35 rewarded rotating** the way
`reward_forward_progress_along_command` rewards translating — the task paid for *being* at the
right facing, never for *turning* towards it.

**The revision, chosen by the user: add the rotational twin.**
`reward_yaw_progress` = `clip(wz_b·sign(cmd_wz), ±ang_vel_max)`, gated on `cmd_wz ≠ 0`, weight 1.0
via `KRABBY_YAW_PROGRESS_W` in **every** window. 1a/1b draw no turning slot and hold their facing,
so no *commanded* turn exists to pay for there; the gate opens in those stages only when the
robot's own yaw drifts past the 0.133 rad dead band, and then the term pays for turning back — a
heading hold, aligned with what those stages already want. (It is not unconditionally zero in
1a/1b; `KRABBY_YAW_PROGRESS_W=0` would make it so, at the cost of splitting the dose.)
Three properties are load-bearing:

* **signed**, unlike the linear twin's `clamp(min=0)` — symmetric wobble nets exactly zero, so the
  gradient points at the *net* drift, which is the quantity that has to move;
* **clipped at the command's own `ang_vel_max`** (read off the command term's cfg at call time,
  `clip: None` in the params — so `KRABBY_OMNI_ANG_VEL_MAX` retunes cap and reward together): the
  reward saturates at the rate the policy is actually asked for and cannot be farmed by spinning
  faster than commanded, the rotational form of the wheelie exploit `max_speed_scale` 1.05 closed;
* **gated on `cmd_wz ≠ 0`**, which IS "|facing error| > the P-control's dead band" — no second
  threshold to tune, and no term pushing against the dead band the command just imposed.

`Metrics/base_velocity/yaw_progress` logs the same signed quantity during training (mean over the
rows commanding a turn), so the failure mode that cost this smoke 12 minutes of silence is now a
number on every print. The other two candidates (`ang_vel_max` 0.2, `r_turn` 2.0) are untouched and
still available if the re-smoke needs them.

### Revision 3 — the facing-error observation clip (2026-09-17)

**With Revision 2 live, the re-smoke turned — but the canary still failed.** A 1500-iteration
turn-only head (`crab_hex_omni_flat_walk/2026-09-16_21-38-25`) trained a real signed yaw-progress
rate, +0.086 rad/s and rising. Yet the harness's 90° canary (`turn_l90`/`turn_r90`) still FAILED —
all 16 episodes fell inside the ±π/2-face holds — while an in-band diagnostic at ±0.5 rad settled
every episode cleanly (16/16, progress 0.05–0.11 rad/s, no falls).

**The cause was the observation, not the command.** The yaw-rate P-control already saturates at
`ang_vel_max` beyond the 0.133 rad dead band, so an out-of-band facing target still drives the SAME
command the policy trained under. But the facing-error OBSERVATION (steering channels 6/7,
`wrap(heading_target − heading_w)`) carries the raw, unsaturated error: a head trained on
`|err| ≤ 0.6` rad (the 1c facing band) was handed **1.57 rad** on the 90° canary — 2.6× anything it
had ever seen — and that out-of-distribution input, not a command-saturation problem, is what
destabilised it. The plan's earlier claim that the gate is "band-independent because the
P-control saturates" was right for the COMMAND and wrong for the OBSERVATION.

**The fix: clip the observation at the band edge.** `facing_obs_clip` (knob
`KRABBY_OMNI_FACING_OBS_CLIP`, `None` by default = the raw error) symmetrically clamps steering
channels 6/7 to `±clip` before they reach the policy
(`crab_hex_omni_observations.py::CrabHexOmniParkourObservations._steering_channels`) — beyond the
dead band the P-control needs only the SIGN of the error, not its magnitude, so clipping loses no
information the command can act on. Re-run on the SAME 1500-iteration head with the clip at 0.6:
**PASS on the turning criterion** — signed yaw-progress rate **+0.063 / +0.094 / +0.071 rad/s** on
`turn_l90` / `turn_r90` / `move_turn`, completion 0.81 (3 falls vs. 16, down from 16/16 without the
clip). (`turn_l90`'s final heading error, 0.78 rad, still misses the WATCHED 0.6-rad line — `gate_facing`
is enforced from 2a, not on a 1c-formation head.)

**Adopted 2026-09-17 (user decision):** `KRABBY_OMNI_FACING_OBS_CLIP=0.6` in EVERY omni preset —
0.6 is the 1c facing band's own edge, and it stays there in 2a/2b/2c even though `KRABBY_OMNI_FACING`
itself widens to `0.2:1.2` then `0.2:2.4`: the band ramp only changes how LONG a turn lasts, not
what the policy must be able to see to resolve the dead-band edge, so the observation clip does not
widen with it.

### Revision 4 — the reward economy (2026-09-18)

**The 1c formation gate failed, and the diagnosis was not about turning.** Seed 3 ran
`1a → 1b → 1c` and `gate_1c` recorded completion 0.00 with 100/100 falls. Two separate things were
wrong, and only one of them was the policy
(`experiments/2026-09-16_1805_omni_formation/ANALYSIS_1c_failure_2026-09-18.md`, a 31-agent
workflow whose claims survived three-lens adversarial refutation):

1. **An eval defect (established, fixed in `d0fdd9c`).** The driver's canary env never carried
   `KRABBY_OMNI_FACING_OBS_CLIP`, so the policy was scored on a raw 1.57-rad facing error against a
   ±0.6-rad training support. The same head under the corrected driver completes **0.39**, with
   survival 1.00 on every translation hold. The recorded canary, and the light/ring rows, are void.
2. **A real policy failure the eval fix does not touch: the robot stopped travelling.** Speed
   tracking, measured in TRAINING, fell **0.747 (1a) → 0.433 (1b) → 0.245 (1c)** while mean reward
   *rose* 61 → 93. The 1c head survives every translation hold by stepping in place with a clean
   0.85-tripod rhythm at ~0.04 m/s against 0.25 commanded.

**Why stillness was the optimum.** Of the 1c head's 2.73/s positive income, clock + apex was 34 %,
the linear-velocity exponential 30 %, the angular-velocity exponential 28 %, facing 6.6 % — and the
terms that *require motion* totalled **1.2 %**. A Monte-Carlo of the 1c command sampler predicts
that a **zero-velocity** policy earns `track_lin_vel_xy_exp` 0.835 and L1 −0.182; the logged head
earned 0.826 / −0.177. It had found the standing solution to three digits.

**The fix — four values, uniform in every omni window** (user decision D, 2026-09-18; `OMNI_COMMON`
in `crab_hex_omni_phases.py`). Uniformity is deliberate: the chain lost forward travel at the
`1a → 1b` handoff, and a window whose economy differs from its neighbour's is exactly that handoff.

| knob | was | now | why |
|---|---|---|---|
| `KRABBY_TRACK_SIGMA2` | `0.1` (forward) | `0.04` | `exp(−‖cmd−v‖²/σ²)` pays a stander 0.835 at 0.1 and ~0.21 at 0.04 — standing stops being 99 % of walking's tracking income |
| `KRABBY_TRACK_L1_W` | `−1.0` (forward) | `−2.0` | the one term whose gradient does not vanish at `v = 0`; break-even for a stander at σ² 0.1 alone would need ≈ −2.9 |
| `KRABBY_OMNI_ANG_VEL_STD` | `0.15` (plan D4 class-body literal) | `0.5` (**the forward value**) | at 0.15 ordinary gait yaw wobble on the 88 % of steps that command NO turn costs 0.36/s at 0.10 rad/s and 0.63/s at 0.15 — a tax on walking larger than the whole translation income. The turning signal is carried by `reward_yaw_progress` (Revision 1), which did not exist when 0.15 was chosen |
| `KRABBY_OMNI_CLOCK_COMBINE` | `sum` (forward literal) | `product` | `reward_clock_schedule` stops paying `(swing half + stance half)` — a creeper collects the whole stance half free, ~0.5 ceiling — and pays `swing_quality × stance_quality`, which a foot that never leaves the ground multiplies toward 0 |

**The clock `combine` mechanism is the forward task's own, not new here.**
`crab_hex_clock_reward.clock_schedule_income(..., combine=...)` has taken `"sum"` / `"product"`
since 2026-08-22 (the basin-lottery fix), with its own unit tests; the forward rewards cfg declares
`params["combine"] = "sum"` and `KRABBY_CLOCK_COMBINE` applies it forward-side. Only the omni
*spelling* is new, so **no forward preset moves**. `KRABBY_OMNI_CLOCK_COMBINE` is applied by
`apply_omni_knobs` — after the forward `__post_init__` — so it wins if both are set. Note what
`"product"` does and does not do: it prices gait QUALITY (a foot that never swings earns nothing),
not travel. Making standing unprofitable is the tracking pair's job.

**Run shape (decision D).** A **from-scratch** 1c window of 5000–8000 iterations on the 1c preset
with these terms (`run_phases.py --phases 1c --from-scratch`), then 2a/2b resumed from its head.
Two safety nets land with it, both reading the TRAINING log rather than an eval:

* every gate (soft 1a/1b and `gate_1c`) additionally requires
  `Metrics/base_parkour/track_ratio_mean ≥ 0.6` averaged over the last 500 iterations — the seed-3
  numbers are 1a **0.693 PASS**, 1b **0.406 FAIL**, 1c **0.246 FAIL**, i.e. this floor alone would
  have halted the chain after 1b;
* a from-scratch window longer than 1000 iterations is aborted at iteration 1000 (`dead_walker`,
  `--continue`-overridable) if its its-800-1000 mean is below 0.5 — 1a formed a walker from scratch
  at 0.747 in exactly that budget.

**Status: the three tuned values are PROVISIONAL.** They are to be confirmed by an offline replay
on the saved 1a/1c raws (does the walker out-earn the creeper under these terms?) before the window
is launched. The plumbing is landed either way; the numbers may move.

#### Lever staged 2026-09-18: `KRABBY_OMNI_ANG_TURN_ONLY` (DEFAULT OFF)

The offline replay that was supposed to confirm the four values found that they do not clear the
campaign's `>= 25 %` walker-vs-creeper margin on their own (`fwd` hold, COMMAND margin **+16.6 %**;
`experiments/2026-09-16_1805_omni_formation/replay_reward_economy.md`), and it named the largest
remaining leak: `track_ang_vel_z_exp` is an *even* function of the yaw-rate **error**, so a
motionless robot with zero yaw error collects ~`0.98`/s of it — and with `std` back at the forward
`0.5`, on the ~88 % of 1c rows that command no turn, that free income is ~45 % of a stander's entire
positive income. Gating the term to rows with a **live** yaw command (`cmd[:, 2] != 0` — i.e. the
P-control outside its 0.133-rad dead band, the same gate `reward_yaw_progress` uses) takes the
stander's angular income to ~0 and moves the margin to **+37.5 %** (gate FAIL → PASS), with the
creeper's motion-requiring income share 2.5 % → 4.5 %. Restate it with
`replay_reward_economy.py --new-ang-turn-only`.

It ships **off**, as a lever rather than as a fifth Revision-4 value, because the gate also removes
the term's yaw-rate **damping** on non-turn rows: today a row with `cmd_wz == 0` pays the policy to
hold `wz_b` near zero — to walk straight and not wobble — and that pressure disappears entirely
under the gate, leaving `reward_facing_exp` (a position term, not a rate term) and the command's own
P-control to price heading there. That is a behavioural change, not a bookkeeping fix, so arming it
is a per-run decision:

```bash
KRABBY_OMNI_PHASE=1c KRABBY_PLANT=A15+B KRABBY_OMNI_ANG_TURN_ONLY=1 \
  python scripts/rsl_rl/train.py --task Isaac-Crab-Hex-Omni-Walk-v0 --headless ...
```

> **Arming it under `run_phases.py` takes an edit, not an export.** The driver's `_subenv`
> deletes **every** `KRABBY_*` key from the parent environment before launching a phase
> (`run_phases.py:519-521` — the same scrubbing that hid the facing-clip defect), so
> `export KRABBY_OMNI_ANG_TURN_ONLY=1; ./launch_phases.sh ...` is silently dropped and the run
> trains the shipped economy while the launcher believes the lever is on. Under the driver the
> knob has to reach the window's `env` dict. Verify either way from the run's own
> `params/env.yaml`: `rewards.track_ang_vel_z_exp.func` must read
> `...crab_hex_omni_rewards:track_ang_vel_z_exp_turn_gated`, not
> `isaaclab.envs.mdp.rewards:track_ang_vel_z_exp`.

Landed: `mdp/crab_hex_omni_rewards.py::track_ang_vel_z_exp_turn_gated` (a drop-in twin with the
stock argument order that **calls** the stock function and multiplies by the mask — it never
re-derives the kernel, and `std` passes straight through, so `KRABBY_OMNI_ANG_VEL_STD` still owns
the width), the knob in `config/crab_hex/omni_knobs.py::_apply_ang_turn_only_knob` (applied right
after the `std` knob, refusing loudly if the term is missing, has lost `std`, or already carries a
func this knob does not know), and unit/delta coverage of both the armed and the shipped state.
The rewards **class body** keeps the stock func and **no preset sets the variable** — the
knob-timing pillar: a cfg class body carries its declared value and knobs move it.

## The three rules

1. **Never export `KRABBY_PHASE` or `KRABBY_SIDE_PHASE` in an omni process.**
   `crab_hex_omni_phases.py::activate_phase` raises `RuntimeError` if `KRABBY_OMNI_PHASE` is set
   together with EITHER other preset variable — one task per process, because the expanded
   `KRABBY_*` knob names are shared across all three packages and whichever activation ran first
   would have `setdefault`-ed them already.
2. **`KRABBY_FLAT_TERRAIN_GEOM=ring` is required in EVERY omni process, preset or not.**
   `apply_omni_ring_knobs` calls `require_ring_geom` unconditionally in `__post_init__` (train,
   play, or eval) and raises if the value is anything else, including unset: a leaked forward/
   sideways corridor preset would otherwise train happily on corridors while every omni reward,
   the ring spawn and the radial exposure ledger quietly measured a course that is not there. The
   omni presets set it in every window; a manual run (e.g. viewing a checkpoint) must set it too.
3. **`KRABBY_HEADING` is refused, not ignored.** `_check_omni` raises if it is set: the
   forward/sideways heading-P-control mechanism is inert under the omni command term (the facing
   sampler and `reward_facing_exp` replace it outright), and a run launched with it set would look
   configured for turning while being nothing of the sort.

## Presets (`KRABBY_OMNI_PHASE`, `crab_hex_omni_phases.py`)

Phase 1 is **three stages**, all built from the forward window-0 env (`FORMATION` +
`WALKING_SLOTS` + `LONG_EPISODES`), differing only in the command sampler's direction band, slot
mix and RSI frame source:

| stage | iterations | resume | task | `KRABBY_OMNI_DIR_BAND` | `KRABBY_OMNI_SLOTS` (stand:move:turn:move_turn) | RSI `source` | facing band |
|---|---|---|---|---|---|---|---|
| `1a` | 1000 | scratch | Omni-Walk | `0:0` (body +x) | `0.2:0.8:0:0` | `0` (forward) | held (no turning slots) |
| `1b` | 1000 | `1a` | Omni-Walk | `1.5708:1.5708` (body +y) | `0.2:0.8:0:0` | `1` (sideways) | held |
| `1c` | 5000 | `1b` | Omni-Walk | unset (world-uniform) | `0.15:0.55:0.10:0.20` | `all` | `0.2:0.6` |
| `2a` | 5000 | `1c` | Omni-Teacher | unset | `0.15:0.45:0.15:0.25` | `all` | `0.2:1.2` |
| `2b` | 5000 | `2a` | Omni-Teacher | unset | same as 2a | `all` | `0.2:2.4` |
| `2c` | 5000 | `2b` | Omni-Teacher | unset | same as 2a | `all` | `0.2:2.4` (clock anneal 1.0→0.5) |
| `2c_clockheld` | 5000 | `2b` (**not** `2c`) | Omni-Teacher | unset | same as 2a | `all` | `0.2:2.4` (`KRABBY_CLOCK_W=1.0`, no phase-out — the anneal-vs-iterations control) |

Per-window deltas re-pointed on every stage: `KRABBY_RSI_BANK` → the omni bank,
`KRABBY_FLAT_TERRAIN_GEOM=ring`, `KRABBY_TERRAIN_PROMOTE=0.45:0.25` (ratio semantics, not the
forward horizon scaling); `KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.25:0.25`, `KRABBY_OMNI_ANG_VEL_MAX=0.3`,
`KRABBY_OMNI_CLOCK_TURN_R=1.2`, `KRABBY_OMNI_YAW_RANGE=-3.14159:3.14159`,
`KRABBY_YAW_PROGRESS_W=1.0` (the added rotational-progress term — armed everywhere; 1a/1b command
no turn, so the only thing it pays there is correcting yaw drift back to the held facing),
`KRABBY_OMNI_FACING_OBS_CLIP=0.6` (Revision 3, 2026-09-17 — the facing-error observation clip,
fixed at the 1c facing band's edge in every window regardless of that window's own facing-band
ramp), and the Revision 4 reward economy (2026-09-18): `KRABBY_TRACK_SIGMA2=0.04`,
`KRABBY_TRACK_L1_W=-2.0`, `KRABBY_OMNI_ANG_VEL_STD=0.5`, `KRABBY_OMNI_CLOCK_COMBINE=product` —
in every window; dropped
everywhere: `KRABBY_HEADING`, `KRABBY_RSI_SPAWN_FIX`, the corridor knobs (`KRABBY_CORRIDOR_HALF_WIDTH`
etc. — defensive, they never appear in a forward window today). Everything else — gait weights,
`KRABBY_PHASEOUT` ramps, `EPISODE_S 40`, `RESAMPLE_S 10:10`, `RSI_FRAC
0.2`, DR, `EDGE_W`/`STUMBLE_W`/`COLLISION_W`/`CLEARANCE_W`/`FOOT_CLEAR_*`/`SWING_MIN_CLEAR_W`,
`GOAL_VEL_W`, `FLAT_TERRAIN_MODE/FLAT_FRAC/DIFF` — is the forward window's own values, carried
unchanged. (`TRACK_SIGMA2`/`TRACK_L1_W` used to be on that carried-unchanged list; Revision 4 moved
them.) No phase 3 (teacher only in this plan).

### Omni-only knobs (`config/crab_hex/omni_knobs.py`, `mdp/omni_terrain_knobs.py`)

`KRABBY_OMNI_SLOTS` (`stand:move:turn:move_turn`, non-negative, not all zero) ·
`KRABBY_OMNI_DIR_BAND` (`lo:hi` body-relative radians, or empty/`none`/`off` for world-uniform) ·
`KRABBY_OMNI_SPEED_ELLIPSE` (`a:b:c` m/s, fore-aft/lateral/backward, must clear the walk floor) ·
`KRABBY_OMNI_FACING` (`lo:hi` radians, `0 <= lo <= hi`) · `KRABBY_OMNI_ANG_VEL_MAX` (rad/s, must
exceed `ang_vel_clip`) · `KRABBY_OMNI_CLOCK_TURN_R` (metres, ≥ 0) · `KRABBY_OMNI_YAW_RANGE`
(`lo:hi` spawn-yaw band radians, applied to BOTH the ring spawn and the RSI re-yaw) ·
`KRABBY_OMNI_RSI_SOURCE` (`0`, `1`, or `all`) · `KRABBY_RING_CLEAR_R` (metres) · `KRABBY_RING_BLOCKS`
(`sparse:dense` counts) · `KRABBY_YAW_PROGRESS_W` (the added `reward_yaw_progress` term's weight;
presence-based like the forward `_W` knobs, so `=0` is a legitimate ablation. Named without the
`OMNI` infix because it is a reward dose, but it is applied by `apply_omni_knobs` rather than by the
forward `_overrides` table — no forward cfg has that term) · `KRABBY_OMNI_FACING_OBS_CLIP` (rad,
must be `> 0`; writes `params["facing_obs_clip"]` onto every omni observation term, raises if none
is installed; symmetrically clips steering channels 6/7 — see Revision 3 above; armed at `0.6` in
every preset) · `KRABBY_OMNI_ANG_VEL_STD` (rad/s, must be `> 0`; writes
`rewards.track_ang_vel_z_exp.params["std"]`, raises if the term is missing — Revision 4, armed at
`0.5` in every preset) · `KRABBY_OMNI_CLOCK_COMBINE` (`sum` | `product`; writes `params["combine"]`
onto every reward term that already HAS that key — today `reward_clock_schedule` only, since
`clock_swing_apex_income` takes no such argument — raises if no term does. Applied after the
forward block, so it wins over the forward-owned `KRABBY_CLOCK_COMBINE`; Revision 4, armed at
`product` in every preset) · `KRABBY_OMNI_ANG_TURN_ONLY` (`0` | `1`, nothing else parses; `1`
swaps `rewards.track_ang_vel_z_exp`'s **func** — not a param — for
`crab_hex_omni_rewards.track_ang_vel_z_exp_turn_gated`, keeping the weight and every param.
**Staged 2026-09-18, DEFAULT OFF — set by no preset**; see the note below). `KRABBY_RSI_BANK` /
`KRABBY_RSI_FRAC` are shared with the forward task, as are `KRABBY_TRACK_SIGMA2` /
`KRABBY_TRACK_L1_W` (the omni presets re-point both: Revision 4).

## Training

From `parkour/` (`$PY` = `/home/nickmagus/krabby/isaac_venv/bin/python`):

```bash
# stage 1a: from scratch, body +x only
KRABBY_OMNI_PHASE=1a $PY scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Omni-Walk-v0 --headless --num_envs 256 --seed 3 --max_iterations 1000

# stage 1b: resume 1a, body +y only
KRABBY_OMNI_PHASE=1b $PY scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Omni-Walk-v0 --headless --num_envs 256 --seed 3 --max_iterations 1000 \
  --resume --checkpoint <abs path to the 1a head>

# stage 1c: resume 1b, any world direction + facing
KRABBY_OMNI_PHASE=1c $PY scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Omni-Walk-v0 --headless --num_envs 256 --seed 3 --max_iterations 5000 \
  --resume --checkpoint <abs path to the 1b head>

# phases 2a/2b/2c/2c_clockheld: teacher, resumed from the previous phase's head
# (2c and 2c_clockheld BOTH resume the 2b head -- never chain one from the other)
KRABBY_OMNI_PHASE=2a $PY scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Omni-Teacher-v0 --headless --num_envs 256 --seed 3 --max_iterations 5000 \
  --resume --checkpoint <abs path to the 1c head>
```

Never set `KRABBY_PHASE` or `KRABBY_SIDE_PHASE` in the same process — see "The three rules" above.

**Registration.** Same mechanism as the sideways task: the package is discovered by
`parkour_tasks/__init__.py::import_packages`, which only runs when the `parkour_tasks` source dir
is on `sys.path` ahead of `parkour/` — `train.py`, `play.py` and the eval harness already put it
first. Import order is forward → omni → sideways.

### Driver (unattended, multi-phase)

```bash
OMNI=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
"$OMNI/experiments/tools/launch_phases.sh" --campaign-dir <campaign dir> --plant A15+B \
  --phases 1a,1b,1c,2a,2b --seed 3
"$OMNI/experiments/tools/heartbeat_phases.sh" <unit> <campaign dir> 1800
# from EACH seed's 2b head, two SEPARATE launches (never listed together):
"$OMNI/experiments/tools/launch_phases.sh" --campaign-dir <campaign dir> --plant A15+B \
  --phases 2c --seed 3 --from-checkpoint <2b head>
"$OMNI/experiments/tools/launch_phases.sh" --campaign-dir <campaign dir> --plant A15+B \
  --phases 2c_clockheld --seed 3 --from-checkpoint <2b head>
```

The driver's default `--phases` is `1a,1b,1c,2a,2b` and default `--pause-after` is `2b` (the
decision point). No student anywhere: the training subprocess env carries only
`KRABBY_OMNI_PHASE` / `KRABBY_PLANT`. Halts (each notifies; `--continue` overrides, `--redo`
reruns): `dead_plant` (soundness at 2000 iterations on FROM-SCRATCH windows only, the sideways rule —
no omni window qualifies (1a is 1000 it), so collapse is covered by the soft gates and `gate_1c`), `gate_1a`/`gate_1b` (soft, short-stage), `gate_1c` (the written
formation gate), `gate_facing` (facing precision, enforced from 2a), `nan`, `infra`.

### Gates

- **`gate_1a`/`gate_1b` (soft, short stages).** After 1a/1b the driver runs the flat canary
  (`slow__omni`) ONLY — no ring rows. **1a:** `fwd` hold walking-tripod ≥ 0.30 AND survival
  THROUGH the `fwd` hold ≥ 0.70. **1b:** `lat_l` hold ≥ 0.30 AND survival through `lat_l` ≥ 0.70.
  Survival through a hold = the share of episodes that reach the end of that hold
  (`survival_by_hold`, cut from per-episode `n_steps` at the manifest's hold ends) — NOT
  whole-schedule completion, which counts falls on the five holds a short stage never trained
  (2026-09-17: seed-3 1a halted at 0.09 completion with 0.93 survival through `fwd`; the gate was
  corrected and the run continued). Floors, not like-for-like bars — they exist so a from-scratch
  formation failure doesn't burn 1b/1c/2a-2b hours unattended.
- **`gate_1c` (the written formation gate, all three rows).** Walking-hold tripod **≥ 0.40 on
  EACH moving hold** (`fwd, lat_l, diag, back, move_turn`) AND completion **≥ 0.70** AND a
  turning-rate criterion on `turn_l90`/`turn_r90`: median **`yaw_progress_rate`** (the signed
  `mean(wz · sign(cmd_wz))` over the turning steps) **≥ 0.05 rad/s** on each — the earlier
  `yaw_rate_ratio × ang_vel_max` proxy was retired on 2026-09-17 after it read +0.28 rad/s on a head
  whose true net rotation was −0.025 rad/s (gait wobble) — AND a per-direction floor: median tracking
  ratio (`speed_ratio`) **≥ 0.3** on each of `lat_l`, `diag`, `back`, `move_turn`, so a forward-only
  walker cannot pass an omnidirectional formation gate.
- **`gate_facing` (facing precision).** WATCHED at 1c, ENFORCED from 2a: median
  `heading_error_final_abs` **≤ 0.30 rad** on each turn hold (the 1c facing band, ±0.6 rad, is
  too narrow for this to be a fair test; 2a widens it to ±1.2).
- **The training-log translation floor (2026-09-18, every gate above).** On top of its own
  criteria, each gate requires the stage's TRAINING log to show
  `Metrics/base_parkour/track_ratio_mean ≥ 0.6` averaged over the last 500 iterations. Tripod and
  survival are gait and robustness metrics, not translation metrics: the seed-3 1b head passed its
  soft gate at 0.360 tripod / 0.96 survival while its speed tracking had already fallen to 0.406,
  and 1c then drifted to 0.245 unwatched. Seed-3 numbers: 1a **0.693 PASS**, 1b **0.406 FAIL**,
  1c **0.246 FAIL**. A missing number is a halt, never a skip. The eval-independent half of this
  matters on its own — the campaign's canaries were themselves defective (Revision 4, cause 1).
- **`dead_walker` (from-scratch early abort, 2026-09-18).** A from-scratch window longer than 1000
  iterations is killed at iteration 1000 if its its-800-1000 `track_ratio_mean` is below 0.5 — no
  walker is forming and the remaining 4000–7000 iterations buy nothing. The checkpoint is kept and
  the record is `--continue`-overridable, like a gate (1a formed a walker from scratch at 0.747 in
  exactly this budget, so the floor clears any run that is forming).
- **Watched, not enforced (any phase):** per-direction `speed_ratio` ≥ 0.5, `|duty A − duty B| ≤
  0.10`, cam `shaft_mean_abs_vel`, slip, knee saturation, `recap_frac`.

## Eval harness

One harness serves all three tasks (`crab_hex_forward_task/scripts/eval_crab_hex_gait.py`); on a
`command_frame: world` row it never writes `vel_command_b` directly — it hands each step's
`(speed, dir, face)` to the command term's `schedule_hook`, which runs the SAME world→body
conversion and facing P-control training uses:

```bash
$PY parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py --headless \
  --manifest parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml \
  --scenario slow__omni --checkpoint <model.pt>
```

Phase-2 heads add `--task Isaac-Crab-Hex-Omni-Teacher-Play-v0`. `scenarios_omni.yaml` (manifest
v2) has three rows, each an 8-hold schedule — 8 s per hold except the two turn holds at 12 s, 80-s
episodes (a 90° turn to ≤ 0.30 rad needs ≥ 0.16 rad/s sustained in 8 s, above the plant's ~0.15–0.2
rad/s) — (stand, fwd 0.25, lat_l 0.22, diag 0.25, back 0.22,
turn_l90 (face +π/2), turn_r90 (a right 90° turn back to face 0, not an absolute −π/2 — that would
be a 180° swing and no hold is long enough), move_turn 0.22 while turning to face +π/2):

- **`slow__omni`** — flat canary, no `env:` block (frozen shape).
- **`light__omni`** — step onset: ring geometry at `0.05:0.2`.
- **`ring__omni`** — no `env:` block; the driver injects the ring at `0.20:0.70` at run time
  (the PLAN H convention).

`facing_metrics` adds `heading_error_rms`, `heading_error_final_abs`, `heading_error_max_abs`,
`settle_time_s`, `yaw_progress_rate` (the gate quantity), `net_yaw_rate_toward` (cross-check from the
unwrapped heading error), `yaw_rate_ratio` (wobble-sensitive, kept for continuity only), `n_turn_steps`;
`facing_by_hold` and `facing_error_rms` land in
`aggregate`. **Manifest shape is frozen** — do not add or remove an `env:` block on an existing
row after it has been evaluated once (process-state sensitivity flips same-seed outcomes between
two repeatable results); a new row may carry whatever it needs.

### Log dirs and eval root

- Training: `parkour/logs/rsl_rl/crab_hex_omni_{flat_walk,teacher}`
- Eval: `parkour/logs/rsl_rl/gait_eval/omni_phases`

## Viewing a checkpoint

No preset is required for a viewer (and none should be set unless you want its whole knob stack) —
but rule 2 above still applies: `KRABBY_FLAT_TERRAIN_GEOM=ring` must be exported explicitly, since
`apply_omni_ring_knobs` runs in every env cfg's `__post_init__` regardless of a phase preset.
Every other omni knob falls back to the command cfg's own defaults (`slot_probs 0.15:0.55:0.10:0.20`,
`speed_ellipse 0.35:0.25:0.25`, `facing_delta 0.2:0.6`, `ang_vel_max 0.3`, world-uniform direction).

```bash
cd "$R/parkour"
export KRABBY_FLAT_TERRAIN_GEOM=ring
$PY scripts/rsl_rl/play.py --task Isaac-Crab-Hex-Omni-Walk-Play-v0 \
  --num_envs 1 --real-time \
  --checkpoint <abs path to model_N.pt>
```

Use `Isaac-Crab-Hex-Omni-Teacher-Play-v0` for a phase-2 head. Add explicit `KRABBY_OMNI_*` knobs
(above) to view a specific slot mix, direction band or facing width instead of the defaults.

## RSI

On by default (`KRABBY_RSI_FRAC 0.2`, unlike the sideways task where RSI is off). The bank
(`rsi_bank_omni.npz`) merges reference states harvested from BOTH heads of record — the forward
2c teacher (`source=0`, body +x) and the sideways 2a teacher (`source=1`, body +y, left-leads) —
2000 frames each, sha256-pinned in `bundle.yaml`. Each frame is re-yawed to a sampled facing, its
world velocity rotated into the new frame, and its **first command staged** so the episode starts
on-command with zero facing error (`reset_from_reference_states_omni`; the RSI reset runs before
the command manager's own resample). Stage 1a draws only `source=0` frames, 1b only `source=1`,
1c onward `all`. See `experiments/2026-09-16_1805_omni_formation/RSI_BANK.md` for the harvest
commands, filters and provenance. **Backward travel has no RSI seed** — rotating a stored frame
never yields backward momentum; the campaign's backward hold is unseeded by design.

## Tests

```bash
pytest tests/unit -k "crab_hex or travel_frame or parkour_command or exposure or policy_of_record or experiment_layout or gait"
RUN_CRAB_HEX_CFG_IDENTITY=1 pytest tests/integration/test_crab_hex_phase_configs.py \
  tests/integration/test_crab_hex_sideways_configs.py tests/integration/test_crab_hex_omni_configs.py -v
```

The pure suite includes the omni-specific modules (`test_crab_hex_omni_{command,observations,
rewards,mirror,rsi,ring_terrain,terrain_knobs,exposure,phases,gate}.py`) plus the seam tests shared
with the forward/sideways tasks (`test_crab_hex_clock_speed.py`, `test_travel_frame_seams.py`,
`test_crab_hex_forward_seam_literals.py`, `test_crab_hex_gait_schedule.py`,
`test_crab_hex_gait_metrics.py`, `test_crab_hex_stability_metrics.py`,
`test_crab_hex_gait_eval_golden.py`, `test_experiment_layout.py`). The Isaac delta test
(`test_crab_hex_omni_configs.py`) asserts the omni-vs-forward config diff equals exactly the
documented delta and that the forward AND sideways config dumps are byte-identical pre/post.

## Caveats

- **Edge-avoidance is inert on both courses at these heights.** The recal2b2 height formulas
  (≤ 0.09 m at difficulty 1) never cross the mesh edge-detection trigger (0.12 m per pixel), so
  `reward_feet_edge` is a dead term on the forward corridor today and on the ring at this plan's
  heights too — pre-existing, not a regression; the both-axes mask is kept and proven correct on
  a synthetic block in the terrain unit test.
- **`reach_obst`/`block_steps_mean` are watched-only telemetry, never gated.** The shipped knob
  loop pins every sub-terrain's `noise_range` to (0.02, 0.02) at 1a and 2a+ alike, so the disc
  roughness is −0.030…+0.025 m at every difficulty and `block_height_tol_m` 0.03 produces NO
  false-positive block pixels from roughness (measured on the real decorated ring at d ∈ {0.05,
  0.2, 0.7} — see `mdp/omni_exposure.py`); the residual bias is under-reporting at 1a, whose lowest
  blocks (2.25–3.3 cm) sit at the tolerance. Tracked for trend, never a promotion or gate signal.
- **Facing has no reference cell.** Neither the forward nor the sideways head has ever commanded
  a world facing target, so the decision table's facing columns are blank by construction for the
  hard-coded forward/sideways reference rows — there is nothing to recompute.
- **The 4-image mirror roughly doubles the symmetry-loss forward-pass cost** per PPO update (4
  images here vs. 2 for the forward/sideways tasks) — expect omni windows to run somewhat slower
  than the naive per-iteration budget suggests.
- **Backward travel has no RSI seed** (see the RSI section above) — allowed, just unseeded.
- Ring obstacle numbers and the 8-hold (6×8 s + 2×12 s) canary schedule are not comparable to the corridor
  rows or the 3×10 s schedules of the forward/sideways tasks; compare only within this campaign.
  The per-hold walking tripod and per-direction `speed_ratio` are the like-for-like cells against
  the forward/sideways reference rows.

## See also

- [`crab_hex_forward_task/README.md`](../crab_hex_forward_task/README.md) — the shared onboarding
  doc (environment setup, plant, the full MDP reference) and the forward task's own train/eval
  commands.
- [`crab_hex_sideways_task/README.md`](../crab_hex_sideways_task/README.md) — the other sibling;
  its 2a teacher head is one of this task's two RSI sources.
- `docs/crab-hex-forward-policy-config.md` — the shared config reference; its `KRABBY_PHASE` row
  names all three preset variables.
- This package's campaign CHARTER
  (`experiments/2026-09-16_1805_omni_formation/CHARTER.md`) — the question, hypotheses, held-fixed
  list, gates, decision table and caveats for the first training campaign; `RSI_BANK.md` in the
  same directory has the harvest provenance.
