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
| `track_ang_vel_z_exp` std | 0.5 | 0.15 (gradient at 0.15–0.3 rad/s where the old std had none) |
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
(gradient across 0.15–0.3 rad/s); the gait clock driven by turning (`r_turn` 1.2 m,
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
`track_ang_vel_z_exp` (std 0.15) and `reward_facing_exp` (weight 0.2) both pay about the same for
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
ramp) in every window; dropped
everywhere: `KRABBY_HEADING`, `KRABBY_RSI_SPAWN_FIX`, the corridor knobs (`KRABBY_CORRIDOR_HALF_WIDTH`
etc. — defensive, they never appear in a forward window today). Everything else — gait weights,
`KRABBY_PHASEOUT` ramps, `TRACK_SIGMA2`/`TRACK_L1_W`, `EPISODE_S 40`, `RESAMPLE_S 10:10`, `RSI_FRAC
0.2`, DR, `EDGE_W`/`STUMBLE_W`/`COLLISION_W`/`CLEARANCE_W`/`FOOT_CLEAR_*`/`SWING_MIN_CLEAR_W`,
`GOAL_VEL_W`, `FLAT_TERRAIN_MODE/FLAT_FRAC/DIFF` — is the forward window's own values, carried
unchanged. No phase 3 (teacher only in this plan).

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
every preset). `KRABBY_RSI_BANK` / `KRABBY_RSI_FRAC`
are shared with the forward task.

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
  (`slow__omni`) ONLY — no ring rows. **1a:** `fwd` hold walking-tripod ≥ 0.30 AND completion ≥
  0.70. **1b:** `lat_l` hold ≥ 0.30 AND completion ≥ 0.70. Floors, not like-for-like bars — they
  exist so a from-scratch formation failure doesn't burn 1b/1c/2a-2b hours unattended.
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
