# M17 Task 6 — HAL → model mapper audit

Audit of every value the parkour model consumes from the real robot, where it
comes from, the scaling the mapper applies, and its M17 status. The mapper is
`compute/parkour/mappers/hardware_to_model.py` (`_extract_proprioceptive`); the
observation source is `JetsonHalServer.set_observation()` in
`hal/server/jetson/hal_server.py`.

Joint count `n` = 18 for `--robot hex` (`KRABBY_HEX_DEFINITION`, 6 legs ×
{hip_yaw, hip_pitch, knee}).

## Proprioceptive vector layout

| Offset | Field | Source (real path) | Mapper scaling | M17 status |
|--------|-------|--------------------|----------------|------------|
| 0:3 | base angular velocity | `base_ang_vel_b` ← ZED IMU (Task 5), camera→body rotated | ×0.25 | **Live** (Task 5). rad/s after deg→rad + mount rotation. |
| 3:5 | IMU roll, pitch | `base_quat_w` ← ZED IMU → `euler_xyz_from_quat` | wrap to [−π, π] | **Live** (Task 5). Sign verified by tilt test (bench). |
| 5 | zero placeholder | — | — | Left 0. |
| 6 | `delta_yaw` | `hw_obs.delta_yaw` | — | **Isaac-only → 0** (see gap doc). |
| 7 | `delta_next_yaw` | `hw_obs.delta_next_yaw` | — | **Isaac-only → 0**. |
| 8 | zero placeholder (vy) | — | — | Left 0. |
| 9 | vx command | `nav_cmd.vx` (controller) | — | Available from command source. |
| 10 | `terrain_type_flag` | `hw_obs.terrain_type_flag` | — | **Isaac-only**, default 1.0. |
| 11 | `flat_terrain_flag` | `hw_obs.flat_terrain_flag` | — | **Isaac-only**, default 0.0. |
| 12 : 12+n | joint positions | `hw_obs.joint_positions` ← MCU telemetry `pos` | rel. to default | **Live (6b)**. Firmware `pos` is normalized [0,1]; falls back to echoed command when a joint has no telemetry yet. |
| 24 : 24+n | joint velocities | `hw_obs.joint_velocities` ← Python-side EMA | **none applied** (see §Velocity) | **Live (6c)**. Differentiated from successive `pos`, EMA α=0.2. |
| 36 : 36+n | previous action | `hw_obs.previous_action` ← last command | — | **Live (6d)**. `_last_joint_positions` echo. |
| 12+3n : … | contact forces (5) | `hw_obs.contact_forces` ← MCU current | clip [−0.5, 0.5] | **First pass (6e)**. See §Contacts. |

## Joint positions (6b)

`set_observation()` overlays `KrabbyMCUSDK.read_telemetry()` onto the echoed
command in `_apply_mcu_telemetry()`. The firmware `JointTelemetry.pos` is already
normalized to [0,1] (Task 2 calibration), so it is written directly — no extra
scaling. Joints without telemetry (none reported yet) keep the echoed-command
value. UNCAL/PARTIAL joints report a *relative* `pos` (still [0,1] units) until
they self-heal to FULL against an end-stop.

**Measured range:** to be filled from a bench log — expected [0,1] per joint,
centered near 0.5 (neutral). Capture with the flail test (6h).

## Joint velocities (6c)

The MCU firmware emits position but no velocity, so velocity is computed in the
HAL server: per joint, `vel = (pos_t − pos_{t−1}) / (t − t_{t−1})` using
`time.monotonic()`, then single-pole EMA-smoothed (`α = 0.2`, see
`_JOINT_VEL_EMA_ALPHA`) to suppress serial-jitter spikes. Units are normalized
position-units per second (`pos` is [0,1]). First tick (no prior sample) yields
0. Firmware-side velocity is a follow-up; M17 ships the Python path.

> **Scaling discrepancy (pre-existing, flagged for M15).** The mapper docstring
> says joint velocities are ×0.05, and `RobotDefinition.observation_scaling`
> carries `joint_vel=0.05`, but **neither the mapper nor the HAL actually applies
> it** — `hardware_to_model.py:179` copies `joint_velocities` through raw (its
> comment: "HAL may already apply scaling", which it does not). By contrast the
> base-angular-velocity ×0.25 *is* applied (hardcoded at `:129`). If the M2
> student was trained on scaled velocities, the magnitude will be off until this
> is reconciled. M17's bar is data flow, not policy quality, so this is logged
> for M15 rather than fixed here.

**Measured range:** to be filled from a bench log.

## Contact forces (6e)

Mapping decision: **Option A — five legs, drop one** (`krabby_mcusdk.py`,
`current_to_contact_forces`). The model's `contact_forces` is 5-wide (trained
against a quadruped foot set); the hex has 6 legs. The two middle legs (ML/MR)
are geometrically redundant for a forward gait, so **MR is dropped** and the five
slots map to `(FL, FR, ML, RL, RR)`.

Per-leg load proxy = sum of that leg's three joint currents (raw ADC counts from
`JointTelemetry.current`). Scaling (first pass):

```
slot = clip(leg_current_sum / _CONTACT_FULLSCALE − 0.5, −0.5, 0.5)
```

with `CONTACT_FULLSCALE = 300.0` (0 current → −0.5 "no contact";
fullscale → +0.5 "firm contact"). Legs with no telemetry map to 0.0 (unknown).
The mapping lives in `firmware/observation_mapping.py` (pure Python, shared with
the `krabby-firmware observe` bench command); the HAL wraps it to `np.float32`.

> **Placeholder scale.** `CONTACT_FULLSCALE` is a structural placeholder, **not**
> calibrated. It must be retuned against Task 4's loaded-vs-unloaded `avgIS`
> ranges once the current-sense IS-line fault (Task 4 finding: bench reads ~6
> under load vs. an expected ≳100) is resolved on the chassis. Expected to be
> refined in M15.

## Notes

- Base position / linear velocity: `base_lin_vel_b` is filled from ZED positional
  tracking when enabled (Task 5 path), else zero.
- The Isaac-populated-but-empty-on-real fields are enumerated in
  [`m17-isaac-vs-real-gaps.md`](m17-isaac-vs-real-gaps.md) — the M15 domain-
  randomization target list.
