# Crab hexapod sideways-walk task (`crab_hex_sideways_task`)

The **sideways-walk sibling** of [`crab_hex_forward_task`](../crab_hex_forward_task/README.md):
the same robot, the same plant (A15+B, `assets/crab.usda`), and the same MDP terms, but the travel
axis is the body **+y** axis instead of +x — **left side leads** (the robot spawns at yaw -pi/2 so
body +y points down the course). This package is a thin subclass of the forward package's config
classes: it reuses `crab_hex_forward_task/mdp/*` unchanged and changes only the handful of
config/reward sites that need to know the travel direction (Shape B: the commanded speed stays in
`vel_command_b[:, 0]`, read as "speed along the travel axis" — a constant `travel_yaw = +pi/2`
rotation of the body frame). See `docs/crab-hex-forward-policy-config.md` for the shared reference
and this README's delta table below for exactly what differs.

Nothing about the *policy* has changed on purpose — this is the first plan for the task and
changes only what is necessary to go from walking forward to walking sideways. Reward weights,
terrain/curriculum knobs, PPO/distillation hyper-parameters, terminations and the 18-channel
action space are byte-identical to the forward recipe (verified by an Isaac config-delta test;
see `tests/integration/test_crab_hex_sideways_configs.py`).

## Delta from the forward task

| Site | Forward | Sideways |
|---|---|---|
| Travel axis | body +x | body +y (`travel_yaw = pi/2`) |
| Spawn rotation | identity | yaw -pi/2 (`w=0.7071, z=-0.7071`) |
| Heading offset | none | `commands.base_velocity.heading_offset = -pi/2` |
| Velocity-tracking reward func | `track_lin_vel_xy_exp` | `track_lin_vel_travel_exp` (same term name/params) |
| Height-scan look-ahead offset | `pos (0.6, 0.0, 20.0)` | `pos (0.0, 0.6, 20.0)` (size/resolution/ordering unchanged — 108 rays, obs 1149) |
| Mirror-symmetry plane | left-right (front/rear legs preserved) | fore-aft (front<->rear legs swap, ML/MR fixed) |
| Preset variable | `KRABBY_PHASE` | `KRABBY_SIDE_PHASE` |
| Gym ids | `Isaac-Crab-Hex-{Flat-Walk,Teacher,Student}-v0` | `Isaac-Crab-Hex-Side-{Walk,Teacher,Student}-v0` |
| RSL-RL experiment dirs | `crab_hex_{flat_walk,teacher,student}` | `crab_hex_side_{flat_walk,teacher,student}` |
| RSI | on (`KRABBY_RSI_FRAC=0.2`, forward bank) | **off** (the forward bank is a forward gait; an RSI reset would teleport a sideways walker back to yaw 0 — see the campaign CHARTER's fallback for a sideways-native bank recipe) |
| Depth camera (phase 3) | +x face | +y face (hardware placeholder; the actual mount is a phase-3 decision). **Known blocker before 3a/3b:** the shared prestartup event `random_camera_position` treats the parent-relative offset rotation as a world orientation, so under the rotated spawn the camera is yawed an extra +90 deg at runtime (config correct, no-op for forward) — see caveat 10 in the campaign CHARTER |

Everything else — reward weights and ramps, the clock/stride/tripod reward math, the contact
proxy, terrain/curriculum/promotion/DR knobs, the command band, PPO and distillation
hyper-parameters, seed and env counts, the eval hold schedules — is the forward recipe unchanged.

## Training

From `parkour/`:

```bash
# phase 1a: pure student, from scratch
KRABBY_SIDE_PHASE=1a $PY scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Side-Walk-v0 --headless --num_envs 256 --seed 3 --max_iterations 5000

# phases 2a/2b/2c: teacher-student, resumed from the previous phase's head
KRABBY_SIDE_PHASE=2a $PY scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Side-Teacher-v0 --headless --num_envs 256 --seed 3 --max_iterations 5000 \
  --resume --checkpoint <abs path to the 1a head>
```

(`$PY` = `/home/nickmagus/krabby/isaac_venv/bin/python`.) Never set `KRABBY_PHASE` in the same
process — see "Two rules" below.

**Registration.** The package is discovered by `parkour_tasks/__init__.py::import_packages` — but that
`__init__` only runs when the `parkour_tasks` *source dir* (`parkour/parkour_tasks`) is on `sys.path`
ahead of `parkour/`; otherwise the bare name resolves to the outer directory as a namespace package
and only packages imported by name elsewhere (the forward task, via `parkour_isaaclab`) get
registered. `train.py`, `play.py` and the eval harness put the source dir first
(`tests/unit/test_rsl_rl_entry_points_resolve_parkour_tasks.py` pins it; the driver never imports
`parkour_tasks` itself — it only spawns those two); a new script that boots Kit and needs the
`Isaac-Crab-Hex-Side-*` ids must do the same before `import parkour_tasks`.

### Driver (unattended, multi-phase)

```bash
"$SIDE/experiments/tools/launch_phases.sh" --campaign-dir <campaign dir> --plant A15+B \
  --phases 1a,2a,2b,2c --seed 3
"$SIDE/experiments/tools/heartbeat_phases.sh" <unit> <campaign dir> 1800
```

`--phases` should always be passed explicitly (the driver's default is
`1a,2a,2b,2c,3a,3b`; phase 3 needs the depth-camera placement decided first). The driver chains
phases, resumes each from the previous head, evaluates after every phase, and halts (with a
notify line) on `dead_plant`, `gate_1a` (the written 1a formation gate — see the campaign
CHARTER), `nan` or `infra`; relaunch with `--continue` or `--redo`.

### Eval harness

One harness serves both tasks (`crab_hex_forward_task/scripts/eval_crab_hex_gait.py` reads
`travel_yaw` off the env cfg):

```bash
$PY parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py --headless \
  --manifest parkour_tasks/parkour_tasks/crab_hex_sideways_task/experiments/eval/scenarios_side.yaml \
  --scenario slow__side --checkpoint <model.pt>
```

Phase-3 (distillation) heads additionally need `--task Isaac-Crab-Hex-Side-Student-v0` and
`KRABBY_STUDENT_MDP=1` in the process env (the driver's eval helpers add both automatically).

### Log dirs and eval root

- Training: `parkour/logs/rsl_rl/crab_hex_side_{flat_walk,teacher,student}`
- Eval: `parkour/logs/rsl_rl/gait_eval/side_phases`

## Two rules

1. **Never export `KRABBY_PHASE` in a sideways process.** `crab_hex_side_phases.py::activate_phase`
   raises `RuntimeError` if both `KRABBY_PHASE` and `KRABBY_SIDE_PHASE` are set — one task per
   process, because the expanded `KRABBY_*` knob names are shared and the forward activation would
   `setdefault` the RSI keys first.
2. **Never leave `KRABBY_SIDE_PHASE` set for a forward run.** The forward package ignores it
   (a leaked value would silently train the un-expanded default MDP instead of a preset). The
   drivers on both sides strip every `KRABBY_*` variable from the subprocess environment before
   each launch, so this only matters for manual runs — `unset KRABBY_SIDE_PHASE` first.

## RSI

Off for this campaign (see the Delta table above and the campaign CHARTER's fallback recipe,
`harvest_side_rsi_bank.py`, if phase 1a needs it).

## See also

- [`crab_hex_forward_task/README.md`](../crab_hex_forward_task/README.md) — the shared onboarding
  doc (environment setup, plant, the full MDP reference) and the forward task's own train/eval
  commands.
- [`crab_hex_forward_task/policy/manifest.yaml`](../crab_hex_forward_task/policy/manifest.yaml) —
  the forward policy of record and the reference eval numbers this task's campaigns are compared
  against.
- `docs/crab-hex-forward-policy-config.md` — the shared config reference; its `KRABBY_PHASE` knob
  row notes this package's `KRABBY_SIDE_PHASE` sibling.
- This package's campaign CHARTER (`experiments/2026-09-11_0414_sideways_formation/CHARTER.md`) —
  the hypotheses, the written 1a gate, and the decision table for the first training campaign.
