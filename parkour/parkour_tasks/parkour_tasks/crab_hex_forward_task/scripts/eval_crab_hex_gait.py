# SPDX-License-Identifier: BSD-3-Clause
"""Gait-metrics eval harness for the crab hexapod (Milestone 18, Task 0).

Plays a checkpoint over a **fixed** command schedule and writes per-episode gait metrics -- the
headline being a scalar tripod-phasing score that later milestone tasks gate on. Trains nothing,
adds no rewards, mutates no MDP semantics: the only env-cfg changes are determinism knobs (seeds,
frozen terrain levels, disabled command resampling).

Why this exists: training logs report ``crab_failure``, episode length, and reward terms, none of
which distinguish a healthy alternating-tripod walk from "tippy-tap" micro-stepping or from a foot
that stays planted and skates. Those distinctions are what Task 1's reward changes are scored on.

Mechanism: the env's own random command resampling is replaced by monkeypatching the *instance*
``compute`` on ``UniformParkourCommand`` (the pattern ``demo_crab_hex_student.py`` uses for teleop),
so every episode sees the identical command sequence and metrics are comparable run to run.

Scoring lives in ``gait_eval/`` (pure numpy, unit-tested without Isaac). This file only simulates
and logs.

Launch from ``krabby-research/parkour`` via ``isaaclab.sh -p`` -- see README section 4.1b.
"""

from __future__ import annotations

from os import environ as _ENVIRON  # module-level: main() has a later local `import os`
import argparse
import contextlib
import hashlib
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
_TASK_DIR = _SCRIPT_DIR.parent
# scripts -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour
_PARKOUR_ROOT = _SCRIPT_DIR.parents[3]
_DEFAULT_MANIFEST = _TASK_DIR / "experiments" / "eval" / "scenarios_v1.yaml"

# Load cli_args without putting ``parkour/scripts`` on sys.path (it shadows pip ``rsl_rl``).
_cli_args_path = _PARKOUR_ROOT / "scripts" / "rsl_rl" / "cli_args.py"
_cli_spec = importlib.util.spec_from_file_location("parkour_rsl_rl_cli_args", _cli_args_path)
if _cli_spec is None or _cli_spec.loader is None:
    raise ImportError(f"Cannot load cli_args from {_cli_args_path}")
cli_args = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(cli_args)

parser = argparse.ArgumentParser(description="Crab-hex gait metrics eval harness (fixed command schedule).")
parser.add_argument("--manifest", type=str, default=None, help=f"Scenario manifest (default {_DEFAULT_MANIFEST}).")
parser.add_argument("--scenario", type=str, default=None, help="Scenario id within the manifest.")
parser.add_argument("--task", type=str, default=None, help="Gym task id (ad-hoc mode, overrides scenario).")
parser.add_argument(
    "--holds",
    type=str,
    default=None,
    help="Ad-hoc schedule 'vx:seconds[:label],...' e.g. '0.45:6:low,0.65:6:mid,0.85:6:high'.",
)
parser.add_argument("--episodes", type=int, default=None, help="Episodes == parallel envs.")
parser.add_argument("--episode-length-s", type=float, default=None)
parser.add_argument("--env-seed", type=int, default=None)
parser.add_argument(
    "--zero-actions",
    action="store_true",
    help="Send zero actions instead of loading a policy (pipeline smoke test; expects duty~1, tripod 0, slip~0).",
)
parser.add_argument("--freeze-friction", action="store_true", help="Disable friction randomization for low-variance A/B.")
parser.add_argument(
    "--policy-role",
    choices=["auto", "teacher", "student"],
    default="auto",
    help="Which head to run: 'auto' picks student (depth actor + encoder) when the task's runner is a "
         "distillation runner, else teacher. 'teacher' forces the privileged actor + estimator path even on "
         "a student task (diagnostic: is the student MDP walkable by the teacher?).",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--output-root", type=str, default=None, help="Default: parkour/logs/rsl_rl/gait_eval/v1.")
parser.add_argument("--save-raw", action="store_true", default=True, help="Write per-episode NPZ arrays.")
parser.add_argument("--no-save-raw", dest="save_raw", action="store_false")
parser.add_argument("--plot", action="store_true", default=True, help="Render gait diagrams after the run.")
parser.add_argument("--no-plot", dest="plot", action="store_false")
parser.add_argument(
    "--plant",
    type=str,
    default=None,
    help="Named plant (crab_hex_phases.PLANTS: A15+B|main = assets/crab.usda, legacy_golden|golden = "
         "assets/variants/crab_simple__splay00_axis5p5in.usda, B/A10/A15/A20/A10+B/A20+B variants). Exported as KRABBY_PLANT before "
         "the task package import (the USD is read at config-import time, so a manifest env block cannot "
         "select it). Default: whatever the environment selects, else the main asset.",
)
parser.add_argument(
    "--waves",
    type=int,
    default=None,
    help="Category-suite rows only: a consistency assertion on the manifest's derived wave count "
         "(one wave per category hold), never a configuration. Refused on any other row.",
)
parser.add_argument(
    "--trained-categories",
    type=str,
    default=None,
    help="Category-suite rows only: CSV of the categories the evaluated recipe trained (recorded in "
         "run_meta and aggregate.suite; scoring never reads it). Refused on any other row.",
)
parser.add_argument(
    "--allow-checkpoint-sha-mismatch",
    action="store_true",
    help="Proceed even if the checkpoint sha256 differs from the manifest's pin.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.plant:
    _ENVIRON.setdefault("KRABBY_PLANT", args_cli.plant)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Isaac / running from ``parkour/scripts`` can add paths that shadow pip ``rsl_rl`` (see zero_agent.py).
_parkour_scripts = _PARKOUR_ROOT / "scripts"
_KRABBY_RESEARCH = _PARKOUR_ROOT.parent
for _p in (str(_parkour_scripts), str(_parkour_scripts / "rsl_rl"), str(_PARKOUR_ROOT / "parkour_tasks")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_KRABBY_RESEARCH))
sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))
# gait_eval lives beside this script; keep its dir importable after the surgery above.
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import parkour_tasks  # noqa: F401,E402
from parkour_tasks.crab_hex_forward_task.config.crab_hex import crab_hex_phases as PH  # noqa: E402
from gait_eval import metrics as M  # noqa: E402
from gait_eval import report as R  # noqa: E402
from gait_eval import schedule as S  # noqa: E402
from isaaclab.sensors.ray_caster import RayCaster  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.warp import raycast_mesh  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from parkour_isaaclab.envs.mdp.parkour_commands.uniform_parkour_command import (  # noqa: E402
    UniformParkourCommand,
)
from parkour_tasks.crab_hex_forward_task.mdp.crab_contact_sensors import (  # noqa: E402
    CRAB_HEX_FOOTPAD_BODY_NAMES,
)
from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner  # noqa: E402
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402

# Command channels the policy can actually observe. observations.py concatenates
# ``0 * commands[:, 0:2]`` then ``commands[:, 0:1]``: vy is explicitly zeroed and wz never enters
# the vector at all. Recorded into every report so an off-axis "tracking error" is never mistaken
# for a capability measurement. Derived at run time from the observation term when it declares
# ``OBSERVED_COMMAND_SLOTS`` (the omni term does: {8: 0, 9: 1, 5: 2}); this literal is the
# forward/sideways answer and the fallback.
POLICY_SEES_COMMAND_CHANNEL = {"vx": True, "vy": False, "wz": False}
OBS_IDX_DELTA_YAW = 6
OBS_IDX_DELTA_NEXT_YAW = 7
OBS_IDX_CMD_VX = 10
#: ``{observation slot -> vel_command_b channel}``. The end-to-end "the schedule reached the
#: policy" check runs over these pairs; today's single pair is obs dim 10 == commanded vx.
DEFAULT_OBSERVED_COMMAND_SLOTS = {OBS_IDX_CMD_VX: 0}
_CMD_CHANNEL_NAMES = ("vx", "vy", "wz")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_holds(spec: str) -> list[S.Hold]:
    holds: list[S.Hold] = []
    for i, part in enumerate(p for p in spec.split(",") if p.strip()):
        fields = part.split(":")
        if len(fields) < 2:
            raise ValueError(f"--holds entry {part!r} must be 'vx:seconds[:label]'")
        holds.append(
            S.Hold(
                vx=float(fields[0]),
                hold_s=float(fields[1]),
                label=(fields[2] if len(fields) > 2 else f"hold{i}"),
            )
        )
    return holds


def _resolve_scenario() -> S.Scenario:
    """Manifest scenario, or an ad-hoc one built from CLI flags."""
    if args_cli.scenario:
        manifest = Path(args_cli.manifest) if args_cli.manifest else _DEFAULT_MANIFEST
        scenarios, _ = S.load_manifest(manifest)
        matches = [s for s in scenarios if s.id == args_cli.scenario]
        if not matches:
            raise SystemExit(f"scenario {args_cli.scenario!r} not in {manifest} ({[s.id for s in scenarios]})")
        scenario = matches[0]
    else:
        if not args_cli.task:
            raise SystemExit("provide --scenario (with --manifest) or --task for an ad-hoc run")
        holds = _parse_holds(args_cli.holds) if args_cli.holds else [S.Hold(vx=0.45, hold_s=6.0, label="default")]
        scenario = S.Scenario(
            id=args_cli.scenario or "adhoc",
            task=args_cli.task,
            checkpoint=str(args_cli.checkpoint or ""),
            schedule=holds,
            episodes=args_cli.episodes or 4,
        )
    # CLI overrides win over the manifest so a scenario can be probed without editing it.
    if args_cli.task:
        scenario.task = args_cli.task
    if args_cli.checkpoint:
        scenario.checkpoint = str(args_cli.checkpoint)
    if args_cli.episodes:
        scenario.episodes = args_cli.episodes
    if args_cli.episode_length_s:
        scenario.episode_length_s = args_cli.episode_length_s
    if args_cli.env_seed is not None:
        scenario.env_seed = args_cli.env_seed
    if args_cli.freeze_friction:
        scenario.freeze_friction = True
    S.validate_scenario(scenario)
    return scenario


def _wrap_to_pi(angle: "torch.Tensor") -> "torch.Tensor":
    """Wrap to ``(-pi, pi]`` without importing isaaclab's math utils into the pure-ish helpers."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _observed_command_slots(unwrapped) -> tuple[dict[int, int], str]:
    """``({obs slot: vel_command_b channel}, source)`` for the override-reached-the-policy check.

    Read from the observation term's ``OBSERVED_COMMAND_SLOTS`` class attribute when it declares
    one (the omni observations do -- three command channels instead of one); otherwise today's
    single ``{10: 0}`` pair, so forward and sideways rows run the identical check.
    """
    candidates = []
    manager = getattr(unwrapped, "observation_manager", None)
    for terms in (getattr(manager, "_group_obs_term_cfgs", None) or {}).values():
        candidates.extend(getattr(term, "func", None) for term in terms)
    obs_cfg = getattr(getattr(unwrapped, "cfg", None), "observations", None)
    if obs_cfg is not None:
        for group in vars(obs_cfg).values():
            if hasattr(group, "__dict__"):
                candidates.extend(getattr(term, "func", None) for term in vars(group).values())
    for func in candidates:
        slots = getattr(func, "OBSERVED_COMMAND_SLOTS", None)
        if slots:
            name = getattr(func, "__name__", type(func).__name__)
            return {int(k): int(v) for k, v in dict(slots).items()}, f"{name}.OBSERVED_COMMAND_SLOTS"
    return dict(DEFAULT_OBSERVED_COMMAND_SLOTS), "default (obs dim 10 == cmd vx)"


def _policy_sees_command_channels(slots: dict[int, int]) -> dict[str, bool]:
    """``{"vx"/"vy"/"wz": bool}`` derived from the observed slots."""
    seen = set(slots.values())
    return {name: (k in seen) for k, name in enumerate(_CMD_CHANNEL_NAMES)}


# The derivation must reproduce the hand-written forward answer, or every forward report's
# ``policy_sees_command_channel`` silently changes meaning.
assert _policy_sees_command_channels(DEFAULT_OBSERVED_COMMAND_SLOTS) == POLICY_SEES_COMMAND_CHANNEL


def _apply_env_overrides(env_cfg, scenario: S.Scenario) -> dict:
    """Determinism knobs only -- no reward/MDP semantics are touched."""
    applied: dict[str, object] = {}
    env_cfg.seed = scenario.env_seed
    applied["seed"] = scenario.env_seed

    if scenario.episode_length_s:
        env_cfg.episode_length_s = scenario.episode_length_s
    else:
        # Leave headroom so the schedule always fits inside one episode.
        env_cfg.episode_length_s = float(scenario.total_duration_s + 2.0)
    applied["episode_length_s"] = env_cfg.episode_length_s

    # The override below fully replaces compute(), but disabling resampling too means a stray
    # reset-driven _resample_command can't leak a random command into a step we log.
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.curriculum = None

    if getattr(env_cfg, "parkours", None) is not None:
        # Terrain levels otherwise advance on every reset, so episodes would not be comparable.
        env_cfg.parkours.base_parkour.freeze_terrain_levels = scenario.freeze_terrain_levels
        env_cfg.parkours.base_parkour.debug_vis = False
        applied["freeze_terrain_levels"] = scenario.freeze_terrain_levels

    terrain = getattr(env_cfg.scene, "terrain", None)
    tg = getattr(terrain, "terrain_generator", None) if terrain is not None else None
    if tg is not None and scenario.terrain_generator_seed is not None:
        tg.seed = scenario.terrain_generator_seed
        applied["terrain_generator_seed"] = scenario.terrain_generator_seed

    events = getattr(env_cfg, "events", None)
    if events is not None:
        if getattr(events, "push_by_setting_velocity", None) is not None:
            events.push_by_setting_velocity = None
            applied["push_by_setting_velocity"] = None
        if scenario.freeze_friction and getattr(events, "physics_material", None) is not None:
            events.physics_material = None
            applied["physics_material"] = None

    # World rows only: pin the spawn heading. The omni reset event samples a uniform yaw, so
    # without this the first hold's facing error is a different (random) number in every episode
    # and the whole facing block measures the spawn, not the policy.
    if scenario.command_frame == "world":
        face0 = scenario.spawn_yaw_rad()
        if face0 is not None:
            reset = getattr(events, "reset_root_state", None) if events is not None else None
            params = getattr(reset, "params", None)
            if params is None or "yaw_range" not in params:
                raise SystemExit(
                    f"scenario {scenario.id!r} sets spawn_yaw={scenario.spawn_yaw!r} but the env's "
                    "events.reset_root_state has no 'yaw_range' param to pin (the omni reset event "
                    "does; a forward/sideways task cannot run a world-frame scenario)."
                )
            params["yaw_range"] = (face0, face0)
            applied["spawn_yaw_pin"] = face0
            # The omni RSI reset (present whenever the preset carries KRABBY_RSI_FRAC > 0) draws
            # its OWN spawn yaw for the reference-state subset; pin it to the same value so no
            # episode starts at a random facing (2026-09-17 harness-smoke finding: 2/8 episodes
            # spawned at -1.40 / -0.40 rad through this event).
            rsi = getattr(events, "rsi_reference_reset", None)
            rsi_params = getattr(rsi, "params", None)
            if rsi_params is not None and "yaw_range" in rsi_params:
                rsi_params["yaw_range"] = (face0, face0)
                applied["spawn_yaw_pin_rsi"] = face0
    return applied


def _install_schedule_override(cmd_term: UniformParkourCommand, cmd_tensor: torch.Tensor) -> dict:
    """Replace ``compute`` so the fixed schedule wins over the env's own resampling.

    Mirrors ``demo_crab_hex_student.py``'s teleop override: call ``_update_metrics`` but never
    ``_update_command`` (it would overwrite ``vel_command_b[:, 2]`` from the absolute heading
    target) and never ``_resample_command``.
    """
    state = {"step": 0}
    n_steps = int(cmd_tensor.shape[0])

    def _schedule_compute(self: UniformParkourCommand, dt: float) -> None:
        idx = min(state["step"], n_steps - 1)
        self.vel_command_b[:] = cmd_tensor[idx]
        self._update_metrics()

    cmd_term.compute = types.MethodType(_schedule_compute, cmd_term)
    cmd_term.time_left[:] = 1.0e9
    # CommandTerm.reset() resamples, and the wrapper's initial reset already ran before this
    # override existed -- so seed the tensor explicitly rather than trusting the first compute().
    cmd_term.vel_command_b[:] = cmd_tensor[0]
    return state


def _install_schedule_override_world(
    cmd_term: UniformParkourCommand, compiled: S.CompiledSchedule, device, hold_labels: list[str]
) -> dict:
    """World-frame rows: drive the command term through its own ``schedule_hook`` each step.

    The world -> body conversion and the facing P-control belong to the command term (plan D1), not
    to the harness: writing ``vel_command_b`` directly, the way body rows do, would freeze the body
    command while the robot turns and would bypass the yaw-rate control entirely. So each step
    hands the term a :class:`gait_eval.schedule.ScheduleRow` -- ``.speed`` / ``.dir`` / ``.face``,
    world-frame values -- and the term writes ``vel_command_b`` itself.

    Row fields are floats whenever every env is on the same hold (``compiled.world_uniform``,
    which is every committed omni scenario); a ``rotate_holds_by_env`` scenario needs per-env
    values instead and gets ``(num_envs,)`` tensors on the env's device.

    WHEN THE WRITE HAPPENS, and why the harness does not also write per step. ``vel_command_b``
    is here a function of the LIVE YAW (the held world direction is re-projected and the facing
    P-control re-evaluated on every call), so *when* the hook runs changes the numbers. Training
    calls it exactly once per control step, from ``command_manager.compute()`` -- after the
    physics, after any reset, immediately before the observation manager reads the command. The
    ``compute`` override below sits in that same slot, so the eval writes the command at the
    training instant and the observation the policy acts on is built from that write. The main
    loop therefore does NOT re-apply the row before each step (a second write, at the *pre*-physics
    yaw, whose value the following observation would never carry); it only advances ``step`` so
    this compute applies the row of the step whose observation it is about to produce.

    Returns a state dict with ``step`` (the row index the next ``compute`` applies), ``row(t)``
    (the compiled row, for the schedule-state check) and ``apply(t)``, used once before the loop to
    overwrite the random command the wrapper's own reset left behind.
    """
    if not callable(getattr(cmd_term, "schedule_hook", None)):
        raise SystemExit(
            f"world-frame scenario needs a command term with schedule_hook(row); {type(cmd_term).__name__} "
            "has none (only the omni command term does)."
        )
    n_steps = int(compiled.n_steps)
    state = {"step": 0}
    uniform = compiled.world_uniform
    if not uniform:
        speed = torch.from_numpy(compiled.world_speed).to(device)
        direction = torch.from_numpy(compiled.world_dir).to(device)
        face = torch.from_numpy(compiled.world_face).to(device)

    def _row(step: int) -> S.ScheduleRow:
        idx = min(max(int(step), 0), n_steps - 1)
        seg = int(compiled.segment_id[idx, 0])
        label = hold_labels[seg] if 0 <= seg < len(hold_labels) else "prologue"
        if uniform:
            return compiled.row(idx, label=label)  # floats
        return S.ScheduleRow(
            speed=speed[idx],  # (num_envs,) on the env's device
            dir=direction[idx],
            face=face[idx],
            label=label,
            step=idx,
        )

    def _schedule_compute(self: UniformParkourCommand, dt: float) -> None:
        # schedule_hook re-runs the term's own _update_command, so the yaw-rate slot tracks the
        # body's CURRENT heading -- the closed loop training uses. Never _resample_command.
        self.schedule_hook(_row(state["step"]))
        self._update_metrics()

    cmd_term.compute = types.MethodType(_schedule_compute, cmd_term)
    cmd_term.time_left[:] = 1.0e9
    cmd_term.schedule_hook(_row(0))
    state["apply"] = lambda t: cmd_term.schedule_hook(_row(t))
    state["row"] = _row
    return state


def _terrain_z_under_feet(foot_pos_w: torch.Tensor, mesh, device) -> torch.Tensor:
    """Exact ground height under each footpad via a downward raycast. ``foot_pos_w`` is (E, 6, 3).

    Beats both the crude ``root_pos_w[2] - 1.0`` approximation used by ``reward_foot_clearance``
    and a height-scanner lookup: the scanner's grid is body-offset and does not cover the rear feet.
    """
    starts = foot_pos_w.clone()
    starts[..., 2] += 5.0
    dirs = torch.zeros_like(starts)
    dirs[..., 2] = -1.0
    hits = raycast_mesh(starts.reshape(-1, 3).contiguous(), dirs.reshape(-1, 3).contiguous(), mesh=mesh)[0]
    return hits.reshape(foot_pos_w.shape)[..., 2]


def _joint_groups(names) -> dict[str, list[int]]:
    """Map joint-name suffix -> column indices.

    Column order follows articulation order (``preserve_order`` is False on the action term), so the
    groups must be derived from the names rather than assumed to be contiguous blocks.
    """
    groups: dict[str, list[int]] = {"camshaft": [], "hip_femur": [], "femur_tibia": []}
    for i, name in enumerate(names):
        if "CamShaft" in name:
            groups["camshaft"].append(i)
        elif "Hip_Femur" in name:
            groups["hip_femur"].append(i)
        elif "Femur_Tibia" in name:
            groups["femur_tibia"].append(i)
    return {k: v for k, v in groups.items() if v}


def main() -> None:
    scenario = _resolve_scenario()
    # --waves / --trained-categories exist for category-suite rows only; on any other row either
    # flag is an error, so a legacy invocation's run_meta key set can never change.
    try:
        trained_categories = S.suite_cli_args(
            scenario, waves=args_cli.waves, trained_categories=args_cli.trained_categories
        )
    except S.ManifestError as exc:
        raise SystemExit(str(exc)) from exc
    suite_mode = S.uses_waves(scenario)
    started = datetime.now(timezone.utc)

    for key, value in scenario.env_vars.items():
        # Read inside the cfg's __post_init__, so this must happen before parse_env_cfg.
        # NOTE(reproducibility, 2026-09-09): writing ANY environment variable here -- i.e. after
        # Kit has started -- moves this process into a different, equally reproducible outcome
        # class for some scenario/policy pairs (measured: the depth student on slow__A15pB gives
        # 0.79/21 falls with an env block of any content and 0.75/25 without one; the obstacle
        # course and the privileged teacher were insensitive). Same-seed numbers are therefore
        # comparable only between runs with the same manifest shape; treat +-0.04 completion /
        # +-4 falls in 100 as the harness's resolution across manifests.
        import os

        os.environ[key] = value

    # NOTE(2026-09-18, the omni 1c eval defect): what the CONFIG actually saw is the whole
    # process KRABBY_* environment, not just the manifest's ``env:`` block -- the driver passes
    # most of it in the child's environment and the manifest rows that matter carry no block at
    # all, so ``run_meta["env_vars"]`` read ``{}`` for every campaign canary. The seed-3 1c gate
    # then recorded completion 0.00 / 100 falls because ``KRABBY_OMNI_FACING_OBS_CLIP`` was
    # missing from the eval process, and NOTHING in the run's own record said so: the defect was
    # only found by re-deriving the child env from the driver's source. Recording the real
    # environment here makes that class of failure legible in the artifact itself.
    # Scope: KRABBY_* only (the project's own namespace) -- never the whole environment, which
    # carries tokens, paths and machine identity into a file that gets bundled and shared.
    krabby_env = {k: v for k, v in sorted(_ENVIRON.items()) if k.startswith("KRABBY_")}

    env_cfg = parse_env_cfg(
        scenario.task,
        device=args_cli.device,
        num_envs=scenario.episodes,
        use_fabric=not args_cli.disable_fabric,
    )
    applied_overrides = _apply_env_overrides(env_cfg, scenario)

    agent_cfg = cli_args.parse_rsl_rl_cfg(scenario.task, args_cli)
    print("[DEBUG] agent_cfg parsed, calling gym.make", flush=True)
    env = ParkourRslRlVecEnvWrapper(
        gym.make(scenario.task, cfg=env_cfg, render_mode=None),
        clip_actions=agent_cfg.clip_actions,
    )
    print("[DEBUG] env created", flush=True)
    unwrapped = env.unwrapped
    device = env.device
    dt = float(unwrapped.step_dt)
    num_envs = int(unwrapped.num_envs)
    # One harness serves both the forward task (travel_yaw absent -> 0.0) and the sideways task
    # (travel_yaw == +pi/2); see gait_eval.metrics / gait_eval.report for the rotation convention.
    travel_yaw = float(getattr(unwrapped.cfg, "travel_yaw", 0.0))

    robot = unwrapped.scene["robot"]
    usd_spawned = str(getattr(getattr(robot.cfg, "spawn", None), "usd_path", "<unknown>"))
    if args_cli.plant:
        # The plant is fixed at config-import time; make sure the requested one actually spawned.
        want = PH.plant_usd_path(args_cli.plant)
        spawned_name = Path(usd_spawned).name
        if (want is None and spawned_name not in PH.MAIN_ASSET_NAMES) or (want is not None and spawned_name != Path(want).name):
            raise SystemExit(f"--plant {args_cli.plant} requested {want or 'the main asset'} but the scene spawned {usd_spawned} "
                             "(KRABBY_HEX_USD_PATH set in the environment wins over --plant; unset it)")
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    action_term = unwrapped.action_manager.get_term("joint_pos")

    # Foot order must match gait_eval.metrics.FOOT_ORDER exactly: getting it wrong silently swaps
    # the tripod sets and yields a plausible but wrong score.
    if tuple(M.FOOT_ORDER) != tuple(CRAB_HEX_FOOTPAD_BODY_NAMES):
        raise SystemExit(
            f"foot order mismatch: metrics.FOOT_ORDER={M.FOOT_ORDER} vs "
            f"CRAB_HEX_FOOTPAD_BODY_NAMES={CRAB_HEX_FOOTPAD_BODY_NAMES}"
        )
    foot_body_ids, foot_body_names = robot.find_bodies(list(CRAB_HEX_FOOTPAD_BODY_NAMES), preserve_order=True)
    if tuple(foot_body_names) != tuple(CRAB_HEX_FOOTPAD_BODY_NAMES):
        raise SystemExit(f"articulation returned feet in unexpected order: {foot_body_names}")
    sensor_foot_ids, sensor_foot_names = contact_sensor.find_bodies(
        list(CRAB_HEX_FOOTPAD_BODY_NAMES), preserve_order=True
    )
    # PLAN G (2026-09-02): leg-link net contact forces make leg-leg / leg-body interference
    # visible offline (reward_collision prices hip/femur only; tibias were invisible).
    sensor_leg_ids, sensor_leg_names = contact_sensor.find_bodies([".*_Hip", ".*_Femur", ".*_Tibia"])

    policy = None
    depth_encoder = None
    estimator = None
    is_student = False
    checkpoint_sha = None
    num_prop = num_scan = num_priv_explicit = 0

    if not args_cli.zero_actions:
        if not scenario.checkpoint:
            raise SystemExit("no checkpoint given; pass --checkpoint or use --zero-actions")
        resume_path = retrieve_file_path(scenario.checkpoint)
        checkpoint_sha = _sha256(Path(resume_path))
        if scenario.checkpoint_sha256 and checkpoint_sha != scenario.checkpoint_sha256:
            msg = (
                f"checkpoint sha256 mismatch for scenario {scenario.id!r}:\n"
                f"  manifest: {scenario.checkpoint_sha256}\n  actual:   {checkpoint_sha}\n"
                "The manifest pins the checkpoint the committed baseline was produced from."
            )
            if not args_cli.allow_checkpoint_sha_mismatch:
                raise SystemExit(msg + "\nPass --allow-checkpoint-sha-mismatch to override.")
            print(f"[WARN] {msg}", flush=True)

        runner = make_on_policy_runner(env, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=device)
        print(f"[INFO] loading checkpoint: {resume_path}", flush=True)
        runner.load(resume_path)
        # NOTE: unlike scripts/rsl_rl/play.py this deliberately does NOT export jit/onnx -- the
        # harness must never write next to a checkpoint it is measuring.
        estimator_paras = agent_cfg.to_dict()["estimator"]
        num_prop = int(estimator_paras["num_prop"])
        num_scan = int(estimator_paras["num_scan"])
        num_priv_explicit = int(estimator_paras["num_priv_explicit"])
        is_student = agent_cfg.algorithm.class_name == "DistillationWithExtractor"
        if args_cli.policy_role != "auto":
            is_student = args_cli.policy_role == "student"
            print(f"[INFO] policy role forced to {args_cli.policy_role}", flush=True)
        if is_student:
            policy = runner.get_inference_depth_policy(device=device)
            depth_encoder = runner.get_depth_encoder_inference_policy(device=device)
        else:
            policy = runner.get_inference_policy(device=device)
            estimator = runner.get_estimator_inference_policy(device=device)

    if suite_mode:
        # Category suite (plan D2): one uniform two-hold schedule [settle, hold_k] per wave, every
        # wave the same length; wave 0's is installed below exactly as a single-pass row's is.
        wave_compiled = [S.compile_wave(scenario, k, dt=dt, num_envs=num_envs) for k in range(scenario.waves)]
        compiled = wave_compiled[0]
    else:
        compiled = S.compile_schedule(scenario, dt=dt, num_envs=num_envs)
    cmd_tensor = torch.from_numpy(compiled.cmd).to(device)
    delta_yaw_tensor = (
        torch.from_numpy(compiled.delta_yaw).to(device) if compiled.delta_yaw is not None else None
    )
    T = compiled.n_steps

    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    if not isinstance(cmd_term, UniformParkourCommand):
        raise SystemExit(f"base_velocity must be UniformParkourCommand, got {type(cmd_term)}")
    world_frame = scenario.command_frame == "world"
    if world_frame:
        override_state = _install_schedule_override_world(
            cmd_term, compiled, device, compiled.hold_labels
        )
    else:
        override_state = _install_schedule_override(cmd_term, cmd_tensor)

    heading_stiffness = float(getattr(env_cfg.commands.base_velocity, "heading_control_stiffness", 0.8))
    ang_vel_clip = float(getattr(env_cfg.commands.base_velocity.clips, "ang_vel_clip", 0.4))
    ang_vel_max = float(getattr(env_cfg.commands.base_velocity, "ang_vel_max", scenario.ang_vel_max))
    observed_command_slots, observed_slots_source = _observed_command_slots(unwrapped)
    policy_sees_command_channel = _policy_sees_command_channels(observed_command_slots)

    mesh = None
    try:
        scanner = unwrapped.scene.sensors.get("height_scanner")
        mesh_key = scanner.cfg.mesh_prim_paths[0] if scanner is not None else "/World/ground"
        mesh = RayCaster.meshes[mesh_key]
    except (KeyError, AttributeError) as exc:  # pragma: no cover - depends on scene
        print(f"[WARN] terrain raycast unavailable ({exc}); falling back to root-offset ground.", flush=True)

    n_joints = int(robot.num_joints)
    n_actions = int(unwrapped.action_manager.total_action_dim)
    buf = {
        "foot_force_norm": torch.zeros(T, num_envs, 6, device=device),
        "foot_pos_w": torch.zeros(T, num_envs, 6, 3, device=device),
        "foot_lin_vel_w": torch.zeros(T, num_envs, 6, 3, device=device),
        "foot_ang_vel_w": torch.zeros(T, num_envs, 6, 3, device=device),
        "terrain_z": torch.zeros(T, num_envs, 6, device=device),
        "root_pos_w": torch.zeros(T, num_envs, 3, device=device),
        "root_quat_w": torch.zeros(T, num_envs, 4, device=device),
        "root_lin_vel_b": torch.zeros(T, num_envs, 3, device=device),
        "root_ang_vel_b": torch.zeros(T, num_envs, 3, device=device),
        "cmd_applied": torch.zeros(T, num_envs, 3, device=device),
        "obs_cmd_vx": torch.zeros(T, num_envs, device=device),
        "actions": torch.zeros(T, num_envs, n_actions, device=device),
        "joint_vel": torch.zeros(T, num_envs, n_joints, device=device),
        "applied_torque": torch.zeros(T, num_envs, n_joints, device=device),
        "done": torch.zeros(T, num_envs, dtype=torch.bool, device=device),
        "crab_failure": torch.zeros(T, num_envs, dtype=torch.bool, device=device),
        # PLAN G additions: exact whole-body CoM (with run_meta body masses), FK cross-checks,
        # and leg-link contact visibility for the morphology campaign's offline metrics.
        "joint_pos": torch.zeros(T, num_envs, n_joints, device=device),
        "body_pos_w": torch.zeros(T, num_envs, int(robot.num_bodies), 3, device=device),
        "leg_contact_force": torch.zeros(T, num_envs, len(sensor_leg_ids), device=device),
    }
    if world_frame:
        # Per-step travel frame (the direction of the body command) and facing error -- the two
        # series the omni scoring needs and a constant travel_yaw cannot express. They live in the
        # raw buffers, not run_meta: (T, E) floats are an npz column, not a JSON field.
        buf["travel_yaw"] = torch.zeros(T, num_envs, device=device)
        buf["heading_err"] = torch.zeros(T, num_envs, device=device)

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    n_steps_env = torch.zeros(num_envs, dtype=torch.long, device=device)
    term_reason = ["running"] * num_envs
    max_cmd_dev = 0.0
    # World rows only: the schedule-state check (see the loop). Body rows write the whole command
    # tensor on every compute, so "the scheduled row is the one in force" is true by construction
    # there and this stays 0.0.
    max_sched_state_dev = 0.0

    # Category-suite rows run W waves in this one process (plan D2); every other row runs ONE pass
    # of the loop below with none of the wave-only steps (no env.seed, no extra reset, no
    # re-install): configure_seed writes process state, and any env write after Kit startup can
    # flip a same-seed eval between two repeatable outcomes.
    n_waves = scenario.waves if suite_mode else 1
    wave_seeds: list[int] = []
    host_parts: list[dict] = []
    term_reason_all: list[str] = []
    n_steps_all: list[int] = []
    depth_latent = None
    yaw_pred = None
    print(f"[INFO] scenario={scenario.id} task={scenario.task} envs={num_envs} steps={T} dt={dt:.4f}", flush=True)

    for wave in range(n_waves):
        if suite_mode:
            print(f"[INFO] wave {wave + 1}/{n_waves} start ({wave_compiled[wave].hold_labels[wave + 1]})", flush=True)
            # Re-seed before EVERY wave (wave 0 included) so wave k's initial states do not depend on
            # how many envs fell in wave k-1 -- the precondition for pairing heads on env_index.
            _seed = S.wave_seed(scenario, wave)
            env.seed(_seed)
            wave_seeds.append(_seed)
            # Per-wave bookkeeping starts from scratch; a fallen env stays frozen for the rest of its
            # wave only.
            for _v in buf.values():
                _v.zero_()
            alive = torch.ones(num_envs, dtype=torch.bool, device=device)
            n_steps_env = torch.zeros(num_envs, dtype=torch.long, device=device)
            term_reason = ["running"] * num_envs
            depth_latent = None
            yaw_pred = None
        # Wave > 0 touches env state that the previous wave's inference-mode steps own, so its
        # prologue runs in inference mode too; the single pass keeps today's (no) context.
        with (torch.inference_mode() if suite_mode and wave > 0 else contextlib.nullcontext()):
            if suite_mode and wave > 0:
                env.reset()  # every env re-spawns at the pinned yaw
                compiled = wave_compiled[wave]
                override_state = _install_schedule_override_world(
                    cmd_term, compiled, device, compiled.hold_labels
                )
            obs, extras = env.get_observations()
            # The wrapper's own reset ran before the override existed, so the first observation can
            # carry a randomly resampled command. Re-apply and re-read before logging anything.
            if world_frame:
                override_state["apply"](0)
            else:
                cmd_term.vel_command_b[:] = cmd_tensor[0]
            obs, extras = env.get_observations()
            # obs at this point reflects the command just seeded above -- track it so the deviation
            # check below always compares an observation against the command that actually produced
            # it, not whatever was written *after*. Getting this backwards produces a spurious
            # "deviation" of exactly the hold-to-hold jump size at every command change, since
            # obs[:, 10] necessarily lags one step behind vel_command_b (the policy acts on last
            # step's observation).
            obs_reflects_cmd = cmd_term.vel_command_b.clone() if world_frame else cmd_tensor[0].clone()

        with torch.inference_mode():
            for t in range(T):
                override_state["step"] = t
                if world_frame:
                    # NOTHING is written here. The term already holds this step's command: the
                    # ``env.step()`` before this one ran ``command_manager.compute()`` ->
                    # ``schedule_hook(row_t)`` -> ``_update_command`` at the post-physics yaw (the
                    # instant training writes at) and the observation manager read that write to build
                    # the ``obs`` this iteration acts on. Re-applying the row here would write it a
                    # SECOND time at the pre-physics yaw of the next step, and since the world -> body
                    # projection and the facing P-control both depend on the live yaw, that second
                    # value is not the one any observation carries. It differs by one control step of
                    # yaw: ~2e-3 m/s on the xy channels, but the yaw-rate channel is DISCONTINUOUS in
                    # yaw -- the dead band flips it between 0 and +-ang_vel_clip at
                    # |facing error| = ang_vel_clip / stiffness, and ``wrap_to_pi`` flips a saturated
                    # command between +-ang_vel_max at |facing error| = pi -- so a fallen robot whose
                    # yaw swept a threshold made obs and the recorded command disagree by up to
                    # 2 * ang_vel_max (0.6). Advancing ``step`` instead keeps ONE write per step, and
                    # obs, ``cmd_applied``, ``travel_yaw``, ``heading_err`` and the deviation check
                    # below all read that single value. ``_row`` clamps, so t + 1 == T is the last row.
                    override_state["step"] = t + 1
                    obs_reflects_cmd = cmd_term.vel_command_b.clone()
                    # The scheduled row must still be the one in force: ``CommandTerm.reset()`` calls
                    # ``_resample_command`` on every env that fell, which draws a RANDOM world
                    # direction / speed / facing. ``_reset_idx`` runs before ``command_manager.compute``
                    # (parkour_manager_based_rl_env.py:142 vs :153), so ``schedule_hook`` overwrites
                    # every env right after -- this asserts it, on the three scheduled world
                    # quantities themselves, which is what a leaked resample would change (the body
                    # path gets the same guarantee for free by writing the whole command tensor).
                    _row_t = override_state["row"](t)
                    max_sched_state_dev = max(
                        max_sched_state_dev,
                        (cmd_term.travel_dir_w - _row_t.dir).abs().max().item(),
                        (cmd_term.speed_cmd - _row_t.speed).abs().max().item(),
                        _wrap_to_pi(cmd_term.heading_target - _row_t.face).abs().max().item(),
                    )
                else:
                    cmd_term.vel_command_b[:] = cmd_tensor[t]

                # Free end-to-end proof the override reaches the policy input: observations.py puts
                # commands[:, 0:1] at obs dim 10 (the omni observations put three channels at
                # OBSERVED_COMMAND_SLOTS instead). Check now, before this step's command overwrite
                # makes `obs_reflects_cmd` stale.
                for _slot, _channel in observed_command_slots.items():
                    dev = (obs[:, _slot] - obs_reflects_cmd[:, _channel]).abs().max().item()
                    max_cmd_dev = max(max_cmd_dev, dev)

                if delta_yaw_tensor is not None:
                    # wz cannot reach the policy; delta_yaw is the only steering channel it observes.
                    obs[:, OBS_IDX_DELTA_YAW] = delta_yaw_tensor[t]
                    obs[:, OBS_IDX_DELTA_NEXT_YAW] = delta_yaw_tensor[t]
                    wz_ref = torch.clamp(delta_yaw_tensor[t] * heading_stiffness, -1.0, 1.0)
                    wz_ref = wz_ref * (wz_ref.abs() > ang_vel_clip)
                    cmd_term.vel_command_b[:, 2] = wz_ref

                if args_cli.zero_actions:
                    actions = torch.zeros(num_envs, n_actions, device=device)
                elif is_student:
                    depth_camera = extras["observations"]["depth_camera"].to(device)
                    # Stay in phase with CrabHexParkourObservations' own 5-step refresh.
                    if unwrapped.common_step_counter % 5 == 0 or depth_latent is None:
                        obs_student = obs[:, :num_prop].clone()
                        obs_student[:, 6:8] = 0
                        latent_and_yaw = depth_encoder(depth_camera, obs_student)
                        depth_latent = latent_and_yaw[:, :-2]
                        yaw_pred = latent_and_yaw[:, -2:]
                    obs[:, 6:8] = 1.5 * yaw_pred
                    if delta_yaw_tensor is not None:
                        obs[:, OBS_IDX_DELTA_NEXT_YAW] = delta_yaw_tensor[t]
                    actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
                else:
                    obs[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = estimator.inference(
                        obs[:, :num_prop]
                    )
                    actions = policy(obs, hist_encoding=True)

                buf["obs_cmd_vx"][t] = obs[:, OBS_IDX_CMD_VX]
                buf["cmd_applied"][t] = cmd_term.vel_command_b
                if world_frame:
                    # Travel frame of THIS step = the direction of the body command the policy is
                    # about to act on; facing error = what the yaw-rate slot is steering to close.
                    # On a commanded stop the body command is (0, 0), atan2 gives 0.0, and the travel
                    # frame is the body frame -- the same convention a forward stop hold has.
                    buf["travel_yaw"][t] = torch.atan2(
                        cmd_term.vel_command_b[:, 1], cmd_term.vel_command_b[:, 0]
                    )
                    buf["heading_err"][t] = _wrap_to_pi(cmd_term.heading_target - robot.data.heading_w)
                buf["actions"][t] = actions
                buf["foot_force_norm"][t] = contact_sensor.data.net_forces_w_history[:, 0, sensor_foot_ids].norm(dim=-1)
                fp = robot.data.body_pos_w[:, foot_body_ids]
                buf["foot_pos_w"][t] = fp
                buf["foot_lin_vel_w"][t] = robot.data.body_lin_vel_w[:, foot_body_ids]
                buf["foot_ang_vel_w"][t] = robot.data.body_ang_vel_w[:, foot_body_ids]
                if mesh is not None:
                    buf["terrain_z"][t] = _terrain_z_under_feet(fp, mesh, device)
                else:
                    buf["terrain_z"][t] = (robot.data.root_pos_w[:, 2] - 1.05).unsqueeze(1).expand(-1, 6)
                buf["root_pos_w"][t] = robot.data.root_pos_w
                buf["root_quat_w"][t] = robot.data.root_quat_w
                buf["root_lin_vel_b"][t] = robot.data.root_lin_vel_b
                buf["root_ang_vel_b"][t] = robot.data.root_ang_vel_b
                buf["joint_vel"][t] = robot.data.joint_vel
                buf["applied_torque"][t] = robot.data.applied_torque
                buf["joint_pos"][t] = robot.data.joint_pos
                buf["body_pos_w"][t] = robot.data.body_pos_w
                buf["leg_contact_force"][t] = contact_sensor.data.net_forces_w_history[:, 0, sensor_leg_ids].norm(dim=-1)

                # The command about to be reflected in the *next* obs is whatever vel_command_b holds
                # right before this step() call (this step's base command plus any yaw override above).
                obs_reflects_cmd = cmd_term.vel_command_b.clone()
                obs, _, dones, extras = env.step(actions)

                dones_b = dones.bool().reshape(-1)
                try:
                    failed = unwrapped.termination_manager.get_term("crab_failure").bool().reshape(-1)
                except (KeyError, AttributeError):
                    failed = torch.zeros_like(dones_b)
                buf["done"][t] = dones_b
                buf["crab_failure"][t] = failed

                n_steps_env = torch.where(alive, n_steps_env + 1, n_steps_env)
                newly_done = alive & dones_b
                for env_idx in torch.nonzero(newly_done).flatten().tolist():
                    # total_terminates conflates timeout / fall / goal-reached, so classify here.
                    if bool(failed[env_idx]):
                        term_reason[env_idx] = "fall"
                    elif int(n_steps_env[env_idx]) >= int(unwrapped.max_episode_length) - 1:
                        term_reason[env_idx] = "timeout"
                    else:
                        term_reason[env_idx] = "terminated_other"
                # env.step already reset the done envs in-place, so anything read after this point is
                # the *next* episode; freeze these envs out instead of recording garbage.
                alive = alive & ~dones_b
                # A suite wave always runs its full length: common_step_counter drives the 5-step
                # observation refresh and the action-delay update and is not reset by env.reset(), so
                # an early break would shift every later wave's phase between heads and break the
                # env_index pairing. Fallen envs are already frozen out.
                if not bool(alive.any()) and not suite_mode:
                    print(f"[INFO] all envs terminated by step {t}", flush=True)
                    break

        for env_idx in range(num_envs):
            if term_reason[env_idx] == "running":
                term_reason[env_idx] = "schedule_complete"
        if suite_mode:
            _falls = sum(1 for _r in term_reason if _r == "fall")
            print(f"[INFO] wave {wave + 1}/{n_waves} end at step {t} (falls {_falls}/{num_envs})", flush=True)
            # Copied: on a CPU device .numpy() aliases the buffer the next wave zeroes.
            host_parts.append({k: v.detach().cpu().numpy().copy() for k, v in buf.items()})
            term_reason_all.extend(term_reason)
            n_steps_all.extend(int(v) for v in n_steps_env.cpu().numpy())

    if suite_mode:
        # Episode axis = wave * num_envs + env; the stacked schedule labels every episode's wave and
        # own hold, and is what scoring, write_run and travel_yaw_by_hold read.
        host = {k: np.concatenate([part[k] for part in host_parts], axis=1) for k in host_parts[0]}
        compiled = S.stack_waves(wave_compiled)
        term_reason = term_reason_all
        n_steps_list = n_steps_all
    else:
        host = {k: v.detach().cpu().numpy() for k, v in buf.items()}
        n_steps_list = [int(v) for v in n_steps_env.cpu().numpy()]
    env.close()

    out_root = Path(args_cli.output_root) if args_cli.output_root else _PARKOUR_ROOT / "logs" / "rsl_rl" / "gait_eval" / "v1"
    run_dir = out_root / scenario.id / f"seed{scenario.env_seed:03d}" / started.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "scenario_id": scenario.id,
        "task": scenario.task,
        "checkpoint": scenario.checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "zero_actions": bool(args_cli.zero_actions),
        "role": "student" if is_student else ("none" if args_cli.zero_actions else "teacher"),
        "started_utc": started.isoformat(),
        "dt": dt,
        "num_envs": num_envs,
        "n_steps_planned": T,
        "travel_yaw": travel_yaw,
        "env_overrides": applied_overrides,
        # The process's whole KRABBY_* environment as the config read it (manifest block already
        # merged in above). Every row, body-frame and world-frame alike; scoring is untouched.
        "env_vars": krabby_env,
        # ... and what the MANIFEST row itself asked for, kept separately so "the row's env block"
        # is still recoverable from the record (it used to BE ``env_vars``).
        "env_vars_manifest": dict(scenario.env_vars),
        "foot_body_names": list(foot_body_names),
        "foot_body_ids": [int(i) for i in foot_body_ids],
        "sensor_foot_ids": [int(i) for i in sensor_foot_ids],
        "action_joint_names": list(getattr(action_term, "_joint_names", [])),
        "robot_num_joints": n_joints,
        # PLAN G: exact-CoM / FK / leg-contact metadata for the raw buffers added 2026-09-02.
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "body_masses_kg": [float(m) for m in robot.root_physx_view.get_masses()[0].cpu()],
        "leg_link_names": list(sensor_leg_names),
        # the plant actually spawned (KRABBY_HEX_USD_PATH is read at config import time, so a
        # manifest env block cannot select it -- pass it in the process environment)
        "usd_path": usd_spawned,
        "usd_path_requested": scenario.env_vars.get("KRABBY_HEX_USD_PATH", _ENVIRON.get("KRABBY_HEX_USD_PATH", "<default>")),
        "plant": PH.plant_name_for_path(usd_spawned),
        "plant_requested": args_cli.plant or _ENVIRON.get("KRABBY_PLANT") or "<default>",
        "action_dim": n_actions,
        "num_prop": num_prop,
        "obs_dim_actual": int(obs.shape[1]),
        "command_override_max_deviation": max_cmd_dev,
        "policy_sees_command_channel": policy_sees_command_channel,
        "observed_command_slots": {str(k): v for k, v in observed_command_slots.items()},
        "observed_command_slots_source": observed_slots_source,
        "command_frame": scenario.command_frame,
        "terrain_source": "raycast_mesh" if mesh is not None else "root_offset_fallback",
        "termination_reason": term_reason,
        "n_steps_per_env": list(n_steps_list),
        "hold_labels": compiled.hold_labels,
        "notes": scenario.notes,
    }
    if world_frame:
        # World-row provenance: what was commanded, how it became a body command, and where the
        # per-step series live. The (T, E) travel_yaw/heading_err arrays go in the raw npz, not
        # here -- a JSON field of 3500 x E floats would dwarf every other number in the file.
        run_meta.update(
            {
                "spawn_yaw_pin": applied_overrides.get("spawn_yaw_pin"),
                "spawn_yaw_pin_rsi": applied_overrides.get("spawn_yaw_pin_rsi"),
                "hold_speeds": [h.speed for h in scenario.schedule],
                "hold_dirs": [h.dir for h in scenario.schedule],
                "hold_faces": [h.face for h in scenario.schedule],
                "speed_ellipse": dict(zip(("fwd", "lat", "back"), scenario.get_speed_ellipse())),
                "ang_vel_max": ang_vel_max,
                "ang_vel_clip": ang_vel_clip,
                "heading_control_stiffness": heading_stiffness,
                "travel_yaw_mode": "per_step",
                "schedule_state_max_deviation": max_sched_state_dev,
                "command_conversion": {
                    "hook": "cmd_term.schedule_hook(ScheduleRow(speed, dir, face))",
                    "row_fields": ["speed (m/s)", "dir (world rad)", "face (world rad)"],
                    "body_command": "written by the command term's _update_command (world->body + facing P-control)",
                    "travel_yaw_per_step": "atan2(vel_command_b[:, 1], vel_command_b[:, 0])",
                    "heading_err_per_step": "wrap_to_pi(heading_target - heading_w)",
                    "raw_series": ["travel_yaw", "heading_err"],
                    # The world->body command depends on the live yaw, so the instant matters:
                    # it is written ONCE per step, by the term's own compute inside env.step()
                    # (post-physics, pre-observation -- training's instant), and every series
                    # above plus command_override_max_deviation reads that single value. So
                    # cmd_applied[t] IS the command obs[t] carries, bit for bit.
                    "written_at": "command_manager.compute() inside env.step(), once per step",
                    "cmd_applied_instant": "the command the step's own observation was built from",
                    "schedule_state_check": (
                        "max |travel_dir_w - row.dir| / |speed_cmd - row.speed| / "
                        "|wrap(heading_target - row.face)| over every step -- proves no "
                        "reset-driven _resample_command leaked a random command into an observation"
                    ),
                    "spawn_yaw_pin_note": (
                        "every reset (including mid-schedule falls) re-spawns at the pinned spawn "
                        "yaw, not at the current hold's face, so an env that falls during a turn "
                        "hold restarts facing face0 with the hold's facing error re-opened. The "
                        "COMMAND is still the scheduled row (schedule_state_max_deviation proves "
                        "it); only the body's starting yaw is the spawn pin."
                    ),
                },
            }
        )
        ty = host["travel_yaw"]
        run_meta["travel_yaw_by_hold"] = {
            label: float(np.arctan2(
                np.sin(ty[compiled.segment_id == i]).mean(), np.cos(ty[compiled.segment_id == i]).mean()
            ))
            for i, label in enumerate(compiled.hold_labels)
            if bool((compiled.segment_id == i).any())
        }
    if suite_mode:
        # Suite rows only (a legacy row's key set is unchanged): the waves, their seeds, the
        # env_index -> (wave, env, own hold, category, hand) map and the suite's run-time geometry.
        run_meta.update(
            S.suite_run_meta(
                scenario,
                dt=dt,
                num_envs=num_envs,
                wave_seeds=wave_seeds,
                trained_categories=trained_categories,
            )
        )
    if num_prop and run_meta["obs_dim_actual"] != num_prop:
        run_meta["obs_dim_warning"] = (
            f"num_prop={num_prop} but observation width is {run_meta['obs_dim_actual']}; train and play "
            "share this slicing so it is self-consistent, but the 'scan' slice is not the height scan."
        )
    if max_cmd_dev > 1e-4:
        run_meta["command_override_warning"] = (
            f"observed command slots {dict(observed_command_slots)} deviated from the commanded "
            f"channel by {max_cmd_dev:.3e}; the schedule may not be reaching the policy."
        )
    if world_frame and max_sched_state_dev > 1e-6:
        # A non-zero value means the term's world state (dir / speed / facing) was NOT the
        # scheduled row at an observation instant -- i.e. a reset-driven _resample_command's random
        # command survived into an observation. The deviation check above cannot see that (it
        # compares the observation against the term's own command), so it is reported separately.
        run_meta["schedule_state_warning"] = (
            f"the command term's scheduled world state (dir/speed/face) deviated from the hold row "
            f"by {max_sched_state_dev:.3e}; a resampled command may have reached the policy."
        )

    groups = _joint_groups(run_meta["action_joint_names"])
    jv_groups = _joint_groups(list(robot.joint_names))

    episode_metrics = R.score_run(
        host,
        compiled=compiled,
        scenario=scenario,
        dt=dt,
        term_reason=term_reason,
        n_steps_env=list(n_steps_list),
        action_groups=groups,
        joint_vel_groups=jv_groups,
        # World rows: the (T, E) per-step command direction replaces the constant travel frame.
        travel_yaw=(host["travel_yaw"] if world_frame else travel_yaw),
    )
    if suite_mode:
        print("[INFO] write_run start", flush=True)
    R.write_run(
        run_dir,
        run_meta=run_meta,
        episode_metrics=episode_metrics,
        raw=host if args_cli.save_raw else None,
        scenario=scenario,
        compiled=compiled,
        history_root=_PARKOUR_ROOT / "logs" / "rsl_rl" / "metrics_history",
    )
    if suite_mode:
        print("[INFO] write_run done", flush=True)
    print(f"\n[INFO] wrote {run_dir}", flush=True)
    print(R.summary_text(run_meta, episode_metrics), flush=True)

    if args_cli.plot and args_cli.save_raw:
        # Post-hoc and optional: a matplotlib import error must not destroy a finished GPU run.
        try:
            import subprocess

            subprocess.run(
                [sys.executable, str(_SCRIPT_DIR / "plot_crab_hex_gait.py"), "--run-dir", str(run_dir)],
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - plotting is never fatal
            print(f"[WARN] gait diagram not rendered: {exc}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
