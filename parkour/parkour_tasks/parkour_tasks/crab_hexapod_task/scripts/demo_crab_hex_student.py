# SPDX-License-Identifier: BSD-3-Clause
"""Play the crab hex **student** policy with a gamepad velocity command (Pro Controller / compatible pad).

Left stick Y: forward speed in ``base_velocity`` ``lin_vel_x`` range (center = stop).
Right stick X: yaw rate + proprio ``delta_yaw`` hint (student was trained with heading=0, not absolute heading).

Gamepad input uses ``krabby-research/controller`` (pygame SDL2), not Isaac Carb gamepad.
Velocity commands are injected in ``UniformParkourCommand.compute`` so the env cannot
resample random commands over teleop.

Launch from ``krabby-research/parkour`` (see crab_hexapod_task README §4.4 gamepad).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

from isaaclab.app import AppLauncher

_TASK_DIR = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
# scripts → crab_hexapod_task → parkour_tasks → parkour_tasks → parkour
_PARKOUR_ROOT = _SCRIPT_DIR.parents[3]
_DEFAULT_CHECKPOINT = _TASK_DIR / "runs" / "2026-05-26_22-57-01" / "model_9800.pt"
# Load cli_args without putting ``parkour/scripts`` on sys.path (shadows pip ``rsl_rl``).
_cli_args_path = _PARKOUR_ROOT / "scripts" / "rsl_rl" / "cli_args.py"
_cli_spec = importlib.util.spec_from_file_location("parkour_rsl_rl_cli_args", _cli_args_path)
if _cli_spec is None or _cli_spec.loader is None:
    raise ImportError(f"Cannot load cli_args from {_cli_args_path}")
cli_args = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(cli_args)

parser = argparse.ArgumentParser(description="Crab hex student policy + gamepad velocity command.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Crab-Hex-Student-Play-v0",
    help="Gym task (default: student play MDP with follow-cam).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (use 1 for teleop).")
parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match sim step dt.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
parser.add_argument(
    "--lin_vel_scale",
    type=float,
    default=1.0,
    help="Scale gamepad forward command within env lin_vel_x range (0–1).",
)
parser.add_argument(
    "--yaw_heading_scale",
    type=float,
    default=1.6,
    help="Right stick X → max |delta_yaw| and |omega_z| scale (rad / rad/s; default 1.6 / 1.0).",
)
parser.add_argument(
    "--max-yaw-rate",
    type=float,
    default=1.0,
    help="Right stick full deflection → |omega_z| on base_velocity (matches parkour clip).",
)
parser.add_argument(
    "--gamepad-device-id",
    type=int,
    default=None,
    help="SDL2 controller index (``python -m controller.input --list``). Default: first pad.",
)
parser.add_argument(
    "--gamepad-deadzone",
    type=float,
    default=0.05,
    help="Stick deadzone for pygame SDL2 axes (matches controller package).",
)
parser.add_argument(
    "--debug-gamepad",
    action="store_true",
    help="Print stick axes and commanded vx every ~2 s.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    args_cli.checkpoint = str(_DEFAULT_CHECKPOINT)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

# Isaac / running from ``parkour/scripts`` can add paths that shadow pip ``rsl_rl`` (see zero_agent.py).
_parkour_scripts = _PARKOUR_ROOT / "scripts"
_KRABBY_RESEARCH = _PARKOUR_ROOT.parent
for _p in (
    str(_parkour_scripts),
    str(_parkour_scripts / "rsl_rl"),
    str(_PARKOUR_ROOT / "parkour_tasks"),
    str(_SCRIPT_DIR),
):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_KRABBY_RESEARCH))
sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import pygame  # noqa: E402
import pygame._sdl2.controller as sdl2_controller  # noqa: E402

import parkour_tasks  # noqa: F401,E402
from controller.input import InputController  # noqa: E402
from controller.input.state import ControllerState  # noqa: E402
from controller.mappers.gamepad_to_parkour_velocity import (  # noqa: E402
    apply_teleop_state_to_parkour_command,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from parkour_isaaclab.envs import ParkourManagerBasedRLEnv  # noqa: E402
from parkour_isaaclab.envs.mdp.parkour_commands.uniform_parkour_command import (  # noqa: E402
    UniformParkourCommand,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (  # noqa: E402
    ParkourRslRlOnPolicyRunnerCfg,
)
from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner  # noqa: E402
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402


def _lin_vel_x_range(env_cfg) -> tuple[float, float]:
    return tuple(env_cfg.commands.base_velocity.ranges.lin_vel_x)


def _install_teleop_command_override(
    cmd_term: UniformParkourCommand,
    *,
    lin_range: tuple[float, float],
    lin_vel_scale: float,
    yaw_heading_scale: float,
    max_yaw_rate: float,
    deadzone: float,
    lin_vel_clip: float,
    ang_vel_clip: float,
) -> None:
    """Replace ``compute`` so teleop wins over env resampling (must run before ``env.step``)."""
    cmd_term._teleop_pad_state = ControllerState()  # type: ignore[attr-defined]
    cmd_term._teleop_lin_range = lin_range  # type: ignore[attr-defined]
    cmd_term._teleop_lin_vel_scale = lin_vel_scale  # type: ignore[attr-defined]
    cmd_term._teleop_yaw_heading_scale = yaw_heading_scale  # type: ignore[attr-defined]
    cmd_term._teleop_max_yaw_rate = max_yaw_rate  # type: ignore[attr-defined]
    cmd_term._teleop_deadzone = deadzone  # type: ignore[attr-defined]
    cmd_term._teleop_lin_vel_clip = lin_vel_clip  # type: ignore[attr-defined]
    cmd_term._teleop_ang_vel_clip = ang_vel_clip  # type: ignore[attr-defined]
    cmd_term._teleop_delta_yaw = 0.0  # type: ignore[attr-defined]

    def _teleop_compute(self: UniformParkourCommand, dt: float) -> None:
        delta_yaw, _omega_z = apply_teleop_state_to_parkour_command(
            self,
            self._teleop_pad_state,
            self._teleop_lin_range,
            lin_vel_scale=self._teleop_lin_vel_scale,
            yaw_heading_scale=self._teleop_yaw_heading_scale,
            max_yaw_rate=self._teleop_max_yaw_rate,
            deadzone=self._teleop_deadzone,
            lin_vel_clip=self._teleop_lin_vel_clip,
            ang_vel_clip=self._teleop_ang_vel_clip,
        )
        self._teleop_delta_yaw = delta_yaw  # type: ignore[attr-defined]
        self._update_metrics()
        # Do not call _update_command: it overwrites omega_z from absolute heading_target,
        # and the student policy never saw heading commands during training.

    cmd_term.compute = types.MethodType(_teleop_compute, cmd_term)


def _init_pygame_main_thread() -> None:
    """Initialize pygame on Isaac's main thread (required for event.pump() axis updates)."""
    if not pygame.get_init():
        pygame.init()
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if not sdl2_controller.get_init():
        sdl2_controller.init()


def _pump_pygame_events(last_pump_time: list[float]) -> None:
    """Pump SDL events on the main thread so joystick axes update inside Isaac Sim."""
    if not pygame.get_init():
        return
    now = time.time()
    if now - last_pump_time[0] < 0.016:  # ~60 Hz max
        return
    pygame.event.pump()
    last_pump_time[0] = now


def _print_gamepad_device(input_controller: InputController) -> None:
    controller = getattr(input_controller, "_controller", None)
    if controller is None:
        print("[WARN] SDL2 controller handle missing; sticks may stay at zero.")
        return
    try:
        name = controller.as_joystick().get_name()
    except Exception:
        name = "unknown"
    print(f"[INFO] SDL2 gamepad: device_id={input_controller._device_id} name={name!r}")


def _ensure_gamepad_ready(input_controller: InputController) -> None:
    """Fail fast if pygame SDL2 did not open a controller."""
    time.sleep(0.3)
    if not input_controller._running:
        raise RuntimeError(
            "InputController did not start (no SDL2 gamepad). Run "
            "`python -m controller.input --list`, connect the Pro Controller, "
            "quit other clients (krabby-uno, Steam), retry with `--gamepad-device-id`."
        )


def main() -> None:
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = retrieve_file_path(args_cli.checkpoint)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = 1.0e6
    env_cfg.curriculum = None
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

    env = ParkourRslRlVecEnvWrapper(
        gym.make(args_cli.task, cfg=env_cfg, render_mode=None),
        clip_actions=agent_cfg.clip_actions,
    )
    device = env.device
    lin_range = _lin_vel_x_range(env_cfg)
    lin_vel_clip = float(env_cfg.commands.base_velocity.clips.lin_vel_clip)
    ang_vel_clip = float(env_cfg.commands.base_velocity.clips.ang_vel_clip)

    runner = make_on_policy_runner(env, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=device)
    runner.load(resume_path)

    is_student = agent_cfg.algorithm.class_name == "DistillationWithExtractor"
    if is_student:
        policy = runner.get_inference_depth_policy(device=device)
        depth_encoder = runner.get_depth_encoder_inference_policy(device=device)
    else:
        raise RuntimeError(
            f"Expected DistillationWithExtractor for {args_cli.task!r}, got {agent_cfg.algorithm.class_name!r}"
        )

    unwrapped: ParkourManagerBasedRLEnv = env.unwrapped
    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    if not isinstance(cmd_term, UniformParkourCommand):
        raise TypeError(f"base_velocity must be UniformParkourCommand, got {type(cmd_term)}")

    _install_teleop_command_override(
        cmd_term,
        lin_range=lin_range,
        lin_vel_scale=args_cli.lin_vel_scale,
        yaw_heading_scale=args_cli.yaw_heading_scale,
        max_yaw_rate=args_cli.max_yaw_rate,
        deadzone=args_cli.gamepad_deadzone,
        lin_vel_clip=lin_vel_clip,
        ang_vel_clip=ang_vel_clip,
    )
    cmd_term.time_left[:] = 1.0e9

    _init_pygame_main_thread()
    input_controller = InputController.get_instance()
    input_controller.start(
        device_id=args_cli.gamepad_device_id,
        update_rate_hz=50.0,
    )
    time.sleep(0.2)
    _ensure_gamepad_ready(input_controller)
    _print_gamepad_device(input_controller)

    num_prop = agent_cfg.estimator.num_prop
    depth_latent = None
    yaw = torch.zeros(env.num_envs, 2, device=device)

    print(f"[INFO] Loaded student checkpoint: {resume_path}")
    print(f"[INFO] lin_vel_x range: {lin_range} (clip below {lin_vel_clip} → stop)")
    print(
        "[INFO] Teleop mode: command_manager will NOT resample velocity; only gamepad drives vx/heading."
    )
    print(
        "[INFO] Gamepad: controller.input (pygame SDL2). "
        "List devices: python -m controller.input --list"
    )
    print(
        "[INFO] Controls: left stick Y = forward (center = stop), "
        "right stick X = yaw rate / delta_yaw hint."
    )

    obs, extras = env.get_observations()
    dt = unwrapped.step_dt
    step_i = 0
    last_debug_t = time.time()
    last_pump_t = [0.0]

    try:
        while simulation_app.is_running():
            start = time.time()
            _pump_pygame_events(last_pump_t)
            pad_state = input_controller.get_state()
            cmd_term._teleop_pad_state = pad_state  # type: ignore[attr-defined]

            teleop_delta_yaw = float(cmd_term._teleop_delta_yaw)  # type: ignore[attr-defined]

            if args_cli.debug_gamepad and (time.time() - last_debug_t) > 2.0:
                vx_cmd = float(cmd_term.vel_command_b[0, 0].item())
                omega_z = float(cmd_term.vel_command_b[0, 2].item())
                print(
                    f"[gamepad] LY={pad_state.LY:+.2f} RX={pad_state.RX:+.2f} "
                    f"→ vx={vx_cmd:.3f} omega_z={omega_z:+.3f} delta_yaw_obs={teleop_delta_yaw:+.3f}"
                )
                last_debug_t = time.time()

            with torch.inference_mode():
                depth_camera = extras["observations"]["depth_camera"].to(device)
                if step_i % 5 == 0:
                    obs_student = obs[:, :num_prop].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5 * yaw
                if abs(teleop_delta_yaw) > args_cli.gamepad_deadzone:
                    obs[:, 7] = teleop_delta_yaw
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)

            obs, _, _, extras = env.step(actions)
            step_i += 1

            if args_cli.real_time:
                sleep_s = dt - (time.time() - start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        input_controller.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
