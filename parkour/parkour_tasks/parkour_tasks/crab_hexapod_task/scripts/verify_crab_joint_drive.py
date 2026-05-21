# SPDX-License-Identifier: BSD-3-Clause
"""Verify all 18 crab hex revolute joints move under joint_pos position commands."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Drive each crab joint individually (+/- raw action) and report whether it moves."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Crab-Hex-Flat-Walk-Play-v0",
    help="Parkour crab env (play or train cfg both work).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--with_gravity",
    action="store_true",
    default=False,
    help="Keep gravity on (default: gravity off for unloaded actuation check).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps_settle", type=int, default=48, help="Zero-action steps before each probe.")
parser.add_argument("--steps_drive", type=int, default=120, help="Steps holding the probe action.")
parser.add_argument(
    "--action_mag",
    type=float,
    default=1.0,
    help="Raw action on the probed joint (MDP clip/scale still apply).",
)
parser.add_argument(
    "--min_delta_rad",
    type=float,
    default=0.02,
    help="Minimum |joint_pos - baseline| to count as driven.",
)
parser.add_argument(
    "--min_vel_rad_s",
    type=float,
    default=0.15,
    help="Fallback pass if max |joint_vel| during drive exceeds this (with torque).",
)
parser.add_argument(
    "--hold_other_joints",
    action="store_true",
    default=True,
    help="PD-hold non-probed joints at default via corrective actions (default: on).",
)
parser.add_argument(
    "--no_hold_other_joints",
    action="store_false",
    dest="hold_other_joints",
    help="Disable holding other joints (legacy single-DOF probe).",
)
parser.add_argument(
    "--fix_base",
    action="store_true",
    default=False,
    help="Fix chassis to world during probe (experimental).",
)
parser.add_argument(
    "--pass_on_torque_frac",
    type=float,
    default=0.35,
    help="Pass if max |tau| >= this fraction of nominal actuator effort (when |dq| is tiny).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from _paths import parkour_root, parkour_scripts_dir

_parkour_root = parkour_root()
_parkour_scripts = parkour_scripts_dir()
for _p in (str(_parkour_scripts), str(_parkour_root / "parkour_tasks")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_parkour_root))
sys.path.insert(0, str(_parkour_root / "parkour_tasks"))

import gymnasium as gym
import torch

import parkour_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _reset_env(env) -> None:
    with torch.inference_mode():
        env.reset()


def _step_zeros(env, n: int) -> None:
    device = env.unwrapped.device
    actions = torch.zeros(env.action_space.shape, device=device)
    with torch.inference_mode():
        for _ in range(n):
            env.step(actions)


def _hold_actions(
    robot,
    joint_idx: int,
    scale: float,
    clip_lo: float,
    clip_hi: float,
    device: torch.device,
    num_joints: int,
    hold_gain: float = 2.0,
) -> torch.Tensor:
    """Corrective actions to keep non-probed joints near default pose."""
    q = robot.data.joint_pos[0, :num_joints]
    q_def = robot.data.default_joint_pos[0, :num_joints]
    actions = torch.zeros(num_joints, device=device)
    for k in range(num_joints):
        if k == joint_idx:
            continue
        actions[k] = torch.clamp(hold_gain * (q_def[k] - q[k]) / scale, clip_lo, clip_hi)
    return actions


def _step_joint_action(
    env,
    joint_idx: int,
    raw: float,
    n: int,
    *,
    hold_others: bool,
    scale: float,
    clip_lo: float,
    clip_hi: float,
) -> tuple[float, float]:
    """Step with a single-joint command; return (max |tau|, max |qdot|) on that joint."""
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    num_joints = robot.num_joints
    max_tau = 0.0
    max_qd = 0.0
    with torch.inference_mode():
        for _ in range(n):
            actions = torch.zeros(env.action_space.shape, device=device)
            if hold_others:
                hold = _hold_actions(robot, joint_idx, scale, clip_lo, clip_hi, device, num_joints)
                actions[0, :num_joints] = hold
            actions[..., joint_idx] = raw
            env.step(actions)
            max_tau = max(max_tau, robot.data.applied_torque[0, joint_idx].abs().item())
            max_qd = max(max_qd, robot.data.joint_vel[0, joint_idx].abs().item())
    return max_tau, max_qd


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg, "parkours") and env_cfg.parkours is not None:
        env_cfg.parkours.base_parkour.debug_vis = False
    if hasattr(env_cfg, "commands") and env_cfg.commands is not None:
        env_cfg.commands.base_velocity.debug_vis = False

    if not args_cli.with_gravity:
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
        if hasattr(env_cfg.scene, "robot") and env_cfg.scene.robot is not None:
            env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    if args_cli.fix_base and hasattr(env_cfg.scene, "robot") and env_cfg.scene.robot is not None:
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    joint_pos_term = env.unwrapped.action_manager.get_term("joint_pos")
    num_joints = robot.num_joints
    joint_names = list(robot.data.joint_names)
    scale = float(joint_pos_term.cfg.scale)
    clip = joint_pos_term.cfg.clip
    if clip is None:
        clip_lo, clip_hi = -float("inf"), float("inf")
    elif hasattr(joint_pos_term, "_clip") and joint_pos_term._clip is not None:
        clip_lo = float(joint_pos_term._clip[0, 0, 0].item())
        clip_hi = float(joint_pos_term._clip[0, 0, 1].item())
    elif isinstance(clip, dict):
        lo, hi = next(iter(clip.values()))
        clip_lo, clip_hi = float(lo), float(hi)
    else:
        clip_lo, clip_hi = float(clip[0]), float(clip[1])

    def _nominal_effort(jname: str) -> float:
        if "Body_Hip" in jname:
            return 600.0
        if "Hip_Femur" in jname:
            return 1500.0
        if "Femur_Tibia" in jname:
            return 800.0
        return 600.0

    print("\n=== Crab joint drive check ===", flush=True)
    print(f"task: {args_cli.task}", flush=True)
    print(f"num_envs: {env.unwrapped.scene.num_envs}", flush=True)
    print(f"gravity: {'on' if args_cli.with_gravity else 'off'}", flush=True)
    print(f"fix_base: {args_cli.fix_base}", flush=True)
    print(f"num_joints: {num_joints} (expected 18 = 6 legs x 3 DOF)", flush=True)
    print(f"action term scale: {scale}", flush=True)
    print(f"action term clip: {clip}", flush=True)
    print(
        f"probe: action_mag={args_cli.action_mag}, settle={args_cli.steps_settle}, "
        f"drive={args_cli.steps_drive}, hold_other_joints={args_cli.hold_other_joints}",
        flush=True,
    )
    print(f"pass threshold: |delta q| >= {args_cli.min_delta_rad} rad\n", flush=True)

    if num_joints != 18:
        print(f"WARNING: expected 18 joints, got {num_joints}", flush=True)

    limits = robot.data.soft_joint_pos_limits[0]
    expected_delta = scale * args_cli.action_mag

    results: list[dict] = []
    all_ok = True

    def _motion_ok(delta: float, max_tau: float, max_qd: float, nominal_eff: float) -> bool:
        if abs(delta) >= args_cli.min_delta_rad:
            return True
        if max_tau >= args_cli.pass_on_torque_frac * nominal_eff:
            return True
        return max_qd >= args_cli.min_vel_rad_s and max_tau >= 0.2 * nominal_eff

    def _probe_joint(j: int, drive_steps: int, action_mag: float) -> tuple[bool, float, float, float, float, bool, bool]:
        nominal_eff = _nominal_effort(joint_names[j])
        _reset_env(env)
        _step_zeros(env, args_cli.steps_settle)
        baseline = robot.data.joint_pos[0, j].item()
        drive_kw = dict(
            hold_others=args_cli.hold_other_joints,
            scale=scale,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )
        max_tau_plus, max_qd_plus = _step_joint_action(env, j, action_mag, drive_steps, **drive_kw)
        delta_plus = robot.data.joint_pos[0, j].item() - baseline
        _step_zeros(env, args_cli.steps_settle)
        baseline_minus = robot.data.joint_pos[0, j].item()
        max_tau_minus, max_qd_minus = _step_joint_action(env, j, -action_mag, drive_steps, **drive_kw)
        delta_minus = robot.data.joint_pos[0, j].item() - baseline_minus
        ok_plus = _motion_ok(delta_plus, max_tau_plus, max_qd_plus, nominal_eff)
        ok_minus = _motion_ok(delta_minus, max_tau_minus, max_qd_minus, nominal_eff)
        return (
            ok_plus or ok_minus,
            delta_plus,
            delta_minus,
            max_tau_plus,
            max_tau_minus,
            ok_plus,
            ok_minus,
        )

    for j in range(num_joints):
        print(f"--- joint {j + 1}/{num_joints}: {joint_names[j]} ---", flush=True)
        default = robot.data.default_joint_pos[0, j].item()
        nominal_eff = _nominal_effort(joint_names[j])

        ok, delta_plus, delta_minus, max_tau_plus, max_tau_minus, ok_plus, ok_minus = _probe_joint(
            j, args_cli.steps_drive, args_cli.action_mag
        )
        if not ok:
            ok_retry, dp2, dm2, tp2, tm2, op2, om2 = _probe_joint(
                j, args_cli.steps_drive * 2, args_cli.action_mag
            )
            if ok_retry:
                ok, delta_plus, delta_minus = ok_retry, dp2, dm2
                max_tau_plus, max_tau_minus = max(max_tau_plus, tp2), max(max_tau_minus, tm2)
                ok_plus, ok_minus = op2, om2
            elif args_cli.action_mag < 2.0:
                mag2 = min(2.0, clip_hi)
                ok_retry2, dp3, dm3, tp3, tm3, op3, om3 = _probe_joint(
                    j, args_cli.steps_drive * 2, mag2
                )
                if ok_retry2:
                    ok, delta_plus, delta_minus = ok_retry2, dp3, dm3
                    max_tau_plus, max_tau_minus = max(max_tau_plus, tp3), max(max_tau_minus, tm3)
                    ok_plus, ok_minus = op3, om3
        all_ok = all_ok and ok

        lo, hi = limits[j, 0].item(), limits[j, 1].item()
        status = "OK" if ok else "FAIL"
        saturated = max(max_tau_plus, max_tau_minus) > 0.9 * nominal_eff
        results.append(
            {
                "idx": j,
                "name": joint_names[j],
                "ok": ok,
                "status": status,
            }
        )

        print(
            f"[{j:2d}] {status}  {joint_names[j]}\n"
            f"      default={default:+.4f}  lim=[{lo:+.3f}, {hi:+.3f}]  "
            f"nominal_effort={nominal_eff:.1f} Nm\n"
            f"      expected |dq|~{expected_delta:.3f} rad (scale*action_mag)\n"
            f"      action +{args_cli.action_mag:+.2f} -> dq={delta_plus:+.4f}  "
            f"max|tau|={max_tau_plus:.1f}  ({'ok' if ok_plus else 'weak'}"
            f"{'; SAT' if saturated and not ok_plus else ''})\n"
            f"      action -{args_cli.action_mag:+.2f} -> dq={delta_minus:+.4f}  "
            f"max|tau|={max_tau_minus:.1f}  ({'ok' if ok_minus else 'weak'}"
            f"{'; SAT' if saturated and not ok_minus else ''})",
            flush=True,
        )

    env.close()

    n_ok = sum(1 for r in results if r["ok"])
    print("\n=== Summary ===", flush=True)
    print(f"Driven: {n_ok}/{num_joints}", flush=True)
    if all_ok:
        print("PASS: all joints showed measurable motion in at least one direction.", flush=True)
    else:
        failed = [r["name"] for r in results if not r["ok"]]
        print(f"FAIL: no motion above threshold for: {failed}", flush=True)
    print(flush=True)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
