# SPDX-License-Identifier: BSD-3-Clause
"""Verify the crab-hex cam-mechanism kinematic coupling: for each converted leg's
``*_Body_CamShaft_RevoluteJoint`` -> ``*_Body_Hip_RevoluteJoint`` pair, drive the shaft to a
few different target angles via normal policy-style actions, let it settle under real PD
dynamics, then assert the hip joint's settled position tracks
``crab_hex_cam_mapping.cam_shaft_to_hip(shaft's settled position)`` within a reasonable
tolerance.

NOTE(cam-mechanism-migration): an earlier version of this script forced the shaft's state
directly via ``Articulation.write_joint_state_to_sim`` and checked the hip joint's response
immediately (no physics stepping), expecting a near-exact match. That matched the coupling
code's *old* design (direct state teleportation each substep), which was abandoned after it
was found to freeze the whole articulation's dynamics -- see parkour_actions.py's docstring.
The coupling now works via ``set_joint_position_target`` (real PD tracking, like every other
joint), so this script settles via ``env.step()`` and checks with a real-PD-appropriate
tolerance instead of near-zero.

Also checks the post-reset default pose is self-consistent: hip's default_joint_pos should
be close to cam_shaft_to_hip(shaft's default_joint_pos), since crab_hex_scene_cfg.py's
init_state.joint_pos values were chosen to satisfy this (allowing for settling under gravity).
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Sweep each converted leg's cam-shaft joint and verify the passive hip joint tracks the mapping."
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
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--raw_actions",
    type=float,
    nargs="+",
    default=[-1.0, -0.5, 0.5, 1.0],
    help="Raw CamShaft VELOCITY actions to test (clipped to [-1,1]; omega = raw * CAM_VEL_SCALE rad/s).",
)
parser.add_argument("--settle_steps", type=int, default=100, help="Steps to hold each action before sampling.")
parser.add_argument(
    "--revolutions",
    type=float,
    default=2.0,
    help="Full shaft revolutions to spin per raw action while checking hip tracking pointwise.",
)
parser.add_argument(
    "--pos_tol_rad",
    type=float,
    default=0.08,
    help="Max allowed |hip_pos - expected(shaft_actual)| sampled during the spin (real PD tracking, not exact).",
)
parser.add_argument(
    "--default_pos_tol_rad",
    type=float,
    default=0.05,
    help="Max allowed default-pose inconsistency (settled under gravity, not an exact write).",
)
parser.add_argument(
    "--debug_frames",
    action="store_true",
    default=False,
    help="Print per-frame shaft/hip telemetry (every 10th frame, all legs' min/max) during spins.",
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

from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_cam_mapping import cam_shaft_to_hip


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

    env = gym.make(args_cli.task, cfg=env_cfg)
    with torch.inference_mode():
        env.reset()

    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    device = env.unwrapped.device

    shaft_ids, shaft_names = robot.find_joints([".*_Body_CamShaft_RevoluteJoint"], preserve_order=True)
    hip_names = [name.replace("_Body_CamShaft_RevoluteJoint", "_Body_Hip_RevoluteJoint") for name in shaft_names]
    hip_ids, _ = robot.find_joints(hip_names, preserve_order=True)

    # Resolve each shaft joint's position within the action vector (order matches action_term's
    # own joint resolution, found by intersecting its _joint_ids with our shaft_ids).
    action_joint_ids = list(action_term._joint_ids)
    shaft_action_idx = [action_joint_ids.index(jid) for jid in shaft_ids]

    print("\n=== Crab cam-mechanism coupling check ===", flush=True)
    print(f"task: {args_cli.task}", flush=True)
    print(f"num_envs: {env.unwrapped.scene.num_envs}", flush=True)
    print(f"converted legs (cam-shaft joints found): {shaft_names}", flush=True)
    print(f"corresponding hip joints: {hip_names}", flush=True)
    if not shaft_names:
        print("FAIL: no *_Body_CamShaft_RevoluteJoint found -- has the prototype been built?", flush=True)
        env.close()
        sys.exit(1)

    all_ok = True

    def _settle(n: int) -> None:
        zero_actions = torch.zeros(env.action_space.shape, device=device)
        with torch.inference_mode():
            for _ in range(n):
                env.step(zero_actions)

    def _check(label: str, tol: float) -> tuple[float, list]:
        shaft_actual = robot.data.joint_pos[:, shaft_ids]
        hip_actual = robot.data.joint_pos[:, hip_ids]
        zero_vel = torch.zeros_like(shaft_actual)
        expected_hip, _ = cam_shaft_to_hip(shaft_actual, zero_vel)
        err = (hip_actual - expected_hip).abs()
        max_err = err.max().item()
        ok = max_err <= tol
        rows = []
        for i, name in enumerate(shaft_names):
            rows.append(
                f"[{name}] shaft={shaft_actual[0, i].item():+.4f}  hip={hip_actual[0, i].item():+.4f}  "
                f"expected={expected_hip[0, i].item():+.4f}  err={err[0, i].item():.4f}"
            )
        print(f"\n--- {label} (tol {tol}) ---", flush=True)
        for row in rows:
            print(row, flush=True)
        print(f"max err = {max_err:.4f}  ({'ok' if ok else 'FAIL'})", flush=True)
        return max_err, [ok]

    # --- 1. Default pose settled under gravity ---
    _settle(args_cli.settle_steps)
    default_err, default_oks = _check("post-settle default pose", args_cli.default_pos_tol_rad)
    all_ok = all_ok and all(default_oks)

    # --- 2. Spin the CamShaft continuously (velocity actions) and check hip tracking
    #        pointwise through >= args_cli.revolutions full revolutions each direction ---
    from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import CAM_VEL_SCALE
    from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_cam_mapping import THETA_HIP_MAX

    # Body_Hip hard limit, read LIVE from the articulation (was a stale 32-deg constant
    # that let a hard-stop contact pass silently after the 2026-08-20 limit change to 28).
    body_hip_ids, _ = robot.find_joints([".*_Body_Hip_RevoluteJoint"], preserve_order=True)
    hard_limit_rad = float(robot.data.joint_pos_limits[0, body_hip_ids, 1].max().item())
    env_dt = float(env.unwrapped.step_dt)
    max_pos_err = default_err
    max_hip_abs = 0.0
    n_checked = 0
    n_failed = 0
    for raw in args_cli.raw_actions:
        omega_cmd = raw * CAM_VEL_SCALE
        n_steps = max(1, int(math.ceil(args_cli.revolutions * 2.0 * math.pi / (abs(omega_cmd) * env_dt))))
        actions = torch.zeros(env.action_space.shape, device=device)
        for idx in shaft_action_idx:
            actions[:, idx] = raw
        spin_err = 0.0
        spin_hip_abs = 0.0
        n_resets = 0
        shaft_travel = torch.zeros(len(shaft_ids), device=device)
        prev_shaft = robot.data.joint_pos[:, shaft_ids].clone()
        prev_ep_len = int(env.unwrapped.episode_length_buf[0].item())
        with torch.inference_mode():
            for step_i in range(n_steps):
                env.step(actions)
                if args_cli.debug_frames and step_i % 10 == 0:
                    sv = robot.data.joint_vel[0, shaft_ids]
                    st = robot.data.applied_torque[0, shaft_ids]
                    svt = robot.data.joint_vel_target[0, shaft_ids]
                    hv = robot.data.joint_vel[0, hip_ids]
                    ht = robot.data.applied_torque[0, hip_ids]
                    print(
                        f"  f{step_i:3d}: shaft_v=[{sv.min().item():+.2f},{sv.max().item():+.2f}]"
                        f" v_tgt=[{svt.min().item():+.2f},{svt.max().item():+.2f}]"
                        f" shaft_tau=[{st.min().item():+.2f},{st.max().item():+.2f}]"
                        f" hip_v=[{hv.min().item():+.2f},{hv.max().item():+.2f}]"
                        f" hip_tau=[{ht.min().item():+.1f},{ht.max().item():+.1f}]",
                        flush=True,
                    )
                ep_len = int(env.unwrapped.episode_length_buf[0].item())
                shaft_actual = robot.data.joint_pos[:, shaft_ids]
                if ep_len <= prev_ep_len:
                    # env terminated and reset: joint state teleported — skip this frame for
                    # error/travel accounting (the linkage itself did nothing wrong).
                    n_resets += 1
                    prev_shaft = shaft_actual.clone()
                    prev_ep_len = ep_len
                    continue
                prev_ep_len = ep_len
                hip_actual = robot.data.joint_pos[:, hip_ids]
                expected_hip, _ = cam_shaft_to_hip(shaft_actual, torch.zeros_like(shaft_actual))
                spin_err = max(spin_err, (hip_actual - expected_hip).abs().max().item())
                spin_hip_abs = max(spin_hip_abs, hip_actual.abs().max().item())
                # PhysX wraps the reported revolute angle (observed at +-2pi); unwrap the
                # per-frame delta so travel isn't credited a phantom full turn at the seam
                d_raw = (shaft_actual - prev_shaft)[0]
                d_unwrapped = torch.atan2(torch.sin(d_raw), torch.cos(d_raw))
                shaft_travel += d_unwrapped.abs()
                prev_shaft = shaft_actual.clone()
        revs = (shaft_travel / (2.0 * math.pi)).min().item()
        # pro-rate the revolutions target when env resets ate frames (~15 frames each for
        # the teleport + re-acceleration); floor at half the nominal target
        usable_frac = max(0.0, (n_steps - 15 * n_resets) / n_steps)
        required_revs = max(args_cli.revolutions * 0.5, args_cli.revolutions * 0.9 * usable_frac)
        ok = spin_err <= args_cli.pos_tol_rad and revs >= required_revs
        print(
            f"\n--- spin raw={raw:+.2f} ({omega_cmd:+.1f} rad/s, {n_steps} steps, "
            f"{n_resets} env resets) ---\n"
            f"min revolutions completed = {revs:.2f} (target {args_cli.revolutions})\n"
            f"max pointwise |hip err| = {spin_err:.4f} rad  ({'ok' if ok else 'FAIL'})\n"
            f"max |hip| = {spin_hip_abs:.4f} rad (mechanism {THETA_HIP_MAX:.4f}, hard limit {hard_limit_rad:.4f})",
            flush=True,
        )
        n_checked += 1
        max_pos_err = max(max_pos_err, spin_err)
        max_hip_abs = max(max_hip_abs, spin_hip_abs)
        if not ok:
            n_failed += 1

    limit_ok = max_hip_abs < hard_limit_rad - 1e-3
    if not limit_ok:
        print(f"FAIL: hip reached the {math.degrees(hard_limit_rad):.0f} deg hard limit", flush=True)
    sweep_ok = n_failed == 0 and limit_ok
    all_ok = all_ok and sweep_ok
    print(f"\nchecked {n_checked} spin commands: {args_cli.raw_actions}", flush=True)
    print(f"max |hip_pos err| across all checks = {max_pos_err:.4f} rad", flush=True)
    print(f"max |hip| across spins = {max_hip_abs:.4f} rad", flush=True)
    print(f"failed: {n_failed}/{n_checked}", flush=True)

    env.close()

    print("\n=== Summary ===", flush=True)
    if all_ok:
        print(
            "PASS: default pose consistent; hip tracks cam_shaft_to_hip() pointwise through "
            "continuous multi-revolution spin in both directions without touching the hard limit.",
            flush=True,
        )
    else:
        print("FAIL: see details above.", flush=True)
    print(flush=True)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
