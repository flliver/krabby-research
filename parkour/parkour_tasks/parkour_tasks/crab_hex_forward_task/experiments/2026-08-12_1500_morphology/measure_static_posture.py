# SPDX-License-Identifier: BSD-3-Clause
"""M1+M2 morphology measurement: zero-action equilibrium posture, static per-foot load split
by tripod set, and CoM location vs foot centroid.

Three reward campaigns (tripod v1-v5, lean L-series, stride S-series) proved the +12deg forward
lean and the 0.146/0.556 tripod-set duty asymmetry are plant-side. This script measures where
the plant anchors them. Modeled on verify_crab_contact_physics.py.

Run (from anywhere):
    $ISAACLAB_PATH/isaaclab.sh -p measure_static_posture.py --headless --steps 300
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure zero-action posture, load split, CoM.")
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300, help="Zero-action settle steps (6 s at dt=0.02).")
parser.add_argument("--output", type=str, default=None)
parser.add_argument(
    "--joint_overrides",
    type=str,
    default=None,
    help='JSON dict of init_state.joint_pos overrides, e.g. \'{"MR_Femur_Tibia_RevoluteJoint": 0.05}\'.',
)

_PARKOUR_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour")
sys.path.insert(0, str(_PARKOUR_ROOT / "scripts" / "rsl_rl"))
import cli_args as _cli_args  # isort: skip

_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_PARKOUR_ROOT))
sys.path.insert(0, str(_PARKOUR_ROOT / "parkour_tasks"))

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401

# FOOT_ORDER convention: (FL, FR, ML, MR, RL, RR); A = {FL, MR, RL}, B = {FR, ML, RR}.
FOOT_NAMES = ["FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]
A_SET = (0, 3, 4)
B_SET = (1, 2, 5)


def signed_pitch_roll(quat_w: torch.Tensor) -> tuple[float, float]:
    """Signed pitch/roll (rad) from wxyz world quaternion, matching score_gait's convention."""
    w, x, y, z = quat_w.tolist()
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    overrides = json.loads(args_cli.joint_overrides) if args_cli.joint_overrides else {}
    if overrides:
        env_cfg.scene.robot.init_state.joint_pos.update(overrides)
        print(f"[INFO] joint_pos overrides applied: {overrides}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    cs = uenv.scene.sensors["contact_forces"]

    foot_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)
    foot_cfg.resolve(uenv.scene)
    foot_body_cfg = SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True)
    foot_body_cfg.resolve(uenv.scene)

    with torch.inference_mode():
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=uenv.device)
        # Track posture over the settle window to confirm equilibrium (last 50 steps averaged).
        pitches, rolls, force_hist = [], [], []
        for i in range(args_cli.steps):
            env.step(actions)
            p, r = signed_pitch_roll(robot.data.root_quat_w[0])
            pitches.append(p)
            rolls.append(r)
            f = cs.data.net_forces_w[0, foot_cfg.body_ids]
            force_hist.append(torch.norm(f, dim=-1).cpu())

    tail = 50
    pitch_eq = sum(pitches[-tail:]) / tail
    roll_eq = sum(rolls[-tail:]) / tail
    forces_eq = torch.stack(force_hist[-tail:]).mean(dim=0)  # [6]
    fa = float(forces_eq[list(A_SET)].sum())
    fb = float(forces_eq[list(B_SET)].sum())

    # CoM (world) from per-body masses and positions; compare to foot centroid.
    masses = robot.root_physx_view.get_masses()[0].to(robot.data.body_pos_w.device)  # [num_bodies]
    body_pos = robot.data.body_pos_w[0]  # [num_bodies, 3]
    com_w = (body_pos * masses.unsqueeze(1)).sum(dim=0) / masses.sum()
    foot_pos = robot.data.body_pos_w[0, foot_body_cfg.body_ids]  # [6, 3]
    centroid_w = foot_pos.mean(dim=0)
    # Longitudinal axis = body x in world; project offset onto it.
    quat = robot.data.root_quat_w[0]
    w, x, y, z = quat.tolist()
    fwd_w = torch.tensor(
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
        device=com_w.device,
    )
    offset = com_w - centroid_w
    lon_offset = float(torch.dot(offset[:2], fwd_w[:2]) / torch.norm(fwd_w[:2]))

    body_names = list(robot.body_names)
    mass_table = {body_names[i]: float(masses[i]) for i in range(len(body_names))}

    report = {
        "task": args_cli.task,
        "settle_steps": args_cli.steps,
        "pitch_eq_rad_last50": pitch_eq,
        "roll_eq_rad_last50": roll_eq,
        "pitch_trace_every25": pitches[::25],
        "per_foot_force_N": {FOOT_NAMES[i]: float(forces_eq[i]) for i in range(6)},
        "A_set_force_N": fa,
        "B_set_force_N": fb,
        "A_share": fa / max(fa + fb, 1e-9),
        "com_world": com_w.tolist(),
        "foot_centroid_world": centroid_w.tolist(),
        "com_longitudinal_offset_m": lon_offset,
        "total_mass_kg": float(masses.sum()),
        "mass_kg_by_body": mass_table,
    }
    out = Path(args_cli.output) if args_cli.output else Path(__file__).parent / "static_posture_report.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n=== M1: zero-action equilibrium (last 50 of settle window) ===")
    print(f"  pitch: {pitch_eq:+.4f} rad ({math.degrees(pitch_eq):+.2f} deg)   [policy gait: +0.209 rad]")
    print(f"  roll:  {roll_eq:+.4f} rad ({math.degrees(roll_eq):+.2f} deg)")
    print(f"  pitch trace (every 25 steps): {[round(p, 4) for p in pitches[::25]]}")
    print("=== M1b: static load split ===")
    for i, n in enumerate(FOOT_NAMES):
        print(f"  {n:12s} {float(forces_eq[i]):8.2f} N")
    print(f"  A set (FL,MR,RL): {fa:8.2f} N   B set (FR,ML,RR): {fb:8.2f} N   A share: {100 * fa / max(fa + fb, 1e-9):.1f}%")
    print("=== M2: CoM vs foot centroid ===")
    print(f"  longitudinal offset (body-forward positive): {lon_offset:+.4f} m")
    print(f"  total mass: {float(masses.sum()):.2f} kg")
    print(f"[INFO] wrote {out}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
