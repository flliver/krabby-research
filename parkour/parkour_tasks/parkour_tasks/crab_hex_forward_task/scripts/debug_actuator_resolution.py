# SPDX-License-Identifier: BSD-3-Clause
"""One-off diagnostic: print resolved actuator groups and drive FL knee directly.

Bypasses the action pipeline (set_joint_position_target straight on the articulation)
to separate 'actuator misconfigured' from 'action term broken'.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
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


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # gravity stays ON: zero-g floats the robot into per-step terminations that silently
    # reset all joint state (see memory: isaac-headless-launch-quirks)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]

    print("\n=== Resolved actuator groups ===", flush=True)
    for group_name, act in robot.actuators.items():
        names = act.joint_names
        stiff = act.stiffness[0].tolist()
        damp = act.damping[0].tolist()
        eff = act.effort_limit[0].tolist()
        vel = act.velocity_limit[0].tolist() if hasattr(act, "velocity_limit") else None
        idx = getattr(act, "joint_indices", getattr(act, "_joint_indices", None))
        print(f"[{group_name}] type={type(act).__name__} joint_indices={idx}", flush=True)
        for i, n in enumerate(names):
            v = f" vel_lim={vel[i]:.1f}" if vel is not None else ""
            print(
                f"    {n}: k={stiff[i]:.2f} d={damp[i]:.3f} effort_lim={eff[i]:.1f}{v}",
                flush=True,
            )

    print("\n=== Sim-side (PhysX) drive gains per DOF ===", flush=True)
    sim_k = robot.root_physx_view.get_dof_stiffnesses()[0]
    sim_d = robot.root_physx_view.get_dof_dampings()[0]
    for i, n in enumerate(robot.data.joint_names):
        print(f"    dof {i:2d} {n}: sim_k={sim_k[i].item():g} sim_d={sim_d[i].item():g}", flush=True)

    joint_names = list(robot.data.joint_names)
    cam_id = joint_names.index("FL_Body_CamShaft_RevoluteJoint")
    term = env.unwrapped.action_manager.get_term("joint_pos")
    cam_col = list(term._joint_names).index("FL_Body_CamShaft_RevoluteJoint")
    yaw_act = robot.actuators["body_hip_yaw"]
    print(f"\n=== Drive FL camshaft via action col {cam_col} (dof {cam_id}) ===", flush=True)
    with torch.inference_mode():
        env.reset()
        zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(48):
            env.step(zero)
        print(f"start theta={robot.data.joint_pos[0, cam_id].item():+.4f}", flush=True)
        act = zero.clone()
        act[..., cam_col] = 1.0
        for step in range(80):
            env.step(act)
            if step % 8 == 0 or step == 79:
                q = robot.data.joint_pos[0, cam_id].item()
                qd = robot.data.joint_vel[0, cam_id].item()
                vt = robot.data.joint_vel_target[0, cam_id].item()
                pt = robot.data.joint_pos_target[0, cam_id].item()
                tau = robot.data.applied_torque[0, cam_id].item()
                # actuator-internal view (group-local slice 0 = FL)
                comp = yaw_act.computed_effort[0, 0].item()
                appl = yaw_act.applied_effort[0, 0].item()
                print(
                    f"  step {step:3d}: theta={q:+.3f} omega={qd:+.3f} v_tgt={vt:+.2f} "
                    f"pos_tgt={pt:+.3f} data_tau={tau:+.2f} act_computed={comp:+.2f} "
                    f"act_applied={appl:+.2f}",
                    flush=True,
                )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
