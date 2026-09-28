# SPDX-License-Identifier: BSD-3-Clause
"""M4 solver: find per-leg knee defaults that equalize static per-foot forces.

Runs one sim session; each candidate overwrites the PD default targets
(``robot.data.default_joint_pos``) so zero-action stepping settles onto the candidate pose
(~150 steps, ~5 s). Damped finite-difference Newton on the 6 knee angles, objective =
per-foot force deviation from mean + attitude penalty. Deterministic sim -> clean Jacobians.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Solve knee defaults for equal static foot loads.")
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle", type=int, default=150)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--fd_eps", type=float, default=0.004)
parser.add_argument("--damping", type=float, default=0.6)

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

FOOT_NAMES = ["FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad"]
KNEE_JOINTS = [
    "FL_Femur_Tibia_RevoluteJoint", "FR_Femur_Tibia_RevoluteJoint",
    "ML_Femur_Tibia_RevoluteJoint", "MR_Femur_Tibia_RevoluteJoint",
    "RL_Femur_Tibia_RevoluteJoint", "RR_Femur_Tibia_RevoluteJoint",
]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    cs = uenv.scene.sensors["contact_forces"]
    foot_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)
    foot_cfg.resolve(uenv.scene)
    knee_ids = [robot.joint_names.index(j) for j in KNEE_JOINTS]

    base_defaults = robot.data.default_joint_pos.clone()
    actions = torch.zeros(env.action_space.shape, device=uenv.device)

    def evaluate(knees: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Set candidate knee defaults, settle, return (per-foot forces [6], pitch, roll)."""
        with torch.inference_mode():
            robot.data.default_joint_pos[:] = base_defaults
            robot.data.default_joint_pos[:, knee_ids] = knees
            env.reset()
            # reset() re-samples from init_state cfg; re-apply candidate as PD target AND state
            robot.data.default_joint_pos[:, knee_ids] = knees
            jp = robot.data.joint_pos.clone()
            jp[:, knee_ids] = knees
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            fh = []
            for i in range(args_cli.settle):
                env.step(actions)
                if i >= args_cli.settle - 30:
                    f = cs.data.net_forces_w[0, foot_cfg.body_ids]
                    fh.append(torch.norm(f, dim=-1).cpu())
        forces = torch.stack(fh).mean(dim=0)
        w, x, y, z = robot.data.root_quat_w[0].tolist()
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        return forces, pitch, roll

    def residual(forces: torch.Tensor) -> torch.Tensor:
        return forces - forces.mean()

    knees = base_defaults[0, knee_ids].clone().cpu()
    history = []
    for rnd in range(args_cli.rounds):
        f0, p0, r0 = evaluate(knees.to(uenv.device))
        res0 = residual(f0)
        history.append({
            "round": rnd,
            "knees": {KNEE_JOINTS[i]: float(knees[i]) for i in range(6)},
            "forces": {FOOT_NAMES[i]: float(f0[i]) for i in range(6)},
            "spread_N": float(f0.max() - f0.min()),
            "rms_dev_N": float(res0.pow(2).mean().sqrt()),
            "pitch_rad": p0, "roll_rad": r0,
        })
        print(f"[round {rnd}] forces={[round(float(v),1) for v in f0]} "
              f"rms_dev={float(res0.pow(2).mean().sqrt()):.1f}N spread={float(f0.max()-f0.min()):.1f}N "
              f"pitch={p0:+.4f} roll={r0:+.4f}", flush=True)
        if float(res0.pow(2).mean().sqrt()) < 12.0:
            print("[done] rms deviation under 12 N — converged", flush=True)
            break
        # finite-difference Jacobian dF/dknee [6x6]
        J = torch.zeros(6, 6)
        for k in range(6):
            kp = knees.clone(); kp[k] += args_cli.fd_eps
            fk, _, _ = evaluate(kp.to(uenv.device))
            J[:, k] = (fk - f0) / args_cli.fd_eps
        # damped least-squares step toward zero residual
        lam = 0.1 * float(J.norm())
        step = torch.linalg.lstsq(
            J.T @ J + lam * torch.eye(6), -J.T @ res0.unsqueeze(1)
        ).solution.squeeze(1)
        step = step.clamp(-0.03, 0.03) * args_cli.damping
        knees = knees + step
        print(f"  step: {[round(float(s),4) for s in step]}", flush=True)

    f_final, p_final, r_final = evaluate(knees.to(uenv.device))
    out = {
        "solved_knee_defaults": {KNEE_JOINTS[i]: round(float(knees[i]), 4) for i in range(6)},
        "final_forces_N": {FOOT_NAMES[i]: float(f_final[i]) for i in range(6)},
        "final_rms_dev_N": float(residual(f_final).pow(2).mean().sqrt()),
        "final_spread_N": float(f_final.max() - f_final.min()),
        "final_pitch_rad": p_final,
        "final_roll_rad": r_final,
        "history": history,
    }
    Path(__file__).with_name("solved_knee_defaults.json").write_text(json.dumps(out, indent=2))
    print(f"[RESULT] {json.dumps(out['solved_knee_defaults'], indent=2)}")
    print(f"[RESULT] final forces {[round(float(v),1) for v in f_final]} "
          f"rms {out['final_rms_dev_N']:.1f}N pitch {p_final:+.4f} roll {r_final:+.4f}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
