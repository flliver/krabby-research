# SPDX-License-Identifier: BSD-3-Clause
"""Audit crab hex contact bodies (tibia filter), link masses, and friction at runtime."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify crab contact bodies, mass distribution, friction.")
parser.add_argument("--task", type=str, default="Isaac-Crab-Hex-Flat-Walk-Play-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=80, help="Zero-action steps before sampling contacts.")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="JSON report path (default: logs/.../diagnostics/contact_physics_audit.json).",
)
from _paths import parkour_root, parkour_scripts_dir

_parkour_root = parkour_root()
_parkour_scripts = parkour_scripts_dir()
_rsl_rl_dir = _parkour_scripts / "rsl_rl"
sys.path.insert(0, str(_rsl_rl_dir))
import cli_args as _cli_args  # isort: skip

_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

for _p in (str(_parkour_scripts), str(_parkour_root / "parkour_tasks")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_parkour_root))
sys.path.insert(0, str(_parkour_root / "parkour_tasks"))

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401
from parkour_tasks.crab_hexapod_task.mdp.crab_contact_sensors import CRAB_HEX_FOOTPAD_BODY_NAMES


def _resolve_footpad_cfg(env) -> SceneEntityCfg:
    cfg = SceneEntityCfg("contact_forces", body_names=".*_Footpad")
    cfg.resolve(env.scene)
    return cfg


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    cs = uenv.scene.sensors["contact_forces"]

    with torch.inference_mode():
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=uenv.device)
        for _ in range(args_cli.steps):
            env.step(actions)

    footpad_cfg = _resolve_footpad_cfg(uenv)
    all_names = list(cs.body_names)
    footpad_names = [all_names[i] for i in footpad_cfg.body_ids]
    link_paths = robot.root_physx_view.link_paths[0]

    masses = robot.root_physx_view.get_masses()[0].detach().cpu().tolist()
    body_names_art = list(robot.body_names)
    mass_by_body = {body_names_art[i]: masses[i] for i in range(len(body_names_art))}

    mats = robot.root_physx_view.get_material_properties()[0].detach().cpu()
    # shape: [num_shapes, 3] -> static friction, dynamic friction, restitution (typical)

    terrain = uenv.scene.terrain
    terrain_mat = None
    if terrain is not None and hasattr(terrain, "cfg") and terrain.cfg.physics_material is not None:
        pm = terrain.cfg.physics_material
        terrain_mat = {
            "static_friction": pm.static_friction,
            "dynamic_friction": pm.dynamic_friction,
            "restitution": getattr(pm, "restitution", None),
            "combine_mode": getattr(pm, "friction_combine_mode", None),
        }

    forces = cs.data.net_forces_w[0]
    in_contact_all = (torch.norm(forces, dim=-1) > 1.0).detach().cpu().tolist()
    footpad_forces = forces[footpad_cfg.body_ids]
    in_contact_footpad = (torch.norm(footpad_forces, dim=-1) > 1.0).detach().cpu().tolist()

    total_mass = sum(masses)
    base_mass = mass_by_body.get("body", None)

    report = {
        "task": args_cli.task,
        "contact_sensor_type": type(cs).__name__,
        "num_contact_bodies_per_env": len(all_names),
        "contact_sensor_body_names": all_names,
        "footpad_regex": ".*_Footpad",
        "footpad_body_ids": list(footpad_cfg.body_ids) if not isinstance(footpad_cfg.body_ids, slice) else "all",
        "footpad_matched_names": footpad_names,
        "expected_footpad_names": list(CRAB_HEX_FOOTPAD_BODY_NAMES),
        "footpad_match_ok": footpad_names == list(CRAB_HEX_FOOTPAD_BODY_NAMES),
        "contact_force_threshold_sensor_cfg": cs.cfg.force_threshold,
        "zero_action_steps": args_cli.steps,
        "bodies_in_contact_norm_gt_1N": {
            all_names[i]: in_contact_all[i] for i in range(len(all_names))
        },
        "footpad_in_contact_norm_gt_1N": {
            footpad_names[i]: in_contact_footpad[i] for i in range(len(footpad_names))
        },
        "num_footpad_in_contact": sum(in_contact_footpad),
        "articulation_body_names": body_names_art,
        "mass_kg_by_body": mass_by_body,
        "total_mass_kg": total_mass,
        "base_body_mass_kg": base_mass,
        "urdf_reference_total_kg_approx": 23.0,
        "urdf_reference_base_kg": 10.0,
        "material_properties_first_env_shape": list(mats.shape),
        "material_static_friction_per_shape": mats[:, 0].tolist() if mats.numel() else [],
        "material_dynamic_friction_per_shape": mats[:, 1].tolist() if mats.numel() else [],
        "terrain_physics_material_cfg": terrain_mat,
        "usd_notes": {
            "footpad_collision_scale_m": "0.06 x 0.06 x 0.04 on sibling *_Footpad rigid bodies",
            "footpad_material": "FootRubber on footpad; *_Tibia_shank_vis PlyWood, no collision",
            "tibia_knee_link_mass_kg": 0.95,
            "footpad_mass_kg": 0.05,
            "explicit_physics_mass_in_crab_simple_usd": True,
        },
        "event_physics_material_randomization": "startup friction_range (0.6, 2.0) on all robot bodies unless disabled",
    }

    out = Path(args_cli.output) if args_cli.output else Path("logs/rsl_rl/crab_hex_flat_walk/diagnostics/contact_physics_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("=== Contact sensor bodies (all) ===")
    for i, name in enumerate(all_names):
        flag = "CONTACT" if in_contact_all[i] else "air"
        print(f"  [{i:2d}] {name:12s}  {flag}")
    print("\n=== .*_Footpad filter ===")
    print(f"  body_ids: {report['footpad_body_ids']}")
    print(f"  matched:  {footpad_names}")
    print(f"  OK:       {report['footpad_match_ok']}")
    print(f"  footpads in contact (>1N): {sum(in_contact_footpad)}/6")
    print("\n=== Mass (kg) ===")
    for name in sorted(mass_by_body, key=lambda n: -mass_by_body[n]):
        print(f"  {name:12s} {mass_by_body[name]:8.3f}")
    print(f"  TOTAL        {total_mass:8.3f}  (URDF ref ~23 kg, base ref 10 kg)")
    if base_mass and base_mass > 30:
        print("  WARNING: base mass looks auto-computed from large collision box, not URDF-scale.")
    print("\n=== Friction (shape 0, env 0) ===")
    if mats.numel():
        print(f"  static={mats[0, 0].item():.3f} dynamic={mats[0, 1].item():.3f} restitution={mats[0, 2].item():.3f}")
        print(f"  (after startup randomization; USD PlyWood was 0.45/0.35)")
    print(f"  terrain cfg: {terrain_mat}")
    print(f"\n[INFO] Wrote {out}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
