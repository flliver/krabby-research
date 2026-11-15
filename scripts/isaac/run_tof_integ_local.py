#!/usr/bin/env python3
"""Local Isaac Lab runner for the ToF HAL integration check (no Docker).

What this does
--------------
Boots Isaac Sim via Isaac Lab ``AppLauncher`` (headless + cameras), attaches
front RGB-D + belly/rear ToF sensors on Unitree Go2, places a box on the
belly-right optical axis, and asserts through ``HalClient`` that:

- ``belly_left_tof``, ``belly_right_tof``, ``rear_tof`` publish depth-only
  100×100 float32 frames on the HAL observation bus
- a cluster of belly-right depth pixels is near 0.35 m (box range)

Same asserts as ``images/isaacsim/test_runner.py`` → ``test_isaacsim_tof_cameras``.

Prerequisites
-------------
- NVIDIA GPU + working Isaac Sim / Isaac Lab **local** install (not the Docker
  image).
- Conda env created by Isaac Lab (default name ``env_isaaclab``)::

      cd /path/to/IsaacLab
      ./isaaclab.sh --conda          # creates env named env_isaaclab
      # or: ./isaaclab.sh --conda my_name

  Activating that env sources Isaac Sim (``_isaac_sim/setup_conda_env.sh``) so
  ``import isaacsim`` / ``AppLauncher`` work.
- This repo (``krabby-research``) on ``PYTHONPATH`` via ``cd`` into the checkout
  (the script adds the repo root itself).
- Parkour tasks importable (``parkour/parkour_tasks`` in this repo).

Run
---
::

    source ~/miniconda3/etc/profile.d/conda.sh   # or your conda install
    conda activate env_isaaclab                  # or the name you passed to --conda
    cd /path/to/krabby-research
    PYTHONUNBUFFERED=1 python scripts/isaac/run_tof_integ_local.py

Optional AppLauncher flags (e.g. ``--device cuda:0``) are accepted; the script
forces ``--headless`` and ``--enable_cameras``.

Success looks like
------------------
::

    [PASS] test_isaacsim_tof_cameras (local)

Exit code 0. A trailing ``ParkourViewportCameraController`` / ``carb.input``
``AttributeError`` on shutdown is harmless teardown noise.

Docker alternative (no local Isaac Lab)
---------------------------------------
Build ``krabby-isaacsim:latest`` (see ``images/isaacsim/README.md``), then::

    PYTHONUNBUFFERED=1 timeout 600 docker run --rm --gpus all \\
      --entrypoint /workspace/run_test_runner.sh \\
      krabby-isaacsim:latest test_isaacsim_tof_cameras

For visual leg checks (PNGs), use ``scripts/isaac/walk_tof_preview.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_parkour_import_order() -> None:
    candidates = (
        Path("/workspace/parkour/parkour_tasks"),
        _REPO_ROOT / "parkour" / "parkour_tasks",
    )
    for path in candidates:
        if path.is_dir() and (path / "parkour_tasks").is_dir():
            resolved = str(path)
            if resolved in sys.path:
                sys.path.remove(resolved)
            sys.path.insert(0, resolved)
            break
    repo = str(_REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def main(argv: list[str] | None = None) -> int:
    print("START", flush=True)
    _ensure_parkour_import_order()

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True
    args.enable_cameras = True
    if not getattr(args, "device", None):
        args.device = "cuda:0"

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    import isaaclab_tasks  # noqa: F401
    import parkour_tasks  # noqa: F401
    from isaaclab.assets import RigidObjectCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

    from hal.server.isaac.sim_rgbd_camera_cfgs import (
        apply_sim_hal_camera_update_periods,
        attach_sim_rgbd_sensors_to_scene_cfg,
    )
    from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION
    from tests.integration.test_isaacsim_tof_cameras import (
        place_tof_test_box_on_optical_axis,
        run_tof_hal_client_check,
    )

    # Play cfg matches the known-good walk_tof_preview Go2 path.
    task_name = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0"
    box_distance_m = 0.35
    rc = 1
    env = None
    try:
        print(f"parse_env_cfg {task_name}", flush=True)
        env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=1, use_fabric=True)
        env_cfg.scene.num_envs = 1
        env_cfg.scene.tof_test_box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TofTestBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, -0.5, 0.4),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        attach_sim_rgbd_sensors_to_scene_cfg(env_cfg.scene, "base")
        apply_sim_hal_camera_update_periods(
            env_cfg.scene, control_dt=env_cfg.sim.dt * env_cfg.decimation
        )

        print("Creating env", flush=True)
        env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        env.reset()
        target = place_tof_test_box_on_optical_axis(env, distance_m=box_distance_m)
        print(f"ToF test box placed at {target}", flush=True)

        print("HAL client ToF depth check", flush=True)
        rgbd = run_tof_hal_client_check(
            env=env,
            robot_definition=UNITREE_GO2_DEFINITION,
            box_expected_m=box_distance_m,
            box_catalog_id="belly_right_tof",
        )
        for cid, entry in sorted(rgbd.items()):
            depth = entry.channels.get("depth")
            if depth is None:
                continue
            finite = depth[np.isfinite(depth) & (depth > 0)]
            med = float(np.median(finite)) if finite.size else float("nan")
            print(f"{cid} shape={depth.shape} median={med:.3f}", flush=True)
        print("[PASS] test_isaacsim_tof_cameras (local)", flush=True)
        rc = 0
    except BaseException as exc:
        print(f"[FAIL] test_isaacsim_tof_cameras (local): {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            simulation_app.close()
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
