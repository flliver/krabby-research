#!/usr/bin/env python3
"""Walk the robot in Isaac while saving ToF depth previews.

Attaches front RGB-D + three SEN0581-spec ToF cameras, runs a short scripted forward
walk, and writes colorized depth PNGs for ``belly_left_tof``, ``belly_right_tof``,
and ``rear_tof``.

Local (conda env_isaaclab)::

    source /home/sanjay/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    cd /home/sanjay/Projects/krabby/krabby-research
    python scripts/isaac/walk_tof_preview.py --out /tmp/tof_preview --steps 120 --robot go2

Hex (plant of record ``assets/crab.usda``; cameras mount on ``chassis/body``)::

    python scripts/isaac/walk_tof_preview.py --out /tmp/tof_preview_hex --steps 120 --robot hex \
      --usd assets/crab.usda

Confirm legs are visible in each belly camera's saved frames.
Go2 uses tighter belly mounts than hex (hex Y would hang off parkour voids).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("walk_tof_preview")

TOF_IDS = ("belly_left_tof", "belly_right_tof", "rear_tof")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_parkour_import_order() -> None:
    """Register gym envs before ``parkour_isaaclab.envs`` (avoids circular import).

    Prefer the local package root ``parkour/parkour_tasks`` (parent of the
    ``parkour_tasks`` package) ahead of the editable ``parkour/`` path that can
    shadow it as a namespace package.
    """
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


def _colorize_depth(depth: np.ndarray, d_min: float = 0.15, d_max: float = 1.5) -> np.ndarray:
    """Return uint8 RGB preview (turbo-like grayscale stretch if cv2 missing)."""
    clipped = np.clip(depth.astype(np.float32), d_min, d_max)
    norm = (clipped - d_min) / max(d_max - d_min, 1e-6)
    u8 = (norm * 255.0).astype(np.uint8)
    try:
        import cv2

        return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    except Exception:
        return np.stack([u8, u8, u8], axis=-1)


def _save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cv2.imwrite(str(path), rgb)
    except Exception:
        h, w = rgb.shape[:2]
        with open(path.with_suffix(".ppm"), "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(np.ascontiguousarray(rgb[:, :, ::-1] if rgb.shape[2] == 3 else rgb).tobytes())


def _default_hex_usd() -> str | None:
    for candidate in (
        _REPO_ROOT / "assets" / "crab.usda",
        Path("/workspace/assets/crab.usda"),
        _REPO_ROOT / "assets" / "crab_hex.usda",
        _REPO_ROOT / "assets" / "crab_hex_ref.usd",
        Path("/workspace/assets/crab_hex_ref.usd"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _hex_camera_mount_link(usd_path: str) -> str:
    """Body prim under ``Robot/`` for ToF/RGB-D offsets.

    Plant-of-record ``crab.usda`` / ``crab_simple.usda`` use ``chassis/body``.
    Legacy ``crab_hex_ref.usd`` uses ``crab_hex``.
    """
    name = Path(usd_path).name.lower()
    if "crab_hex_ref" in name:
        return "crab_hex"
    return "chassis/body"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/tof_preview"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--robot", choices=("hex", "go2"), default="go2")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--usd", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    # AppLauncher owns --device / --headless / --enable_cameras (do not redefine them).
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)

    # Force headless + cameras for this preview script.
    args.headless = True
    args.enable_cameras = True
    if not getattr(args, "device", None):
        args.device = "cuda:0"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    _ensure_parkour_import_order()
    import isaaclab_tasks  # noqa: F401
    import parkour_tasks  # noqa: F401

    from isaaclab_tasks.utils import parse_env_cfg
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

    from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
    from hal.client.client import HalClient
    from hal.client.config import HalClientConfig
    from hal.server import HalServerConfig
    from hal.server.isaac import IsaacSimHalServer
    from hal.server.isaac.sim_rgbd_camera_cfgs import (
        apply_sim_hal_camera_update_periods,
        attach_sim_rgbd_sensors_to_scene_cfg,
    )
    from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
    from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION
    from hal.server.zmq_endpoints import zmq_bind_to_connect_url

    if args.usd or args.robot == "hex":
        usd = args.usd or _default_hex_usd()
        if not usd:
            raise SystemExit(
                "Hex run needs --usd or assets/crab.usda. "
                "Or use --robot go2 to validate ToF frames without a hex USD."
            )
        os.environ["KRABBY_HEX_USD_PATH"] = usd
        task = args.task or "Isaac-CrabHex-Joystick-v0"
        robot = "hex"
        link = _hex_camera_mount_link(usd)
        robot_def = KRABBY_HEX_DEFINITION
    else:
        usd = None
        task = args.task or "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0"
        robot = "go2"
        link = "base"
        robot_def = UNITREE_GO2_DEFINITION

    obs_dims = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(robot_def)

    logger.info("task=%s robot=%s link=%s usd=%s out=%s", task, robot, link, usd, args.out)
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1, use_fabric=True)
    env_cfg.scene.num_envs = 1
    # Joystick hex cfg hardcodes height_scanner on Robot/crab_hex; plant-of-record uses chassis/body.
    height_scanner = getattr(env_cfg.scene, "height_scanner", None)
    if height_scanner is not None and robot == "hex":
        height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{link}"
    attach_sim_rgbd_sensors_to_scene_cfg(env_cfg.scene, link)
    apply_sim_hal_camera_update_periods(
        env_cfg.scene, control_dt=env_cfg.sim.dt * env_cfg.decimation
    )

    env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    env.reset()

    obs_bind = "inproc://walk_tof_obs"
    cmd_bind = "inproc://walk_tof_cmd"
    server = IsaacSimHalServer(
        HalServerConfig(observation_bind=obs_bind, command_bind=cmd_bind),
        robot_def,
        env=env,
        observation_dimensions=obs_dims,
    )
    server.initialize()
    # inproc requires the client's sockets to share the server's ZMQ context.
    client = HalClient(
        HalClientConfig(
            observation_endpoint=zmq_bind_to_connect_url(obs_bind),
            command_endpoint=zmq_bind_to_connect_url(cmd_bind),
        ),
        context=server.get_transport_context(),
    )
    client.initialize()
    # PUB/SUB slow-joiner: give the SUB socket a beat before the first publish.
    time.sleep(0.05)

    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    n_act = int(np.prod(env.action_space.shape))
    saved = 0
    empty_polls = 0
    empty_rgbd = 0
    run_error: BaseException | None = None
    try:
        for step in range(args.steps):
            # Hold default pose so flop/thrash does not empty the ToF FOV.
            action = torch.zeros((1, n_act), device=env.device, dtype=torch.float32)
            obs_dict, *_ = env.step(action)
            # Cache step observation so set_observation uses the env's exact tensor.
            if isinstance(obs_dict, dict) and "policy" in obs_dict:
                server._latest_obs_dict = obs_dict
                server._latest_obs_tensor = obs_dict["policy"]
            server.set_observation()
            obs = client.poll(timeout_ms=50)
            if obs is None:
                empty_polls += 1
                continue
            if not obs.rgbd_by_catalog_id:
                empty_rgbd += 1
                if empty_rgbd <= 3:
                    logger.warning(
                        "step %d: observation has no rgbd_by_catalog_id "
                        "(cameras may still be warming up); sensors=%s",
                        step,
                        sorted(getattr(server, "camera_sensors", {})),
                    )
                continue
            if step % args.save_every != 0 and step != args.steps - 1:
                continue
            for cid in TOF_IDS:
                chunk = obs.rgbd_by_catalog_id.get(cid)
                if chunk is None:
                    logger.warning("step %d: missing %s keys=%s", step, cid, sorted(obs.rgbd_by_catalog_id))
                    continue
                depth = chunk.channels.get("depth")
                if depth is None:
                    continue
                preview = _colorize_depth(depth)
                out_path = args.out / f"{cid}_step{step:04d}.png"
                _save_png(out_path, preview)
                saved += 1
                finite = depth[np.isfinite(depth) & (depth > 0)]
                med = float(np.median(finite)) if finite.size else float("nan")
                logger.info(
                    "saved %s shape=%s median_depth=%.3f m",
                    out_path,
                    depth.shape,
                    med,
                )
    except BaseException as exc:
        run_error = exc
        logger.exception("walk_tof_preview failed during step/set_observation: %s", exc)
    finally:
        try:
            client.close()
        except Exception:
            logger.exception("client.close failed")
        try:
            server.close()
        except Exception:
            logger.exception("server.close failed")
        try:
            env.close()
        except Exception:
            logger.exception("env.close failed")
        try:
            simulation_app.close()
        except Exception:
            logger.exception("simulation_app.close failed")

    if run_error is not None:
        print(f"FATAL: {type(run_error).__name__}: {run_error}", flush=True)
        return 2

    if saved == 0:
        msg = (
            f"No ToF previews written (empty_polls={empty_polls} empty_rgbd={empty_rgbd}). "
            "Check enable_cameras and shared inproc ZMQ context."
        )
        logger.error(msg)
        print(f"FATAL: {msg}", flush=True)
        return 1

    logger.info(
        "Done (%d files). Inspect %s — belly cameras should show legs if poses are correct.",
        saved,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
