"""Multi-sensor live view: list HAL sensors, print pipelines, show real camera/sim frames.

**Jetson** (default display): requires a **ZED** camera (same stack as `hal.server.jetson.zed_camera`).
Shows front RGB + depth in a grid. Other sensors listed by `JetsonSensorInterface` are not
captured here—only the ZED front pair.

**Isaac Sim** (``--backend isaac`` without ``--no-display``): starts Isaac Lab with
``ZedLikeSceneCfg`` (``front_rgb`` + ``front_camera`` depth). Requires ``isaaclab`` and
the same ``AppLauncher`` flags as other Isaac scripts.

**``--no-display``**: prints sensor metadata and example GStreamer pipeline strings only
(no windows, no hardware/sim). Isaac uses ``sim_rgbd_camera_cfgs`` + ``IsaacSensorInterface``
introspection (no duplicate catalog).

Usage:
  python -m hal.tools.multi_stream_display --backend jetson
  python -m hal.tools.multi_stream_display --backend isaac
  python -m hal.tools.multi_stream_display --backend jetson --no-display
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import numpy as np

from types import SimpleNamespace

from hal.server.isaac.sensor_backend_isaac import IsaacSensorInterface
from hal.server.jetson.sensor_backend_jetson import JetsonSensorInterface

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[misc, assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    interface = _get_interface(args.backend)
    _print_sensors_and_pipelines(interface)

    if args.no_display:
        return 0

    if args.backend == "jetson":
        print("Opening live ZED view (q to quit).")
        return run_jetson_zed_display()

    if args.backend == "isaac":
        print("Starting Isaac Sim with ZedLikeSceneCfg (q in window to quit).")
        return run_isaac_zed_like_display(args)

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List HAL sensors and pipelines; optional live view from ZED (Jetson) or Isaac ZedLike scene."
    )
    parser.add_argument(
        "--backend",
        choices=("jetson", "isaac"),
        default="jetson",
        help="jetson: ZED hardware. isaac: Isaac Sim with ZedLikeSceneCfg.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Only print sensors and pipeline strings (no OpenCV window).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Isaac Sim parallel envs (display backend only).",
    )
    # ``--backend`` selects Jetson vs Isaac *after* parse. We only need Isaac Lab here to
    # register AppLauncher flags (--headless, --device, …) when that package exists; Jetson
    # images usually lack isaaclab and skip this block without affecting ``--backend jetson``.
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        pass
    return parser


def run_isaac_zed_like_display(args: argparse.Namespace) -> int:
    """Isaac Lab + ZedLikeSceneCfg: live front_rgb + front_camera depth."""
    # Deferred import: isaaclab is only needed to construct AppLauncher on this code path.
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        logger.error("isaaclab is not installed; cannot run --backend isaac with display.")
        return 1
    if cv2 is None:
        logger.error("Install opencv-python for display mode.")
        return 1

    setattr(args, "enable_cameras", True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from hal.server.isaac.zed_like_scene_cfg import ZedLikeSceneCfg

    device = getattr(args, "device", "cuda:0")
    sim_cfg = sim_utils.SimulationCfg(device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 0.0, 1.5], target=[0.0, 0.0, 0.5])

    scene_cfg = ZedLikeSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    if not hasattr(scene, "sensors") or "front_camera" not in scene.sensors or "front_rgb" not in scene.sensors:
        logger.error("ZedLikeSceneCfg must provide front_camera and front_rgb.")
        simulation_app.close()
        return 1

    depth_sensor = scene.sensors["front_camera"]
    rgb_sensor = scene.sensors["front_rgb"]
    sim_dt = sim.get_physics_dt()

    def tensor_to_bgr_rgb(t: torch.Tensor) -> np.ndarray:
        if t.dim() > 3:
            t = t[0]
        arr = t.detach().cpu().numpy()
        if arr.ndim == 2:
            u8 = arr.astype(np.float32)
            if u8.max() <= 1.0:
                u8 = (np.clip(u8, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                u8 = np.clip(u8, 0.0, 255.0).astype(np.uint8)
            return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        if arr.dtype != np.uint8:
            if float(arr.max()) <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        if arr.shape[-1] >= 3:
            rgb = arr[..., :3]
        else:
            rgb = np.repeat(arr[..., :1], 3, axis=-1)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    try:
        while simulation_app.is_running():
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            tiles: list[tuple[str, np.ndarray]] = []
            if hasattr(rgb_sensor, "data") and hasattr(rgb_sensor.data, "output"):
                rgb_t = rgb_sensor.data.output.get("rgb")
                if rgb_t is not None:
                    tiles.append(("front_rgb", tensor_to_bgr_rgb(rgb_t)))
            if hasattr(depth_sensor, "data") and hasattr(depth_sensor.data, "output"):
                depth_t = depth_sensor.data.output.get("distance_to_camera")
                if depth_t is None:
                    depth_t = depth_sensor.data.output.get("distance_to_image_plane")
                if depth_t is not None:
                    d = depth_t[0] if depth_t.dim() > 2 else depth_t
                    d_np = d.detach().cpu().numpy().astype(np.float32)
                    if d_np.ndim == 3 and d_np.shape[-1] == 1:
                        d_np = d_np.squeeze(-1)
                    tiles.append(("front_camera_depth", _depth_to_bgr(d_np)))

            if not tiles:
                time.sleep(0.01)
                continue
            if _show_grid_frame(tiles, scale=0.5, window="isaac multi_stream_display"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


def run_jetson_zed_display() -> int:
    """Live front RGB-D camera via catalog driver (default: ZED; requires driver deps + hardware)."""
    if cv2 is None:
        logger.error("Install opencv-python for display mode.")
        return 1
    from hal.server.jetson.front_camera_factory import create_front_rgb_depth_camera
    from hal.server.jetson.sensor_backend_jetson import front_observation_camera_catalog_entry

    obs = front_observation_camera_catalog_entry()
    driver = obs.camera_driver
    if not driver:
        logger.error("JETSON_SENSOR_CATALOG is_primary row has no camera_driver.")
        return 1
    cam = create_front_rgb_depth_camera(
        driver,
        resolution=obs.resolution,
        fps=obs.fps,
        depth_mode=obs.depth_mode,
        maixsense_host_env=obs.maixsense_host_env,
        maixsense_port_env=obs.maixsense_port_env,
    )
    if cam is None:
        logger.error("Front RGB-D camera not available (init failed or no device).")
        return 1
    logger.info(
        "Live view: front RGB + depth (driver=%s, id=%s). "
        "Other JetsonSensorInterface entries have no capture in this tool.",
        obs.camera_driver,
        obs.id,
    )
    win = "jetson multi_stream_display"
    try:
        while True:
            rgb, depth = cam.get_camera_frames()
            if rgb is None:
                time.sleep(0.01)
                continue
            bgr = rgb
            tile_list: list[tuple[str, np.ndarray]] = [("front_rgbd_rgb", bgr)]
            if depth is not None:
                tile_list.append(("front_rgbd_depth", _depth_to_bgr(depth)))
            if _show_grid_frame(tile_list, scale=0.5, window=win):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


def _print_sensors_and_pipelines(interface) -> None:
    sensors = interface.list_sensors()
    print(f"Sensors: {len(sensors)}")
    for s in sensors:
        pose_str = f" pose={s.pose.to_tuple()}" if s.pose else ""
        print(f"  {s.id}: type={s.type} modality={s.modality} resolution={s.resolution} fps={s.fps}{pose_str}")

    print("\nGenerated pipelines (encoding=h264, sink=fakesink):")
    for s in sensors:
        handle = interface.get_gstreamer_handle(s)
        pipeline_str = interface.build_pipeline(handle, encoding="h264", output_element="fakesink")
        print(f"  [{s.id}]\n    {pipeline_str}\n")


def _show_grid_frame(
    tiles: list[tuple[str, np.ndarray]],
    scale: float,
    window: str,
) -> bool:
    """Show one frame; returns True if user pressed 'q'."""
    grid = _compose_grid_bgr(tiles, scale=scale)
    if grid is None:
        return False
    cv2.imshow(window, grid)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def _compose_grid_bgr(
    tiles: list[tuple[str, np.ndarray]],
    scale: float = 0.5,
) -> Optional[np.ndarray]:
    """Build one BGR image from labeled tiles, or None if empty."""
    if cv2 is None:
        raise RuntimeError("OpenCV required for display")
    if not tiles:
        return None
    cols = 2 if len(tiles) >= 2 else 1
    rows = (len(tiles) + cols - 1) // cols
    scaled: list[np.ndarray] = []
    for label, bgr in tiles:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Tile {label}: expected HxWx3 BGR, got {bgr.shape}")
        small = cv2.resize(bgr, None, fx=scale, fy=scale)
        cv2.putText(
            small,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        scaled.append(small)
    while len(scaled) < cols * rows:
        scaled.append(np.zeros_like(scaled[0]))
    grid = None
    for r in range(rows):
        row_imgs = scaled[r * cols : (r + 1) * cols]
        row_img = row_imgs[0] if len(row_imgs) == 1 else cv2.hconcat(row_imgs)
        grid = row_img if grid is None else cv2.vconcat([grid, row_img])
    return grid


def _depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    """Float depth (m) -> BGR uint8 for OpenCV."""
    if cv2 is None:
        raise RuntimeError("OpenCV required for display")
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mask = d > 0.01
    if not np.any(mask):
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = float(np.percentile(d[mask], 5)), float(np.percentile(d[mask], 95))
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)


def _isaac_interface_from_sim_cfgs(robot_link: str = "base") -> IsaacSensorInterface:
    """Same sensor catalog as HAL Isaac teleop (``sim_rgbd_camera_cfgs``), without a running env."""

    from hal.server.isaac.sim_rgbd_camera_cfgs import sim_rgbd_camera_cfgs_for_robot_link

    fc, fr, sc, sr = sim_rgbd_camera_cfgs_for_robot_link(robot_link)
    scene_sensors = {
        "front_camera": SimpleNamespace(cfg=fc),
        "front_rgb": SimpleNamespace(cfg=fr),
        "side_camera": SimpleNamespace(cfg=sc),
        "side_rgb": SimpleNamespace(cfg=sr),
    }
    return IsaacSensorInterface(scene_sensors=scene_sensors)


def _get_interface(backend: str):
    if backend == "jetson":
        return JetsonSensorInterface()
    if backend == "isaac":
        return _isaac_interface_from_sim_cfgs()
    raise ValueError(f"Unknown backend: {backend}. Use jetson or isaac.")


if __name__ == "__main__":
    sys.exit(main())
