"""Scene configuration that adds a front ZED-like camera (RGB + depth) to the parkour default scene.

Subclasses ParkourDefaultSceneCfg (parkour terrain, robot, sky) and adds front_camera
(depth) and front_rgb (RGB) without modifying any parkour files.
"""

from isaaclab.utils import configclass

from hal.server.isaac.sim_rgbd_camera_cfgs import sim_rgbd_camera_cfgs_for_robot_link
from parkour_tasks.default_cfg import ParkourDefaultSceneCfg

_front_camera_cfg, _front_rgb_cfg, *_rest = sim_rgbd_camera_cfgs_for_robot_link("base")

# Re-export for callers that referenced the standalone RGB cfg.
FRONT_RGB_CAMERA_CFG = _front_rgb_cfg


@configclass
class ZedLikeSceneCfg(ParkourDefaultSceneCfg):
    """Parkour default scene plus front ZED-like camera (depth + RGB)."""

    front_camera = _front_camera_cfg
    front_rgb = _front_rgb_cfg
