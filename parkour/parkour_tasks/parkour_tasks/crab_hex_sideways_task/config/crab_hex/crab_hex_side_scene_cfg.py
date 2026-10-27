# SPDX-License-Identifier: BSD-3-Clause
"""Scene cfgs for the sideways-walk task: the forward scenes, turned a quarter turn.

The robot travels along its body **+y** axis and the course runs along world +x, so the spawn
yaw is **-pi/2** (``TRAVEL_YAW = +pi/2`` is the body->travel rotation; the spawn rotation is its
inverse). Three single-config sensors/placements follow the travel axis and nothing else about
the scene changes:

===================  ===========================  ===============================
thing                forward                      sideways
===================  ===========================  ===============================
``robot.init_state.rot``   ``(1, 0, 0, 0)``       ``(sqrt(.5), 0, 0, -sqrt(.5))``
``height_scanner.offset.pos``  ``(0.6, 0, 20)``   ``(0, 0.6, 20)``
``depth_camera.offset``    ``pos (0.37, 0, 0.10)``  ``pos (0, 0.625, 0.10)``, rot turned +90 deg about z
===================  ===========================  ===============================

Everything else -- the scan ``size``/``resolution``/``ordering``, the contact sensor, the plant,
the terrain -- is inherited from the forward scene classes and MUST stay equal (the sideways
config-delta test asserts exactly the rows above).
"""

import math
import warnings

from isaaclab.utils import configclass

from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_scene_cfg import (
    CrabHexStudentSceneCfg,
    CrabHexTeacherSceneCfg,
)
from parkour_tasks.default_cfg import CAMERA_CFG

#: Body -> travel rotation. The commanded speed stays in ``vel_command_b[:, 0]`` and is read as
#: "speed along the travel axis" = the body frame rotated by ``TRAVEL_YAW`` (0 for forward).
TRAVEL_YAW = math.pi / 2

#: Spawn attitude (w, x, y, z): yaw **-pi/2** exactly, so body +y points down the course (world
#: +x). Written as ``sqrt(0.5)`` literals rather than through a trig helper because the config
#: delta is compared at a 1e-9 tolerance and ``cos(pi/4)`` round-trips are not bit-stable.
SPAWN_ROT_WXYZ = (math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5))

#: Height-scan look-ahead: the forward 0.6 m bias moved from body +x to body **+y**.
#: The grid ``size``/``resolution``/``ordering`` are deliberately NOT transposed -- the scan
#: keeps its 108 rays in their original layout, so the scan mirror permutation, the 1149-dim
#: observation and every warm start stay valid. Consequence (accepted, recorded in the plan):
#: along-travel coverage becomes y in [-1.1, +2.3] and cross-travel x in [-1.2, +1.2].
SCANNER_OFFSET_POS = (0.0, 0.6, 20.0)

#: Depth camera on the +y body face (leg-mount half-width 0.6096 m plus the forward mount's
#: margin). HARDWARE PLACEHOLDER: the real mount is a decision taken at the phase-3 point; the
#: ray-caster camera has no collision, so the 0.625 offset is a convention, not a constraint.
DEPTH_CAMERA_POS = (0.0, 0.625, 0.10)

#: Quarter turn about body +z, as a (w, x, y, z) quaternion.
_YAW_PLUS_90_WXYZ = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))


def _quat_mul(q1, q2):
    """Hamilton product of two ``(w, x, y, z)`` quaternions (pure Python, no torch)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


#: The forward camera rotation pre-multiplied by the quarter turn, computed at import.
#:
#: NOTE: editing the Euler yaw term of ``default_cfg.CAMERA_CFG`` does NOT work here.
#: ``quat_from_euler_xyz_tuple`` NEGATES ``qz`` before returning, so the tuple it hands back is
#: not the quaternion of the Euler triple it was given; feeding it ``[180, 70, 0]`` (or any other
#: yaw) rolls the image about the optical axis instead of swinging the camera onto the +y face.
#: Composing the finished forward quaternion with a real +90 deg z-rotation is exact: the optical
#: axis moves from body ``(0.94, 0, -0.342)`` to body ``(0, 0.94, -0.342)``.
DEPTH_CAMERA_ROT = _quat_mul(_YAW_PLUS_90_WXYZ, tuple(CAMERA_CFG.offset.rot))

#: Tripwire: a silent change in ``CAMERA_CFG.offset.rot`` would otherwise point the student's
#: only exteroceptive sensor somewhere nobody checked. A WARNING, not a raise -- every forward
#: process imports this module through ``import_packages``, so an import-time exception here
#: would be a forward-task outage; the hard assertion lives in
#: ``tests/integration/test_crab_hex_sideways_configs.py`` (student pairs). (The forward rot is
#: built in float32, so the residual ``qz`` is ~4e-8 rather than exactly 0.)
_DEPTH_CAMERA_ROT_EXPECTED = (-0.5735764, 0.8191520, 0.0, 0.0)
if any(abs(a - b) > 1e-6 for a, b in zip(DEPTH_CAMERA_ROT, _DEPTH_CAMERA_ROT_EXPECTED)):
    warnings.warn(
        "sideways depth-camera rotation drifted: "
        f"{DEPTH_CAMERA_ROT} != {_DEPTH_CAMERA_ROT_EXPECTED} (tol 1e-6). "
        "parkour_tasks.default_cfg.CAMERA_CFG.offset.rot changed underneath this module; "
        "re-check the phase-3 camera placement (plan Verification 4b).",
        stacklevel=1,
    )


def _apply_side_spawn_and_scanner(scene) -> None:
    """Point the spawn attitude and the scan look-ahead down the travel axis.

    Called from ``__post_init__`` AFTER ``super().__post_init__()``, i.e. after the forward
    ``_crab_robot_cfg()`` / ``_apply_crab_height_scanner()`` have run: this only re-points what
    they built. The offset is rebuilt exactly the way the forward helper builds it
    (``type(offset)(pos=...)``, no ``rot``), so the two configs differ in ``pos`` alone.
    """
    scene.robot.init_state.rot = SPAWN_ROT_WXYZ
    scene.height_scanner.offset = type(scene.height_scanner.offset)(pos=SCANNER_OFFSET_POS)


@configclass
class CrabHexSideTeacherSceneCfg(CrabHexTeacherSceneCfg):
    """Teacher / flat-walk scene: the forward teacher scene with the travel axis on body +y."""

    def __post_init__(self):
        super().__post_init__()
        _apply_side_spawn_and_scanner(self)


@configclass
class CrabHexSideStudentSceneCfg(CrabHexStudentSceneCfg):
    """Student scene: same, plus the depth camera moved onto the +y body face."""

    depth_camera = CAMERA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/body",
        offset=CAMERA_CFG.offset.replace(pos=DEPTH_CAMERA_POS, rot=DEPTH_CAMERA_ROT),
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_side_spawn_and_scanner(self)
