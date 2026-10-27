# SPDX-License-Identifier: BSD-3-Clause
"""Scene cfg for the omnidirectional task: the forward teacher scene on a ring course (plan D7).

Two single-config changes, and nothing else about the scene:

===============================  =========================  =========================
thing                            forward                    omni
===============================  =========================  =========================
``terrain.terrain_generator``    ``EXTREME_PARKOUR_...``    ``OMNI_RING_TERRAINS_CFG``
``height_scanner.offset.pos``    ``(0.6, 0.0, 20.0)``       ``(0.0, 0.0, 20.0)``
===============================  =========================  =========================

**Why the scanner is centred.** The forward scan is biased 0.6 m along the travel axis, which
is a constant only because the travel axis is. Here travel is uniform over ``+-pi``, so any
bias points the sensor away from half the commanded directions; worse, the omni symmetry
augmentation reflects through BOTH body planes (L/R ``y = 0`` and fore-aft ``x = 0``), and an
off-plane scan offset makes the scan permutation wrong for the observation it augments. The
origin lies on both planes. The grid ``size`` / ``resolution`` / ``ordering`` are deliberately
UNCHANGED (2.4 x 3.4 m at 0.3 m -> 108 rays), so the 1149-dim observation, the scan
permutation and every warm start stay valid; only the patch's centre moves, from
``x in [-0.6, +1.8]`` to ``x in [-1.2, +1.2]``.

The spawn ``init_state.rot`` stays IDENTITY: the omni reset event
(``mdp/omni_events.py::reset_root_state_ring``) draws a uniform world yaw at every reset, so
a baked-in spawn attitude would only bias the first frame of the first episode. This is
stated rather than asserted because "the sideways scene turns it, the omni one must not" is
exactly the kind of copy-paste the delta test exists to catch (it pins the identity rot).

The omni plans were teacher-only; the Task 4 pipeline (2026-10-16, step 4 / M1b) adds
:class:`CrabHexOmniStudentSceneCfg` -- the teacher scene plus the four depth cameras of
``omni_student_spec`` (``PL/cameras.yaml`` v0). It is used ONLY by the omni student gym ids; the
teacher / walk scenes above are untouched.
"""

from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass

from parkour_isaaclab.terrains.omni.config.omni_ring import OMNI_RING_TERRAINS_CFG
from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_scene_cfg import (
    CrabHexTeacherSceneCfg,
)
from parkour_tasks.crab_hex_omni_task.config.crab_hex import omni_student_spec as _student_spec
from parkour_tasks.default_cfg import CAMERA_CFG

#: Height-scan ray origin, body frame. On both mirror planes (see the module docstring).
SCANNER_OFFSET_POS = (0.0, 0.0, 20.0)


def _apply_omni_terrain_and_scanner(scene) -> None:
    """Install the ring generator and centre the scan.

    Called from ``__post_init__`` AFTER ``super().__post_init__()``, i.e. after the forward
    ``_crab_robot_cfg()`` / ``_apply_crab_height_scanner()`` and after
    ``ParkourTeacherSceneCfg.__post_init__`` has installed the corridor generator: this only
    re-points what they built. The offset is rebuilt exactly the way the forward helper
    builds it (``type(offset)(pos=...)``, no ``rot``), so the two configs differ in ``pos``
    alone.
    """
    scene.terrain.terrain_generator = OMNI_RING_TERRAINS_CFG
    scene.height_scanner.offset = type(scene.height_scanner.offset)(pos=SCANNER_OFFSET_POS)


@configclass
class CrabHexOmniTeacherSceneCfg(CrabHexTeacherSceneCfg):
    """Teacher / omni-walk scene: the forward teacher scene on the ring course."""

    def __post_init__(self):
        super().__post_init__()
        _apply_omni_terrain_and_scanner(self)


def _student_camera(name: str):
    """One depth camera of the spec, as a ``RayCasterCameraCfg`` (the forward ``CAMERA_CFG`` fields replaced).

    Everything that defines the image is set from ``omni_student_spec`` (pose, ROS quaternion composed
    there, a 58 x 87 pinhole grid over the datasheet FOV, 2 m clip, ``/World/ground`` only). The update
    period is set by the env cfg (``sim.dt * decimation``, as the forward student does). The rotation
    tripwire runs in the env cfg's ``__post_init__`` (see ``omni_student_spec``'s module docstring).
    """
    spec = _student_spec.camera_specs()[name]
    return CAMERA_CFG.replace(
        prim_path=_student_spec.CAMERA_PRIM_PATH,
        offset=CAMERA_CFG.offset.replace(pos=tuple(spec["pos"]), rot=tuple(spec["rot"]), convention="ros"),
        data_types=["distance_to_camera"],
        depth_clipping_behavior="max",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=_student_spec.FOCAL_LENGTH,
            horizontal_aperture=spec["horizontal_aperture"],
            vertical_aperture=spec["vertical_aperture"],
            height=spec["height"],
            width=spec["width"],
        ),
        mesh_prim_paths=list(_student_spec.MESH_PRIM_PATHS),
        max_distance=_student_spec.MAX_DISTANCE_M,
    )


@configclass
class CrabHexOmniStudentSceneCfg(CrabHexOmniTeacherSceneCfg):
    """Omni depth-student scene (Task 4 M1b): the omni teacher scene + four ray-caster depth cameras.

    Front ZED 2i + left / right / rear SEN0583 (``omni_student_spec.CAMERA_NAMES`` order), all on
    ``chassis/body``, all PLACEHOLDER mounts (M15). Deliberately NOT derived from the forward student scene:
    that one inherits the Go2 student scene's terrain rewrite. There is no camera-pose event (the teacher
    env sets ``events.random_camera_position = None``; the env cfg asserts it).
    """

    depth_camera_front = _student_camera("front")
    depth_camera_left = _student_camera("left")
    depth_camera_right = _student_camera("right")
    depth_camera_rear = _student_camera("rear")
