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

There is no student scene: this plan is teacher-only (no depth camera, no phase 3).
"""

from isaaclab.utils import configclass

from parkour_isaaclab.terrains.omni.config.omni_ring import OMNI_RING_TERRAINS_CFG
from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_scene_cfg import (
    CrabHexTeacherSceneCfg,
)

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
