# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL runner cfgs for the sideways-walk task.

Two deltas against the forward runners, and nothing else (same optimizer, same networks, same
schedules -- the campaign changes the travel axis, not the policy):

1. ``experiment_name`` -> ``crab_hex_side_{flat_walk,teacher,student}``, so the sideways runs land
   in their own ``parkour/logs/rsl_rl/`` trees and never resume a forward head by accident.
2. the mirror-symmetry loss is re-pointed at the **fore-aft** mirror. The forward flat-walk runner
   arms symmetry unconditionally (``_apply_flat_walk_symmetry``, coef ``KRABBY_SYM_LOSS_COEF``,
   default 0.5) and the forward teacher runner arms it under the phase modes 2a/2b/2c -- always
   true under a sideways preset. Both side runners let the forward chain build the cfg and then
   swap only ``data_augmentation_func``, so a ``KRABBY_SYM_LOSS_COEF=0`` ablation (symmetry_cfg
   left None) still behaves exactly as it does forward.

Neither forward RL runner's ``__post_init__`` calls its own ``super()``: ``super().__post_init__()``
from the classes below therefore lands on the forward body and runs exactly it, once.
"""

from isaaclab.utils import configclass

from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.rsl_rl_ppo_cfg import (
    CrabHexFlatWalkPPORunnerCfg,
    CrabHexStudentPPORunnerCfg,
    CrabHexTeacherPPORunnerCfg,
)

#: ``mdp/crab_hex_side_mirror.py``: the same builder, reflected through the plane x = 0.
SIDE_MIRROR = (
    "parkour_tasks.crab_hex_sideways_task.mdp.crab_hex_side_mirror:"
    "crab_hex_side_symmetry_augmentation"
)


def _repoint_symmetry(cfg) -> None:
    """Point an already-armed symmetry loss at the fore-aft mirror (no-op when it is off)."""
    sym = cfg.algorithm.symmetry_cfg
    if sym is not None:
        sym.data_augmentation_func = SIDE_MIRROR


@configclass
class CrabHexSideFlatWalkPPORunnerCfg(CrabHexFlatWalkPPORunnerCfg):
    """Phase 1a (``Isaac-Crab-Hex-Side-Walk-v0``)."""

    experiment_name = "crab_hex_side_flat_walk"

    def __post_init__(self):
        super().__post_init__()
        _repoint_symmetry(self)


@configclass
class CrabHexSideTeacherPPORunnerCfg(CrabHexTeacherPPORunnerCfg):
    """Phases 2a / 2b / 2c (``Isaac-Crab-Hex-Side-Teacher-v0``)."""

    experiment_name = "crab_hex_side_teacher"

    def __post_init__(self):
        super().__post_init__()
        _repoint_symmetry(self)


@configclass
class CrabHexSideStudentPPORunnerCfg(CrabHexStudentPPORunnerCfg):
    """Phases 3a / 3b (``Isaac-Crab-Hex-Side-Student-v0``).

    Distillation has no symmetry loss (and the forward student runner has no ``__post_init__``),
    so ``experiment_name`` is the whole delta. ``clip_actions = 1.0`` is inherited and is
    load-bearing -- see the forward runner's note.
    """

    experiment_name = "crab_hex_side_student"
