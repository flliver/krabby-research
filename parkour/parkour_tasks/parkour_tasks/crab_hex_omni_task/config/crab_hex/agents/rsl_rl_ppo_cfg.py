# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL runner cfgs for the omnidirectional task.

Two deltas against the forward runners, and nothing else (same optimizer, same networks, same
schedules -- the campaign changes the command space, not the policy):

1. ``experiment_name`` -> ``crab_hex_omni_{flat_walk,teacher}``, so the omni runs land in
   their own ``parkour/logs/rsl_rl/`` trees and never resume a forward or sideways head by
   accident.
2. the mirror-symmetry loss is re-pointed at the **four-image** omni mirror (L/R x fore-aft =
   R_z(pi)). The forward flat-walk runner arms symmetry unconditionally
   (``_apply_flat_walk_symmetry``, coef ``KRABBY_SYM_LOSS_COEF``, default 0.5) and the forward
   teacher runner arms it under the phase modes 2a/2b/2c -- always true under an omni preset.
   Both runners below let the forward chain build the cfg and then swap only
   ``data_augmentation_func``, so a ``KRABBY_SYM_LOSS_COEF=0`` ablation (symmetry_cfg left
   None) still behaves exactly as it does forward.

There is no student runner: this plan is teacher-only.

Neither forward RL runner's ``__post_init__`` calls its own ``super()``:
``super().__post_init__()`` from the classes below therefore lands on the forward body and
runs exactly it, once.

NOTE (plan caveat 13): with ``use_mirror_loss=True`` the fork runs ``act_inference`` on the
augmented batch -- four images here against the forward two -- so the symmetry-loss forward
pass per PPO update roughly doubles. Expect omni windows to be somewhat slower than the
forward ones at the same iteration count.
"""

from isaaclab.utils import configclass

from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.rsl_rl_ppo_cfg import (
    CrabHexFlatWalkPPORunnerCfg,
    CrabHexTeacherPPORunnerCfg,
)

#: ``mdp/crab_hex_omni_mirror.py``: the composed L/R x fore-aft four-image augmentation.
OMNI_MIRROR = (
    "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_mirror:"
    "crab_hex_omni_symmetry_augmentation"
)


def _repoint_symmetry(cfg) -> None:
    """Point an already-armed symmetry loss at the omni mirror (no-op when it is off)."""
    sym = cfg.algorithm.symmetry_cfg
    if sym is not None:
        sym.data_augmentation_func = OMNI_MIRROR


@configclass
class CrabHexOmniFlatWalkPPORunnerCfg(CrabHexFlatWalkPPORunnerCfg):
    """Phase-1 stages 1a / 1b / 1c (``Isaac-Crab-Hex-Omni-Walk-v0``)."""

    experiment_name = "crab_hex_omni_flat_walk"

    def __post_init__(self):
        super().__post_init__()
        _repoint_symmetry(self)


@configclass
class CrabHexOmniTeacherPPORunnerCfg(CrabHexTeacherPPORunnerCfg):
    """Phases 2a / 2b / 2c (+ ``2c_clockheld``) (``Isaac-Crab-Hex-Omni-Teacher-v0``)."""

    experiment_name = "crab_hex_omni_teacher"

    def __post_init__(self):
        super().__post_init__()
        _repoint_symmetry(self)
