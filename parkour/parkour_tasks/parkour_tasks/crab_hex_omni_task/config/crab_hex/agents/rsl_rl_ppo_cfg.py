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

The omni plans were teacher-only. The Task 4 pipeline (step 4 / M1b, 2026-10-16) adds the depth-student
runner at the end of this module (:class:`CrabHexOmniStudentPPORunnerCfg`); the two runners above are
unchanged.

A THIRD, DEFAULT-OFF delta (motion-quality plan step 2, 2026-09-22): the three runner-side
exploration knobs -- ``KRABBY_ACTION_STD_MAX`` (``policy.action_std_max``),
``KRABBY_ACTION_STD_RESET`` (``policy.action_std_reset``) and ``KRABBY_ENTROPY_COEF``
(``algorithm.entropy_coef``) -- via ``motion_knobs.apply_exploration_knobs``. Unset, each writes
nothing, so the runner cfg (and ``params/agent.yaml``) is unchanged; set, the effective value is
on the cfg ``train.py`` dumps, which is where the campaign ladder's agent-side audit reads it.

A FOURTH, DEFAULT-OFF delta (side-speed-bisection plan step 1a, 2026-09-28):
``KRABBY_OMNI_MIRROR_IMAGES=fa`` re-points an already-armed symmetry loss's
``data_augmentation_func`` at the fore-aft-only, two-image mirror
(``crab_hex_omni_mirror.crab_hex_omni_symmetry_augmentation_fa``) instead of the shipped
four-image one. Unset, ``_repoint_symmetry`` writes the same ``OMNI_MIRROR`` path it always
has, so every existing run is byte-identical.

A FIFTH, DEFAULT-OFF delta (PPO-construction plan step 2, 2026-10-01): the PPO-construction knobs --
``KRABBY_ACTION_BOUND_LOSS`` / ``_BOUND_RAMP`` / ``_BOUND``, ``KRABBY_STD_PARAM``,
``KRABBY_INIT_STD``, ``KRABBY_PPO_SCHEDULE`` / ``_LR`` / ``_DESIRED_KL`` / ``_KL_STOP`` /
``_HIST_LR`` / ``_HAND_DIAG`` -- through the SAME ``apply_exploration_knobs`` call below (so both
runners apply them). Unset, each writes nothing; set, each prints ``[KNOB] <NAME>=<value> consumed``
and lands on the dumped cfg, and the runner refuses any algorithm / policy key its classes do not
accept, so none can be dropped silently.

Neither forward RL runner's ``__post_init__`` calls its own ``super()``:
``super().__post_init__()`` from the classes below therefore lands on the forward body and
runs exactly it, once.

NOTE (plan caveat 13): with ``use_mirror_loss=True`` the fork runs ``act_inference`` on the
augmented batch -- four images here against the forward two -- so the symmetry-loss forward
pass per PPO update roughly doubles. Expect omni windows to be somewhat slower than the
forward ones at the same iteration count.
"""

import os

from isaaclab.utils import configclass

from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.crab_hex_rl_cfg import (
    CrabHexParkourRslRlDepthEncoderCfg,
)
from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.rsl_rl_ppo_cfg import (
    CrabHexFlatWalkPPORunnerCfg,
    CrabHexStudentPPORunnerCfg,
    CrabHexTeacherPPORunnerCfg,
)
from parkour_tasks.crab_hex_omni_task.config.crab_hex import omni_knobs as _knobs
from parkour_tasks.crab_hex_omni_task.config.crab_hex import omni_student_spec as _student_spec

#: ``mdp/crab_hex_omni_mirror.py``: the composed L/R x fore-aft four-image augmentation.
OMNI_MIRROR = (
    "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_mirror:"
    "crab_hex_omni_symmetry_augmentation"
)
#: The fore-aft-only, two-image augmentation (``KRABBY_OMNI_MIRROR_IMAGES=fa``; side-speed-
#: bisection plan step 1a, default OFF).
FA_MIRROR = (
    "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_mirror:"
    "crab_hex_omni_symmetry_augmentation_fa"
)


def _apply_exploration_knobs(cfg) -> None:
    """The runner-side motion-quality and PPO-construction knobs (no-op while all are unset)."""
    from parkour_tasks.crab_hex_omni_task.config.crab_hex.motion_knobs import (  # noqa: PLC0415
        apply_exploration_knobs,
    )

    apply_exploration_knobs(cfg)


def _repoint_symmetry(cfg) -> None:
    """Point an already-armed symmetry loss at the omni mirror (no-op when it is off).

    ``KRABBY_OMNI_MIRROR_IMAGES=fa`` (side-speed-bisection plan step 1a, default OFF) swaps the
    four-image mirror for the two-image fore-aft-only one instead. Unset or blank, this reads
    nothing and the repoint below is exactly what it always was -- byte-identical to before the
    knob existed.
    """
    sym = cfg.algorithm.symmetry_cfg
    if sym is None:
        return
    raw_mode = os.environ.get(_knobs.MIRROR_IMAGES_VAR)
    mode = _knobs.parse_mirror_images(raw_mode) if raw_mode and str(raw_mode).strip() else None
    sym.data_augmentation_func = FA_MIRROR if mode == "fa" else OMNI_MIRROR


@configclass
class CrabHexOmniFlatWalkPPORunnerCfg(CrabHexFlatWalkPPORunnerCfg):
    """Phase-1 stages 1a / 1b / 1c (``Isaac-Crab-Hex-Omni-Walk-v0``)."""

    experiment_name = "crab_hex_omni_flat_walk"

    def __post_init__(self):
        super().__post_init__()
        _repoint_symmetry(self)
        _apply_exploration_knobs(self)


@configclass
class CrabHexOmniTeacherPPORunnerCfg(CrabHexTeacherPPORunnerCfg):
    """Phases 2a / 2b / 2c (+ ``2c_clockheld``) (``Isaac-Crab-Hex-Omni-Teacher-v0``)."""

    experiment_name = "crab_hex_omni_teacher"

    def __post_init__(self):
        super().__post_init__()
        _repoint_symmetry(self)
        _apply_exploration_knobs(self)


# ---------------------------------------------------------------------------- omni depth student
@configclass
class CrabHexOmniRslRlDepthEncoderCfg(CrabHexParkourRslRlDepthEncoderCfg):
    """The K-image depth encoder of the omni student (Task 4 M1b, DESIGN B4 / D4). A subclass, so the forward
    ``CrabHexParkourRslRlDepthEncoderCfg`` -- and ``FWD/policy/mdp_pins.yaml`` -- do not move.

    ``RecurrentMultiDepthBackbone`` (``scripts/rsl_rl/modules/feature_extractors/depth_backbone.py``): the four
    camera images through ONE shared conv -> ``combination_mlp(32 K + num_prop)`` -> GRU(512) -> a 32-d latent
    (no yaw head). ``num_prop`` 89 = the whole per-step proprio head (D4: restores the last-action tail and the
    current-sense proxy, columns 75-88, which the teacher saw through its scan slice). ``omni_student`` is the
    routing flag every entry point reads (``omni_student_step.is_omni_student``); it lands in
    ``params/agent.yaml``. ``grad_checkpoint`` recomputes the shared conv in the backward pass: at 192 envs x
    K = 4 x 24 encodes per 120-step rollout the stored conv activations alone are ~23 GB (vs ~6 GB at the
    forward K = 1) -- more than the 16 GB card; the recompute is mathematically the same gradient.
    """

    encoder_class_name: str = "RecurrentMultiDepthBackbone"
    num_prop: int = 89
    num_cameras: int = len(_student_spec.CAMERA_NAMES)
    depth_shape: tuple[int, int] = (_student_spec.IMAGE_ROWS, _student_spec.IMAGE_COLS)
    latent_dim: int = 32
    grad_checkpoint: bool = True
    omni_student: bool = True
    #: DEFAULT OFF diagnostic: all-zero images into the same step (the step-8 depth vs zeroed-depth pilot).
    #: No KRABBY knob sets it; a pilot passes the Hydra override ``agent.depth_encoder.zero_depth=true``.
    zero_depth: bool = False


@configclass
class CrabHexOmniStudentPPORunnerCfg(CrabHexStudentPPORunnerCfg):
    """Omni depth-student distillation runner (``Isaac-Crab-Hex-Omni-Student*``; Task 4 M1b, DESIGN B3).

    The forward student runner (``clip_actions`` 1.0 inherited -- load-bearing: the omni teacher's raw action
    clip is wider than its trained authority, exactly the forward phase-3 root cause), with:

    * ``experiment_name`` -> ``crab_hex_omni_student``;
    * the omni K = 4 depth encoder (:class:`CrabHexOmniRslRlDepthEncoderCfg`);
    * ONLY ``KRABBY_STD_PARAM`` / ``KRABBY_INIT_STD`` applied (``soft`` is needed to load the teacher's
      ``std_raw``); every other runner knob is REFUSED (``omni_student_spec.apply_student_runner_knobs``).
    """

    experiment_name = "crab_hex_omni_student"
    depth_encoder = CrabHexOmniRslRlDepthEncoderCfg(
        hidden_dims=512,
        learning_rate=1e-3,
        num_steps_per_env=24 * 5,
    )

    def __post_init__(self):
        # (the forward student runner has no __post_init__ of its own; configclass adds its own after this)
        _student_spec.apply_student_runner_knobs(self)
