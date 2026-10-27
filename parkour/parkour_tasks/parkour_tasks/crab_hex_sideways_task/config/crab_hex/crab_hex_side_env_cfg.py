# SPDX-License-Identifier: BSD-3-Clause
"""Env cfgs for the sideways-walk task -- the forward env cfgs with the travel axis on body +y.

The whole delta against the forward task is eight config paths (plus the student's camera):

* ``travel_yaw = +pi/2`` -- a field the forward cfg classes do NOT have. Every travel-frame seam
  in ``parkour_isaaclab`` reads it through ``getattr(env.cfg, "travel_yaw", 0.0)``, so the forward
  task keeps the identity path bit-for-bit.
* ``commands.base_velocity.heading_offset = -travel_yaw`` -- likewise a field only the sideways
  command cfg has. It rotates the WORLD heading target the command term steers to; without it the
  yaw-rate command saturates at +-1 and the robot spins back to yaw 0.
* ``rewards.track_lin_vel_xy_exp.func`` -> the travel-frame twin (same term NAME, weight, params).
* the scene's spawn rot / scan offset (and the student's depth camera) -- see
  ``crab_hex_side_scene_cfg.py``.
* ``events.rsi_reference_reset`` absent: the presets drop the RSI keys (the bank of record is a
  forward gait), so the knob block never builds the term.

**KNOB TIMING (the pillar that makes this file look redundant).** ``import_packages`` imports the
forward package FIRST, so every forward cfg class body -- and with it every cfg object those
bodies construct -- is evaluated BEFORE the sideways activation (``crab_hex_side_phases``) has
exported the ``KRABBY_*`` knobs. A configclass default is a deep copy of the object built in the
class body, so simply inheriting ``scene`` / ``observations`` / ``actions`` / ``commands`` /
``rewards`` / ``terminations`` / ``parkours`` / ``events`` from the forward classes would hand the
sideways task the UN-EXPANDED forward defaults (wrong plant, wrong command band, wrong episode
length, ...) with no error anywhere. Therefore every cfg-valued class attribute is RE-DECLARED
below with the forward class-body literal, so each is constructed at *this* module's import --
which happens after ``config/crab_hex/__init__.py`` has run the sideways activation. The
sideways-vs-forward delta test proves each one still equals the forward instance built under the
forward activation.
"""

import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp import rewards as mdp_rewards
from parkour_isaaclab.envs.mdp.parkour_commands import ParkourCommandCfg
from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import (
    CrabHexFlatWalkActionsCfg,
    CrabHexFlatWalkRewardsCfg,
    CrabHexFlatWalkTerminationsCfg,
    CrabHexRewardsCfg,
    CrabHexStudentActionsCfg,
    CrabHexStudentObservationsCfg,
    CrabHexStudentRewardsCfg,
    CrabHexTeacherObservationsCfg,
    CrabHexTerminationsCfg,
    EventCfg,
    ParkourEventsCfg,
)
from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_env_cfg import (
    CRAB_HEX_VIEWER,
    PHASE_TEACHER_MODES,
    CrabHexFlatWalkEnvCfg,
    CrabHexStudentEnvCfg,
    CrabHexTeacherEnvCfg,
    _crab_hex_teacher_mode,
)
from parkour_tasks.crab_hex_sideways_task.config.crab_hex.crab_hex_side_scene_cfg import (
    TRAVEL_YAW,
    CrabHexSideStudentSceneCfg,
    CrabHexSideTeacherSceneCfg,
)


# ---------------------------------------------------------------------------- commands


@configclass
class CrabHexSideVelocityCommandCfg(ParkourCommandCfg):
    """The parkour velocity command with a constant heading offset.

    This is the ONLY place ``heading_offset`` exists as a field: the shared command term reads it
    through ``getattr(self.cfg, "heading_offset", 0.0)``, so the forward command cfg keeps its
    exact draw order and its dumps are unchanged.

    The offset is added to the sampled heading target, which is a WORLD yaw. The sideways robot
    spawns at yaw ``-pi/2``, so a heading band of ``(0, 0)`` plus an offset of ``-pi/2`` steers it
    to hold the spawn attitude -- i.e. "straight ahead" along the course. ``heading_target`` may
    leave ``[-pi, pi]``; every consumer wraps the difference.
    """

    heading_offset: float = -TRAVEL_YAW


@configclass
class CrabHexSideCommandsCfg:
    """The Go2 ``CommandsCfg`` literals verbatim, on the sideways command class.

    Standalone rather than a subclass of the Go2 ``CommandsCfg``: the only member is
    ``base_velocity``, and it has to be rebuilt anyway (different class, and knob timing), so
    inheriting would add nothing but a second place for the literals to drift.
    """

    base_velocity = CrabHexSideVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        heading_control_stiffness=0.8,
        ranges=ParkourCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.8),
            heading=(-1.6, 1.6),
        ),
        clips=ParkourCommandCfg.Clips(
            lin_vel_clip=0.2,
            ang_vel_clip=0.4,
        ),
    )


# ---------------------------------------------------------------------------- rewards


@configclass
class CrabHexSideFlatWalkRewardsCfg(CrabHexFlatWalkRewardsCfg):
    """The 35 flat-walk reward terms with ONE function swapped for its travel-frame twin.

    Same attribute name, same weight, same params: ``Episode_Reward/track_lin_vel_xy_exp``
    telemetry, the ``KRABBY_TRACK_SIGMA2`` override (which assigns
    ``self.track_lin_vel_xy_exp.params["std"]`` by name in the parent's ``__post_init__``), the
    ``KRABBY_PHASEOUT`` term names and the driver's ``exposure_from_log`` all key on that name.
    Re-declaring the field keeps its position in the parent's field order.
    """

    track_lin_vel_xy_exp = RewTerm(
        func=mdp_rewards.track_lin_vel_travel_exp,
        weight=1.25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.02)},
    )


# ---------------------------------------------------------------------------- helpers


def _check_side(cfg) -> None:
    """Post-init guard: the command term must still carry the sideways heading offset.

    Anything that replaced ``cfg.commands`` with a forward ``CommandsCfg`` (an inherited class
    attribute that escaped the re-declarations above, a knob block, a future refactor) would train
    a policy that is commanded to hold yaw 0 while its rewards are read in the +y travel frame --
    silent in the config, and only visible ten hours into a run.
    """
    offset = getattr(cfg.commands.base_velocity, "heading_offset", None)
    if offset is None or not math.isclose(offset, -cfg.travel_yaw):
        raise ValueError(
            "sideways env cfg: commands.base_velocity.heading_offset is "
            f"{offset!r}, expected -travel_yaw = {-cfg.travel_yaw!r}. The command term is not a "
            "CrabHexSideVelocityCommandCfg (it was replaced with the forward CommandsCfg)."
        )


def _require_phase_teacher_mode(cls_name: str) -> None:
    """The sideways teacher only exists under a paradigm phase mode (2a / 2b / 2c).

    The legacy ``bridge`` / ``2b1`` / ``2b2`` / ``full*`` branches of
    ``_crab_hex_apply_teacher_mode`` install forward-only reward classes and terrain recipes and
    never consult ``_phase_rewards_cfg`` -- under them the sideways teacher would silently train
    on the forward tracking term. Fail at config build instead.
    """
    mode = _crab_hex_teacher_mode()
    if mode not in PHASE_TEACHER_MODES:
        raise ValueError(
            f"{cls_name}: KRABBY_HEX_TEACHER_MODE resolves to {mode!r}; the sideways teacher "
            f"supports only the paradigm phase modes {PHASE_TEACHER_MODES}. Launch through a "
            "KRABBY_SIDE_PHASE preset (2a / 2b / 2c), which exports the mode for you."
        )


def _apply_play(cfg) -> None:
    """The head every forward PLAY ``__post_init__`` shares (``crab_hex_env_cfg.py``).

    ``episode_length_s`` is set AFTER the train post-init, so it overrides ``KRABBY_EPISODE_S``
    exactly as the forward PLAY cfgs do.

    NOTE: the forward *teacher* PLAY additionally re-mixes the terrain -- but it returns early
    under ``_crab_hex_phase_mode_active()``, and the sideways teacher only ever runs under a phase
    mode (``_require_phase_teacher_mode``). That block is therefore a no-op here and is
    intentionally dropped rather than copied.
    """
    cfg.episode_length_s = 60.0
    cfg.parkours.base_parkour.debug_vis = True
    cfg.commands.base_velocity.debug_vis = True
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.max_init_terrain_level = None


# ---------------------------------------------------------------------------- env cfgs


@configclass
class CrabHexSideTeacherEnvCfg(CrabHexTeacherEnvCfg):
    """``Isaac-Crab-Hex-Side-Teacher-v0`` -- phases 2a / 2b / 2c of the sideways schedule.

    Every attribute below is the forward ``CrabHexTeacherEnvCfg`` literal, re-declared for knob
    timing (see the module docstring). ``rewards`` is the forward ``CrabHexRewardsCfg`` because
    that is what the forward class body holds; the phase branch replaces it in ``__post_init__``
    through ``_phase_rewards_cfg()``, which this class overrides.
    """

    viewer = CRAB_HEX_VIEWER
    scene: CrabHexSideTeacherSceneCfg = CrabHexSideTeacherSceneCfg(num_envs=6144, env_spacing=1.0)
    observations: CrabHexTeacherObservationsCfg = CrabHexTeacherObservationsCfg()
    actions: CrabHexFlatWalkActionsCfg = CrabHexFlatWalkActionsCfg()
    commands: CrabHexSideCommandsCfg = CrabHexSideCommandsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()
    travel_yaw: float = TRAVEL_YAW

    def _phase_rewards_cfg(self) -> CrabHexSideFlatWalkRewardsCfg:
        """Phases 2a/2b/2c wire the rewards HERE (``_crab_hex_apply_teacher_mode``).

        A method, not a dataclass field -- it does not appear in ``class_to_dict`` dumps.
        """
        return CrabHexSideFlatWalkRewardsCfg()

    def __post_init__(self):
        _require_phase_teacher_mode(type(self).__name__)
        super().__post_init__()
        _check_side(self)


@configclass
class CrabHexSideFlatWalkEnvCfg(CrabHexFlatWalkEnvCfg):
    """``Isaac-Crab-Hex-Side-Walk-v0`` -- phase 1a (gait formation) of the sideways schedule.

    Phase 1a wires the rewards through the class attribute, NOT through ``_phase_rewards_cfg``:
    ``CrabHexFlatWalkEnvCfg.__post_init__`` pins the teacher mode to ``"full"`` and never consults
    the hook. (It reaches the Go2 base through an explicit ``super(CrabHexTeacherEnvCfg, self)``
    skip, which resolves correctly for any subclass depth -- including this one.)
    """

    viewer = CRAB_HEX_VIEWER
    scene: CrabHexSideTeacherSceneCfg = CrabHexSideTeacherSceneCfg(num_envs=6144, env_spacing=1.0)
    observations: CrabHexTeacherObservationsCfg = CrabHexTeacherObservationsCfg()
    actions: CrabHexFlatWalkActionsCfg = CrabHexFlatWalkActionsCfg()
    commands: CrabHexSideCommandsCfg = CrabHexSideCommandsCfg()
    rewards: CrabHexSideFlatWalkRewardsCfg = CrabHexSideFlatWalkRewardsCfg()
    terminations: CrabHexFlatWalkTerminationsCfg = CrabHexFlatWalkTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()
    travel_yaw: float = TRAVEL_YAW

    def _phase_rewards_cfg(self) -> CrabHexSideFlatWalkRewardsCfg:
        """Inert today (the flat-walk chain pins mode ``"full"``), overridden so that a future
        forward change which does route flat-walk through the hook cannot hand this task the
        forward tracking term."""
        return CrabHexSideFlatWalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _check_side(self)


@configclass
class CrabHexSideStudentEnvCfg(CrabHexStudentEnvCfg):
    """``Isaac-Crab-Hex-Side-Student-v0`` -- phases 3a / 3b (depth distillation).

    The re-declared attributes are ``CrabHexStudentParkourEnvCfg``'s class-body literals.
    The forward ``__post_init__`` chooses the phase-3 MDP through the FORWARD
    ``is_student_phase()``, which reads ``KRABBY_PHASE`` / ``KRABBY_STUDENT_MDP`` only -- the
    sideways 3a/3b presets export ``KRABBY_STUDENT_MDP=1`` for exactly that reason (see
    ``crab_hex_side_phases.build_phases``); without it this would silently build the legacy 2b2
    MDP. The terrain it installs (``CrabHexTeacherSceneCfg().terrain``) is direction-agnostic and
    is inherited as-is.
    """

    scene: CrabHexSideStudentSceneCfg = CrabHexSideStudentSceneCfg(num_envs=192, env_spacing=1.0)
    observations: CrabHexStudentObservationsCfg = CrabHexStudentObservationsCfg()
    actions: CrabHexStudentActionsCfg = CrabHexStudentActionsCfg()
    commands: CrabHexSideCommandsCfg = CrabHexSideCommandsCfg()
    rewards: CrabHexStudentRewardsCfg = CrabHexStudentRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()
    travel_yaw: float = TRAVEL_YAW

    def __post_init__(self):
        super().__post_init__()
        _check_side(self)


# ---------------------------------------------------------------------------- PLAY variants


@configclass
class CrabHexSideFlatWalkEnvCfgPLAY(CrabHexSideFlatWalkEnvCfg):
    """Flat-walk visualization: follow-cam and command debug."""

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


@configclass
class CrabHexSideTeacherEnvCfgPLAY(CrabHexSideTeacherEnvCfg):
    """Teacher visualization / evaluation. Play must use the same phase mode as training."""

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


@configclass
class CrabHexSideStudentEnvCfgPLAY(CrabHexSideStudentEnvCfg):
    """Student visualization. ``KRABBY_HEX_PLAY_FLAT=1`` gives 100% flat tiles (MDP/obs unchanged)."""

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)
        self.events.push_by_setting_velocity = None
        if os.environ.get("KRABBY_HEX_PLAY_FLAT", "").strip().lower() in ("1", "true", "yes"):
            self.parkours.base_parkour.freeze_terrain_levels = True
            tg = getattr(self.scene.terrain, "terrain_generator", None) if self.scene.terrain else None
            if tg is not None:
                tg.curriculum = False
                tg.difficulty_range = (0.1, 0.25)
                for key, sub_terrain in tg.sub_terrains.items():
                    if key == "parkour_flat":
                        sub_terrain.proportion = 1.0
                    else:
                        sub_terrain.proportion = 0.0
