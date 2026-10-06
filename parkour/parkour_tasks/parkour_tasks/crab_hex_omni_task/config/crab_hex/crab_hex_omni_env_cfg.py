# SPDX-License-Identifier: BSD-3-Clause
"""Env cfgs for the omnidirectional task -- the forward env cfgs, omnidirectionalised (plan D12).

The delta against the forward task, in full (and the omni-vs-forward delta test asserts
exactly this list -- plan Appendix A):

* ``omni_command = True`` and ``omni_clock_turn_radius = 1.2`` -- two fields the forward cfg
  classes do NOT have. Every shared seam in ``parkour_isaaclab`` / the forward action term
  reads them through ``getattr(env.cfg, ..., <identity default>)``, so the forward and
  sideways tasks keep their bit-identical paths;
* ``commands`` -> the four-slot omni command term (world travel direction + speed, world
  facing), with ``clips.ang_vel_clip`` tightened 0.4 -> 0.2;
* ``observations`` -> the omni observation (facing error on every tile, the 2-D command and
  its norm, the yaw-rate command), same 1149 width and same term params;
* ``rewards`` -> the omni twins of four terms UNDER THE FORWARD NAMES, ``track_ang_vel_z_exp``'s
  ``std`` 0.5 -> 0.15, and ONE added term (``reward_yaw_progress``, the 36th name, dosed by
  ``KRABBY_YAW_PROGRESS_W`` in ``apply_omni_knobs``); ``terminations`` -> the goal cutoff removed;
* ``parkours.base_parkour`` -> ``OmniParkourEvent`` (ring exposure, tracking-ratio promotion,
  a goal index that never advances);
* ``events.reset_root_state`` -> the ring spawn (tile centre, uniform world yaw), and
  ``events.rsi_reference_reset`` re-pointed by :func:`~...omni_knobs.apply_omni_knobs`;
* ``scene`` -> the ring generator and the centred height scan (``crab_hex_omni_scene_cfg``).

There is no student cfg: this plan is teacher-only.

**KNOB TIMING (the pillar that makes this file look redundant).** ``import_packages`` imports
the forward package FIRST, so every forward cfg class body -- and with it every cfg object
those bodies construct -- is evaluated BEFORE the omni activation
(``crab_hex_omni_phases``) has exported the ``KRABBY_*`` knobs. A configclass default is a
deep copy of the object built in the class body, so simply inheriting ``scene`` /
``observations`` / ``actions`` / ``commands`` / ``rewards`` / ``terminations`` / ``parkours``
/ ``events`` from the forward classes would hand the omni task the UN-EXPANDED forward
defaults (wrong plant, wrong episode length, wrong terrain band, ...) with no error anywhere.
Therefore every cfg-valued class attribute is RE-DECLARED below with the forward class-body
literal, so each is constructed at *this* module's import -- which happens after
``config/crab_hex/__init__.py`` has run the omni activation. The omni-vs-forward delta test
proves each one still equals the forward instance built under the forward preset.
"""

import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import (
    CrabHexFlatWalkActionsCfg,
    CrabHexRewardsCfg,
    CrabHexTerminationsCfg,
    EventCfg,
)
from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_env_cfg import (
    CRAB_HEX_VIEWER,
    PHASE_TEACHER_MODES,
    CrabHexFlatWalkEnvCfg,
    CrabHexTeacherEnvCfg,
    _crab_hex_teacher_mode,
)
from parkour_tasks.crab_hex_omni_task.config.crab_hex.crab_hex_omni_scene_cfg import (
    CrabHexOmniTeacherSceneCfg,
)
from parkour_tasks.crab_hex_omni_task.config.crab_hex.omni_knobs import apply_omni_knobs
from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command import CrabHexOmniCommandsCfg
from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_observations import (
    CrabHexOmniParkourObservations,
    _check_omni,
)
from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_rewards import (
    CrabHexOmniFlatWalkRewardsCfg,
    CrabHexOmniFlatWalkTerminationsCfg,
)
from parkour_tasks.crab_hex_omni_task.mdp.omni_events import reset_root_state_ring
from parkour_tasks.crab_hex_omni_task.mdp.omni_parkour_event import OmniParkourEventsCfg
from parkour_tasks.crab_hex_omni_task.mdp.omni_terrain_knobs import install_omni_terrain_generator

#: Gait-clock turn radius (m): the lever arm the clock prices a yaw-rate command at, so that a
#: turn-in-place command still runs the gait. Knob ``KRABBY_OMNI_CLOCK_TURN_R`` (plan D3).
OMNI_CLOCK_TURN_RADIUS = 1.2

#: Spawn-yaw band for the ring reset event. Knob ``KRABBY_OMNI_YAW_RANGE``.
SPAWN_YAW_RANGE = (-math.pi, math.pi)


# ---------------------------------------------------------------------------- observations


@configclass
class CrabHexOmniObservationsCfg:
    """``CrabHexTeacherObservationsCfg``'s literal with the omni observation function.

    Same term name, same clip, same 1149-dim output, and the forward params plus the one
    omni-only key the encoding knob needs a home for (``facing_obs_encoding``): only the five
    slots the omni policy needs (facing error on every tile, ``cmd_vx_b`` / ``cmd_vy_b`` /
    ``|cmd|``, the yaw-rate command) are filled differently. Standalone rather than a subclass because
    the single member has to be rebuilt anyway (different function, and knob timing).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=CrabHexOmniParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
                "parkour_name": "base_parkour",
                "history_length": 10,
                # 2026-09-18 (plan Part A2): the facing-error encoding, DECLARED here at the
                # class default so ``KRABBY_OMNI_FACING_OBS_ENCODING`` has a key to overwrite
                # -- ``omni_knobs._apply_facing_obs_encoding_knob`` refuses a term that does
                # not already carry it, because writing a keyword the observation function
                # does not accept would kill the run inside the observation manager on its
                # first step. "raw" is bitwise today's observation.
                "facing_obs_encoding": "raw",
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------- parkour / events


@configclass
class CrabHexOmniParkoursCfg:
    """The ``base_parkour`` term, on the omni parkour event.

    The term NAME is load-bearing: every reward, observation and termination that reads the
    course names ``"base_parkour"`` in its params.
    """

    base_parkour = OmniParkourEventsCfg(asset_name="robot")


@configclass
class CrabHexOmniEventCfg(EventCfg):
    """The forward event block with ONLY ``reset_root_state`` re-pointed at the ring spawn.

    Re-declaring an inherited dataclass field keeps its POSITION in the parent's field order,
    which is the order the event manager applies reset-mode terms in -- so the spawn still
    runs before ``reset_robot_joints``, and ``rsi_reference_reset`` (created by the forward
    knob block and re-pointed in :func:`~...omni_knobs.apply_omni_knobs`) still runs after
    both. Nothing else changes: the startup DR terms, the joint reset and the push event are
    direction-agnostic and are inherited verbatim.

    ``offset`` is kept as a param name even though it is 0 here: ``ParkourEvent.__init__``
    hard-reads ``reset_root_state.params['offset']``, and the omni term inherits that
    ``__init__``.
    """

    reset_root_state = EventTerm(
        func=reset_root_state_ring,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "offset": 0.0,
            "yaw_range": SPAWN_YAW_RANGE,
        },
        mode="reset",
    )


# ---------------------------------------------------------------------------- helpers


def _require_phase_teacher_mode(cls_name: str) -> None:
    """The omni teacher only exists under a paradigm phase mode (2a / 2b / 2c).

    The legacy ``bridge`` / ``2b1`` / ``2b2`` / ``full*`` branches of
    ``_crab_hex_apply_teacher_mode`` install forward-only reward classes and corridor terrain
    recipes and never consult ``_phase_rewards_cfg`` -- under them the omni teacher would
    silently train on the forward tracking terms, on a corridor, with an omni command. Fail at
    config build instead. (Same guard as the sideways teacher's.)
    """
    mode = _crab_hex_teacher_mode()
    if mode not in PHASE_TEACHER_MODES:
        raise ValueError(
            f"{cls_name}: KRABBY_HEX_TEACHER_MODE resolves to {mode!r}; the omni teacher "
            f"supports only the paradigm phase modes {PHASE_TEACHER_MODES}. Launch through a "
            "KRABBY_OMNI_PHASE preset (2a / 2b / 2c), which exports the mode for you."
        )


def _install_omni_total_terminates(cfg) -> None:
    """Put the goal-free episode termination back after the phase-2 branch overwrote it.

    ``_crab_hex_apply_teacher_mode``'s paradigm branch assigns
    ``self.terminations = CrabHexFlatWalkTerminationsCfg()`` by name -- there is no
    ``_phase_terminations_cfg`` hook beside ``_phase_rewards_cfg`` -- so the omni teacher would
    silently inherit the forward ``terminate_episode``, whose ``cur_goal_idx`` cutoff ends an
    episode as soon as the robot drifts past a ring sector anchor it was never commanded to.

    Only the ONE field is swapped, not the whole block: ``apply_flat_walk_knobs`` has already
    written ``KRABBY_TERM_ANGLE`` / ``KRABBY_TERM_CONTACT_N`` into ``crab_failure.params`` by
    the time this runs, and replacing the container would drop them. The result is dump-identical
    to the flat-walk task's ``CrabHexOmniFlatWalkTerminationsCfg`` (``class_to_dict`` records the
    terms, not the container class) -- pinned by the phase-2-teacher == Omni-Walk test.
    """
    cfg.terminations.total_terminates = CrabHexOmniFlatWalkTerminationsCfg().total_terminates


PLAY_EPISODE_S_VAR = "KRABBY_PLAY_EPISODE_S"
PLAY_EPISODE_S_DEFAULT = 60.0


def play_episode_length_s(environ) -> float:
    """PLAY episode length: ``KRABBY_PLAY_EPISODE_S`` (seconds, > 0) or the forward PLAY literal 60 s.

    The PLAY head overwrites ``episode_length_s`` AFTER the train post-init, so the training
    knob ``KRABBY_EPISODE_S`` cannot reach a viewer/recording; this PLAY-only knob can
    (2026-09-18: continuous recordings with "no resets except a fall" need 300 s).
    """
    raw = environ.get(PLAY_EPISODE_S_VAR)
    if raw is None or str(raw).strip() == "":
        return PLAY_EPISODE_S_DEFAULT
    val = float(raw)
    if not val > 0.0:
        raise RuntimeError(f"{PLAY_EPISODE_S_VAR} must be > 0 s, got {raw!r}")
    return val


PLAY_DEBUG_VIS_VAR = "KRABBY_PLAY_DEBUG_VIS"


def play_debug_vis(environ) -> bool:
    """``KRABBY_PLAY_DEBUG_VIS`` -> whether the PLAY cfg turns the command/parkour debug markers on.

    Absent or "1" -> True (viewers, recordings). "0" -> False (the eval harness: headless, 100 envs).
    Anything else raises so a typo cannot silently pick a default.
    """
    raw = environ.get(PLAY_DEBUG_VIS_VAR)
    if raw is None or str(raw).strip() in ("", "1"):
        return True
    if str(raw).strip() == "0":
        return False
    raise RuntimeError(f"{PLAY_DEBUG_VIS_VAR} must be '0' or '1', got {raw!r}")


def _apply_play(cfg) -> None:
    """The head every forward PLAY ``__post_init__`` shares (``crab_hex_env_cfg.py``).

    ``episode_length_s`` is set AFTER the train post-init, so it overrides ``KRABBY_EPISODE_S``
    exactly as the forward PLAY cfgs do.

    NOTE: the forward *teacher* PLAY additionally re-mixes the terrain -- but it returns early
    under ``_crab_hex_phase_mode_active()``, and the omni teacher only ever runs under a phase
    mode (``_require_phase_teacher_mode``). That block is therefore a no-op here and is
    intentionally dropped rather than copied.
    """
    cfg.episode_length_s = play_episode_length_s(os.environ)
    # Debug markers (viewer arrows / ring anchors) are for humans: viewers and recordings keep them
    # (default "1"); the headless eval harness runs 100 envs and must not pay for them --
    # 2026-09-18: with the markers on, 100-env evals grew to ~59 GB RSS and were OOM-killed.
    dv = play_debug_vis(os.environ)
    cfg.parkours.base_parkour.debug_vis = dv
    cfg.commands.base_velocity.debug_vis = dv
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.max_init_terrain_level = None


# ---------------------------------------------------------------------------- env cfgs


@configclass
class CrabHexOmniTeacherEnvCfg(CrabHexTeacherEnvCfg):
    """``Isaac-Crab-Hex-Omni-Teacher-v0`` -- phases 2a / 2b / 2c of the omni schedule.

    Every attribute below is the forward ``CrabHexTeacherEnvCfg`` literal with the omni class
    substituted where the plan calls for one, re-declared for knob timing (see the module
    docstring). ``rewards`` is the forward ``CrabHexRewardsCfg`` because that is what the
    forward class body holds; the phase branch replaces it in ``__post_init__`` through
    ``_phase_rewards_cfg()``, which this class overrides.
    """

    viewer = CRAB_HEX_VIEWER
    scene: CrabHexOmniTeacherSceneCfg = CrabHexOmniTeacherSceneCfg(num_envs=6144, env_spacing=1.0)
    observations: CrabHexOmniObservationsCfg = CrabHexOmniObservationsCfg()
    actions: CrabHexFlatWalkActionsCfg = CrabHexFlatWalkActionsCfg()
    commands: CrabHexOmniCommandsCfg = CrabHexOmniCommandsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: CrabHexOmniParkoursCfg = CrabHexOmniParkoursCfg()
    events: CrabHexOmniEventCfg = CrabHexOmniEventCfg()
    omni_command: bool = True
    omni_clock_turn_radius: float = OMNI_CLOCK_TURN_RADIUS

    def _phase_rewards_cfg(self) -> CrabHexOmniFlatWalkRewardsCfg:
        """Phases 2a/2b/2c wire the rewards HERE (``_crab_hex_apply_teacher_mode``).

        A method, not a dataclass field -- it does not appear in ``class_to_dict`` dumps.
        """
        return CrabHexOmniFlatWalkRewardsCfg()

    def __post_init__(self):
        _require_phase_teacher_mode(type(self).__name__)
        # Omni phase-2 plan step 2 (2026-10-06): KRABBY_FLAT_TERRAIN_GEOM=field swaps in the field
        # generator FIRST, so the forward knob block below (flat fraction, difficulty band,
        # curriculum) applies to it as it does to the ring. ``ring`` is a no-op.
        install_omni_terrain_generator(self, os.environ)
        super().__post_init__()
        _install_omni_total_terminates(self)
        apply_omni_knobs(self)
        _check_omni(self)


@configclass
class CrabHexOmniFlatWalkEnvCfg(CrabHexFlatWalkEnvCfg):
    """``Isaac-Crab-Hex-Omni-Walk-v0`` -- phase-1 stages 1a / 1b / 1c of the omni schedule.

    Phase 1 wires the rewards through the class attribute, NOT through ``_phase_rewards_cfg``:
    ``CrabHexFlatWalkEnvCfg.__post_init__`` pins the teacher mode to ``"full"`` and never
    consults the hook. (It reaches the Go2 base through an explicit
    ``super(CrabHexTeacherEnvCfg, self)`` skip, which resolves correctly for any subclass
    depth -- including this one.)
    """

    viewer = CRAB_HEX_VIEWER
    scene: CrabHexOmniTeacherSceneCfg = CrabHexOmniTeacherSceneCfg(num_envs=6144, env_spacing=1.0)
    observations: CrabHexOmniObservationsCfg = CrabHexOmniObservationsCfg()
    actions: CrabHexFlatWalkActionsCfg = CrabHexFlatWalkActionsCfg()
    commands: CrabHexOmniCommandsCfg = CrabHexOmniCommandsCfg()
    rewards: CrabHexOmniFlatWalkRewardsCfg = CrabHexOmniFlatWalkRewardsCfg()
    terminations: CrabHexOmniFlatWalkTerminationsCfg = CrabHexOmniFlatWalkTerminationsCfg()
    parkours: CrabHexOmniParkoursCfg = CrabHexOmniParkoursCfg()
    events: CrabHexOmniEventCfg = CrabHexOmniEventCfg()
    omni_command: bool = True
    omni_clock_turn_radius: float = OMNI_CLOCK_TURN_RADIUS

    def _phase_rewards_cfg(self) -> CrabHexOmniFlatWalkRewardsCfg:
        """Inert today (the flat-walk chain pins mode ``"full"``), overridden so that a future
        forward change which does route flat-walk through the hook cannot hand this task the
        forward reward functions."""
        return CrabHexOmniFlatWalkRewardsCfg()

    def __post_init__(self):
        # Omni phase-2 plan step 2: the field generator, if selected, before anything else (see the
        # teacher cfg). ``ring`` is a no-op.
        install_omni_terrain_generator(self, os.environ)
        super().__post_init__()
        apply_omni_knobs(self)
        _check_omni(self)


# ---------------------------------------------------------------------------- PLAY variants


@configclass
class CrabHexOmniFlatWalkEnvCfgPLAY(CrabHexOmniFlatWalkEnvCfg):
    """Omni-walk visualization: follow-cam and command debug."""

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


@configclass
class CrabHexOmniTeacherEnvCfgPLAY(CrabHexOmniTeacherEnvCfg):
    """Teacher visualization / evaluation. Play must use the same phase mode as training."""

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)
