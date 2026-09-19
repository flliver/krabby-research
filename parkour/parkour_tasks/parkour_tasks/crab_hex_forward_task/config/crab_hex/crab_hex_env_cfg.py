import os
from pathlib import Path

from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import (
    CommandsCfg,
    CrabHexFlatWalkActionsCfg,
    CrabHexFlatWalkRewardsCfg,
    CrabHexFlatWalkTerminationsCfg,
    CrabHexRewardsCfg,
    CrabHexTeacherObservationsCfg,
    CrabHexStage2BPhase1RewardsCfg,
    CrabHexStage2BPhase2RewardsCfg,
    CrabHexTeacherBridgeRewardsCfg,
    CrabHexTerminationsCfg,
    EventCfg,
    ParkourEventsCfg,
    _crab_action_clip,
    _crab_action_scale,
)
from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_scene_cfg import CrabHexTeacherSceneCfg
from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_student_cfg import (
    CrabHexStudentParkourEnvCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import (
    UnitreeGo2TeacherParkourEnvCfg,
)

# Front 3/4 view: FL/FR at −x; Go2 ``VIEWER`` is a tight +y side shot.
CRAB_HEX_VIEWER = ViewerCfg(
    eye=(-4.0, 0.5, 1.55),
    lookat=(0.0, 0.0, 0.35),
    asset_name="robot",
    origin_type="asset_root",
)
# Top view: directly above root (raise z for wider view)
# CRAB_HEX_VIEWER = ViewerCfg(
#     eye=(0.0, 0.0, 6.0),      # directly above root (raise z for wider view)
#     lookat=(0.0, 0.0, 0.35),  # same as now — chassis height
#     asset_name="robot",
#     origin_type="asset_root",
# )

# ---------------------------------------------------------------------------
# ``KRABBY_HEX_TEACHER_MODE`` — teacher curriculum (``Isaac-Crab-Hex-Teacher-v0``)
#
# Set before train/play:  export KRABBY_HEX_TEACHER_MODE=<mode>
# Unset or omit for:     default full parkour teacher (``full``).
#
# Paradigm phases (2026-09-07, pipeline of record): ``KRABBY_HEX_TEACHER_MODE=2a|2b|2c`` (exported
# by ``KRABBY_PHASE``) = the flat-walk MDP + that window's elements (``apply_flat_walk_knobs``); the
# student (3a/3b) distills from the 2c teacher. The staged chain below (bridge -> 2b1 -> 2b2 ->
# full1 -> full2 -> full) is the LEGACY May-2026 recipe kept for reproduction; ``full1``/``full2``
# are ramp stages between 2b2 and ``full``.
#
# Pipeline (checkpoint chain):
#   Stage 1  Flat walk     →  task ``Isaac-Crab-Hex-Flat-Walk-v0`` (NOT this flag)
#   Stage 2a bridge        →  ``bridge``   resume flat ``model_6000``
#   Stage 2b phase 1       →  ``2b1``      resume bridge ``model_6099``
#   Stage 2b phase 2       →  ``2b2``      teacher-ready obstacle walk (distillation source)
#   Stage 3  Student       →  ``Isaac-Crab-Hex-Student-v0``  distill from 2b2 teacher
#   Stage 4  Full parkour  →  ``full``     TODO — only after 2b2 student pipeline is stable
#
# --- bridge (Appendix D) — “easy mixed walk” ---
#   Intent: Keep the flat-walk gait on mostly flat ground with a little shallow
#   parkour geometry; learn velocity + posture, not goal chasing yet.
#   Terrain: ~82% flat tiles, ~18% shallow gaps/pits; difficulty 0.08–0.30;
#            terrain level frozen (no curriculum demotion).
#   Actions: scale 0.24, clip ±1 (same family as flat-walk).
#   Rewards: ``CrabHexTeacherBridgeRewardsCfg`` — forward speed/progress, upright,
#            anti-stall; parkour goal/yaw terms OFF.
#   Typical play: stable forward walk on flat + light tiles; some heading drift OK.
#
# --- 2b1 — “hybrid walk + light parkour hints” ---
#   Intent: Same bridge-lite physics/terrain as ``bridge``, but gently introduce
#   parkour goal velocity and yaw (aux weights) plus teacher body regularizers.
#   Terrain/actions/events: same as ``bridge``.
#   Rewards: ``CrabHexStage2BPhase1RewardsCfg`` — bridge core + goal_vel 0.75, yaw 0.2.
#   Resume: always from bridge ``model_6099`` (do not use ``full`` from 6099).
#
# --- 2b2 — “teacher-ready obstacle walk” ---
#   Intent: Polished 2b2-phase-2 policy for **student distillation** (robustness over raw speed).
#   Terrain: 50/50 flat/parkour; curriculum 0.20–0.70; moderate geometry; actions 0.24, ±1.
#   Rewards: ``CrabHexStage2BPhase2RewardsCfg`` — clearance **+1.8**, foot **+2.0**, swing-vz **+0.4**;
#            recover **+0.4**, micro-swing **−0.2**, forward **+0.25**, low-speed **−0.8**.
#   Stop at sweet-spot checkpoint (play + metric gates); bundle as 2b2-teacher before student train.
#   Bundled teacher: Appendix F ``2026-05-26_21-46-37/model_6300.pt`` (supersedes ``2026-05-26_11-30-18``).
#
# --- full (TODO stage 4) — “Go2-style parkour teacher” ---
#   Intent: Full extreme-parkour teacher MDP (goal velocity primary).
#   Terrain: full sub-terrain mix, difficulty 0.0–1.0, curriculum on.
#   Actions: scale 0.25, clip ±4.8; push/mass/COM domain randomization on.
#   Rewards: ``CrabHexRewardsCfg`` (goal_vel 2.25, collision -6, …).
#   Warning: resuming bridge/2b1 checkpoints into ``full`` without staging thrashes.
#
# Play must use the same ``KRABBY_HEX_TEACHER_MODE`` as training for that checkpoint.
# ---------------------------------------------------------------------------


def _crab_hex_teacher_mode() -> str:
    """Resolve ``KRABBY_HEX_TEACHER_MODE`` → ``2a`` | ``2b`` | ``2c`` (paradigm phases) | ``bridge`` |
    ``2b1`` | ``2b2`` | ``full1`` | ``full2`` | ``full`` (see module comment above).

    ``full1``/``full2`` are intermediate ramp stages between 2b2 and true ``full`` -- jumping
    straight from 2b2's bridge-lite MDP to full's (terrain 0-1, full domain randomization,
    0.25/±4.8 actions, strict 500N failure threshold) all at once collapses training almost
    immediately (verified empirically: crab_failure pinned at 100% within ~50 iterations, never
    recovering). Each ramp stage narrows the gap on every axis at once by a smaller amount.
    """
    raw = os.environ.get("KRABBY_HEX_TEACHER_MODE", "").strip().lower()
    if raw in ("bridge",):
        return "bridge"
    if raw in ("2b1", "2b_1", "stage2b1", "stage2b-1", "stage2b_1"):
        return "2b1"
    if raw in ("2b2", "2b_2", "stage2b2", "stage2b-2", "stage2b_2"):
        return "2b2"
    if raw in ("full1", "full_1", "full-1", "fullramp1", "full-ramp-1"):
        return "full1"
    if raw in ("full2", "full_2", "full-2", "fullramp2", "full-ramp-2"):
        return "full2"
    if raw in PHASE_TEACHER_MODES:      # paradigm phases 2a / 2b / 2c (2026-09-07)
        return raw
    return "full"


PHASE_TEACHER_MODES = ("2a", "2b", "2c")


def _crab_hex_phase_mode_active() -> bool:
    """Phase-2 teacher mode: the flat-walk MDP + elements (train and play must match)."""
    return _crab_hex_teacher_mode() in PHASE_TEACHER_MODES


def _crab_hex_bridge_like_mdp_active() -> bool:
    """Train/play uses bridge-lite physics (scale 0.24, ±1), not default teacher 0.25 / ±4.8."""
    return _crab_hex_teacher_mode() in ("bridge", "2b1", "2b2")


def _teacher_lin_vel_x_band() -> tuple[float, float]:
    """Teacher-stage command band, KRABBY_LIN_VEL_X-overridable (gait-formation-v2 hand-off,
    2026-08-26). The historical (0.45, 0.85) band belongs to the 6 rad/s-cam plant (kinematic
    ceiling ~1.1 m/s); the measured-hardware plant tops out near 0.5 m/s at full cam speed, so
    the old band demands untrackable speeds. Presence-based override, same syntax as flat-walk.
    """
    raw = os.environ.get("KRABBY_LIN_VEL_X")
    if raw is not None:
        lo, hi = (float(x) for x in raw.split(":"))
        return (lo, hi)
    return (0.45, 0.85)


def _apply_crab_hex_stage_2b_bridge_lite_env(cfg, *, action_scale: float = 0.24) -> None:
    """Shared bridge-lite physics/events/terminations for stage-2b (phase 1 and 2)."""
    _apply_crab_hex_bridge_actions_and_events(cfg, action_scale=action_scale)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.heading_control_stiffness = 1.5
    cfg.commands.base_velocity.ranges.lin_vel_x = _teacher_lin_vel_x_band()


def _apply_crab_hex_stage_2b_phase1_terrain(cfg) -> None:
    """Bridge-equivalent terrain: frozen levels, easy mix, shallow gaps."""
    cfg.parkours.base_parkour.freeze_terrain_levels = True
    tg = getattr(cfg.scene.terrain, "terrain_generator", None) if cfg.scene.terrain else None
    if tg is not None:
        _apply_crab_hex_easy_mixed_terrain(tg, flat_proportion=0.825, difficulty_range=(0.08, 0.30))
        _apply_crab_hex_bridge_shallow_parkour_geometry(tg)


def _apply_crab_hex_stage_2b_phase2_parkour_geometry(tg) -> None:
    """Moderate parkour geometry for 2b2 (between bridge-shallow and full teacher).

    TODO(hardware-measurements, 2026-08-20): these obstacle scales (and the Go2-derived
    EXTREME_PARKOUR_TERRAINS_CFG defaults) were sized for the OLD half-size robot. The
    measured robot is ~2x taller with ~40% longer legs and 10-25x slower joints -- hurdles
    and steps at these heights are likely trivial while gaps may bind differently.
    Re-scale deliberately when the retrain campaign reaches parkour stages; the flat-walk
    stages are unaffected."""
    if "parkour_gap" in tg.sub_terrains:
        gap = tg.sub_terrains["parkour_gap"]
        gap.gap_depth = (0.08, 0.18)
        gap.gap_size = "0.10 + 0.45 * difficulty"
        gap.half_valid_width = (0.85, 1.15)
    if "parkour" in tg.sub_terrains:
        stone = tg.sub_terrains["parkour"]
        stone.pit_depth = (0.08, 0.18)
        stone.incline_height = "0.20*difficulty"
        stone.last_incline_height = "incline_height + 0.08 - 0.06*difficulty"
    if "parkour_step" in tg.sub_terrains:
        tg.sub_terrains["parkour_step"].step_height = "0.12 + 0.28*difficulty"
    if "parkour_hurdle" in tg.sub_terrains:
        tg.sub_terrains["parkour_hurdle"].hurdle_height_range = (
            "0.12+0.10*difficulty, 0.16+0.20*difficulty"
        )


def _apply_crab_hex_stage_2b_phase2_terrain(cfg) -> None:
    """Phase 2: curriculum on, 50/50 mix, gently ramping obstacle difficulty."""
    cfg.parkours.base_parkour.freeze_terrain_levels = False
    tg = getattr(cfg.scene.terrain, "terrain_generator", None) if cfg.scene.terrain else None
    if tg is not None:
        tg.curriculum = True
        tg.difficulty_range = (0.20, 0.70)
        active = [k for k in tg.sub_terrains if k not in ("parkour_flat", "parkour_demo")]
        n_other = len(active)
        share = (0.5 / n_other) if n_other else 0.0
        for key, sub_terrain in tg.sub_terrains.items():
            if key == "parkour_flat":
                sub_terrain.proportion = 0.5
            elif key == "parkour_demo":
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = share
            sub_terrain.noise_range = (0.02, 0.02)
        _apply_crab_hex_stage_2b_phase2_parkour_geometry(tg)


def _apply_crab_hex_student_2b2_teacher_mdp(cfg) -> None:
    """Student distillation MDP aligned with ``KRABBY_HEX_TEACHER_MODE=2b2`` teacher train.

    Same terrain mix, difficulty, geometry, commands, and bridge-lite DR as 2b2 teacher.
    Keeps student action delay (``use_delay=True``) — not overwritten by bridge-lite actions.
    """
    _apply_crab_hex_stage_2b_phase2_terrain(cfg)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.heading_control_stiffness = 1.5
    cfg.commands.base_velocity.ranges.lin_vel_x = _teacher_lin_vel_x_band()
    cfg.events.push_by_setting_velocity = None
    cfg.events.randomize_rigid_body_mass = None
    cfg.events.randomize_rigid_body_com = None
    cfg.terminations.crab_failure.params["contact_force_threshold"] = 800.0
    tg = getattr(cfg.scene.terrain, "terrain_generator", None) if cfg.scene.terrain else None
    if tg is not None:
        tg.horizontal_scale = 0.08
        for sub_terrain in tg.sub_terrains.values():
            sub_terrain.horizontal_scale = 0.08
            sub_terrain.use_simplified = False


def _apply_crab_hex_bridge_actions_and_events(cfg, *, action_scale: float = 0.24) -> None:
    """Flat-walk-compatible actions/events for the flat-walk → teacher bridge."""
    cfg.actions.joint_pos.scale = _crab_action_scale(action_scale)
    cfg.actions.joint_pos.clip = _crab_action_clip((-1.0, 1.0))
    cfg.actions.joint_pos.use_delay = False
    cfg.actions.joint_pos.history_length = 1

    cfg.events.push_by_setting_velocity = None
    cfg.events.randomize_rigid_body_mass = None
    cfg.events.randomize_rigid_body_com = None

    cfg.terminations.crab_failure.params["contact_force_threshold"] = 800.0


def _apply_crab_hex_full_actions(cfg) -> None:
    """Flat-walk / phase 2a-2c / phase-3 student / ``full`` action config: 0.25 scale
    (``KRABBY_ACTION_SCALE``) / ±4.8 raw clip, 1-step action delay on -- matches the original
    Go2-imported ``ActionsCfg`` this repo used before the cam-mechanism migration. Only the legacy
    bridge / 2b1 / 2b2 modes override to 0.24 / ±1 (see ``_apply_crab_hex_bridge_actions_and_events``).
    ``cfg.actions.joint_pos.joint_names`` itself is unaffected -- already fixed at the class-level
    default (``CrabHexFlatWalkActionsCfg``) to exclude the passive ``*_Body_Hip_RevoluteJoint``.
    """
    # NOTE(gait-formation-v2 Phase 4, 2026-08-23): KRABBY_ACTION_SCALE overrides the pitch
    # joints' target authority (rad). The default 0.25 caps toe lift at ~4.6 cm (analytic,
    # matches G1-G3's measured 4.2-4.7 plateau across a 4x clearance-income range); the
    # user-approved lift route raises it. Cam channels keep CAM_VEL_SCALE regardless.
    import os as _os

    _asc = _os.environ.get("KRABBY_ACTION_SCALE")
    cfg.actions.joint_pos.scale = _crab_action_scale(float(_asc) if _asc else 0.25)
    cfg.actions.joint_pos.clip = _crab_action_clip((-4.8, 4.8))
    cfg.actions.joint_pos.use_delay = True
    cfg.actions.joint_pos.history_length = 8


def _apply_crab_hex_full_ramp1_actions(cfg) -> None:
    """full-ramp-1: 0.245 scale / ±2.4 clip -- midpoint between 2b2-lite (0.24/±1) and true full
    (0.25/±4.8), so the policy adapts to a larger raw action range gradually instead of in one jump.
    """
    cfg.actions.joint_pos.scale = _crab_action_scale(0.245)
    cfg.actions.joint_pos.clip = _crab_action_clip((-2.4, 2.4))
    cfg.actions.joint_pos.use_delay = True
    cfg.actions.joint_pos.history_length = 8


def _apply_crab_hex_full_ramp2_actions(cfg) -> None:
    """full-ramp-2: 0.25 scale / ±3.6 clip -- most of the way to true full's ±4.8."""
    cfg.actions.joint_pos.scale = _crab_action_scale(0.25)
    cfg.actions.joint_pos.clip = _crab_action_clip((-3.6, 3.6))
    cfg.actions.joint_pos.use_delay = True
    cfg.actions.joint_pos.history_length = 8


def _apply_crab_hex_full_ramp1_dr(cfg) -> None:
    """Half-strength push/mass/com domain randomization (vs. Go2's original EventCfg ranges --
    see push_by_setting_velocity/randomize_rigid_body_mass/randomize_rigid_body_com in
    extreme_parkour_task/config/go2/parkour_mdp_cfg.py) -- 2b2 disables these entirely, true full
    uses them at 100%; ramp1 uses half so the jump isn't instant.
    """
    if cfg.events.push_by_setting_velocity is not None:
        cfg.events.push_by_setting_velocity.params["velocity_range"] = {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}
    if cfg.events.randomize_rigid_body_mass is not None:
        cfg.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (-0.5, 1.5)
    if cfg.events.randomize_rigid_body_com is not None:
        cfg.events.randomize_rigid_body_com.params["com_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}


def _apply_crab_hex_full_ramp1_terrain(cfg) -> None:
    """full-ramp-1: curriculum on, difficulty (0.15, 0.60), 40% flat, gentle 2b2-style obstacle
    geometry (not yet raw Go2 formulas) -- a smaller step up from 2b2's own (0.20, 0.70) mix.
    """
    cfg.parkours.base_parkour.freeze_terrain_levels = False
    tg = getattr(cfg.scene.terrain, "terrain_generator", None) if cfg.scene.terrain else None
    if tg is not None:
        tg.curriculum = True
        tg.difficulty_range = (0.15, 0.60)
        active = [k for k in tg.sub_terrains if k not in ("parkour_flat", "parkour_demo")]
        n_other = len(active)
        share = (0.60 / n_other) if n_other else 0.0
        for key, sub_terrain in tg.sub_terrains.items():
            if key == "parkour_flat":
                sub_terrain.proportion = 0.40
            elif key == "parkour_demo":
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = share
        _apply_crab_hex_stage_2b_phase2_parkour_geometry(tg)


def _apply_crab_hex_full_ramp2_terrain(cfg) -> None:
    """full-ramp-2: curriculum on, difficulty (0.05, 0.90), 15% flat, raw (unmodified) Go2
    ``EXTREME_PARKOUR_TERRAINS_CFG`` obstacle geometry -- the last step before true full's (0, 1).
    """
    cfg.parkours.base_parkour.freeze_terrain_levels = False
    tg = getattr(cfg.scene.terrain, "terrain_generator", None) if cfg.scene.terrain else None
    if tg is not None:
        tg.curriculum = True
        tg.difficulty_range = (0.05, 0.90)
        active = [k for k in tg.sub_terrains if k not in ("parkour_flat", "parkour_demo")]
        n_other = len(active)
        share = (0.85 / n_other) if n_other else 0.0
        for key, sub_terrain in tg.sub_terrains.items():
            if key == "parkour_flat":
                sub_terrain.proportion = 0.15
            elif key == "parkour_demo":
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = share


def _apply_crab_hex_easy_mixed_terrain(
    tg, *, flat_proportion: float, difficulty_range: tuple[float, float]
) -> None:
    """``parkour_flat`` + easy parkour sub-terrains (e.g. 50/50 or 70/30)."""
    tg.curriculum = False
    tg.difficulty_range = difficulty_range
    active = [k for k in tg.sub_terrains if k not in ("parkour_flat", "parkour_demo")]
    n_other = len(active)
    share = ((1.0 - flat_proportion) / n_other) if n_other else 0.0
    for key, sub_terrain in tg.sub_terrains.items():
        if key == "parkour_flat":
            sub_terrain.proportion = flat_proportion
        elif key == "parkour_demo":
            sub_terrain.proportion = 0.0
        else:
            sub_terrain.proportion = share
        sub_terrain.noise_range = (0.02, 0.02)


def _apply_crab_hex_bridge_shallow_parkour_geometry(tg) -> None:
    """Shallow, narrow gaps/pits for bridge (depth is not scaled by ``difficulty_range``)."""
    if "parkour_gap" in tg.sub_terrains:
        gap = tg.sub_terrains["parkour_gap"]
        gap.gap_depth = (0.05, 0.12)
        gap.gap_size = "0.08 + 0.35 * difficulty"
        gap.half_valid_width = (0.9, 1.2)
    if "parkour" in tg.sub_terrains:
        tg.sub_terrains["parkour"].pit_depth = (0.05, 0.12)


def _apply_crab_hex_recal_2b2_parkour_geometry(tg) -> None:
    """PLAN F (2026-08-26): the 2b2 end-state geometry RE-DERIVED for the measured plant,
    resolving the old TODO — the stock 2b2 numbers were sized for the half-size robot.
    Anchors: tibia 0.83 m (spans wide gaps), swing lift ~0.05 typical / 0.09 max (steps
    and hurdles must fit inside that envelope or they are unnegotiable by construction),
    steady walk 0.10-0.20 m/s.
    """
    if "parkour_gap" in tg.sub_terrains:
        gap = tg.sub_terrains["parkour_gap"]
        gap.gap_depth = (0.06, 0.14)
        gap.gap_size = "0.10 + 0.25 * difficulty"  # 0.15-0.275 m over d 0.2-0.7
        gap.half_valid_width = (0.85, 1.15)
    if "parkour" in tg.sub_terrains:
        stone = tg.sub_terrains["parkour"]
        stone.pit_depth = (0.06, 0.14)
        stone.incline_height = "0.10 * difficulty"
        stone.last_incline_height = "incline_height + 0.04 - 0.03 * difficulty"
    if "parkour_step" in tg.sub_terrains:
        tg.sub_terrains["parkour_step"].step_height = "0.02 + 0.06 * difficulty"  # 3.2-6.2 cm
    if "parkour_hurdle" in tg.sub_terrains:
        # Single eval'd expression string (terrain generator evals the comma expression
        # into the tuple); capped under the 0.09 lift envelope.
        tg.sub_terrains["parkour_hurdle"].hurdle_height_range = (
            "0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty"
        )


def _apply_crab_hex_recal_2b2w_parkour_geometry(tg) -> None:
    """PLAN H B0a (2026-09-03): recal2b2 with the obstacle corridors WIDENED to fit the
    robot. The recal2b2 half-widths (gap 0.85-1.15, hurdle 0.4-0.8, step 0.5-1.0, stones
    0.5 m) are all below the crab's 1.19 m half-stance, so the outer feet ride the 0.06-0.14 m
    side trench from x ~ 2.5 m on -- the terrain z under the feet at the moment of fall and
    why light ~= hard (0.26 vs 0.27). (1.40, 1.70) clears the stance with margin (tile 4.0 m
    wide, bound 1.85). The corridor's lateral offset per segment is narrowed from +-0.4 to
    +-0.2 m (stones: 0.08 m) because the crab is blind to it: the corridor must cover
    half-stance + |offset|, and at least one side trench (>= 0.22 m) always remains.
    ``recal2b2`` stays for reproducing old runs.
    """
    from parkour_tasks.crab_hex_forward_task.mdp import exposure_knobs as _xk

    _apply_crab_hex_recal_2b2_parkour_geometry(tg)
    _xk.apply_corridor_widths(tg.sub_terrains, _xk.RECAL2B2W_HALF_VALID_WIDTH, _xk.RECAL2B2W_STONE_WIDTH,
                              y_range=_xk.RECAL2B2W_Y_RANGE, stone_y_range=_xk.RECAL2B2W_STONE_Y_RANGE)


@configclass
class CrabHexTeacherEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    viewer = CRAB_HEX_VIEWER
    scene: CrabHexTeacherSceneCfg = CrabHexTeacherSceneCfg(num_envs=6144, env_spacing=1.0)
    observations: CrabHexTeacherObservationsCfg = CrabHexTeacherObservationsCfg()
    # NOTE(cam-mechanism-migration): reuses CrabHexFlatWalkActionsCfg (not the Go2-imported
    # ``ActionsCfg``) -- the cam-shaft joints replace ``Body_Hip`` as the driven yaw DOF, so the
    # action space must exclude the now-passive ``Body_Hip`` joints the same way flat-walk's does.
    # Also required for checkpoint-shape compatibility when resuming Stage 2a from the flat-walk
    # checkpoint (both must have the same 18-dim action space).
    actions: CrabHexFlatWalkActionsCfg = CrabHexFlatWalkActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.enable_external_forces_every_iteration = True
        # Skew velocity commands toward meaningful forward speed (reduces near-zero command_vel in rewards).
        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.85)
        base_body_cfg = SceneEntityCfg("robot", body_names="body")
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_mass is not None:
            self.events.randomize_rigid_body_mass.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_com is not None:
            self.events.randomize_rigid_body_com.params["asset_cfg"] = base_body_cfg

        self._crab_hex_apply_teacher_mode(_crab_hex_teacher_mode())

    def _crab_hex_apply_teacher_mode(self, mode: str) -> None:
        """Mode-specific MDP (rewards / terrain / actions). ``CrabHexFlatWalkEnvCfg`` calls this
        with ``"full"`` regardless of ``KRABBY_HEX_TEACHER_MODE`` (a ``KRABBY_PHASE`` preset exports
        a phase-2 mode; the flat-walk task must not apply the phase branch AND its own knobs)."""
        if mode in PHASE_TEACHER_MODES:
            # Paradigm phase 2 (teacher-student): the flat-walk MDP with the phase's elements,
            # built from the same knobs the flat-walk task uses (see apply_flat_walk_knobs).
            # Mirrors ``CrabHexFlatWalkEnvCfg`` exactly: its class-level rewards/terminations, the
            # ``full`` action restore its parent chain applies (mode unset -> "full"), then the knobs.
            self.rewards = CrabHexFlatWalkRewardsCfg()
            self.terminations = CrabHexFlatWalkTerminationsCfg()
            _apply_crab_hex_full_actions(self)
            apply_flat_walk_knobs(self)
        elif mode == "bridge":
            _apply_crab_hex_stage_2b_bridge_lite_env(self)
            self.rewards = CrabHexTeacherBridgeRewardsCfg()
            _apply_crab_hex_stage_2b_phase1_terrain(self)
        elif mode == "2b1":
            _apply_crab_hex_stage_2b_bridge_lite_env(self)
            self.rewards = CrabHexStage2BPhase1RewardsCfg()
            _apply_crab_hex_stage_2b_phase1_terrain(self)
        elif mode == "2b2":
            _apply_crab_hex_stage_2b_bridge_lite_env(self)
            self.rewards = CrabHexStage2BPhase2RewardsCfg()
            _apply_crab_hex_stage_2b_phase2_terrain(self)
        elif mode == "full1":
            # Ramp stage 1/2 toward full (see _crab_hex_teacher_mode docstring): eases actions,
            # DR, terrain, and the failure threshold partway from 2b2's bridge-lite MDP toward
            # true full, instead of jumping all four axes to their hardest values simultaneously.
            _apply_crab_hex_full_ramp1_actions(self)
            _apply_crab_hex_full_ramp1_dr(self)
            _apply_crab_hex_full_ramp1_terrain(self)
            self.terminations.crab_failure.params["contact_force_threshold"] = 750.0
        elif mode == "full2":
            # Ramp stage 2/2: DR left at class-level EventCfg defaults (full strength) --
            # only full1 needs the half-strength override.
            _apply_crab_hex_full_ramp2_actions(self)
            _apply_crab_hex_full_ramp2_terrain(self)
            self.terminations.crab_failure.params["contact_force_threshold"] = 600.0
        else:
            # mode == "full": rewards (CrabHexRewardsCfg) and terrain (full Go2 sub-terrain mix,
            # difficulty 0-1, domain randomization on) stay at their class-level defaults --
            # only actions need restoring to 0.25/±4.8 (see _apply_crab_hex_full_actions docstring).
            _apply_crab_hex_full_actions(self)


def apply_flat_walk_knobs(cfg, *, include_reward_anneal: bool = True) -> None:
    """The flat-walk MDP knobs (``KRABBY_*`` -> cfg), shared by every training phase.

    Extracted verbatim from ``CrabHexFlatWalkEnvCfg.__post_init__`` (paradigm restore, 2026-09-07)
    so that phase-2 teacher modes (``KRABBY_HEX_TEACHER_MODE=2a|2b|2c``) and the phase-3 student
    MDP build the identical environment from the same knobs. ``include_reward_anneal=False`` skips
    the ``KRABBY_PHASEOUT`` reward-weight curriculum (distillation has no RL reward).
    """
    # Straight flat-walk: fixed world heading 0; P-control corrects slow yaw drift in play/train.
    # NOTE(gait-formation Phase 0, 2026-08-20): command range overridable per-arm as
    # "lo:hi" (e.g. KRABBY_LIN_VEL_X="0.0:0.35" for a stand-first survival arm).
    # NOTE(gait-formation Phase A, 2026-08-21): command resampling override "lo:hi"
    # seconds. The eval holds commands 10 s but training resampled every 6 s -- C5/C9
    # both plateaued at ~0.3 eval completion while passing training gates; policies
    # never trained against a sustained hold.
    _rs = os.environ.get("KRABBY_RESAMPLE_S")
    if _rs is not None:
        _rlo, _rhi = (float(x) for x in _rs.split(":"))
        cfg.commands.base_velocity.resampling_time_range = (_rlo, _rhi)
    _lvx = os.environ.get("KRABBY_LIN_VEL_X")
    if _lvx is not None:
        _lo, _hi = (float(x) for x in _lvx.split(":"))
        cfg.commands.base_velocity.ranges.lin_vel_x = (_lo, _hi)
    else:
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.30, 0.65)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    # NOTE(gait-formation-v2 Phase 1): RSI arm — seed a fraction of resets from the
    # Phase-0 scripted-gait reference bank (see crab_hex_rsi.py). Presence-based.
    _rsi = os.environ.get("KRABBY_RSI_FRAC")
    if _rsi is not None and float(_rsi) > 0.0:
        from isaaclab.managers import EventTermCfg as _EventTerm

        from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_rsi as _rsi_mod

        _bank_default = str(
            Path(__file__).resolve().parents[2]
            / "experiments/2026-08-22_1200_gait_formation_v2/rsi_bank_setAB.npz"
        )
        cfg.events.rsi_reference_reset = _EventTerm(
            func=_rsi_mod.reset_from_reference_states,
            mode="reset",
            params={
                "bank_path": os.environ.get("KRABBY_RSI_BANK", _bank_default),
                "fraction": float(_rsi),
                # PLAN H B4: place RSI resets where reset_root_state places (unset = old
                # tile-centre placement, bit-identical).
                "fix_spawn": os.environ.get("KRABBY_RSI_SPAWN_FIX", "").strip().lower()
                in ("1", "true", "yes", "on"),
            },
        )
    cfg.commands.base_velocity.heading_control_stiffness = 1.5
    # NOTE(gait-formation Phase 0): unidirectional cam clamp (structural
    # continuous-spin lever, lit-review "clamp + cadence-linked income" recipe).
    # KRABBY_CAM_CLIP_LO sets the LOWER raw-action bound on the six cam velocity
    # channels only (upper stays +1.0): e.g. 0.1 -> commanded shaft speed in
    # [0.1*pi, pi], oscillation not expressible. Unset = symmetric (-1, 1).
    _cam_lo = os.environ.get("KRABBY_CAM_CLIP_LO")
    if _cam_lo is not None:
        clip = dict(cfg.actions.joint_pos.clip)
        clip[".*_Body_CamShaft_RevoluteJoint"] = (float(_cam_lo), 1.0)
        cfg.actions.joint_pos.clip = clip
    # NOTE(PLAN F gated lineage, 2026-08-26): DR events default OFF (as always for
    # flat-walk) but each is now individually armable for the P6 elements.
    _dr_push = os.environ.get("KRABBY_DR_PUSH")
    if _dr_push is not None and cfg.events.push_by_setting_velocity is not None:
        _pv = float(_dr_push)
        cfg.events.push_by_setting_velocity.params["velocity_range"] = {
            "x": (-_pv, _pv), "y": (-_pv, _pv)
        }
    else:
        cfg.events.push_by_setting_velocity = None
    _dr_mass = os.environ.get("KRABBY_DR_MASS")
    if _dr_mass is not None and cfg.events.randomize_rigid_body_mass is not None:
        _mlo, _mhi = (float(x) for x in _dr_mass.split(":"))
        cfg.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (_mlo, _mhi)
    else:
        cfg.events.randomize_rigid_body_mass = None
    _dr_com = os.environ.get("KRABBY_DR_COM")
    if _dr_com is not None and cfg.events.randomize_rigid_body_com is not None:
        _cv = float(_dr_com)
        cfg.events.randomize_rigid_body_com.params["com_range"] = {
            "x": (-_cv, _cv), "y": (-_cv, _cv), "z": (-_cv, _cv)
        }
    else:
        cfg.events.randomize_rigid_body_com = None
    # PLAN F: turning school — heading command band ("lo:hi", rad). Default stays (0,0).
    _hd = os.environ.get("KRABBY_HEADING")
    if _hd is not None:
        _hlo, _hhi = (float(x) for x in _hd.split(":"))
        cfg.commands.base_velocity.ranges.heading = (_hlo, _hhi)
    _hcs = os.environ.get("KRABBY_HEADING_STIFFNESS")
    if _hcs is not None:  # pre-authorized turning-school fallback (1.5 -> 0.75)
        cfg.commands.base_velocity.heading_control_stiffness = float(_hcs)
    # PLAN F: episode length switch (P2 onward trains at 40 s).
    _eps = os.environ.get("KRABBY_EPISODE_S")
    if _eps is not None:
        cfg.episode_length_s = float(_eps)
    # PLAN F: termination knobs (flat defaults 0.5 rad / 500 N stay unless set).
    _ta = os.environ.get("KRABBY_TERM_ANGLE")
    if _ta is not None:
        cfg.terminations.crab_failure.params["limit_angle"] = float(_ta)
    _tc = os.environ.get("KRABBY_TERM_CONTACT_N")
    if _tc is not None:
        cfg.terminations.crab_failure.params["contact_force_threshold"] = float(_tc)
    # PLAN F: terrain-level promotion fractions recalibrated for the plant's ~0.5
    # tracking ratio (stock 0.8/0.4 can never promote here) — "up:down".
    _tp = os.environ.get("KRABBY_TERRAIN_PROMOTE")
    if _tp is not None:
        _up, _down = (float(x) for x in _tp.split(":"))
        cfg.parkours.base_parkour.move_up_frac = _up
        cfg.parkours.base_parkour.move_down_frac = _down
    tg = getattr(cfg.scene.terrain, "terrain_generator", None) if cfg.scene.terrain else None
    if tg is not None:
        tg.curriculum = False
        tg.difficulty_range = (0.1, 0.25)
        for key, sub_terrain in tg.sub_terrains.items():
            if key == "parkour_flat":
                sub_terrain.proportion = 1.0
            else:
                sub_terrain.proportion = 0.0
    # NOTE(phased-flat Phase C 2026-08-18): ``KRABBY_FLAT_TERRAIN_MODE=light`` blends
    # easy obstacles into flat-walk so lifting is learned while the gait is still
    # plastic (the carry-up lineages showed obstacles-after-consolidation plateaus).
    # Same terrain recipe family as the bridge stage but gentler: frozen levels,
    # shallow parkour geometry, defaults 80% flat / difficulty 0.05-0.2. Knobs:
    # ``KRABBY_FLAT_TERRAIN_FLAT_FRAC`` and ``KRABBY_FLAT_TERRAIN_DIFF`` ("lo:hi").
    _flat_mode = os.environ.get("KRABBY_FLAT_TERRAIN_MODE", "").strip().lower()
    if _flat_mode in ("light", "1", "true", "yes") and tg is not None:
        cfg.parkours.base_parkour.freeze_terrain_levels = True
        _frac = float(os.environ.get("KRABBY_FLAT_TERRAIN_FLAT_FRAC", "0.8"))
        _lo, _hi = (
            float(x) for x in os.environ.get("KRABBY_FLAT_TERRAIN_DIFF", "0.05:0.2").split(":")
        )
        _apply_crab_hex_easy_mixed_terrain(tg, flat_proportion=_frac, difficulty_range=(_lo, _hi))
        # PLAN F: geometry preset — "shallow" (bridge-era default) or "recal2b2" (the
        # plant-recalibrated end-state geometry).
        _geom = os.environ.get("KRABBY_FLAT_TERRAIN_GEOM", "shallow").strip().lower()
        if _geom == "recal2b2":
            _apply_crab_hex_recal_2b2_parkour_geometry(tg)
        elif _geom == "recal2b2w":  # PLAN H B0a: corridors widened to the plant
            _apply_crab_hex_recal_2b2w_parkour_geometry(tg)
        else:
            _apply_crab_hex_bridge_shallow_parkour_geometry(tg)
        # PLAN F: terrain-level curriculum enable (applied AFTER the mixed-terrain
        # helper, which forces curriculum False).
        if os.environ.get("KRABBY_FLAT_TERRAIN_CURRICULUM", "").strip() in ("1", "true", "yes"):
            cfg.parkours.base_parkour.freeze_terrain_levels = False
            tg.curriculum = True
    # PLAN H (2026-09-03) obstacle-exposure knobs. All unset = bit-identical. Parsing,
    # bounds and the corridor helpers live in mdp/exposure_knobs.py (unit-tested).
    #   KRABBY_CORRIDOR_HALF_WIDTH=lo:hi / KRABBY_STONE_WIDTH=w  -- applied AFTER any preset
    #   KRABBY_STAND_FRAC=p        -- B1 Bernoulli standing slots (command term)
    #   KRABBY_SPAWN_OFFSET=m      -- B2 platform spawn offset (default 3.0 -> tile-local 1.0 m)
    #   KRABBY_SPAWN_SPREAD=lo:hi[:frac] -- B3 spawn along the course (tile-local m)
    from parkour_tasks.crab_hex_forward_task.mdp import exposure_knobs as _xk

    _half, _stone = _xk.corridor_overrides_from_env(os.environ)
    if tg is not None and (_half is not None or _stone is not None):
        _xk.apply_corridor_widths(tg.sub_terrains, _half, _stone)
    _sf = os.environ.get("KRABBY_STAND_FRAC")
    if _sf:
        cfg.commands.base_velocity.stand_frac = _xk.parse_stand_frac(_sf)
    _so = os.environ.get("KRABBY_SPAWN_OFFSET")
    if _so:
        cfg.events.reset_root_state.params["offset"] = _xk.parse_spawn_offset(_so)
    _ss = os.environ.get("KRABBY_SPAWN_SPREAD")
    if _ss:
        _slo, _shi, _sfrac = _xk.parse_spawn_spread(_ss)
        cfg.events.reset_root_state.params["spread"] = (_slo, _shi)
        cfg.events.reset_root_state.params["spread_frac"] = _sfrac
    # NOTE(PLAN G gait-income phase-out, 2026-08-31): in-run cosine anneal of reward
    # term weights via the (already-constructed) CurriculumManager. Armed only by
    # KRABBY_PHASEOUT="term:w0:w1:t0:t1[,...]" (t in env steps, relative to process
    # start); unset leaves cfg.curriculum = None — bit-identical to before.
    # Targets are epsilon-clamped (>=1e-3): weight-exactly-0.0 terms are skipped by
    # ParkourRewardManager and their Episode_Reward telemetry (the campaign's gait
    # gate metric) would flatline.
    _phaseout = os.environ.get("KRABBY_PHASEOUT") if include_reward_anneal else None
    if _phaseout:
        from parkour_tasks.crab_hex_forward_task.mdp.curriculums import (
            phaseout_curriculum_cfg_from_env,
        )

        cfg.curriculum = phaseout_curriculum_cfg_from_env(_phaseout)


@configclass
class CrabHexFlatWalkEnvCfg(CrabHexTeacherEnvCfg):
    """Stage 1 **gait** (``Isaac-Crab-Hex-Flat-Walk-v0``): no ``KRABBY_HEX_TEACHER_MODE``.

    Learn alternating hex footfall on 100% flat tiles before any teacher mode.
    Rewards emphasize commanded speed, forward progress, upright pose, and light
    swing/stance shaping — not parkour goals. See README §2 / §3 and
    ``CrabHexFlatWalkRewardsCfg``.
    """

    actions: CrabHexFlatWalkActionsCfg = CrabHexFlatWalkActionsCfg()
    rewards: CrabHexFlatWalkRewardsCfg = CrabHexFlatWalkRewardsCfg()
    terminations: CrabHexFlatWalkTerminationsCfg = CrabHexFlatWalkTerminationsCfg()

    def __post_init__(self):
        # The teacher parent's post-init with the mode pinned to "full" (see
        # ``_crab_hex_apply_teacher_mode``), then the flat-walk knobs -- the chain every
        # flat-walk lineage trained through, independent of KRABBY_HEX_TEACHER_MODE.
        super(CrabHexTeacherEnvCfg, self).__post_init__()
        self.sim.physx.enable_external_forces_every_iteration = True
        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.85)
        base_body_cfg = SceneEntityCfg("robot", body_names="body")
        for ev in (self.events.base_external_force_torque, self.events.randomize_rigid_body_mass,
                   self.events.randomize_rigid_body_com):
            if ev is not None:
                ev.params["asset_cfg"] = base_body_cfg
        self._crab_hex_apply_teacher_mode("full")
        apply_flat_walk_knobs(self)


@configclass
class CrabHexFlatWalkEnvCfgPLAY(CrabHexFlatWalkEnvCfg):
    """Flat-walk visualization: follow-cam and command debug."""

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = None


@configclass
class CrabHexTeacherEnvCfgPLAY(CrabHexTeacherEnvCfg):
    """Visualization / evaluation: follow-cam, parkour debug, longer episodes, structured parkour mix.

    **Default terrain is the easy / flat-heavy mix** so stance checks are not confused with hard parkour.
    Set ``KRABBY_HEX_PLAY_HARD=1`` for the previous high-difficulty play preset (no flat, 0.7–1.0 difficulty).
    """

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = None
        # Bridge / stage-2b / phase-2 train sets terrain in ``CrabHexTeacherEnvCfg``; play must match train.
        if _crab_hex_bridge_like_mdp_active() or _crab_hex_phase_mode_active():
            return
        tg = getattr(self.scene.terrain, "terrain_generator", None) if self.scene.terrain else None
        if tg is not None:
            play_easy_flag = os.environ.get("KRABBY_HEX_PLAY_EASY", "").strip().lower() in ("1", "true", "yes")
            play_hard_flag = os.environ.get("KRABBY_HEX_PLAY_HARD", "").strip().lower() in ("1", "true", "yes")
            # Default easy unless explicitly requesting hard parkour (legacy was hard unless PLAY_EASY).
            easy = play_easy_flag or not play_hard_flag
            if easy:
                tg.difficulty_range = (0.15, 0.55)
                active = [
                    k
                    for k in tg.sub_terrains
                    if k not in ("parkour_flat", "parkour_demo")
                ]
                n_other = len(active)
                share = (0.5 / n_other) if n_other else 0.0
                for key, sub_terrain in tg.sub_terrains.items():
                    if key == "parkour_flat":
                        sub_terrain.proportion = 0.5
                    elif key == "parkour_demo":
                        sub_terrain.proportion = 0.0
                    else:
                        sub_terrain.proportion = share
                    sub_terrain.noise_range = (0.02, 0.02)
            else:
                tg.difficulty_range = (0.7, 1.0)
                for key, sub_terrain in tg.sub_terrains.items():
                    if key == "parkour_flat":
                        sub_terrain.proportion = 0.0
                    else:
                        sub_terrain.proportion = 0.2
                        sub_terrain.noise_range = (0.02, 0.02)


@configclass
class CrabHexStudentEnvCfg(CrabHexStudentParkourEnvCfg):
    """Gym entry ``Isaac-Crab-Hex-Student-v0``: depth student MDP — the phase-2c teacher MDP under a
    phase-3 preset / ``KRABBY_STUDENT_MDP=1``, else the legacy 2b2-teacher MDP."""

    def __post_init__(self):
        super().__post_init__()
        from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_phases import is_student_phase

        base_body_cfg = SceneEntityCfg("robot", body_names="body")
        if is_student_phase():
            # Paradigm phase 3: the student's MDP is the phase-2c teacher MDP -- the teacher's
            # terrain generator (0.08 m full-resolution tiles, 40 columns; the student scene's
            # simplified 0.1 m / 20-column mesh would change the obstacle geometry the teacher
            # was trained on), the flat-walk terminations, the ``full`` action space the 2c teacher
            # acts in, push / mass / CoM DR on the chassis body, and the same knobs (terrain band,
            # walking slots, horizon, DR, RSI, plant). No reward anneal (distillation has no reward).
            self.scene.terrain = CrabHexTeacherSceneCfg().terrain
            self.sim.physx.enable_external_forces_every_iteration = True
            self.terminations = CrabHexFlatWalkTerminationsCfg()
            _apply_crab_hex_full_actions(self)
            for ev in (self.events.randomize_rigid_body_mass, self.events.randomize_rigid_body_com):
                if ev is not None:
                    ev.params["asset_cfg"] = base_body_cfg
            apply_flat_walk_knobs(self, include_reward_anneal=False)
        else:
            _apply_crab_hex_student_2b2_teacher_mdp(self)
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = base_body_cfg


@configclass
class CrabHexStudentEnvCfgPLAY(CrabHexStudentEnvCfg):
    """Visualization: ``CRAB_HEX_VIEWER`` follow-cam (same as ``CrabHexTeacherEnvCfgPLAY``).

    Set ``KRABBY_HEX_PLAY_FLAT=1`` for 100% flat tiles (student MDP / obs unchanged).
    """

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = None
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
