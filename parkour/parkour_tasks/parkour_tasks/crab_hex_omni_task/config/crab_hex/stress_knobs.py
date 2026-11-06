# SPDX-License-Identifier: BSD-3-Clause
"""DR stress-sweep knobs (omni rung 1, 2026-11-06) -> single-parameter perturbations of a frozen policy's plant.

EVERY KNOB IS DEFAULT OFF AND SET BY NO PRESET. Unset or blank, a knob writes NOTHING into the cfg -- no attribute,
no param, no event term -- so a knob-off config dump is byte-identical to one built before this module existed.
The sweep (``experiments/2026-11-06_dr_stress_sweep``, CHARTER) arms ONE knob per cell; nothing refuses two, but a
cell with two knobs is not a single-parameter read.

====================================  ==============================================================================
variable                              effect (plant constants: ``crab_hex_dimensions`` / ``crab_hex_scene_cfg``)
====================================  ==============================================================================
``KRABBY_STRESS_ROD_SPEED=s``         both lead-screw rods' slew limit x s (hip 28, knee 33 mm/s nominal); the
                                      current proxy's drive gate keeps the rated speed
``KRABBY_STRESS_ROD_EFFORT=s``        ``hip_femur`` / ``femur_tibia`` ``effort_limit_sim`` x s (1000 / 400 N m: the
                                      holding-level joint torque cap -- the sim's only force bound; the rod DRIVE
                                      force is not enforced in sim at all)
``KRABBY_STRESS_CAM=s``               the cam motor x s: the cam acceleration limit (2 pi rad/s^2) and, below 1, a
                                      speed cap of s x pi rad/s; ``body_hip_yaw`` effort / saturation (20 / 24 N m)
                                      are scaled too but are INERT in sim -- the leg's yaw load never reaches the cam
                                      shaft (the hip follows the shaft through a software target, not a joint)
``KRABBY_STRESS_KPKD=s``              ``hip_femur`` / ``femur_tibia`` stiffness AND damping x s (16000 / 600, 6000 /
                                      200): the screw-compliance proxy
``KRABBY_STRESS_LEG_MASS=s``          every leg link's mass x s (Hip, Femur, Tibia, Footpad; not the placeholder
                                      CamShaft), inertia scaled with it (startup event ``stress_leg_mass``)
``KRABBY_STRESS_MU=mu``               foot-ground friction pinned at mu: the robot material event draws (mu, mu),
                                      and the ground and sim default materials are static = dynamic = mu, so the
                                      combined coefficient is mu under average / min / max combine
``KRABBY_STRESS_ACT_DELAY=d``         the action delay fixed at d policy steps (nominal 1 = 20 ms); needs
                                      ``use_delay`` and ``d < history_length``
``KRABBY_STRESS_OBS_NOISE=k``         uniform IMU + encoder noise, k x ``stress_obs.OBS_NOISE_REF`` (sensor side)
``KRABBY_STRESS_CUR_SCALE=s``         the current-proxy gain x s (sensor side)
``KRABBY_STRESS_CUR_NOISE=sigma``     Gaussian noise of std sigma on the normalised current proxy (sensor side)
``KRABBY_STRESS_VXY_NOISE=sigma``     Gaussian noise of std sigma m/s on the body-frame xy velocity slot
``KRABBY_STRESS_VXY_ZERO=1``          the xy velocity slot zeroed (exclusive with ``_VXY_NOISE``)
``KRABBY_STRESS_DEPTH_LAG=n``         the depth student's image n refreshes (n x 0.1 s) old (nominal 1); needs the
                                      four depth groups
``KRABBY_STRESS_PUSH_DV=dv``          the additive push (``crab_hex_forward_task/mdp/stress_push.py``, stepped by the
                                      action term): +dv m/s at a random heading every :data:`STRESS_PUSH_INTERVAL_S` s
                                      per env, timers restarted at each reset; no event-manager RNG draws
====================================  ==============================================================================

Any OTHER ``KRABBY_STRESS_*`` name in the environment is refused (a typo would otherwise run the cell at nominal).
Pure at module scope (no Isaac, no torch), path-loaded and re-exported by ``omni_knobs.py`` the way
``payload_knobs.py`` is; the Isaac import is lazy, inside the leg-mass event builder. See
``tests/unit/test_dr_stress_knobs.py``.
"""

from __future__ import annotations

import math
import os

# ------------------------------------------------------------------------------ variables
ROD_SPEED_VAR = "KRABBY_STRESS_ROD_SPEED"
ROD_EFFORT_VAR = "KRABBY_STRESS_ROD_EFFORT"
CAM_VAR = "KRABBY_STRESS_CAM"
KPKD_VAR = "KRABBY_STRESS_KPKD"
LEG_MASS_VAR = "KRABBY_STRESS_LEG_MASS"
MU_VAR = "KRABBY_STRESS_MU"
ACT_DELAY_VAR = "KRABBY_STRESS_ACT_DELAY"
OBS_NOISE_VAR = "KRABBY_STRESS_OBS_NOISE"
CUR_SCALE_VAR = "KRABBY_STRESS_CUR_SCALE"
CUR_NOISE_VAR = "KRABBY_STRESS_CUR_NOISE"
VXY_NOISE_VAR = "KRABBY_STRESS_VXY_NOISE"
VXY_ZERO_VAR = "KRABBY_STRESS_VXY_ZERO"
DEPTH_LAG_VAR = "KRABBY_STRESS_DEPTH_LAG"
PUSH_DV_VAR = "KRABBY_STRESS_PUSH_DV"
#: Every stress knob.
STRESS_VARS = (
    ROD_SPEED_VAR, ROD_EFFORT_VAR, CAM_VAR, KPKD_VAR, LEG_MASS_VAR, MU_VAR, ACT_DELAY_VAR, OBS_NOISE_VAR,
    CUR_SCALE_VAR, CUR_NOISE_VAR, VXY_NOISE_VAR, VXY_ZERO_VAR, DEPTH_LAG_VAR, PUSH_DV_VAR,
)

#: Actuator groups of ``crab_hex_scene_cfg._crab_robot_cfg`` the knobs touch.
ROD_ACTUATORS = ("hip_femur", "femur_tibia")
CAM_ACTUATOR = "body_hip_yaw"
#: The leg links ``KRABBY_STRESS_LEG_MASS`` scales (the CamShaft is a 0.3 kg placeholder with a pinned inertia
#: for actuator stability, so it is left alone).
LEG_BODY_EXPR = "(FL|FR|ML|MR|RL|RR)_(Hip|Femur|Tibia|Footpad)"
#: The env-cfg attribute the sensor knobs write (``crab_hex_forward_task/mdp/stress_obs.STRESS_OBS_ATTR``).
STRESS_OBS_ATTR = "stress_obs"
#: The action-cfg attribute the push knob writes (``crab_hex_forward_task/mdp/stress_push.STRESS_PUSH_ATTR``).
STRESS_PUSH_ATTR = "stress_push"
#: The depth groups ``KRABBY_STRESS_DEPTH_LAG`` re-points (``omni_student_spec.DEPTH_GROUPS``) and their term.
DEPTH_GROUPS = ("depth_front", "depth_left", "depth_right", "depth_rear")
DEPTH_TERM = "depth_image"
#: The event the leg-mass knob creates (absent while off).
LEG_MASS_EVENT = "stress_leg_mass"
#: The additive push's period, per env (s).
STRESS_PUSH_INTERVAL_S = 10.0

#: Typo guards, not physics (raise them here, with a reason, if a sweep needs more).
SCALE_MAX = 4.0
MU_MAX = 3.0
ACT_DELAY_MAX = 7
OBS_NOISE_MAX = 10.0
CUR_NOISE_MAX = 1.0
VXY_NOISE_MAX = 2.0
DEPTH_LAG_MAX = 10
PUSH_DV_MAX = 3.0

__all__ = [
    "ROD_SPEED_VAR", "ROD_EFFORT_VAR", "CAM_VAR", "KPKD_VAR", "LEG_MASS_VAR", "MU_VAR", "ACT_DELAY_VAR",
    "OBS_NOISE_VAR", "CUR_SCALE_VAR", "CUR_NOISE_VAR", "VXY_NOISE_VAR", "VXY_ZERO_VAR", "DEPTH_LAG_VAR",
    "PUSH_DV_VAR", "STRESS_VARS", "ROD_ACTUATORS", "CAM_ACTUATOR", "LEG_BODY_EXPR", "STRESS_OBS_ATTR",
    "DEPTH_GROUPS", "DEPTH_TERM", "LEG_MASS_EVENT", "STRESS_PUSH_ATTR", "STRESS_PUSH_INTERVAL_S",
    "STRESS_KNOB_PARSERS", "parse_stress_knobs", "apply_stress_knobs",
]


# ------------------------------------------------------------------------------ parsers
def _blank(text) -> bool:
    return text is None or not str(text).strip()


def _number(text, var: str) -> float:
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{var} must be a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{var} must be finite, got {text!r}")
    return value


def _in_range(var: str, lo: float, hi: float, *, lo_open: bool):
    def parse(text) -> float | None:
        if _blank(text):
            return None
        v = _number(text, var)
        ok = (lo < v if lo_open else lo <= v) and v <= hi
        if not ok:
            raise ValueError(f"{var} must be in {'(' if lo_open else '['}{lo:g}, {hi:g}], got {text!r}")
        return v + 0.0
    parse.__name__ = f"parse_{var.lower()}"
    return parse


def _int_in_range(var: str, lo: int, hi: int):
    def parse(text) -> int | None:
        if _blank(text):
            return None
        v = _number(text, var)
        if v != int(v) or not lo <= int(v) <= hi:
            raise ValueError(f"{var} must be an integer in [{lo}, {hi}], got {text!r}")
        return int(v)
    parse.__name__ = f"parse_{var.lower()}"
    return parse


def parse_flag(text, var: str) -> bool:
    """``1`` / ``0`` (or blank = off); anything else is refused rather than guessed."""
    if _blank(text):
        return False
    t = str(text).strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{var} must be 1 or 0, got {text!r}")


#: ``{variable: parser}`` (the driver's overlay round-trip table, ``run_candidates.knob_parsers``).
STRESS_KNOB_PARSERS = {
    ROD_SPEED_VAR: _in_range(ROD_SPEED_VAR, 0.0, SCALE_MAX, lo_open=True),
    ROD_EFFORT_VAR: _in_range(ROD_EFFORT_VAR, 0.0, SCALE_MAX, lo_open=True),
    CAM_VAR: _in_range(CAM_VAR, 0.0, SCALE_MAX, lo_open=True),
    KPKD_VAR: _in_range(KPKD_VAR, 0.0, SCALE_MAX, lo_open=True),
    LEG_MASS_VAR: _in_range(LEG_MASS_VAR, 0.0, SCALE_MAX, lo_open=True),
    MU_VAR: _in_range(MU_VAR, 0.0, MU_MAX, lo_open=True),
    ACT_DELAY_VAR: _int_in_range(ACT_DELAY_VAR, 0, ACT_DELAY_MAX),
    OBS_NOISE_VAR: _in_range(OBS_NOISE_VAR, 0.0, OBS_NOISE_MAX, lo_open=True),
    CUR_SCALE_VAR: _in_range(CUR_SCALE_VAR, 0.0, SCALE_MAX, lo_open=False),
    CUR_NOISE_VAR: _in_range(CUR_NOISE_VAR, 0.0, CUR_NOISE_MAX, lo_open=True),
    VXY_NOISE_VAR: _in_range(VXY_NOISE_VAR, 0.0, VXY_NOISE_MAX, lo_open=True),
    VXY_ZERO_VAR: lambda text: parse_flag(text, VXY_ZERO_VAR) or None,
    DEPTH_LAG_VAR: _int_in_range(DEPTH_LAG_VAR, 1, DEPTH_LAG_MAX),
    PUSH_DV_VAR: _in_range(PUSH_DV_VAR, 0.0, PUSH_DV_MAX, lo_open=True),
}


def parse_stress_knobs(environ=None) -> dict:
    """``{variable: value}`` of the ARMED knobs only (empty while all are off). Raises ``ValueError`` for a
    malformed value, for an unknown ``KRABBY_STRESS_*`` name and for ``KRABBY_STRESS_VXY_ZERO=1`` together with
    ``KRABBY_STRESS_VXY_NOISE``."""
    env = os.environ if environ is None else environ
    unknown = sorted(k for k in env if str(k).startswith("KRABBY_STRESS_") and k not in STRESS_KNOB_PARSERS)
    if unknown:
        raise ValueError(f"unknown stress knob(s) {unknown}: a misspelled knob would run the cell at nominal "
                         f"(known: {list(STRESS_VARS)})")
    armed = {}
    for var, parser in STRESS_KNOB_PARSERS.items():
        value = parser(env.get(var))
        if value is not None:
            armed[var] = value
    if VXY_ZERO_VAR in armed and VXY_NOISE_VAR in armed:
        raise ValueError(f"{VXY_ZERO_VAR}=1 and {VXY_NOISE_VAR} are exclusive (a zeroed slot carries no noise)")
    return armed


# ------------------------------------------------------------------------------ application
def _actuator(cfg, name: str, var: str):
    acts = getattr(getattr(getattr(cfg, "scene", None), "robot", None), "actuators", None)
    if not acts or name not in acts:
        raise RuntimeError(f"{var} is set but the env cfg has no scene.robot.actuators[{name!r}]")
    return acts[name]


def _scaled(value, s: float, var: str, field: str):
    """A gain / limit field (a float or a ``{joint_expr: float}`` dict) x s, as a NEW object."""
    if isinstance(value, dict):
        return {k: float(v) * s for k, v in value.items()}
    if value is None:
        raise RuntimeError(f"{var}: actuator field {field!r} is None, nothing to scale")
    return float(value) * s


def _scale_fields(cfg, names, fields, s: float, var: str) -> None:
    for name in names:
        act = _actuator(cfg, name, var)
        for field in fields:
            setattr(act, field, _scaled(getattr(act, field), s, var, field))


def _action(cfg, var: str):
    action = getattr(getattr(cfg, "actions", None), "joint_pos", None)
    if action is None:
        raise RuntimeError(f"{var} is set but the env cfg has no actions.joint_pos")
    return action


def _events(cfg, var: str):
    events = getattr(cfg, "events", None)
    if events is None:
        raise RuntimeError(f"{var} is set but the env cfg has no events block")
    return events


def leg_mass_event_cfg(scale: float):
    """The startup ``randomize_rigid_body_mass`` term pinning every leg link at ``scale`` x its mass."""
    from isaaclab.envs.mdp.events import randomize_rigid_body_mass  # noqa: PLC0415
    from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: PLC0415

    return EventTermCfg(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_BODY_EXPR),
            "mass_distribution_params": (float(scale), float(scale)),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )


def _set_friction(material, mu: float) -> None:
    material.static_friction = float(mu)
    material.dynamic_friction = float(mu)


def apply_stress_knobs(cfg, environ=None, *, leg_mass_event=None) -> dict:
    """Apply every armed stress knob to a built omni env cfg; writes nothing while all are off.

    Called by ``omni_knobs.apply_omni_knobs`` right after the payload knobs. ``leg_mass_event`` replaces the Isaac
    event builder (the CPU unit tests). Returns the armed ``{variable: value}``.
    """
    armed = parse_stress_knobs(environ)
    if not armed:
        return armed
    lines = []

    s = armed.get(ROD_SPEED_VAR)
    if s is not None:
        _action(cfg, ROD_SPEED_VAR).stress_rod_speed_scale = s
        lines.append(f"{ROD_SPEED_VAR}={s:g} -> rod slew limits x{s:g}")
    s = armed.get(ROD_EFFORT_VAR)
    if s is not None:
        _scale_fields(cfg, ROD_ACTUATORS, ("effort_limit_sim",), s, ROD_EFFORT_VAR)
        lines.append(f"{ROD_EFFORT_VAR}={s:g} -> {'/'.join(ROD_ACTUATORS)} effort_limit_sim x{s:g}")
    s = armed.get(CAM_VAR)
    if s is not None:
        _scale_fields(cfg, (CAM_ACTUATOR,), ("effort_limit", "saturation_effort"), s, CAM_VAR)
        _action(cfg, CAM_VAR).stress_cam_scale = s
        lines.append(f"{CAM_VAR}={s:g} -> cam accel limit x{s:g}"
                     + (f", cam speed cap {s:g} x full-action speed" if s < 1.0 else "")
                     + f"; {CAM_ACTUATOR} effort/saturation x{s:g} (inert in sim: no leg load on the shaft)")
    s = armed.get(KPKD_VAR)
    if s is not None:
        _scale_fields(cfg, ROD_ACTUATORS, ("stiffness", "damping"), s, KPKD_VAR)
        lines.append(f"{KPKD_VAR}={s:g} -> {'/'.join(ROD_ACTUATORS)} stiffness and damping x{s:g}")
    s = armed.get(LEG_MASS_VAR)
    if s is not None:
        build = leg_mass_event if leg_mass_event is not None else leg_mass_event_cfg
        setattr(_events(cfg, LEG_MASS_VAR), LEG_MASS_EVENT, build(s))
        lines.append(f"{LEG_MASS_VAR}={s:g} -> events.{LEG_MASS_EVENT} (leg links {LEG_BODY_EXPR} x{s:g}, "
                     "inertia scaled)")
    mu = armed.get(MU_VAR)
    if mu is not None:
        events = _events(cfg, MU_VAR)
        material_event = getattr(events, "physics_material", None)
        if material_event is None:
            raise RuntimeError(f"{MU_VAR} is set but events.physics_material is off (nothing sets the robot side)")
        material_event.params["friction_range"] = (float(mu), float(mu))
        terrain_material = getattr(getattr(getattr(cfg, "scene", None), "terrain", None), "physics_material", None)
        if terrain_material is None:
            raise RuntimeError(f"{MU_VAR} is set but the env cfg has no scene.terrain.physics_material")
        _set_friction(terrain_material, mu)
        sim_material = getattr(getattr(cfg, "sim", None), "physics_material", None)
        if sim_material is not None:
            _set_friction(sim_material, mu)
        lines.append(f"{MU_VAR}={mu:g} -> robot friction ({mu:g}, {mu:g}), ground and sim default static = "
                     f"dynamic = {mu:g}")
    d = armed.get(ACT_DELAY_VAR)
    if d is not None:
        action = _action(cfg, ACT_DELAY_VAR)
        if not getattr(action, "use_delay", False):
            raise RuntimeError(f"{ACT_DELAY_VAR} is set but actions.joint_pos.use_delay is off")
        hist = int(getattr(action, "history_length", 0) or 0)
        if not d < hist:
            raise RuntimeError(f"{ACT_DELAY_VAR}={d} needs d < actions.joint_pos.history_length ({hist})")
        action.action_delay_steps = [int(d)]
        lines.append(f"{ACT_DELAY_VAR}={d} -> actions.joint_pos.action_delay_steps [{d}]")

    sensor = {}
    if OBS_NOISE_VAR in armed:
        sensor["noise_k"] = armed[OBS_NOISE_VAR]
    if CUR_SCALE_VAR in armed:
        sensor["cur_scale"] = armed[CUR_SCALE_VAR]
    if CUR_NOISE_VAR in armed:
        sensor["cur_noise"] = armed[CUR_NOISE_VAR]
    if VXY_NOISE_VAR in armed:
        sensor["vxy_noise"] = armed[VXY_NOISE_VAR]
    if VXY_ZERO_VAR in armed:
        sensor["vxy_zero"] = True
    if sensor:
        setattr(cfg, STRESS_OBS_ATTR, sensor)
        lines.append(f"sensor stress -> cfg.{STRESS_OBS_ATTR} = {sensor}")

    n = armed.get(DEPTH_LAG_VAR)
    if n is not None:
        obs = getattr(cfg, "observations", None)
        missing = [g for g in DEPTH_GROUPS if getattr(getattr(obs, g, None), DEPTH_TERM, None) is None]
        if missing:
            raise RuntimeError(f"{DEPTH_LAG_VAR} is set but the env cfg has no depth groups {missing} "
                               "(a student env only)")
        for g in DEPTH_GROUPS:
            term = getattr(getattr(obs, g), DEPTH_TERM)
            term.params["buffer_len"] = int(n) + 1
            term.params["lag"] = int(n)
        lines.append(f"{DEPTH_LAG_VAR}={n} -> {len(DEPTH_GROUPS)} depth terms lag {n} (buffer_len {n + 1})")
    dv = armed.get(PUSH_DV_VAR)
    if dv is not None:
        setattr(_action(cfg, PUSH_DV_VAR), STRESS_PUSH_ATTR, {"dv": float(dv), "interval_s": STRESS_PUSH_INTERVAL_S})
        lines.append(f"{PUSH_DV_VAR}={dv:g} -> actions.joint_pos.{STRESS_PUSH_ATTR} (+{dv:g} m/s, random heading, "
                     f"every {STRESS_PUSH_INTERVAL_S:g} s per env, own timers)")
    for line in lines:
        print(f"[KNOB] {line}", flush=True)
    return armed
