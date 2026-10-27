# SPDX-License-Identifier: BSD-3-Clause
"""The omni depth student's pure-Python spec (Task 4 pipeline, step 4 / M1b; DEFAULT OFF everywhere).

Nothing here imports Isaac, so the CPU unit tests path-load it (``tests/unit/test_task4_student.py``).
Three things live here, each used by exactly one Isaac-side module:

1. **The four cameras** (``PL/cameras.yaml`` v0, approved at STOP-0 2026-10-16; every mount pose is a
   PLACEHOLDER for M15 pose matching): ZED 2i front, DFRobot SEN0583 ToF left / right / rear. For each:
   mount position on ``chassis/body``, the ROS-convention quaternion, the pinhole apertures that give the
   datasheet FOV on a 58 x 87 ray grid (cast directly at the depth-encoder input size, no crop / resize),
   2 m clip, ``/World/ground`` only. ``crab_hex_omni_scene_cfg.CrabHexOmniStudentSceneCfg`` builds the
   ``RayCasterCameraCfg`` objects from :func:`camera_specs`.
2. **The rotation tripwire** (:func:`check_scene_cameras`). The quaternion is composed, never typed in:
   ``q_z(yaw) (x) q_fwd(pitch)``, with ``q_fwd`` the forward camera's own Euler recipe
   (``default_cfg.quat_from_euler_xyz_tuple(180, 90 - pitch, -90)``, which NEGATES ``qz``). Editing the
   Euler yaw term instead rolls the image about the optical axis (``crab_hex_side_scene_cfg`` NOTE), so
   the check rotates the three ROS camera axes into the body frame and compares them with the axes the
   spec's yaw / pitch imply (optical axis ``(cos p cos y, cos p sin y, -sin p)``, image right
   ``(sin y, -cos y, 0)``, image down ``(-sin p cos y, -sin p sin y, -cos p)``). It RAISES, but only
   from the student ENV cfg's ``__post_init__`` (``crab_hex_omni_env_cfg._apply_student_sensors``):
   the scene class bodies are instantiated at import by every process through ``import_packages``, and
   an import-time raise there would be an outage of every task (the sideways precedent's reason for a
   warning). The env cfg is built only when the student task itself is launched.
3. **The student runner's knob policy** (:func:`apply_student_runner_knobs`): it applies ONLY
   ``KRABBY_STD_PARAM`` (``soft``) and ``KRABBY_INIT_STD`` -- the two the teacher checkpoint needs to
   load (``std_raw``) -- and REFUSES every other runner-side knob (the PPO-construction, statue,
   re-stance, BC, mirror and symmetry knobs): the distillation algorithm has no PPO, no symmetry loss and
   no entropy, so a set knob would be a recipe that says something it does not train. The distill
   launcher strips them from T*'s overlay (DESIGN stage S1: "minus the runner-only keys").
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------------------------- cameras
#: The camera order of every K-image tensor (scene attribute, observation group, student step, export).
CAMERA_NAMES = ("front", "left", "right", "rear")
#: Scene attribute per camera (``env.scene[<attr>]``); the observation groups are ``depth_<name>``.
CAMERA_ATTRS = tuple(f"depth_camera_{n}" for n in CAMERA_NAMES)
DEPTH_GROUPS = tuple(f"depth_{n}" for n in CAMERA_NAMES)

#: The ray grid every camera casts: the forward depth encoder's input size (cameras.yaml ``encoder_input``).
IMAGE_ROWS = 58
IMAGE_COLS = 87
#: Far clip, all four cameras (the forward normalisation clip; SEN0583 range; the ZED's 20 m is unused).
MAX_DISTANCE_M = 2.0
#: Pinhole focal length (tenths of a world unit, IsaacLab convention); only the aperture / focal RATIO
#: matters to the ray pattern. The forward camera's value, kept so its apertures stay comparable.
FOCAL_LENGTH = 11.041
#: cameras.yaml ``update.proposal_period_s``: refreshed every policy step (sim.dt * decimation), consumed
#: every :data:`DEPTH_PERIOD` steps by the image term and the student step.
UPDATE_PERIOD_S = 0.02
DEPTH_PERIOD = 5
#: Ray-cast target: the terrain mesh only (field blocks are stamped into it; tables / robot are not).
MESH_PRIM_PATHS = ("/World/ground",)
CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/chassis/body"

#: cameras.yaml v0 (2026-10-16). Every ``pos`` / ``pitch_down_deg`` is a PLACEHOLDER for M15.
_CAMERAS = {
    "front": {"sensor": "ZED 2i (2.1 mm)", "pos": (0.37, 0.0, 0.10), "yaw_deg": 0.0, "pitch_down_deg": 55.0,
              "hfov_deg": 110.0, "vfov_deg": 70.0},
    "left": {"sensor": "DFRobot SEN0583 ToF", "pos": (0.0, 0.625, 0.10), "yaw_deg": 90.0, "pitch_down_deg": 54.0,
             "hfov_deg": 55.0, "vfov_deg": 72.0},
    "right": {"sensor": "DFRobot SEN0583 ToF", "pos": (0.0, -0.625, 0.10), "yaw_deg": -90.0, "pitch_down_deg": 54.0,
              "hfov_deg": 55.0, "vfov_deg": 72.0},
    "rear": {"sensor": "DFRobot SEN0583 ToF", "pos": (-0.37, 0.0, 0.10), "yaw_deg": 180.0, "pitch_down_deg": 54.0,
             "hfov_deg": 55.0, "vfov_deg": 72.0},
}
#: cameras.yaml ``optical_axis_body`` (3 decimals) -- a second, independent statement of the pointing.
YAML_OPTICAL_AXES = {
    "front": (0.574, 0.0, -0.819),
    "left": (0.0, 0.588, -0.809),
    "right": (0.0, -0.588, -0.809),
    "rear": (-0.588, 0.0, -0.809),
}
#: Tripwire tolerance on each body-frame axis component.
AXIS_TOL = 1e-6


def quat_mul(q1, q2):
    """Hamilton product of two ``(w, x, y, z)`` quaternions (pure Python)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_rotate(q, v):
    """``q`` applied to the 3-vector ``v`` (pure Python)."""
    w, x, y, z = q
    vx, vy, vz = v
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))


def forward_euler_quat(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """``parkour_tasks.default_cfg.quat_from_euler_xyz_tuple`` in float64 -- including its ``qz`` negation.

    That helper returns ``(qw, qx, qy, -qz)`` of the Euler triple, which is why the forward recipe
    ``(180, 70, -90)`` points the forward camera 20 deg down along body +x and why its yaw term cannot be
    used to turn a camera (see the module docstring).
    """
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return (qw, qx, qy, -qz)


def camera_quat(yaw_deg: float, pitch_down_deg: float):
    """ROS-convention offset rotation: ``q_z(yaw) (x) q_fwd(pitch)`` (cameras.yaml ``rotation_rule``)."""
    base = forward_euler_quat(180.0, 90.0 - pitch_down_deg, -90.0)
    half = math.radians(yaw_deg) * 0.5
    return quat_mul((math.cos(half), 0.0, 0.0, math.sin(half)), base)


def expected_axes(yaw_deg: float, pitch_down_deg: float) -> dict:
    """Body-frame ROS camera axes a camera with this yaw / pitch must have (forward, right, down)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_down_deg)
    return {
        "forward": (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)),
        "right": (math.sin(y), -math.cos(y), 0.0),
        "down": (-math.sin(p) * math.cos(y), -math.sin(p) * math.sin(y), -math.cos(p)),
    }


def body_axes(rot) -> dict:
    """The ROS camera axes (z forward, x right, y down) of the offset ``rot`` in the body frame."""
    return {
        "forward": quat_rotate(rot, (0.0, 0.0, 1.0)),
        "right": quat_rotate(rot, (1.0, 0.0, 0.0)),
        "down": quat_rotate(rot, (0.0, 1.0, 0.0)),
    }


def aperture(fov_deg: float, focal_length: float = FOCAL_LENGTH) -> float:
    """Pinhole aperture giving ``fov_deg`` at ``focal_length`` (``fov = 2 atan(aperture / 2f)``)."""
    return 2.0 * focal_length * math.tan(math.radians(fov_deg) * 0.5)


def camera_specs() -> dict:
    """``name -> dict`` with everything the scene cfg needs for each camera (in :data:`CAMERA_NAMES` order)."""
    out = {}
    for name in CAMERA_NAMES:
        c = dict(_CAMERAS[name])
        c["attr"] = f"depth_camera_{name}"
        c["group"] = f"depth_{name}"
        c["rot"] = camera_quat(c["yaw_deg"], c["pitch_down_deg"])
        c["horizontal_aperture"] = aperture(c["hfov_deg"])
        c["vertical_aperture"] = aperture(c["vfov_deg"])
        c["height"], c["width"] = IMAGE_ROWS, IMAGE_COLS
        out[name] = c
    return out


def camera_rotation_errors(name: str, rot) -> list[str]:
    """Every body-axis component of ``rot`` that misses the spec's yaw / pitch by more than :data:`AXIS_TOL`."""
    c = _CAMERAS[name]
    want = expected_axes(c["yaw_deg"], c["pitch_down_deg"])
    got = body_axes(tuple(float(v) for v in rot))
    errs = []
    for axis in ("forward", "right", "down"):
        dev = max(abs(a - b) for a, b in zip(got[axis], want[axis]))
        if dev > AXIS_TOL:
            errs.append(f"{name}.{axis}: body {tuple(round(v, 6) for v in got[axis])} != "
                        f"{tuple(round(v, 6) for v in want[axis])} (|d|={dev:.2e} > {AXIS_TOL})")
    return errs


def check_scene_cameras(scene) -> None:
    """The tripwire: the built scene's four cameras are exactly the spec (pose, rotation, grid, clip, target).

    Raises ``ValueError`` listing every mismatch. Called from the student ENV cfg's ``__post_init__``.
    """
    specs = camera_specs()
    errs = []
    for name in CAMERA_NAMES:
        spec = specs[name]
        cam = getattr(scene, spec["attr"], None)
        if cam is None:
            errs.append(f"scene.{spec['attr']} is missing")
            continue
        off = cam.offset
        if getattr(off, "convention", None) != "ros":
            errs.append(f"{name}: offset convention {getattr(off, 'convention', None)!r} != 'ros'")
        if any(abs(float(a) - b) > 1e-9 for a, b in zip(off.pos, spec["pos"])) or len(tuple(off.pos)) != 3:
            errs.append(f"{name}: offset pos {tuple(off.pos)} != {spec['pos']}")
        errs.extend(camera_rotation_errors(name, off.rot))
        pat = cam.pattern_cfg
        if (int(pat.height), int(pat.width)) != (IMAGE_ROWS, IMAGE_COLS):
            errs.append(f"{name}: ray grid {pat.height}x{pat.width} != {IMAGE_ROWS}x{IMAGE_COLS}")
        for key in ("horizontal_aperture", "vertical_aperture"):
            if abs(float(getattr(pat, key)) - spec[key]) > 1e-9:
                errs.append(f"{name}: {key} {getattr(pat, key)} != {spec[key]}")
        if abs(float(pat.focal_length) - FOCAL_LENGTH) > 1e-12:
            errs.append(f"{name}: focal_length {pat.focal_length} != {FOCAL_LENGTH}")
        if abs(float(cam.max_distance) - MAX_DISTANCE_M) > 1e-12:
            errs.append(f"{name}: max_distance {cam.max_distance} != {MAX_DISTANCE_M}")
        if tuple(cam.mesh_prim_paths) != MESH_PRIM_PATHS:
            errs.append(f"{name}: mesh_prim_paths {list(cam.mesh_prim_paths)} != {list(MESH_PRIM_PATHS)}")
        if cam.prim_path != CAMERA_PRIM_PATH:
            errs.append(f"{name}: prim_path {cam.prim_path!r} != {CAMERA_PRIM_PATH!r}")
        if list(cam.data_types) != ["distance_to_camera"]:
            errs.append(f"{name}: data_types {list(cam.data_types)} != ['distance_to_camera']")
    if errs:
        raise ValueError("omni student cameras drifted from omni_student_spec (cameras.yaml v0):\n  "
                         + "\n  ".join(errs))


# ----------------------------------------------------------------------------------- runner knob policy
STD_PARAM_VAR = "KRABBY_STD_PARAM"
INIT_STD_VAR = "KRABBY_INIT_STD"
#: The only runner-side knobs the student runner applies.
STUDENT_RUNNER_KNOBS_APPLIED = (STD_PARAM_VAR, INIT_STD_VAR)
#: Every OTHER runner-side knob: ``motion_knobs.apply_exploration_knobs`` reads these (pinned against its
#: source by the unit test), plus the teacher runners' symmetry knobs. Set -> the student runner refuses.
STUDENT_RUNNER_KNOBS_REFUSED = (
    "KRABBY_ACTION_STD_MAX", "KRABBY_ACTION_STD_RESET", "KRABBY_ENTROPY_COEF",
    "KRABBY_ACTION_BOUND_LOSS", "KRABBY_ACTION_BOUND_RAMP", "KRABBY_ACTION_BOUND",
    "KRABBY_PPO_SCHEDULE", "KRABBY_PPO_LR", "KRABBY_PPO_DESIRED_KL", "KRABBY_PPO_KL_STOP",
    "KRABBY_PPO_HIST_LR", "KRABBY_PPO_HAND_DIAG",
    "KRABBY_ANCHOR_W", "KRABBY_ANCHOR_CKPT", "KRABBY_STAND_PRIOR_W", "KRABBY_STAND_PRIOR_TARGET",
    "KRABBY_RESTANCE", "KRABBY_BC_W", "KRABBY_BC_BANK", "KRABBY_BC_BATCH",
    "KRABBY_OMNI_MIRROR_IMAGES", "KRABBY_SYM_LOSS_COEF",
)

_MOTION_KNOBS_PY = Path(__file__).resolve().parent / "motion_knobs.py"
_MOTION_KNOBS_NAME = "crab_hex_omni_motion_knobs"  # the same module object omni_knobs path-loads


def _motion_knobs():
    """``motion_knobs.py`` (pure), by path -- the same object for every loader of this module."""
    mod = sys.modules.get(_MOTION_KNOBS_NAME)
    if mod is None:
        spec = importlib.util.spec_from_file_location(_MOTION_KNOBS_NAME, _MOTION_KNOBS_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_MOTION_KNOBS_NAME] = mod
        spec.loader.exec_module(mod)
    return mod


def _blank(raw) -> bool:
    return raw is None or str(raw).strip() == ""


def student_runner_knob_refusals(environ) -> list[str]:
    """``VAR=value`` for every refused runner knob that is set (non-blank) in ``environ``."""
    return [f"{v}={environ[v]!r}" for v in STUDENT_RUNNER_KNOBS_REFUSED if not _blank(environ.get(v))]


def apply_student_runner_knobs(agent_cfg, environ=None) -> None:
    """Refuse every set runner knob but the two std knobs, then apply ``KRABBY_STD_PARAM`` / ``KRABBY_INIT_STD``.

    Both unset -> writes nothing (the forward student runner's policy cfg). The parsers and the soft-band
    rule are ``motion_knobs``'s own, so the student reads the two knobs exactly as the teacher runners do.
    """
    env = os.environ if environ is None else environ
    refused = student_runner_knob_refusals(env)
    if refused:
        raise ValueError(
            "the omni student runner (distillation) applies only KRABBY_STD_PARAM / KRABBY_INIT_STD; refusing "
            + ", ".join(refused)
            + " -- strip the runner-only keys from the teacher overlay (DESIGN stage S1)"
        )
    mk = _motion_knobs()
    std_param = mk.parse_std_param(env.get(STD_PARAM_VAR))
    init_std = mk.parse_init_std(env.get(INIT_STD_VAR))
    if std_param == "soft":
        if init_std is not None and not init_std < mk.STOCK_ACTION_STD_MAX:
            raise ValueError(f"{INIT_STD_VAR}={init_std} must lie strictly inside the soft band "
                             f"({mk.ACTION_STD_MIN}, {mk.STOCK_ACTION_STD_MAX})")
    elif init_std is not None and init_std > mk.STOCK_ACTION_STD_MAX:
        raise ValueError(f"{INIT_STD_VAR}={init_std} exceeds the std clamp ceiling {mk.STOCK_ACTION_STD_MAX}")
    if std_param is not None:
        agent_cfg.policy.noise_std_type = std_param
        mk._consumed(STD_PARAM_VAR, env.get(STD_PARAM_VAR))
    if init_std is not None:
        agent_cfg.policy.init_noise_std = init_std
        mk._consumed(INIT_STD_VAR, env.get(INIT_STD_VAR))
