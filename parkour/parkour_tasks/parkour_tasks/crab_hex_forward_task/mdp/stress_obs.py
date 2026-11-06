# SPDX-License-Identifier: BSD-3-Clause
"""Sensor-side stress for the pre-hardware DR stress sweep (omni rung 1, 2026-11-06). DEFAULT OFF.

Armed only by ``crab_hex_omni_task/config/crab_hex/stress_knobs.py``, which writes the env-cfg attribute
:data:`STRESS_OBS_ATTR` (a plain dict) when one of the sensor knobs is set. Without that attribute
``CrabHexParkourObservations`` never builds this object and its observation is bitwise what it was.

What it perturbs, on the RAW sensor values, before the proprio scales and before the history push (a real
sensor reading is one value: the policy's current row and the row that enters the history carry the same
noise):

* ``noise_k`` (``KRABBY_STRESS_OBS_NOISE``) -- i.i.d. uniform noise, half-width ``noise_k`` x the reference
  table :data:`OBS_NOISE_REF` (legged_gym's ``noise_scales`` convention, uniform in ``[-s, s]`` per channel per
  step), on the IMU (body angular velocity, roll / pitch) and the joint encoders (position, velocity). The
  joint-velocity reference is scaled to this plant (lead-screw joints move at 0.2-2.3 rad/s, the cams at
  <= pi rad/s), not legged_gym's 1.5 rad/s for a 30 rad/s quadruped joint.
* ``cur_scale`` / ``cur_noise`` (``KRABBY_STRESS_CUR_SCALE`` / ``_NOISE``) -- the motor-current proxy channels:
  the normalised current is multiplied by ``cur_scale`` (a gain error) and gets Gaussian noise of std
  ``cur_noise``, then the proxy's own [0, 1.5] clip and -0.5 offset are re-applied.
* ``vxy_noise`` / ``vxy_zero`` (``KRABBY_STRESS_VXY_NOISE`` / ``_ZERO``) -- the body-frame planar velocity slot
  (dims 13-14): Gaussian noise of std ``vxy_noise`` m/s, or the slot zeroed (the "no velocity estimate" read).

Draws come from a PRIVATE generator seeded from the env seed (:func:`stress_obs_seed`), so arming a sensor knob
never shifts any other random stream of the run (commands, spawns, friction buckets).
"""

from __future__ import annotations

import torch

#: The env-cfg attribute ``stress_knobs`` writes (absent = off).
STRESS_OBS_ATTR = "stress_obs"

#: Reference noise half-widths in RAW units (rad/s, rad, rad, rad/s), multiplied by ``noise_k``.
OBS_NOISE_REF = {
    "ang_vel": 0.2,      # legged_gym ang_vel
    "roll_pitch": 0.05,  # legged_gym gravity (projected-gravity components ~ rad at small tilt)
    "joint_pos": 0.01,   # legged_gym dof_pos
    "joint_vel": 0.1,    # plant-scaled (legged_gym dof_vel 1.5 is for a ~30 rad/s joint)
}

#: The current proxy's normalised-current clip (``observations._CURRENT_SENSE_MAX``) and offset.
CURRENT_MAX = 1.5
CURRENT_OFFSET = 0.5

#: ``b"stress"`` -- added to the env seed so the private stream never equals the global stream.
STRESS_OBS_SEED_SALT = 0x737472657373

_KEYS = ("noise_k", "cur_scale", "cur_noise", "vxy_noise", "vxy_zero")


def stress_obs_seed(env_seed) -> int:
    """The private generator's seed for an env seed (``None`` -> as for 0)."""
    base = 0 if env_seed is None else int(env_seed)
    return (base + STRESS_OBS_SEED_SALT) % (1 << 63)


class StressObs:
    """Applies the armed sensor stresses (see the module docstring); built only when the spec is present."""

    def __init__(self, spec: dict, device, env_seed=None):
        unknown = sorted(set(spec) - set(_KEYS))
        if unknown:
            raise ValueError(f"{STRESS_OBS_ATTR}: unknown keys {unknown} (known: {list(_KEYS)})")
        self.noise_k = float(spec.get("noise_k", 0.0) or 0.0)
        self.cur_scale = spec.get("cur_scale")
        self.cur_noise = float(spec.get("cur_noise", 0.0) or 0.0)
        self.vxy_noise = float(spec.get("vxy_noise", 0.0) or 0.0)
        self.vxy_zero = bool(spec.get("vxy_zero", False))
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(stress_obs_seed(env_seed))

    def _uniform(self, like: torch.Tensor, half_width: float) -> torch.Tensor:
        u = torch.rand(like.shape, generator=self.generator, device=like.device, dtype=like.dtype)
        return (2.0 * u - 1.0) * half_width

    def _normal(self, like: torch.Tensor, std: float) -> torch.Tensor:
        return torch.randn(like.shape, generator=self.generator, device=like.device, dtype=like.dtype) * std

    def __call__(self, ang_vel, roll_pitch, lin_vel_xy, joint_pos_delta, joint_vel, contact_fill):
        """The six raw channel blocks -> the same six, stressed (new tensors; the inputs are not modified)."""
        if self.noise_k > 0.0:
            k = self.noise_k
            ang_vel = ang_vel + self._uniform(ang_vel, k * OBS_NOISE_REF["ang_vel"])
            roll_pitch = roll_pitch + self._uniform(roll_pitch, k * OBS_NOISE_REF["roll_pitch"])
            joint_pos_delta = joint_pos_delta + self._uniform(joint_pos_delta, k * OBS_NOISE_REF["joint_pos"])
            joint_vel = joint_vel + self._uniform(joint_vel, k * OBS_NOISE_REF["joint_vel"])
        if self.vxy_zero:
            lin_vel_xy = torch.zeros_like(lin_vel_xy)
        elif self.vxy_noise > 0.0:
            lin_vel_xy = lin_vel_xy + self._normal(lin_vel_xy, self.vxy_noise)
        if self.cur_scale is not None or self.cur_noise > 0.0:
            current = contact_fill + CURRENT_OFFSET
            if self.cur_scale is not None:
                current = current * float(self.cur_scale)
            if self.cur_noise > 0.0:
                current = current + self._normal(current, self.cur_noise)
            contact_fill = current.clamp(0.0, CURRENT_MAX) - CURRENT_OFFSET
        return ang_vel, roll_pitch, lin_vel_xy, joint_pos_delta, joint_vel, contact_fill


def build_stress_obs(env_cfg, device) -> StressObs | None:
    """``StressObs`` for an env cfg carrying :data:`STRESS_OBS_ATTR`, else None (the unarmed path)."""
    spec = getattr(env_cfg, STRESS_OBS_ATTR, None)
    if not spec:
        return None
    return StressObs(dict(spec), device, getattr(env_cfg, "seed", None))
