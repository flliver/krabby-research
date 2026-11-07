# SPDX-License-Identifier: BSD-3-Clause
"""Per-env rod-speed domain randomisation for TRAINING (rod-speed DR campaign, 2026-11-07). DEFAULT OFF.

Armed only by ``crab_hex_omni_task/config/crab_hex/dr_knobs.py`` (``KRABBY_DR_ROD_SPEED=lo:hi[:ramp_steps]``),
which writes the action-cfg attribute :data:`ROD_SPEED_DR_ATTR` (a plain dict). The crab action term builds one
:class:`RodSpeedDR` and ALIASES its ``scale`` -- an ``(N, 1)`` tensor, written in place and never rebound -- as its
``_rod_speed_scale``, the factor the hip and knee slew limits already multiply (``parkour_actions.py``), so the slew
code changes by zero lines. Hip and knee share one draw per env, constant within an episode.

At every reset of an env its scale is redrawn ``s ~ U(lo_eff, hi)``, the reset envs only, in env-id order, from a
PRIVATE CPU ``torch.Generator`` seeded from the env seed plus :data:`ROD_SPEED_DR_SEED_SALT` -- no draw from the
global CPU or CUDA streams, so arming the DR does not shift the reset events' initial states against the same-seed
run without it. ``lo_eff`` ramps linearly from 1.0 (nominal) to ``lo`` over ``ramp_steps`` env steps, keyed on the
env's ``common_step_counter`` (:func:`rod_speed_bounds`; 0 = no ramp). Only the low edge ramps.

The current-sense proxy's drive gate keeps the RATED rod speed (a sensor constant), as under the stress knob.
"""

from __future__ import annotations

import hashlib
import math

import torch

#: The action-cfg attribute ``dr_knobs`` writes (absent = off).
ROD_SPEED_DR_ATTR = "rod_speed_dr"
#: ``b"rodspd"`` -- mixed into the generator seed so the stream never coincides with the stress push
#: (``0x70757368``), the stress sensor (``0x737472657373``) or the payload (``0x7061796C6F6164``) streams.
ROD_SPEED_DR_SEED_SALT = 0x726F64737064
#: The ``extras["log"]`` keys each reset returns (the low bound in force, then the scale over all N envs).
LOG_KEYS = ("DR/rod_speed_lo", "DR/rod_speed_mean", "DR/rod_speed_min", "DR/rod_speed_max")
_SPEC_KEYS = ("lo", "hi", "ramp_steps")


def rod_speed_dr_seed(env_seed) -> int:
    """The private generator's seed for an env seed (``None`` -> as for 0)."""
    base = 0 if env_seed is None else int(env_seed)
    return (base + ROD_SPEED_DR_SEED_SALT) % (1 << 63)


def rod_speed_bounds(step: int, lo: float, hi: float, ramp_steps: int) -> tuple[float, float]:
    """``(lo_eff, hi)`` at env step ``step``: ``lo_eff`` goes linearly from 1.0 at step 0 to ``lo`` at
    ``ramp_steps`` and stays there (``ramp_steps`` 0 = ``lo`` from the start)."""
    lo, hi, ramp_steps = float(lo), float(hi), int(ramp_steps)
    if ramp_steps <= 0:
        return lo, hi
    frac = min(max(int(step), 0), ramp_steps) / ramp_steps
    return 1.0 + (lo - 1.0) * frac, hi


def _sha16(values: torch.Tensor) -> str:
    """sha256 (first 16 hex) of a CPU float32 tensor's bytes."""
    raw = values.detach().to(device="cpu", dtype=torch.float32).contiguous().view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()[:16]


class RodSpeedDR:
    """The per-env rod-speed scale and its redraws; the action term reads ``scale`` as its slew-limit factor.

    ``scale`` ``(N, 1)`` starts at 1.0 (nominal) and is only ever written in place, so an alias taken at the action
    term's construction stays live. :meth:`reset` redraws the reset envs and returns the :data:`LOG_KEYS` dict.
    """

    def __init__(self, spec: dict, num_envs: int, device, env_seed=None):
        unknown = sorted(set(spec) - set(_SPEC_KEYS))
        missing = [k for k in _SPEC_KEYS if k not in spec]
        if unknown or missing:
            raise ValueError(f"{ROD_SPEED_DR_ATTR}: unknown keys {unknown}, missing keys {missing} "
                             f"(known: {list(_SPEC_KEYS)})")
        self.lo = float(spec["lo"])
        self.hi = float(spec["hi"])
        ramp = spec["ramp_steps"]
        if isinstance(ramp, bool) or float(ramp) != int(ramp) or int(ramp) < 0:
            raise ValueError(f"{ROD_SPEED_DR_ATTR}: ramp_steps must be an integer >= 0, got {ramp!r}")
        self.ramp_steps = int(ramp)
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and 0.0 < self.lo <= self.hi):
            raise ValueError(f"{ROD_SPEED_DR_ATTR}: need finite 0 < lo <= hi, got lo={spec['lo']!r} hi={spec['hi']!r}")
        if self.ramp_steps > 0 and not self.lo <= 1.0 <= self.hi:
            raise ValueError(f"{ROD_SPEED_DR_ATTR}: a ramp starts at 1.0, so it needs lo <= 1 <= hi, "
                             f"got lo={self.lo:g} hi={self.hi:g}")
        self.device = device
        self.num_envs = int(num_envs)
        self.seed = rod_speed_dr_seed(env_seed)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(self.seed)
        self.scale = torch.ones((self.num_envs, 1), dtype=torch.float32, device=device)
        self.resets = 0

    def bounds(self, step: int) -> tuple[float, float]:
        """``(lo_eff, hi)`` at env step ``step`` (:func:`rod_speed_bounds`)."""
        return rod_speed_bounds(step, self.lo, self.hi, self.ramp_steps)

    def reset(self, env_ids=None, step: int = 0) -> dict:
        """Redraw ``scale`` for the reset envs, in the order given (``None`` = every env, ascending).

        Draws ``k`` uniforms from the private CPU generator -- the i-th goes to ``env_ids[i]`` -- maps them onto
        ``[lo_eff(step), hi)`` and writes them in place. Returns ``{LOG_KEYS: 0-dim tensor}`` (no host sync: the
        stats are device reductions over all N envs). Prints one ``[DR]`` line on the first call.
        """
        if env_ids is None:
            ids = torch.arange(self.num_envs, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
        lo_eff, hi = self.bounds(step)
        u = torch.rand(int(ids.numel()), generator=self.gen, dtype=torch.float32)
        s = lo_eff + (hi - lo_eff) * u
        self.scale[ids.to(self.scale.device), 0] = s.to(self.scale.device)
        if self.resets == 0:
            stats = (f"min/mean/max {float(s.min()):.4f}/{float(s.mean()):.4f}/{float(s.max()):.4f}"
                     if s.numel() else "no envs")
            print(f"[DR] rod_speed n={int(s.numel())} step={int(step)} bounds=[{lo_eff:g}, {hi:g}] "
                  f"(lo={self.lo:g} hi={self.hi:g} ramp_steps={self.ramp_steps}) seed={self.seed} {stats} "
                  f"sha16={_sha16(s)} module={__file__}", flush=True)
        self.resets += 1
        col = self.scale[:, 0]
        return {
            LOG_KEYS[0]: torch.tensor(lo_eff, dtype=torch.float32),
            LOG_KEYS[1]: col.mean(),
            LOG_KEYS[2]: col.min(),
            LOG_KEYS[3]: col.max(),
        }


def build_rod_speed_dr(action_cfg, num_envs: int, device, env_seed=None) -> RodSpeedDR | None:
    """``RodSpeedDR`` for an action cfg carrying :data:`ROD_SPEED_DR_ATTR`, else None (the unarmed path: nothing
    is built, no generator exists)."""
    spec = getattr(action_cfg, ROD_SPEED_DR_ATTR, None)
    if not spec:
        return None
    return RodSpeedDR(dict(spec), num_envs, device, env_seed)
