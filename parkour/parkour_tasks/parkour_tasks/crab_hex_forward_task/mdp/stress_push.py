# SPDX-License-Identifier: BSD-3-Clause
"""The additive push of the pre-hardware DR stress sweep (omni rung 1, 2026-11-06). DEFAULT OFF.

Armed only by ``crab_hex_omni_task/config/crab_hex/stress_knobs.py`` (``KRABBY_STRESS_PUSH_DV``), which writes the
action-cfg attribute :data:`STRESS_PUSH_ATTR` (a plain dict). The crab action term then calls this object once per env
step, before physics: every ``interval_s`` per env (counted in env steps, restarted at each reset of that env), the
root's current world velocity gets a horizontal change of fixed magnitude ``dv`` m/s at a uniform random heading -- an
impulse of ``m * dv`` N s.

Why not an IsaacLab interval EVENT: the event manager draws the per-env timers from the GLOBAL CUDA generator at its
construction, at every reset and at every fire, so arming the push would shift the initial states the reset events
draw (suite wave 0 and the whole field_light row are not re-seeded by the harness) against the same-seed nominal. Here
the timers are plain counters and consume no random stream at all. (It is also why the push is not the inherited
``push_by_setting_velocity``, which REPLACES the xy velocity and which the eval harness nulls on every row.)

The heading is COUNTER-BASED, not a stream position: ``h = 2 pi u(seed, epoch, env, k)`` with ``u`` a splitmix64 hash
of the env seed, the EPOCH (the number of full resets so far -- the harness resets every env at each suite wave), the
env id and ``k``, that env's push count within the epoch. So the k-th push of env e in wave w has the same heading in
every cell and at every push level, whatever fell in earlier waves.
"""

from __future__ import annotations

import math

import torch

#: The action-cfg attribute ``stress_knobs`` writes (absent = off).
STRESS_PUSH_ATTR = "stress_push"
#: ``b"push"`` -- mixed into the heading hash so it never coincides with another seeded stream.
STRESS_PUSH_SEED_SALT = 0x70757368
_MASK64 = (1 << 64) - 1


def stress_push_seed(env_seed) -> int:
    """The heading hash's seed for an env seed (``None`` -> as for 0)."""
    base = 0 if env_seed is None else int(env_seed)
    return (base + STRESS_PUSH_SEED_SALT) % (1 << 63)


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def push_heading(seed: int, epoch: int, env: int, k: int) -> float:
    """The heading (rad, in [0, 2 pi)) of env ``env``'s ``k``-th push in reset epoch ``epoch``: a pure function."""
    x = _splitmix64(int(seed) & _MASK64)
    x = _splitmix64(x ^ (int(epoch) & _MASK64))
    x = _splitmix64(x ^ (int(env) & _MASK64))
    x = _splitmix64(x ^ (int(k) & _MASK64))
    return (x >> 11) * (1.0 / (1 << 53)) * (2.0 * math.pi)


class StressPush:
    """Per-env push timers and draws; the caller writes the velocity change it returns into the sim.

    ``interval_steps`` env steps after every reset (and then every ``interval_steps``), :meth:`step` returns the env
    ids to push and their ``(k, 2)`` world-xy velocity change. ``pushes`` ``(N,)`` counts the pushes per env (all
    time); ``epoch`` counts the full resets and ``epoch_pushes`` ``(N,)`` the pushes per env within the epoch.
    """

    def __init__(self, spec: dict, num_envs: int, step_dt: float, device, env_seed=None):
        unknown = sorted(set(spec) - {"dv", "interval_s"})
        if unknown:
            raise ValueError(f"{STRESS_PUSH_ATTR}: unknown keys {unknown} (known: ['dv', 'interval_s'])")
        self.dv = float(spec["dv"])
        interval_s = float(spec["interval_s"])
        if not (math.isfinite(self.dv) and self.dv > 0.0):
            raise ValueError(f"{STRESS_PUSH_ATTR}: dv must be > 0 m/s, got {spec['dv']!r}")
        if not (math.isfinite(interval_s) and interval_s > 0.0):
            raise ValueError(f"{STRESS_PUSH_ATTR}: interval_s must be > 0 s, got {spec['interval_s']!r}")
        self.interval_steps = max(1, int(round(interval_s / float(step_dt))))
        self.device = device
        self.num_envs = int(num_envs)
        self.timer = torch.full((self.num_envs,), self.interval_steps, dtype=torch.long, device=device)
        self.pushes = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.seed = stress_push_seed(env_seed)
        self.epoch = 0
        self.epoch_pushes = [0] * self.num_envs

    def reset(self, env_ids=None) -> None:
        """Restart the reset envs' timers; a reset of EVERY env (the harness's per-wave reset) opens a new epoch."""
        if env_ids is None:
            ids = None
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids is None or int(torch.unique(ids).numel()) == self.num_envs:
            self.timer[:] = self.interval_steps
            self.epoch += 1
            self.epoch_pushes = [0] * self.num_envs
        else:
            self.timer[ids] = self.interval_steps

    def step(self):
        """Advance one env step. Returns ``(ids, dvel)`` -- ``ids`` ``(k,)`` long, ``dvel`` ``(k, 2)`` -- or None."""
        self.timer -= 1
        ids = (self.timer <= 0).nonzero().flatten()
        if ids.numel() == 0:
            return None
        self.timer[ids] = self.interval_steps
        self.pushes[ids] += 1
        headings = []
        for e in ids.tolist():
            headings.append(push_heading(self.seed, self.epoch, e, self.epoch_pushes[e]))
            self.epoch_pushes[e] += 1
        heading = torch.tensor(headings, dtype=torch.float64)
        dvel = (torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1) * self.dv).to(
            device=self.device, dtype=torch.float32)
        return ids, dvel


def build_stress_push(action_cfg, num_envs: int, step_dt: float, device, env_seed=None) -> StressPush | None:
    """``StressPush`` for an action cfg carrying :data:`STRESS_PUSH_ATTR`, else None (the unarmed path)."""
    spec = getattr(action_cfg, STRESS_PUSH_ATTR, None)
    if not spec:
        return None
    return StressPush(dict(spec), num_envs, step_dt, device, env_seed)
