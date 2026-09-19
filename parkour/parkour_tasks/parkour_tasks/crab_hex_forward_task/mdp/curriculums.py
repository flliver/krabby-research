"""In-run reward-weight annealing for the gait-income phase-out campaign (PLAN G).

``ramp_reward_weight`` is a ramped variant of Isaac Lab's stock
``modify_reward_weight`` (a hard switch at ``num_steps``): the term weight follows a
cosine from ``w0`` to ``w1`` over the env-step window ``[t0, t1]``, holding ``w0``
before and ``w1`` after. Two campaign contracts are enforced here:

* **Epsilon floor** — targets are clamped to magnitude >= ``EPS_WEIGHT`` (1e-3).
  ``ParkourRewardManager.compute`` skips terms at weight exactly 0.0 and their
  ``Episode_Reward/<term>`` telemetry flatlines (parkour_reward_manager.py:27) — the
  campaign's gate metrics must stay alive at the floor.
* **Settled tails** — gates read income only after ``t1`` (constant weight); the
  logged curriculum value (the current weight) lets the orchestrator verify the
  schedule and normalize income by it.

Armed exclusively by ``KRABBY_PHASEOUT="term:w0:w1:t0:t1[,term2:...]"`` (t in env
steps, converted from training iterations by the orchestrator; the counter is
relative to process start, so per-segment ramps start at t0=0). Unset = no
curriculum attached = today's behavior, bit-identical.

The interpolation and the spec parser are pure (unit-tested without Isaac); the
manager-term class needs ``isaaclab`` and is defined only when it is importable.
"""

from __future__ import annotations

import math

EPS_WEIGHT = 1e-3


def ramp_value(step: int, w0: float, w1: float, t0: int, t1: int,
               eps: float = EPS_WEIGHT) -> float:
    """Cosine interpolation of a reward weight over env steps, epsilon-clamped.

    Returns ``w0`` for ``step <= t0``, ``w1`` for ``step >= t1``, cosine in between.
    Any output whose magnitude falls below ``eps`` is clamped to ``+-eps`` (sign of
    the nearer endpoint) so a term is never parked at exactly 0.0.
    """
    if t1 <= t0:
        raise ValueError(f"ramp window must satisfy t1 > t0, got [{t0}, {t1}]")
    if step <= t0:
        w = w0
    elif step >= t1:
        w = w1
    else:
        frac = (step - t0) / (t1 - t0)
        w = w1 + (w0 - w1) * 0.5 * (1.0 + math.cos(math.pi * frac))
    if abs(w) < eps:
        anchor = w1 if w1 != 0.0 else (w0 if w0 != 0.0 else 1.0)
        w = math.copysign(eps, anchor)
    return w


def parse_phaseout_spec(spec: str) -> list[dict]:
    """Parse ``"term:w0:w1:t0:t1[,term2:...]"`` into ramp param dicts.

    Raises ``ValueError`` on malformed entries so a typo fails the launch loudly
    instead of silently training the wrong schedule.
    """
    entries = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"KRABBY_PHASEOUT entry {chunk!r} must be term:w0:w1:t0:t1"
            )
        name = parts[0].strip()
        if not name.isidentifier():
            raise ValueError(f"KRABBY_PHASEOUT term name {name!r} is not an identifier")
        w0, w1 = float(parts[1]), float(parts[2])
        t0, t1 = int(parts[3]), int(parts[4])
        if t1 <= t0:
            raise ValueError(f"KRABBY_PHASEOUT entry {chunk!r}: t1 must exceed t0")
        entries.append({"term_name": name, "w0": w0, "w1": w1, "t0": t0, "t1": t1})
    if not entries:
        raise ValueError(f"KRABBY_PHASEOUT {spec!r} parsed to no entries")
    return entries


def format_phaseout_spec(entries: list[dict]) -> str:
    """Inverse of :func:`parse_phaseout_spec` (round-trip used by the orchestrator)."""
    return ",".join(
        f"{e['term_name']}:{e['w0']}:{e['w1']}:{e['t0']}:{e['t1']}" for e in entries
    )


try:  # pragma: no cover - importable only inside the Isaac Sim process
    from collections.abc import Sequence

    from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

    class ramp_reward_weight(ManagerTermBase):
        """Curriculum term: cosine-ramp one reward term's weight (see module docstring)."""

        def __init__(self, cfg: CurriculumTermCfg, env):
            super().__init__(cfg, env)
            self._term_name = cfg.params["term_name"]
            self._term_cfg = env.reward_manager.get_term_cfg(self._term_name)

        def __call__(
            self,
            env,
            env_ids: Sequence[int],
            term_name: str,
            w0: float,
            w1: float,
            t0: int,
            t1: int,
        ) -> float:
            weight = ramp_value(env.common_step_counter, w0, w1, t0, t1)
            if weight != self._term_cfg.weight:
                self._term_cfg.weight = weight
                env.reward_manager.set_term_cfg(self._term_name, self._term_cfg)
            return self._term_cfg.weight

    class _PhaseoutCurriculumCfg:
        """Attribute bag of CurriculumTermCfg entries (CurriculumManager iterates __dict__)."""

    def phaseout_curriculum_cfg_from_env(spec: str) -> _PhaseoutCurriculumCfg:
        cfg = _PhaseoutCurriculumCfg()
        for entry in parse_phaseout_spec(spec):
            setattr(
                cfg,
                f"phaseout_{entry['term_name']}",
                CurriculumTermCfg(func=ramp_reward_weight, params=dict(entry)),
            )
        return cfg

except ImportError:  # pure-python context (unit tests): pure functions above still work
    pass
