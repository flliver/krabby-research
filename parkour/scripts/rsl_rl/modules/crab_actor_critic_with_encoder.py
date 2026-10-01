"""Crab-hex policy: ``ActorCriticRMA`` with capped exploration std (Go2 uses unclamped shared class)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic_with_encoder import ActorCriticRMA

ACTION_STD_MIN = 0.05
ACTION_STD_MAX = 2.0

#: The ``soft`` std type's log-std band (PPO-construction plan step 3, 2026-10-01): the same
#: [0.05, 2.0] as the scalar clamp, but reached smoothly, so the gradient never dies at a cap.
SOFT_LOG_STD_LO = math.log(ACTION_STD_MIN)
SOFT_LOG_STD_HI = math.log(ACTION_STD_MAX)


def soft_log_std(x: torch.Tensor) -> torch.Tensor:
    """``lo + 0.5 (hi - lo) (tanh(x) + 1)`` for the unconstrained parameter ``x``.

    Computed as ``hi - (hi - lo) sigmoid(-2x)`` for ``x >= 0`` and ``lo + (hi - lo) sigmoid(2x)``
    below (the same function: ``0.5 (tanh(x) + 1) = sigmoid(2x) = 1 - sigmoid(-2x)``). The literal
    tanh form rounds ``tanh(x)`` to exactly 1 in float32 past ``|x| ~ 9`` and its gradient to exactly
    0; each branch here keeps the small sigmoid tail, so the gradient is small but finite near both
    caps (non-zero until the tail underflows, ``|x| > ~44``).
    """
    span = SOFT_LOG_STD_HI - SOFT_LOG_STD_LO
    return torch.where(
        x >= 0,
        SOFT_LOG_STD_HI - span * torch.sigmoid(-2.0 * x),
        SOFT_LOG_STD_LO + span * torch.sigmoid(2.0 * x),
    )


def soft_std(x: torch.Tensor) -> torch.Tensor:
    """The ``soft`` type's sigma: ``exp(soft_log_std(x))``, strictly inside (0.05, 2.0)."""
    return torch.exp(soft_log_std(x))


def soft_std_inverse(std: float) -> float:
    """The exact inverse of :func:`soft_std` (in float64): the ``x`` whose sigma is ``std``.

    ``x = atanh(2u - 1) = 0.5 ln(u / (1 - u))`` with ``u = (ln std - lo) / (hi - lo)``. Only an
    OPEN-interval value has one: the caps themselves sit at ``x = +-inf``.
    """
    std = float(std)
    if not (math.isfinite(std) and ACTION_STD_MIN < std < ACTION_STD_MAX):
        raise ValueError(f"soft std must lie strictly inside ({ACTION_STD_MIN}, {ACTION_STD_MAX}), got {std!r}")
    u = (math.log(std) - SOFT_LOG_STD_LO) / (SOFT_LOG_STD_HI - SOFT_LOG_STD_LO)
    return 0.5 * math.log(u / (1.0 - u))


class CrabHexActorCriticRMA(ActorCriticRMA):
    """Same architecture as ``ActorCriticRMA``; clamps Gaussian std during training rollouts.

    Two OPTIONAL policy-cfg keys (motion-quality plan step 2, 2026-09-22; DEFAULT OFF -- no shipped
    runner cfg carries them, so the stock clamp and the untouched load path apply), written by
    ``crab_hex_omni_task/config/crab_hex/motion_knobs.apply_exploration_knobs``:

    ``action_std_max`` (``KRABBY_ACTION_STD_MAX``)
        replaces :data:`ACTION_STD_MAX` as the clamp ceiling.
    ``action_std_reset`` (``KRABBY_ACTION_STD_RESET``)
        on a :meth:`load_state_dict` whose loaded std EXCEEDS the clamp ceiling, the std parameter
        is overwritten with this value after the checkpoint's weights are loaded -- the fork from a
        head whose std sits above the clamp (where the clamp's gradient is zero, so it can never come
        back down) starts from the asked exploration level instead. A load whose std is already
        within the ceiling (a later stage or a crash resume of a run that already forked) keeps its
        learned std (review 2026-09-23: an every-load reset would discard it at each stage boundary).

    A third std TYPE (PPO-construction plan step 3, 2026-10-01; DEFAULT OFF, ``KRABBY_STD_PARAM=soft``
    writes ``noise_std_type: soft``): the parameter ``std_raw`` is unconstrained and the sigma is
    :func:`soft_std` of it -- a tanh-bounded log-std in ``[ln 0.05, ln 2.0]``. There is NO hard clamp
    for this type, so the gradient reaches sigma everywhere (small but finite near the caps; the
    clamped scalar's is exactly zero above 2.0, which is where every head of record sits). It is
    initialised from ``init_noise_std`` by the exact inverse (:func:`soft_std_inverse`), and
    :meth:`reset_action_std` writes through the inverse too. Its ceiling is fixed at 2.0, so
    ``action_std_max`` / ``action_std_reset`` are refused with it (neither has a meaning: no
    clamp to move, and no loaded sigma can exceed the ceiling).
    """

    def __init__(self, *args, action_std_max: float | None = None, action_std_reset: float | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if self.noise_std_type == "soft" and (action_std_max is not None or action_std_reset is not None):
            raise ValueError("noise_std_type 'soft' has a fixed (0.05, 2.0) band and no clamp: "
                             "action_std_max / action_std_reset do not apply to it")
        self.action_std_max = ACTION_STD_MAX if action_std_max is None else float(action_std_max)
        if not self.action_std_max > ACTION_STD_MIN:
            raise ValueError(f"action_std_max must exceed {ACTION_STD_MIN}, got {action_std_max!r}")
        self.action_std_reset = None if action_std_reset is None else float(action_std_reset)

    def _build_noise_std(self, init_noise_std, num_actions) -> None:
        if self.noise_std_type == "soft":
            x0 = soft_std_inverse(init_noise_std)
            self.std_raw = nn.Parameter(x0 * torch.ones(num_actions))
            return
        super()._build_noise_std(init_noise_std, num_actions)

    def effective_std(self) -> torch.Tensor:
        """``(num_actions,)`` -- the std :meth:`update_distribution` actually uses (post clamp / soft map)."""
        if self.noise_std_type == "soft":
            return soft_std(self.std_raw)
        return super().effective_std().clamp(ACTION_STD_MIN, self.action_std_max)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        if self.noise_std_type == "soft":
            # no clamp: soft_std is already inside (0.05, 2.0) and differentiable everywhere
            self.distribution = Normal(mean, soft_std(self.std_raw).expand_as(mean))
            return
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar', 'log' or 'soft'"
            )
        std = std.clamp(ACTION_STD_MIN, self.action_std_max)
        self.distribution = Normal(mean, std)

    def reset_action_std(self, value: float) -> None:
        """Write ``value`` into the std parameter in place (the optimizer keeps its reference)."""
        with torch.no_grad():
            if self.noise_std_type == "scalar":
                self.std.fill_(float(value))
            elif self.noise_std_type == "log":
                self.log_std.fill_(float(torch.log(torch.tensor(float(value)))))
            elif self.noise_std_type == "soft":
                self.std_raw.fill_(soft_std_inverse(value))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        if self.action_std_reset is not None:
            loaded = self.std if self.noise_std_type == "scalar" else torch.exp(self.log_std)
            # read BEFORE the reset: ``loaded`` aliases the parameter the reset fills in place
            loaded_max = float(loaded.max())
            if loaded_max > self.action_std_max:
                self.reset_action_std(self.action_std_reset)
                print(f"[INFO] action_std_reset: loaded std max {loaded_max:.4f} > ceiling "
                      f"{self.action_std_max}; exploration std set to {self.action_std_reset}")
            else:
                print(f"[INFO] action_std_reset: loaded std max {loaded_max:.4f} within ceiling "
                      f"{self.action_std_max}; kept")
        return result
