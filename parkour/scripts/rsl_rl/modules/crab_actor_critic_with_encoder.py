"""Crab-hex policy: ``ActorCriticRMA`` with capped exploration std (Go2 uses unclamped shared class)."""

from __future__ import annotations

import torch
from torch.distributions import Normal

from .actor_critic_with_encoder import ActorCriticRMA

ACTION_STD_MIN = 0.05
ACTION_STD_MAX = 2.0


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
    """

    def __init__(self, *args, action_std_max: float | None = None, action_std_reset: float | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.action_std_max = ACTION_STD_MAX if action_std_max is None else float(action_std_max)
        if not self.action_std_max > ACTION_STD_MIN:
            raise ValueError(f"action_std_max must exceed {ACTION_STD_MIN}, got {action_std_max!r}")
        self.action_std_reset = None if action_std_reset is None else float(action_std_reset)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
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
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        if self.action_std_reset is not None:
            loaded = self.std if self.noise_std_type == "scalar" else torch.exp(self.log_std)
            if float(loaded.max()) > self.action_std_max:
                self.reset_action_std(self.action_std_reset)
                print(f"[INFO] action_std_reset: loaded std max {float(loaded.max()):.4f} > ceiling "
                      f"{self.action_std_max}; exploration std set to {self.action_std_reset}")
            else:
                print(f"[INFO] action_std_reset: loaded std max {float(loaded.max()):.4f} within ceiling "
                      f"{self.action_std_max}; kept")
        return result
