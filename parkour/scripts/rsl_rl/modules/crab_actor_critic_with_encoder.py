"""Crab-hex policy: ``ActorCriticRMA`` with capped exploration std (Go2 uses unclamped shared class)."""

from __future__ import annotations

import torch
from torch.distributions import Normal

from .actor_critic_with_encoder import ActorCriticRMA

ACTION_STD_MIN = 0.05
ACTION_STD_MAX = 2.0


class CrabHexActorCriticRMA(ActorCriticRMA):
    """Same architecture as ``ActorCriticRMA``; clamps Gaussian std during training rollouts."""

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
        std = std.clamp(ACTION_STD_MIN, ACTION_STD_MAX)
        self.distribution = Normal(mean, std)
