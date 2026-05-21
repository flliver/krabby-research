"""On-policy runner for crab-hex tasks (resolves ``CrabHexActorCriticRMA`` policy class)."""

from __future__ import annotations

from .modules.crab_actor_critic_with_encoder import CrabHexActorCriticRMA
from .modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor


class OnPolicyRunnerCrabHex(OnPolicyRunnerWithExtractor):
    """Uses ``CrabHexActorCriticRMA`` when ``policy.class_name`` is ``CrabHexActorCriticRMA``."""

    def _resolve_policy_class(self, class_name: str):
        if class_name == "CrabHexActorCriticRMA":
            return CrabHexActorCriticRMA
        return super()._resolve_policy_class(class_name)
