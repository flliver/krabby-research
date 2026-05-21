"""Factory for RSL-RL on-policy runners (Go2 + crab hex)."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

from rsl_rl.env import VecEnv

from .crab_on_policy_runner import OnPolicyRunnerCrabHex
from .modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

# Scalar runner fields that must not come from a corrupted ``configclass.to_dict()`` (e.g.
# ``num_steps_per_env`` replaced by ``obs_groups`` dict on multi-inherit Isaac Lab cfgs).
_RUNNER_SCALAR_KEYS = (
    "runner_class_name",
    "num_steps_per_env",
    "save_interval",
    "max_iterations",
    "clip_actions",
    "experiment_name",
    "device",
    "seed",
    "empirical_normalization",
)


def agent_cfg_to_train_dict(agent_cfg: Any) -> dict:
    """Build train dict for ``OnPolicyRunnerWithExtractor``; keep scalars from the live cfg object."""
    train_cfg = agent_cfg.to_dict()
    for key in _RUNNER_SCALAR_KEYS:
        if hasattr(agent_cfg, key):
            val = getattr(agent_cfg, key)
            if val is MISSING:
                continue
            train_cfg[key] = val
    policy = train_cfg.get("policy")
    if isinstance(policy, dict) and hasattr(agent_cfg, "policy") and hasattr(agent_cfg.policy, "class_name"):
        policy["class_name"] = agent_cfg.policy.class_name
    return train_cfg


def _use_crab_runner(train_cfg: dict) -> bool:
    if train_cfg.get("runner_class_name") == "OnPolicyRunnerCrabHex":
        return True
    policy_class = train_cfg.get("policy", {}).get("class_name")
    if policy_class == "CrabHexActorCriticRMA":
        return True
    estimator = train_cfg.get("estimator", {})
    return estimator.get("num_prop") == 75


def make_on_policy_runner(
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None,
    device: str,
) -> OnPolicyRunnerWithExtractor:
    """Go2 uses ``OnPolicyRunnerWithExtractor``; crab uses ``OnPolicyRunnerCrabHex``."""
    if _use_crab_runner(train_cfg):
        return OnPolicyRunnerCrabHex(env, train_cfg, log_dir=log_dir, device=device)
    return OnPolicyRunnerWithExtractor(env, train_cfg, log_dir=log_dir, device=device)
