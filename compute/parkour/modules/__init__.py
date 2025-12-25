"""Parkour modules for inference.

This module contains copies of OnPolicyRunnerWithExtractor and dependencies
from parkour.scripts.rsl_rl.modules, updated to use the compute.parkour namespace.
This avoids the transitive dependency on isaaclab_rl that exists in the original
parkour package.
"""

from .on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

__all__ = ["OnPolicyRunnerWithExtractor"]



