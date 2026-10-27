# SPDX-License-Identifier: BSD-3-Clause
"""Sideways-task MDP helpers.

The MDP itself is the forward task's (``crab_hex_forward_task/mdp``, reused unchanged); the
only piece that is genuinely direction-specific is the symmetry mirror, whose plane is the
**fore-aft** plane ``x = 0`` instead of the forward task's left/right plane ``y = 0``.

``parkour_tasks.__init__`` blacklists the ``".mdp"`` substring, so this package is NOT imported
by ``import_packages``; it is reached through the rsl-rl symmetry dotted string
``parkour_tasks.crab_hex_sideways_task.mdp.crab_hex_side_mirror:crab_hex_side_symmetry_augmentation``
(see ``config/crab_hex/agents/rsl_rl_ppo_cfg.py``).
"""

from .crab_hex_side_mirror import crab_hex_side_symmetry_augmentation  # noqa: F401
