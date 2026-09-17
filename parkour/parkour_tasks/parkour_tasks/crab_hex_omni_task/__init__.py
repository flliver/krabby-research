# SPDX-License-Identifier: BSD-3-Clause
"""Crab hexapod omnidirectional task package (any world travel direction, any world facing).

A thin sibling of ``crab_hex_forward_task``: the plant, the action space, the forward 35 reward
term NAMES, the terminations and every knob are the forward ones; what differs is the command
space (a world travel direction with a speed, plus an independent world facing target), the
ring course it is trained on, the handful of reward / observation / event functions those two
imply, and exactly ONE added reward term (``reward_yaw_progress``, the 36th name -- rotational
progress, which nothing in the forward 35 paid for). See
``config/crab_hex/crab_hex_omni_env_cfg.py`` for the full delta.

Teacher only -- no depth student in this plan.
"""

from .config import crab_hex  # noqa: F401
