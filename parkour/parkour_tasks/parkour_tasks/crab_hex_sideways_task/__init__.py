# SPDX-License-Identifier: BSD-3-Clause
"""Crab hexapod sideways-walk task package (travel along the body **+y** axis).

A thin sibling of ``crab_hex_forward_task``: the plant, the action space, the 35-term reward
class, the terminations and every knob are the forward ones; only the travel axis differs.
See ``config/crab_hex/crab_hex_side_env_cfg.py`` for the full delta.
"""

from .config import crab_hex  # noqa: F401
