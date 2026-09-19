# SPDX-License-Identifier: BSD-3-Clause
"""Pure (no Isaac Sim) scoring layer for the crab-hex gait eval harness.

Deliberately imports nothing from ``torch``, ``isaaclab``, or ``parkour_tasks`` -- anything inside
the ``parkour_tasks`` package pulls in ``isaaclab_tasks`` on import, which is why this lives under
``scripts/`` and is imported as a top-level module rather than a package submodule.

``eval_crab_hex_gait.py`` does the simulating; everything here just scores arrays.
"""

from gait_eval import metrics, schedule  # noqa: F401

__all__ = ["metrics", "schedule"]
