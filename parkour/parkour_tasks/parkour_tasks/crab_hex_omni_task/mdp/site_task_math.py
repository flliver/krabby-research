# SPDX-License-Identifier: BSD-3-Clause
"""Pure-torch math of the construction-site per-kind telemetry (ported, TRIMMED, from the closed site teacher's
``site_task_math.py`` at ``worktree-sen0581-cameras`` 2a3e3c51; rough-terrain campaign
``experiments/2026-11-10_rough_terrain/PLAN.md`` step 1, 2026-11-10). DEFAULT OFF: nothing here runs unless the
terrain generator carries site layers (``KRABBY_FLAT_TERRAIN_GEOM=site``).

What is kept: the kind / policy-class literals (pinned to ``site_raster`` by ``tests/unit/test_site_task_math.py``),
:func:`crossed` (the per-kind ledger's "crossed the hazard" share) and the hazard inner-radius tables
(:func:`hazard_inner_radius`, :func:`hazard_inner_radius_table`), which the offline read-out may use.

What is NOT ported (the rough-terrain plan keeps main's ratio promotion rule ``KRABBY_TERRAIN_PROMOTE`` and has no
blockers, slack, edge / void rewards or impact termination): the site promotion table (``promotion_site``), rule
``loc`` (``promotion_site_loc`` / ``promotion_site_rule``), the blocked-path / corridor slack, the foot / edge / void
helpers, the blocker contact and impact math. Their constants are not restated either.

This file path-loads standalone (module scope imports ``math`` and ``torch`` only).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "KIND_GROUND", "VOID_KINDS", "BLOCKER_KINDS",
    "POLICY_LEGACY", "POLICY_CROSS", "POLICY_VOID", "POLICY_BLOCKER",
    "crossed",
    "CROSS_LEAD_M", "TRENCH_CENTRE_R_M", "STRADDLE_CENTRE_R_M",
    "HAZARD_INNER_R_M", "hazard_inner_radius", "hazard_inner_radius_table",
]

# ------------------------------------------------------------------------------ kind ids (= site_raster)
#: ``site_raster.KIND_GROUND`` / ``VOID_KINDS`` / ``BLOCKER_KINDS`` (literals: this file loads standalone).
KIND_GROUND = 0
VOID_KINDS = (1, 2)            # KIND_PIT, KIND_TRENCH
BLOCKER_KINDS = (7, 8)         # KIND_BLOCKER_CHEST, KIND_BLOCKER_WALL
#: ``site_raster.POLICY_*`` (the column classes, ``SITE_KIND_POLICY``).
POLICY_LEGACY, POLICY_CROSS, POLICY_VOID, POLICY_BLOCKER = 0, 1, 2, 3

# ------------------------------------------------------------------------------ hazard radii
#: ``R_cross = centre + w / 2 + CROSS_LEAD_M`` for the two ring kinds whose width is drawn (``site_terrains`` trench /
#: straddle), so their inner radius ``centre - w / 2`` follows from the tile's ``cross_r``; the centres are
#: ``site_terrains.TRENCH_R_M`` / ``STRADDLE_R_M`` (literals: this file loads standalone; the tests pin them).
CROSS_LEAD_M = 0.5
TRENCH_CENTRE_R_M = 2.4
STRADDLE_CENTRE_R_M = 2.6
#: The fixed inner hazard radius (m, tile-centre frame) of every other CROSS / VOID kind: the annulus band's inner
#: edge 2.0 (pits, rubble, ramps; ``site_terrains.BAND_R_IN_M``), the pad edge / first riser 1.7 (ledges, terraces;
#: ``PAD_R_M``, ``TERRACE_RISER_R_M[0]``), the inner snag ring's inner edge 2.2 - 0.16 / 2.
HAZARD_INNER_R_M = {"pits": 2.0, "rubble": 2.0, "ramps": 2.0, "ledge_down": 1.7, "ledge_up": 1.7, "terraces": 1.7,
                    "snags": 2.2 - 0.08}
_HAZARD_CENTRE_R_M = {"trench": TRENCH_CENTRE_R_M, "straddle": STRADDLE_CENTRE_R_M}


# ============================================================================ crossing
def crossed(r_max: torch.Tensor, cross_r: torch.Tensor) -> torch.Tensor:
    """``r_max >= cross_r`` per episode, False where ``cross_r`` is NaN (no crossing defined: legacy,
    blocker and flat columns). Both ``(n,)``, metres from the tile centre."""
    cross_r = cross_r.to(r_max.device, dtype=r_max.dtype)
    return torch.isfinite(cross_r) & (r_max >= cross_r)


def hazard_inner_radius(kind: str, cross_r):
    """The inner radius (m, tile-centre frame) of a CROSS / VOID kind's hazard: trench / straddle ``c - ((cross_r -
    0.5) - c)`` (``c`` the ring centre; ``= c - w / 2``), every other kind its :data:`HAZARD_INNER_R_M` constant, NaN
    for a kind without one (flat, fields, blockers). ``cross_r`` is a float or a tensor; the result has its type."""
    centre = _HAZARD_CENTRE_R_M.get(kind)
    if centre is not None:
        return centre - ((cross_r - CROSS_LEAD_M) - centre)
    value = HAZARD_INNER_R_M.get(kind, math.nan)
    if isinstance(cross_r, torch.Tensor):
        return torch.full_like(cross_r, value)
    return float(value)


def hazard_inner_radius_table(tile_kind: torch.Tensor, tile_cross_r: torch.Tensor,
                              kinds: tuple[str, ...]) -> torch.Tensor:
    """``(R, C)`` float32 :func:`hazard_inner_radius` of every tile: ``tile_kind`` indexes ``kinds`` (-1 = not a site
    key -> NaN), ``tile_cross_r`` the tile's crossing radius (``site_raster.SiteLayers``)."""
    cross = tile_cross_r.to(torch.float64)
    tk = tile_kind.to(cross.device).long()
    out = torch.full(cross.shape, math.nan, dtype=torch.float64, device=cross.device)
    for i, name in enumerate(kinds):
        m = tk == i
        if bool(m.any()):
            out = torch.where(m, hazard_inner_radius(name, cross), out)
    return out.to(torch.float32)
