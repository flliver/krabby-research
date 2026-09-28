# SPDX-License-Identifier: BSD-3-Clause
"""Env-var knob parsing + corridor geometry helpers for the obstacle-exposure campaign
(PLAN H, 2026-09-03). Pure Python, no Isaac imports -- unit-tested; the crab-hex env cfg
wires the results into the manager cfgs.

Knobs (all unset = bit-identical to before):
  KRABBY_FLAT_TERRAIN_GEOM=recal2b2w   widened recal preset (B0a)
  KRABBY_CORRIDOR_HALF_WIDTH=lo:hi     override gap/hurdle/step half_valid_width (m)
  KRABBY_STONE_WIDTH=w                 override stepping-stone width (m)
  KRABBY_STAND_FRAC=p                  Bernoulli standing slots (B1)
  KRABBY_SPAWN_OFFSET=m                platform spawn offset (B2; default 3.0)
  KRABBY_SPAWN_SPREAD=lo:hi[:frac]     tile-local spawn-x spread (B3)
  KRABBY_RSI_SPAWN_FIX=1               RSI resets placed where reset_root_state places (B4)
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TILE_WIDTH_M = 4.0
"""Sub-terrain width (size[1]); 3.92 m usable inside the 1-px border."""

CORRIDOR_HALF_WIDTH_MAX_M = 1.85
"""Above this the side trenches vanish and a gap/hurdle/step stops being an obstacle."""

CORRIDOR_HALF_WIDTH_MIN_M = 0.2
STONE_WIDTH_MAX_M = 2.0 * CORRIDOR_HALF_WIDTH_MAX_M
CRAB_HALF_STANCE_M = 1.19
"""Measured half-stance of the crab; the widened corridor must clear it with margin."""

RECAL2B2W_HALF_VALID_WIDTH = (1.40, 1.70)
RECAL2B2W_STONE_WIDTH = 2.8
RECAL2B2W_Y_RANGE = (-0.2, 0.2)
"""Lateral offset of each gap/hurdle/step corridor segment (stock (-0.4, 0.4)). The crab
walks the tile centre line blind to the offset, so the corridor must cover
half-stance + |offset| = 1.19 + 0.2 = 1.39 <= 1.40 (the preset's narrowest draw); at the
stock 0.4 m the outer foot would still ride the trench in ~20% of segments."""
RECAL2B2W_STONE_Y_RANGE = "0.05, 0.2"
"""Stepping-stone alternation offset (stock '0.2, 0.3+0.1*difficulty'): 1 px = 0.08 m at the
crab's 0.08 m grid, so a 2.8 m stone (half 1.40) covers 1.19 + 0.08."""
MAX_ABS_Y_OFFSET_M = 0.4

CORRIDOR_TERRAINS = ("parkour_gap", "parkour_hurdle", "parkour_step")
STONE_TERRAIN = "parkour"

SPAWN_OFFSET_MIN_M = 1.6
"""``reset_root_state`` spawns at tile-local x = 0.5*size_x - (size_y + offset) = 4.0 - offset
(16 x 4 m tiles): offset 3.0 -> 1.0 m (today), 2.0 -> 2.0 m (arm C3), 1.6 -> 2.4 m (just
inside the 2.48 m platform edge). Below 1.52 the spawn is already off the platform, so the
bound is 1.6, not the plan text's 1.0 (corrected during implementation, see CHANGELOG)."""
SPAWN_OFFSET_MAX_M = 3.5

SPREAD_X_MIN_M = 1.0
SPREAD_X_MAX_M = 15.0
SPREAD_FRAC_DEFAULT = 0.5


def parse_pair(spec: str, name: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(spec).split(":")]
    if len(parts) != 2:
        raise ValueError(f"{name}: expected 'lo:hi', got {spec!r}")
    lo, hi = (float(p) for p in parts)
    if not (lo <= hi):
        raise ValueError(f"{name}: lo must be <= hi, got {spec!r}")
    return lo, hi


def check_corridor_half_width(lo: float, hi: float) -> tuple[float, float]:
    if lo < CORRIDOR_HALF_WIDTH_MIN_M or hi > CORRIDOR_HALF_WIDTH_MAX_M or lo > hi:
        raise ValueError(
            f"corridor half_valid_width ({lo}, {hi}) must lie in "
            f"[{CORRIDOR_HALF_WIDTH_MIN_M}, {CORRIDOR_HALF_WIDTH_MAX_M}] m (tile {TILE_WIDTH_M} m wide)"
        )
    return float(lo), float(hi)


def check_stone_width(w: float) -> float:
    if not (0.0 < w <= STONE_WIDTH_MAX_M):
        raise ValueError(f"stone_width {w} must lie in (0, {STONE_WIDTH_MAX_M}] m")
    return float(w)


def apply_corridor_widths(
    sub_terrains: Mapping[str, Any],
    half_valid_width: tuple[float, float] | None = None,
    stone_width: float | None = None,
    y_range: tuple[float, float] | None = None,
    stone_y_range: str | None = None,
) -> dict[str, Any]:
    """Set corridor widths (and, for the preset, lateral offsets) on the sub-terrain cfgs
    present. Returns what was applied."""
    applied: dict[str, Any] = {}
    if y_range is not None:
        lo, hi = float(y_range[0]), float(y_range[1])
        if not (-MAX_ABS_Y_OFFSET_M <= lo <= hi <= MAX_ABS_Y_OFFSET_M):
            raise ValueError(f"corridor y_range {y_range} must lie within +-{MAX_ABS_Y_OFFSET_M} m")
        for key in CORRIDOR_TERRAINS:
            if key in sub_terrains:
                sub_terrains[key].y_range = (lo, hi)
                applied[f"{key}.y_range"] = (lo, hi)
    if stone_y_range is not None and STONE_TERRAIN in sub_terrains:
        sub_terrains[STONE_TERRAIN].y_range = str(stone_y_range)
        applied[f"{STONE_TERRAIN}.y_range"] = str(stone_y_range)
    if half_valid_width is not None:
        lo, hi = check_corridor_half_width(*half_valid_width)
        for key in CORRIDOR_TERRAINS:
            if key in sub_terrains:
                sub_terrains[key].half_valid_width = (lo, hi)
                applied[key] = (lo, hi)
    if stone_width is not None and STONE_TERRAIN in sub_terrains:
        sub_terrains[STONE_TERRAIN].stone_width = check_stone_width(stone_width)
        applied[STONE_TERRAIN] = float(stone_width)
    return applied


def corridor_overrides_from_env(environ: Mapping[str, str]) -> tuple[tuple[float, float] | None, float | None]:
    hw = environ.get("KRABBY_CORRIDOR_HALF_WIDTH")
    sw = environ.get("KRABBY_STONE_WIDTH")
    half = check_corridor_half_width(*parse_pair(hw, "KRABBY_CORRIDOR_HALF_WIDTH")) if hw else None
    stone = check_stone_width(float(sw)) if sw else None
    return half, stone


def parse_stand_frac(spec: str) -> float:
    p = float(spec)
    if not (0.0 <= p < 1.0):
        raise ValueError(f"KRABBY_STAND_FRAC must be in [0, 1), got {spec!r}")
    return p


def parse_spawn_offset(spec: str) -> float:
    off = float(spec)
    if not (SPAWN_OFFSET_MIN_M <= off <= SPAWN_OFFSET_MAX_M):
        raise ValueError(
            f"KRABBY_SPAWN_OFFSET must lie in [{SPAWN_OFFSET_MIN_M}, {SPAWN_OFFSET_MAX_M}] m, got {spec!r}"
        )
    return off


def parse_spawn_spread(spec: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in str(spec).split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"KRABBY_SPAWN_SPREAD: expected 'lo:hi[:frac]', got {spec!r}")
    lo, hi = float(parts[0]), float(parts[1])
    frac = float(parts[2]) if len(parts) == 3 else SPREAD_FRAC_DEFAULT
    if not (SPREAD_X_MIN_M <= lo < hi <= SPREAD_X_MAX_M):
        raise ValueError(
            f"KRABBY_SPAWN_SPREAD: need {SPREAD_X_MIN_M} <= lo < hi <= {SPREAD_X_MAX_M} (tile-local m), got {spec!r}"
        )
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"KRABBY_SPAWN_SPREAD: frac must be in (0, 1], got {spec!r}")
    return lo, hi, frac


def truthy(spec: str | None) -> bool:
    return bool(spec) and spec.strip().lower() in ("1", "true", "yes", "on")
