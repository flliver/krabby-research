"""Hall ``saf`` scale: simulated edge counts per joint stop-to-stop travel."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_SCALE_JSON = Path(__file__).with_name("hall_encoder_scale.json")
_FALLBACK_EDGES_PER_FULL_TRAVEL = 3200


@lru_cache(maxsize=1)
def load_hall_encoder_scale(path: Path | None = None) -> dict[str, Any]:
    """Load ``hall_encoder_scale.json`` (cached)."""
    scale_path = path or _SCALE_JSON
    with scale_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if "edges_per_full_travel" not in data:
        raise KeyError(f"{scale_path}: missing edges_per_full_travel")
    return data


def resolve_hall_edges_per_full_travel(*, scale_path: Path | None = None) -> int:
    """Effective stop-to-stop edge count for sim hall synthesis."""
    override = os.environ.get("KRABBY_HALL_EDGES_PER_FULL_TRAVEL", "").strip()
    if override:
        return int(override)
    try:
        return int(load_hall_encoder_scale(scale_path)["edges_per_full_travel"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _FALLBACK_EDGES_PER_FULL_TRAVEL


DEFAULT_HALL_EDGES_PER_FULL_TRAVEL = resolve_hall_edges_per_full_travel()


def edges_for_angle_delta(
    abs_delta_rad: float,
    joint_span_rad: float,
    *,
    edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
) -> int:
    """Map |Δθ| over a joint's soft-limit span to hall edge increments."""
    if joint_span_rad <= 0.0:
        return 0
    if abs_delta_rad <= 0.0:
        return 0
    return int(round(abs_delta_rad / joint_span_rad * edges_per_full_travel))


def edges_per_radian(
    joint_span_rad: float,
    *,
    edges_per_full_travel: int = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
) -> float:
    if joint_span_rad <= 0.0:
        return 0.0
    return float(edges_per_full_travel) / joint_span_rad
