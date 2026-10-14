"""Hall ``saf`` scale: simulated edge counts per joint stop-to-stop travel."""

from __future__ import annotations

# Placeholder until a stop-to-stop ``saf`` delta is measured on hardware (fleet
# telemetry while jogging one joint min→max) or supplied by the mechanical team.
# Replace with measured stop-to-stop encoder triggers when available.
DEFAULT_HALL_EDGES_PER_FULL_TRAVEL = 3200


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
