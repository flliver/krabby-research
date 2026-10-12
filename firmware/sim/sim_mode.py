"""Host sim mode flags (environment-driven)."""

from __future__ import annotations

import os

_SINGLE_LEADER_ENV = "KRABBY_SIM_SINGLE_LEADER_18"


def single_leader_eighteen_enabled() -> bool:
    """One USB instance owns 18 joints and emits three role telemetry lines."""
    raw = os.environ.get(_SINGLE_LEADER_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")
