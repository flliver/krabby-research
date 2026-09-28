"""When bench E2E runs strict (pass/fail) vs skipped locally."""
from __future__ import annotations

import os


def bench_e2e_requested() -> bool:
    """True when bench integration should run (CI or explicit local enable)."""
    flag = os.environ.get("BENCH_E2E", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    return os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true"
