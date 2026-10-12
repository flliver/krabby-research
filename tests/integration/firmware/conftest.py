"""Firmware Isaac integration pytest hooks.

When ``FIRMWARE_ISAAC_RESULTS_DIR`` is set (``run_firmware_isaac_pytest.sh`` in Docker),
each completed test call phase is appended to ``outcomes.jsonl`` on disk. The host reads
that file for pass/fail — do not rely on the container process exit code alone; Kit teardown
can exit 0 before pytest prints its summary.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

_RESULTS_DIR: Path | None = None
_ISAAC_DEFERRED_TEARDOWN: list[Callable[[], None]] = []


def defer_isaac_teardown(callback: Callable[[], None]) -> None:
    """Close Kit only after pytest finishes reporting (``simulation_app.close()`` kills the process)."""
    _ISAAC_DEFERRED_TEARDOWN.append(callback)


def _results_dir() -> Path | None:
    global _RESULTS_DIR
    if _RESULTS_DIR is not None:
        return _RESULTS_DIR
    raw = os.environ.get("FIRMWARE_ISAAC_RESULTS_DIR", "").strip()
    if not raw:
        return None
    _RESULTS_DIR = Path(raw)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def _append_result(record: dict[str, Any]) -> None:
    directory = _results_dir()
    if directory is None:
        return
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with (directory / "outcomes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def pytest_collection_finish(session) -> None:
    directory = _results_dir()
    if directory is None:
        return
    meta = {
        "collected": len(session.items),
        "nodeids": [item.nodeid for item in session.items],
    }
    (directory / "collected.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def pytest_sessionfinish(session, exitstatus: int) -> None:
    directory = _results_dir()
    if directory is not None:
        payload = {
            "pytest_exitstatus": int(exitstatus),
            "collected": len(session.items),
        }
        (directory / "session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    while _ISAAC_DEFERRED_TEARDOWN:
        callback = _ISAAC_DEFERRED_TEARDOWN.pop()
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup after tests recorded
            _append_result({"nodeid": "isaac_teardown", "outcome": "error", "longrepr": str(exc)})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    _append_result(
        {
            "nodeid": item.nodeid,
            "outcome": report.outcome,
            "duration_s": report.duration,
            "longrepr": str(report.longrepr) if report.failed or report.skipped else None,
        }
    )
