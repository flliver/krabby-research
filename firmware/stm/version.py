"""Leg-board firmware version string for host tools and CI."""

from __future__ import annotations

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent / "VERSION"

__version__ = _VERSION_FILE.read_text(encoding="utf-8").strip()
