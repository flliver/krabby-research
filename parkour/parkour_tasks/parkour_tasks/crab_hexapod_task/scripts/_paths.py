"""Shared paths for crab_hex USD tools and Isaac diagnostics."""

from __future__ import annotations

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
# scripts → crab_hexapod_task → parkour_tasks → parkour_tasks → parkour → krabby-research
_REPO_ROOT = _SCRIPT_DIR.parents[4]
_PARKOUR_ROOT = _REPO_ROOT / "parkour"


def repo_root() -> Path:
    return _REPO_ROOT


def parkour_root() -> Path:
    return _PARKOUR_ROOT


def parkour_scripts_dir() -> Path:
    return _PARKOUR_ROOT / "scripts"


def crab_simple_usd() -> Path:
    return _REPO_ROOT / "assets" / "crab_simple.usda"
