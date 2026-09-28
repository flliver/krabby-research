"""Shared paths for crab_hex USD tools and Isaac diagnostics."""

from __future__ import annotations

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
# scripts → crab_hex_forward_task → parkour_tasks → parkour_tasks → parkour → krabby-research
_REPO_ROOT = _SCRIPT_DIR.parents[4]
_PARKOUR_ROOT = _REPO_ROOT / "parkour"


def repo_root() -> Path:
    return _REPO_ROOT


def parkour_root() -> Path:
    return _PARKOUR_ROOT


def parkour_scripts_dir() -> Path:
    return _PARKOUR_ROOT / "scripts"


def crab_usd() -> Path:
    """The main training plant (A15+B geometry, plant of record since 2026-09-09)."""
    return _REPO_ROOT / "assets" / "crab.usda"


def legacy_golden_usd() -> Path:
    """The legacy golden plant (2026-08-20 measured-hardware build): heads trained before the a15b lineage."""
    return _REPO_ROOT / "assets" / "variants" / "crab_simple__splay00_axis5p5in.usda"

