"""STO-SCN-091 — tests for the capture-profile resolver.

Validates the conclusions that motivated the story (STO-SCN-096 #3, HUG-SCN-004):
- known camera + fisheye  -> SIMPLE_RADIAL_FISHEYE, COLMAP-compatible
- known camera + dewarped -> no COLMAP model, COLMAP-incompatible (dead-end)
- unknown camera          -> fail loud (no guessed default)
- missing mode            -> fail loud (mode is not in EXIF)
"""
import importlib.util
from pathlib import Path

import pytest

# Load the module directly (real2sim is not a package).
_MOD = Path(__file__).resolve().parents[1] / "capture_profile.py"
_spec = importlib.util.spec_from_file_location("capture_profile", _MOD)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def test_registry_loads():
    reg = cp.load_registry()
    assert reg, "registry must contain seeded profiles"
    assert all({"make", "model", "mode"} <= set(p) for p in reg)


def test_dji_fisheye_resolves_to_simple_radial_fisheye():
    prof = cp.resolve("DJI", "DJI Action 3", "fisheye")
    assert prof["colmap_camera_model"] == "SIMPLE_RADIAL_FISHEYE"
    assert prof["colmap_compatible"] is True
    assert prof["dewarp_dead_end"] is False
    assert prof["single_camera"] is True


def test_dji_dewarped_is_colmap_dead_end():
    prof = cp.resolve("DJI", "DJI Action 3", "dewarped")
    assert prof["colmap_camera_model"] is None
    assert prof["colmap_compatible"] is False
    assert prof["dewarp_dead_end"] is True


def test_case_insensitive_match():
    prof = cp.resolve("dji", "dji action 3", "FISHEYE")
    assert prof["colmap_camera_model"] == "SIMPLE_RADIAL_FISHEYE"


def test_unknown_camera_fails_loud():
    with pytest.raises(cp.ProfileError):
        cp.resolve("GoPro", "Hero 12", "fisheye")


def test_missing_mode_fails_loud():
    # mode is NOT derivable from EXIF — refusing to guess is the point.
    with pytest.raises(cp.ProfileError):
        cp.resolve("DJI", "DJI Action 3", None)


def test_resolved_provenance_stub_present():
    prof = cp.resolve("DJI", "DJI Action 3", "fisheye")
    assert prof["resolved"] == {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye"}
