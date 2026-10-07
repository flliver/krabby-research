"""Tests for krabby pair-pro packaging."""
from __future__ import annotations

from importlib import resources
from pathlib import Path


def test_pair_pro_script_is_package_data():
    ref = resources.files("krabby").joinpath("scripts/pair_pro_controller.sh")
    with resources.as_file(ref) as path:
        assert Path(path).is_file()
        text = Path(path).read_text(encoding="utf-8")
        assert "Pro Controller" in text
        assert "bluetoothctl" in text
        # F5: warn text must blame incomplete bond / Sync, not hid_nintendo first;
        # third-party pads get USB / bluetoothctl guidance.
        assert "Found too quickly" in text
        assert "Continuing this attempt anyway" in text
        assert "Paired: no" in text
        assert "Do not chase hid_nintendo" in text
        assert "Home ≠ Sync" in text or "Home wake" in text
        assert "Third-party" in text
        assert "USB" in text
        assert "CONNECT_PRO_CONTROLLER.md" in text


def test_pair_pro_cmd_importable():
    from krabby.pair_pro import cmd_pair_pro

    assert callable(cmd_pair_pro)
