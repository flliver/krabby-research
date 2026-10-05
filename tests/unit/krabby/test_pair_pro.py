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


def test_pair_pro_cmd_importable():
    from krabby.pair_pro import cmd_pair_pro

    assert callable(cmd_pair_pro)
