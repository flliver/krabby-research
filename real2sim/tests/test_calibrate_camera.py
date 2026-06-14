"""STO-SCN-102 — calibrate_camera profile-write + board parsing.

The OpenCV calibration math is exercised when the operator runs it on real
checkerboard frames; here we cover the pure, cv2-free paths (board parsing +
writing the calibration into the right capture-profile entry, fail-loud on a
missing profile).
"""
import importlib.util
import json
from pathlib import Path

import pytest

_MOD = Path(__file__).resolve().parents[1] / "calibrate_camera.py"
_spec = importlib.util.spec_from_file_location("calibrate_camera", _MOD)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def test_parse_board():
    assert cc.parse_board("9x6") == (9, 6)
    assert cc.parse_board("10X7") == (10, 7)


def _profiles(tmp_path):
    p = tmp_path / "capture_profiles.json"
    p.write_text(json.dumps({
        "schema": 1,
        "profiles": [
            {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye",
             "colmap_camera_model": "SIMPLE_RADIAL_FISHEYE"},
            {"make": "DJI", "model": "DJI Action 3", "mode": "dewarped",
             "colmap_camera_model": None},
        ],
    }, indent=2))
    return p


def test_write_calibration_sets_matching_entry(tmp_path):
    p = _profiles(tmp_path)
    calib = {"model": "OPENCV_FISHEYE", "image_size": [2704, 1520],
             "K": [[1, 0, 1352], [0, 1, 760], [0, 0, 1]], "D": [0.1, 0.0, 0.0, 0.0],
             "rms_reproj_px": 0.42}
    cc.write_calibration(p, "DJI", "DJI Action 3", "fisheye", calib)
    data = json.loads(p.read_text())
    fish = next(x for x in data["profiles"] if x["mode"] == "fisheye")
    dew = next(x for x in data["profiles"] if x["mode"] == "dewarped")
    assert fish["calibration"]["rms_reproj_px"] == 0.42        # written on fisheye
    assert "calibration" not in dew                            # not on dewarped


def test_write_calibration_case_insensitive(tmp_path):
    p = _profiles(tmp_path)
    cc.write_calibration(p, "dji", "dji action 3", "FISHEYE", {"rms_reproj_px": 1.0})
    data = json.loads(p.read_text())
    assert any(x.get("calibration") for x in data["profiles"])


def test_write_calibration_unknown_profile_fails_loud(tmp_path):
    p = _profiles(tmp_path)
    with pytest.raises(ValueError):
        cc.write_calibration(p, "GoPro", "Hero 12", "fisheye", {"rms_reproj_px": 1.0})
