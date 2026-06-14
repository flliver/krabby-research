"""STO-SCN-093 — undistort_fisheye calibration-resolve + intrinsics paths.

The cv2 remap is exercised on real frames fleet-side; here we cover the pure,
cv2-free paths: pulling the fisheye calibration out of a capture profile (and
failing loud when it's absent / wrong model) and the pinhole intrinsics dict.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_MOD = Path(__file__).resolve().parents[1] / "undistort_fisheye.py"
_spec = importlib.util.spec_from_file_location("undistort_fisheye", _MOD)
uf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uf)


def _profiles(tmp_path, with_calib=True, model="OPENCV_FISHEYE"):
    entry = {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye",
             "colmap_camera_model": "SIMPLE_RADIAL_FISHEYE"}
    if with_calib:
        entry["calibration"] = {"model": model, "image_size": [3840, 2160],
                                "K": [[2065.6, 0, 1946.2], [0, 2061.7, 1097.6], [0, 0, 1]],
                                "D": [0.28, -0.09, 0.15, -0.14], "rms_reproj_px": 0.86}
    p = tmp_path / "capture_profiles.json"
    p.write_text(json.dumps({"schema": 1, "profiles": [entry]}, indent=2))
    return p


def test_load_calibration_returns_block(tmp_path):
    p = _profiles(tmp_path)
    calib = uf.load_fisheye_calibration("DJI", "DJI Action 3", "fisheye", p)
    assert calib["model"] == "OPENCV_FISHEYE"
    assert calib["image_size"] == [3840, 2160]
    assert calib["rms_reproj_px"] == 0.86


def test_load_calibration_absent_fails_loud(tmp_path):
    p = _profiles(tmp_path, with_calib=False)
    with pytest.raises(ValueError):
        uf.load_fisheye_calibration("DJI", "DJI Action 3", "fisheye", p)


def test_load_calibration_wrong_model_fails_loud(tmp_path):
    p = _profiles(tmp_path, model="STANDARD_BROWN")
    with pytest.raises(ValueError):
        uf.load_fisheye_calibration("DJI", "DJI Action 3", "fisheye", p)


def test_intrinsics_dict_shape():
    Kp = [[2012.6, 0.0, 1952.0], [0.0, 2008.7, 1098.9], [0.0, 0.0, 1.0]]
    intr = uf.intrinsics_dict(Kp, (3840, 2160), {"rms_reproj_px": 0.86}, balance=0.0)
    assert intr["model"] == "PINHOLE"
    assert intr["fx"] == 2012.6 and intr["cy"] == 1098.9
    assert intr["params"] == [2012.6, 2008.7, 1952.0, 1098.9]
    assert intr["width"] == 3840 and intr["height"] == 2160
