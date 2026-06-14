"""STO-SCN-093 — solver dispatch (solve_plan). Pure dict logic; runs everywhere."""
import importlib.util
from pathlib import Path

_MOD = Path(__file__).resolve().parents[1] / "solve_plan.py"
_spec = importlib.util.spec_from_file_location("solve_plan", _MOD)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

FISHEYE = {"mode": "fisheye", "colmap_compatible": True, "dewarp_dead_end": False,
           "calibration": {"model": "OPENCV_FISHEYE", "rms_reproj_px": 0.86}}
FISHEYE_NOCAL = {"mode": "fisheye", "colmap_compatible": True, "dewarp_dead_end": False}
DEWARPED = {"mode": "dewarped", "colmap_compatible": False, "dewarp_dead_end": True}
PINHOLE = {"mode": "rectilinear", "colmap_compatible": True, "dewarp_dead_end": False}


def test_fisheye_hyperlapse():
    p = sp.plan_solve(FISHEYE, "hyperlapse")
    assert p["solver"] == "fastmap"
    assert p["undistort"] is True
    assert p["precull_target"] == 0            # keep full pool
    assert p["matcher"] == "exhaustive_matcher"
    assert p["solve_camera_model"] == "SIMPLE_PINHOLE"
    assert not p["warnings"]                    # calibration present


def test_fisheye_without_calibration_warns():
    p = sp.plan_solve(FISHEYE_NOCAL, "hyperlapse")
    assert p["undistort"] is True
    assert any("NO calibration" in w for w in p["warnings"])


def test_dewarped_routes_to_da3():
    p = sp.plan_solve(DEWARPED, "video")
    assert p["solver"] == "da3"
    assert p["undistort"] is False
    assert p["precull_target"] == sp.DA3_VIEW_CEILING
    assert p["solve_camera_model"] is None


def test_pinhole_video():
    p = sp.plan_solve(PINHOLE, "video")
    assert p["solver"] == "fastmap"
    assert p["undistort"] is False
    assert p["matcher"] == "sequential_matcher"     # ordered video
    assert p["solve_camera_model"] == "SIMPLE_RADIAL"


def test_fisheye_photos_uses_exhaustive():
    p = sp.plan_solve(FISHEYE, "photos")
    assert p["matcher"] == "exhaustive_matcher"
