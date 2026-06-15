"""STO-SCN-105 — scout→solve gauge registration (scale_factor).

The DA3 scout gaussian lives in DA3's normalized frame (solve / scale_factor
about the origin). These tests pin the read path + the point-cloud cross-check
that recovers the same scale, so the verify surface registers the splat with
no manual reconciliation.
"""
import importlib.util
import json
import sys
from pathlib import Path

_R2S = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("scout_gauge", _R2S / "scout_gauge.py")
sg = importlib.util.module_from_spec(_spec)
sys.modules["scout_gauge"] = sg
_spec.loader.exec_module(sg)


def _scout(tmp_path, payload):
    (tmp_path / "scout_gauge.json").write_text(json.dumps(payload))
    return tmp_path


def test_read_scale_factor_present(tmp_path):
    d = _scout(tmp_path, {"scale_factor": 2.1, "is_metric": True})
    assert sg.read_scale_factor(d) == 2.1


def test_read_scale_factor_absent_returns_none(tmp_path):
    assert sg.read_scale_factor(tmp_path) is None            # pre-105 scout: no file


def test_read_scale_factor_nonmetric_none(tmp_path):
    d = _scout(tmp_path, {"scale_factor": None, "is_metric": False})
    assert sg.read_scale_factor(d) is None


def test_splat_transform_registered(tmp_path):
    d = _scout(tmp_path, {"scale_factor": 2.1, "is_metric": True})
    tf = sg.splat_transform(d)
    assert tf == {"scale": 2.1, "registered": True, "source": "scale_factor"}


def test_splat_transform_unregistered_is_identity(tmp_path):
    tf = sg.splat_transform(tmp_path)
    assert tf["scale"] == 1.0 and tf["registered"] is False  # safe no-op + warnable


def test_estimate_scale_matches_measured():
    # 001-patio: gaussian core std vs SfM points std → ~2.1 (the scale_factor)
    est = sg.estimate_scale_from_points([1.54, 0.93, 1.48], [0.75, 0.53, 0.57])
    assert 2.0 < est < 2.25


def test_estimate_scale_zero_spread_raises():
    try:
        sg.estimate_scale_from_points([1, 1, 1], [0, 0, 0])
    except ValueError:
        return
    raise AssertionError("expected ValueError on zero solve spread")
