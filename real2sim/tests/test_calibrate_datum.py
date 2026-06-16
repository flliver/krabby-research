"""STO-SCN-016 — tests for the datum-scale recompute from a MEASURE export.

Run: uv run --quiet --python 3.11 --with numpy --with pytest python3 -m pytest real2sim/tests/test_calibrate_datum.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import calibrate_datum as cd  # noqa: E402


def _pick(cam_center, point):
    o = np.asarray(cam_center, float); p = np.asarray(point, float)
    d = p - o; d = d / np.linalg.norm(d)
    return {"o": o.tolist(), "d": d.tolist()}


def test_recompute_roundtrip_recovers_scale_and_flags_da3():
    cL, cR = [-1, 0, 0], [1, 0, 0]
    Pa, Pb = [0, 0, 4], [1.5, 0, 4]                 # 1.5 solve-units apart
    export = {"distances": [{"D": 3.0,              # really 3.0 m -> s = 2.0 m/unit
                             "picks1": [_pick(cL, Pa), _pick(cR, Pa)],
                             "picks2": [_pick(cL, Pb), _pick(cR, Pb)]}],
              "da3_scale_factors": [3.346, 3.481, 11.220]}
    rec = cd.recompute(export)
    assert abs(rec["s_meters_per_solve_unit"] - 2.0) < 1e-6
    assert rec["n_distances"] == 1
    assert rec["weak_triangulation"] is False
    assert rec["da3_scouts_disagree"] is True       # 11.22 spread > 1.5
    assert rec["da3_gate"]["passed"] is None         # no converted prior -> gate inert


def test_recompute_falls_back_to_stored_points():
    export = {"distances": [{"D": 1.0, "P1": [0, 0, 0], "P2": [0, 0, 2],
                             "parallax": 30, "gaps": [0, 0]}]}
    rec = cd.recompute(export)
    assert abs(rec["s_meters_per_solve_unit"] - 0.5) < 1e-9


def test_two_distances_log_median_and_spread():
    export = {"distances": [
        {"D": 1.0, "P1": [0, 0, 0], "P2": [0, 0, 2], "parallax": 30, "gaps": [0, 0]},   # s=0.5
        {"D": 1.2, "P1": [0, 0, 0], "P2": [0, 2, 0], "parallax": 30, "gaps": [0, 0]}]}  # s=0.6
    rec = cd.recompute(export)
    assert rec["n_distances"] == 2
    assert 0.5 < rec["s_meters_per_solve_unit"] < 0.6
    assert rec["spread"] == 1.2


def test_weak_triangulation_flagged():
    export = {"distances": [{"D": 1.0, "P1": [0, 0, 0], "P2": [0, 0, 2],
                             "parallax": 1.0, "gaps": [0.4, 0.5]}]}   # par < 2 deg
    rec = cd.recompute(export)
    assert rec["weak_triangulation"] is True


def test_gate_applies_when_converted_prior_supplied():
    export = {"distances": [{"D": 1.0, "P1": [0, 0, 0], "P2": [0, 0, 2],
                             "parallax": 30, "gaps": [0, 0]}]}        # s = 0.5
    rec = cd.recompute(export, s_monocular=[0.49, 0.51])
    assert rec["da3_gate"]["passed"] is True
    rec2 = cd.recompute(export, s_monocular=[1.6])                   # 3.2x off
    assert rec2["da3_gate"]["passed"] is False


def test_apply_to_gauge_writes_additive_sidecar(tmp_path):
    out = cd.apply_to_gauge(str(tmp_path), 0.4167,
                            datum_frame={"scale": 0.4167}, provenance={"D": 1.0})
    rec = json.loads((tmp_path / "datum.json").read_text())
    assert out.endswith("datum.json")
    assert rec["scale_m_per_unit"] == 0.4167
    assert rec["datum_frame"]["scale"] == 0.4167
    assert rec["provenance"]["D"] == 1.0


def test_apply_to_gauge_refuses_clobber(tmp_path):
    cd.apply_to_gauge(str(tmp_path), 0.4)
    with pytest.raises(FileExistsError):
        cd.apply_to_gauge(str(tmp_path), 0.5)              # preserve existing (T-018)
    cd.apply_to_gauge(str(tmp_path), 0.5, force=True)      # explicit override
    assert json.loads((tmp_path / "datum.json").read_text())["scale_m_per_unit"] == 0.5
