"""STO-SCN-148 — scene_subsets(): camera-subset listing for the Subsets view.

Pure function of a scene dir. Asserts the `primary` symlink resolves to a flag
(not a duplicate entry), member/solve counts, and the datum flag. One live test
against the real 001-patio store.
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_R2S = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "rr_server", _R2S / "rate_renders" / "server.py")
srv = importlib.util.module_from_spec(_spec)
sys.modules["rr_server"] = srv
_spec.loader.exec_module(srv)


def _scene(name: str) -> Path:
    return Path(tempfile.mkdtemp()) / name


def test_empty():
    d = _scene("001-x"); d.mkdir(parents=True)
    assert srv.scene_subsets(d) == {"scene": "001-x", "subsets": []}


def test_primary_symlink_flagged_not_duplicated():
    d = _scene("002-x")
    real = d / "images" / "subsets" / "ABCD"
    (real / "cameras" / "SOL1").mkdir(parents=True)
    (real / "subset.json").write_text(json.dumps({"members": ["h1", "h2", "h3"]}))
    (real / "metadata.json").write_text(json.dumps({"label": "precull-3", "mechanism": "precull"}))
    os.symlink("ABCD", d / "images" / "subsets" / "primary")
    r = srv.scene_subsets(d)["subsets"]
    assert len(r) == 1                      # the symlink is NOT a second entry
    assert r[0]["id"] == "ABCD"
    assert r[0]["is_primary"] is True
    assert r[0]["member_count"] == 3
    assert r[0]["members"] == ["h1", "h2", "h3"]
    assert r[0]["solves"] == ["SOL1"]
    assert r[0]["label"] == "precull-3"


def test_datum_flag_and_primary_first():
    d = _scene("003-x")
    a = d / "images" / "subsets" / "AAAA" / "cameras" / "S1"
    a.mkdir(parents=True)
    (a / "datum.json").write_text("{}")
    (d / "images" / "subsets" / "AAAA" / "subset.json").write_text(json.dumps({"members": ["x"]}))
    b = d / "images" / "subsets" / "BBBB"
    b.mkdir(parents=True)
    (b / "subset.json").write_text(json.dumps({"members": ["y", "z"]}))
    os.symlink("BBBB", d / "images" / "subsets" / "primary")
    r = srv.scene_subsets(d)["subsets"]
    assert r[0]["id"] == "BBBB" and r[0]["is_primary"] is True     # primary first
    aa = next(s for s in r if s["id"] == "AAAA")
    assert aa["has_datum"] is True


@pytest.mark.skipif(not Path("/var/krabby/scenes/001-patio").is_dir(),
                    reason="real store not present")
def test_real_001_patio():
    r = srv.scene_subsets(Path("/var/krabby/scenes/001-patio"))["subsets"]
    assert len(r) >= 5
    assert any(s["is_primary"] for s in r)
    assert any(s["has_datum"] for s in r)        # 6EHLYO3MF3QU carries the datum
