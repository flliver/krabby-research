"""STO-SCN-153 — scene_meta(): read-only metadata for the Scenes-tab Metadata view.

scene_meta is a pure function of a scene directory (no server/network state),
so these tests build synthetic trees + assert the derived identity, counts,
datum/scale status, and pipeline-state flags. One test runs against the real
001-patio store when present (calibrated at s=4.45) as a live anchor.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_R2S = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "rr_server", _R2S / "rate_renders" / "server.py")
srv = importlib.util.module_from_spec(_spec)
sys.modules["rr_server"] = srv
_spec.loader.exec_module(srv)


def _tmp() -> Path:
    import tempfile
    return Path(tempfile.mkdtemp())


def _mk(tmp: Path, name: str) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    return d


def test_identity_split():
    d = _mk(_tmp(), "007-sky-house")
    m = srv.scene_meta(d)
    assert m["code"] == "007"
    assert m["name"] == "sky-house"
    assert m["scene"] == "007-sky-house"


def test_empty_scene_counts_zero():
    d = _mk(_tmp(), "001-empty")
    m = srv.scene_meta(d)
    assert m["capture_mode"] == "empty"
    assert m["counts"] == {"images": 0, "subsets": 0, "solves": 0, "render_views": 0}
    assert m["datum"]["calibrated"] is False
    assert m["state"]["ingested"] is False


def test_image_capture_counts():
    d = _mk(_tmp(), "002-patio")
    for h in ("AAAA", "BBBB", "CCCC"):
        (d / "images" / h).mkdir(parents=True)
        (d / "images" / h / "image.jpg").write_bytes(b"x")
    # subsets + ingress must NOT count as canonical images
    (d / "images" / "subsets" / "primary" / "cameras" / "SOLVE1").mkdir(parents=True)
    (d / "images" / "ingress").mkdir(parents=True)
    m = srv.scene_meta(d)
    assert m["capture_mode"] == "images"
    assert m["counts"]["images"] == 3
    assert m["counts"]["subsets"] == 1
    assert m["counts"]["solves"] == 1
    assert m["state"]["ingested"] is True
    assert m["state"]["solved"] is True


def test_video_capture_mode():
    d = _mk(_tmp(), "003-firepit")
    (d / "videos" / "capture").mkdir(parents=True)
    (d / "videos" / "capture" / "video.mp4").write_bytes(b"x")
    m = srv.scene_meta(d)
    assert m["capture_mode"] == "video"
    assert m["state"]["ingested"] is True


def test_datum_calibrated():
    d = _mk(_tmp(), "004-meadow")
    cam = d / "images" / "subsets" / "S1" / "cameras" / "C1"
    cam.mkdir(parents=True)
    (cam / "datum.json").write_text(json.dumps({
        "scale_m_per_unit": 4.45,
        "provenance": {"method": "two anchors", "status": "ok",
                       "scene_extent_m": 12.3},
    }))
    m = srv.scene_meta(d)
    assert m["datum"]["calibrated"] is True
    assert m["datum"]["scale_m_per_unit"] == 4.45
    assert m["datum"]["method"] == "two anchors"
    assert m["state"]["calibrated"] is True


def test_datum_unreadable_is_uncalibrated():
    d = _mk(_tmp(), "005-broken")
    cam = d / "images" / "subsets" / "S1" / "cameras" / "C1"
    cam.mkdir(parents=True)
    (cam / "datum.json").write_text("{ not json")
    m = srv.scene_meta(d)
    assert m["datum"]["calibrated"] is False


def test_meshed_flag():
    d = _mk(_tmp(), "006-kubota")
    p = d / "represent" / "x" / "id" / "meshify" / "m" / "mid"
    p.mkdir(parents=True)
    (p / "mesh.ply").write_bytes(b"ply")
    m = srv.scene_meta(d)
    assert m["state"]["meshed"] is True


@pytest.mark.skipif(not Path("/var/krabby/scenes/001-patio").is_dir(),
                    reason="real store not present")
def test_real_001_patio():
    m = srv.scene_meta(Path("/var/krabby/scenes/001-patio"))
    assert m["code"] == "001"
    assert m["name"] == "patio"
    assert m["counts"]["images"] > 100
    assert m["datum"]["calibrated"] is True
    assert m["datum"]["scale_m_per_unit"] == 4.45
