"""STO-SCN-149 — scene_ingest: new-scene creation + canonicalize.

Pure/stdlib (+ffmpeg) functions: kebab/next_code/create_scene and the
canonicalize path (images and, when ffmpeg is present, video frames). Asserts
content-hash dedup + the images/<hash>/{image,metadata} layout the rest of the
pipeline reads.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_R2S = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("scene_ingest", _R2S / "scene_ingest.py")
si = importlib.util.module_from_spec(_spec)
sys.modules["scene_ingest"] = si
_spec.loader.exec_module(si)


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def test_kebab():
    assert si.kebab("My Cool Scene! 2") == "my-cool-scene-2"
    assert si.kebab("  Trailing--Dashes  ") == "trailing-dashes"
    assert si.kebab("") == "scene"


def test_next_code():
    store = _tmp()
    (store / "001-foo").mkdir()
    (store / "003-bar").mkdir()
    (store / "notscene").mkdir()
    assert si.next_code(store) == "004"          # max + 1, ignores non-NNN dirs
    assert si.next_code(_tmp()) == "001"


def test_create_scene():
    store = _tmp()
    r = si.create_scene(store, "Patio Test")
    assert r["scene"] == "001-patio-test"
    assert r["code"] == "001"
    assert (store / "001-patio-test" / "images").is_dir()
    # a second scene gets the next code
    r2 = si.create_scene(store, "Another")
    assert r2["code"] == "002"


def test_canonicalize_dedup():
    store = _tmp()
    sd = store / "001-x"
    (sd / "images").mkdir(parents=True)
    src = _tmp()
    (src / "a.jpg").write_bytes(b"AAAA")
    (src / "b.jpg").write_bytes(b"BBBB")
    (src / "b2.jpg").write_bytes(b"BBBB")     # dup of b
    (src / "notes.txt").write_bytes(b"x")     # ignored
    res = si.ingest_images(sd, [src], move=False)
    assert res["n"] == 3                       # 3 images processed
    assert len(set(res["hashes"])) == 2        # 2 unique content hashes
    canon = [d for d in (sd / "images").iterdir()
             if d.is_dir() and d.name not in ("subsets", "ingress")]
    assert len(canon) == 2
    md = json.loads((canon[0] / "metadata.json").read_text())
    assert md["mechanism"] == "ingest-ui" and "original_name" in md


def test_progress_callback():
    store = _tmp()
    sd = store / "001-x"
    (sd / "images").mkdir(parents=True)
    src = _tmp()
    for i in range(3):
        (src / f"img{i}.jpg").write_bytes(bytes([i]) * 8)
    seen = []
    si.ingest_images(sd, [src], move=False, progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1] == (3, 3)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_video_ingest():
    store = _tmp()
    sd = store / "001-vid"
    (sd / "images").mkdir(parents=True)
    vid = _tmp() / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
         "-y", str(vid)], check=True, capture_output=True)
    res = si.ingest_video(sd, vid, fps=2.0, move=False)
    assert res["frames_extracted"] >= 3
    assert res["n"] == res["frames_extracted"]
    assert (sd / "videos" / "capture" / "video.mp4").exists()
