"""v4exec capture-modality auto-heal — a MISSING `modality` in capture.json is
inferred from STORE FACTS instead of hard-failing the solve (operator hit this
on 003-firepit: make/model/mode present, modality absent).

`_infer_modality` / `_read_capture_decl` are pulled out of v4exec.py by AST so
the test does NOT import v4exec (importing it runs its argparse main()). The
functions only use json + Path, so this is faithful.
"""
import ast
import json
import sys
import tempfile
from pathlib import Path

import pytest

_R2S = Path(__file__).resolve().parents[1]


def _load_funcs(*names):
    tree = ast.parse((_R2S / "v4exec.py").read_text())
    ns = {"json": json, "sys": sys, "Path": Path}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), "v4exec.py", "exec"), ns)
    return [ns[n] for n in names]


infer, read_decl = _load_funcs("_infer_modality", "_read_capture_decl")


def _scene(name):
    d = Path(tempfile.mkdtemp()) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _img(scene, h, original_name):
    d = scene / "images" / h
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"original_name": original_name}))


def test_video_file_infers_video():
    s = _scene("003-firepit")
    cap = s / "videos" / "capture"
    cap.mkdir(parents=True)
    (cap / "video.mp4").write_bytes(b"x")
    mod, why = infer(s)
    assert mod == "video" and "video.mp4" in why


def test_extracted_frames_infer_video():
    s = _scene("011-frames")
    for i in range(4):
        _img(s, f"H{i}", f"frame_{i:02d}.jpg")
    assert infer(s)[0] == "video"


def test_discrete_photos_infer_photos():
    s = _scene("010-photos")
    for i, nm in enumerate(["DJI_0001.jpg", "DJI_0002.jpg", "IMG_3.jpg"]):
        _img(s, f"H{i}", nm)
    assert infer(s)[0] == "photos"


def test_nothing_to_infer_returns_none():
    s = _scene("012-empty")
    (s / "images").mkdir(parents=True)
    assert infer(s)[0] is None          # fails loud later, never guesses


def test_read_decl_autoheals_missing_modality():
    s = _scene("003-firepit")
    (s / "videos" / "capture").mkdir(parents=True)
    (s / "videos" / "capture" / "video.mp4").write_bytes(b"x")
    (s / "capture.json").write_text(json.dumps(
        {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye"}))   # no modality
    make, model, mode, modality = read_decl(s)
    assert (make, model, mode) == ("DJI", "DJI Action 3", "fisheye")
    assert modality == "video"          # auto-healed from the video file


def test_declared_modality_wins_over_inference():
    s = _scene("013-hyper")
    (s / "videos" / "capture").mkdir(parents=True)
    (s / "videos" / "capture" / "video.mp4").write_bytes(b"x")     # would infer 'video'
    (s / "capture.json").write_text(json.dumps(
        {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye", "modality": "hyperlapse"}))
    assert read_decl(s)[3] == "hyperlapse"   # declaration is authoritative


def test_missing_mode_still_fails():
    s = _scene("014-nomode")
    (s / "capture.json").write_text(json.dumps({"make": "DJI", "model": "X"}))
    with pytest.raises(SystemExit):
        read_decl(s)
