"""v4exec capture-declaration auto-heal — capture.json is OPTIONAL for
EXIF-identifiable, single-mode registered cameras (STO-SCN-091/093):

  - make/model  ← EXIF of a canonical image when not declared
  - mode        ← the registry's SOLE mode for that camera (multi-mode cameras
                  like DJI fisheye/dewarped still require an explicit mode)
  - modality    ← store facts (`_infer_modality`)

Operators hit this on 007-kubota: an iPhone capture with NO capture.json at all.

`_infer_modality` / `_read_capture_decl` are pulled out of v4exec.py by AST so
the test does not import v4exec (importing it runs its argparse main()). The
real `capture_profile` is on sys.path so the EXIF + registry paths are faithful.
"""
import ast
import json
import sys
import tempfile
from pathlib import Path

import pytest

_R2S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R2S))    # so the extracted fn's `import capture_profile` resolves


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


# ---- modality inference (store facts) --------------------------------------

def test_video_file_infers_video():
    s = _scene("003-firepit")
    (s / "videos" / "capture").mkdir(parents=True)
    (s / "videos" / "capture" / "video.mp4").write_bytes(b"x")
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


# ---- full capture-decl resolution -----------------------------------------

def test_modality_autoheals_when_make_model_mode_declared():
    s = _scene("003-firepit")
    (s / "videos" / "capture").mkdir(parents=True)
    (s / "videos" / "capture" / "video.mp4").write_bytes(b"x")
    (s / "capture.json").write_text(json.dumps(
        {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye"}))   # no modality
    assert read_decl(s) == ("DJI", "DJI Action 3", "fisheye", "video")


def test_mode_autoheals_from_sole_registry_mode():
    # iPhone 15 Pro has exactly one registry mode (rectilinear) -> no declaration needed.
    s = _scene("020-iphone")
    _img(s, "H0", "IMG_0001.jpg")
    (s / "capture.json").write_text(json.dumps(
        {"make": "Apple", "model": "iPhone 15 Pro"}))   # no mode, no modality
    make, model, mode, modality = read_decl(s)
    assert (make, model) == ("Apple", "iPhone 15 Pro")
    assert mode == "rectilinear"        # sole registry mode
    assert modality == "photos"         # discrete photo names


def test_declared_modality_wins_over_inference():
    s = _scene("013-hyper")
    (s / "videos" / "capture").mkdir(parents=True)
    (s / "videos" / "capture" / "video.mp4").write_bytes(b"x")
    (s / "capture.json").write_text(json.dumps(
        {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye", "modality": "hyperlapse"}))
    assert read_decl(s)[3] == "hyperlapse"


def test_multimode_camera_requires_mode():
    # DJI Action 3 has fisheye AND dewarped -> mode is NOT auto-derivable.
    s = _scene("014-dji-nomode")
    _img(s, "H0", "frame_01.jpg")
    (s / "capture.json").write_text(json.dumps({"make": "DJI", "model": "DJI Action 3"}))
    with pytest.raises(SystemExit):
        read_decl(s)


def test_unknown_camera_fails_loud():
    s = _scene("015-unknown")
    _img(s, "H0", "frame_01.jpg")
    (s / "capture.json").write_text(json.dumps({"make": "Acme", "model": "Z9"}))
    with pytest.raises(SystemExit):     # no profile -> add one, never guess
        read_decl(s)


@pytest.mark.skipif(not Path("/var/krabby/scenes/007-kubota").is_dir(),
                    reason="real store not present")
def test_real_007_kubota_full_autoheal():
    # No capture.json on disk — everything from facts (iPhone EXIF + registry + store).
    make, model, mode, modality = read_decl(Path("/var/krabby/scenes/007-kubota"))
    assert make == "Apple" and model == "iPhone 15 Pro"
    assert mode == "rectilinear"
    assert modality in ("video", "photos")
