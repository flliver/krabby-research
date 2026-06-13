"""STO-SCN-092 — cmd_precull store wiring (option i: opt-in, primary untouched).

Builds a synthetic v4 store, runs cmd_precull against it (STORE monkeypatched),
and asserts a curated subset is written with mechanism=precull and that `primary`
is left alone unless --set-primary. Skips where numpy/PIL aren't present.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

_R2S = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v4exec", _R2S / "v4exec.py")
vex = importlib.util.module_from_spec(_spec)
sys.modules["v4exec"] = vex
_spec.loader.exec_module(vex)


def _build_store(tmp_path):
    store = tmp_path / "scenes"
    sdir = store / "t"
    k = 0
    for seg in range(4):                      # 4 segments x 6 near-dups = 24
        base = np.random.default_rng(seg).integers(0, 256, (48, 48, 3), dtype=np.uint8)
        for _ in range(6):
            x = base.copy(); x[0, 0, 0] = (int(x[0, 0, 0]) + 1) % 256
            d = sdir / "images" / f"hash{k:03d}"
            d.mkdir(parents=True)
            Image.fromarray(x).save(d / "image.png")
            (d / "metadata.json").write_text('{"schema":4}')
            k += 1
    return store, sdir


def _args(**kw):
    base = dict(scene="t", target=300, phash_thresh=8, blur_rel=0.2,
                max_gap=20, dup_window=12, score_edge=480, set_primary=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_cmd_precull_writes_curated_subset_and_leaves_primary(tmp_path, monkeypatch):
    store, sdir = _build_store(tmp_path)
    monkeypatch.setattr(vex.v4, "STORE", store)

    vex.cmd_precull(_args())

    subs = list((sdir / "images" / "subsets").glob("*/subset.json"))
    assert len(subs) == 1
    md = json.loads((subs[0].parent / "metadata.json").read_text())
    assert md["mechanism"] == "precull"
    assert md["source_pool_n"] == 24
    assert 4 <= md["kept_n"] < 24                      # near-dups collapsed
    members = json.loads(subs[0].read_text())["members"]
    assert len(members) == md["kept_n"]
    # opt-in: primary must NOT be set
    assert not (sdir / "images" / "subsets" / "primary").exists()


def test_cmd_precull_noop_and_set_primary(tmp_path, monkeypatch):
    store, sdir = _build_store(tmp_path)
    monkeypatch.setattr(vex.v4, "STORE", store)

    vex.cmd_precull(_args())                            # creates subset
    vex.cmd_precull(_args(set_primary=True))            # NOOP subset, then set primary
    primary = sdir / "images" / "subsets" / "primary"
    assert primary.is_symlink()
    # only one curated subset exists (idempotent identity); exclude the `primary`
    # symlink, which resolves back into the same subset dir.
    real = [p for p in (sdir / "images" / "subsets").glob("*/subset.json")
            if p.parent.name != "primary"]
    assert len(real) == 1
