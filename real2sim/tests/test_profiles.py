#!/usr/bin/env python3
"""STO-SCN-108 — studio profiles (server-side, passwordless rater identity).

`_read_profiles` = explicitly-added profiles (profiles.json) ∪ raters seen in scores.jsonl
across scenes, deduped + case-insensitively sorted, `__diag__` excluded. `_add_profile`
appends + dedups + persists. Tested against a temp store (monkeypatched SCENES_ROOT).
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rate_renders"))
import server  # noqa: E402


def _handler():
    return server.Handler.__new__(server.Handler)


def _store(tmp):
    """Lay down a temp store + point the server at it. Returns the root."""
    root = Path(tmp)
    server.SCENES_ROOT = root
    return root


def _scene_scores(root, scene, raters):
    d = root / scene
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "scores.jsonl", "w") as f:
        for i, r in enumerate(raters):
            f.write(json.dumps({"schema": 4, "at": "x", "slot": "01",
                                "rank": 1, "rater": r, "ts": f"t{i}"}) + "\n")


def test_read_union_of_scores_and_file_sorted_deduped():
    with tempfile.TemporaryDirectory() as tmp:
        root = _store(tmp)
        _scene_scores(root, "001", ["Jeremy", "bob"])
        _scene_scores(root, "002", ["bob", "Jeremy"])           # dup across scenes
        (root / "profiles.json").write_text(json.dumps(["Carol", "bob"]))
        h = _handler()
        assert h._read_profiles() == ["bob", "Carol", "Jeremy"]  # case-insensitive sort, deduped


def test_diag_rater_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        root = _store(tmp)
        _scene_scores(root, "001", ["Jeremy", "__diag__"])
        profs = _handler()._read_profiles()
        assert "__diag__" not in profs
        assert profs == ["Jeremy"]


def test_add_persists_and_dedups():
    with tempfile.TemporaryDirectory() as tmp:
        root = _store(tmp)
        _scene_scores(root, "001", ["Jeremy"])
        h = _handler()
        out = h._add_profile("Alice")
        assert out == ["Alice", "Jeremy"]
        # persisted to profiles.json
        assert json.loads((root / "profiles.json").read_text()) == ["Alice"]
        # dedup: adding again is a no-op on the file
        h._add_profile("Alice")
        assert json.loads((root / "profiles.json").read_text()) == ["Alice"]
        # adding a rater that already exists via scores doesn't duplicate the read
        assert h._add_profile("Jeremy").count("Jeremy") == 1


def test_add_empty_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        root = _store(tmp)
        _scene_scores(root, "001", ["Jeremy"])
        h = _handler()
        assert h._add_profile("  ") == ["Jeremy"]
        assert not (root / "profiles.json").exists()           # nothing written


def test_survives_restart():
    """A fresh handler (new 'process') sees the persisted profile."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _store(tmp)
        _handler()._add_profile("Dave")
        server.SCENES_ROOT = root                              # simulate re-read
        assert "Dave" in _handler()._read_profiles()


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
