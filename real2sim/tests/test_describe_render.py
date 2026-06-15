#!/usr/bin/env python3
"""STO-SCN-106 — render-description synthesizer tests.

`describe_render(manifest)` turns a render's manifest (v4 algo+settings, or the legacy
transform chain) into an ultra-succinct narrative of how it was built. Telegraphic,
dot-joined, distinguishes pipelines, never raises.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rate_renders"))
from server import describe_render  # noqa: E402


def test_v4_da3_posed_voxel():
    m = {"variant_name": "da3-XYZ [tetra+conditioned]", "pipeline": "da3",
         "transforms": {"da3@1": {"parameters": {"sfm": "posed", "selector": "voxel",
                                                 "n": 24, "process_res": 504,
                                                 "dense_regul": "default"}}},
         "mesh": {"verts": 778127, "faces": 1200000}, "notes": ""}
    d = describe_render(m)
    assert d.startswith("da3@1")
    assert "posed" in d and "voxel-select" in d and "24v" in d
    assert "tetra+conditioned" in d           # mesh method from the label
    assert "1.2M tris" in d
    assert "default" not in d                  # defaults are dropped


def test_v4_matcha_distinct_and_flags():
    m = {"variant_name": "matcha-ABC [tsdf]", "pipeline": "matcha",
         "transforms": {"matcha@1": {"parameters": {"sfm": "posed", "dense_regul": "strong"}}},
         "mesh": {"verts": 900000, "faces": 1800000}, "notes": "NOT DELIVERABLE: CC-BY-NC"}
    d = describe_render(m)
    assert d.startswith("matcha@1")            # distinguishes from da3
    assert "dense-strong" in d
    assert "tsdf" in d and "1.8M tris" in d
    assert "non-deliverable" in d              # surfaced flag


def test_misaligned_flag():
    m = {"variant_name": "da3-Q [tsdf]", "pipeline": "da3",
         "transforms": {"da3@1": {"parameters": {"sfm": "posed"}}},
         "mesh": {}, "notes": "MIS-ALIGNED: flagged (ICP fitness 0.3)"}
    assert "mis-aligned" in describe_render(m)


def test_legacy_transform_chain():
    m = {"variant_name": "matcha--12-strong", "pipeline": "matcha",
         "transforms": {
             "transform-00-sfm": {"kind": "solve-cameras", "parameters": {}},
             "transform-01-train": {"kind": "represent-via-matcha",
                                    "parameters": {"dense_regul": "strong"}},
             "transform-02-mesh": {"kind": "meshify-via-tsdf", "parameters": {}}},
         "notes": ""}
    d = describe_render(m)
    assert d.startswith("matcha")
    assert "solve-cameras" in d and "represent-via-matcha" in d and "meshify-via-tsdf" in d
    assert "dense-strong" in d


def test_degraded_never_raises():
    assert describe_render({"variant_name": "weird"}) == "weird"
    assert describe_render({}) == "(unknown)"
    # malformed transforms must not crash
    assert describe_render({"variant_name": "x", "transforms": {"da3@1": None}}) == "da3@1"


def test_dedup_keeps_order():
    m = {"variant_name": "m [tsdf]", "pipeline": "matcha",
         "transforms": {"matcha@0": {"parameters": {"dense_regul": "strong",
                                                    "alignment_config": "strong"}}}, "mesh": {}}
    d = describe_render(m)
    # dense-strong and align-strong are distinct tokens (not collapsed to one "strong")
    assert "dense-strong" in d and "align-strong" in d


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
