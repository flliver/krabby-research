"""STO-SCN-145 — content-identity + backwards-compat for the cull-mesh@2 `primitives` tunable.

Mirrors the STO-SCN-136/137 backwards-compat proof: the new tunable flows into identity, default
injection holds ({} == explicit-null), distinct primitives -> distinct nodes, and the @2 namespace
does NOT collide with / re-key the prior @1 nodes.

Run: uv run --quiet --python 3.11 --with pytest python3 -m pytest real2sim/tests/test_cull_primitives_identity.py -q
(v4core is pure stdlib — no numpy needed.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import v4core as v4  # noqa: E402

CULL = v4.tasks()["cull-mesh"]
ALGO = CULL["algo"]


def tid(overrides):
    return v4.identity_hash({"mesh": "VARIANT"}, v4.hashable_settings(CULL, overrides), ALGO)


def test_algo_bumped_to_v2():
    assert ALGO == "cull-mesh@2"


def test_primitives_default_null_injected():
    s = v4.hashable_settings(CULL, {})
    assert "primitives" in s and s["primitives"] is None


def test_default_equality_empty_equals_explicit_null():
    assert tid({}) == tid({"primitives": None})


def test_primitives_change_identity():
    prims = {"primitives": [{"type": "sphere", "op": "keep", "center": [0, 0, 0], "radius": 2}]}
    assert tid({}) != tid(prims)


def test_distinct_primitives_distinct_identity():
    a = {"primitives": [{"type": "sphere", "op": "keep", "center": [0, 0, 0], "radius": 2}]}
    b = {"primitives": [{"type": "sphere", "op": "keep", "center": [0, 0, 0], "radius": 3}]}
    assert tid(a) != tid(b)


def test_identity_stable_on_repeat():
    a = {"primitives": [{"type": "box", "op": "keep", "center": [0, 0, 0], "half": [1, 1, 1]}]}
    assert tid(a) == tid(a)


def test_v2_does_not_collide_with_v1_namespace():
    # a prior @1-style cull (no primitives key) hashed under @1 must NOT equal the @2 default node,
    # so existing @1 (STO-137 cambox / STO-136) nodes are preserved untouched, not re-keyed.
    s_v1 = {k: v for k, v in v4.hashable_settings(CULL, {}).items() if k != "primitives"}
    id_v1 = v4.identity_hash({"mesh": "VARIANT"}, s_v1, "cull-mesh@1")
    assert id_v1 != tid({})
