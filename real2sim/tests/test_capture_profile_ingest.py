"""STO-SCN-091 — ingest wiring (B): graph node + store-write primitives.

Covers the parts of cmd_ingest's capture-profile step that are pure/testable
without a GPU host: the task def + graph topology, and the emit primitives
(hashable_settings / identity_hash / write_metadata) producing a
capture-profile artifact with the resolved camera model. Full end-to-end ingest
is operator-verified on the next real scene ingest (T-020).
"""
import importlib.util
import json
from pathlib import Path

_R2S = Path(__file__).resolve().parents[1]


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _R2S / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cp = _load("capture_profile", "capture_profile.py")
v4 = _load("v4core", "v4core.py")


def test_task_def_loads():
    t = v4.tasks()["resolve-capture-profile"]
    assert t["algo"] == "capture-profile@0"
    assert t["placement"] == "images/capture-profile/{identity}"


def test_graph_has_capture_profile_between_pool_and_solve():
    g = v4.graphs()["ingest-scene"]
    ids = [n["id"] for n in g["nodes"]]
    assert "capture-profile" in ids
    # topo (same algorithm as v4core.plan)
    indeg = {n["id"]: 0 for n in g["nodes"]}
    adj = {n["id"]: [] for n in g["nodes"]}
    for a, b in g["edges"]:
        adj[a].append(b)
        indeg[b] += 1
    order, q = [], [i for i, d in indeg.items() if d == 0]
    while q:
        i = q.pop(0)
        order.append(i)
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    assert order.index("pool") < order.index("capture-profile") < order.index("solve")


def test_declaration_settings_are_hashable_and_deterministic():
    tdef = v4.tasks()["resolve-capture-profile"]
    decl = {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye"}
    s = v4.hashable_settings(tdef, decl)
    assert s == decl  # all three enter the hash (none are pins)
    a = v4.identity_hash({}, s, "capture-profile@0")
    b = v4.identity_hash({}, s, "capture-profile@0")
    assert a == b and a  # deterministic, non-empty


def test_emit_writes_capture_profile_artifact(tmp_path):
    prof = cp.resolve("DJI", "DJI Action 3", "fisheye")
    tdef = v4.tasks()["resolve-capture-profile"]
    s = v4.hashable_settings(tdef, {"make": "DJI", "model": "DJI Action 3", "mode": "fisheye"})
    cpid = v4.identity_hash({}, s, "capture-profile@0")
    cpdir = tmp_path / "images" / "capture-profile" / cpid
    cpdir.mkdir(parents=True)
    (cpdir / "capture-profile.json").write_text(json.dumps(prof, indent=2))
    v4.write_metadata(
        cpdir, task="resolve-capture-profile", algo="capture-profile@0",
        identity=cpid, resolved_inputs={}, settings=s, mechanism="job",
        extra={"camera_model": prof["colmap_camera_model"],
               "colmap_compatible": prof["colmap_compatible"],
               "dewarp_dead_end": prof["dewarp_dead_end"]},
    )
    md = json.loads((cpdir / "metadata.json").read_text())
    assert md["task"] == "resolve-capture-profile"
    assert md["camera_model"] == "SIMPLE_RADIAL_FISHEYE"
    assert md["colmap_compatible"] is True
    assert md["dewarp_dead_end"] is False
    assert json.loads((cpdir / "capture-profile.json").read_text())["mode"] == "fisheye"
