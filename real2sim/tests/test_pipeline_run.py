"""STO-SCN-150 — pipeline_run: ingest-scene orchestration (plan + dry-run).

The id-resolvers (newest solve under the primary subset) + the phase plan +
the dry-run sequencer are deterministic and testable without a GPU. The REAL
run (v4exec on a host via ssh+docker) is operator-verified (T-020) and NOT
exercised here.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_R2S = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pipeline_run", _R2S / "pipeline_run.py")
pr = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_run"] = pr
_spec.loader.exec_module(pr)


def _scene_with_solve() -> Path:
    store = Path(tempfile.mkdtemp())
    sc = store / "009-syn"
    (sc / "images" / "subsets" / "ABC" / "cameras" / "SOLVE1").mkdir(parents=True)
    (sc / "images" / "subsets" / "ABC" / "cameras" / "SOLVE1" / "cameras.json").write_text("{}")
    os.symlink("ABC", sc / "images" / "subsets" / "primary")
    return sc


def test_gpu_hosts_default_and_env(monkeypatch):
    monkeypatch.delenv("KRABBY_GPU_HOSTS", raising=False)
    assert pr.gpu_hosts() == ["tbeeprz"]
    monkeypatch.setenv("KRABBY_GPU_HOSTS", "a, b ,c")
    assert pr.gpu_hosts() == ["a", "b", "c"]


def test_resolvers():
    sc = _scene_with_solve()
    assert pr.resolve_primary_subset(sc) == "ABC"
    assert pr.resolve_latest_solve(sc) == "SOLVE1"


def test_plan_threads_solve():
    sc = _scene_with_solve()
    pl = pr.plan("009-syn", "tbeeprz", scene_dir=sc)
    assert [p["key"] for p in pl] == ["precull", "solve", "covis", "scout", "mesh"]
    covis = next(p for p in pl if p["key"] == "covis")
    assert "--solve" in covis["cmd"] and "SOLVE1" in covis["cmd"]
    assert "--host" in covis["cmd"] and "tbeeprz" in covis["cmd"]
    mesh = next(p for p in pl if p["key"] == "mesh")
    assert "reconstruct-da3" in mesh["cmd"] and "--sfm" in mesh["cmd"]
    # precull has no host (CPU)
    precull = next(p for p in pl if p["key"] == "precull")
    assert "--host" not in precull["cmd"] and "--set-primary" in precull["cmd"]


def test_dry_run_sequences_without_executing():
    sc = _scene_with_solve()
    recs = []
    final = pr.run_pipeline(sc, "tbeeprz", dry_run=True, status_cb=lambda r: recs.append(r))
    assert final["status"] == "done"
    assert all(p["status"] == "planned" for p in final["phases"])
    assert [p["key"] for p in final["phases"]] == ["precull", "solve", "covis", "scout", "mesh"]
    assert len(recs) >= 6   # initial + per-phase transitions
