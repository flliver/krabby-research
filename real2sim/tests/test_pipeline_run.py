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
    assert [p["key"] for p in pl] == ["ingest", "precull", "solve", "covis", "scout", "mesh"]
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
    assert [p["key"] for p in final["phases"]] == ["ingest", "precull", "solve", "covis", "scout", "mesh"]
    # no video on this synthetic scene -> ingest skips; the rest are planned
    assert final["phases"][0]["status"] == "skipped"
    assert all(p["status"] == "planned" for p in final["phases"][1:])
    assert len(recs) >= 6   # initial + per-phase transitions


def test_deduce_fps_band():
    assert pr.deduce_fps(500)[0] == 1.0           # 500/500=1.0
    assert pr.deduce_fps(60)[0] == 4.0            # clamp high
    assert pr.deduce_fps(2000)[0] == 1.0          # clamp low
    fps, exp = pr.deduce_fps(331)
    assert 1.0 <= fps <= 4.0 and exp and exp > 300


def test_ingest_plan_extract_vs_skip(tmp_path):
    import shutil
    sd = tmp_path / "003-x"
    (sd / "videos" / "capture").mkdir(parents=True)
    # a 1s test clip so ffprobe has a real duration
    vid = sd / "videos" / "capture" / "video.mp4"
    if shutil.which("ffmpeg"):
        import subprocess
        subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10",
                        "-y", str(vid)], check=True, capture_output=True)
    else:
        vid.write_bytes(b"x")
    (sd / "images").mkdir(parents=True)
    ip = pr.ingest_plan(sd)
    assert ip["action"] == "extract"              # video + 0 images -> extract
    # with images present -> skip
    (sd / "images" / "HASH").mkdir()
    assert pr.ingest_plan(sd)["action"] == "skip"


def test_ingest_plan_skip_no_video(tmp_path):
    sd = tmp_path / "010-photos"
    (sd / "images" / "H0").mkdir(parents=True)
    assert pr.ingest_plan(sd)["action"] == "skip"  # no video


def test_phase0_is_ingest():
    assert pr.PHASES[0]["key"] == "ingest" and pr.PHASES[0].get("local")


def test_resize_target_mode_aware(tmp_path):
    import json
    # declared fisheye -> native (None) — undistort needs full res
    fe = tmp_path / "003"; fe.mkdir()
    (fe / "capture.json").write_text(json.dumps({"make": "DJI", "model": "DJI Action 3", "mode": "fisheye"}))
    assert pr.resize_target(fe) is None
    # rectilinear -> downscale
    rc = tmp_path / "007"; rc.mkdir()
    (rc / "capture.json").write_text(json.dumps({"mode": "rectilinear"}))
    assert pr.resize_target(rc) == pr.INGEST_MAX_LONG_EDGE
    # no capture.json -> downscale (safe; an undeclared fisheye can't pass the solve gate anyway)
    bare = tmp_path / "x"; bare.mkdir()
    assert pr.resize_target(bare) == pr.INGEST_MAX_LONG_EDGE
