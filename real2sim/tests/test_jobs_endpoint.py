"""STO-SCN-088 — /api/jobs/<scene> feedback channel (source-of-truth slice).

The job.json records (locked #8) are the source of truth; the retained-MQTT
overlay is an optional fast path that must degrade to {} when no broker is
configured. These tests cover the deterministic file-truth half + the
no-broker degradation (the MQTT happy path is exercised against a live
broker in the lib_progress.sh shell test, not here).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_R2S = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "rr_server", _R2S / "rate_renders" / "server.py")
srv = importlib.util.module_from_spec(_spec)
sys.modules["rr_server"] = srv
_spec.loader.exec_module(srv)


def _write_job(scene_dir: Path, name: str, payload: dict) -> None:
    jd = scene_dir / "jobs" / name
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "job.json").write_text(json.dumps(payload))


def test_jobs_files_newest_first_and_tagged(tmp_path):
    sdir = tmp_path / "001-patio"
    _write_job(sdir, "20260614T100000-aaaa",
               {"graph": "render-missing", "status": "done", "outcome": {"rendered": 2}})
    _write_job(sdir, "20260614T110000-bbbb",
               {"graph": "render-missing", "status": "running", "outcome": {"rendered": 1}})
    recs = srv.Handler._jobs_files(sdir)
    assert [r["job"] for r in recs] == [
        "20260614T110000-bbbb", "20260614T100000-aaaa"]   # reverse-sorted = newest first
    assert recs[0]["status"] == "running"


def test_jobs_files_skips_corrupt_record(tmp_path):
    sdir = tmp_path / "s"
    _write_job(sdir, "j-ok", {"graph": "render-missing"})
    bad = sdir / "jobs" / "j-bad"
    bad.mkdir(parents=True)
    (bad / "job.json").write_text("{not json")
    recs = srv.Handler._jobs_files(sdir)
    assert [r["job"] for r in recs] == ["j-ok"]           # corrupt one dropped, no raise


def test_jobs_files_empty_when_no_jobs(tmp_path):
    assert srv.Handler._jobs_files(tmp_path / "nope") == []


def test_jobs_live_empty_without_broker(monkeypatch):
    monkeypatch.delenv("KRABBY_MQTT_HOST", raising=False)
    assert srv.Handler._jobs_live("001-patio") == {}      # no broker configured → file truth stands
