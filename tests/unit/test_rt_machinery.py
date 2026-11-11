"""CPU tests of the rough-terrain run machinery (PLAN steps 5, 6, 8, 9): env composition, the build pin, the row
plans of every block, the sentinels and the log-sentinel watcher, the training-log telemetry parse, the trainer argv,
and the pure halves of the step-5 checks. No Kit, no GPU, no launch: gate_row.sh / rt_kit_run.sh run in their dry
modes only."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-11-10_rough_terrain"
SCRIPTS = CAMP / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.dont_write_bytecode = True
M = _load("rt_machinery")
C = _load("rt_smoke_check")
Q = _load("rt_queue")
B = _load("build_check")
T = _load("rt_train")


# ============================================================================== env composition
def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def test_compose_child_env_takes_krabby_only_from_the_env_file(tmp_path):
    envf = _write(tmp_path / "a.env", "# c\nKRABBY_A=1\nKRABBY_EMPTY=\nPYTHONPATH=%s\n" % M.PYTHONPATH_W)
    lvl = _write(tmp_path / "l.env", "KRABBY_PAYLOAD_KG=90:90\nKRABBY_OTHER=9\n")
    base = {"KRABBY_STALE": "x", "TP_OUT": "/x", "RSD_Y": "1", "PYTHONPATH": "/elsewhere", "HOME": "/h"}
    env, kr, probs = M.compose_child_env(base, envf, lvl)
    assert probs == []
    assert kr == {"KRABBY_A": "1", "KRABBY_EMPTY": "", "KRABBY_PAYLOAD_KG": "90:90"}
    assert {k for k in env if k.startswith("KRABBY_")} == set(kr)
    assert "TP_OUT" not in env and "RSD_Y" not in env and env["HOME"] == "/h"
    assert env["PYTHONPATH"] == M.PYTHONPATH_W and env["OMNI_KIT_ACCEPT_EULA"] == "yes"


def test_compose_child_env_refusals(tmp_path):
    envf = _write(tmp_path / "a.env", "KRABBY_A=1\nKRABBY_A=2\nKRABBY_B=$REPO/x\nbad line\nPYTHONPATH=/other\n")
    lvl = _write(tmp_path / "l.env", "KRABBY_PAYLOAD_KG=1\n")
    envf2 = _write(tmp_path / "b.env", "KRABBY_PAYLOAD_KG=0:180\n")
    _e, _k, probs = M.compose_child_env({}, envf)
    joined = " | ".join(probs)
    assert "repeated" in joined and "'$'" in joined and "not a KEY=VALUE" in joined and "PYTHONPATH" in joined
    _e, _k, probs = M.compose_child_env({}, envf2, lvl)
    assert any("already set" in p for p in probs)


def test_rt_train_env_is_tdrs_with_exactly_the_planned_changes(tmp_path):
    assert M.train_env_problems() == []
    text = M.RT_TRAIN_ENV.read_text()
    bad = _write(tmp_path / "x.env", text.replace("KRABBY_DR_ROD_SPEED=0.6:1.35\n", "KRABBY_DR_ROD_SPEED=0.6:1.35:12000\n"))
    assert any("KRABBY_DR_ROD_SPEED" in p for p in M.train_env_problems(bad))
    bad = _write(tmp_path / "y.env", text.replace("KRABBY_RT_FRICTION=gravel:0.2:0.5\n", ""))
    assert any("missing ['KRABBY_RT_FRICTION']" in p for p in M.train_env_problems(bad))
    bad = _write(tmp_path / "z.env", text.replace("rubble=2", "rubble=3"))
    assert any("KRABBY_SITE_MIX" in p for p in M.train_env_problems(bad))
    bad = _write(tmp_path / "w.env", text.replace(str(M.W), "/somewhere/else"))
    assert any("non-KRABBY" in p for p in M.train_env_problems(bad))


def test_smoke_env_files_match_their_derivation_and_never_reach_training():
    assert M.make_envs(check=True) == []
    rt, ref = M.krabby_of(M.SMOKE_TRAIN_ENV), M.krabby_of(M.SMOKE_REF_ENV)
    train = M.krabby_of(M.RT_TRAIN_ENV)
    assert "KRABBY_EPISODE_S" not in train
    assert rt == {**train, "KRABBY_EPISODE_S": M.SMOKE_EPISODE_S}
    tdr = M.krabby_of(M.TDR_TRAIN_ENV)
    assert ref == {**tdr, "KRABBY_DR_ROD_SPEED": "0.6:1.35", "KRABBY_EPISODE_S": M.SMOKE_EPISODE_S}
    assert not any(k.startswith(("KRABBY_SITE_", "KRABBY_RT_")) for k in ref)


# ============================================================================== pin
LIVE = {"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": [], "head": "h1", "repo": "/r"}


def _pin(**kw):
    p = {"schema": "rt_pin_v1", "code_sha": "c" * 64, "eval_sha": "e" * 64, "head": "h1", "repo": "/r",
         "frozen_files": {"manifests/a.json": "1"}, "reported_files": {"scripts/x.py": "2"}}
    p.update(kw)
    return p


def test_pin_problems():
    f, r = {"manifests/a.json": "1"}, {"scripts/x.py": "2"}
    assert M.pin_problems(_pin(), LIVE, f, r) == ([], [])
    bad, _ = M.pin_problems(_pin(code_sha="d" * 64), LIVE, f, r)
    assert any(b.startswith("code_sha") for b in bad)
    bad, _ = M.pin_problems(_pin(eval_sha="d" * 64), LIVE, f, r)
    assert any(b.startswith("eval_sha") for b in bad)
    bad, _ = M.pin_problems(_pin(), {**LIVE, "dirty": ["parkour/x.py"]}, f, r)
    assert any("dirty" in b for b in bad)
    bad, _ = M.pin_problems(_pin(), LIVE, {"manifests/a.json": "9"}, r)
    assert any("frozen campaign files" in b for b in bad)
    bad, warn = M.pin_problems(_pin(), {**LIVE, "head": "h2"}, f, {"scripts/x.py": "3"})
    assert bad == [] and len(warn) == 2
    bad, _ = M.pin_problems(None, {**LIVE, "dirty": ["a"]}, f, r)
    assert len(bad) == 2 and "no pin" in bad[0]


def test_build_check_pin_is_written_once_and_refuses_a_dirty_tree(tmp_path, monkeypatch, capsys):
    pin = tmp_path / "pin.json"
    monkeypatch.setattr(M, "live_build", lambda: {**LIVE, "dirty": ["parkour/x.py"]})
    assert B.main(["pin", "--pin-path", str(pin)]) == 1 and not pin.exists()
    monkeypatch.setattr(M, "live_build", lambda: dict(LIVE))
    assert B.main(["pin", "--pin-path", str(pin)]) == 0 and pin.is_file()
    doc = json.loads(pin.read_text())
    assert doc["code_sha"] == LIVE["code_sha"] and doc["schema"] == "rt_pin_v1" and doc["frozen_files"]
    assert B.main(["pin", "--pin-path", str(pin)]) == 1
    assert B.main(["check", "--pin-path", str(pin)]) == 0
    monkeypatch.setattr(M, "live_build", lambda: {**LIVE, "eval_sha": "f" * 64})
    assert B.main(["check", "--pin-path", str(pin)]) == 1
    assert "BUILD_CHECK BAD" in capsys.readouterr().out


def test_live_build_uses_run_candidates_own_functions():
    r = M._run_candidates()
    live = M.live_build()
    assert live["code_sha"] == r.code_sha() and live["eval_sha"] == r.eval_sha() and Path(live["repo"]) == M.W


# ============================================================================== plans
def _abs_exist(p) -> bool:
    return os.path.isabs(str(p)) and Path(p).exists()


def test_smoke_plan():
    steps = M.plan("smoke")
    lab = [s.label for s in steps]
    assert lab[:3] == ["S.check", "S.envs", "S.trainenv"] and all(s.stop_on_fail for s in steps[:3])
    ev = [s for s in steps if s.label.startswith("S.eval.rt_")]
    pr = [s for s in steps if s.label.startswith("S.probe.")]
    assert len(ev) == 21 and len(pr) == 21
    for s in ev:
        assert s.how == "gate" and s.argv[-2:] == ["--episodes", str(M.EVAL_SMOKE_EPISODES)] and s.slot[3] == 1
        assert s.env["TP_OUT"] == str(M.SMOKE_OUT)
    neu = next(s for s in steps if s.label == "S.neutral")
    assert neu.argv[4].endswith("T00_nominal_L090_A__field_light.env") and neu.env["TP_LEVEL_FILE"].endswith("L090.env")
    assert neu.argv[3] == str(M.TDR_CK) and neu.slot[2] == "field_light__omni"
    cmp_ = next(s for s in steps if s.label == "S.neutral.cmp")
    assert "@stamp:S.neutral" in cmp_.argv
    t50 = next(s for s in steps if s.label == "S.train50")
    assert t50.how == "train" and "--smoke" in t50.argv and "50" in t50.argv
    kits = [s for s in steps if s.how == "kit"]
    assert all(s.argv[1] == str(M.RT_KIT_RUN) for s in kits)
    assert {s.argv[2] for s in kits if s.label.startswith("S.train_")} == {str(M.SMOKE_TRAIN_ENV), str(M.SMOKE_REF_ENV)}


def test_baseline_plan_rows_are_the_step4_rows_verbatim():
    rows = M.rt_spec().load_rows()
    by = {r["id"]: r for r in rows["kind_rows"] + rows["retention_rows"]}
    steps = [s for s in M.plan("baseline") if s.how == "gate"]
    assert len(steps) == 21 + sum(len(r["env_seeds"]) for r in rows["retention_rows"]) == 40
    for s in steps:
        rid = s.label.split(".")[2]
        es = int(s.label.rsplit(".es", 1)[1])
        r = by[rid]
        name, ck, envf, man, sc, es_s, *args = s.argv[2:]
        assert name == M.row_name("RT6", M.TDR_SHA, rid) and ck == str(M.TDR_CK) and sc == r["scenario"]
        assert args == list(r["args"]) and int(es_s) == es and s.env["TP_OUT"] == str(M.OUT)
        assert envf == str(M.OMNI / r["env_file"]) and man == str(M.OMNI / r["manifest"])
        if r.get("level_file"):
            assert s.env["TP_LEVEL_FILE"] == str(M.OMNI / r["level_file"])
        else:
            assert "TP_LEVEL_FILE" not in s.env
        assert es in (r["env_seeds"]["baseline"] if isinstance(r["env_seeds"], dict) else r["env_seeds"])
        for p in (envf, man, *(([s.env["TP_LEVEL_FILE"]]) if "TP_LEVEL_FILE" in s.env else [])):
            assert _abs_exist(p), p
    assert len({s.slot for s in steps}) == len(steps)


def _thr(tmp_path) -> Path:
    k = {"slopes": ("gated", "b2"), "hills": ("gated", "b3"), "ledge_up": ("solved", "b2"),
         "ledge_down": ("gated", "b1"), "pits": ("solved", "b2"), "gravel": ("solved", "b3"), "rubble": ("gated", "b2")}
    doc = {"schema": "rt_thresholds_v1", "decision": "proceed", "tdr_sha256": M.TDR_SHA,
           "rule": {"gate_env_seeds": [1, 2], "k": 2},
           "kinds": {n: ({"status": s, "gate_band": b, "r_band": None} if s == "gated"
                         else {"status": s, "gate_band": None, "r_band": b}) for n, (s, b) in k.items()}}
    p = tmp_path / "thr.json"
    p.write_text(json.dumps(doc))
    return p


def test_gate_plan_both_teachers_on_gate_and_r_bands_and_the_new_teachers_retention(tmp_path):
    thr = _thr(tmp_path)
    new = tmp_path / "model_34990.pt"
    new.write_bytes(b"not TDR")
    steps = M.plan("gate", new, thr)
    nsha = M.sha256_file(new)
    rows = [s for s in steps if s.how == "gate"]
    kind = [s for s in rows if ".rt_" in s.label]
    assert len(kind) == 7 * 2 * 2
    assert {s.argv[2] for s in kind if s.label.startswith("G.new.")} == {
        M.row_name("RT9", nsha, r) for r, _e in M.gate_rows(json.loads(thr.read_text()))}
    assert all(s.argv[3] == str(M.TDR_CK) for s in kind if s.label.startswith("G.tdr."))
    ret = [s for s in rows if ".R_" in s.label]
    assert len(ret) == 19 and all(s.label.startswith("G.new.") and s.argv[3] == str(new) for s in ret)
    assert [s.label for s in steps][-2:] == ["G.audit", "G.read"]
    with pytest.raises(ValueError):
        M.plan("gate", None)


def test_gate_rows_skip_kinds_without_a_band():
    thr = {"rule": {"gate_env_seeds": [1, 2]}, "kinds": {"slopes": {"status": "gated", "gate_band": "b1"},
                                                           "pits": {"status": "above", "gate_band": None}}}
    assert M.gate_rows(thr) == [("rt_slopes_b1__omni", (1, 2))]


def test_train_plan_and_trainer_argv():
    steps = M.plan("train")
    assert [s.label for s in steps] == ["T.check", "T.envcheck", "T.train"]
    assert steps[-1].stop_on_fail and "--name" in steps[-1].argv and M.TRAIN_NAME in steps[-1].argv
    argv = M.trainer_argv("RT_s19", "/abs/model_28991.pt")
    assert argv == [M.PY, str(M.PARKOUR / "scripts/rsl_rl/train.py"), "--task", "Isaac-Crab-Hex-Omni-Teacher-v0",
                    "--headless", "--num_envs", "256", "--seed", "19", "--max_iterations", "6000", "--resume",
                    "--checkpoint", "/abs/model_28991.pt", "--run_name", "RT_s19"]
    assert "-I" not in argv
    assert M.final_iteration(28991, 6000) == 34990


def test_rt_train_cli_pre_registration():
    with pytest.raises(T.Usage):
        T.parse_args(["--name", "RT_s19", "--iterations", "50"])
    with pytest.raises(T.Usage):
        T.parse_args(["--name", "RT_x", "--smoke"])
    a = T.parse_args(["--name", "RT5_smoke50", "--smoke", "--iterations", "50"])
    assert a.iterations == 50 and a.checkpoint == str(M.TDR_CK)


def test_rt_train_environ_check():
    kr = {"KRABBY_A": "1"}
    good = {"PYTHONPATH": M.PYTHONPATH_W, "KRABBY_A": "1", "OMNI_KIT_ACCEPT_EULA": "yes"}
    assert T.environ_problems(good, kr) == []
    assert T.environ_problems({**good, "PYTHONPATH": "/main/parkour"}, kr)
    assert T.environ_problems({**good, "KRABBY_LEAK": "1"}, kr)


# ============================================================================== sentinels and the watcher
LOG_LINES = ["RTQ_START block=smoke", "RTQ S.check rc=0", "  RTQ S.x rc=1 (indented: not a sentinel)",
             "RTQ S.eval.rt_pits_b1__omni rc=8 (no stamp)", "RTQ NOTIFY early_read it=1500 a=0.1->0.2",
             "RTQ S.skip rc=0 (slot complete: /x; skipped)", "RTQ_DONE smoke fail=1"]


def test_sentinel_format_and_parse():
    assert M.sentinel("S.a", 0, "(x)") == "RTQ S.a rc=0 (x)" and M.done_line("smoke", 2) == "RTQ_DONE smoke fail=2"
    ev = M.parse_events(LOG_LINES)
    assert [e["kind"] for e in ev] == ["fail", "notify", "done"]
    assert ev[0]["label"] == "S.eval.rt_pits_b1__omni" and ev[0]["rc"] == 8 and ev[2]["fail"] == 1


def test_wait_script_reports_events_and_times_out(tmp_path):
    log = tmp_path / "q.log"
    log.write_text("\n".join(LOG_LINES) + "\n")
    env = {**os.environ, "RT_WAIT_POLL_S": "1"}
    r = subprocess.run(["bash", str(SCRIPTS / "rt_wait.sh"), str(log), "0", "1"], capture_output=True, text=True,
                       env=env, timeout=60)
    lines = r.stdout.strip().splitlines()
    assert [ln.split(" ", 2)[1] for ln in lines[:-1]] == ["line=4", "line=5", "line=7"]
    assert lines[-1] == f"RT_WAIT lines={len(LOG_LINES)}"
    r = subprocess.run(["bash", str(SCRIPTS / "rt_wait.sh"), str(log), str(len(LOG_LINES)), "0"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.startswith("RT_WAIT TIMEOUT") and "RTQ_DONE smoke fail=1" in r.stdout
    code = [ln for ln in (SCRIPTS / "rt_wait.sh").read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert not any("pgrep" in ln or "ps " in ln for ln in code)            # the log is the only signal


def _stamp(slot: Path, name: str, rc: int = 0, code_sha: str = "c" * 64, eval_sha: str = "e" * 64,
           dirty: str = "False") -> Path:
    d = slot / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "code_state.json").write_text(json.dumps({"code_sha": code_sha, "eval_sha": eval_sha, "dirty": dirty}))
    (d / "tp_row.json").write_text(json.dumps({"rc": rc}))
    return d


def test_queue_resolves_stamp_placeholders(tmp_path):
    slot = tmp_path / "lr" / "N" / "sc" / "seed001"
    (slot / "2026-01-01_00-00-00").mkdir(parents=True)
    st = M.Step("S.r", "gate", ["x"], slot=(str(tmp_path), "N", "sc", 1))
    argv, why = Q.resolve(["a", "@stamp:S.r"], {"S.r": st}, _pin())
    assert why and "no complete stamp" in why
    _stamp(slot, "2026-01-01_00-00-00")
    _stamp(slot, "2026-01-02_00-00-00", rc=3)
    argv, why = Q.resolve(["a", "@stamp:S.r"], {"S.r": st}, _pin())
    assert why is None and argv == ["a", str(slot / "2026-01-01_00-00-00")]
    # a newer complete stamp at another build (a pre-re-pin smoke) is never substituted; nor anything without a pin
    _stamp(slot, "2026-01-03_00-00-00", code_sha="d" * 64)
    _stamp(slot, "2026-01-04_00-00-00", dirty="True")
    argv, why = Q.resolve(["a", "@stamp:S.r"], {"S.r": st}, _pin())
    assert why is None and argv == ["a", str(slot / "2026-01-01_00-00-00")]
    argv, why = Q.resolve(["a", "@stamp:S.r"], {"S.r": st}, _pin(code_sha="d" * 64))
    assert why is None and argv == ["a", str(slot / "2026-01-03_00-00-00")]
    argv, why = Q.resolve(["a", "@stamp:S.r"], {"S.r": st}, None)
    assert why and "no pin" in why


def test_complete_stamps_filter_on_the_pinned_build(tmp_path):
    slot = tmp_path / "slot"
    a = _stamp(slot, "a")
    b = _stamp(slot, "b", code_sha="d" * 64)
    c = _stamp(slot, "c", eval_sha="f" * 64)
    d = _stamp(slot, "d", dirty="True")
    e = _stamp(slot, "e", rc=1)
    f = slot / "f"
    f.mkdir()
    (f / "tp_row.json").write_text(json.dumps({"rc": 0}))                  # no code_state.json
    assert M.complete_stamps(slot) == [a, b, c, d, f]                         # no pin: the rc rule only
    assert M.complete_stamps(slot, pin=_pin()) == [a]
    assert M.build_mismatch(M.stamp_build(a), _pin()) == []
    assert M.build_mismatch(M.stamp_build(a), None) and M.build_mismatch(None, _pin())
    assert e not in M.complete_stamps(slot, pin=_pin(code_sha="c" * 64))


def _queue_main(monkeypatch, capsys, tmp_path, pin):
    slot = tmp_path / "lr" / "N" / "sc" / "seed001"
    step = M.Step("S.row", "gate", [sys.executable, "-c", "print('ROW RAN')"], slot=(str(tmp_path), "N", "sc", 1))
    monkeypatch.setattr(M, "plan", lambda *a, **k: [step])
    monkeypatch.setattr(M, "read_pin", lambda *a, **k: pin)
    rc = Q.main(["--block", "smoke", "--dry"])
    return rc, capsys.readouterr().out


def test_queue_never_skips_a_slot_complete_at_another_build(tmp_path, monkeypatch, capsys):
    """Finding (step-5 rerun after a re-pin): a slot holding a complete stamp at a different code_sha is NOT
    skipped; one at the pinned build is; with no pin nothing is skipped."""
    slot = tmp_path / "lr" / "N" / "sc" / "seed001"
    _stamp(slot, "2026-01-01_00-00-00", code_sha="d" * 64)               # the old build's complete stamp
    rc, out = _queue_main(monkeypatch, capsys, tmp_path, _pin())
    assert "skipped" not in out and "at another build" in out and "RTQ S.row rc=0 (dry)" in out, out
    rc, out = _queue_main(monkeypatch, capsys, tmp_path, None)
    assert "skipped" not in out and "RTQ S.row rc=0 (dry)" in out, out
    _stamp(slot, "2026-01-02_00-00-00")                                    # now one at the pinned build
    rc, out = _queue_main(monkeypatch, capsys, tmp_path, _pin())
    assert "RTQ S.row rc=0 (slot complete at the pinned build" in out and "skipped" in out, out


def _neutral_stamp(root: Path, code_sha: str) -> Path:
    st = _stamp(root, "new", code_sha=code_sha)
    (st / "run_meta.json").write_text(json.dumps({"checkpoint_sha256": M.TDR_SHA, "env_vars": {}}))
    return st


def test_neutrality_fails_on_a_stamp_at_another_build(tmp_path):
    st = _neutral_stamp(tmp_path / "a", "d" * 64)
    res = C.neutrality(st, [st], pin=_pin())
    assert res["verdict"] == "FAIL" and "not at the pinned build" in res["why"] and not res["pairs"]
    assert C.neutrality(st, [st], pin=None)["verdict"] == "FAIL"
    st = _neutral_stamp(tmp_path / "b", "c" * 64)                           # pinned build: the compare runs
    res = C.neutrality(st, [], pin=_pin())
    assert res["verdict"] == "FAIL" and "no saved TDR" in res["why"]


def test_audit_refuses_ok_rows_at_a_build_other_than_the_pin(monkeypatch):
    class _Rows:
        def __init__(self, roots, shas=None):
            pass

        def found(self, rid, es, sha):
            return {"status": "OK", "stamp": "/s", "problems": [], "code_sha": "d" * 64, "eval_sha": "e" * 64}

    class _RO:
        Rows = _Rows

    monkeypatch.setattr(M, "readout", lambda: _RO)
    res = M.audit("baseline", pin=_pin())
    assert res["bad"] == 0 and len(res["builds"]) == 1 and res["pin_problems"], res["pin_problems"]
    assert not M.audit("baseline", pin=_pin(code_sha="d" * 64))["pin_problems"]
    monkeypatch.setattr(M, "read_pin", lambda *a, **k: None)
    assert "no pin" in M.audit("baseline")["pin_problems"][0]


def test_queue_runner_env_strips_parent_knobs(monkeypatch):
    monkeypatch.setenv("KRABBY_LEAK", "1")
    monkeypatch.setenv("TP_STRIP_KEYS", "KRABBY_X")
    step = M.Step("S.r", "gate", ["x"], env={"TP_OUT": "/o"})
    env = Q.runner_env(step, dry=True)
    assert "KRABBY_LEAK" not in env and env["TP_OUT"] == "/o" and env["TP_ROW_DRY"] == "1"
    assert "TP_STRIP_KEYS" not in env and env["CUDA_VISIBLE_DEVICES"] == ""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for how in ("train", "gate", "kit"):                     # a CPU-only parent must not hide the GPU from Kit
        assert "CUDA_VISIBLE_DEVICES" not in Q.runner_env(M.Step("T.t", how, ["x"]), dry=False)
    assert Q.runner_env(M.Step("T.c", "cpu", ["x"]), dry=False)["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.skipif(not M.TDR_CK.is_file(), reason="TDR's policy file is not in this checkout")
def test_gate_row_and_kit_run_dry_compose_from_the_env_file_only(monkeypatch):
    steps = {s.label: s for s in M.plan("smoke")}
    env = {**os.environ, "KRABBY_LEAK": "1"}
    s = steps["S.neutral"]
    r = subprocess.run(s.argv, env={**env, **s.env, "TP_ROW_DRY": "1"}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "KRABBY_LEAK" not in r.stdout and "KRABBY_PAYLOAD_KG=90:90" in r.stdout and "TPROW END" in r.stdout
    s = steps["S.train_rt"]
    r = subprocess.run(s.argv, env={**env, **s.env, "KIT_RUN_DRY": "1"}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "KRABBY_LEAK" not in r.stdout and "KRABBY_EPISODE_S=6" in r.stdout
    assert "MemoryMax=48G" in r.stdout and "timeout --signal=TERM" in r.stdout and M.PYTHONPATH_W in r.stdout


# ============================================================================== guards
PROCS = [(10, "/home/nickmagus/krabby/isaac_venv/bin/python /w/scripts/rt_kit_train_smoke.py --headless"),
         (11, "/home/nickmagus/krabby/isaac_venv/bin/python -I -B /w/scripts/rt_queue.py --block smoke"),
         (12, "/home/nickmagus/krabby/isaac_venv/bin/python -I -B /w/scripts/rt_smoke_check.py eval"),
         (13, "/home/nickmagus/krabby/isaac_venv/bin/python -u /w/eval_crab_hex_gait.py --headless"),
         (14, "/home/nickmagus/krabby/isaac_venv/bin/python -I -B /w/scripts/rt_train.py --name RT_s19"),
         (15, "/usr/bin/python3 /w/train.py")]


def test_busy_pattern_sees_this_campaigns_kit_scripts_only():
    hit = {pid for pid, _ in M.busy(procs=PROCS)}
    assert hit == {10, 13, 14}
    assert {pid for pid, _ in M.busy(exclude=(14,), procs=PROCS)} == {10, 13}


def test_guard_problems():
    assert M.guard_problems(10.0, {"/": 100.0}) == []
    assert len(M.guard_problems(M.DDC_MAX_GB + 1, {"/": M.FREE_MIN_GB - 1})) == 2


# ============================================================================== training log telemetry
def _block(it: int, tot: int, counts: dict) -> str:
    lines = [f"                    Learning iteration {it}/{tot}                    ", "Mean reward: 1.0"]
    for kind, (failed, ended) in counts.items():
        lines += [f"Metrics/base_parkour/site_cnt_failed/{kind}: {failed:.4f}",
                  f"   Metrics/base_parkour/site_cnt_end/{kind}: {ended:.4f}"]
    return "\n".join(lines) + "\n"


def test_fall_shares_and_the_iteration_1500_line():
    it0 = 28991
    text = "".join(_block(it0 + k, it0 + 6000, {"slopes": (0.1, 1.0) if k < 1000 else (0.05, 1.0),
                                                  "gravel": (0.2, 0.5)}) for k in range(0, 1400, 10))
    assert M.early_line(text, it0) is None
    sh = M.fall_shares(M.iteration_blocks(text), it0, 300, 500)
    assert sh["slopes"]["share"] == pytest.approx(0.1) and sh["gravel"]["share"] == pytest.approx(0.4)
    text += "".join(_block(it0 + k, it0 + 6000, {"slopes": (0.05, 1.0), "gravel": (0.2, 0.5)})
                    for k in range(1400, 1510, 10))
    line = M.early_line(text, it0)
    assert line.startswith("RTQ NOTIFY early_read it=1500 ") and "slopes=0.100->0.050" in line
    assert "gravel=0.400->0.400" in line and "report only" in line
    assert M.NOTIFY_RE.match(line)


# ============================================================================== pure step-5 checks
def test_fit_decel_recovers_mu_and_the_slide_verdict_separates_combine_modes():
    t = np.arange(1, 40) * 0.02
    for mu in (0.2, 1.5):
        v = np.clip(1.0 - mu * C.G * t, 0, None)
        f = C.fit_decel(t, v)
        assert f["n"] >= 3 and f["mu_meas"] == pytest.approx(mu, rel=1e-6)
    assert C.slide_verdict(0.21, 0.2, low=True)["pass"]
    assert not C.slide_verdict(0.6, 0.2, low=True)["pass"]          # what an 'average' combine would read
    hi_ok, hi_low = C.slide_verdict(1.2, 1.5, low=False), C.slide_verdict(0.9, 1.5, low=False)
    assert hi_ok["pass"] and hi_low["pass"] and hi_ok["report_only"] and hi_low["report_only"]   # high mu: report only
    assert hi_ok["would_pass_old_line"] and not hi_low["would_pass_old_line"]
    assert not C.slide_verdict(None, 0.2, low=True)["pass"]
    # the sliding phase ends at the first rebound (the first smoke's pivoting high-mu body: speed falls, then rises)
    v = np.concatenate([np.clip(1.0 - 0.5 * C.G * t[:10], 0, None), [0.08, 0.17, 0.27, 0.35]])
    f = C.fit_decel(np.arange(1, len(v) + 1) * 0.02, v)
    assert f["n"] <= 11 and f["mu_meas"] == pytest.approx(0.5, rel=0.15)


def test_within_spawn_and_early_flags():
    assert C.within_spawn([1.08, 1.10])["pass"] and not C.within_spawn([1.20])["pass"]
    ref = {"flat": {"episodes": 100, "early": 2}}
    out = C.early_flags({"slopes": {"episodes": 20, "early": 10}, "gravel": {"episodes": 20, "early": 10},
                         "hills": {"episodes": 20, "early": 1}}, ref)
    assert out["flagged"] == ["slopes"] and "note" in out["kinds"]["gravel"] and not out["kinds"]["hills"]["flag"]


def _dump(level, surface0, kind, height=None):
    sr = C._sr()
    w = level.shape[0]
    fr = sr.TileFrame(size_xy=(10.0, 10.0), hs=0.08, vs=0.005, tile_px=(w, w), grid_rc=(1, 1))
    q = (lambda a: np.rint(np.asarray(a) / 0.005).astype(np.int32))
    return {"frame": fr, "terrain_origins": np.zeros((1, 1, 3)), "level": q(level), "surface0": q(surface0),
            "height_field": q(surface0 if height is None else height), "kind": np.asarray(kind, np.int8)}


def _grid(w=126):
    x = np.arange(w) * 0.08 - 5.0
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.hypot(xx, yy)


@pytest.mark.parametrize("grade,factor,ok", [(10.0, 1.0, True), (20.0, 1.0, True), (20.0, 0.5, False)])
def test_slope_checks_on_a_synthetic_frustum(grade, factor, ok):
    r = _grid()
    g = math.tan(math.radians(grade)) * factor
    r_out = min(1.7 + 0.8 / math.tan(math.radians(grade)), 4.4)
    h = -g * np.clip(r_out - r, 0.0, r_out - 1.7)
    tile = C.Tile(_dump(h, h, np.zeros_like(r, np.int8)), 0, 0)
    res = C.slope_checks(tile, grade, (0.0, 0.0), float(h[62, 62]))
    assert res["pass"] is ok and res["variant"] == "up_sunk_pad"


def test_realised_dose_ledge_pits_gravel():
    sr = C._sr()
    r = _grid()
    lvl = np.where(r < 1.7, -0.10, 0.0)
    tile = C.Tile(_dump(lvl, lvl, np.zeros_like(r, np.int8)), 0, 0)
    d = (0.10 - 0.02) / (0.30 - 0.02)
    assert C.realised_dose("ledge_up", tile, d, (0, 0), -0.10)["pass"]
    assert not C.realised_dose("ledge_down", tile, d, (0, 0), -0.10)["pass"]       # wrong sign
    kind = np.where((r > 2.2) & (r < 2.4), sr.KIND_PIT, 0).astype(np.int8)
    s0 = np.where(kind == sr.KIND_PIT, -0.20, 0.0)
    tile = C.Tile(_dump(np.zeros_like(r), s0, kind), 0, 0)
    dp = (0.20 - 0.03) / (0.35 - 0.03)
    assert C.realised_dose("pits", tile, dp, (0, 0), 0.0)["pass"]
    assert not C.realised_dose("pits", tile, 1.0, (0, 0), 0.0)["pass"]
    rough = np.where((r > 2.0) & (r < 4.1), 0.05 * np.sign(np.sin(r * 40)), 0.0)
    tile = C.Tile(_dump(np.zeros_like(r), np.zeros_like(r), np.zeros_like(r, np.int8), rough), 0, 0)
    assert C.realised_dose("gravel", tile, 1.0, (0, 0), 0.0)["pass"]


# ============================================================================== recorded data (skip when absent)
@pytest.mark.skipif(not M.RSD_LR.is_dir(), reason="the rod-speed campaign's eval rows are not on this machine")
def test_neutral_refs_are_keyed_by_sha_and_composition():
    refs = M.neutral_refs()
    assert refs, "TDR's nominal L090 field_light es1 stamps"
    for st in refs:
        meta = json.loads((st / "run_meta.json").read_text())
        assert meta["checkpoint_sha256"] == M.TDR_SHA and meta["scenario_id"] == "field_light__omni"
        assert meta["env_vars"]["KRABBY_PAYLOAD_KG"] == "90:90"
    assert C.bitwise_compare(refs[0], refs[0])["equal"]


def test_queue_echoes_key_child_lines_and_the_notify_line_at_column_0(tmp_path, capsys):
    code = ("print('noise'); print('TPROW END name=x rc=0 stamp=/s'); "
            "print('RTQ NOTIFY early_read it=1500 a=0.1->0.2'); print('RTQ S.fake rc=1')")
    log = tmp_path / "step.log"
    rc = Q.run_child([sys.executable, "-c", code], dict(os.environ), log)
    out = capsys.readouterr().out.splitlines()
    assert rc == 0 and "noise" not in "\n".join(out) and "noise" in log.read_text()
    assert "  TPROW END name=x rc=0 stamp=/s" in out and "RTQ NOTIFY early_read it=1500 a=0.1->0.2" in out
    assert not any(ln.startswith("RTQ S.fake") for ln in out)        # a child never forges a queue sentinel


def _fake_eval(tmp_path, *, pad, d, spawn_z=1.085, early=False, probe_pad=None, terrain_source="raycast_mesh",
               surface=None):
    """A synthetic eval-smoke stamp (4 episodes) + a probe dump of a 1 x 1 ledge_up grid with a sunk pad."""
    rows = {r["id"]: r for r in M.rt_spec().load_rows()["kind_rows"]}
    row = rows["rt_ledge_up_b2__omni"]
    st = tmp_path / "stamp"
    (st / "raw").mkdir(parents=True)
    eps = []
    for i in range(4):
        n = 50 if (early and i == 0) else 3675
        r = np.zeros((n, 3))
        r[:, 0] = np.linspace(0, 3.5, n)
        r[0] = (0.0, 0.0, pad + spawn_z)
        np.savez(st / "raw" / f"episode_{i:02d}.npz", root_pos_w=r, terrain_z=np.full((n, 6), pad),
                 dt=np.float64(0.02))
        eps.append({"env_index": i, "n_steps": n,
                    "termination_reason": "crab_failure" if (early and i == 0) else "schedule_complete"})
    (st / "scenario_metrics.json").write_text(json.dumps({"episodes": eps}))
    (st / "run_meta.json").write_text(json.dumps({"dt": 0.02, "terrain_source": terrain_source}))
    (st / "code_state.json").write_text(json.dumps({"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": "False"}))
    rr = _grid()
    lvl = np.where(rr < 1.7, pad, 0.0)

    def q(a):
        return np.rint(np.asarray(a) / 0.005).astype(np.int16)

    npz = tmp_path / "probe.npz"
    np.savez(npz, height_field=q(lvl), vs=np.float64(0.005), hs=np.float64(0.08), size_xy=np.array([10.0, 10.0]),
             tile_px=np.array([126, 126]), grid_rc=np.array([1, 1]), terrain_origins=np.zeros((1, 1, 3), np.float32),
             site=np.bool_(True), kind=np.zeros((126, 126), np.int8), level=q(lvl), surface0=q(lvl),
             material=np.zeros((126, 126), np.int8), tile_kind=np.zeros((1, 1), np.int8),
             tile_cross_r=np.full((1, 1), 3.2, np.float32), tile_d=np.full((1, 1), d, np.float32),
             tile_centre_level_m=np.full((1, 1), pad, np.float32), terrain_names=np.array([["site_ledge_up"]]),
             mix=np.str_("ledge_up=20"), site_d_range=np.array([0.3, 0.6]), gen_d_range=np.array([0.2, 1.0]))
    envs = [{"index": i, "row": 0, "col": 0, "tile_kind": "ledge_up", "tile_d": d,
             "centre_level_m": pad if probe_pad is None else probe_pad, "root_reset": [0.0, 0.0, pad + 1.085],
             "surface_under_root_reset": (pad if probe_pad is None else probe_pad) if surface is None else surface,
             "mu_shape_min": 1.0, "mu_shape_max": 1.0, "mu_eff_attr": None} for i in range(4)]
    pj = tmp_path / "probe.json"
    pj.write_text(json.dumps({"complete": True, "layers_npz": str(npz), "envs": envs, "env_seed": 1,
                              "pythonpath": M.PYTHONPATH_W,
                              "build": {"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": "False"},
                              "modules": {"site_raster": str(M.OMNI / "mdp" / "site_raster.py")}}))
    return row, st, pj


def test_eval_row_on_a_synthetic_stamp_and_probe(tmp_path):
    d = 0.45
    pad = -(0.02 + (0.30 - 0.02) * d)
    row, st, pj = _fake_eval(tmp_path, pad=pad, d=d)
    res = C.eval_row(row["id"], row, st, pj, pin=_pin())
    assert res["pass"], {k: v for k, v in res["checks"].items() if not v["pass"]}
    assert res["fall_share"] == 0.0
    # the stamp or the probe at another build (or no pin) FAILs the row before any join
    res = C.eval_row(row["id"], row, st, pj, pin=_pin(code_sha="d" * 64))
    assert not res["pass"] and not res["checks"]["stamp_build"]["pass"]
    assert not C.eval_row(row["id"], row, st, pj, pin=None)["pass"]
    doc = json.loads(pj.read_text())
    doc["build"]["eval_sha"] = "f" * 64
    pj.write_text(json.dumps(doc))
    res = C.eval_row(row["id"], row, st, pj, pin=_pin())
    assert not res["pass"] and res["checks"]["stamp_build"]["pass"] and not res["checks"]["probe_build"]["pass"]
    row, st, pj = _fake_eval(tmp_path / "b", pad=pad, d=d, spawn_z=1.2, early=True, probe_pad=pad + 0.2)
    res = C.eval_row(row["id"], row, st, pj, pin=_pin())
    bad = {k for k, v in res["checks"].items() if not v["pass"]}
    assert {"spawn_height", "no_early_termination", "probe_matches_harness"} <= bad and res["fall_share"] == 0.25
    row, st, pj = _fake_eval(tmp_path / "c", pad=pad, d=0.9)
    assert not C.eval_row(row["id"], row, st, pj, pin=_pin())["checks"]["tile_d_in_band"]["pass"]


def test_eval_spawn_check_refuses_the_root_offset_fallback_and_a_surface_mismatch(tmp_path):
    """The harness's no-mesh fallback (terrain_z = root_z - 1.05, inside 1.085 +- 0.05 by construction) must not
    pass the spawn check; nor may a stamp whose pad disagrees with the probe's surface under the root at reset."""
    d = 0.45
    pad = -(0.02 + (0.30 - 0.02) * d)
    row, st, pj = _fake_eval(tmp_path / "ok", pad=pad, d=d)
    res = C.eval_row(row["id"], row, st, pj, pin=_pin())
    assert res["pass"] and res["checks"]["terrain_mesh"]["pass"] and res["checks"]["spawn_height"]["pass"]
    # the fallback: terrain_z = root_z - 1.05 puts the spawn at 1.05 (within tolerance) yet the row must FAIL
    row, st, pj = _fake_eval(tmp_path / "fb", pad=pad, d=d, spawn_z=1.05, terrain_source="root_offset_fallback")
    res = C.eval_row(row["id"], row, st, pj, pin=_pin())
    assert C.within_spawn([1.05])["pass"]                      # the numbers alone would pass ...
    assert not res["pass"] and not res["checks"]["terrain_mesh"]["pass"] and not res["checks"]["spawn_height"]["pass"]
    # an older stamp with no terrain_source at all is not a measured surface either
    row, st, pj = _fake_eval(tmp_path / "none", pad=pad, d=d, terrain_source=None)
    assert not C.eval_row(row["id"], row, st, pj, pin=_pin())["checks"]["terrain_mesh"]["pass"]
    # the probe's own surface under the root at reset disagrees with the stamp's pad by > 0.02 m
    row, st, pj = _fake_eval(tmp_path / "surf", pad=pad, d=d, surface=pad + 0.03)
    res = C.eval_row(row["id"], row, st, pj, pin=_pin())
    assert not res["pass"] and not res["checks"]["probe_matches_harness"]["pass"]
    assert abs(res["checks"]["probe_matches_harness"]["envs"][0]["dsurf"] - 0.03) < 1e-9
    row, st, pj = _fake_eval(tmp_path / "surf_ok", pad=pad, d=d, surface=pad + 0.015)
    assert C.eval_row(row["id"], row, st, pj, pin=_pin())["pass"]


def test_eval_probe_records_the_live_build_before_kit_without_leaking_imports():
    """rt_kit_eval_probe.py's ``build`` (what rt_smoke_check eval compares with the pin): its _probe_build() run on
    its own (the script imports Kit at top level, so only that function is exercised)."""
    import ast

    src = (SCRIPTS / "rt_kit_eval_probe.py").read_text()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_probe_build"][0]
    ns = {"sys": sys, "HERE": SCRIPTS}
    exec(compile(ast.Module([fn], []), "rt_kit_eval_probe.py", "exec"), ns)
    path0, mods0 = list(sys.path), set(sys.modules)
    b = ns["_probe_build"]()
    assert "error" not in b, b
    assert sys.path == path0 and set(sys.modules) <= mods0
    live = M.live_build()
    assert (b["code_sha"], b["eval_sha"], b["dirty"]) == (live["code_sha"], live["eval_sha"], str(bool(live["dirty"])))
    assert '"build": PROBE_BUILD' in src
    assert src.index("PROBE_BUILD = _probe_build()") < src.index("AppLauncher(args)")


def test_eval_all_picks_the_newest_stamp_at_the_pinned_build(tmp_path, monkeypatch):
    rows = M.rt_spec().load_rows()["kind_rows"]
    r = rows[0]
    slot = tmp_path / "lr" / M.row_name("RT5", M.TDR_SHA, r["id"]) / r["scenario"] / "seed001"
    good = _stamp(slot, "2026-01-01_00-00-00")
    _stamp(slot, "2026-01-02_00-00-00", code_sha="d" * 64)                 # newer, other build: ignored
    seen = {}

    def fake_row(rid, row, stamp, probe, pin=None):
        seen[rid] = stamp
        return {"row": rid, "pass": True}

    monkeypatch.setattr(C, "eval_row", fake_row)
    C.eval_all(tmp_path / "lr", tmp_path / "probe", pin=_pin())
    assert seen[r["id"]] == good
    C.eval_all(tmp_path / "lr", tmp_path / "probe", pin=None)
    assert seen[r["id"]] is None


# ============================================================================== run-machinery minors (review follow-up)
def test_queue_crash_still_prints_the_done_sentinel_and_exits_non_zero(monkeypatch, capsys):
    """Any exception in the step loop ends the log with ``RTQ_DONE <block> fail=<n+1> (queue crashed: ...)``
    (rt_wait.sh would otherwise sit until MAXH with the GPU idle)."""
    def boom(argv, env, log):
        raise OSError("No space left on device")

    monkeypatch.setattr(Q, "run_child", boom)
    rc = Q.main(["--block", "baseline", "--dry"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1
    assert out[-1] == "RTQ_DONE baseline fail=1 (queue crashed: OSError: No space left on device)"
    assert M.DONE_RE.match(out[-1])
    # a crash after one counted failure reports n + 1 and a multi-line message stays on one line
    calls = []

    def flaky(argv, env, log):
        calls.append(argv)
        if len(calls) == 1:
            return 3                                           # a failed step: counted
        raise RuntimeError("first line\nsecond line")

    monkeypatch.setattr(Q, "run_child", flaky)
    rc = Q.main(["--block", "smoke", "--dry"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-1] == "RTQ_DONE smoke fail=2 (queue crashed: RuntimeError: first line second line)"
    # a crash in the plan itself (not a refusal type) is also covered
    monkeypatch.setattr(M, "plan", lambda *a, **k: (_ for _ in ()).throw(TypeError("bad plan")))
    rc = Q.main(["--block", "train", "--dry"])
    assert rc == 1 and capsys.readouterr().out.splitlines()[-1] == "RTQ_DONE train fail=1 (queue crashed: TypeError: bad plan)"


def _wait(log, frm, env=None, maxh="0"):
    r = subprocess.run(["bash", str(SCRIPTS / "rt_wait.sh"), str(log), str(frm), maxh], capture_output=True,
                       text=True, env={**os.environ, "RT_WAIT_POLL_S": "1", **(env or {})}, timeout=60)
    return r.stdout.strip().splitlines()


def test_wait_script_next_from_comes_from_the_same_snapshot_and_the_last_reported_hit(tmp_path):
    log = tmp_path / "q.log"
    lines = ["RTQ_START"] + [f"noise {i}" if i % 2 else f"RTQ S.r{i} rc=1" for i in range(1, 10)] + ["tail noise"]
    log.write_text("\n".join(lines) + "\n")
    hits = [i + 1 for i, ln in enumerate(lines) if ln.startswith("RTQ S.")]           # 1-based line numbers
    assert len(hits) == 4
    # more hits than the cap: report the first 3, FROM = the last REPORTED hit's line (not EOF), a 'more' line
    out = _wait(log, 0, {"RT_WAIT_MAX_HITS": "3"})
    assert [ln.split(" ")[1] for ln in out if ln.startswith("RT_WAIT line=")] == [f"line={n}" for n in hits[:3]]
    assert out[-1] == f"RT_WAIT lines={hits[2]}" and any(ln.startswith("RT_WAIT more=1 ") for ln in out)
    # re-armed from there the remaining hit is reported, then FROM = end of the snapshot's complete lines
    out = _wait(log, hits[2], {"RT_WAIT_MAX_HITS": "3"})
    assert [ln.split(" ")[1] for ln in out if ln.startswith("RT_WAIT line=")] == [f"line={hits[3]}"]
    assert out[-1] == f"RT_WAIT lines={len(lines)}" and not any(ln.startswith("RT_WAIT more") for ln in out)
    # a line still being written (no newline yet) is neither reported nor counted: a re-arm sees it whole, once
    log.write_text("\n".join(lines) + "\nRTQ S.partial rc=1 (still being wr")
    out = _wait(log, hits[3])
    assert out[-1].startswith("RT_WAIT TIMEOUT")
    log.write_text("\n".join(lines) + "\nRTQ S.partial rc=1 (written)\n")
    out = _wait(log, hits[3])
    assert out[0] == f"RT_WAIT line={len(lines) + 1} RTQ S.partial rc=1 (written)" and out[-1] == f"RT_WAIT lines={len(lines) + 1}"


def test_wait_script_uses_one_snapshot_of_the_log():
    code = [ln for ln in (SCRIPTS / "rt_wait.sh").read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert sum("tail -n +" in ln for ln in code) == 1                  # one read of the log per poll
    assert not any('wc -l < "$LOG"' in ln and "TIMEOUT" not in ln for ln in code)   # no second count of the live log


def test_runner_env_passes_only_the_guard_knobs_of_rtq(monkeypatch):
    for k, v in (("RTQ_DDC_MAX_GB", "90"), ("RTQ_FREE_MIN_GB", "10"), ("RTQ_RUN_X", "1"), ("RTQ_OTHER", "1")):
        monkeypatch.setenv(k, v)
    env = Q.runner_env(M.Step("S.r", "gate", ["x"]), dry=False)
    assert env["RTQ_DDC_MAX_GB"] == "90" and env["RTQ_FREE_MIN_GB"] == "10"
    assert "RTQ_RUN_X" not in env and "RTQ_OTHER" not in env
    assert "every KRABBY_* / TP_* / KIT_* / RTQ_* key" in Q.runner_env.__doc__ and "RTQ_PASS" in Q.runner_env.__doc__


def test_kit_run_honours_rtq_ddc_max_gb():
    sh = SCRIPTS / "rt_kit_run.sh"
    text = sh.read_text()
    assert "DDC_MAX_GB=${RTQ_DDC_MAX_GB:-80}" in text and "DDC_MAX_GB=80\n" not in text
    envf = CAMP / "envs" / "RT_train.env"

    def run(val):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "RTQ_"))}
        if val is not None:
            e["RTQ_DDC_MAX_GB"] = val
        e["KIT_RUN_DRY"] = "1"
        return subprocess.run(["bash", str(sh), str(envf), str(SCRIPTS / "rt_kit_train_smoke.py")], env=e,
                              capture_output=True, text=True, timeout=60)

    assert run(None).returncode == 0 and run("120").returncode == 0 and run("80.5").returncode == 0
    r = run("lots")
    assert r.returncode == 2 and "RTQ_DDC_MAX_GB must be a positive number" in r.stdout


def test_dry_run_preflight_failure_is_non_zero_except_the_expected_pre_pin_pf1(monkeypatch, capsys):
    def ok(i):
        return T.pf(i, True, "fine")

    pf3 = {**T.pf("PF3", True, "fine"), "data": {"expected": {"final_iter": 34990}}}
    pre_pin = T.pf("PF1", False, "no pin x; dirty code/eval paths ['a']",
                   problems=["no pin x (write it once)", "dirty code/eval paths ['a']"])
    moved = T.pf("PF1", False, "code_sha moved", problems=["code_sha abc != pin def"])
    base = {"PF2": ok("PF2"), "PF3": pf3, "PF4": ok("PF4"), "PF5": ok("PF5"), "PF6": ok("PF6"), "PF7": ok("PF7")}
    assert T.dry_failures({"PF1": pre_pin, **base}) == []                  # expected: behaves as before
    assert T.dry_failures({"PF1": moved, **base}) == ["PF1"]
    assert T.dry_failures({"PF1": T.pf("PF1", False, "build check failed: boom"), **base}) == ["PF1"]
    assert T.dry_failures({"PF1": pre_pin, **{**base, "PF5": T.pf("PF5", False, "x")}}) == ["PF5"]
    monkeypatch.setattr(T, "base_env", lambda: {})
    for res, want_rc, tail in (({"PF1": pre_pin, **base}, 0, "preflight=FAIL"),
                               ({"PF1": ok("PF1"), **base}, 0, "preflight=PASS"),
                               ({"PF1": ok("PF1"), **{**base, "PF4": T.pf("PF4", False, "argv")}}, 5,
                                "preflight=FAIL unexpected=['PF4']")):
        monkeypatch.setattr(T, "preflight", lambda *a, _r=res, **k: _r)
        assert T.main(["--name", "RT_s19", "--dry-run"]) == want_rc
        assert capsys.readouterr().out.strip().splitlines()[-1].endswith(tail)
    monkeypatch.setattr(T, "preflight", lambda *a, **k: {"PF1": moved, **base})
    assert T.main(["--name", "RT_s19", "--preflight-only"]) == 5


def test_queue_dry_tally_counts_a_failed_dry_preflight(monkeypatch, capsys):
    monkeypatch.setattr(Q, "run_child", lambda argv, env, log: 5 if "--dry-run" in argv else 0)
    Q.main(["--block", "train", "--dry"])
    out = capsys.readouterr().out
    assert "RTQ T.train rc=5 (dry)" in out and "RTQ_DONE train fail=1" in out


def test_smoke_plan_has_the_train50_params_step_and_the_gate_ident_record(tmp_path):
    lab = [s.label for s in M.plan("smoke")]
    assert lab.index("S.train50.params") == lab.index("S.train50") + 1
    st = {s.label: s for s in M.plan("smoke")}["S.train50.params"]
    assert st.how == "cpu" and not st.stop_on_fail and st.argv[4:6] == ["params-diff", "--name"] and "--smoke" in st.argv
    ck = tmp_path / "model_34990.pt"
    ck.write_bytes(b"x")
    steps = M.plan("gate", ck)
    g = {s.label: s for s in steps}["G.ident"]
    assert g.argv[g.argv.index("--record") + 1] == str(M.RESULTS / "runs" / f"{M.TRAIN_NAME}.json")
    assert [s.label for s in steps].index("G.ident") < [s.label for s in steps].index("G.audit")


# ---- params diff (PF14)
try:
    import yaml  # noqa: F401

    HAVE_YAML = True
except ImportError:                                        # a CI runner without PyYAML: the params tests skip
    HAVE_YAML = False
TDR_PARAMS_OK = HAVE_YAML and (M.TDR_PARAMS / "env.yaml").is_file() and (M.TDR_PARAMS / "agent.yaml").is_file()


def _expect(name="RT5_smoke50", iterations=50, seed=19):
    return {"name": name, "iterations": iterations, "seed": seed, "checkpoint": str(M.TDR_CK)}


def _planned_params(tmp_path, *, iterations=50):
    """A synthetic run params/ dir: TDR's, with exactly the planned differences applied (text-level edits)."""
    import re

    new = tmp_path / "params"
    new.mkdir(parents=True)
    env = M.normalise_roots((M.TDR_PARAMS / "env.yaml").read_text())
    agent = M.normalise_roots((M.TDR_PARAMS / "agent.yaml").read_text())
    env = env.replace("      ramp_steps: 12000", "      ramp_steps: 0")
    env = env.replace("friction_combine_mode: average", "friction_combine_mode: multiply", 1)
    env += "rt_friction:\n  gravel_range:\n  - 0.2\n  - 0.5\n"
    agent = agent.replace("run_name: RSD_TDR_s19", "run_name: RT5_smoke50")
    agent = agent.replace("max_iterations: 3000", f"max_iterations: {iterations}")
    agent = re.sub(r"load_checkpoint: .*", f"load_checkpoint: {M.TDR_CK}", agent)
    (new / "env.yaml").write_text(env)
    (new / "agent.yaml").write_text(agent)
    return new


@pytest.mark.skipif(not TDR_PARAMS_OK, reason="TDR's recorded run params are not on this machine")
def test_params_diff_identity_and_planned_differences(tmp_path):
    kr = M.krabby_of(M.RT_TRAIN_ENV)
    same = M.params_check(M.TDR_PARAMS, M.TDR_PARAMS, _expect(name="RSD_TDR_s19", iterations=3000), kr)
    # identical params: no differences at all (the loader reads every python tag TDR's files carry)
    assert same["ok"] and all(not r["allowed"] and not r["disallowed"] for r in same["files"].values())
    assert same["files"]["env.yaml"]["n_compared"] > 500
    new = _planned_params(tmp_path)
    res = M.params_check(new, M.TDR_PARAMS, _expect(), kr, smoke=True)
    assert res["ok"], res["files"]
    paths = {d["path"] for r in res["files"].values() for d in r["allowed"]}
    assert {"actions.joint_pos.rod_speed_dr.ramp_steps", "run_name", "max_iterations", "rt_friction.gravel_range"} <= paths
    assert any(p.startswith("sim.physics_material") or p.startswith("scene.terrain") for p in paths)


@pytest.mark.skipif(not TDR_PARAMS_OK, reason="TDR's recorded run params are not on this machine")
def test_params_diff_fails_on_an_unplanned_difference(tmp_path):
    kr = M.krabby_of(M.RT_TRAIN_ENV)
    new = _planned_params(tmp_path)
    env = (new / "env.yaml").read_text()
    # a quiet recipe drift: one reward weight changed
    i = env.index("\nrewards:")
    j = env.index("weight:", i)
    k = env.index("\n", j)
    (new / "env.yaml").write_text(env[:j] + "weight: 123.456" + env[k:])
    res = M.params_check(new, M.TDR_PARAMS, _expect(), kr, smoke=True)
    bad = [d for r in res["files"].values() for d in r["disallowed"]]
    assert not res["ok"] and len(bad) == 1 and bad[0]["path"].startswith("rewards.") and "allow-list" in bad[0]["why"]
    # an agent-side drift (learning rate) and wrong planned VALUES (ramp left at 12000; the wrong iteration count)
    new2 = _planned_params(tmp_path / "b", iterations=999)
    (new2 / "agent.yaml").write_text((new2 / "agent.yaml").read_text().replace("learning_rate: 0.0003",
                                                                               "learning_rate: 0.0005"))
    (new2 / "env.yaml").write_text((new2 / "env.yaml").read_text().replace("ramp_steps: 0", "ramp_steps: 12000"))
    res = M.params_check(new2, M.TDR_PARAMS, _expect(), kr, smoke=True)
    why = " | ".join(d["path"] + ": " + d["why"] for r in res["files"].values() for d in r["disallowed"])
    assert not res["ok"] and "learning_rate" in why and "max_iterations: new 999" in why
    # rules derive from the env file: with no friction contract armed, the rt_friction block is unplanned
    kr_off = {k: v for k, v in kr.items() if k != "KRABBY_RT_FRICTION"}
    res = M.params_check(_planned_params(tmp_path / "c"), M.TDR_PARAMS, _expect(), kr_off, smoke=True)
    assert not res["ok"] and any(d["path"].startswith("rt_friction") for r in res["files"].values()
                                 for d in r["disallowed"])
    # a missing params file is a failure, not a pass
    (new / "agent.yaml").unlink()
    assert not M.params_check(new, M.TDR_PARAMS, _expect(), kr)["ok"]


@pytest.mark.skipif(not TDR_PARAMS_OK, reason="TDR's recorded run params are not on this machine")
def test_params_diff_cli_prints_every_difference(tmp_path, capsys):
    new = _planned_params(tmp_path)
    args = ["params-diff", "--name", "RT5_smoke50", "--smoke", "--iterations", "50", "--run-dir", str(tmp_path)]
    rc = M.main(args)
    out = capsys.readouterr().out.splitlines()
    assert rc == 0 and out[-1].startswith("RT PARAMS OK ") and any(ln.startswith("RT PARAMS allowed env.yaml") for ln in out)
    (new / "agent.yaml").write_text((new / "agent.yaml").read_text().replace("gamma: 0.99", "gamma: 0.98"))
    rc = M.main(args)
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[-1].startswith("RT PARAMS BAD ") and any(
        ln.startswith("RT PARAMS DISALLOWED agent.yaml algorithm.gamma") for ln in out)
    assert M.main(["params-diff", "--name", "RT_nonexistent_zz"]) == 1


def test_rt_train_runs_the_params_check_in_run():
    src = (SCRIPTS / "rt_train.py").read_text()
    assert "res.append(params_pf(a, krabby))" in src and 'pf("PF14"' in src and "M.TDR_PARAMS" in src
    assert "PF14" in T.__doc__


# ---- G.ident: the run record
def _record(tmp_path, ck, **over):
    rec = {"name": M.TRAIN_NAME, "status": "ok", "cli": {"smoke": False},
           "final": {"path": str(ck), "sha256": M.sha256_file(ck), "iter": M.final_iteration(M.TDR_ITER, M.TRAIN_ITERS)}}
    rec.update(over)
    p = tmp_path / "RT_s19.json"
    p.write_text(json.dumps(rec))
    return p


def test_gate_ident_requires_the_step8_run_record(tmp_path):
    ck = tmp_path / "model_34990.pt"
    ck.write_bytes(b"the final teacher")
    assert M.final_iteration(M.TDR_ITER, M.TRAIN_ITERS) == 34990
    assert M.gate_ident_problems(ck, _record(tmp_path, ck)) == []
    assert any("no readable" in w for w in M.gate_ident_problems(ck, tmp_path / "nope.json"))
    # another file (an intermediate checkpoint, the smoke model, a wrong path): its sha is not the record's
    other = tmp_path / "model_30000.pt"
    other.write_bytes(b"an intermediate checkpoint")
    assert any("!= the run record's final" in w for w in M.gate_ident_problems(other, _record(tmp_path, ck)))
    # a run that did not finish ok, or stopped short of 34990
    assert any("status" in w for w in M.gate_ident_problems(ck, _record(tmp_path, ck, status="crashed")))
    bad_iter = {"path": str(ck), "sha256": M.sha256_file(ck), "iter": 33000}
    assert any("final iter" in w for w in M.gate_ident_problems(ck, _record(tmp_path, ck, final=bad_iter)))
    # the RT5_smoke50 launcher smoke is refused outright, even when its sha matches
    smoke = _record(tmp_path, ck, name="RT5_smoke50", cli={"smoke": True})
    why = " ".join(M.gate_ident_problems(ck, smoke))
    assert "launcher smoke" in why and "RT_s19" in why
    # through the CLI (the thresholds file is absent here: its own refusal, the record's rides along)
    rc = M.main(["gate-ident", "--new-ckpt", str(ck), "--record", str(smoke), "--thresholds", str(tmp_path / "t.json")])
    assert rc == 1
