"""Rod-speed DR campaign launchers (``crab_hex_omni_task/experiments/2026-11-07_rod_speed_dr/scripts``), on CPU.

``rsd_train.py`` launches and supervises the teacher fine-tune (TDR / TCTL), ``rsd_distill.sh`` wraps W's
``distill.sh`` for the student, ``rsd_distill_check.py`` runs the distill's +10-iteration checks; ``rsd_launch.py``
holds what they share. Nothing here starts Isaac: the trainer is a fake script and ``systemd-run`` is never called.
Pinned, because each failure is silent in a real run:

* the busy pattern is distill.sh's own and matches ``rsd_train.py`` itself (so the launcher excludes its own pid);
* the env file is read verbatim (empty values kept), the composed environment drops every inherited
  ``KRABBY_*`` / ``TP_*`` / ``DRS_*`` / ``RSD_*`` / ``PYTHONPATH`` and refuses the stress / phase / DR-in-file keys;
* the trainer command is run_phases.train's order, absolute checkpoint, no ``-I``; the wrapper is the capped scope;
* PF7's predicted ``[DR]`` line is the module's own print for the draw an independent generator makes;
* the in-run checks (PF9-13) and PF14 pass on a faithful log and fail on each planted defect;
* the stop escalates SIGTERM, SIGTERM, SIGKILL in that order, and only when the trainer stays alive;
* an end-to-end launch with a fake trainer records ``ok``, and a counter mismatch stops it with ``preflight_fail``;
* the distill wrapper writes ``p3b.distill.env`` + the knob and refuses a relative checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
CD = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task" / "experiments" / \
    "2026-11-07_rod_speed_dr"
SCRIPTS = CD / "scripts"
if not (SCRIPTS / "rsd_lib.py").is_file():
    pytest.skip("the rod-speed DR campaign (experiments/2026-11-07_rod_speed_dr) is not in this checkout",
                allow_module_level=True)

torch = pytest.importorskip("torch")
P3B_DISTILL_ENV = CD.parent / "2026-10-16_task4_pipeline" / "F19" / "envs" / "p3b.distill.env"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(SCRIPTS))
    try:
        RL = _load("rsd_launch", SCRIPTS / "rsd_launch.py")
        T = _load("rsd_train", SCRIPTS / "rsd_train.py")
        DC = _load("rsd_distill_check", SCRIPTS / "rsd_distill_check.py")
    finally:
        sys.path.remove(str(SCRIPTS))
    return SimpleNamespace(RL=RL, T=T, DC=DC)


def _args(**kw):
    d = dict(arm="dr", name="RSD_TEST", checkpoint="/abs/model_100.pt", env_file="/abs/p.env", dr="0.6:1.35:240",
             iterations=30, num_envs=8, seed=19, timeout_s=600, expect_counter=None, wait_idle=False, dry_run=False,
             preflight_only=False)
    d.update(kw)
    return SimpleNamespace(**d)


# ============================================================================== layout
def test_layout_no_package_marker_no_bytecode_no_forbidden_names():
    assert not list(CD.rglob("__init__.py"))
    assert not list(CD.rglob("__pycache__")) and not list(CD.rglob("*.pyc"))
    for bad in ("cells", "stress_read", "run_sweep", "tp_lib"):
        assert not (SCRIPTS / f"{bad}.py").exists()


def test_paths_resolve_to_this_worktree(mods):
    RL = mods.RL
    assert RL.W == REPO
    assert RL.TRAIN.is_file() and RL.DISTILL_SH.is_file() and RL.RSD_PY.is_file() and RL.DR_KNOBS_PY.is_file()
    assert RL.CD == CD


# ============================================================================== busy pattern
def test_busy_pattern_is_distill_sh_and_matches_the_launcher_itself(mods):
    RL = mods.RL
    isaac, nonkit = RL.isaac_patterns()
    text = RL.DISTILL_SH.read_text()
    assert f"ISAAC_PAT='{isaac}'" in text and f"NONKIT_PAT='{nonkit}'" in text
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    trainer = f"{py} /w/parkour/scripts/rsl_rl/train.py --task X --run_name N"
    launcher = f"{py} -I -B /w/scripts/rsd_train.py --arm dr --name N"
    procs = [(11, trainer), (12, launcher), (13, "bash /w/rsd_distill.sh --name N"),
             (14, f"{py} -I -B /w/scripts/rsd_distill_check.py check --name N"), (15, f"{py} /x/sh_dagger.py --read")]
    got = RL.busy(procs=procs)
    assert [p for p, _ in got] == [11, 12]  # rsd_train.py matches `[^ ]*train.py` -> the launcher excludes itself
    assert [p for p, _ in RL.busy(exclude=(12,), procs=procs)] == [11]
    assert RL.trainer_pids(None, "N", procs=procs) == [11]
    assert RL.trainer_pids(None, "M", procs=procs) == []


# ============================================================================== env
def test_env_file_is_read_verbatim(tmp_path, mods):
    RL = mods.RL
    p = tmp_path / "x.env"
    p.write_text("# c\n\nKRABBY_A=1\nKRABBY_EMPTY=\nKRABBY_SP= a b \n")
    lines, problems = RL.read_env_lines(p)
    assert lines == [("KRABBY_A", "1"), ("KRABBY_EMPTY", ""), ("KRABBY_SP", " a b ")] and not problems
    p.write_text("KRABBY_A=1\nKRABBY_A=2\nFOO=1\nKRABBY_B=$REPO/x\n")
    _, problems = RL.read_env_lines(p)
    assert len(problems) == 3


def test_compose_env_drops_inherited_keys_and_refuses(mods):
    T = mods.T
    base = {"HOME": "/h", "PATH": "/p", "KRABBY_OLD": "1", "TP_X": "1", "DRS_Y": "1", "RSD_Z": "1",
            "PYTHONPATH": "/main", "DBUS_SESSION_BUS_ADDRESS": "d"}
    env, kr, refusals = T.compose_env(_args(), base, [("KRABBY_PLANT", "A15+B"), ("KRABBY_PHASEOUT", "")])
    assert not refusals
    assert kr == {"KRABBY_PLANT": "A15+B", "KRABBY_PHASEOUT": "", "KRABBY_DR_ROD_SPEED": "0.6:1.35:240"}
    for k in ("KRABBY_OLD", "TP_X", "DRS_Y", "RSD_Z"):
        assert k not in env
    assert env["PYTHONPATH"] == f"{REPO}/parkour/parkour_tasks:{REPO}/parkour"
    assert (env["OMNI_KIT_ACCEPT_EULA"], env["TERM"], env["PYTHONUNBUFFERED"]) == ("yes", "xterm", "1")
    assert env["HOME"] == "/h" and env["DBUS_SESSION_BUS_ADDRESS"] == "d"
    env, kr, _ = T.compose_env(_args(arm="ctl", dr=None), base, [("KRABBY_PLANT", "A15+B")])
    assert "KRABBY_DR_ROD_SPEED" not in env and kr == {"KRABBY_PLANT": "A15+B"}
    for key in ("KRABBY_STRESS_ROD_SPEED", "KRABBY_PHASE", "KRABBY_SIDE_PHASE", "KRABBY_DR_ROD_SPEED",
                "KRABBY_DR_ROD_SPD"):
        _, _, refusals = T.compose_env(_args(arm="ctl", dr=None), base, [(key, "1")])
        assert refusals, key


def test_cli_refusals(tmp_path, mods):
    T = mods.T
    ck = tmp_path / "model_1.pt"
    ck.write_bytes(b"x")
    ef = tmp_path / "e.env"
    ef.write_text("KRABBY_A=1\n")
    ok = ["--name", "RSD_A", "--checkpoint", str(ck), "--env-file", str(ef)]
    assert T.parse_args(["--arm", "dr", *ok]).dr == T.DEFAULT_DR
    assert T.parse_args(["--arm", "ctl", *ok]).dr is None
    for bad in (["--arm", "ctl", *ok, "--dr", "0.6:1.35"],
                ["--arm", "dr", "--name", "DRST_a", "--checkpoint", str(ck), "--env-file", str(ef)],
                ["--arm", "dr", "--name", "a-b", "--checkpoint", str(ck), "--env-file", str(ef)],
                ["--arm", "dr", "--name", "A", "--checkpoint", "model_1.pt", "--env-file", str(ef)],
                ["--arm", "dr", *ok, "--iterations", "0"]):
        with pytest.raises(T.Usage):
            T.parse_args(bad)


# ============================================================================== command
def test_trainer_argv_is_run_phases_order_and_the_wrapper_is_the_capped_scope(mods):
    T, RL = mods.T, mods.RL
    a = _args(checkpoint="/abs/x/model_25992.pt", name="RSD_TDR", iterations=3000, num_envs=256, seed=19)
    argv = T.trainer_argv(a)
    assert argv == [RL.PY, str(RL.TRAIN), "--task", "Isaac-Crab-Hex-Omni-Teacher-v0", "--headless", "--num_envs",
                    "256", "--seed", "19", "--max_iterations", "3000", "--resume", "--checkpoint",
                    "/abs/x/model_25992.pt", "--run_name", "RSD_TDR"]
    assert argv == T.template_argv(a) and "-I" not in argv
    rp = RL.RUN_PHASES_PY.read_text()
    assert all(t in rp for t in T.RUN_PHASES_CMD_TEXT)
    w = T.wrap(argv, "rsd-train-RSD_TDR-1", 14400)
    assert w[:12] == ["systemd-run", "--user", "--scope", "-q", "--unit=rsd-train-RSD_TDR-1", "-p", "MemoryMax=48G",
                      "-p", "MemorySwapMax=0", "timeout", "--signal=TERM", "14400"]
    assert w[12:] == argv


@pytest.mark.skipif(not P3B_DISTILL_ENV.is_file(), reason="no F19 env file")
def test_pf4_on_the_real_env_file(mods):
    T, RL = mods.T, mods.RL
    if not RL.P3B_ORCH_LOG.is_file():
        pytest.skip("the main checkout's F19 orchestrator.log is not here")
    lines, problems = RL.read_env_lines(P3B_DISTILL_ENV)
    for arm, dr in (("dr", "0.6:1.35:12000"), ("ctl", None)):
        a = _args(arm=arm, dr=dr, env_file=str(P3B_DISTILL_ENV))
        _, kr, refusals = T.compose_env(a, {}, lines)
        r = T.pf4(a, kr, refusals, problems)
        assert r["ok"], r["detail"]
    a = _args(arm="ctl", dr=None, env_file=str(P3B_DISTILL_ENV))
    _, kr, refusals = T.compose_env(a, {}, lines[:-1])
    assert not T.pf4(a, kr, refusals, problems)["ok"]


# ============================================================================== the predicted draw
def test_predicted_dr_is_the_modules_own_draw(mods):
    RL = mods.RL
    spec = {"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}
    p = RL.predicted_dr(spec, 256, 19, 0)
    mod = RL.rod_speed_dr()
    g = torch.Generator(device="cpu")
    g.manual_seed(mod.rod_speed_dr_seed(19))
    u = torch.rand(256, generator=g, dtype=torch.float32)
    s = 1.0 + (1.35 - 1.0) * u
    assert p["sha16"] == mod._sha16(s)
    assert p["min"] == float(s.min()) and p["max"] == float(s.max())
    assert p["line"].startswith("[DR] rod_speed n=256 step=0 bounds=[1, 1.35] (lo=0.6 hi=1.35 ramp_steps=12000)")
    assert p["line"].endswith(f"sha16={p['sha16']} module={RL.RSD_PY}")
    p2 = RL.predicted_dr({"lo": 0.6, "hi": 1.35, "ramp_steps": 0}, 192, 19, 0)
    assert p2["lo_eff"] == 0.6 and "bounds=[0.6, 1.35]" in p2["line"]


def test_dr_knob_line_is_dr_knobs_print(mods):
    RL = mods.RL
    spec, line = RL.dr_knob_line("0.6:1.35:12000")
    assert spec == {"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}
    assert line.startswith("[KNOB] KRABBY_DR_ROD_SPEED=0.6:1.35:12000 -> actions.joint_pos.rod_speed_dr = ")
    with pytest.raises(ValueError):
        RL.dr_knob_line("1.1:1.4:100")


# ============================================================================== yaml compare
def test_compare_params_normalises_roots_and_excepts_only_the_named_keys(tmp_path, mods):
    T, RL = mods.T, mods.RL
    ref, new = tmp_path / "ref", tmp_path / "new"
    ref.mkdir()
    new.mkdir()
    env_ref = f"seed: 19\nscene:\n  usd: {RL.MAIN}/assets/crab.usda\nactions:\n  joint_pos:\n    use_delay: true\n"
    env_new = (f"seed: 19\nscene:\n  usd: {RL.W}/assets/crab.usda\nactions:\n  joint_pos:\n    rod_speed_dr:\n"
               f"      lo: 0.6\n      hi: 1.35\n      ramp_steps: 12000\n    use_delay: true\n")
    (ref / "env.yaml").write_text(env_ref)
    (new / "env.yaml").write_text(env_new)
    (ref / "agent.yaml").write_text("seed: 19\nrun_name: p3b\nload_checkpoint: /m/a.pt\nmax_iterations: 3000\n")
    (new / "agent.yaml").write_text("seed: 19\nrun_name: RSD\nload_checkpoint: /w/b.pt\nmax_iterations: 3000\n")
    spec = {"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}
    assert T.compare_params(new, ref, "dr", spec)["ok"]
    assert not T.compare_params(new, ref, "ctl", None)["ok"]  # a DR block on the ctl arm
    (new / "agent.yaml").write_text("seed: 20\nrun_name: RSD\nload_checkpoint: /w/b.pt\nmax_iterations: 3000\n")
    assert not T.compare_params(new, ref, "dr", spec)["ok"]


# ============================================================================== in-run checks and PF14
KNOBS = ["[omni-knobs] a (2.0) synced", "[KNOB] KRABBY_PAYLOAD_KG=0:180 kg", "[KNOB] KRABBY_STD_PARAM=soft consumed"]


def _fake_log(RL, *, ck, it0, iters, spec, counter=5, lr=1e-5, knobs=None, dr_line=True, n_blocks=None,
              lo_shift=0.0, dr_knob=True, num_envs=8, seed=19, lo_lag=0):
    lines = ["[INFO][AppLauncher]: boot"] + list(knobs if knobs is not None else KNOBS)
    if spec is not None and dr_knob:
        value = (f"{spec['lo']}:{spec['hi']}:{spec['ramp_steps']}" if spec["ramp_steps"]
                 else f"{spec['lo']}:{spec['hi']}")
        lines.append(RL.dr_knob_line(value)[1])
    if spec is not None and dr_line:
        lines.append(RL.predicted_dr(spec, num_envs, seed, 0)["line"])
    lines += [f"[INFO]: Loading model checkpoint from: {ck}",
              f"[INFO]: Restored algorithm state: counter {counter}, learning_rate {lr:.3g}"]
    mod = RL.rod_speed_dr()
    for k in range(iters if n_blocks is None else n_blocks):
        lines += ["#" * 80, f"                    \x1b[1m Learning iteration {it0 + k}/{it0 + iters} \x1b[0m ",
                  "                       Mean reward: 0.2000"]
        if spec is not None:
            lo_true = mod.rod_speed_bounds(24 * max(0, k - lo_lag) + 12, spec["lo"], spec["hi"],
                                           spec["ramp_steps"])[0]
            lo_eff = lo_true + lo_shift
            # faithful to the module: every draw so far came from [lo_eff(reset), hi), so min >= the logged lo_eff
            lines += [f"               DR/rod_speed_lo: {lo_eff:.4f}",
                      f"             DR/rod_speed_mean: {(lo_true + spec['hi']) / 2:.4f}",
                      f"              DR/rod_speed_min: {lo_true + 0.01:.4f}",
                      f"              DR/rod_speed_max: {spec['hi'] - 0.01:.4f}"]
        lines.append("               Mean episode length: 100.00")
    return "\n".join(lines) + "\n"


def _pre(T, RL, a, ck, it0, counter=5, lr=1e-5):
    spec = None
    knob_line = None
    if a.arm == "dr":
        spec, knob_line = RL.dr_knob_line(a.dr)
    pre = {
        "PF3": {"data": {"facts": {"iter": it0}, "expected": {
            "loading": f"[INFO]: Loading model checkpoint from: {ck}",
            "restored": f"[INFO]: Restored algorithm state: counter {counter}, learning_rate {lr:.3g}"}}},
        "PF6": {"data": {"spec": spec, "knob_line": knob_line, "steps_per_iteration": 24}},
        "PF7": {"data": {"line": RL.predicted_dr(spec, a.num_envs, a.seed, 0)["line"] if spec else None}},
        "_p3b_knob_lines": list(KNOBS),
    }
    return spec, pre


@pytest.fixture()
def params_dirs(tmp_path, mods):
    ref, new = tmp_path / "ref", tmp_path / "run" / "params"
    ref.mkdir()
    new.mkdir(parents=True)
    (ref / "env.yaml").write_text("seed: 19\nactions:\n  joint_pos:\n    use_delay: true\n")
    (ref / "agent.yaml").write_text("seed: 19\nnum_steps_per_env: 24\nrun_name: p3b\nload_checkpoint: /a.pt\n")
    (new / "env.yaml").write_text("seed: 19\nactions:\n  joint_pos:\n    rod_speed_dr:\n      lo: 0.6\n"
                                  "      hi: 1.35\n      ramp_steps: 240\n    use_delay: true\n")
    (new / "agent.yaml").write_text("seed: 19\nnum_steps_per_env: 24\nrun_name: RSD\nload_checkpoint: /b.pt\n")
    return ref, new.parent


def test_inrun_checks_pass_on_a_faithful_log_and_fail_on_each_defect(monkeypatch, params_dirs, mods):
    T, RL = mods.T, mods.RL
    ref, run_dir = params_dirs
    monkeypatch.setattr(RL, "P3B_PARAMS", ref)
    a = _args()
    ck = "/abs/model_100.pt"
    spec, pre = _pre(T, RL, a, ck, 100)
    good = _fake_log(RL, ck=ck, it0=100, iters=30, spec=spec)
    res = {r["id"]: r for r in T.inrun_checks(a, good, pre, run_dir)}
    assert all(r["ok"] for r in res.values()), {k: r["detail"] for k, r in res.items() if not r["ok"]}
    bad = {
        "PF9": _fake_log(RL, ck=ck, it0=100, iters=30, spec=spec, counter=6),
        "PF10": _fake_log(RL, ck=ck, it0=100, iters=30, spec=spec, knobs=KNOBS + ["[KNOB] KRABBY_X=1 consumed"]),
        "PF11": _fake_log(RL, ck=ck, it0=100, iters=30, spec=spec, dr_line=False),
        "PF13": _fake_log(RL, ck=ck, it0=101, iters=30, spec=spec),
    }
    for pid, text in bad.items():
        res = {r["id"]: r for r in T.inrun_checks(a, text, pre, run_dir)}
        assert not res[pid]["ok"], pid
    res = {r["id"]: r for r in T.inrun_checks(a, good, pre, None)}
    assert not res["PF12"]["ok"]
    # ctl: a [DR] line or DR/ values are failures
    ac = _args(arm="ctl", dr=None)
    _, prec = _pre(T, RL, ac, ck, 100)
    res = {r["id"]: r for r in T.inrun_checks(ac, good, prec, run_dir)}
    assert not res["PF10"]["ok"] and not res["PF11"]["ok"] and not res["PF13"]["ok"]


def test_pf14_follows_the_ramp(tmp_path, mods):
    T, RL = mods.T, mods.RL
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    a = _args(iterations=30)
    ck = "/abs/model_100.pt"
    spec, pre = _pre(T, RL, a, ck, 100)
    torch.save({"iter": 129, "alg_state": {"counter": 9, "learning_rate": 1e-5}}, run_dir / "model_129.pt")
    old = T.LATE_ITERS, T.LAG_ITERS
    T.LATE_ITERS, T.LAG_ITERS = 10, 3
    try:
        good = _fake_log(RL, ck=ck, it0=100, iters=30, spec=spec)
        # a stale (lagging) lo within LAG_ITERS passes; one that never ramps (stuck at 1.0) fails
        assert T.pf14(a, _fake_log(RL, ck=ck, it0=100, iters=30, spec=spec, lo_lag=2), run_dir, pre)["ok"]
        assert not T.pf14(a, _fake_log(RL, ck=ck, it0=100, iters=30, spec=spec, lo_lag=6), run_dir, pre)["ok"]
        stuck = re.sub(r"DR/rod_speed_lo: \d\.\d+", "DR/rod_speed_lo: 1.0000", good)
        assert not T.pf14(a, stuck, run_dir, pre)["ok"]
        r = T.pf14(a, good, run_dir, pre)
        assert r["ok"], r["detail"]
        assert not T.pf14(a, _fake_log(RL, ck=ck, it0=100, iters=30, spec=spec, lo_shift=0.05), run_dir, pre)["ok"]
        assert not T.pf14(_args(iterations=31), good, run_dir, pre)["ok"]  # no model_130.pt
        ac = _args(arm="ctl", dr=None, iterations=30)
        _, prec = _pre(T, RL, ac, ck, 100)
        assert not T.pf14(ac, good, run_dir, prec)["ok"]
        assert T.pf14(ac, _fake_log(RL, ck=ck, it0=100, iters=30, spec=None), run_dir, prec)["ok"]
    finally:
        T.LATE_ITERS, T.LAG_ITERS = old


# ============================================================================== the stop
def test_stop_trainer_escalates_term_term_kill(mods):
    RL = mods.RL
    clock = [0.0]

    def sleep(s):
        clock[0] += s

    sent = []
    r = RL.stop_trainer(lambda: [7], grace=120, alive=lambda p: True, send=lambda p, s: sent.append(s),
                        sleep=sleep, clock=lambda: clock[0], say=lambda m: None)
    assert sent == [signal.SIGTERM, signal.SIGTERM, signal.SIGKILL] and r["sigkill"]
    assert clock[0] >= 240
    state = {"alive": True}
    sent2 = []

    def send(p, s):
        sent2.append(s)
        state["alive"] = False

    r = RL.stop_trainer(lambda: [7], grace=120, alive=lambda p: state["alive"], send=send, sleep=sleep,
                        clock=lambda: clock[0], say=lambda m: None)
    assert sent2 == [signal.SIGTERM] and not r["sigkill"]
    assert RL.stop_trainer(lambda: [], say=lambda m: None)["pids"] == []


# ============================================================================== end to end (fake trainer)
FAKE_TRAINER = r'''
import json, os, sys, time
cfg = json.load(open(sys.argv[1]))
print("\n".join(cfg["head"]), flush=True)
rd = cfg["run_dir"]
os.makedirs(os.path.join(rd, "params"), exist_ok=True)
for f, t in cfg["params"].items():
    open(os.path.join(rd, "params", f), "w").write(t)
for blk in cfg["blocks"]:
    print(blk, flush=True)
    time.sleep(cfg.get("sleep", 0.0))
import torch
torch.save({"iter": cfg["final_iter"], "alg_state": {"counter": 9, "learning_rate": 1e-5}}, cfg["final"])
'''


def _e2e(tmp_path, monkeypatch, mods, *, counter_in_log=5, sleep=0.0):
    T, RL = mods.T, mods.RL
    out = tmp_path / "out"
    for attr, p in (("TRAIN_OUT", out / "train"), ("RUNS", tmp_path / "results" / "runs"),
                    ("ENVS", tmp_path / "envs"), ("EVENTS_LOG", out / "train" / "events.log")):
        monkeypatch.setattr(RL, attr, p)
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "env.yaml").write_text("seed: 19\nactions:\n  joint_pos:\n    use_delay: true\n")
    (ref / "agent.yaml").write_text("seed: 19\nnum_steps_per_env: 24\nrun_name: p3b\nload_checkpoint: /a.pt\n")
    monkeypatch.setattr(RL, "P3B_PARAMS", ref)
    p3b_log = tmp_path / "p3b_train.log"
    p3b_log.write_text("\n".join(KNOBS) + "\n Learning iteration 1/2 \n")
    monkeypatch.setattr(RL, "P3B_TRAIN_LOG", p3b_log)
    orch = tmp_path / "orchestrator.log"
    d = {"KRABBY_OMNI_PHASE": "2a", "PYTHONUNBUFFERED": "1", "KRABBY_PLANT": "A15+B", "KRABBY_PHASEOUT": ""}
    orch.write_text("\n" * (RL.P3B_ORCH_LINE - 1) + f"[t] {RL.P3B_TAG}: Isaac mode 2a | 3 its | env {d!r}\n")
    monkeypatch.setattr(RL, "P3B_ORCH_LOG", orch)
    envf = tmp_path / "p3b.distill.env"
    envf.write_text("KRABBY_OMNI_PHASE=2a\nKRABBY_PLANT=A15+B\nKRABBY_PHASEOUT=\n")
    ck = tmp_path / "ck" / "model_100.pt"
    ck.parent.mkdir()
    torch.save({"iter": 100, "alg_state": {"counter": 5, "learning_rate": 1e-5}}, ck)
    # the fake checkpoint stands in for T* (PF3: a ramped dr launch resumes T* only)
    monkeypatch.setitem(RL.KNOWN_CHECKPOINTS, RL.sha256_file(ck),
                        {"label": "T* (fake)", "iter": 100, "counter": 5, "learning_rate": 1e-5, "tstar": True})
    run_dir = tmp_path / "logs" / "2026-11-07_10-00-00_RSD_E2E"
    monkeypatch.setattr(RL, "systemd_user_env", lambda: {"HOME": str(tmp_path), "PATH": os.environ["PATH"],
                                                         "KRABBY_LEAK": "1"})
    monkeypatch.setattr(T, "pf1", lambda: T.pf("PF1", True, "faked"))
    monkeypatch.setattr(T, "pf2", lambda env: T.pf("PF2", True, "faked"))
    monkeypatch.setattr(RL, "existing_runs", lambda name, roots=None: [])
    monkeypatch.setattr(RL, "guard", lambda *a, **k: {"free_gb": {"x": 999.0}, "ddc_gb": 1.0, "problems": []})
    monkeypatch.setattr(RL, "busy", lambda *a, **k: [])
    monkeypatch.setattr(RL, "find_run_dir", lambda name, exp, root=None: run_dir if run_dir.is_dir() else None)
    monkeypatch.setattr(T, "POLL_S", 0.05)
    a = _args(name="RSD_E2E", checkpoint=str(ck), env_file=str(envf), iterations=30, num_envs=8)
    spec, knob_line = RL.dr_knob_line(a.dr)
    text = _fake_log(RL, ck=str(ck), it0=100, iters=30, spec=spec, counter=counter_in_log)
    head, _, rest = text.partition("#" * 80)
    blocks = [("#" * 80 + b) for b in rest.split("#" * 80)]
    cfg = tmp_path / "fake.json"
    cfg.write_text(json.dumps({"head": head.rstrip("\n").split("\n"), "run_dir": str(run_dir), "blocks": blocks,
                               "sleep": sleep, "final_iter": 129, "final": str(run_dir / "model_129.pt"),
                               "params": {"env.yaml": "seed: 19\nactions:\n  joint_pos:\n    rod_speed_dr:\n"
                                                      "      lo: 0.6\n      hi: 1.35\n      ramp_steps: 240\n"
                                                      "    use_delay: true\n",
                                          "agent.yaml": "seed: 19\nnum_steps_per_env: 24\nrun_name: RSD_E2E\n"
                                                        f"load_checkpoint: {ck}\n"}}))
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(FAKE_TRAINER)
    seen = {}

    def fake_wrap(argv, unit, timeout_s):
        seen["argv"] = argv
        return [sys.executable, "-B", str(fake), str(cfg)]

    monkeypatch.setattr(T, "wrap", fake_wrap)
    procs = []
    real_popen = subprocess.Popen

    def popen(*args, **kw):
        seen["env"] = kw.get("env")
        p = real_popen(*args, **kw)
        procs.append(p)
        return p

    monkeypatch.setattr(T.subprocess, "Popen", popen)
    stops = []

    def fake_stop(pids_fn, **kw):
        stops.append(True)
        for p in procs:
            p.terminate()
        return {"pids": [p.pid for p in procs], "terms": 1, "sigkill": False}

    monkeypatch.setattr(RL, "stop_trainer", fake_stop)
    old = T.LATE_ITERS
    T.LATE_ITERS = 10
    try:
        rc = T.main(["--arm", "dr", "--name", "RSD_E2E", "--checkpoint", str(ck), "--env-file", str(envf),
                     "--dr", a.dr, "--iterations", "30", "--num-envs", "8"])
    finally:
        T.LATE_ITERS = old
    rec = json.loads((RL.RUNS / "RSD_E2E.json").read_text())
    log = (RL.TRAIN_OUT / "RSD_E2E.log").read_text()
    return rc, rec, log, seen, stops


def test_end_to_end_fake_trainer_ok(tmp_path, monkeypatch, mods):
    rc, rec, log, seen, stops = _e2e(tmp_path, monkeypatch, mods)
    assert rc == 0, (rec.get("status"), rec.get("why"), [r["detail"] for r in rec.get("preflight_inrun", [])
                                                         if not r["ok"]])
    assert rec["status"] == "ok" and not stops
    assert all(r["ok"] for r in rec["preflight_inrun"]) and rec["pf14"]["ok"]
    assert re.search(r"^RSD TRAIN START RSD_E2E ", log, re.M) and re.search(r"^RSD TRAIN END RSD_E2E status=ok", log,
                                                                           re.M)
    assert "KRABBY_LEAK" not in seen["env"] and seen["env"]["KRABBY_DR_ROD_SPEED"] == "0.6:1.35:240"
    assert seen["argv"][-2:] == ["--run_name", "RSD_E2E"]
    env_text = (mods.RL.ENVS / "RSD_E2E.train.env").read_text()
    assert "KRABBY_PHASEOUT=\n" in env_text and "KRABBY_DR_ROD_SPEED=0.6:1.35:240\n" in env_text


def test_end_to_end_counter_mismatch_stops_with_preflight_fail(tmp_path, monkeypatch, mods):
    rc, rec, log, _, stops = _e2e(tmp_path, monkeypatch, mods, counter_in_log=6, sleep=0.05)
    assert rc == mods.T.EXIT_PF_INRUN and rec["status"] == "preflight_fail"
    assert stops, "an in-run pre-flight failure must stop the trainer"
    assert re.search(r"^RSD TRAIN FAIL RSD_E2E status=preflight_fail", log, re.M)
    assert [r["id"] for r in rec["preflight_inrun"] if not r["ok"]] == ["PF9"]


# ============================================================================== distill wrapper
@pytest.mark.skipif(not P3B_DISTILL_ENV.is_file(), reason="no F19 env file")
def test_distill_wrapper_dry_run_writes_the_knob_into_the_env_file(tmp_path):
    ck = tmp_path / "model_100.pt"
    ck.write_bytes(b"x")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "TP_", "DRS_", "RSD_"))}
    env["TP_STRAY"] = "1"
    r = subprocess.run(["bash", str(SCRIPTS / "rsd_distill.sh"), "--name", "RSD_TEST_DRY", "--checkpoint", str(ck),
                        "--dry-run"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    envf = CD / "envs" / "p3b_dr.distill.env"
    assert envf.read_text() == P3B_DISTILL_ENV.read_text() + "KRABBY_DR_ROD_SPEED=0.6:1.35\n"
    assert "KRABBY_DR_ROD_SPEED=0.6:1.35" in r.stdout  # survived distill.sh's strip
    assert f"PYTHONPATH={REPO}/parkour/parkour_tasks:{REPO}/parkour" in r.stdout
    assert "--num_envs 192 --seed 19 --max_iterations 5000 --resume --checkpoint" in r.stdout
    assert "Isaac-Crab-Hex-Omni-Student-v0" in r.stdout
    r = subprocess.run(["bash", str(SCRIPTS / "rsd_distill.sh"), "--name", "RSD_TEST_DRY", "--checkpoint",
                        "model_100.pt", "--dry-run"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 5


def test_distill_check_reads_dr_and_teacher_lines(tmp_path, mods):
    RL, DC = mods.RL, mods.DC
    teacher = tmp_path / "t" / "model_100.pt"
    teacher.parent.mkdir()
    teacher.write_bytes(b"teacher")
    ref_log = tmp_path / "ref.log"
    ref_log.write_text("\n".join(KNOBS) + "\n Learning iteration 1/2 \n")
    spec = {"lo": 0.6, "hi": 1.35, "ramp_steps": 0}
    log = tmp_path / "st.log"
    run_dir = tmp_path / "run"
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "params" / "agent.yaml").write_text(f"seed: 19\nload_checkpoint: {teacher}\n")
    text = _fake_log(RL, ck=str(teacher), it0=100, iters=12, spec=spec, num_envs=192)
    log.write_text(text)
    kw = dict(name="X", teacher=teacher, log=log, run_dir=run_dir, dr="0.6:1.35", num_envs=192, seed=19,
              teacher_run_dir=tmp_path / "t", min_iters=10, reference_log=ref_log, live=False)
    rc, out = DC.checks(**kw)
    by = {r["id"]: r for r in out}
    assert all(by[k]["ok"] for k in ("D1", "D2", "D3", "D4")), {k: r["detail"] for k, r in by.items()}
    assert not by["D5"]["ok"] and rc == DC.RC_KILL  # no tfevents in the fake teacher dir
    log.write_text(_fake_log(RL, ck=str(teacher), it0=100, iters=12, spec=spec, num_envs=191))
    assert not {r["id"]: r for r in DC.checks(**kw)[1]}["D2"]["ok"]
    log.write_text(_fake_log(RL, ck=str(teacher), it0=100, iters=12, n_blocks=5, spec=spec, num_envs=192))
    assert DC.checks(**kw)[0] == DC.RC_WAIT


# ##############################################################################################################
# The campaign's GATE code: ``rsd_lib.py`` (rows, stamp checks, reads, verdicts) and ``rsd_gate.py`` (the row queue),
# CHARTER sections 3.3-3.6. A wrong rule here is silent (a launch error read as a reading, a near-line cell rescued by
# es3, an EXTEND that fires when a PASS is impossible), so each is pinned on synthetic records / fake stamps; the
# recorded-stamp replays (Task 4's selection and F19's zero-kg verdicts) run where the main checkout's stamps exist.
# ##############################################################################################################
@pytest.fixture(scope="module")
def gate(tmp_path_factory):
    np = pytest.importorskip("numpy")  # noqa: F841
    base = tmp_path_factory.mktemp("rsd_gate")
    saved = {k: os.environ.get(k) for k in ("RSD_RESULTS", "RSD_TP_OUT")}
    os.environ["RSD_RESULTS"] = str(base / "results")
    os.environ["RSD_TP_OUT"] = str(base / "out")
    sys.path.insert(0, str(SCRIPTS))
    try:
        for n in ("rsd_lib", "rsd_gate"):
            sys.modules.pop(n, None)
        Lb = _load("rsd_lib", SCRIPTS / "rsd_lib.py")
        G = _load("rsd_gate", SCRIPTS / "rsd_gate.py")
    finally:
        sys.path.remove(str(SCRIPTS))
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return SimpleNamespace(L=Lb, G=G, base=base)


@pytest.fixture()
def gl(gate, tmp_path, monkeypatch):
    """The gate modules with a fresh results dir / output root per test."""
    L, G = gate.L, gate.G
    monkeypatch.setattr(L, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(L, "TP_OUT", tmp_path / "out")
    monkeypatch.setattr(G, "LOG", tmp_path / "results" / "rsd_gate.log")
    monkeypatch.setattr(G, "NOTIFY", tmp_path / "results" / "notify.jsonl")
    monkeypatch.setattr(G, "STOP_FILE", tmp_path / "STOP_QUEUE")
    monkeypatch.setattr(G, "STOP_REQUESTED", [])
    return gate


def _rec(lines: dict, es=1, n=None, stamps=None):
    """A synthetic cell-seed record: {key: (value, line, direction)}."""
    out = {}
    for k, (v, line, d) in lines.items():
        ok = v >= line - 1e-12 if d == ">=" else v <= line + 1e-12
        out[k] = {"value": v, "line": line, "direction": d, "pass": ok}
    return {"es": es, "lines": out, "n_lines": 18, "pass": all(x["pass"] for x in out.values()),
            "failed": sorted(k for k, x in out.items() if not x["pass"]), "problems": [],
            "extras": {"n": n or {"fwd": 100, "side": 200, "back": 100, "diag": 100, "turn": 200, "move_turn": 200}},
            "stamps": stamps or {"suite": f"/s{es}", "field_light": f"/f{es}"}}


GOOD = {"field_light_completion": (0.95, 0.85, ">="), "survival.diag": (1.0, 0.9, ">="),
        "roll_rms.suite": (0.05, 0.1, "<="), "tracking.diag": (0.05, 0.25, "<="), "h_meas.suite": (0.9, 0.85, ">=")}


def _with(**kw):
    d = dict(GOOD)
    for k, v in kw.items():
        key = k.replace("__", ".")
        d[key] = (v, d[key][1], d[key][2])
    return d


def test_gate_layout_paths_and_sweep_loader(gl, monkeypatch):
    L = gl.L
    assert L.W == REPO and L.CAMPAIGN == CD
    assert L.GATE_ROW.is_file() and L.SPEC_PATH.is_file() and L.TH_PATH.name == "thresholds_v1_2.json"
    for n in ("rsd_sweep_cells", "rsd_sweep_stress_read"):
        sys.modules.pop(n, None)
    L._MODS.pop("sweep", None)
    monkeypatch.setenv("DRS_TP_OUT", "/bogus/drs")
    monkeypatch.setenv("DRS_RESULTS", "/bogus/drs_results")
    had_cells = "cells" in sys.modules
    sw = L.sweep()
    assert str(sw.C.TP_OUT) == "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/dr_stress"
    assert sw.SR.RESULTS == L.SWEEP / "results"
    assert os.environ["DRS_TP_OUT"] == "/bogus/drs"            # restored after the load
    assert ("cells" in sys.modules) == had_cells and "run_sweep" not in sys.modules
    assert not list(CD.rglob("__pycache__"))


def test_gate_names_rows_and_phase_counts(gl):
    L, G = gl.L, gl.G
    T = L.placeholder_policy("TDR", "teacher")
    S = L.placeholder_policy("PICK", "student")
    assert L.cell_name(T, L.X07) == f"RSDT_{T['sha8']}_T01_rod_speed_0p7_L090"
    assert L.cell_name(S, L.NOM["L135"], L.RELAUNCH) == f"RSDS_{S['sha8']}_T00_nominal_L135_A_RL"
    assert L.sel_name(S, 90) == f"RSDSEL_{S['sha8']}_L090" and L.sel_name(T, None) == f"RSDSELT_{T['sha8']}_L000"
    rows = G.g1_rows("TDR", dry=True)
    assert len(rows) == 39 and rows[0]["cell"] == L.X07 and rows[0]["es"] == 1
    assert sum(1 for r in rows if r["kind"] == "sel") == 7
    assert len(G.tctl_rows("TCTL", dry=True)) == 12
    lv = {r["cell"]: (r["level_kg"], r["rod_s"]) for r in rows if r["kind"] == "cell"}
    assert lv[L.X07] == (90, 0.7) and lv[L.X135_L000] == (None, 1.35) and lv[L.NOM["L135"]] == (135, 1.0)
    assert Path(next(r for r in rows if r["cell"] == L.NOM["L135"])["level_file"]).name == "L135.env"
    sr = L.sel_row(T, "side22@1")
    assert sr["strip"] == L.refused_keys() and "--policy-role" in sr["args"] and sr["args"][-1] == "teacher"
    su = L.sel_row(S, "suite@2", 90)
    assert su["args"][-2:] == ["--trained-categories", "stand,fwd,side,back,diag,turn,move_turn"]
    assert su["level_kg"] == 90 and Path(su["level_file"]).name == "L090.env"
    for r in rows + [sr, su]:
        assert re.match(r"^[A-Za-z0-9_]+$", r["name"])
    if L.TSTAR_CKPT.is_file():
        assert len(G.refarm_rows()) == 27
    if L.C28000_CKPT.is_file():
        assert len(G.g2_rows("PICK", dry=True)) == 54


def test_gate_policy_registry_refuses_a_rebind(gl, tmp_path):
    L = gl.L
    a = tmp_path / "ta" / "model_100.pt"
    b = tmp_path / "tb" / "model_100.pt"
    for p, body in ((a, b"A"), (b, b"B")):
        p.parent.mkdir(parents=True)
        p.write_bytes(body)
    p1 = L.policy_add("TDR", str(a), "teacher")
    assert p1["sha8"] == L.sha256_file(a)[:8] and p1["iteration"] == 100
    assert L.policy_add("TDR", str(a), "teacher")["sha256"] == p1["sha256"]      # idempotent
    with pytest.raises(L.PolicyError, match="refusing to rebind"):
        L.policy_add("TDR", str(b), "teacher")
    with pytest.raises(L.PolicyError, match="agent.yaml"):
        L.policy_add("STU", str(b), "student")
    with pytest.raises(L.PolicyError, match="name"):
        L.policy_add("bad-name", str(b), "teacher")
    with pytest.raises(L.PolicyError, match="pre-bound"):
        L.policy_add("TSTAR", str(b), "teacher")
    a.write_bytes(b"changed")
    with pytest.raises(L.PolicyError, match="now hashes"):
        L.policy("TDR")


def test_gate_dose_check(gl, tmp_path):
    import numpy as np
    L = gl.L
    st = tmp_path / "stamp"
    (st / "raw").mkdir(parents=True)
    r = np.zeros((50, 12), np.float32)
    r[10, :6], r[20, 6:] = -0.7 * 0.028, 0.7 * 0.033
    np.savez(st / "raw" / "episode_00.npz", rod_speed=r)
    assert L.dose_problems(st, 0.7, "x") == []
    assert len(L.dose_problems(st, 1.0, "x")) == 2 and "dose hip" in L.dose_problems(st, 1.0, "x")[0]
    r[30, 0] = 0.7 * 0.028 + 1e-5
    np.savez(st / "raw" / "episode_01.npz", rod_speed=r)
    assert any("hip" in p for p in L.dose_problems(st, 0.7, "x"))


def _fake_stamp(L, row, root, *, env=None, dose_s=None, role=None, seed=None, pin=None):
    import numpy as np
    st = Path(root) / "lr" / row["name"] / row["scenario"] / f"seed{row['es']:03d}" / "2026-11-08_00-00-00"
    (st / "raw").mkdir(parents=True)
    s = row["rod_s"] if dose_s is None else dose_s
    r = np.zeros((5, 12), np.float32)
    r[1, :6], r[2, 6:] = s * 0.028, s * 0.033
    np.savez(st / "raw" / "episode_00.npz", rod_speed=r)
    meta = {"env_vars": env if env is not None else L.expect_env(row), "checkpoint_sha256": row["ckpt_sha"],
            "task": L.TASK, "role": role or row["role"], "env_overrides": {"seed": row["es"] if seed is None else seed},
            "scenario_id": row["scenario"]}
    if row["trained"] is not None:
        meta["trained_categories"] = row["trained"]
    if row["role"] == "student":
        meta["student"] = {"zero_depth": False}
    (st / "run_meta.json").write_text(json.dumps(meta))
    (st / "scenario_metrics.json").write_text("{}")
    (st / "summary.md").write_text("x")
    pin = pin or {"code_sha": "c" * 64, "eval_sha": L.EVAL_SHA}
    (st / "code_state.json").write_text(json.dumps({**pin, "dirty": "False"}))
    (st / "tp_row.json").write_text(json.dumps({
        "by": "TP gate_row.sh", "name": row["name"], "checkpoint": row["ckpt"], "env_file": row["env_file"],
        "level_file": row["level_file"], "manifest": row["manifest"], "scenario": row["scenario"],
        "env_seed": row["es"], "stripped_keys": L.expected_stripped(row), "args": row["args"], "rc": 0}))
    return st


def test_gate_row_checks_on_a_fake_stamp(gl, tmp_path):
    L = gl.L
    T = L.placeholder_policy("TDR", "teacher")
    pin = {"code_sha": "c" * 64, "eval_sha": L.EVAL_SHA}
    row = L.sel_row(T, "field_light@1")
    st = _fake_stamp(L, row, tmp_path / "a")
    assert L.row_problems(st, row, pin) == []
    assert any("code_state code_sha" in p for p in L.row_problems(st, row, {**pin, "code_sha": "d" * 64}))
    bad = {**L.expect_env(row), "KRABBY_DR_ROD_SPEED": "0.6:1.35"}
    st2 = _fake_stamp(L, row, tmp_path / "b", env=bad)
    probs = L.row_problems(st2, row, pin)
    assert any("KRABBY_DR_" in p for p in probs) and any("env_vars differ" in p for p in probs)
    assert any("env seed" in p for p in L.row_problems(_fake_stamp(L, row, tmp_path / "c", seed=2), row, pin))
    assert any("run_meta.role" in p for p in L.row_problems(_fake_stamp(L, row, tmp_path / "d", role="student"),
                                                            row, pin))
    assert any("dose" in p for p in L.row_problems(_fake_stamp(L, row, tmp_path / "e", dose_s=0.7), row, pin))
    lv = L.sel_row(T, "field_light@1", 90)
    assert any("payload" in p for p in L.row_problems(_fake_stamp(L, lv, tmp_path / "f"), lv, pin))


def test_gate_near_line_rules(gl):
    L = gl.L
    r1, r2 = _rec(_with(field_light_completion=0.84)), _rec(_with(), es=2)
    assert L.near_line([(1, "field_light_completion")], {1: r1, 2: r2}, False)["near"]
    r1 = _rec(_with(field_light_completion=0.83))
    assert not L.near_line([(1, "field_light_completion")], {1: r1, 2: r2}, False)["near"]
    r1 = _rec(_with(survival__diag=0.89))                                    # 89 / 100: one episode
    assert L.near_line([(1, "survival.diag")], {1: r1, 2: r2}, False)["near"]
    r1 = _rec(_with(survival__diag=0.88))
    assert not L.near_line([(1, "survival.diag")], {1: r1, 2: r2}, False)["near"]
    r1, r2b = _rec(_with(roll_rms__suite=0.102)), _rec(_with(roll_rms__suite=0.102), es=2)
    assert L.near_line([(1, "roll_rms.suite")], {1: r1, 2: r2}, False)["near"]
    assert not L.near_line([(1, "roll_rms.suite"), (2, "roll_rms.suite")], {1: r1, 2: r2b}, False)["near"]
    r1 = _rec(_with(field_light_completion=0.84, survival__diag=0.89))      # two failing lines: a clear failure
    assert not L.near_line([(1, "field_light_completion"), (1, "survival.diag")], {1: r1, 2: r2}, False)["near"]
    assert not L.near_line([(1, "tracking.diag")], {1: _rec(_with(tracking__diag=0.26)), 2: r2}, False)["near"]


def test_gate_cell_eval_l135_shared_limit_and_es3(gl, monkeypatch):
    L = gl.L
    ref = {1: _rec(_with(h_meas__suite=0.80)), 2: _rec(_with(), es=2)}
    recs = {1: _rec(_with(field_light_completion=0.5, h_meas__suite=0.80)), 2: _rec(_with(), es=2)}
    e = L.cell_eval(L.NOM["L135"], recs, ref, carve=False, l135=True, its=0)
    assert e["status"] == "pass" and e["seeds"][1]["shared_limit"] == ["h_meas.suite"]
    assert e["seeds"][1]["excluded"] == ["field_light_completion"]
    e = L.cell_eval(L.NOM["L090"], recs, ref, carve=False, l135=False, its=0)
    assert e["status"] == "fail"
    recs = {1: _rec(_with(field_light_completion=0.84)), 2: _rec(_with(), es=2)}
    assert L.cell_eval(L.X08, recs, None, carve=False, l135=False, its=0)["status"] == "es3_due"
    rate = {"v": 0.86}
    monkeypatch.setattr(L, "thm", lambda: SimpleNamespace(field_completion=lambda fl: {"rate": rate["v"]}))
    recs[3] = _rec(_with(field_light_completion=0.86), es=3)                 # 2 of 3 seeds pass, pooled 0.86
    assert L.cell_eval(L.X08, recs, None, carve=False, l135=False, its=0)["status"] == "pass"
    rate["v"] = 0.84
    assert L.cell_eval(L.X08, recs, None, carve=False, l135=False, its=0)["status"] == "fail"
    recs[2] = _rec(_with(field_light_completion=0.85), es=2)
    recs[3] = _rec(_with(field_light_completion=0.80), es=3)
    rate["v"] = 0.90
    assert L.cell_eval(L.X08, recs, None, carve=False, l135=False, its=0)["status"] == "fail"   # 1 of 3


def _g1(**kw):
    st = {"s_cells": {"x07": (True, [])}, "n_cells": {"n": (True, [])}, "n_pooled_ok": True, "r_ok": True,
          "trig": True, "a_ok": True, "tctl_moved": False, "half_gap": True, "extended": False}
    st.update(kw)
    return st


def test_gate_g1_verdict_table(gl):
    L = gl.L
    d = L.g1_decide
    assert d(_g1())[0] == "PASS"
    assert d(_g1(r_ok=False))[0] == "FAIL-REGRESSION"
    assert d(_g1(n_pooled_ok=False))[0] == "FAIL-REGRESSION"
    assert d(_g1(n_cells={"n": (False, ["h_meas.suite"])}))[0] == "FAIL-REGRESSION"
    assert d(_g1(a_ok=False))[0] == "UNATTRIBUTED"
    assert d(_g1(a_ok=None))[0] == "TCTL-DUE"
    s_cs = {"x07": (False, ["field_light_completion", "survival.diag"])}
    assert d(_g1(s_cells=s_cs))[0] == "EXTEND"
    assert d(_g1(s_cells={"x07": (False, ["roll_rms.suite"])}))[0] == "FAIL-STRESS"      # roll/pitch-only miss
    v, n = d(_g1(s_cells=s_cs, tctl_moved=True))
    assert v == "FAIL-STRESS" and "no-DR continuation moved x0.7" in n[0]
    assert d(_g1(s_cells=s_cs, half_gap=False))[0] == "FAIL-STRESS"
    assert d(_g1(s_cells=s_cs, a_ok=False))[0] == "FAIL-STRESS"
    v, n = d(_g1(s_cells=s_cs, trig=False, a_ok=None))
    assert v == "FAIL-STRESS" and "DR null" in n[0]
    assert d(_g1(s_cells=s_cs, extended=True))[0] == "FAIL-STRESS"                     # EXTEND fires once
    assert d(_g1(extended=True))[0] == "PASS"
    assert d(_g1(extended=True, tctl_moved=True))[0] == "FAIL-STRESS"                   # post-EXTEND PASS rule


def test_gate_resolve_pending_es3_and_cap(gl):
    L = gl.L
    cells = {"x07": {"status": "pass", "fail_keys": []}, "x08": {"status": "es3_due",
                                                                 "fail_keys": ["field_light_completion"]},
             "n": {"status": "pass", "fail_keys": []}}
    base = {**_g1(), "s_cells": ["x07", "x08"], "n_cells": ["n"]}
    r = L.resolve(cells, L.g1_decide, base, cap_left=4)
    assert r["verdict"] == "PENDING-ES3" and r["pending"] == ["x08"] and set(r["possible"]) == {"PASS", "EXTEND"}
    assert L.resolve(cells, L.g1_decide, base, cap_left=0)["verdict"] == "UNRESOLVED"
    assert L.resolve(cells, L.g1_decide, {**base, "r_ok": False}, cap_left=0)["verdict"] == "FAIL-REGRESSION"


def test_gate_g2_verdicts(gl):
    L = gl.L
    st = {"s_cells": {"a": (True, [])}, "n_cells": {"n": (True, [])}, "n_pooled": {"n": "pass"}}
    assert L.g2_decide(st)[0] == "PASS"
    assert L.g2_decide({**st, "n_pooled": {"n": "ambiguous"}})[0] == "RELAUNCH-DUE"
    assert L.g2_decide({**st, "n_pooled": {"n": "fail"}})[0] == "FAIL"
    assert L.g2_decide({**st, "s_cells": {"a": (False, ["survival.diag"])}})[0] == "FAIL"


def test_gate_runner_skips_valid_reruns_invalid_and_stops_after_two_failures(gl, monkeypatch, tmp_path):
    L, G = gl.L, gl.G
    row = L.sel_row(L.placeholder_policy("TDR", "teacher"), "side22@1")
    state = {"stamp": None, "probs": [], "launches": 0, "rc": 0}
    monkeypatch.setattr(L, "check_tree", lambda: {"code_sha": "c", "eval_sha": "e"})
    monkeypatch.setattr(L, "launch_pin", lambda: {"code_sha": "c", "eval_sha": "e"})
    monkeypatch.setattr(L, "disk_problems", lambda *a, **k: [])
    monkeypatch.setattr(L, "row_stamp", lambda r, root=None: state["stamp"])
    monkeypatch.setattr(L, "row_problems", lambda st, r, pin, **k: list(state["probs"]))

    def launch(self, r, attempt):
        state["launches"] += 1
        state["stamp"] = tmp_path / f"st{state['launches']}"
        return state["rc"]
    monkeypatch.setattr(G.Runner, "launch", launch)
    R = G.Runner(False, "t")
    assert R.run(row) == tmp_path / "st1" and state["launches"] == 1         # no stamp -> one launch
    assert R.run(row) == tmp_path / "st1" and state["launches"] == 1         # valid -> SKIP
    log = G.LOG.read_text()
    assert "RSD ROW" in log and "RSD SKIP" in log
    state["probs"] = ["dose hip"]
    with pytest.raises(L.Stop, match="row_failed_twice"):
        R.run(row)
    assert state["launches"] == 3 and "RSD RERUN" in G.LOG.read_text()
    state["probs"] = []
    G.STOP_FILE.write_text("")
    state["stamp"] = None
    with pytest.raises(L.Stop, match="STOP_QUEUE"):
        R.run(row)
    G.STOP_FILE.unlink()
    G.STOP_REQUESTED.append("SIGTERM")
    with pytest.raises(L.Stop, match="SIGTERM"):
        R.run(row)


def test_gate_pin_and_tree_refusals(gl, monkeypatch):
    L = gl.L
    cs = {"code_sha": "a" * 64, "eval_sha": L.EVAL_SHA, "head": "h", "dirty": ["x.py"]}
    monkeypatch.setattr(L, "code_state", lambda: dict(cs))
    with pytest.raises(L.Stop, match="dirty_tree"):
        L.write_pin()
    cs["dirty"] = []
    cs["eval_sha"] = "b" * 64
    with pytest.raises(L.Stop, match="eval_sha"):
        L.write_pin()
    cs["eval_sha"] = L.EVAL_SHA
    with pytest.raises(L.Stop, match="no pin"):
        L.check_tree()
    pin = L.write_pin()
    assert pin["code_sha"] == "a" * 64 and pin["head"] == "h" and L.check_tree() == pin
    cs["code_sha"] = "z" * 64
    with pytest.raises(L.Stop, match="code_changed_since_pin"):
        L.check_tree()
    with pytest.raises(L.Stop, match="different pin"):
        L.write_pin()


def test_gate_disk_guard(gl, monkeypatch):
    L = gl.L
    monkeypatch.setattr(L, "free_gb", lambda p: 29.0)
    monkeypatch.setattr(L, "ddc_gb", lambda p=None: 79.0)
    bad = L.disk_problems()
    assert len(bad) == 2 and "29.0 GB < 30" in bad[0] and "79 GB > 78" in bad[1]
    monkeypatch.setattr(L, "free_gb", lambda p: 175.0)
    monkeypatch.setattr(L, "ddc_gb", lambda p=None: 72.0)
    assert L.disk_problems() == []


def test_gate_env_files_are_the_sweeps_and_the_strip_is_a_noop(gl, monkeypatch, tmp_path):
    L = gl.L
    monkeypatch.setattr(L, "ENVS", tmp_path / "envs")
    for cid in (L.NOM["L000"], L.NOM["L090"], L.X07, L.X08):
        L.prepare_cell(cid)
        for fam in ("suite", "field_light"):
            assert L.env_file(cid, fam).read_text() == (L.SWEEP / "envs" / f"{cid}__{fam}.env").read_text()
    L.prepare_cell(L.NOM["L135"])
    assert L.env_file(L.NOM["L135"], "suite").read_text() == L.env_file(L.NOM["L000"], "suite").read_text()
    assert "KRABBY_STRESS_ROD_SPEED=0.7\n" in L.env_file(L.X07, "suite").read_text()
    assert L.strip_noop_problems() == []


_T4_LR = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/task4/F19/lr")


@pytest.mark.skipif(not (_T4_LR / "T4F19_sel_tstar").is_dir() or not (_T4_LR / "T4F19_p3a").is_dir(),
                    reason="the main checkout's Task 4 F19 stamps are not on this machine")
def test_gate_replay_reproduces_task4_selection_and_the_f19_zero_kg_verdicts(gl):
    L = gl.L
    zk = L.replay_zero_kg(dose=False)
    assert zk["exact"], zk
    assert zk["stages"]["p3a"]["mine"]["verdict"] == "FAIL" and zk["stages"]["p3b"]["mine"]["verdict"] == "PASS"
    assert abs(zk["stages"]["p3a"]["mine"]["max_margin_used"] - 2.7099) < 1e-4
    assert abs(zk["stages"]["p3b"]["mine"]["max_margin_used"] - 0.6620) < 1e-4
    sel = L.replay_sel(dose=False)
    assert sel["exact"] and sel["checks_clean"], {k: v for k, v in sel["diffs"].items() if v}
    assert sel["pick"]["cand"] == "c28000" and sel["failed_blocks"] == []


# ##############################################################################################################
# Campaign-wide pins (the ``test_ct_*`` tests). Fake stamps / synthetic records only: stamp launch checks, the
# reference and stage guards, the bridge rule, every verdict-table branch of G1 and G2 driven through ``g1_read`` /
# ``g2_read`` with the stamp readers replaced by a synthetic world, the es3 and carve-out rules, the selection's
# candidate numbering and 90-kg level rows, and the launchers' shared helpers. The one place a fake stands in for a
# real function is ``paired_completion`` in the verdict-table worlds (a fixed margin of 0.05, the floor of
# max(0.05, 2 SE)); ``test_ct_paired_completion_*`` runs the real function on fake episodes.
# ##############################################################################################################
import copy  # noqa: E402
import shutil  # noqa: E402


@pytest.fixture()
def cx(gl, tmp_path, monkeypatch):
    """``gl`` with the campaign's composed env files in a tmp dir."""
    monkeypatch.setattr(gl.L, "ENVS", tmp_path / "envs")
    return gl


def _ck(tmp_path, name, it, body=None, student=False):
    p = tmp_path / "ck" / name / f"model_{it}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body or f"{name}-{it}".encode())
    if student:
        (p.parent / "params").mkdir(exist_ok=True)
        (p.parent / "params" / "agent.yaml").write_text("seed: 19\n")
    return p


def _edit(path: Path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def _full_stamp(L, row, root, **kw):
    """``_fake_stamp`` plus the payload readback a level row needs."""
    st = _fake_stamp(L, row, root, **kw)
    if row["level_kg"] is not None:
        kg = float(row["level_kg"])
        _edit(st / "run_meta.json", lambda m: m.update(payload={
            "payload_kg": [kg, kg], "kg_range": [kg, kg], "zero_frac": 0.0, "com_xy": 0.0, "com_z": 0.309,
            "payload_com_m": [[0.0, 0.0, 0.309], [0.0, 0.0, 0.309]]}))
    return st


# ============================================================================== names
def test_ct_layout_no_tracked_log_or_checkpoint(gl):
    assert not list(CD.rglob("*.pt")) and not list(CD.rglob("__init__.py"))
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", str(CD)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no git here")
    assert [f for f in r.stdout.split("\n") if f.endswith((".log", ".pt", ".pyc", "__init__.py"))] == []
    for bad in ("cells", "stress_read", "run_sweep", "tp_lib"):
        assert not (SCRIPTS / f"{bad}.py").exists()


def test_ct_gate_names_are_legal_unique_and_carry_the_sha8(cx, tmp_path):
    L, G = cx.L, cx.G
    L.policy_add("TDR", _ck(tmp_path, "tdr", 28991), "teacher")
    L.policy_add("TCTL", _ck(tmp_path, "tctl", 28991), "teacher")
    pick = L.policy_add("PICK", _ck(tmp_path, "pick", 29991, student=True), "student")
    cands = [L.ephemeral_policy(_ck(tmp_path, f"c{i}", i, student=True), "student", f"c{i}")
             for i in (29991, 30991, 31991, 32991, 33991)]
    T = L.policy("TDR")
    groups = {"g1": G.g1_rows("TDR"), "tctl": G.tctl_rows("TCTL"),
              "closing": L.cell_rows(T, L.CLOSING_CELL, 1),
              "es3": L.cell_rows(T, L.X08, 3),
              "relaunch": L.cell_rows(T, L.NOM["L090"], 1, L.RELAUNCH) + L.cell_rows(pick, L.NOM["L090"], 1, L.RELAUNCH),
              "g2": G.cellseed_rows(pick, L.S_CELLS + [L.NOM["L135"]] + L.G2_WATCH)}
    for c in cands:
        groups[f"sel_{c['name']}"] = list(L.student_sel_rows(c, "full").values()) + list(
            L.student_sel_rows(c, "full_level").values())
    if L.TSTAR_CKPT.is_file():
        groups["refarm"] = G.refarm_rows()
    allrows = [r for rows in groups.values() for r in rows]
    for gname, rows in groups.items():
        keys = [(r["name"], r["scenario"], r["es"]) for r in rows]
        assert len(keys) == len(set(keys)), f"{gname}: two rows share one stamp directory"
    ident_names: dict = {}
    for r in allrows:
        m = re.fullmatch(r"(RSDSELT|RSDSEL|RSDT|RSDS)_([0-9a-f]{8})_([A-Za-z0-9_]+)", r["name"])
        assert m, r["name"]
        assert m.group(1) == {("teacher", "cell"): "RSDT", ("student", "cell"): "RSDS",
                              ("teacher", "sel"): "RSDSELT", ("student", "sel"): "RSDSEL"}[(r["role"], r["kind"])]
        assert m.group(2) == r["ckpt_sha"][:8], "the name must carry the checkpoint's own sha8"
        assert not r["name"].startswith(("DRS_", "DRST_", "T4F19_"))        # never a sweep / Task 4 name
        ident = (r["ckpt_sha"], r["kind"], r["cell"], r["suffix"], r["level_kg"])
        ident_names.setdefault(ident, set()).add(r["name"])
    assert all(len(v) == 1 for v in ident_names.values())                    # one name per (checkpoint, cell, launch)
    flat = [next(iter(v)) for v in ident_names.values()]
    assert len(flat) == len(set(flat))                                       # ... and one cell per name
    assert len({p["sha8"] for p in [T, L.policy("TCTL"), pick, *cands]}) == 8


def test_ct_policy_registry_refuses_every_rebind(cx, tmp_path, monkeypatch):
    L = cx.L
    a, b = _ck(tmp_path, "ta", 100, b"A"), _ck(tmp_path, "tb", 100, b"B")
    L.policy_add("TDR", a, "teacher")
    with pytest.raises(L.PolicyError, match="refusing to rebind"):
        L.policy_add("TDR", b, "teacher")
    a_student = _ck(tmp_path, "ta", 100, b"A", student=True)                       # the same bytes, as a student
    with pytest.raises(L.PolicyError, match="role"):
        L.policy_add("TDR", a_student, "student")
    assert L.load_registry()["policies"]["TDR"]["sha256"] == L.sha256_file(a)        # the refusal wrote nothing
    # a pre-bound policy: the file must hash to the recorded sha, and only its own file may bind to the name
    monkeypatch.setattr(L, "TSTAR_CKPT", a)
    with pytest.raises(L.PolicyError, match="does not start"):
        L.policy_add("TSTAR", a, "teacher")
    with pytest.raises(L.PolicyError, match="pre-bound"):
        L.policy_add("TSTAR", b, "teacher")


# ============================================================================== stamp launch checks
def _x07_row(L, T, es=1, fam=0):
    L.prepare_cell(L.X07)
    return L.cell_rows(T, L.X07, es)[fam]


STAMP_DEFECTS = [
    ("wrong checkpoint sha", lambda st: _edit(st / "run_meta.json", lambda m: m.update(checkpoint_sha256="0" * 64)),
     "checkpoint_sha256"),
    ("wrong role", lambda st: _edit(st / "run_meta.json", lambda m: m.update(role="student")), "run_meta.role"),
    ("wrong task", lambda st: _edit(st / "run_meta.json", lambda m: m.update(task="Isaac-Crab-Hex-Omni-Teacher-v0")),
     "task"),
    ("stray DR key", lambda st: _edit(st / "run_meta.json",
                                      lambda m: m["env_vars"].update(KRABBY_DR_ROD_SPEED="0.6:1.35")),
     "KRABBY_DR_"),
    ("missing stress key", lambda st: _edit(st / "run_meta.json",
                                            lambda m: m["env_vars"].pop("KRABBY_STRESS_ROD_SPEED")),
     "env_vars differ"),
    ("wrong stress value", lambda st: _edit(st / "run_meta.json",
                                            lambda m: m["env_vars"].update(KRABBY_STRESS_ROD_SPEED="0.8")),
     "env_vars differ"),
    ("dirty tree", lambda st: _edit(st / "code_state.json", lambda m: m.update(dirty="True")), "dirty"),
    ("code_sha off the pin", lambda st: _edit(st / "code_state.json", lambda m: m.update(code_sha="d" * 64)),
     "code_sha"),
    ("eval_sha off the pin", lambda st: _edit(st / "code_state.json", lambda m: m.update(eval_sha="e" * 64)),
     "eval_sha"),
    ("no code_state", lambda st: (st / "code_state.json").unlink(), "code_state"),
    ("no tp_row", lambda st: (st / "tp_row.json").unlink(), "tp_row"),
    ("tp_row checkpoint", lambda st: _edit(st / "tp_row.json", lambda m: m.update(checkpoint="/other/model_1.pt")),
     "tp_row.checkpoint"),
    ("tp_row rc", lambda st: _edit(st / "tp_row.json", lambda m: m.update(rc=8)), "tp_row.rc"),
    ("seed mismatch", lambda st: _edit(st / "run_meta.json", lambda m: m["env_overrides"].update(seed=2)),
     "env seed"),
    ("scenario mismatch", lambda st: _edit(st / "run_meta.json", lambda m: m.update(scenario_id="other")),
     "scenario_id"),
    ("student block on a teacher row", lambda st: _edit(st / "run_meta.json",
                                                        lambda m: m.update(student={"zero_depth": False})),
     "student present"),
    ("payload off the level", lambda st: _edit(st / "run_meta.json", lambda m: m["payload"].update(
        payload_kg=[135.0, 135.0], kg_range=[135.0, 135.0])), "payload"),
    ("no raw", lambda st: shutil.rmtree(st / "raw"), "raw"),
]


def test_ct_clean_stamp_has_no_problem_and_each_defect_is_flagged(cx, tmp_path):
    L = cx.L
    T = L.placeholder_policy("TDR", "teacher")
    row = _x07_row(L, T)
    pin = {"code_sha": "c" * 64, "eval_sha": L.EVAL_SHA}
    clean = _full_stamp(L, row, tmp_path / "clean")
    assert L.row_problems(clean, row, pin) == []
    for i, (what, mutate, needle) in enumerate(STAMP_DEFECTS):
        st = _full_stamp(L, row, tmp_path / f"d{i}")
        mutate(st)
        probs = L.row_problems(st, row, pin)
        assert probs, what
        assert any(needle in p for p in probs), (what, probs)
    assert any("no pin" in p for p in L.row_problems(clean, row, None))


def test_ct_dose_mismatch_and_nominal_dose(cx, tmp_path):
    L = cx.L
    T = L.placeholder_policy("TDR", "teacher")
    pin = {"code_sha": "c" * 64, "eval_sha": L.EVAL_SHA}
    row = _x07_row(L, T)
    assert row["rod_s"] == 0.7
    probs = L.row_problems(_full_stamp(L, row, tmp_path / "a", dose_s=0.8), row, pin)
    assert any("dose hip" in p for p in probs) and any("dose knee" in p for p in probs)
    assert L.row_problems(_full_stamp(L, row, tmp_path / "b", dose_s=0.7 + 1e-4), row, pin)       # 2.8e-6 > 1e-6 m/s
    assert L.row_problems(_full_stamp(L, row, tmp_path / "b2", dose_s=0.7 + 1e-5), row, pin) == []  # 2.8e-7: inside
    nom = L.cell_rows(T, L.NOM["L090"], 1)[0]
    L.prepare_cell(L.NOM["L090"])
    assert nom["rod_s"] == 1.0 and L.row_problems(_full_stamp(L, nom, tmp_path / "c"), nom, pin) == []
    assert any("dose" in p for p in L.row_problems(_full_stamp(L, nom, tmp_path / "d", dose_s=0.7), nom, pin))
    st = _full_stamp(L, row, tmp_path / "e")
    shutil.rmtree(st / "raw")
    (st / "raw").mkdir()
    assert any("dose unreadable" in p for p in L.dose_problems(st, 0.7, "x"))


def test_ct_student_stamp_zero_depth_polarity_and_the_newest_complete_stamp_wins(cx, tmp_path):
    L = cx.L
    P = L.placeholder_policy("PICK", "student")
    pin = {"code_sha": "c" * 64, "eval_sha": L.EVAL_SHA}
    L.prepare_cell(L.NOM["L000"])
    row = L.cell_rows(P, L.NOM["L000"], 1)[0]
    st = _full_stamp(L, row, tmp_path / "s")
    assert L.row_problems(st, row, pin) == []
    _edit(st / "run_meta.json", lambda m: m["student"].update(zero_depth=True))
    assert any("zero_depth" in p for p in L.row_problems(st, row, pin))
    _edit(st / "run_meta.json", lambda m: m.pop("student"))
    assert any("zero_depth" in p for p in L.row_problems(st, row, pin))
    # row_stamp: the newest stamp that has tp_row.json (written last); an unfinished newer one is ignored
    old = _full_stamp(L, row, tmp_path / "r")
    new = old.parent / "2026-11-09_00-00-00"
    shutil.copytree(old, new)
    assert L.row_stamp(row, tmp_path / "r") == new
    (new / "tp_row.json").unlink()
    assert L.row_stamp(row, tmp_path / "r") == old


# ============================================================================== references and stage guards
def test_ct_recorded_reference_that_disagrees_with_a_reread_is_refused(cx, monkeypatch, tmp_path):
    L, G = cx.L, cx.G
    mine = {**_rec({f"k{i:02d}": (0.5 + i / 100, 0.1, ">=") for i in range(18)}), "line_pass": True, "problems": [],
            "survival_all": True, "lenient_feasible": False}
    rec = {**copy.deepcopy(mine), "pass": True}
    rec.pop("line_pass"), rec.pop("problems")
    assert L._rec_diff(mine, rec) == []
    moved = copy.deepcopy(rec)
    moved["lines"]["k03"]["value"] += 1e-9                                          # exact equality, no tolerance
    assert any("k03" in d for d in L._rec_diff(mine, moved))
    flipped = {**copy.deepcopy(rec), "pass": False}
    assert any(d.startswith("pass:") for d in L._rec_diff(mine, flipped))
    assert any("launch checks flag" in d for d in L._rec_diff({**mine, "problems": ["dose hip"]}, rec))
    assert any("launch error" in d for d in L._rec_diff(mine, {**rec, "launch_error": ["x"]}))
    assert L._rec_diff(None, None) == [] and L._rec_diff(mine, None) and L._rec_diff(None, rec)
    # the replay CLI refuses (rc 3) when any of its three comparisons is not exact or its launch checks are not clean
    def part(exact=True, clean=True):
        return {"exact": exact, "checks_clean": clean, "cellseeds": 1, "closing": {"mine": {}}, "pick": {"cand": "c1"},
                "ranked": [], "failed_blocks": [],
                "stages": {"p3a": {"its_past_p2a": 1, "diffs": [], "mine": {"verdict": "PASS",
                                                                              "max_margin_used": 0.1}}}}
    monkeypatch.setattr(G, "LOG", tmp_path / "log")
    for exact, clean, rc in ((True, True, 0), (False, True, 3), (True, False, 3)):
        monkeypatch.setattr(L, "replay_sweep", lambda dose=True, e=exact, c=clean: part(e, c))
        monkeypatch.setattr(L, "replay_sel", lambda dose=True: part())
        monkeypatch.setattr(L, "replay_zero_kg", lambda dose=True: part())
        a = SimpleNamespace(no_dose=True, out=str(tmp_path / f"replay_{rc}_{exact}_{clean}.json"))
        assert G.ph_replay(a) == rc


def test_ct_stage_guards_refuse_a_stale_or_foreign_record(cx, monkeypatch, tmp_path):
    L, G = cx.L, cx.G
    tdr = L.policy_add("TDR", _ck(tmp_path, "tdr", 28991), "teacher")
    other = L.policy_add("TCTL", _ck(tmp_path, "tctl", 28991), "teacher")
    pick = L.policy_add("PICK", _ck(tmp_path, "pick", 29991, student=True), "student")
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    R = G.Runner(False, "t")
    sel_a = SimpleNamespace(teacher="TDR", dry=False, student_run=str(tmp_path), pick_name="PICK",
                            approve_extra_block=False, dry_teacher_iter=None)
    for g1 in ({}, {"verdict": "EXTEND", "policies": {"tdr": {"sha256": tdr["sha256"]}}},
               {"verdict": "PASS", "policies": {"tdr": {"sha256": other["sha256"]}}}):
        L.write_json(L.RESULTS / "g1.json", g1)
        with pytest.raises(L.Stop, match="G1 PASS only") as e:
            G.ph_sel(sel_a, R)
        assert e.value.code == 3
    # g2 runs only the selection's pick
    monkeypatch.setattr(L, "check_tree", lambda: {"code_sha": "c", "eval_sha": "e"})
    monkeypatch.setattr(L, "disk_problems", lambda *a, **k: [])
    monkeypatch.setattr(L, "strip_noop_problems", lambda: [])
    g2a = SimpleNamespace(cmd="g2", dry=False, policy="PICK")
    for sel in ({}, {"verdict": "HALT"}, {"verdict": "SELECTED", "pick": {"sha256": "0" * 64}}):
        L.write_json(L.RESULTS / "sel.json", sel)
        with pytest.raises(L.Stop, match="not the selection's pick"):
            G.run_phase(g2a)
    # es3 only for a cell the last read left due, and only inside the 4 cell-seed allowance
    es3 = SimpleNamespace(policy="TDR", gate="g1", cell=L.X08, dry=False)
    L.write_json(L.RESULTS / "g1.json", {"es3_due": [L.X07], "es3_used": 0})
    with pytest.raises(L.Stop, match="not in results/g1.json es3_due") as e:
        G.ph_es3(es3, R)
    assert e.value.code == 2
    L.write_json(L.RESULTS / "g1.json", {"es3_due": [L.X08], "es3_used": 4})
    with pytest.raises(L.Stop, match="allowance spent") as e:
        G.ph_es3(es3, R)
    assert e.value.code == 3
    assert not (L.TP_OUT / "lr").exists()                                              # nothing was launched
    # the class-ambiguity relaunch needs the read's relaunch_due and 4 free cell-seeds
    real_policy = L.policy
    monkeypatch.setattr(L, "policy", lambda n: pick if n == "C28000" else real_policy(n))
    rl = SimpleNamespace(policy="PICK", cell=L.NOM["L090"], dry=False)
    L.write_json(L.RESULTS / "g2.json", {"relaunch_due": [], "es3_used": 0})
    with pytest.raises(L.Stop, match="not class-ambiguous"):
        G.ph_g2relaunch(rl, R)
    L.write_json(L.RESULTS / "g2.json", {"relaunch_due": [L.NOM["L090"]], "es3_used": 1})
    with pytest.raises(L.Stop, match="4 cell-seeds"):
        G.ph_g2relaunch(rl, R)


# ============================================================================== the bridge rule
def _l18(**delta):
    d = {f"line{i:02d}": 0.1 * i + 0.123456789 for i in range(18)}
    d.update(delta)
    return {k: (v, 0.0, ">=") for k, v in d.items()}


class _BridgeWorld:
    def __init__(self, L, monkeypatch):
        self.L = L
        self.sweep: dict = {}
        self.mine: dict = {}
        real = L.sweep()
        self.fake_sr = SimpleNamespace(launch_pin=lambda: {"code_sha": "s" * 64, "eval_sha": "e" * 64},
                                       cellseed=lambda name, es, term, level, base, policy=None:
                                       self.sweep.get((name, es)))
        pol = L.placeholder_policy("TSTAR", "teacher")
        monkeypatch.setattr(L, "sweep", lambda: SimpleNamespace(C=real.C, SR=self.fake_sr))
        monkeypatch.setattr(L, "policy", lambda name: pol)
        monkeypatch.setattr(L, "read_cellseed", lambda rows, *, pin, root=None, **k:
                            self.mine.get((rows[0]["name"], rows[0]["es"])))
        self.pol, self.C = pol, real.C

    def seed(self, cid, es, sweep_lines, first=None, relaunch=None, first_problems=(), sweep_error=None):
        L, C = self.L, self.C
        c = L.cell(cid)
        sname = C.gate_name(cid, "teacher")
        rec = _rec(sweep_lines, es=es)
        rec["launch_error"] = sweep_error
        self.sweep[(sname, es)] = rec
        if first is not None:
            r = _rec(first, es=es)
            r["problems"] = list(first_problems)
            r["pass"] = None if first_problems else r["pass"]
            self.mine[(L.cell_name(self.pol, cid), es)] = r
        if relaunch is not None:
            self.mine[(L.cell_name(self.pol, cid, L.RELAUNCH), es)] = _rec(relaunch, es=es)
        return c


def _bridge(L, monkeypatch, plan):
    """plan: {(cid, es): dict(first=, relaunch=, sweep_lines=...)}; unlisted cell-seeds are bitwise equal."""
    w = _BridgeWorld(L, monkeypatch)
    for cid, es in L.BRIDGE_TEACHER:
        kw = {"sweep_lines": _l18(), "first": _l18()}
        kw.update(plan.get((cid, es), {}))
        w.seed(cid, es, **kw)
    return L.bridge_read("TSTAR", pin={})


def test_ct_bridge_rule_is_bitwise_on_18_values_with_one_relaunch(cx, monkeypatch):
    L = cx.L
    key = (L.X07, 1)
    r = _bridge(L, monkeypatch, {})
    assert r["verdict"] == "PASS" and {e["status"] for e in r["cell_seeds"].values()} == {"bitwise_first"}
    # one ulp-sized difference in one of 18 values is not bitwise: relaunch due, then PASS / STOP on the relaunch
    off = _l18(line07=0.1 * 7 + 0.123456789 + 1e-12)
    r = _bridge(L, monkeypatch, {key: {"first": off}})
    assert r["verdict"] == "RELAUNCH_DUE" and r["relaunch_due"] == [f"{L.X07}@es1"]
    assert r["cell_seeds"][f"{L.X07}@es1"]["diff_first"] == {"line07": [0.1 * 7 + 0.123456789 + 1e-12,
                                                                        0.1 * 7 + 0.123456789]}
    r = _bridge(L, monkeypatch, {key: {"first": off, "relaunch": _l18()}})
    assert r["verdict"] == "PASS" and r["cell_seeds"][f"{L.X07}@es1"]["status"] == "bitwise_relaunch"
    r = _bridge(L, monkeypatch, {key: {"first": off, "relaunch": off}})
    assert r["verdict"] == "STOP" and "diagnose on CPU" in r["why"]
    assert r["cell_seeds"][f"{L.X07}@es1"]["status"] == "STOP_diagnose"
    # a 17-value record (a line went missing) never counts as bitwise, even when both sides agree
    short = {k: v for k, v in _l18().items() if k != "line17"}
    r = _bridge(L, monkeypatch, {key: {"sweep_lines": short, "first": short}})
    assert r["cell_seeds"][f"{L.X07}@es1"]["status"] == "relaunch_due" and r["verdict"] == "RELAUNCH_DUE"
    # a launch-check failure on the first launch is not a reading; no stamp yet is not a pass either
    r = _bridge(L, monkeypatch, {key: {"first": _l18(), "first_problems": ["dose hip"]}})
    assert r["cell_seeds"][f"{L.X07}@es1"]["status"] == "invalid_first" and r["verdict"] == "INCOMPLETE"
    r = _bridge(L, monkeypatch, {key: {"first": None}})
    assert r["cell_seeds"][f"{L.X07}@es1"]["status"] == "missing" and r["verdict"] == "INCOMPLETE"
    r = _bridge(L, monkeypatch, {key: {"sweep_error": ["launch error"]}})
    assert r["cell_seeds"][f"{L.X07}@es1"]["status"] == "sweep_unread" and r["verdict"] == "INCOMPLETE"
    assert len(r["cell_seeds"]) == 6


def test_ct_bridge_phase_relaunches_only_the_due_cell_seeds_and_stops_on_a_class_failure(cx, monkeypatch):
    L, G = cx.L, cx.G
    pol = L.placeholder_policy("TSTAR", "teacher")
    monkeypatch.setattr(L, "policy", lambda n: pol)
    monkeypatch.setattr(L, "launch_pin", lambda: {})
    monkeypatch.setattr(G, "LOG", cx.base / "bridge.log")
    due = [f"{L.X07}@es1", f"{L.X08}@es2"]
    reads = iter([{"verdict": "RELAUNCH_DUE", "relaunch_due": due, "cell_seeds": {}},
                  {"verdict": "PASS", "relaunch_due": [], "cell_seeds": {}}])
    monkeypatch.setattr(L, "bridge_read", lambda *a, **k: next(reads))
    launched = []
    R = SimpleNamespace(run_all=lambda rows: launched.extend(rows))
    assert G.ph_bridge(SimpleNamespace(policy="TSTAR", dry=False, relaunch=True), R) == 0
    assert {(r["cell"], r["es"], r["suffix"]) for r in launched} == {(L.X07, 1, "_RL"), (L.X08, 2, "_RL")}
    assert all(r["name"].endswith("_RL") for r in launched) and len(launched) == 4        # suite + field_light each
    # without --relaunch nothing is launched; a STOP read stops the campaign (rc 3)
    launched.clear()
    reads = iter([{"verdict": "RELAUNCH_DUE", "relaunch_due": due, "cell_seeds": {}}])
    G.ph_bridge(SimpleNamespace(policy="TSTAR", dry=False, relaunch=False), R)
    assert launched == []
    reads = iter([{"verdict": "STOP", "why": "diagnose", "relaunch_due": [], "cell_seeds": {}}])
    with pytest.raises(L.Stop) as e:
        G.ph_bridge(SimpleNamespace(policy="TSTAR", dry=False, relaunch=True), R)
    assert e.value.code == 3


# ============================================================================== near-line (es3) definition
def test_ct_near_line_band_edges(cx):
    L = cx.L
    good2 = _rec(_with(), es=2)

    def near(fails, r1, r2=good2, l135=False):
        return L.near_line(fails, {1: r1, 2: r2}, l135)["near"]

    # completion: 84/100 .. 86/100 inclusive, one episode either side of the 85 line
    for v, want in ((0.84, True), (0.85, True), (0.86, True), (0.839, False), (0.861, False), (0.83, False)):
        assert near([(1, "field_light_completion")], _rec(_with(field_light_completion=v))) is want, v
    assert not near([(1, "field_light_completion")], _rec(_with(field_light_completion=0.84)), l135=True)
    # a survival count: within one episode of its line, by the category's own n
    assert near([(1, "survival.diag")], _rec(_with(survival__diag=0.89)))             # 89 / 100 vs 90: one episode
    assert not near([(1, "survival.diag")], _rec(_with(survival__diag=0.885)))        # 88.5: more than one
    r200 = _rec(_with(survival__diag=0.895), n={"diag": 200})                          # 179 / 200 vs 180
    assert near([(1, "survival.diag")], r200)
    assert not near([(1, "survival.diag")], _rec(_with(survival__diag=0.89), n={"diag": 200}))   # 178 / 200
    # roll / pitch RMS: within 0.003 of the line, on ONE seed only
    assert near([(1, "roll_rms.suite")], _rec(_with(roll_rms__suite=0.103)))         # the edge itself is inside
    assert not near([(1, "roll_rms.suite")], _rec(_with(roll_rms__suite=0.1031)))
    p1 = _rec({**GOOD, "pitch_rms.suite": (0.1029, 0.1, "<=")})
    assert near([(1, "pitch_rms.suite")], p1)
    assert not near([(1, "pitch_rms.suite")], _rec({**GOOD, "pitch_rms.suite": (0.1031, 0.1, "<=")}))
    r2b = _rec(_with(roll_rms__suite=0.102), es=2)
    assert not near([(1, "roll_rms.suite"), (2, "roll_rms.suite")], _rec(_with(roll_rms__suite=0.102)), r2b)
    # a completion miss on both seeds is still a (completion) near-line cell; two different lines are not
    assert near([(1, "field_light_completion"), (2, "field_light_completion")],
                _rec(_with(field_light_completion=0.84)), _rec(_with(field_light_completion=0.86), es=2))
    two = _rec(_with(field_light_completion=0.84, roll_rms__suite=0.102))
    assert not near([(1, "field_light_completion"), (1, "roll_rms.suite")], two)
    # a tracking / h_meas / turn line is never es3-eligible, however small the miss
    assert not near([(1, "tracking.diag")], _rec(_with(tracking__diag=0.2501)))
    assert not near([(1, "h_meas.suite")], _rec(_with(h_meas__suite=0.8499)))


def test_ct_es3_is_once_two_of_three_and_pooled_completion(cx, monkeypatch):
    L = cx.L
    rate = {"v": 0.9}
    monkeypatch.setattr(L, "thm", lambda: SimpleNamespace(field_completion=lambda fl: {"rate": rate["v"]}))
    f84 = _with(field_light_completion=0.84)
    s1 = {1: _rec(f84), 2: _rec(_with(), es=2)}
    assert L.cell_eval(L.X08, s1, None, carve=False, l135=False, its=0)["status"] == "es3_due"
    ok3 = {**s1, 3: _rec(_with(), es=3)}
    e = L.cell_eval(L.X08, ok3, None, carve=False, l135=False, its=0)
    assert e["status"] == "pass" and e["es3"]["seeds_passing"] == 2
    rate["v"] = 0.8499
    assert L.cell_eval(L.X08, ok3, None, carve=False, l135=False, its=0)["status"] == "fail"    # pooled < 0.85
    # at L135 the pooled completion clause is dropped
    assert L.cell_eval(L.NOM["L135"], {**s1, 3: _rec(_with(h_meas__suite=0.5), es=3)}, None, carve=False,
                       l135=True, its=0)["status"] == "pass"
    # a clear failure (not near-line) never reaches es3 and an es3 launch error leaves the cell due
    clear = {1: _rec(_with(field_light_completion=0.5)), 2: _rec(_with(), es=2)}
    assert L.cell_eval(L.X08, clear, None, carve=False, l135=False, its=0)["status"] == "fail"
    bad3 = _rec(_with(), es=3)
    bad3["pass"], bad3["problems"] = None, ["dose hip"]
    e = L.cell_eval(L.X08, {**s1, 3: bad3}, None, carve=False, l135=False, its=0)
    assert e["status"] == "es3_due" and e["missing"]


# ============================================================================== L135
def test_ct_l135_completion_exemption_and_the_shared_limit_is_seed_specific(cx):
    L = cx.L
    h_fail = _with(h_meas__suite=0.80)
    ref = {1: _rec(h_fail), 2: _rec(_with(), es=2)}
    # completion is excluded at L135 on every seed; elsewhere it fails the cell
    low = {1: _rec(_with(field_light_completion=0.4)), 2: _rec(_with(field_light_completion=0.4), es=2)}
    e = L.cell_eval(L.NOM["L135"], low, ref, carve=False, l135=True, its=0)
    assert e["status"] == "pass" and e["seeds"][1]["excluded"] == ["field_light_completion"]
    assert L.cell_eval(L.NOM["L090"], low, ref, carve=False, l135=False, its=0)["status"] == "fail"
    # the reference must fail the same line on the SAME seed: seed 1 is shared, seed 2 is a regression
    both = {1: _rec(h_fail), 2: _rec(h_fail, es=2)}
    e = L.cell_eval(L.NOM["L135"], both, ref, carve=False, l135=True, its=0)
    assert e["status"] == "fail" and e["seeds"][1]["shared_limit"] == ["h_meas.suite"]
    assert e["seeds"][2]["failing"] == ["h_meas.suite"] and e["seeds"][2]["shared_limit"] == []
    # no reference read (invalid) -> no shared limit; and a head line the reference passes always fails
    inv = {1: {**_rec(h_fail), "pass": None}, 2: _rec(_with(), es=2)}
    assert L.cell_eval(L.NOM["L135"], {1: _rec(h_fail), 2: _rec(_with(), es=2)}, inv, carve=False, l135=True,
                       its=0)["status"] == "fail"
    other = {1: _rec(_with(survival__diag=0.5)), 2: _rec(_with(), es=2)}
    assert L.cell_eval(L.NOM["L135"], other, ref, carve=False, l135=True, its=0)["status"] == "fail"
    # the exemption is L135's only: completion at 84-86 elsewhere is near-line, at L135 there is nothing to rescue
    only_c = {1: _rec(_with(field_light_completion=0.84)), 2: _rec(_with(), es=2)}
    assert L.cell_eval(L.NOM["L135"], only_c, ref, carve=False, l135=True, its=0)["status"] == "pass"


# ============================================================================== the carve-out margin
def test_ct_carve_out_margin_is_max_of_R_3se_and_scaled_drift(cx, monkeypatch):
    L = cx.L
    margin = L.thm().zero_kg_margin
    st = {}
    fake = SimpleNamespace(
        suite_episodes=lambda p: {},
        zero_kg_suite_rows=lambda A, B, cats, R, Rg, translate=None: [
            {"statistic": f"track_err {cats[0]}", "watch": False, **st["row"]}],
        zero_kg_margin=margin)
    monkeypatch.setattr(L, "thm", lambda: fake)
    monkeypatch.setattr(L, "resolution", lambda: (None, None))
    monkeypatch.setattr(L, "drift_table", lambda: st["drift"])
    rec, ref = {"stamps": {"suite": "/a"}}, {"stamps": {"suite": "/b"}}

    def run(R, se, worse, drift, its, es=1):
        st["row"] = {"R": R, "se_diff": se, "worse_by": worse, "head": 0.1, "p2a": 0.1}
        st["drift"] = drift
        return L.carve_out(rec, ref, "diag", es, its)

    d1 = {"track_err diag|suite es1": 0.05, "track_err diag|suite es2": -0.2}
    o = run(0.04, 0.01, 0.04, d1, 2999)                                       # R binds (3 SE .03, drift .03)
    assert o["margin"]["binds"] == "R" and o["pass"] and o["label"] == "CAPABILITY-LIMITED"
    assert not run(0.04, 0.01, 0.0401, d1, 2999)["pass"]
    o = run(0.04, 0.02, 0.06, d1, 2999)                                       # 3 SE = .06 binds
    assert o["margin"]["binds"] == "3SE" and o["pass"] and not run(0.04, 0.02, 0.061, d1, 2999)["pass"]
    o = run(0.04, 0.01, 0.0, {"track_err diag|suite es1": 0.2}, 2999)         # |drift| * its / 5000 = .11996
    assert o["margin"]["binds"] == "drift_scaled" and abs(o["margin"]["margin"] - 0.2 * 2999 / 5000) < 1e-12
    assert run(0.04, 0.01, 0.119, {"track_err diag|suite es1": 0.2}, 2999)["pass"]
    assert not run(0.04, 0.01, 0.121, {"track_err diag|suite es1": 0.2}, 2999)["pass"]
    assert run(0.04, 0.01, 0.04, {"track_err diag|suite es1": 0.2}, 0)["margin"]["binds"] == "R"  # its_past_p2a 0
    assert run(0.04, 0.01, -0.3, d1, 2999)["pass"]                           # better than the reference always passes
    # es3 has no drift of its own: max |drift| of es1 / es2 (0.2 here)
    o = run(0.04, 0.01, 0.1, d1, 2999, es=3)
    assert abs(o["drift"] - 0.2) < 1e-12 and "max" in o["drift_source"] and o["pass"]
    # a missing statistic / R / drift never passes
    assert not run(0.04, 0.01, 0.0, {}, 2999)["pass"]
    assert not run(None, 0.01, 0.0, d1, 2999)["pass"]


def test_ct_carve_out_applies_to_tracking_lines_at_x07_x08_only(cx, monkeypatch):
    L = cx.L
    calls = []
    monkeypatch.setattr(L, "carve_out", lambda rec, ref, cat, es, its: calls.append((cat, es)) or
                        {"pass": True, "line": f"tracking.{cat}"})
    trk = _with(tracking__diag=0.30)
    recs = {1: _rec(trk), 2: _rec(trk, es=2)}
    ref = {1: _rec(_with()), 2: _rec(_with(), es=2)}
    assert L.cell_eval(L.X07, recs, ref, carve=True, l135=False, its=2999)["status"] == "pass"
    assert calls == [("diag", 1), ("diag", 2)]
    calls.clear()
    assert L.cell_eval(L.X135_L090, recs, None, carve=False, l135=False, its=2999)["status"] == "fail"
    assert calls == []                                                          # no carve-out at a band edge
    # a roll line is never carved; and without a valid reference read there is nothing to compare to
    roll = {1: _rec(_with(roll_rms__suite=0.2)), 2: _rec(_with(), es=2)}
    assert L.cell_eval(L.X07, roll, ref, carve=True, l135=False, its=2999)["status"] == "fail" and calls == []
    assert L.cell_eval(L.X07, recs, {1: {**ref[1], "pass": None}, 2: ref[2]}, carve=True, l135=False,
                       its=2999)["status"] == "fail"


# ============================================================================== pooled paired completion (real)
def test_ct_paired_completion_margin_floor_se_and_missing(cx, monkeypatch):
    L = cx.L
    eps = {}
    monkeypatch.setattr(L, "thm", lambda: SimpleNamespace(field_episodes=lambda p: eps[str(p)]))

    def rec(tag, flags):
        eps[f"/{tag}"] = {i: bool(f) for i, f in enumerate(flags)}
        return {"pass": True, "stamps": {"field_light": f"/{tag}"}}

    n = 200
    a_hi = rec("a1", [1] * 160 + [0] * 40)
    b_lo = rec("b1", [1] * 120 + [0] * 80)
    r = L.paired_completion([a_hi, a_hi], [b_lo, b_lo])
    assert abs(r["diff"] - 0.2) < 1e-12 and r["margin"] >= 0.05 and r["noninferior"] and r["beats_by_m"]
    assert r["head"] > r["ref"] and r["pass"] is True
    # paired and identical: SE 0 -> the 0.05 floor binds; a 0.04 gap is within it, a 0.06 gap is not
    base = [1] * 150 + [0] * 50
    same = rec("s1", base)
    r = L.paired_completion([rec("h1", [1] * 142 + [0] * 58)], [same])             # -0.04
    assert r["margin"] >= 0.05 and r["noninferior"] and not r["beats_by_m"]
    r = L.paired_completion([rec("h2", [1] * 138 + [0] * 62)], [same])             # -0.06
    assert not r["noninferior"] and not r["pass"]
    # unpaired noise: equal rates with disjoint successes -> the 2 SE term exceeds the floor
    x = rec("x", [1, 0] * 100)
    y = rec("y", [0, 1] * 100)
    r = L.paired_completion([x], [y])
    assert r["diff"] == 0 and r["margin"] > 0.05 and r["binds"] == "se"
    # missing / invalid / mismatched launches are not a reading
    inv = {"pass": None, "stamps": {"field_light": "/x"}}
    for a, b in (([inv], [x]), ([x], [inv]), ([x, x], [y]), ([], []), ([None], [x])):
        r = L.paired_completion(a, b)
        assert r["missing"] and r["pass"] is False
    # es1 + es2 are paired by (launch index, env): a one-sided es3 record cannot be passed in
    assert L.paired_completion(L.recs_list({1: x, 2: x, 3: x}), L.recs_list({1: y, 2: y}))["n"] == 400


# ============================================================================== G1 through g1_read (synthetic world)
class _World:
    """Stand-ins for the stamp readers: ``recs[(policy, cell, suffix)] = {es: record}`` (the records ``read_cellseed``
    would return) with a completion rate per record for the pooled paired read."""

    def __init__(self, L, monkeypatch):
        self.L, self.recs = L, {}
        self.R, self.used, self.pooled3, self.carve, self.carve_calls = "PASS", 0, 0.9, True, []
        pols = {}

        def policy(name):
            if name not in pols:
                sha = (name.encode().hex() + "0" * 64)[:64]
                pols[name] = {"name": name, "checkpoint": f"/x/{name}/model_1.pt", "sha256": sha, "sha8": sha[:8],
                              "role": "student" if name in ("PICK", "C28000") else "teacher", "iteration": 1}
            return pols[name]

        def seeds_of(pol, cid, *, pin, ess=(1, 2, 3), suffix="", root=None, rows_fn=None):
            d = self.recs.get((pol["name"], cid, suffix), {})
            return {es: d[es] for es in ess if es in d}

        def pc(a, b):
            if (not a or len(a) != len(b) or any(r is None or r.get("pass") is None for r in list(a) + list(b))):
                return {"missing": True, "pass": False, "why": "unread"}
            ha = sum(r["comp"] for r in a) / len(a)
            hb = sum(r["comp"] for r in b) / len(b)
            m = 0.05
            return {"head": ha, "ref": hb, "diff": ha - hb, "margin": m, "noninferior": ha - hb >= -m - 1e-12,
                    "beats_by_m": ha - hb >= m - 1e-12, "pass": ha - hb >= -m - 1e-12, "n": 100 * len(a)}

        for name, fn in (("policy", policy), ("launch_pin", lambda: {"code_sha": "c", "eval_sha": "e"}),
                         ("seeds_of", seeds_of), ("paired_completion", pc), ("zero_kg_rows", lambda pol: []),
                         ("zero_kg_stamps", lambda rows, **k: ({}, [])),
                         ("r_check", lambda zh, zo, its, tl=None: {"verdict": self.R, "its": its}),
                         ("closing_read", lambda *a, **k: {"status": "match", "flag": False}),
                         ("es3_used", lambda pols, cids, **k: self.used),
                         ("carve_out", lambda rec, ref, cat, es, its: self.carve_calls.append((cat, es)) or
                          {"pass": self.carve, "line": f"tracking.{cat}"}),
                         ("thm", lambda: SimpleNamespace(field_completion=lambda fl: {"rate": self.pooled3}))):
            monkeypatch.setattr(L, name, fn)

    def put(self, pol, cid, lines=None, comp=0.9, ess=(1, 2), suffix="", per_seed=None):
        d = self.recs.setdefault((pol, cid, suffix), {})
        for es in ess:
            ln = dict((per_seed or {}).get(es) or lines or GOOD)
            r = _rec(ln, es=es)
            r["comp"] = comp
            r["extras"].update(stand_survival=1.0, side_ratio_by_hand={"l": 1.0, "r": 1.0})
            r["name"] = f"{pol}_{cid}{suffix}"
            d[es] = r


OLD_COMP = {"T00_nominal_L000_A": 0.9, "T00_nominal_L090_A": 0.9, "T00_nominal_L135_A": 0.9,
            "T01_rod_speed_0p7_L090": 0.60, "T01_rod_speed_0p8_L090": 0.70}


def _g1_world(L, mp, tctl=True):
    w = _World(L, mp)
    for cid, c in OLD_COMP.items():
        w.put("TSTAR", cid, comp=c)
    for cid in L.S_CELLS + L.N_CELLS:
        w.put("TDR", cid, comp={L.X07: 0.80, L.X08: 0.82}.get(cid, 0.9))
    if tctl:
        for cid, c in ((L.NOM["L090"], 0.9), (L.X07, 0.62), (L.X08, 0.70)):
            w.put("TCTL", cid, comp=c)
    return w


def _read_g1(L, tctl="TCTL", **kw):
    return L.g1_read("TDR", "TSTAR", tctl, **kw)


def test_ct_g1_read_pass_and_the_lazy_control_trigger(cx, monkeypatch):
    L = cx.L
    w = _g1_world(L, monkeypatch)
    r = _read_g1(L)
    assert r["verdict"] == "PASS" and r["lazy_control_trigger"] and r["A"]["beats_by_m"] and r["half_gap"]
    assert r["its_past_p2a"] == 2999 and r["tctl_due"] is False and r["missing"] == []
    # no TCTL read yet: the control is due (TDR beat old T* by >= m, no N / R line failed)
    r = _read_g1(L, tctl=None)
    assert r["verdict"] == "TCTL-DUE" and r["tctl_due"] is True and r["A"] is None
    # the trigger is TDR - old >= m at x0.7: the edge counts (0.65 - 0.60), 0.649 does not
    w.put("TDR", L.X07, comp=0.65)
    assert _read_g1(L, tctl=None)["lazy_control_trigger"]
    w.put("TDR", L.X07, comp=0.649)
    r = _read_g1(L, tctl=None)
    assert not r["lazy_control_trigger"] and not r["tctl_due"] and r["verdict"] == "UNATTRIBUTED"
    assert "DR null" in r["notes"][0]
    # DR null with a failing S line is FAIL-STRESS; a failing N / R line outranks the null
    w.put("TDR", L.X07, lines=_with(roll_rms__suite=0.2), comp=0.6)
    assert _read_g1(L, tctl=None)["verdict"] == "FAIL-STRESS"
    w.put("TDR", L.X07, comp=0.6)
    w.R = "FAIL"
    assert _read_g1(L, tctl=None)["verdict"] == "FAIL-REGRESSION"
    # a failing N or R read suppresses the control launch (it would be GPU spent on a verdict already made)
    w.put("TDR", L.X07, comp=0.8)
    assert _read_g1(L, tctl=None)["tctl_due"] is False


def test_ct_g1_read_every_branch_of_the_verdict_table(cx, monkeypatch):
    L = cx.L
    # FAIL-REGRESSION: N pooled completion below old by more than m at L135, a failing N line, R fails
    w = _g1_world(L, monkeypatch)
    w.put("TDR", L.NOM["L135"], comp=0.8)
    assert _read_g1(L)["verdict"] == "FAIL-REGRESSION"
    w = _g1_world(L, monkeypatch)
    w.put("TDR", L.NOM["L000"], lines=_with(survival__diag=0.5))
    r = _read_g1(L)
    assert r["verdict"] == "FAIL-REGRESSION" and r["N"][L.NOM["L000"]]["status"] == "fail"
    w = _g1_world(L, monkeypatch)
    w.R = "FAIL"
    assert _read_g1(L)["verdict"] == "FAIL-REGRESSION"
    # UNATTRIBUTED: S, N, R pass but TDR - TCTL < m at x0.7
    w = _g1_world(L, monkeypatch)
    w.put("TCTL", L.X07, comp=0.78)
    r = _read_g1(L)
    assert r["verdict"] == "UNATTRIBUTED" and not r["A"]["beats_by_m"]
    # EXTEND: only a stress completion miss; A >= m; TCTL - old < m; x0.7 above the half-gap; TDR - old >= m
    w = _g1_world(L, monkeypatch)
    w.put("TDR", L.X07, lines=_with(field_light_completion=0.80), comp=0.80)
    r = _read_g1(L)
    assert r["verdict"] == "EXTEND" and r["S"][L.X07]["fail_keys"] == ["field_light_completion"]
    assert _read_g1(L, extended=True)["verdict"] == "FAIL-STRESS"                      # EXTEND is once; the re-read is final
    assert _read_g1(L, extended=True)["its_past_p2a"] == 5998
    # ... and a survival miss counts like a completion miss
    w.put("TDR", L.X07, lines=_with(survival__diag=0.5), comp=0.80)
    assert _read_g1(L)["verdict"] == "EXTEND"
    # EXTEND with TCTL - old >= m at x0.7: the no-DR continuation moved it -> FAIL-STRESS with that note
    w.put("TDR", L.X07, lines=_with(field_light_completion=0.80), comp=0.80)
    w.put("TCTL", L.X07, comp=0.66)
    r = _read_g1(L)
    assert r["verdict"] == "FAIL-STRESS" and "moved x0.7" in r["notes"][0]
    w.put("TCTL", L.X07, comp=0.649)                                              # TCTL - old = 0.049 < m: EXTEND
    assert _read_g1(L)["verdict"] == "EXTEND"
    # FAIL-STRESS: a roll / pitch-only miss; a tracking miss that the carve-out does not rescue; below the half-gap
    w.put("TDR", L.X07, lines=_with(roll_rms__suite=0.2), comp=0.80)
    assert _read_g1(L)["verdict"] == "FAIL-STRESS"
    w.put("TDR", L.X07, lines=_with(field_light_completion=0.80), comp=0.70)       # 0.70 < 0.6 + (0.85 - 0.6) / 2
    r = _read_g1(L)
    assert r["verdict"] == "FAIL-STRESS" and r["half_gap"] is False
    w.put("TDR", L.X07, lines=_with(field_light_completion=0.80), comp=0.725)     # ... exactly the half-gap passes
    assert _read_g1(L)["half_gap"] is True
    w.put("TDR", L.X07, lines=_with(field_light_completion=0.80), comp=0.80)
    w.put("TCTL", L.X07, comp=0.80)                                               # A < m on top of the S miss
    assert _read_g1(L)["verdict"] == "FAIL-STRESS"
    assert any("A < m" in n for n in _read_g1(L)["notes"])


def test_ct_g1_read_tracking_carve_out_only_at_x07_x08(cx, monkeypatch):
    L = cx.L
    w = _g1_world(L, monkeypatch)
    w.put("TDR", L.X07, lines=_with(tracking__diag=0.30))
    assert _read_g1(L)["verdict"] == "PASS" and set(w.carve_calls) == {("diag", 1), ("diag", 2)}   # carve passes
    w.carve = False
    assert _read_g1(L)["verdict"] == "FAIL-STRESS"                                     # a tracking miss is not completion
    w.carve, w.carve_calls[:] = True, []
    w.put("TDR", L.X07)
    w.put("TDR", L.X135_L090, lines=_with(tracking__diag=0.30))
    assert _read_g1(L)["verdict"] == "FAIL-STRESS" and w.carve_calls == []              # no carve-out at the band edges


def test_ct_g1_read_es3_resolution_and_the_cell_seed_cap(cx, monkeypatch):
    L = cx.L
    w = _g1_world(L, monkeypatch)
    near = {1: _with(field_light_completion=0.84), 2: dict(GOOD)}
    w.put("TDR", L.X08, per_seed=near, comp=0.82)
    r = _read_g1(L)
    assert r["es3_due"] == [L.X08] and r["verdict"] == "PENDING-ES3" and set(r["possible"]) == {"EXTEND", "PASS"}
    w.put("TDR", L.X08, lines=dict(GOOD), comp=0.82, ess=(3,))                       # es3 passes, pooled 0.9
    r = _read_g1(L)
    assert r["verdict"] == "PASS" and r["es3_due"] == []
    # the allowance is 4 cell-seeds per gate: two pending cells with one left is UNRESOLVED, with two it waits
    w = _g1_world(L, monkeypatch)
    for cid in (L.X08, L.X07_L000):
        w.put("TDR", cid, per_seed=near, comp=0.82)
    w.used = 3
    r = _read_g1(L)
    assert r["verdict"] == "UNRESOLVED" and "allowance" in r["notes"][0] and r["es3_cap"] == 4
    w.used = 2
    assert _read_g1(L)["verdict"] == "PENDING-ES3"
    # a verdict that every es3 outcome leaves unchanged stands without spending any cell-seed
    w = _g1_world(L, monkeypatch)
    w.put("TDR", L.NOM["L000"], per_seed=near)
    w.put("TDR", L.NOM["L135"], lines=_with(h_meas__suite=0.5))
    w.put("TSTAR", L.NOM["L135"], lines=dict(GOOD))
    w.used = 4
    assert _read_g1(L)["verdict"] == "FAIL-REGRESSION"


def test_ct_g1_read_incomplete_while_rows_are_missing_unless_a_regression_is_already_clear(cx, monkeypatch):
    L = cx.L
    w = _g1_world(L, monkeypatch)
    del w.recs[("TDR", L.X135_L000, "")]
    r = _read_g1(L)
    assert r["verdict"] == "INCOMPLETE" and any(f"TDR {L.X135_L000} es1" in m for m in r["missing"])
    w.put("TDR", L.NOM["L135"], comp=0.5)                                          # a clear N failure is final
    assert _read_g1(L)["verdict"] == "FAIL-REGRESSION"
    w = _g1_world(L, monkeypatch)
    del w.recs[("TSTAR", L.X07, "")]
    assert _read_g1(L)["verdict"] == "INCOMPLETE" and any("old" in m for m in _read_g1(L)["missing"])


# ============================================================================== G2 through g2_read (synthetic world)
def _g2_world(L, mp):
    real = L.sweep()
    w = _World(L, mp)
    mp.setattr(L, "sweep", lambda: SimpleNamespace(C=real.C, SR=SimpleNamespace(cellseed=lambda *a, **k: None)))
    mp.setattr(L, "g2_r_watch", lambda *a, **k: {"missing": []})
    for cid in L.REF_CELLS:
        w.put("C28000", cid, comp=OLD_COMP[cid])
    for cid in L.S_CELLS + L.N_CELLS:
        w.put("PICK", cid, comp={L.X07: 0.85, L.X08: 0.86}.get(cid, 0.9))
    return w


def _g2(L):
    return L.g2_read("PICK", "C28000")


def test_ct_g2_class_ambiguity_relaunch_rule(cx, monkeypatch):
    L = cx.L
    nom = L.NOM["L090"]
    w = _g2_world(L, monkeypatch)
    r = _g2(L)
    assert r["verdict"] == "PASS" and r["its_past_p2a"] == 0 and r["relaunch_due"] == []
    # within m (0.05): pass; in (m, m + 0.04]: class-ambiguous, relaunch both sides; beyond: FAIL at once
    w.put("PICK", nom, comp=0.86)
    r = _g2(L)
    assert r["N_pooled_state"][nom] == "pass" and r["verdict"] == "PASS"
    w.put("PICK", nom, comp=0.83)                                                  # shortfall 0.07
    r = _g2(L)
    assert r["N_pooled_state"][nom] == "ambiguous" and r["relaunch_due"] == [nom] and r["verdict"] == "RELAUNCH-DUE"
    w.put("PICK", nom, comp=0.90 - 0.0901)                                         # 0.0901 > m + 0.04: no relaunch
    r = _g2(L)
    assert r["N_pooled_state"][nom] == "fail" and r["verdict"] == "FAIL" and r["relaunch_due"] == []
    w.put("PICK", nom, comp=0.90 - 0.0899)                                         # the upper edge is still ambiguous
    assert _g2(L)["N_pooled_state"][nom] == "ambiguous"
    # the relaunch decides on the pooled FOUR-launch read: both sides must be relaunched
    w.put("PICK", nom, comp=0.83)
    w.put("PICK", nom, comp=0.90, suffix=L.RELAUNCH)
    r = _g2(L)                                                                      # the reference side is still missing
    assert r["relaunch_due"] == [nom] and r["verdict"] == "RELAUNCH-DUE"            # a half relaunch is resumable
    assert r["es3_used"] == 4 and r["es3_only"] == 0 and r["relaunch_charged"] == 4  # charged once, not per stamp
    w.put("C28000", nom, comp=0.90, suffix=L.RELAUNCH)
    r = _g2(L)                                                                      # (0.83 + 0.90) / 2 vs 0.90: 0.035
    assert r["N_pooled_state"][nom] == "pass" and r["verdict"] == "PASS" and "four_launch" in r["N_pooled"][nom]
    w.put("PICK", nom, comp=0.83, suffix=L.RELAUNCH)                                # the shortfall survives the relaunch
    r = _g2(L)
    assert r["N_pooled_state"][nom] == "fail" and r["verdict"] == "FAIL"
    # a relaunch needs 4 cell-seeds of the es3 allowance; with fewer left the verdict is UNRESOLVED, not a guess
    w = _g2_world(L, monkeypatch)
    w.put("PICK", nom, comp=0.83)
    w.used = 1
    r = _g2(L)
    assert r["verdict"] == "UNRESOLVED" and "4 cell-seeds" in r["notes"][0]


def test_ct_g2_s_and_n_lines_and_the_carve_out_reference(cx, monkeypatch):
    L = cx.L
    w = _g2_world(L, monkeypatch)
    w.put("PICK", L.X07, lines=_with(survival__diag=0.5))
    r = _g2(L)
    assert r["verdict"] == "FAIL" and r["S"][L.X07]["status"] == "fail"       # G2 has no EXTEND: any S miss fails
    w = _g2_world(L, monkeypatch)
    w.put("PICK", L.X08, lines=_with(tracking__diag=0.30))                      # carved against in-pin c28000
    assert _g2(L)["verdict"] == "PASS" and set(w.carve_calls) == {("diag", 1), ("diag", 2)}
    w = _g2_world(L, monkeypatch)
    w.put("PICK", L.NOM["L135"], lines=_with(field_light_completion=0.3))      # L135 completion is WATCH only
    r = _g2(L)
    assert r["verdict"] == "PASS" and r["watch"]["L135_completion"]["es1"] == 0.3
    w = _g2_world(L, monkeypatch)
    del w.recs[("C28000", L.NOM["L135"], "")]
    r = _g2(L)
    assert r["verdict"] == "INCOMPLETE" and any("ref" in m for m in r["missing"])


# ============================================================================== verdict tables, directly
def test_ct_g1_decide_precedence_and_dr_null(cx):
    L = cx.L
    d = L.g1_decide
    fail_cs = {"x07": (False, ["field_light_completion"])}
    # N / R outrank everything, including a null DR
    assert d(_g1(r_ok=False, trig=False))[0] == "FAIL-REGRESSION"
    assert d(_g1(n_pooled_ok=False, s_cells=fail_cs))[0] == "FAIL-REGRESSION"
    # TDR - old < m at x0.7: UNATTRIBUTED when S passes, FAIL-STRESS ("DR null") when it does not
    assert d(_g1(trig=False, a_ok=None))[0] == "UNATTRIBUTED"
    v, n = d(_g1(trig=False, a_ok=None, s_cells=fail_cs))
    assert v == "FAIL-STRESS" and "DR null" in n[0]
    # S passes with A failing is UNATTRIBUTED; with A passing but TCTL moved after EXTEND, FAIL-STRESS
    assert d(_g1(a_ok=False, tctl_moved=True))[0] == "UNATTRIBUTED"
    assert d(_g1(extended=True, tctl_moved=True))[0] == "FAIL-STRESS"
    assert d(_g1(extended=True, tctl_moved=False))[0] == "PASS" and "length-unmatched" in d(_g1(extended=True))[1][0]
    # EXTEND needs every clause: completion / survival only, A, half-gap, not moved, not already extended
    for kw in ({"a_ok": False}, {"half_gap": False}, {"tctl_moved": True}, {"extended": True}):
        assert d(_g1(s_cells=fail_cs, **kw))[0] == "FAIL-STRESS", kw
    for keys in (["tracking.diag"], ["h_meas.suite"], ["turn_fc.suite"], ["roll_rms.suite", "field_light_completion"],
                 ["pitch_rms.suite"]):
        assert d(_g1(s_cells={"x07": (False, keys)}))[0] == "FAIL-STRESS", keys
    for keys in (["field_light_completion"], ["survival.side", "survival.turn"], ["field_light_completion",
                                                                                   "survival.diag"]):
        assert d(_g1(s_cells={"x07": (False, keys)}))[0] == "EXTEND", keys
    assert d(_g1(a_ok=None, s_cells=fail_cs))[0] == "TCTL-DUE"


def test_ct_resolve_cap_arithmetic(cx):
    L = cx.L
    cells = {"s1": {"status": "es3_due", "fail_keys": ["field_light_completion"]},
             "s2": {"status": "es3_due", "fail_keys": ["field_light_completion"]},
             "n": {"status": "pass", "fail_keys": []}}
    base = {**_g1(), "s_cells": ["s1", "s2"], "n_cells": ["n"]}
    assert L.resolve(cells, L.g1_decide, base, cap_left=2)["verdict"] == "PENDING-ES3"
    r = L.resolve(cells, L.g1_decide, base, cap_left=1)
    assert r["verdict"] == "UNRESOLVED" and r["pending"] == ["s1", "s2"] and set(r["possible"]) == {"EXTEND", "PASS"}
    # nothing pending: the decision is taken as is, whatever the cap
    done = {k: dict(v, status="pass") for k, v in cells.items()}
    assert L.resolve(done, L.g1_decide, base, cap_left=0)["verdict"] == "PASS"
    assert L.ES3_CAP == 4 and L.G2_AMBIG == 0.04 and L.ITS_PAST_P2A == {"g1": 2999, "g1_extended": 5998, "g2": 0}


# ============================================================================== selection
def _sel_world(cx, tmp_path):
    L = cx.L
    L.policy_add("TDR", _ck(tmp_path, "tdr", 28991), "teacher")
    run = tmp_path / "srun"
    (run / "params").mkdir(parents=True)
    (run / "params" / "agent.yaml").write_text("seed: 19\n")
    for i in (25000, 26992, 28000, 29991, 30991, 31991, 32991, 33991, 34100):
        (run / f"model_{i}.pt").write_bytes(f"student-{i}".encode())
    return run


def test_ct_selection_candidates_number_from_the_teacher_iteration(cx, tmp_path):
    L = cx.L
    run = _sel_world(cx, tmp_path)
    ss = L.ss()
    cands = ss.candidate_checkpoints(run, 28991, L.sel_spec()["offsets"])
    assert [c["target"] for c in cands] == [29991, 30991, 31991, 32991, 33991]
    assert [c["iter"] for c in cands] == [29991, 30991, 31991, 32991, 33991]
    # T* 's own iteration (25992) would have picked other checkpoints: the start must be TDR's
    assert [c["iter"] for c in ss.candidate_checkpoints(run, 25992, L.sel_spec()["offsets"])] != [c["iter"] for c in cands]
    # the nearest saved checkpoint wins, the earlier on a tie; two offsets on one file are one candidate
    for i in (30000, 30002):
        (run / f"model_{i}.pt").write_bytes(b"x")
    c2 = ss.candidate_checkpoints(run, 28991, [1000, 1010, 1011])           # targets 29991, 30001, 30002
    assert [c["iter"] for c in c2] == [29991, 30000, 30002][:len(c2)] and len(c2) == 3
    assert ss.candidate_checkpoints(run, 28991, [1000, 1000])[0]["iter"] == 29991
    assert len(ss.candidate_checkpoints(run, 28991, [1000, 1000])) == 1


def test_ct_sel_dry_rows_use_the_explicit_90_kg_level_and_never_zero(cx, tmp_path, monkeypatch):
    L, G = cx.L, cx.G
    run = _sel_world(cx, tmp_path)
    seen = []
    monkeypatch.setattr(G.Runner, "run", lambda self, row: seen.append(row))
    a = SimpleNamespace(teacher="TDR", dry=True, student_run=str(run), pick_name="PICK", approve_extra_block=False,
                        dry_teacher_iter=None)
    assert G.ph_sel(a, G.Runner(True, "sel")) == 0
    assert len(seen) == 35                                                           # 25 prescreen + 10 first block
    lvl = [r for r in seen if r["level_file"]]
    assert len(lvl) == 12 and {r["level_kg"] for r in lvl} == {90}
    assert {Path(r["level_file"]).name for r in lvl} == {"L090.env"}
    assert all(r["level_kg"] is None for r in seen if not r["level_file"])
    assert all(r["name"].endswith("_L090") for r in lvl) and not any(r["name"].endswith("_L090") for r in seen
                                                                     if not r["level_file"])
    cand_sha = {r["name"].split("_")[1] for r in seen}
    assert len(cand_sha) == 5                                                        # five distinct candidates
    assert all(r["role"] == "student" and r["kind"] == "sel" for r in seen)
    # an unregistered TDR (dry) maps from --dry-teacher-iter: the same five candidates
    seen2 = []
    monkeypatch.setattr(G.Runner, "run", lambda self, row: seen2.append(row))
    a2 = SimpleNamespace(**{**vars(a), "teacher": "TDRX", "dry_teacher_iter": "28991"})
    G.ph_sel(a2, G.Runner(True, "sel"))
    assert {r["ckpt"] for r in seen2} == {r["ckpt"] for r in seen}
    # the stage's own level rows: always 90 (set from the spec), and the level row set is the spec's
    st = L.sel_spec()
    P = L.ephemeral_policy(_ck(tmp_path, "cc", 29991, student=True), "student", "c29991")
    assert {r["level_kg"] for r in L.student_sel_rows(P, "prescreen_level").values()} == {90}
    assert {r["level_kg"] for r in L.student_sel_rows(P, "full_level").values()} == {90}
    assert set(L.student_sel_rows(P, "full_level")) == set(st["full_block_level_rows"])
    assert {r["level_kg"] for r in L.student_sel_rows(P, "full").values()} == {None}
    assert st["prescreen_level"]["kg"] == 90


def test_ct_sel_row_at_zero_kg_would_collide_with_the_no_level_row(cx, tmp_path):
    """A level row at 0 kg would get the SAME name / scenario / seed as the no-level row (``sel_name(pol, 0)`` ==
    ``sel_name(pol, None)``), so a 0 sneaking in (pipeline.py:2362's ``max_stable or 0``) would reuse the no-level
    stamp as if it were a level stamp: ``sel_row`` refuses a level <= 0 (fix 2026-11-07)."""
    L = cx.L
    P = L.ephemeral_policy(_ck(tmp_path, "cc", 29991, student=True), "student", "c29991")
    assert L.sel_name(P, 0) == L.sel_name(P, None) != L.sel_name(P, 90)
    for bad in (0, -90):
        with pytest.raises(L.PolicyError, match="positive kg"):
            L.sel_row(P, "suite@1", bad)
    plain, lv = L.sel_row(P, "suite@1"), L.sel_row(P, "suite@1", 90)
    assert plain["level_file"] is None and lv["level_file"] is not None and lv["name"] != plain["name"]


# ============================================================================== launcher helpers
def test_ct_knob_multiset_comparison_is_order_free_and_counts_duplicates(mods):
    RL = mods.RL
    a = ["[KNOB] A=1", "[omni-knobs] x (2.0) synced", "[KNOB] B=2"]
    assert RL.multiset_diff(list(reversed(a)), a) == {"missing": [], "extra": []}
    assert RL.multiset_diff(a + ["[KNOB] B=2"], a) == {"missing": [], "extra": ["[KNOB] B=2"]}      # a repeat is extra
    assert RL.multiset_diff(a[:2], a) == {"missing": ["[KNOB] B=2"], "extra": []}
    assert RL.multiset_diff(a[:2] + ["[KNOB] B=3"], a) == {"missing": ["[KNOB] B=2"], "extra": ["[KNOB] B=3"]}
    text = f"boot\n[KNOB] A=1\nnoise [KNOB] mid\n[omni-knobs] x\r\n  [KNOB] indented\n{RL.W}/p [KNOB] no\n[KNOB] root {RL.W}/a {RL.MAIN}/b\n"
    got = RL.knob_lines(text)
    assert got[0] == "[KNOB] A=1" and "[omni-knobs] x" in got and "[KNOB] root <ROOT>/a <ROOT>/b" in got
    assert not any("noise" in g or "indented" in g or "no" == g[-2:] for g in got)


def test_ct_yaml_root_normaliser_replaces_the_worktree_before_main(mods):
    RL = mods.RL
    assert str(RL.W).startswith(str(RL.MAIN) + "/")                                  # the worktree lives inside main
    t = f"a: {RL.W}/assets/crab.usda\nb: {RL.MAIN}/parkour/logs\nc: {RL.MAIN}/.claude/worktrees/x\n"
    assert RL.normalise_roots(t) == "a: <ROOT>/assets/crab.usda\nb: <ROOT>/parkour/logs\nc: <ROOT>/.claude/worktrees/x\n"
    # the wrong order would leave the worktree's own suffix in the text
    wrong = t.replace(str(RL.MAIN), "<ROOT>").replace(str(RL.W), "<ROOT>")
    assert wrong != RL.normalise_roots(t) and "<ROOT>/.claude/worktrees/dr-stress-sweep/assets" in wrong
    other = Path("/tmp/elsewhere")                                                    # explicit roots are honoured
    assert RL.normalise_roots(f"{other}/a {RL.MAIN}/b", w=other) == "<ROOT>/a <ROOT>/b"


def test_ct_predicted_first_draw_is_what_rod_speed_dr_produces(mods):
    import contextlib
    import io
    RL = mods.RL
    mod = RL.rod_speed_dr()
    for spec, n, seed, step in (({"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}, 256, 19, 0),
                                ({"lo": 0.6, "hi": 1.35, "ramp_steps": 0}, 192, 19, 0),
                                ({"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}, 8, 3, 6000)):
        obj = mod.RodSpeedDR(dict(spec), n, "cpu", seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ret = obj.reset(None, step)                                            # the env's first reset: all envs
        line = [ln for ln in buf.getvalue().split("\n") if ln][0]
        p = RL.predicted_dr(spec, n, seed, step)
        assert p["line"] == line and p["sha16"] == mod._sha16(obj.scale[:, 0]) and f"sha16={p['sha16']}" in line
        assert abs(p["min"] - float(obj.scale.min())) == 0 and abs(p["max"] - float(obj.scale.max())) == 0
        assert abs(float(ret["DR/rod_speed_lo"]) - p["lo_eff"]) < 1e-6
        lo_eff = 1.0 + (spec["lo"] - 1.0) * min(step / spec["ramp_steps"], 1.0) if spec["ramp_steps"] else spec["lo"]
        assert lo_eff - 1e-6 <= p["min"] and p["max"] <= spec["hi"] + 1e-6         # the draw lies on [lo_eff, hi]
    # a different seed / env count is a different sha (what PF11 distinguishes)
    base = RL.predicted_dr({"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}, 256, 19)["sha16"]
    assert RL.predicted_dr({"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}, 256, 20)["sha16"] != base
    assert RL.predicted_dr({"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}, 255, 19)["sha16"] != base
    # mid-ramp the draw is on [lo_eff(step), hi]
    mid = RL.predicted_dr({"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}, 256, 19, 6000)
    assert abs(mid["lo_eff"] - 0.8) < 1e-12 and 0.8 <= mid["min"] and mid["max"] <= 1.35 + 1e-6


def test_ct_launcher_env_argv_and_relative_checkpoint(mods, tmp_path):
    T, RL = mods.T, mods.RL
    a = _args(checkpoint="/abs/m/model_5.pt", name="RSD_X", iterations=7, num_envs=9, seed=3, env_file="/abs/p.env")
    argv = T.trainer_argv(a)
    assert argv[:2] == [RL.PY, str(RL.TRAIN)] and argv[-2:] == ["--run_name", "RSD_X"]
    assert "-I" not in argv and "--resume" in argv and argv[argv.index("--checkpoint") + 1] == "/abs/m/model_5.pt"
    assert argv[argv.index("--max_iterations") + 1] == "7" and argv[argv.index("--seed") + 1] == "3"
    # the composed env: stress / phase / DR-in-file keys are refused, empties are kept verbatim, inherited keys dropped
    base = {"PATH": "/p", "KRABBY_LEAK": "1", "TP_X": "1", "DRS_Y": "1", "RSD_Z": "1", "PYTHONPATH": "/main"}
    for key in ("KRABBY_STRESS_ROD_SPEED", "KRABBY_STRESS_PUSH", "KRABBY_PHASE", "KRABBY_SIDE_PHASE",
                "KRABBY_DR_ROD_SPEED"):
        _, _, ref = T.compose_env(_args(arm="ctl", dr=None), base, [(key, "1")])
        assert ref, key
    env, kr, ref = T.compose_env(_args(), base, [("KRABBY_PHASEOUT", ""), ("KRABBY_OMNI_DIR_BAND", "")])
    assert not ref and kr["KRABBY_PHASEOUT"] == "" and kr["KRABBY_OMNI_DIR_BAND"] == "" and env["KRABBY_PHASEOUT"] == ""
    assert "KRABBY_LEAK" not in env and "PYTHONPATH" in env and env["PYTHONPATH"] != "/main"
    # a relative checkpoint is a usage error; so is one that is not there
    ef = tmp_path / "e.env"
    ef.write_text("KRABBY_A=1\n")
    with pytest.raises(T.Usage):
        T.parse_args(["--arm", "dr", "--name", "N", "--checkpoint", "rel/model_1.pt", "--env-file", str(ef)])
    with pytest.raises(T.Usage):
        T.parse_args(["--arm", "dr", "--name", "N", "--checkpoint", str(tmp_path / "model_9.pt"), "--env-file", str(ef)])


# ##############################################################################################################
# Fixes after the independent code-check (2026-11-07, the ``test_fx_*`` tests): the G2 half-relaunch accounting,
# the per-gate es3 allowance across EXTEND, the es3 / relaunch launch guards (fresh read of THIS policy), the c28000
# bridge before the pick's rows, the launcher's resume / ramp guards and lo_eff-tight min bound, the distill wrapper's
# G1-PASS guard, and the verdict layer (paired_completion, carve_out) on recorded sweep stamps.
# ##############################################################################################################
def _no_launch():
    ran = []
    return ran, SimpleNamespace(run_all=lambda rows: ran.extend(rows))


def test_fx_g2relaunch_resumes_a_half_finished_relaunch_and_refuses_a_stale_or_foreign_read(cx, monkeypatch):
    L, G = cx.L, cx.G
    nom = L.NOM["L090"]
    w = _g2_world(L, monkeypatch)
    w.put("PICK", nom, comp=0.83)
    w.put("PICK", nom, comp=0.90, suffix=L.RELAUNCH)                  # the pick side ran, the reference side did not
    r = _g2(L)
    assert r["verdict"] == "RELAUNCH-DUE" and r["es3_only"] == 0 and r["es3_used"] == 4
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    L.write_json(L.RESULTS / "g2.json", r)
    a = SimpleNamespace(policy="PICK", cell=nom, dry=False)
    ran, R = _no_launch()
    G.ph_g2relaunch(a, R)                                             # resumable: 0 + 4 <= 4
    assert {row["name"] for row in ran} == {L.cell_name(L.policy("PICK"), nom, L.RELAUNCH),
                                            L.cell_name(L.policy("C28000"), nom, L.RELAUNCH)} and len(ran) == 8
    w.used = 1                                                        # an es3 ran after the read: stale
    with pytest.raises(L.Stop, match="read g2 again"):
        G.ph_g2relaunch(a, _no_launch()[1])
    assert _g2(L)["verdict"] == "UNRESOLVED"                          # and a fresh read is UNRESOLVED (1 + 4 > 4)
    w.used = 0
    bad = copy.deepcopy(r)
    bad["policies"]["pick"]["sha256"] = "f" * 64
    L.write_json(L.RESULTS / "g2.json", bad)
    with pytest.raises(L.Stop, match="read for pick"):
        G.ph_g2relaunch(a, _no_launch()[1])
    bad = copy.deepcopy(r)
    bad["verdict"] = "UNRESOLVED"
    L.write_json(L.RESULTS / "g2.json", bad)
    with pytest.raises(L.Stop, match="RELAUNCH-DUE / PENDING-ES3 only"):
        G.ph_g2relaunch(a, _no_launch()[1])


def test_fx_es3_launch_needs_a_fresh_pending_read_of_this_policy(cx, monkeypatch):
    L, G = cx.L, cx.G
    w = _g1_world(L, monkeypatch)
    w.put("TDR", L.X08, per_seed={1: _with(field_light_completion=0.84), 2: dict(GOOD)}, comp=0.82)
    r = _read_g1(L)
    assert r["verdict"] == "PENDING-ES3" and r["es3_policies"] == ["TDR"]
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    L.write_json(L.RESULTS / "g1.json", r)
    a = SimpleNamespace(policy="TDR", gate="g1", cell=L.X08, dry=False)
    ran, R = _no_launch()
    G.ph_es3(a, R)
    assert [row["es"] for row in ran] == [3, 3]
    w.used = 1                                                        # the es3 ran; the json still says 0
    with pytest.raises(L.Stop, match="read g1 again"):
        G.ph_es3(a, _no_launch()[1])
    w.used = 0
    for edit, msg in ((lambda d: d.update(verdict="UNRESOLVED"), "PENDING-ES3 read"),
                      (lambda d: d["policies"]["tdr"].update(sha256="e" * 64), "read for tdr")):
        d = copy.deepcopy(r)
        edit(d)
        L.write_json(L.RESULTS / "g1.json", d)
        with pytest.raises(L.Stop, match=msg):
            G.ph_es3(a, _no_launch()[1])


def test_fx_the_es3_allowance_is_per_gate_across_extend(cx, monkeypatch, tmp_path):
    L, G = cx.L, cx.G
    _g1_world(L, monkeypatch)
    counts = {"TPRIOR": 3}
    monkeypatch.setattr(L, "es3_used", lambda pols, cids, **k: sum(counts.get(p["name"], 0) for p in pols))
    r = L.g1_read("TDR", "TSTAR", "TCTL", extended=True, prior_tdr_name="TPRIOR")
    assert r["es3_used"] == 3 and r["es3_policies"] == ["TDR", "TPRIOR"] and r["its_past_p2a"] == 5998
    assert L.g1_read("TDR", "TSTAR", "TCTL", extended=True)["es3_used"] == 0
    a = SimpleNamespace(which="g1", tdr="TDR", old="TSTAR", tctl="TCTL", extended=True, prior_tdr=None,
                        out=str(tmp_path / "g1x.json"))
    with pytest.raises(L.Stop, match="--prior-tdr") as e:
        G.ph_read(a)
    assert e.value.code == 2
    assert G.parser().parse_args(["read", "g1", "--tdr", "T", "--extended", "--prior-tdr", "P"]).prior_tdr == "P"


def test_fx_g2_runs_the_picks_rows_only_after_the_c28000_bridge(cx, monkeypatch):
    L, G = cx.L, cx.G
    ref = {"name": "C28000", "sha256": "c" * 64, "sha8": "c" * 8, "role": "student", "checkpoint": "/x/model_1.pt",
           "iteration": 1}
    monkeypatch.setattr(L, "policy", lambda n: ref)
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    flip = SimpleNamespace(bridge_class_flip=False)
    for doc in ({}, {"verdict": "RELAUNCH_DUE", "sha256": ref["sha256"]}, {"verdict": "PASS", "sha256": "d" * 64},
                {"verdict": "STOP", "sha256": ref["sha256"]}):
        L.write_json(L.RESULTS / "bridge_C28000.json", doc)
        with pytest.raises(L.Stop, match="bridge_C28000.json") as e:
            G.g2_bridge_guard(flip)
        assert e.value.code == 3
    G.g2_bridge_guard(SimpleNamespace(bridge_class_flip=True))         # a recorded class flip proceeds
    L.write_json(L.RESULTS / "bridge_C28000.json", {"verdict": "PASS", "sha256": ref["sha256"]})
    G.g2_bridge_guard(flip)
    rows = G.g2ref_rows()
    assert len(rows) == 20 and {r["policy"] for r in rows} == {"C28000"} and {r["cell"] for r in rows} == set(L.REF_CELLS)
    assert "g2ref" in G.ROW_PHASES and G.ROW_PHASES.index("g2ref") < G.ROW_PHASES.index("g2")
    assert G.parser().parse_args(["g2", "--policy", "PICK", "--bridge-class-flip"]).bridge_class_flip
    assert G.parser().parse_args(["g2ref", "--dry"]).cmd == "g2ref"


def test_mt_g1_tctl_and_extend_run_only_after_the_t_star_bridge(cx, monkeypatch):
    L, G = cx.L, cx.G
    ref = {"name": "TSTAR", "sha256": "a" * 64, "sha8": "a" * 8, "role": "teacher", "checkpoint": "/x/model_1.pt",
           "iteration": 1}
    monkeypatch.setattr(L, "policy", lambda n: ref)
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    for cmd in ("g1", "tctl", "extend"):
        flip = SimpleNamespace(cmd=cmd, bridge_class_flip=False)
        for doc in ({}, {"verdict": "RELAUNCH_DUE", "sha256": ref["sha256"]}, {"verdict": "PASS", "sha256": "d" * 64},
                    {"verdict": "STOP", "sha256": ref["sha256"]}, {"verdict": "INCOMPLETE", "sha256": ref["sha256"]}):
            L.write_json(L.RESULTS / "bridge_TSTAR.json", doc)
            with pytest.raises(L.Stop, match=f"{cmd}: results/bridge_TSTAR.json") as e:
                G.g1_bridge_guard(flip)
            assert e.value.code == 3
        with pytest.raises(L.Stop):                                        # a flip excuses a STOP only
            G.g1_bridge_guard(SimpleNamespace(cmd=cmd, bridge_class_flip=True))
        L.write_json(L.RESULTS / "bridge_TSTAR.json", {"verdict": "STOP", "sha256": ref["sha256"]})
        G.g1_bridge_guard(SimpleNamespace(cmd=cmd, bridge_class_flip=True))  # a recorded class flip proceeds
        L.write_json(L.RESULTS / "bridge_TSTAR.json", {"verdict": "PASS", "sha256": ref["sha256"]})
        G.g1_bridge_guard(flip)
        L.write_json(L.RESULTS / "bridge_TSTAR.json", {})
        assert G.parser().parse_args([cmd, "--policy", "TDR", "--bridge-class-flip"]).bridge_class_flip


def test_fx_pf3_ties_the_resume_to_t_star_and_the_extension_to_no_ramp(mods, tmp_path, monkeypatch):
    T, RL = mods.T, mods.RL
    tstar, other = tmp_path / "a" / "model_100.pt", tmp_path / "b" / "model_130.pt"
    for p, it, c in ((tstar, 100, 5), (other, 130, 9)):
        p.parent.mkdir()
        torch.save({"iter": it, "alg_state": {"counter": c, "learning_rate": 1e-5}}, p)
    monkeypatch.setitem(RL.KNOWN_CHECKPOINTS, RL.sha256_file(tstar),
                        {"label": "T* (fake)", "iter": 100, "counter": 5, "learning_rate": 1e-5, "tstar": True})

    def pf3(**kw):
        return T.pf3(_args(**kw))

    assert pf3(checkpoint=str(tstar))["ok"]                                       # TDR: T*, ramped
    assert pf3(checkpoint=str(tstar), arm="ctl", dr=None)["ok"]                    # TCTL: T*
    r = pf3(checkpoint=str(tstar), dr="0.6:1.35")
    assert not r["ok"] and "ramps" in r["detail"]                                 # a T* fine-tune without the ramp
    r = pf3(checkpoint=str(other), arm="ctl", dr=None)
    assert not r["ok"] and "T* only" in r["detail"]
    r = pf3(checkpoint=str(other))                                                # EXTEND that forgot --dr / counter
    assert not r["ok"] and "--expect-counter" in r["detail"] and "without a ramp" in r["detail"]
    assert not pf3(checkpoint=str(other), dr="0.6:1.35", expect_counter=10)["ok"]  # wrong counter
    assert pf3(checkpoint=str(other), dr="0.6:1.35", expect_counter=9)["ok"]       # the EXTEND launch
    # the real T* is registered as T*
    assert RL.KNOWN_CHECKPOINTS["03b6009a6e952133f6c21c36c8d315b76f945d431615ef6d6798b7f4a5f59fc7"]["tstar"]


def test_fx_dr_min_bound_follows_lo_eff_during_the_ramp(mods):
    T = mods.T
    spec = {"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}                           # 500 iterations
    ok = {"DR/rod_speed_lo": 0.9995, "DR/rod_speed_mean": 1.17, "DR/rod_speed_min": 0.9996, "DR/rod_speed_max": 1.34}
    assert T.dr_block_problems(ok, spec, 0, 24) == []
    early_wide = dict(ok, **{"DR/rod_speed_min": 0.61})                           # logged lo ramps, draws do not
    assert any("rod_speed_min/max" in p for p in T.dr_block_problems(early_wide, spec, 0, 24))
    late = {"DR/rod_speed_lo": 0.6, "DR/rod_speed_mean": 0.975, "DR/rod_speed_min": 0.601, "DR/rod_speed_max": 1.349}
    assert T.dr_block_problems(late, spec, 600, 24) == []
    assert T.dr_block_problems(late, {"lo": 0.6, "hi": 1.35, "ramp_steps": 0}, 0, 24) == []   # no ramp: lo at once


@pytest.mark.skipif(not P3B_DISTILL_ENV.is_file(), reason="no F19 env file")
def test_fx_distill_wrapper_reports_the_g1_pass_guard(tmp_path):
    """Dry runs only (a real launch is never attempted here): without --teacher-policy, and with T* (no G1 PASS
    read for it), the wrapper reports WOULD REFUSE; a real launch dies on the same text before anything is written."""
    ck = tmp_path / "model_100.pt"
    ck.write_bytes(b"x")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "TP_", "DRS_", "RSD_"))}
    for extra, want in (([], "--teacher-policy is required"), (["--teacher-policy", "TSTAR"], "G1 PASS only")):
        r = subprocess.run(["bash", str(SCRIPTS / "rsd_distill.sh"), "--name", "RSD_TEST_DRY", "--checkpoint",
                            str(ck), *extra, "--dry-run"], capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr
        assert "# WOULD REFUSE:" in r.stdout and want in r.stdout and "!= results/g1.json TDR" in r.stdout


_SWEEP_LR = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/dr_stress/lr")


@pytest.mark.skipif(not _SWEEP_LR.is_dir(), reason="the sweep's DRS / DRST stamps are not on this machine")
def test_fx_verdict_layer_on_recorded_sweep_stamps(gl):
    """paired_completion and carve_out (the gates' verdict layer, unit-tested elsewhere on fakes) on the sweep's
    recorded T* DRST stamps: x0.7 L090 vs L090 nominal, es1 + es2 (dose off, the sweep's own pin)."""
    L = gl.L
    C, SR = L.sweep().C, L.sweep().SR
    pin = SR.launch_pin()
    c = L.cell(L.X07)
    x = [L._sweep_read(L.X07, C.TERM[c["term"]], c["level"], c["base"], es, "teacher", pin, False) for es in (1, 2)]
    n = [L._sweep_read(C.nominal_id("L090", "A"), None, None, "L090", es, "teacher", pin, False) for es in (1, 2)]
    assert all(r is not None and r["pass"] is not None for r in x + n)
    comp = [[r["lines"]["field_light_completion"]["value"] for r in side] for side in (x, n)]
    pc = L.paired_completion(x, n)
    assert abs(pc["head"] - sum(comp[0]) / 2) < 1e-9 and abs(pc["ref"] - sum(comp[1]) / 2) < 1e-9   # 100 eps each
    assert abs(pc["margin"] - max(0.05, 2 * pc["se_diff"])) < 1e-12 and pc["n"] == 200
    assert pc["beats_by_m"] == (pc["diff"] >= pc["margin"] - 1e-12)
    assert pc["noninferior"] == (pc["diff"] >= -pc["margin"] - 1e-12) and not pc["noninferior"]       # x0.7 FRAGILE
    self_pc = L.paired_completion(n, n)
    assert self_pc["diff"] == 0 and self_pc["noninferior"] and not self_pc["beats_by_m"]
    cats = sorted(k.split(".", 1)[1] for k in x[0]["lines"] if k.startswith("tracking."))
    assert cats
    for cat in cats[:2]:
        same = L.carve_out(x[0], x[0], cat, 1, 2999)
        assert same["pass"] and same["worse_by"] == 0 and same["margin"]["margin"] >= same["R"] - 1e-12
        co = L.carve_out(x[0], n[0], cat, 1, 2999)
        assert co["R"] is not None and co["drift"] is not None and co["se_diff"] is not None
        want = max(co["R"], 3 * co["se_diff"], abs(co["drift"]) * 2999 / 5000)
        assert abs(co["margin"]["margin"] - want) < 1e-12
        assert co["pass"] == (co["worse_by"] <= co["margin"]["margin"] + 1e-12)
