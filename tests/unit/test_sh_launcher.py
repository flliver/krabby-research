"""Stand-height campaign (``2026-10-13_stand_height``) step 2: the ``stand_height`` eval launcher
``scripts/sh_eval_crab_hex_gait.py`` (+ ``sh_eval_common.py``, ``sh_bridge_read.py``, ``sh_eval_rows.sh`` and the
campaign-local schedule manifest ``scenarios_sh.yaml``). CPU only: nothing here starts Isaac / Kit or a GPU; the
"harness" is the phase-2 / stand-network tests' fake harness, extended with a manifest path, the wrapper clip, the
harness's root / terrain buffers, a hidden clock column and the env's step_dt.

Pinned:

* import / pre-harness hygiene: torch-, numpy- (and, on zero-command rows, yaml-) free before the harness; no bytecode;
  every argv / environment / checkpoint-pairing / manifest refusal fires before the harness file is executed;
* the schedule manifest: loads with the REAL harness loader, both height rows pass the campaign rules (frozen range,
  >= 5 populated bins spanning h_min..h_max, in-stand raises AND lowers, gate windows), labels name their heights, the
  identity rows point at the shared manifests (which carry no height block); every rule refuses its violation; the
  per-step height table (holds, walk holds at 1.064, the prologue);
* the mode: private P2L install; per controller the EXECUTED action is sh_core's router (S1 rows clamp(S1, +-1), stand
  rows q + clamp(a_net) / R_h / R / q) while the RECORDED action keeps the harness's pre-clip convention; the stand
  input is sh_core.stand_net_input with the limited command and restore rows; the env.step swap refuses foreign tensors;
  the H6 restore holds the stand controller with the command columns zeroed until joint-space completion;
* end to end: with the REAL S1 as the stand checkpoint, mask 0, zero command and the restore armed, every raw key of the
  unpatched run is reproduced BITWISE (single row and suite waves, actions included) although S1 saturates beyond +-1
  and the wrapper clip is 4.8; a height row records every series and passes the reader's smoke; scripted controllers,
  restore off, the real A-mask s17 with W[:, 15] zeroed (stand_net == amask_old at a zero command); post-Kit refusals
  write no stamp and still close the app;
* run_arm / rows script / reader (identity miss rule, live, smoke, doctored stamps);
* the residual deadband (CHARTER 2026-10-14): ``--sh-deadband`` parse / refusals (scripted controllers, the identity
  bridge, bad values), the deadbanded executed action on a height row (restore rows included, S1 rows untouched, raws
  keep a_net / q18), its records, the reader's re-derivation at the RECORDED value (absent = 0), the rows script's
  ``--deadband`` and its guards.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
EXP = TASK / "experiments"
CAMP = EXP / "2026-10-13_stand_height"
SCRIPTS = CAMP / "scripts"
LAUNCHER = SCRIPTS / "sh_eval_crab_hex_gait.py"
COMMON = SCRIPTS / "sh_eval_common.py"
READER = SCRIPTS / "sh_bridge_read.py"
ROWS_SH = SCRIPTS / "sh_eval_rows.sh"
SCEN_SH = CAMP / "scenarios_sh.yaml"
P2 = EXP / "2026-10-07_omni_phase2"
P2L_PATH = P2 / "scripts/p2r_mode_eval_crab_hex_gait.py"
RUN_ARM = P2 / "scripts/run_arm.sh"
SN_LAUNCHER = EXP / "2026-10-11_stand_network/scripts/sn_stand_net_eval_crab_hex_gait.py"
REAL_HARNESS = TASK.parent / "crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
SCHEDULE_PY = TASK.parent / "crab_hex_forward_task/scripts/gait_eval/schedule.py"
AMASK = TASK / "policy/stand_net/model_4000.pt"
D3 = ("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
      "tilt_max=0.1,eps_step=0.3")
NP, P0, P1, W, NA = 75, 207, 216, 1149, 18


def _load(path: Path, name: str):
    """By path, never writing ``__pycache__`` next to campaign / phase-2 / predecessor scripts."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


SNT = _load(UNIT / "test_sn_launcher.py", "_sh_sn_launcher_tests")     # stubs at the real layout + real-S1 runner
P2T = SNT.P2T
SC = SNT.SC


@pytest.fixture(scope="module")
def SHL():
    return _load(LAUNCHER, "_t_sh_launcher")


@pytest.fixture(scope="module")
def EC(SHL):
    return SHL.EC


@pytest.fixture(scope="module")
def CORE(SHL):
    return SHL.CORE


@pytest.fixture(scope="module")
def RD():
    return _load(READER, "_t_sh_bridge_read")


def good_env(SHL, **extra) -> dict:
    env = {"KRABBY_OMNI_STAND_OBS": "0", **SHL.SN.ENV4}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _apply(env: dict, patch: dict) -> dict:
    env = dict(env)
    for k, v in patch.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


# ===================================================================================== stubs / fake harness
EXTRA_STUBS = textwrap.dedent('''

    # ---- stand-height tests: the wrapper clip, a scaled actor (S1-like saturation), the hidden clock, hips at
    # their default (restores can complete), the env's step_dt, a manifest-compiled schedule
    ACT_SCALE = 1.0
    HIP_ZERO = False
    CLIP0 = 1.0


    def compile_manifest(path, sid, n, dt=0.02):
        """The harness's compile: segment_id = the hold's INDEX (labels may repeat, e.g. every stand is "stand")."""
        import yaml as _yaml
        data = _yaml.safe_load(open(path).read())
        sc = [s for s in data["scenarios"] if s["id"] == sid][0]
        holds = [(f"h{i}", float(h["speed"]), float(h["face"]), int(round(float(h["hold_s"]) / dt)))
                 for i, h in enumerate(sc["schedule"])]
        comp = compile_holds(holds, n)
        comp.hold_labels = [str(h.get("label")) for h in sc["schedule"]]
        return comp
''')


def stubs_text() -> str:
    s = SNT._stubs_text()
    for old, new in (
            ("        self.num_envs, self.num_actions = n, N_ACT\n",
             "        self.num_envs, self.num_actions = n, N_ACT\n        self.clip_actions = CLIP0\n"
             "        self.executed = []\n"),
            ("        actions = actions.clamp(-1.0, 1.0)\n",
             "        if self.clip_actions is not None:\n"
             "            actions = actions.clamp(-self.clip_actions, self.clip_actions)\n"
             "        self.executed.append(actions.detach().clone())\n"),
            ("        o = self.state.clone()\n",
             "        o = self.state.clone()\n        o[:, 15:17] = 0.0\n        if HIP_ZERO:\n"
             "            o[:, 29:35] = 0.0\n"),
            ("        self.episode_length_buf = torch.zeros(n, dtype=torch.long)\n",
             "        self.episode_length_buf = torch.zeros(n, dtype=torch.long)\n        self.step_dt = 0.02\n"),
            ("        return torch.tanh(obs @ (self.w_hist if hist_encoding else self.w_priv))\n",
             "        return ACT_SCALE * torch.tanh(obs @ (self.w_hist if hist_encoding else self.w_priv))\n"),
            # the real report's per-episode payload also carries segment_id and dt (gait_eval/report.py write_run)
            ("                            **{k: np.asarray(v)[:n, e] for k, v in raw.items()})\n",
             "                            **{k: np.asarray(v)[:n, e] for k, v in raw.items()},\n"
             "                            segment_id=np.asarray(compiled.segment_id)[:n, e],\n"
             "                            dt=np.asarray(run_meta['dt'], dtype=np.float64))\n")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s + EXTRA_STUBS


def harness_text() -> str:
    h = SNT._harness_text()
    for old, new in (
            ('ap.add_argument("--fall", type=str, default="")\n',
             'ap.add_argument("--fall", type=str, default="")\nap.add_argument("--manifest", default=None)\n'
             'ap.add_argument("--scenario", default=None)\n'),
            ("        waves = [SB.compile_holds(SB.SINGLE, N)]\n",
             "        waves = [SB.compile_manifest(args_cli.manifest, args_cli.scenario, N) if args_cli.manifest\n"
             "                 else SB.compile_holds(SB.SINGLE, N)]\n"),
            ("        acts = torch.zeros(T, N, SB.N_ACT)\n",
             "        acts = torch.zeros(T, N, SB.N_ACT)\n        rz = torch.zeros(T, N, 3)\n"
             "        tz = torch.zeros(T, N, 6)\n"),
            ("                acts[t] = actions\n",
             "                acts[t] = actions\n                rz[t, :, 2] = 1.0 + 0.05 * obs[:, 0]\n"
             "                tz[t] = 0.01 * obs[:, 1:7]\n"),
            ("        parts.append(acts.numpy().copy())\n",
             "        parts.append((acts.numpy().copy(), rz.numpy().copy(), tz.numpy().copy()))\n"),
            ('    host = {"actions": np.concatenate(parts, axis=1)}\n',
             '    host = {k: np.concatenate([p[i] for p in parts], axis=1)\n'
             '            for i, k in enumerate(("actions", "root_pos_w", "terrain_z"))}\n'),
            ('"env_vars": dict(SB.ENV_VARS)}',
             '"env_vars": dict(SB.ENV_VARS), "terrain_source": "raycast_mesh", '
             '"hold_labels": list(compiled.hold_labels)}'),
            ('"id": "fake"})()', '"id": args_cli.scenario or "fake"})()')):
        assert h.count(old) == 1, old
        h = h.replace(old, new)
    return h


@pytest.fixture()
def SB(tmp_path, monkeypatch, SHL):
    (tmp_path / "p2r_stubs.py").write_text(stubs_text())
    (tmp_path / "fake_harness.py").write_text(harness_text())
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("p2r_stubs", None)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import p2r_stubs  # noqa: PLC0415
    finally:
        sys.dont_write_bytecode = old
    assert (p2r_stubs.P0, p2r_stubs.P1, p2r_stubs.OBS_W, p2r_stubs.N_ACT) == (P0, P1, W, NA)
    p2r_stubs.StubRunner.builds.clear()
    p2r_stubs.ENV_VARS.clear()
    p2r_stubs.ENV_VARS.update(good_env(SHL))
    yield p2r_stubs
    sys.modules.pop("p2r_stubs", None)


#: a short height row for the fake harness (dt 0.02): walk -> 1.2 -> 0.9 (in-stand lower) -> walk (restore from 0.9);
#: both stands labelled "stand" (positional heights), as the campaign rows
FAKE_ROW = "hfake__omni"
FAKE_MANIFEST = textwrap.dedent(f"""\
    manifest_version: 2
    scenarios:
      - id: {FAKE_ROW}
        task: Fake
        checkpoint: x
        schedule:
          - {{speed: 0.2, dir: 0.0, face: 0.0, hold_s: 0.5, label: fwd}}
          - {{speed: 0.0, dir: 0.0, face: 0.0, hold_s: 5.0, label: stand}}
          - {{speed: 0.0, dir: 0.0, face: 0.0, hold_s: 7.0, label: stand}}
          - {{speed: 0.2, dir: 0.0, face: 0.0, hold_s: 6.0, label: fwd2}}
    stand_height:
      version: 1
      rows:
        {FAKE_ROW}: {{gates: [K2, K4], hold_heights: [null, 1.2, 0.9, null]}}
    """)


def fake_manifest(tmp_path) -> Path:
    p = tmp_path / "scenarios_fake_sh.yaml"
    p.write_text(FAKE_MANIFEST)
    return p


def run_ref(tmp_path, args, seed=0):
    torch.manual_seed(seed)
    old = sys.argv
    sys.argv = [str(tmp_path / "fake_harness.py")] + args
    try:
        g = runpy.run_path(str(tmp_path / "fake_harness.py"), run_name="p2r_harness")
        g["main"]()
    finally:
        sys.argv = old


def run_sh(SHL, tmp_path, args, seed=0, environ=None):
    torch.manual_seed(seed)
    old = sys.argv
    try:
        SHL.main([str(LAUNCHER)] + args, harness=tmp_path / "fake_harness.py",
                 environ=good_env(SHL) if environ is None else environ)
    finally:
        sys.argv = old


def episodes(stamp: Path):
    return [dict(np.load(p)) for p in sorted((Path(stamp) / "raw").glob("episode_*.npz"))]


def meta(stamp: Path) -> dict:
    return json.loads((Path(stamp) / "run_meta.json").read_text())


def sh_args(controller, command, restore="armed", ckpt=None, mask=None):
    a = ["--sh-controller", controller, "--sh-command", command, "--sh-restore", restore]
    if ckpt is not None:
        a += ["--stand-ckpt", str(ckpt), "--stand-mask", str(mask)]
    return a


def sh_infos(mask: int) -> dict:
    """A stand-height trainer checkpoint's provenance (the keys ``trained_arm`` reads)."""
    return {"sh_dagger": {"mode": "dagger", "stand_mask": mask, "warm_start": "A-mask s17 W15 zeroed"}}


def stub_ckpts(SB, tmp_path) -> dict:
    """Stub-format checkpoints: S1; a stand head with S1's estimator as a trainer checkpoint of each mask; the same
    head without provenance; one with its own estimator."""
    for name, seed in (("s1", 1), ("other", 2)):
        SB.save_ckpt(tmp_path / f"{name}.pt", seed)
    s1, other = torch.load(tmp_path / "s1.pt"), torch.load(tmp_path / "other.pt")
    for m in (0, 1):
        torch.save({"policy": other["policy"], "est": s1["est"], "infos": sh_infos(m)}, tmp_path / f"st{m}.pt")
    torch.save({"policy": other["policy"], "est": s1["est"]}, tmp_path / "st_noinfo.pt")
    torch.save({"policy": other["policy"], "est": other["est"], "infos": sh_infos(1)}, tmp_path / "st_badest.pt")
    return {k: str(tmp_path / f"{k}.pt") for k in ("s1", "st0", "st1", "st_noinfo", "st_badest")}


@pytest.fixture(scope="module")
def s1_state():
    if not SC.S1_CKPT.is_file():
        pytest.skip("S1 model_19994.pt is not on this machine")
    ck = torch.load(SC.S1_CKPT, map_location="cpu", weights_only=False)
    return {"model_state_dict": {k: v.detach().cpu().clone() for k, v in ck["model_state_dict"].items()},
            "estimator_state_dict": {k: v.detach().cpu().clone() for k, v in ck["estimator_state_dict"].items()},
            "iter": ck["iter"], "infos": ck["infos"]}


@pytest.fixture()
def real(SB, s1_state, tmp_path):
    """The real-S1 stub world (CPU copy of S1) + the real A-mask s17 file when it is on this machine."""
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        crab_cls, est_cls = SC.rsl_modules()
    finally:
        sys.dont_write_bytecode = old
    pc, ec = SC.s1_cfgs()
    SB.REAL.update(on=True, crab=crab_cls, est=est_cls, policy_cfg=pc, est_cfg=ec)
    p = tmp_path / "s1_cpu.pt"
    torch.save(s1_state, p)
    return {"s1": str(p), "amask": (str(AMASK) if AMASK.is_file() else None)}


# ===================================================================================== import / names
def test_import_and_pre_harness_path_are_torch_numpy_yaml_free(SHL, tmp_path):
    """L6: importing the launcher (sh_eval_common + sh_core + the stand_net launcher + P2L) and everything up to the
    harness import loads no torch / numpy (and no yaml on a zero-command row); no bytecode; the flag left as found."""
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    man = fake_manifest(tmp_path)
    harness = tmp_path / "probe_harness.py"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2R_", "SN_", "SH_"))}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.update(good_env(SHL), CUDA_VISIBLE_DEVICES="")
    for args, yaml_ok in ((sh_args("stand_net", "zero", ckpt=ck, mask=1) + ["--checkpoint", "/s1.pt"], False),
                          (sh_args("r_h", "manifest") + ["--manifest", str(man), "--scenario", FAKE_ROW,
                                                         "--checkpoint", "/s1.pt"], True)):
        report = tmp_path / "pre_kit.json"
        harness.write_text(textwrap.dedent(f"""\
            import json, sys
            json.dump({{"torch": "torch" in sys.modules, "numpy": "numpy" in sys.modules, "yaml": "yaml" in sys.modules,
                       "argv": sys.argv[1:], "bytecode": sys.dont_write_bytecode}}, open({str(report)!r}, "w"))
            raise SystemExit(0)
            """))
        code = textwrap.dedent(f"""\
            import importlib.util, sys
            s = importlib.util.spec_from_file_location("_x", {str(LAUNCHER)!r})
            m = importlib.util.module_from_spec(s)
            sys.dont_write_bytecode = True
            s.loader.exec_module(m)
            sys.dont_write_bytecode = False
            assert not ({{"torch", "numpy", "yaml"}} & set(sys.modules)), "import"
            m.main([{str(LAUNCHER)!r}] + {args!r}, harness={str(harness)!r})
            """)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        got = json.loads(report.read_text())
        assert got["torch"] is False and got["numpy"] is False and got["bytecode"] is False, got
        assert got["yaml"] is yaml_ok
        assert "--checkpoint" in got["argv"] and not any(a.startswith(("--sh", "--stand", "--p2r"))
                                                         for a in got["argv"])
    assert not list(SCRIPTS.glob("__pycache__")) and not list(CAMP.glob("__pycache__"))


def test_launcher_name_matches_run_arm_idle_check():
    assert LAUNCHER.name.endswith("eval_crab_hex_gait.py") and " " not in str(LAUNCHER)
    line = f"/home/nickmagus/krabby/isaac_venv/bin/python {LAUNCHER} --headless --manifest m.yaml"
    pat = r"^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"
    assert subprocess.run(["grep", "-qE", pat], input=line, text=True).returncode == 0
    assert f"drivers_idle() {{ while ps -eo args | grep -qE \"{pat}\"" in RUN_ARM.read_text()


def test_contract_names_and_one_core(SHL, EC, CORE):
    assert SHL.CORE is EC.CORE and SHL.SN is CORE.SN and SHL.SNL.__file__ == str(SN_LAUNCHER)
    assert EC.VARIANT == "stand_height" and EC.VARIANT != SHL.SNL.VARIANT
    assert set(EC.INNER_MODE) == set(EC.CONTROLLERS)
    assert {c for c, m in EC.INNER_MODE.items() if m == "st2"} == set(EC.NET_CONTROLLERS)
    assert set(EC.OLD_PATH_CONTROLLERS) <= set(EC.CONTROLLERS) and set(EC.EXECUTORS) == set(EC.CONTROLLERS)
    assert len(set(EC.RAW_KEYS)) == len(EC.RAW_KEYS) and all(k.startswith("sh_") for k in EC.RAW_KEYS)
    assert EC.RAW_H_MEAS not in EC.MODE_RAW_KEYS and EC.RAW_KEYS[-1] == EC.RAW_H_MEAS
    # the names the K-readers (sh_read.py) import from sh_eval_common
    assert (EC.RAW_STAND_MODE, EC.RAW_STAND_SLOT, EC.RAW_H_CMD, EC.RAW_H_LIM, EC.RAW_Q, EC.RAW_A_NET,
            EC.RAW_RESTORING) == ("p2r_stand_mode", "sh_stand_slot", "sh_h_cmd", "sh_h_lim", "sh_q18", "sh_a_net",
                                  "sh_restoring")
    assert (EC.RAW_ENGAGED, EC.RAW_COMPLETED, EC.RAW_TIMED_OUT, EC.RAW_MERGED) == (
        "sh_engaged", "sh_completed", "sh_timed_out", "sh_merged")
    assert EC.S1_TRAIN_CLIP == 1.0 and CORE.HARNESS_CLIP == 4.8 and CORE.S1_CLIP == 1.0
    # the REFS raw key set never meets the new series (the bridge diff compares REFS's keys only)
    refs = SC.REFS_SUITE1.parent.parent.parent / "walkstop__omni" / "seed001"
    hits = sorted(refs.glob("*/raw/episode_00.npz")) if refs.is_dir() else []
    if hits:
        assert not set(np.load(hits[-1]).files) & set(EC.RAW_KEYS)


# ===================================================================================== the schedule manifest
def _load_registered(path: Path, name: str):
    """By path, registered in sys.modules while it executes (dataclasses resolve their module there)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
        sys.modules.pop(name, None)
    return mod


def test_scenarios_sh_loads_with_the_real_harness_loader(EC):
    S = _load_registered(SCHEDULE_PY, "_t_sh_gait_schedule")
    scen, _ = S.load_manifest(SCEN_SH)
    assert {s.id for s in scen} == {"hwalkstop__omni", "hraise__omni"}
    for s in scen:
        comp = S.compile_schedule(s, dt=0.02, num_envs=2)
        assert comp.hold_labels == EC.height_row(SCEN_SH, s.id)["labels"]
        assert s.command_frame == "world" and s.env_vars == {} and s.episodes == 100
        assert s.total_duration_s <= s.episode_length_s
    raw = yaml.safe_load(SCEN_SH.read_text())
    p2s = yaml.safe_load((P2 / "scenarios_p2s.yaml").read_text())
    for k in ("manifest_version", "speed_ellipse", "ang_vel_max", "defaults"):
        assert raw[k] == p2s[k], k                                        # the header verbatim
    assert all("env" not in s for s in raw["scenarios"])                  # flat ground as walkstop: no env block


def test_height_rows_cover_the_frozen_range_and_both_directions(EC, CORE):
    hw = EC.height_row(SCEN_SH, "hwalkstop__omni")
    hr = EC.height_row(SCEN_SH, "hraise__omni")
    for row in (hw, hr):
        for lab, h in zip(row["labels"], row["hold_heights"]):
            assert (lab == "stand") == (h is not None)                    # every stand is "stand" (p2s_stand's key)
            assert h is None or CORE.H_MIN <= h <= CORE.H_MAX
    hs = sorted({h for h in hw["hold_heights"] if h is not None})
    assert hs[0] == CORE.H_MIN and hs[-1] == CORE.H_MAX and CORE.H_SETTLED in hs
    edges = np.linspace(CORE.H_MIN, CORE.H_MAX, 6)
    counts = np.histogram(hs, bins=edges)[0]
    assert (counts >= 1).all(), counts                                    # >= 5 populated bins h_min..h_max
    assert set(hw["gates"]) >= {"K1", "K6", "K4"} and set(hr["gates"]) >= {"K2", "K4"}
    # hraise: consecutive stand holds whose height changes inside the stand, raises and lowers
    deltas = [b - a for a, b in zip(hr["hold_heights"], hr["hold_heights"][1:]) if a is not None and b is not None]
    assert sum(d > 0.05 for d in deltas) >= 3 and sum(d < -0.05 for d in deltas) >= 3
    # every stand of hwalkstop sits between walks (a hand-back per height)
    by = hw["hold_heights"]
    assert by[0] is None and by[-1] is None and all((a is None) != (b is None) for a, b in zip(by, by[1:]))


def test_identity_rows_are_the_shared_manifests(EC):
    assert tuple(EC.identity_rows()) == EC.IDENTITY_ROW_IDS == ("suite__omni", "walkstop__omni", "field_hard__omni")
    rows = EC.zero_command_rows()
    assert tuple(rows) == EC.ZERO_COMMAND_ROW_IDS == EC.IDENTITY_ROW_IDS + ("field_turn__omni",)
    want = {"suite__omni": EXP / "eval/scenarios_omni.yaml", "walkstop__omni": P2 / "scenarios_p2s.yaml",
            "field_hard__omni": P2 / "scenarios_p2.yaml", "field_turn__omni": P2 / "scenarios_p2.yaml"}
    for sid, path in rows.items():
        assert path == want[sid].resolve() and path.is_file()
        assert any(s.get("id") == sid for s in yaml.safe_load(path.read_text())["scenarios"])
        assert not EC.has_height_row(path, sid) and "stand_height" not in yaml.safe_load(path.read_text())
    assert EC.has_height_row(SCEN_SH, "hwalkstop__omni") and not EC.has_height_row(SCEN_SH, "walkstop__omni")


def _row_yaml(tmp_path, holds, hold_heights, **scen_extra) -> Path:
    s = {"id": "r", "task": "T", "checkpoint": "x", "schedule": holds, **scen_extra}
    p = tmp_path / f"m{len(list(tmp_path.glob('m*.yaml')))}.yaml"
    p.write_text(yaml.safe_dump({"manifest_version": 2, "scenarios": [s],
                                 "stand_height": {"version": 1, "rows": {"r": {"hold_heights": hold_heights}}}}))
    return p


def H(label, speed, hold_s, face=0.0):
    return {"label": label, "speed": speed, "dir": 0.0, "face": face, "hold_s": hold_s}


@pytest.mark.parametrize("holds, hh, extra, needle", [
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("w2", 0.2, 8)], [None, None, None], {}, "without a height"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("w2", 0.2, 8)], [None, 1.0, 1.0], {}, "carries a height"),
    ([H("w", 0.2, 8), H("s", 0.0, 12), H("w2", 0.2, 8)], [None, 1.0, None], {}, "must be labelled 'stand'"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("stand", 0.2, 8), H("w3", 0.2, 8)], [None, 1.0, None, None], {},
     "a walk hold (speed 0.2) labelled 'stand'"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("w2", 0.2, 8)], [None, 0.70, None], {}, "outside the frozen range"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("w2", 0.2, 8)], [None, 1.40, None], {}, "outside the frozen range"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("w2", 0.2, 8)], [None, 1.0], {}, "one per hold"),
    ([H("stand", 0.0, 12), H("w2", 0.2, 8)], [1.0, None], {}, "first hold"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12)], [None, 1.0], {}, "last hold"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12, face=0.5), H("w2", 0.2, 8, face=0.5)], [None, 1.0, None], {},
     "one facing"),
    ([H("w", 0.2, 8), H("stand", 0.0, 6.0), H("w2", 0.2, 8)], [None, 0.719, None], {}, "the 2-s window"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("stand", 0.0, 9.0), H("w2", 0.2, 8)], [None, 1.346, 0.719, None], {},
     "the 2-s window"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("w2", 0.2, 6.0)], [None, 0.719, None], {}, "2 s of walking"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("w2", 0.2, 8)], [None, 1.0, None], {"rotate_holds_by_env": True},
     "rotates"),
    ([H("w", 0.2, 8), H("stand", 0.0, 12), H("w2", 0.2, 8)], [None, True, None], {}, "neither null nor"),
])
def test_height_row_refusals(EC, tmp_path, holds, hh, extra, needle):
    p = _row_yaml(tmp_path, holds, hh, **extra)
    with pytest.raises(ValueError) as ei:
        EC.height_row(p, "r")
    assert needle in str(ei.value), str(ei.value)


def test_height_row_block_refusals_and_the_minimal_valid_row(EC, tmp_path):
    ok = _row_yaml(tmp_path, [H("w", 0.2, 8), H("stand", 0.0, 7.4), H("w2", 0.2, 7.9)], [None, 0.719, None])
    row = EC.height_row(ok, "r")                                          # 4.3125 + 3 <= 7.4 ; 4.3125 + 3.5 <= 7.9
    assert row["labels"] == ["w", "stand", "w2"] and row["hold_heights"] == [None, 0.719, None]
    with pytest.raises(ValueError, match="no height row"):
        EC.height_row(ok, "other")
    d = yaml.safe_load(ok.read_text())
    for patch, needle in (({"version": 2}, "version"),
                          ({"rows": {"r": {"hold_heights": [None, 0.8, None], "extra": 1}}}, "unknown keys"),
                          ({"rows": {"r": {"heights": {"stand": 0.8}}}}, "unknown keys"),
                          ({"rows": {"r": {"hold_heights": []}}}, "non-empty list")):
        dd = copy.deepcopy(d)
        dd["stand_height"].update(patch)
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(dd))
        with pytest.raises(ValueError, match=needle):
            EC.height_row(p, "r")
    with pytest.raises(ValueError, match="no 'stand_height' block"):
        EC.height_row(P2 / "scenarios_p2s.yaml", "walkstop__omni")


def test_height_table_maps_holds_walks_and_the_prologue(EC, CORE):
    seg = np.array([[-1, -1], [0, 1], [0, 1], [1, 2], [2, 2]], dtype=np.int16)
    tab = EC.height_table(seg, [None, 0.8, 1.3])
    assert tab.dtype == np.float64 and tab.shape == (5, 2)
    assert tab[:, 0].tolist() == [CORE.H_SETTLED, CORE.H_SETTLED, CORE.H_SETTLED, 0.8, 1.3]   # prologue = hold 0 (walk)
    assert tab[:, 1].tolist() == [0.8, 0.8, 0.8, 1.3, 1.3]                                   # prologue = its first hold
    assert (EC.height_table(seg, [None, None, None]) == CORE.H_SETTLED).all()
    with pytest.raises(ValueError):
        EC.height_table(seg, [None])
    assert EC.zero_hold_heights(["fwd", "stand", "lat_l", "stand"]) == [None, CORE.H_SETTLED, None, CORE.H_SETTLED]


# ===================================================================================== argv / environment (pre-Kit)
def test_parse_keeps_harness_argv_in_order_and_env_fallbacks(SHL, tmp_path):
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    argv = ["--headless", "--manifest", "m.yaml", "--sh-controller", "stand_net", "--scenario", "walkstop__omni",
            "--stand-ckpt", str(ck), "--env-seed", "1", "--sh-command=zero", "--stand-mask=1", "--checkpoint",
            "/s1.pt", "--save-raw", "--no-plot"]
    o, rest = SHL.parse_sh_args(argv, good_env(SHL))
    assert (o.controller, o.command, o.restore, o.stand_ckpt, o.stand_mask) == ("stand_net", "zero", "armed", str(ck), 1)
    assert type(o.stand_mask) is int and o.height_row is None and o.pre_kind is None
    assert rest == ["--headless", "--manifest", "m.yaml", "--scenario", "walkstop__omni", "--env-seed", "1",
                    "--checkpoint", "/s1.pt", "--save-raw", "--no-plot"]
    assert (o.harness_manifest, o.harness_scenario) == ("m.yaml", "walkstop__omni")
    o, rest = SHL.parse_sh_args(["--scenario", "x"], good_env(SHL, SH_CONTROLLER="r_h", SH_COMMAND="zero",
                                                              SH_RESTORE="off"))
    assert (o.controller, o.command, o.restore, o.stand_ckpt, o.stand_mask, rest) == ("r_h", "zero", "off", None, None,
                                                                                     ["--scenario", "x"])
    o, _ = SHL.parse_sh_args(sh_args("stand_net", "zero", ckpt=ck, mask=0),
                             good_env(SHL, SH_STAND_CKPT=ck, SH_STAND_MASK="0"))
    assert o.stand_mask == 0


def _refusal_cases(ck: str, s1: str):
    g = {}
    net = sh_args("stand_net", "zero", ckpt=ck, mask=1)
    return [
        ([], g, "--sh-controller must be one of"),
        (["--sh-controller", "nope", "--sh-command", "zero"], g, "--sh-controller must be one of"),
        (["--sh-controller", "r_h"], g, "--sh-command must be one of"),
        (["--sh-controller", "r_h", "--sh-command", "height"], g, "--sh-command must be one of"),
        (sh_args("r_h", "zero", restore="maybe"), g, "--sh-restore must be one of"),
        (sh_args("stand_net", "zero"), g, "needs --stand-ckpt"),
        (sh_args("amask_old", "zero") + ["--stand-ckpt", ck], g, "needs --stand-mask"),
        (sh_args("stand_net", "zero", ckpt=ck, mask=2), g, "0 or 1"),
        (sh_args("r_h", "zero") + ["--stand-ckpt", ck], g, "scripted"),
        (sh_args("posture_only", "zero") + ["--stand-mask", "1"], g, "scripted"),
        (sh_args("stand_net", "zero", ckpt="rel/st.pt", mask=1), g, "absolute"),
        (sh_args("stand_net", "zero"), {"SH_STAND_CKPT": ck.replace("st.pt", "s t.pt"), "SH_STAND_MASK": "1"},
         "whitespace"),
        (sh_args("stand_net", "zero", ckpt=ck.replace("st.pt", "s*.pt"), mask=1), g, "glob"),
        (sh_args("stand_net", "zero", ckpt=ck + ".missing", mask=1), g, "not found"),
        (net + ["--stand-mask", "1"], g, "2 times"),
        (net + ["--sh-command=zero"], g, "2 times"),
        (net + ["--sh-extra", "1"], g, "forbidden"),
        (net + ["--stand-maskx", "1"], g, "forbidden"),
        (net + ["--sn-x"], g, "forbidden"),
        (net + ["--p2r-mode", "st2"], g, "forbidden"),
        (net, {"P2R_MODE": "hold"}, "P2R_"),
        (net, {"P2R_BANK_MAX": "5"}, "P2R_"),
        (net, {"SN_STAND_CKPT": ck}, "SN_STAND_CKPT"),
        (net, {"SH_STAND_MASK": "0"}, "disagree"),
        (net, {"SH_CONTROLLER": "r_h"}, "disagree"),
        (net, {"KRABBY_OMNI_STAND_OBS": "clock"}, "STAND_OBS"),
        (net, {"KRABBY_OMNI_STAND_OBS": None}, "STAND_OBS"),
        (net, {"KRABBY_OMNI_STAND_MASK": "1"}, "STAND_MASK"),
        (net, {"KRABBY_RESTANCE": D3}, "KRABBY_RESTANCE"),
        (net, {"KRABBY_OMNI_CLOCK_HIDE": None}, "CLOCK_HIDE"),
        (sh_args("r_h", "manifest"), g, "needs the harness's --manifest"),
        (sh_args("r_h", "manifest") + ["--manifest", str(P2 / "scenarios_p2s.yaml"), "--scenario", "walkstop__omni"],
         g, "no 'stand_height' block"),
        (sh_args("r_h", "manifest") + ["--manifest", str(SCEN_SH), "--scenario", "walkstop__omni"], g,
         "no height row"),
        # checkpoint pairing (hash-based, pre-Kit)
        (sh_args("stand_net", "manifest", ckpt=s1, mask=0) + ["--checkpoint", s1], g, "identity bridge only"),
        (sh_args("amask_old", "zero", ckpt=s1, mask=0) + ["--checkpoint", s1], g, "identity bridge only"),
        (sh_args("amask_old", "zero", ckpt=ck, mask=1), g, "amask_old evaluates the A-mask s17"),
    ]


def test_parse_refusals(SHL, tmp_path):
    ck, s1 = tmp_path / "st.pt", tmp_path / "s1.pt"
    ck.write_bytes(b"x")
    s1.write_bytes(b"s1")
    for argv, patch, needle in _refusal_cases(str(ck), str(s1)):
        with pytest.raises(SystemExit) as ei:
            SHL.parse_sh_args(argv + ["--scenario", "x"] if "--scenario" not in argv else argv,
                              _apply(good_env(SHL), patch))
        assert needle in str(ei.value), (argv, patch, str(ei.value))


@pytest.mark.skipif(not AMASK.is_file(), reason="the A-mask s17 checkpoint is not on this machine")
def test_parse_amask_needs_mask_1(SHL):
    for ctl in ("stand_net", "amask_old"):
        with pytest.raises(SystemExit, match="--stand-mask must be 1"):
            SHL.parse_sh_args(sh_args(ctl, "zero", ckpt=AMASK, mask=0), good_env(SHL))
        o, _ = SHL.parse_sh_args(sh_args(ctl, "zero", ckpt=AMASK, mask=1), good_env(SHL))
        assert o.pre_kind == "amask_s17_w15zeroed"


def test_refusals_never_import_the_harness(SHL, tmp_path):
    ck, s1 = tmp_path / "st.pt", tmp_path / "s1.pt"
    ck.write_bytes(b"x")
    s1.write_bytes(b"s1")
    marker = tmp_path / "harness_ran"
    harness = tmp_path / "marker_harness.py"
    harness.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('1')\n")
    for argv, patch, _ in _refusal_cases(str(ck), str(s1)):
        with pytest.raises(SystemExit):
            SHL.main([str(LAUNCHER)] + argv + ["--checkpoint", str(ck)], harness=harness,
                     environ=_apply(good_env(SHL), patch))
    assert not marker.exists()
    with pytest.raises((KeyError, SystemExit, TypeError)):        # control: a valid call reaches the harness file
        SHL.main([str(LAUNCHER)] + sh_args("r_h", "zero") + ["--checkpoint", str(ck)], harness=harness,
                 environ=good_env(SHL))
    assert marker.exists()


# ===================================================================================== the mode (unit, stubs)
def test_private_install_and_mode_classes(SHL):
    other = _load(P2L_PATH, "_sh_test_other_p2l")
    orig = (other.P2RMode, other.make_factory, other.p2r_block, other.make_write_run)
    for ctl, base in (("stand_net", "StandNetMode"), ("r_h", "P2RMode")):
        p2r = SHL.SNL.load_p2r()
        opts = argparse.Namespace(controller=ctl, command="zero", restore="armed",
                                  stand_ckpt=("/x.pt" if ctl == "stand_net" else None),
                                  stand_mask=(1 if ctl == "stand_net" else None), pre_kind=None, height_row=None,
                                  harness_manifest=None, harness_scenario=None)
        p0 = p2r.P2RMode
        cls = SHL.install(p2r, opts)
        assert p2r.P2RMode is cls and cls.__mro__[1] is SHL._HeightMixin and cls.__mro__[2].__name__ == base
        assert (cls.__mro__[3] if base == "StandNetMode" else cls.__mro__[2]) is p0
        assert cls.INNER == SHL.EC.INNER_MODE[ctl] and "__call__" in vars(SHL._HeightMixin)
        with pytest.raises(RuntimeError, match="already"):
            SHL.install(p2r, opts)
        assert SHL.SNL.load_p2r() is not p2r
    assert (other.P2RMode, other.make_factory, other.p2r_block, other.make_write_run) == orig      # L7


def _mode(SHL, SB, tmp_path, controller="stand_net", command="zero", restore="armed", n=4, mask=1, height_row=None):
    """A StandHeightMode on stubs, wired as install / the factory wire it (S1 runner, ctx, the 4.8 clip)."""
    EC = SHL.EC
    net = controller in EC.NET_CONTROLLERS
    p2r = SHL.SNL.load_p2r()
    ck = tmp_path / f"st_{controller}_{mask}.pt"
    torch.save({"infos": sh_infos(mask)}, ck)
    opts = argparse.Namespace(controller=controller, command=command, restore=restore,
                              stand_ckpt=(str(ck) if net else None), stand_mask=(mask if net else None), pre_kind=None,
                              height_row=height_row, harness_manifest=None, harness_scenario=None)
    cls = SHL.make_mode_class(p2r, opts)
    env = SB.StubEnv(n)
    ct = env.cmd
    p2r.arm_slot_latch(ct, n, "cpu")
    s1, s1_est = SB.StubActor(), SB.StubEst()
    st = st_est = None
    if net:
        st, st_est = SB.StubActor(), SB.StubEst()
        st_est.load_state_dict(s1_est.state_dict())
        m = cls("st2", ct, env.unwrapped, priv=(P0, P1), num_prop=NP, stand=(st.act_inference, st_est))
        m.s1_runner = types.SimpleNamespace(alg=types.SimpleNamespace(estimator=s1_est))
        amask = controller == "amask_old"           # amask_old runs the A-mask s17 file only (its W15 load stubbed)
        m.sn_ctx = types.SimpleNamespace(s1_sha="a" * 64,
                                         specialist={"sha256": SHL.CORE.STAND_NET_SHA256 if amask else "b" * 64})
        if amask:
            m._zero_w15 = lambda: {"stub": True}
    else:
        m = cls("hold", ct, env.unwrapped, priv=(P0, P1), num_prop=NP)
    m.bind_s1(s1.act_inference, s1)
    env.clip_actions = 4.8
    m.clip_before, m.clip_env = 1.0, env
    return env, ct, (s1, s1_est), (st, st_est), m


def _begin(m, n, T=400, labels=("stand",)):
    seg = np.zeros((T, n), np.int16)
    m.begin_pass(types.SimpleNamespace(segment_id=seg, hold_labels=list(labels), n_steps=T))


def _obs(env, s1_est):
    obs, _ = env.get_observations()
    obs[:, P0:P1] = s1_est.inference(obs[:, :NP])          # the harness's in-place S1 estimate
    return obs


ROW_09 = {"manifest": "/x.yaml", "manifest_sha256": "0" * 64, "scenario": "x", "labels": ["stand"],
          "hold_heights": [0.9], "gates": []}


@pytest.mark.parametrize("controller", ["stand_net", "amask_old", "r_h", "r_old", "posture_only"])
@pytest.mark.parametrize("command", ["zero", "manifest"])
def test_step_executes_the_core_router_and_records_pre_clip(SHL, SB, CORE, tmp_path, controller, command):
    SB.ACT_SCALE = 3.0                                     # S1-like saturation: raw outputs beyond +-1
    env, ct, (s1, s1_est), (st, st_est), m = _mode(SHL, SB, tmp_path, controller, command,
                                                   height_row=(ROW_09 if command == "manifest" else None))
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 1, 0, 2])                # stand, move, stand, turn
    _begin(m, 4, labels=("stand",))
    obs = _obs(env, s1_est)
    m.observe(obs)
    before = obs.clone()
    with torch.inference_mode():
        ret = m(obs, hist_encoding=True)
        ex = m.executed_for(ret, env)
    assert torch.equal(obs, before)                        # the harness obs is never written
    h = 0.9 if command == "manifest" else CORE.H_SETTLED
    stand = torch.tensor([True, False, True, False])
    # walk-handover start: the limiter starts at the command; stand rows hold it, the others step toward 1.064
    h_lim = m._cur["rst"]["sh_h_lim"][0].clone()
    assert h_lim[stand].tolist() == [h, h]
    assert torch.allclose(h_lim[~stand], torch.full((2,), min(h + 0.08 * 0.02, CORE.H_SETTLED), dtype=torch.float64),
                          rtol=0, atol=1e-12)
    a_s1 = SB.ACT_SCALE * torch.tanh(before @ s1.w_hist)
    assert s1.calls[-1][0] is True and torch.equal(s1.calls[-1][1], before)
    assert (a_s1.abs() > 1).any()
    assert torch.equal(ret[~stand], a_s1[~stand])                        # recorded: S1's raw (pre-clip) output
    assert torch.equal(ex[~stand], a_s1.clamp(-1, 1)[~stand])            # executed: clamp(S1, +-1)
    old = controller in SHL.EC.OLD_PATH_CONTROLLERS
    hx = CORE.H_SETTLED if old else h_lim
    if controller in ("stand_net", "amask_old"):
        x = CORE.stand_net_input(before, st_est, p0=P0, p1=P1, num_prop=NP, mask=True, h_cmd_limited=hx,
                                 restoring=torch.zeros(4, dtype=torch.bool))
        assert st.calls[-1][0] is True and torch.equal(st.calls[-1][1], x)                # the D1 path + col 15
        want15 = 0.0 if old else (h - CORE.H_SETTLED) / 0.4
        assert abs(float(x[0, 15]) - want15) < 1e-6 and (want15 != 0.0) == (command == "manifest" and not old)
        a_net = SB.ACT_SCALE * torch.tanh(x @ st.w_hist)
        ro = CORE.router_actions(stand, a_net, a_s1, hx)
        assert torch.equal(ex, ro.actions)                                                # sh_core's router
        q = ro.q18
        assert torch.equal(ret[stand], torch.where(q != 0, a_net + q, a_net)[stand])     # recorded pre-clamp
        assert torch.equal(m._cur["rst"]["sh_a_net"][0][stand], a_net[stand])
    elif controller == "r_h":
        tgt, _ = CORE.rh_target(before, h_lim)
        assert torch.equal(ex[stand], tgt[stand]) and torch.equal(ret[stand], tgt[stand])
    elif controller == "r_old":
        tgt, _ = SHL.SN.r_label(before)
        assert torch.equal(ex[stand], tgt[stand]) and torch.equal(ret[stand], tgt[stand])
        assert ex[stand].abs().max() <= 1.0
    else:
        q = CORE.q18(h_lim)
        assert torch.equal(ex[stand], q[stand]) and torch.equal(ret[stand], q[stand])
    if command == "manifest" and not old:
        assert (ex[stand][:, CORE.HIP_ACT] < -1.0).any() or controller == "stand_net"     # authority beyond -1
        assert torch.equal(m._cur["rst"]["sh_q18"][0][stand], CORE.q18(h_lim)[stand])
    rec = m._cur["rst"]
    assert rec["sh_stand_slot"][0].tolist() == [True, False, True, False]
    assert torch.equal(rec["sh_stand_action"][0][stand], ex[stand]) and not rec["sh_stand_action"][0][~stand].any()
    assert torch.equal(rec["sh_s1_action"][0][stand], a_s1.clamp(-1, 1)[stand])
    tgt, br = CORE.rh_target(before, h_lim)
    assert torch.equal(rec["sh_rh_target"][0][stand], tgt[stand]) and (rec["sh_rh_branch"][0][~stand] == -1).all()
    assert torch.equal(rec["sh_rh_branch"][0][stand], br[stand].to(torch.int8))
    assert rec["sh_h_cmd"][0].tolist() == [h] * 4
    assert m._cur["stand"][0].tolist() == stand.tolist() and m.calls == 1 and m.exec_swaps == 1
    c = m.counts()
    assert c["stand_rows_alive"] == 2 and c["s1_exec_abs_max"] <= 1.0


def test_executed_for_refusals(SHL, SB, tmp_path):
    env, ct, (s1, s1_est), _, m = _mode(SHL, SB, tmp_path, "r_h", "zero")
    env.unwrapped.episode_length_buf[:] = 3
    _begin(m, 4)
    with pytest.raises(RuntimeError, match="without a stand_height action"):
        m.executed_for(torch.zeros(4, NA), env)
    with torch.inference_mode():
        obs = _obs(env, s1_est)
        m.observe(obs)
        ret = m(obs)
        with pytest.raises(RuntimeError, match="never reached env.step"):
            m(obs)
        with pytest.raises(RuntimeError, match="did not return"):
            m.executed_for(ret.clone(), env)
        env.clip_actions = 1.0
        with pytest.raises(RuntimeError, match="realised"):
            m.executed_for(ret, env)
        env.clip_actions = 4.8
        assert m.executed_for(ret, env) is not ret and m._ret is None


def test_restore_holds_the_stand_net_with_a_stand_command_until_joint_space_completion(SHL, SB, CORE, tmp_path):
    """H6 in the launcher: a stand at 0.9 m left for a walk engages the restore; the stand net keeps acting (effective
    stand) on an input that shows a stand (cols 5, 8-10 zeroed) with column 15 = the limited command ramping to 1.064;
    completion = limiter at 1.064 + hip rods within delta for 0.2 s; then S1 acts. A 1.064 stand never engages."""
    SB.HIP_ZERO = True
    env, ct, (s1, s1_est), (st, st_est), m = _mode(SHL, SB, tmp_path, "stand_net", "manifest", height_row=ROW_09)
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 0, 0, 1])
    _begin(m, 4, T=400, labels=ROW_09["labels"])
    seen, events = [], []
    with torch.inference_mode():
        obs = _obs(env, s1_est)
        for t in range(260):
            if t == 50:
                ct.slot_id = torch.tensor([1, 0, 1, 1])                   # rows 0, 2 leave the 0.9 stand
            m.observe(obs)
            ret = m(obs)
            seen.append(st.calls[-1][1].clone())
            ex = m.executed_for(ret, env)
            obs, _, dones, _ = env.step(ex)
            obs[:, P0:P1] = s1_est.inference(obs[:, :NP])
            rec = m._cur["rst"]
            events.append({k: rec[k][t].clone() for k in ("sh_engaged", "sh_completed", "sh_restoring",
                                                          "sh_timed_out")})
    eng = torch.stack([e["sh_engaged"] for e in events])
    comp = torch.stack([e["sh_completed"] for e in events])
    rest = torch.stack([e["sh_restoring"] for e in events])
    assert eng[50].tolist() == [True, False, True, False] and int(eng.sum()) == 2
    ramp = int(np.ceil((CORE.H_SETTLED - 0.9) / (0.08 * 0.02) - 1e-9))  # 103 limiter steps from 0.9 to 1.064
    t_arrive = 50 + ramp - 1                                              # the step whose limited command is 1.064
    t_done = t_arrive + 10                                                # then 10 settled observations (0.2 s)
    h_lim = m._cur["rst"]["sh_h_lim"]
    assert h_lim[t_arrive, 0] == CORE.H_SETTLED and h_lim[t_arrive - 1, 0] < CORE.H_SETTLED
    assert comp[t_done, 0] and comp[t_done, 2] and int(comp.sum()) == 2
    assert rest[50:t_done, 0].all() and not rest[t_done:, 0].any() and not rest[:, 1].any()
    assert not torch.stack([e["sh_timed_out"] for e in events]).any()
    x = seen[60]                                                          # the stand net's input at step 60
    assert (x[0, [5, 8, 9, 10]] == 0).all() and (x[1, [5, 8, 9, 10]] != 0).any()      # restore row vs stand row
    assert abs(float(x[0, 15]) - float((h_lim[60, 0] - CORE.H_SETTLED) / 0.4)) < 1e-6
    eff = m._cur["stand"]                                                 # p2r_stand_mode: who acted
    assert eff[60, 0] and not m._cur["rst"]["sh_stand_slot"][60, 0]       # effective stand, latched slot walk
    assert not eff[t_done, 0] and eff[t_done - 1, 0]                      # handed to S1 on the completion step


# ===================================================================================== end to end (fake harness)
@pytest.mark.parametrize("suite", [False, True])
def test_e2e_real_identity_bridge_is_bitwise(SHL, SB, RD, real, tmp_path, suite):
    """S1 (real networks, CPU copy) as the stand checkpoint, mask 0, zero command, restore armed, the 4.8 wrapper clip:
    every raw key of the unpatched run, bitwise -- actions included although S1 saturates beyond +-1 -- and the env
    executed exactly the unpatched run's clamped actions."""
    base = ["--checkpoint", real["s1"], "--fall", "9:1"] + (["--suite"] if suite else [])
    run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    run_sh(SHL, tmp_path, sh_args("stand_net", "zero", ckpt=real["s1"], mask=0) + base + ["--out", str(tmp_path / "id")])
    ref, sh = episodes(tmp_path / "ref" / "stamp"), episodes(tmp_path / "id" / "stamp")
    assert len(ref) == len(sh) == (6 if suite else 3)
    n_stand = 0
    sat = False
    for a, b in zip(ref, sh):
        for k in a:
            assert np.array_equal(a[k], b[k]), k
        sat |= bool((np.abs(a["actions"]) > 1).any())
        st = b["p2r_stand_mode"]
        n_stand += int(st.sum())
        assert np.array_equal(st, b["sh_stand_slot"])                     # no restore: who acted == the slot
        assert np.array_equal(b["sh_stand_action"][st], b["sh_s1_action"][st])
        assert not b["sh_restoring"].any() and (b["sh_h_lim"] == 1.064).all() and not b["sh_q18"].any()
    assert n_stand > 0 and sat                       # the stand rows are live and the S1 clamp is load-bearing
    blk = meta(tmp_path / "id" / "stamp")["p2r_mode"]
    sh_blk = blk["stand_height"]
    assert blk["variant"] == "stand_height" and blk["stand_mask"] == 0 and type(blk["stand_mask"]) is int
    assert sh_blk["stand_ckpt_kind"] == "s1_identity" and sh_blk["stats"]["stand_rows_differ_from_s1"] == 0
    assert sh_blk["clip"]["wrapper_clip_before"] == 1.0 and sh_blk["clip"]["wrapper_clip_realised"] == 4.8
    assert sh_blk["exec_swaps"] == sh_blk["calls"] > 0 and sh_blk["events_alive"]["engaged"] == 0
    assert sh_blk["height_row"] is None and sh_blk["command"] == "zero" and sh_blk["restore"] == "armed"
    assert sh_blk["control"] == "stand_net" and sh_blk["controller"]["restore"] == "armed"
    assert sh_blk["hold_heights"] == ([None, None, None] if suite else [None, 1.064, None, None])   # 1.064 on "stand"
    res = RD.read_row(tmp_path / "ref" / "stamp", tmp_path / "id" / "stamp", how="identity")
    assert res["verdict"] == "PASS" and res["bridge"]["verdict"] == "BITWISE", res
    assert res["meta_fails"] == [] and res["raws"]["fails"] == []
    assert RD.read_row(tmp_path / "ref" / "stamp", tmp_path / "id" / "stamp", how="live")["verdict"] == "FAIL"


def test_e2e_identity_executes_what_the_reference_executed(SHL, SB, real, tmp_path):
    """The env.step swap: the stub env (wrapper clip 4.8 in the launcher run, 1.0 in the reference) received exactly
    the same executed tensors in both runs."""
    base = ["--checkpoint", real["s1"]]
    run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    envs = []
    real_init = SB.StubEnv.__init__

    def spy(self, *a, **k):
        real_init(self, *a, **k)
        envs.append(self)

    SB.StubEnv.__init__ = spy
    try:
        run_ref(tmp_path, base + ["--out", str(tmp_path / "ref2")])
        run_sh(SHL, tmp_path, sh_args("stand_net", "zero", ckpt=real["s1"], mask=0) + base
               + ["--out", str(tmp_path / "id")])
    finally:
        SB.StubEnv.__init__ = real_init
    ref_env, sh_env = envs[0], envs[-1]
    assert ref_env.clip_actions == 1.0 and sh_env.clip_actions == 4.8
    assert len(ref_env.executed) == len(sh_env.executed) > 0
    assert all(torch.equal(a, b) for a, b in zip(ref_env.executed, sh_env.executed))


def _height_run(SHL, SB, tmp_path, controller, *, restore="armed", ckpt=None, mask=None, out="h", extra=()):
    SB.HIP_ZERO = True
    man = fake_manifest(tmp_path)
    envs = []
    real_init = SB.StubEnv.__init__

    def spy(self, *a, **k):
        real_init(self, *a, **k)
        envs.append(self)

    SB.StubEnv.__init__ = spy
    try:
        run_sh(SHL, tmp_path, sh_args(controller, "manifest", restore=restore, ckpt=ckpt, mask=mask)
               + ["--checkpoint", ckpt_s1(SB, tmp_path), "--manifest", str(man), "--scenario", FAKE_ROW,
                  "--out", str(tmp_path / out)] + list(extra))
    finally:
        SB.StubEnv.__init__ = real_init
    return tmp_path / out / "stamp", envs[-1]


def ckpt_s1(SB, tmp_path) -> str:
    p = tmp_path / "s1.pt"
    if not p.is_file():
        SB.save_ckpt(p, 1)
    return str(p)


def test_e2e_height_row_stand_net(SHL, SB, CORE, RD, tmp_path):
    ck = stub_ckpts(SB, tmp_path)
    stamp, env = _height_run(SHL, SB, tmp_path, "stand_net", ckpt=ck["st1"], mask=1)
    m = meta(stamp)
    blk = m["p2r_mode"]
    sh = blk["stand_height"]
    assert blk["variant"] == "stand_height" and blk["stand_mask"] == 1 and blk["specialist"]["sha256"] == \
        hashlib.sha256(Path(ck["st1"]).read_bytes()).hexdigest()
    assert (sh["control"], sh["command"], sh["restore"], sh["inner_mode"]) == ("stand_net", "manifest", "armed", "st2")
    assert sh["stand_ckpt_kind"] == "sh_dagger" and sh["trained_stand_mask"] == 1 and sh["estimator_equals_s1"]
    assert sh["hold_heights"] == [None, 1.2, 0.9, None] == sh["height_row"]["hold_heights"]
    assert sh["height_row"]["scenario"] == FAKE_ROW and sh["height_row"]["gates"] == ["K2", "K4"]
    assert m["hold_labels"] == ["fwd", "stand", "stand", "fwd2"] == sh["height_row"]["labels"] == sh["hold_labels"]
    assert sh["clip"] == {"wrapper_clip_before": 1.0, "wrapper_clip_realised": 4.8, **CORE.clip_record()}
    assert sh["h_meas"]["valid"] is True and sh["controller"]["timeout_steps"] == 266
    assert sh["controller"]["rod_tol_m"] == CORE.RESTORE_ROD_TOL_M and sh["controller"]["engage_m"] == 0.02
    assert sh["controller"]["settle_steps"] == 10 and sh["controller"]["restore"] == "armed"
    assert sh["core"]["sh_core_sha256"] == hashlib.sha256((SCRIPTS / "sh_core.py").read_bytes()).hexdigest()
    assert sh["wrapper_sha256"] == hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()
    assert sh["events_alive"]["engaged"] == 3 and sh["events_alive"]["completed"] == 3
    assert sh["stats"]["hip_below_minus_1_rows"] > 0 and sh["stats"]["s1_exec_abs_max"] <= 1.0
    assert blk["raw_series"][0] == "p2r_stand_mode" and set(blk["raw_series"]) >= set(SHL.EC.RAW_KEYS)
    eps = episodes(stamp)
    T = eps[0]["actions"].shape[0]
    for e, ep in enumerate(eps):
        for k in SHL.EC.RAW_BOOL_KEYS:
            assert ep[k].shape == (T,) and ep[k].dtype == bool, k
        for k in SHL.EC.RAW_H_KEYS + (SHL.EC.RAW_H_MEAS,):
            assert ep[k].shape == (T,) and ep[k].dtype == np.float64, k
        for k in SHL.EC.RAW_ACT_KEYS:
            assert ep[k].shape == (T, NA) and ep[k].dtype == np.float32, k
        assert ep["sh_rh_branch"].shape == (T, 6) and ep["sh_rh_branch"].dtype == np.int8
        np.testing.assert_allclose(ep["sh_h_meas"], ep["root_pos_w"][:, 2].astype(np.float64)
                                   - ep["terrain_z"].astype(np.float64).mean(-1), rtol=0, atol=1e-12)
        # executed (env) == sh_core's router on the recorded series == where(stand, stand action, clamp(actions))
        mode = ep["p2r_stand_mode"]
        exe = torch.stack([x[e] for x in env.executed]).numpy()
        assert np.array_equal(exe, np.where(mode[:, None], ep["sh_stand_action"], np.clip(ep["actions"], -1, 1)))
        s1_raw = torch.from_numpy(np.where(mode[:, None], ep["sh_s1_action"], ep["actions"]))
        ro = CORE.router_actions(torch.from_numpy(mode), torch.from_numpy(ep["sh_a_net"]), s1_raw,
                                 torch.from_numpy(ep["sh_h_lim"]))
        assert np.array_equal(ro.actions.numpy()[mode], exe[mode])
        # the restore: slot walk, stand controller acting, until completion
        r = ep["sh_restoring"]
        assert r.any() and (mode[r]).all() and not ep["sh_stand_slot"][r].any()
        assert ep["sh_h_lim"][r].min() >= 0.9 and ep["sh_h_cmd"][-1] == 1.064
    rd = RD.read_row(None, stamp, how="smoke")
    assert rd["verdict"] == "PASS", (rd["meta_fails"], rd["raws"]["fails"])
    assert rd["raws"]["stand_holds"] == 6 and rd["raws"]["stand_holds_arrived"] == 6
    assert rd["raws"]["restores"] == {"engaged": 3, "completed": 3, "timed_out": 0, "merged": 0, "open_at_end": 0}
    assert RD.read_row(None, stamp, how="live")["verdict"] == "PASS"
    assert RD.main(["--sh-stamp", str(stamp), "--smoke", "--json", str(tmp_path / "r.json")]) == 0


def test_e2e_restore_off_hands_back_at_height(SHL, SB, RD, tmp_path):
    stamp, _ = _height_run(SHL, SB, tmp_path, "r_h", restore="off")
    sh = meta(stamp)["p2r_mode"]["stand_height"]
    assert sh["restore"] == "off" and sh["controller"]["engage_m"] is None and sh["controller"]["restore"] == "off"
    assert sh["events_alive"]["engaged"] == 0 and sh["events_alive"]["leave_off_height"] == 3
    for ep in episodes(stamp):
        assert not ep["sh_restoring"].any() and ep["sh_leave_off_height"].sum() == 1
        t = int(np.argmax(ep["sh_leave_off_height"]))
        assert not ep["p2r_stand_mode"][t] and ep["p2r_stand_mode"][t - 1]       # S1 at once, at height
        assert ep["sh_h_lim"][t] > 0.9 and ep["sh_h_lim"][t] < 0.91
    rd = RD.read_row(None, stamp, how="smoke")
    assert rd["verdict"] == "PASS", (rd["meta_fails"], rd["raws"]["fails"])


@pytest.mark.parametrize("controller", ["r_h", "r_old", "posture_only"])
def test_e2e_scripted_controllers(SHL, SB, CORE, RD, tmp_path, controller):
    stamp, _ = _height_run(SHL, SB, tmp_path, controller)
    blk = meta(stamp)["p2r_mode"]
    sh = blk["stand_height"]
    assert sh["inner_mode"] == "hold" and blk["specialist"] is None and "stand_mask" not in blk
    assert sh["stand_ckpt_kind"] is None and sh["executor"] == SHL.EC.EXECUTORS[controller]
    for ep in episodes(stamp):
        mode = ep["p2r_stand_mode"]
        assert np.array_equal(ep["actions"][mode], ep["sh_stand_action"][mode]) and not ep["sh_a_net"].any()
        if controller == "r_h":
            assert np.array_equal(ep["sh_stand_action"][mode], ep["sh_rh_target"][mode])
        if controller == "r_old":
            assert np.abs(ep["sh_stand_action"][mode]).max() <= 1.0 and not ep["sh_q18"].any()
        if controller == "posture_only":
            assert np.array_equal(ep["sh_stand_action"][mode], ep["sh_q18"][mode])
    rd = RD.read_row(None, stamp, how="smoke")      # (the old +-1 path is not asked for authority beyond +-1)
    assert rd["verdict"] == "PASS", (rd["meta_fails"], rd["raws"]["fails"])
    assert RD.read_row(None, stamp, how="live")["verdict"] == "PASS"


@pytest.mark.skipif(not AMASK.is_file(), reason="the A-mask s17 checkpoint is not on this machine")
def test_e2e_real_amask_w15_zeroed_and_old_path_equals_new_at_zero_command(SHL, SB, CORE, RD, real, tmp_path):
    base = ["--checkpoint", real["s1"]]
    outs = {}
    for ctl in ("stand_net", "amask_old"):
        run_sh(SHL, tmp_path, sh_args(ctl, "zero", ckpt=real["amask"], mask=1) + base
               + ["--out", str(tmp_path / ctl)])
        outs[ctl] = tmp_path / ctl / "stamp"
        sh = meta(outs[ctl])["p2r_mode"]["stand_height"]
        assert sh["stand_ckpt_kind"] == "amask_s17_w15zeroed" and sh["w15"]["w15_max_abs_after"] == 0.0
        assert abs(sh["w15"]["w15_norm_before"] - 1.7404) < 1e-3
        assert sh["w15"]["source_sha256"] == CORE.STAND_NET_SHA256
        assert sh["stats"]["stand_rows_differ_from_s1"] > 0                       # a live stand network
    for a, b in zip(episodes(outs["stand_net"]), episodes(outs["amask_old"])):
        for k in a:
            assert np.array_equal(a[k], b[k]), k                                  # q = 0, column 15 = 0: one path
    rd = RD.read_row(None, outs["stand_net"], how="live")
    assert rd["verdict"] == "PASS", (rd["meta_fails"], rd["raws"]["fails"])


@pytest.mark.skipif(not AMASK.is_file(), reason="the A-mask s17 checkpoint is not on this machine")
def test_e2e_real_amask_plus_q_on_a_height_row(SHL, SB, RD, real, tmp_path):
    SB.HIP_ZERO = True
    man = fake_manifest(tmp_path)
    run_sh(SHL, tmp_path, sh_args("stand_net", "manifest", ckpt=real["amask"], mask=1)
           + ["--checkpoint", real["s1"], "--manifest", str(man), "--scenario", FAKE_ROW, "--out", str(tmp_path / "a")])
    stamp = tmp_path / "a" / "stamp"
    sh = meta(stamp)["p2r_mode"]["stand_height"]
    assert sh["stats"]["hip_below_minus_1_rows"] > 0 and sh["stats"]["stand_rod_exec_abs_max"] > 1.0
    # review item (accepted, reported): the estimate's per-column range on off-height stand rows vs on S1 rows
    st = sh["stats"]
    assert st["est_rows_off_height"] > 0
    assert len(st["est_abs_max_stand_off_height"]) == len(st["est_abs_max_s1_rows"]) > 0
    assert max(st["est_abs_max_s1_rows"]) > 0 and max(st["est_abs_max_stand_off_height"]) > 0
    rd = RD.read_row(None, stamp, how="smoke")
    assert rd["verdict"] == "PASS", (rd["meta_fails"], rd["raws"]["fails"])


@pytest.mark.parametrize("which, mask, extra, needle", [
    ("st0", 1, [], "trained with stand_mask 0"),
    ("st1", 0, [], "trained with stand_mask 1"),
    ("st_noinfo", 1, [], "carries no infos.sh_dagger"),
    ("st_badest", 1, [], "estimator differs"),
])
def test_e2e_post_kit_refusals_write_no_stamp_and_close_the_app(SHL, SB, tmp_path, which, mask, extra, needle):
    ck = stub_ckpts(SB, tmp_path)
    with pytest.raises(SystemExit, match=needle):
        run_sh(SHL, tmp_path, sh_args("stand_net", "zero", ckpt=ck[which], mask=mask) + extra
               + ["--checkpoint", ck["s1"], "--out", str(tmp_path / "bad")])
    assert not (tmp_path / "bad" / "stamp").exists() and (tmp_path / "bad" / "app_closed").exists()


def test_e2e_zero_command_on_a_height_row_and_a_foreign_clip_are_refused(SHL, SB, tmp_path):
    ck = stub_ckpts(SB, tmp_path)
    man = fake_manifest(tmp_path)
    with pytest.raises(SystemExit, match="is a height row"):
        run_sh(SHL, tmp_path, sh_args("r_h", "zero") + ["--checkpoint", ck["s1"], "--manifest", str(man),
                                                        "--scenario", FAKE_ROW, "--out", str(tmp_path / "z")])
    assert not (tmp_path / "z" / "stamp").exists() and (tmp_path / "z" / "app_closed").exists()
    SB.CLIP0 = 2.0
    with pytest.raises(SystemExit, match="not S1's 1.0"):
        run_sh(SHL, tmp_path, sh_args("r_h", "zero") + ["--checkpoint", ck["s1"], "--out", str(tmp_path / "c")])
    assert not (tmp_path / "c" / "stamp").exists() and (tmp_path / "c" / "app_closed").exists()


def test_e2e_harness_schedule_must_be_the_manifest_row(SHL, SB, tmp_path):
    """A compiled schedule whose hold labels are not the height row's is refused before the first step."""
    ck = stub_ckpts(SB, tmp_path)
    man = fake_manifest(tmp_path)
    real_compile = SB.compile_manifest
    SB.compile_manifest = lambda path, sid, n, dt=0.02: SB.compile_holds(
        [("fwd", 0.2, 0.0, 25), ("stand_x", 0.0, 0.0, 30), ("fwd2", 0.2, 0.0, 30)], n)
    try:
        with pytest.raises(RuntimeError, match="not the height row"):
            run_sh(SHL, tmp_path, sh_args("r_h", "manifest") + ["--checkpoint", ck["s1"], "--manifest", str(man),
                                                                "--scenario", FAKE_ROW, "--out", str(tmp_path / "x")])
    finally:
        SB.compile_manifest = real_compile
    assert not (tmp_path / "x" / "stamp").exists()


# ===================================================================================== the reader (doctored stamps)
def _doctor(stamp: Path, fn):
    for p in sorted((stamp / "raw").glob("episode_*.npz")):
        np.savez_compressed(p, **fn(dict(np.load(p))))


def test_reader_identity_miss_rule_and_meta(SHL, SB, RD, real, tmp_path):
    base = ["--checkpoint", real["s1"]]
    run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    run_sh(SHL, tmp_path, sh_args("stand_net", "zero", ckpt=real["s1"], mask=0) + base + ["--out", str(tmp_path / "id")])
    ref, sh = tmp_path / "ref" / "stamp", tmp_path / "id" / "stamp"
    assert RD.main(["--refs-stamp", str(ref), "--sh-stamp", str(sh), "--identity"]) == 0

    def nudge(change):
        def fn(d):
            st = d["p2r_stand_mode"]
            t = int(np.nonzero(~st & (np.arange(st.size) > int(np.argmax(st))))[0][0])
            d["actions"] = d["actions"].copy()
            d["actions"][t, 0] = change(d["actions"][t, 0])
            return d
        return fn

    small = tmp_path / "small"
    shutil.copytree(sh, small)
    _doctor(small, nudge(lambda v: np.nextafter(v, np.float32(np.inf))))
    r = RD.read_row(ref, small, how="identity")
    assert r["bridge"]["verdict"] == "MISS_SMALL" and r["verdict"] == "MISS_SMALL"
    assert RD.main(["--refs-stamp", str(ref), "--sh-stamp", str(small), "--identity"]) == 3
    big = tmp_path / "big"
    shutil.copytree(sh, big)
    _doctor(big, nudge(lambda v: v + np.float32(1e-3)))
    assert RD.read_row(ref, big, how="identity")["verdict"] == "FAIL"
    assert RD.main(["--refs-stamp", str(ref), "--sh-stamp", str(big), "--identity"]) == 4
    bad = tmp_path / "badmeta"
    shutil.copytree(sh, bad)
    m = meta(bad)
    blk = m["p2r_mode"]
    blk["stand_height"]["clip"]["wrapper_clip_realised"] = 1.0
    blk["stand_height"]["exec_swaps"] = blk["stand_height"]["calls"] - 1
    blk["stand_height"]["stats"]["s1_exec_abs_max"] = 1.5
    blk["stand_mask"] = False
    m["env_vars"]["KRABBY_OMNI_STAND_OBS"] = "clock"
    (bad / "run_meta.json").write_text(json.dumps(m))
    fails = RD.meta_reasons(bad, ref, identity=True)
    for needle in ("wrapper_clip_realised", "executed-action swaps", "s1_exec_abs_max", "stand_mask", "STAND_OBS"):
        assert any(needle in f for f in fails), (needle, fails)
    m["p2r_mode"]["variant"] = "stand_net"
    del m["p2r_mode"]["stand_height"]
    (bad / "run_meta.json").write_text(json.dumps(m))
    assert any("block missing" in f for f in RD.meta_reasons(bad, ref))
    assert RD.read_row(None, sh, how="identity")["verdict"] == "MISSING"
    assert RD.exit_code({"a": {"verdict": "PASS"}, "b": {"verdict": "MISSING"}}) == 2


def test_reader_smoke_and_live_catch_doctored_raws(SHL, SB, RD, tmp_path):
    ck = stub_ckpts(SB, tmp_path)
    SB.ACT_SCALE = 3.0                         # raw network outputs beyond +-1: pre-clamp != executed on stand rows
    stamp, _ = _height_run(SHL, SB, tmp_path, "stand_net", ckpt=ck["st1"], mask=1)
    assert RD.read_row(None, stamp, how="smoke")["verdict"] == "PASS"
    cases = {
        "exec": (lambda d: dict(d, sh_stand_action=d["sh_stand_action"] + np.float32(1e-3) * d["p2r_stand_mode"][
            :, None]), "not q + clamp(a_net"),
        "rec": (lambda d: dict(d, actions=np.where(d["p2r_stand_mode"][:, None], d["sh_stand_action"], d["actions"])),
                "pre-clamp"),
        "cmd": (lambda d: dict(d, sh_h_cmd=np.where(d["sh_h_cmd"] == 0.9, 0.95, d["sh_h_cmd"])), "per-hold map"),
        "rate": (lambda d: dict(d, sh_h_lim=np.where(np.arange(d["sh_h_lim"].size) == 40, 1.1, d["sh_h_lim"])),
                 "faster than"),
        "life": (lambda d: dict(d, sh_completed=np.zeros_like(d["sh_completed"])), "lifecycle"),
        "q": (lambda d: dict(d, sh_q18=d["sh_q18"] * np.float32(1.001)), "q18"),
        "mode": (lambda d: dict(d, p2r_stand_mode=d["sh_stand_slot"]), "p2r_stand_mode != sh_stand_slot | sh_restoring"),
    }
    for name, (fn, needle) in cases.items():
        dd = tmp_path / f"doc_{name}"
        shutil.copytree(stamp, dd)
        _doctor(dd, fn)
        r = RD.read_row(None, dd, how="smoke")
        assert r["verdict"] == "FAIL" and any(needle in f for f in r["raws"]["fails"]), (name, r["raws"]["fails"])
    drift = tmp_path / "drift"
    shutil.copytree(stamp, drift)
    m = meta(drift)
    m["p2r_mode"]["stand_height"]["height_row"]["hold_heights"][2] = 0.91
    (drift / "run_meta.json").write_text(json.dumps(m))
    assert any("drift" in f for f in RD.meta_reasons(drift))


# ===================================================================================== run_arm / rows script
SH_EVAL_STUB = textwrap.dedent("""\
    import json, os, sys, pathlib
    a = sys.argv
    sc = a[a.index('--scenario') + 1]
    out = pathlib.Path(a[a.index('--output-root') + 1]) / sc / ('seed%03d' % int(a[a.index('--env-seed') + 1])) / 'ts'
    out.mkdir(parents=True)
    (out / 'complete.json').write_text('{}')
    (out / 'argv.json').write_text(json.dumps({'argv': a, 'env': {k: v for k, v in os.environ.items()
                                                                if k.startswith(('KRABBY_', 'P2R_', 'SN_', 'SH_'))}}))
    """)


def test_run_arm_eval_extra_args_reach_the_launcher_and_parse(SHL, tmp_path):
    ev = tmp_path / "sh_stub_eval_crab_hex_gait.py"
    ev.write_text(SH_EVAL_STUB)
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    extra = f"--sh-controller stand_net --sh-command zero --sh-restore armed --stand-ckpt {ck} --stand-mask 1"
    env = P2T._env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1, P2_EVAL=ev, EVAL_EXTRA_ARGS=extra)
    env = {k: v for k, v in env.items() if not k.startswith(("SN_", "SH_"))}
    st = tmp_path / "stamp"
    st.mkdir()
    (st / "run_meta.json").write_text(json.dumps({"env_vars": good_env(SHL)}))
    script = f"""
        source {RUN_ARM}
        code_guard() {{ return 0; }}
        drivers_idle() {{ :; }}
        write_sidecar() {{ echo '{{}}' > "$1/code_state.json"; }}
        row_eval N {tmp_path}/m.pt {st} {tmp_path}/man.yaml walkstop__omni 1; echo "rc=$? dir=$ROW_DIR"
    """
    r = P2T._bash(script, env)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    got = json.loads((tmp_path / "out/lr/N/walkstop__omni/seed001/ts/argv.json").read_text())
    assert got["argv"][-10:] == extra.split()
    assert got["env"] == good_env(SHL)
    o, rest = SHL.parse_sh_args(got["argv"][1:], got["env"])
    assert (o.controller, o.command, o.restore, o.stand_ckpt, o.stand_mask) == ("stand_net", "zero", "armed", str(ck), 1)
    assert rest[:2] == ["--headless", "--manifest"] and "--checkpoint" in rest and "--save-raw" in rest


@pytest.mark.skipif(not (SC.S1_CKPT.is_file() and (SC.REFS_SUITE1 / "run_meta.json").is_file()),
                    reason="S1 checkpoint / REFS env stamp not on this machine")
def test_rows_script_dry_run_and_guards(EC, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("P2R_", "P2_", "SH_", "SN_STAND_"))}
    log = REPO / "parkour/outputs/omni_plan/stand_height/sh_rows.log"
    before = log.stat().st_mtime_ns if log.exists() else None
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run"], capture_output=True, text=True, timeout=60, env=env)
    assert (log.stat().st_mtime_ns if log.exists() else None) == before            # a dry run writes no log
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    ident = EC.identity_rows()
    assert len(lines) == 3
    for line, row in zip(lines, EC.IDENTITY_ROW_IDS):
        assert f"P2_EVAL={LAUNCHER}" in line and "P2_GUARD_REF=REFH_S1_s18" in line and "P2_ENV_EXTRA= " in line
        assert ("--sh-controller\\ stand_net\\ --sh-command\\ zero\\ --sh-restore\\ armed\\ --stand-ckpt\\ "
                f"{SC.S1_CKPT}\\ --stand-mask\\ 0") in line
        assert f"--row-eval SH_ID_s18 {SC.S1_CKPT} {SC.REFS_SUITE1} {ident[row]} {row} 1" in line
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run", "SH_RH_ctl", "r_h", "manifest", "off", "-", "-",
                        "hwalkstop__omni@2"], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    assert "--sh-controller\\ r_h\\ --sh-command\\ manifest\\ --sh-restore\\ off " in r.stdout
    assert "--stand-ckpt" not in r.stdout and f"{SCEN_SH} hwalkstop__omni 2" in r.stdout
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run", "SH_X", "r_h", "manifest", "armed", "-", "-"],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0 and len(r.stdout.strip().splitlines()) == 2               # default height rows
    trained = tmp_path / "model_100.pt"
    trained.write_bytes(b"x")
    s1_copy = tmp_path / "s1_copy.pt"
    s1_copy.write_bytes(SC.S1_CKPT.read_bytes())
    for args, extra, needle in (
            (["X", "stand_net", "zero", "armed", str(tmp_path / "nope.pt"), "0"], {}, "no stand checkpoint"),
            (["X", "stand_net", "zero", "armed", "rel.pt", "0"], {}, "absolute"),
            (["X", "stand_net", "zero", "armed", str(trained), "2"], {}, "MASK"),
            (["X", "r_h", "zero", "armed", str(trained), "1"], {}, "scripted controller"),
            (["X", "nope", "zero", "armed", "-", "-"], {}, "CONTROLLER"),
            (["X", "r_h", "height", "armed", "-", "-"], {}, "COMMAND"),
            (["X", "r_h", "zero", "maybe", "-", "-"], {}, "RESTORE"),
            (["X", "r_h", "zero", "armed", "-", "-", "hwalkstop__omni@1"], {}, "runs with --sh-command manifest"),
            (["X", "r_h", "manifest", "armed", "-", "-", "walkstop__omni@1"], {}, "runs with --sh-command zero"),
            (["X", "r_h", "zero", "armed", "-", "-", "ring__omni@1"], {}, "no manifest"),
            (["bad-name", "r_h", "zero", "armed", "-", "-"], {}, "NAME"),
            (["X", "r_h", "zero"], {}, "needs CONTROLLER"),
            ([], {"P2_ENV_EXTRA": "KRABBY_OMNI_STAND_OBS=clock"}, "P2_ENV_EXTRA"),
            ([], {"P2R_MODE": "st2"}, "P2R_"),
            ([], {"SH_COMMAND": "zero"}, "SH_"),
            (["SH_ID_s18", "stand_net", "zero", "armed", str(trained), "0"], {}, "identity bridge"),
            (["SH_ID_s18", "stand_net", "zero", "armed", str(SC.S1_CKPT), "1"], {}, "identity bridge"),
            (["SH_ID_s18", "stand_net", "zero", "off", str(SC.S1_CKPT), "0"], {}, "identity bridge"),
            (["SH_A", "stand_net", "zero", "armed", str(s1_copy), "0"], {}, "only under an SH_ID")):
        r = subprocess.run(["bash", str(ROWS_SH), "--dry-run"] + args, capture_output=True, text=True, timeout=60,
                           env={**env, **extra})
        assert r.returncode == 2 and not r.stdout and needle in r.stderr, (args, extra, r.stdout, r.stderr)


def _k5_row_reads():
    """``{row: {env seed, ...}}`` sn_calib reads per head for G1-G5: p2_read.Z2_LIGHT_STAMPS (the light point every
    sn_calib reference resolves; "suite" = the suite__omni row), from the module constants (no stamps needed)."""
    snc = _load(EXP / "2026-10-11_stand_network/scripts/sn_calib.py", "_t_shl_sn_calib")
    rows = {}
    for k in snc.P.Z2_LIGHT_STAMPS:
        row, es = k.rsplit("@", 1)
        rows.setdefault(snc.SUITE if row == "suite" else row, set()).add(int(es))
    return snc, rows


@pytest.mark.skipif(not (SC.S1_CKPT.is_file() and (SC.REFS_SUITE1 / "run_meta.json").is_file()),
                    reason="S1 checkpoint / REFS env stamp not on this machine")
def test_k5_rows_cover_sn_calib_and_the_rows_script_queues_them(EC, tmp_path):
    """Review fix: K5 (sn_calib G1-G5) reads a head's zero-command stamps on suite / walkstop es1-2, field_turn es1 and
    field_hard es1-4; the rows script knows every one of them (field_turn__omni on its shared manifest, command
    zero) and queues the whole K5 list in one pass."""
    snc, need = _k5_row_reads()
    assert need["field_turn__omni"] == {1} and need["field_hard__omni"] == {1, 2, 3, 4}
    got = {}
    for spec in EC.K5_ROWS:
        row, es = spec.split("@")
        got.setdefault(row, set()).add(int(es))
    assert got == need
    assert set(got) == set(EC.ZERO_COMMAND_ROW_IDS)
    text = ROWS_SH.read_text()
    assert " ".join(EC.K5_ROWS) in text.replace("\\\n#        ", "")      # the header's one-pass K5 command
    env = {k: v for k, v in os.environ.items() if not k.startswith(("P2R_", "P2_", "SH_", "SN_STAND_"))}
    ck = tmp_path / "model_100.pt"
    ck.write_bytes(b"x")
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run", "SH_HEAD_s18", "stand_net", "zero", "armed", str(ck), "1",
                        *EC.K5_ROWS], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    ident = EC.zero_command_rows()
    assert len(lines) == len(EC.K5_ROWS)
    for line, spec in zip(lines, EC.K5_ROWS):
        row, es = spec.split("@")
        assert f"--row-eval SH_HEAD_s18 {SC.S1_CKPT} {SC.REFS_SUITE1} {ident[row]} {row} {es}" in line, line
        assert "--sh-command\\ zero" in line
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run", "SH_X", "r_h", "manifest", "armed", "-", "-",
                        "field_turn__omni@1"], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 2 and "runs with --sh-command zero" in r.stderr


def _rows_pat(name: str) -> str:
    import re  # noqa: PLC0415

    m = re.search(rf"^{name}='([^']*)'$", ROWS_SH.read_text(), flags=re.M)
    assert m, f"{name} not found in {ROWS_SH.name}"
    return m.group(1)


def test_rows_script_waits_for_every_isaac_process():
    """Review fix: before each row the rows script waits for ANY Isaac python (trainers, table / ground tests, video
    recorders, train / play, evals and drivers), not just trainers and evals -- the table test's busy list."""
    import re  # noqa: PLC0415

    pat, nonkit, trainer = (re.compile(_rows_pat(k)) for k in ("ISAAC_PAT", "NONKIT_PAT", "TRAINER_PAT"))
    table_pat = re.search(r"^ISAAC_PAT='([^']*)'$", (SCRIPTS / "sh_table_test.sh").read_text(), flags=re.M).group(1)
    assert _rows_pat("ISAAC_PAT") == table_pat
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    busy = [f"{py} /x/sh_table_test.py --headless --controller r_h --prefix a",
            f"{py} /x/sn_table_test.py --headless", f"{py} /x/sh_record_video.py --headless --enable_cameras",
            f"{py} /x/sn_record_video.py --headless", f"{py} /x/play_stream_video.py --task X",
            f"{py} /x/sh_dagger.py --mode dagger --headless", f"{py} /x/sn_dagger.py --mode dagger",
            f"{py} /x/sh_eval_crab_hex_gait.py --headless", f"{py} -u /x/scripts/rsl_rl/train.py --task X",
            f"{py} /x/run_phases.py --x", f"{py} /x/run_candidates.py"]
    for line in busy:
        assert pat.search(line) and not nonkit.search(line), line
    assert trainer.search(busy[5]) and trainer.search(busy[6]) and not trainer.search(busy[0])
    for line in (f"{py} -B /x/sh_dagger.py --read /y", f"{py} /x/sh_dagger.py --emit-env",
                 f"{py} -B /x/sh_table_test.py --merge /y", f"{py} -B /x/sh_table_test.py --emit-facts --out /y",
                 f"{py} /x/scripts/sh_read.py --spec s", f"{py} /x/scripts/sh_bridge_read.py --head H --identity",
                 f"{py} /x/scripts/sh_observe_read.py --head H"):
        assert not (pat.search(line) and not nonkit.search(line)), line
    for line in (f"{py} -m pytest -q tests/unit/test_sh_table.py", f"bash -c {py} /x/sh_table_test.py"):
        assert not pat.search(line), line


# ===================================================================================== the residual deadband (CHARTER 2026-10-14)
def test_parse_deadband_option_and_refusals(SHL, tmp_path):
    ck, s1 = tmp_path / "st.pt", tmp_path / "s1.pt"
    ck.write_bytes(b"x")
    s1.write_bytes(b"s1")
    man = ["--manifest", str(SCEN_SH), "--scenario", "hwalkstop__omni"]
    net = sh_args("stand_net", "manifest", ckpt=ck, mask=1) + man
    o, _ = SHL.parse_sh_args(net, good_env(SHL))
    assert o.deadband == 0.0 and type(o.deadband) is float                       # default: off
    o, rest = SHL.parse_sh_args(net + ["--sh-deadband", "0.25"], good_env(SHL))
    assert o.deadband == 0.25 and "--sh-deadband" not in rest and "0.25" not in rest
    assert SHL.parse_sh_args(net + ["--sh-deadband=0.25"], good_env(SHL))[0].deadband == 0.25
    assert SHL.parse_sh_args(net, good_env(SHL, SH_DEADBAND="0.25"))[0].deadband == 0.25
    assert SHL.parse_sh_args(sh_args("r_h", "manifest") + man + ["--sh-deadband", "0"], good_env(SHL))[0].deadband == 0
    for argv, patch, needle in (
            (net + ["--sh-deadband", "-0.1"], {}, "--sh-deadband"),
            (net + ["--sh-deadband", "1"], {}, "--sh-deadband"),
            (net + ["--sh-deadband", "nan"], {}, "--sh-deadband"),
            (net + ["--sh-deadband", "abc"], {}, "--sh-deadband"),
            (net + ["--sh-deadband", "0.25", "--sh-deadband", "0.25"], {}, "2 times"),
            (net + ["--sh-deadband", "0.25"], {"SH_DEADBAND": "0.3"}, "disagree"),
            (net + ["--sh-deadbandx", "0.25"], {}, "forbidden"),
            (sh_args("r_h", "manifest") + man + ["--sh-deadband", "0.25"], {}, "stand network"),
            (sh_args("posture_only", "manifest") + man, {"SH_DEADBAND": "0.25"}, "stand network"),
            (sh_args("r_old", "zero") + ["--scenario", "x", "--sh-deadband", "0.25"], {}, "stand network"),
            # the identity bridge (S1 as the stand checkpoint) never takes a deadband
            (sh_args("stand_net", "zero", ckpt=s1, mask=0) + ["--checkpoint", str(s1), "--scenario", "x",
                                                             "--sh-deadband", "0.25"], {}, "identity bridge")):
        with pytest.raises(SystemExit) as ei:
            SHL.parse_sh_args(argv, _apply(good_env(SHL), patch))
        assert needle in str(ei.value), (argv, patch, str(ei.value))
    EC = SHL.EC
    assert EC.ckpt_kind_reasons(EC.CKPT_KIND_S1, controller="stand_net", command="zero", stand_mask=0) == []
    assert EC.ckpt_kind_reasons(EC.CKPT_KIND_S1, controller="stand_net", command="zero", stand_mask=0,
                                deadband=0.25)
    assert EC.ckpt_kind_reasons(None, controller="stand_net", command="manifest", stand_mask=1, deadband=0.25) == []


def test_e2e_height_row_stand_net_with_the_deadband(SHL, SB, CORE, RD, tmp_path):
    """--sh-deadband 0.25 on a stand_net height row: recorded in the block (+ its clip), the raws keep a_net and q18
    apart, sh_stand_action and the env's executed action are sh_core's deadbanded router (restore rows included; S1 rows
    untouched); the bridge reader re-derives it at the RECORDED deadband (a record without the key reads as 0)."""
    d = CORE.RESIDUAL_DEADBAND_PROBE
    ck = stub_ckpts(SB, tmp_path)
    stamp, env = _height_run(SHL, SB, tmp_path, "stand_net", ckpt=ck["st1"], mask=1, out="db",
                             extra=["--sh-deadband", str(d)])
    sh = meta(stamp)["p2r_mode"]["stand_height"]
    assert sh["residual_deadband"] == d and sh["clip"]["residual_deadband"] == d
    assert sh["core"]["clip"]["residual_deadband"] == d and CORE.recorded_deadband(sh) == d
    assert sh["trained_residual_deadband"] == 0.0                     # the stub head's infos carry no deadband
    assert sh["clip"] == {"wrapper_clip_before": 1.0, "wrapper_clip_realised": 4.8, **CORE.clip_record(d)}
    eps = episodes(stamp)
    acted = zeroed = 0
    for e, ep in enumerate(eps):
        mode = ep["p2r_stand_mode"]
        a_net, q, sa, h = (torch.from_numpy(ep[k]) for k in ("sh_a_net", "sh_q18", "sh_stand_action", "sh_h_lim"))
        want, wq = CORE.stand_action(a_net, h, deadband=d)
        plain, _ = CORE.stand_action(a_net, h)
        assert torch.equal(sa[mode], want[mode]) and torch.equal(q[mode], wq[mode])
        acted += int((sa[mode] != plain[mode]).any(dim=1).sum())
        rest = torch.from_numpy(ep["sh_restoring"])
        assert rest.any() and torch.equal(sa[rest], want[rest])          # restore rows are stand rows: deadbanded
        small = torch.zeros_like(a_net, dtype=torch.bool)
        small[:, 6:] = a_net[:, 6:].clamp(-1, 1).abs() < d
        sm = small & torch.from_numpy(mode)[:, None]
        zeroed += int(sm.sum())
        assert torch.equal(sa[sm], q[sm])                                # zeroed residual -> the posture itself
        # the raws keep the network's raw output (not the deadbanded residual) and the recorded pre-clip actions
        assert np.array_equal(ep["actions"][mode], np.where(ep["sh_q18"] != 0, ep["sh_a_net"] + ep["sh_q18"],
                                                            ep["sh_a_net"])[mode])
        exe = torch.stack([x[e] for x in env.executed]).numpy()
        assert np.array_equal(exe, np.where(mode[:, None], ep["sh_stand_action"], np.clip(ep["actions"], -1, 1)))
    assert acted > 0 and zeroed > 0
    rd = RD.read_row(None, stamp, how="smoke")
    assert rd["verdict"] == "PASS", (rd["meta_fails"], rd["raws"]["fails"])
    assert rd["raws"]["residual_deadband"] == d
    # trained at 0 (the stub's infos), evaluated at 0.25: both recorded, a NOTE (the probe's mismatch), never a FAIL
    assert rd["trained_residual_deadband"] == 0.0 and len(rd["meta_notes"]) == 1, rd["meta_notes"]
    assert "0.25" in rd["meta_notes"][0] and "trained at 0" in rd["meta_notes"][0]
    # the reader re-derives at the RECORDED deadband: a wrong or missing record makes the executed action a FAIL
    for name, fn, needle in (
            ("missing", lambda b: [b.pop("residual_deadband"), b["clip"].pop("residual_deadband")], "not q + clamp"),
            ("other", lambda b: [b.__setitem__("residual_deadband", 0.3),
                                 b["clip"].__setitem__("residual_deadband", 0.3)], "not q + clamp"),
            ("split", lambda b: b["clip"].__setitem__("residual_deadband", 0.3), "disagrees"),
            ("trained_bad", lambda b: b.__setitem__("trained_residual_deadband", True), "trained_residual_deadband")):
        dd = tmp_path / f"db_{name}"
        shutil.copytree(stamp, dd)
        m = meta(dd)
        fn(m["p2r_mode"]["stand_height"])
        (dd / "run_meta.json").write_text(json.dumps(m))
        r = RD.read_row(None, dd, how="smoke")
        assert r["verdict"] == "FAIL" and any(needle in f for f in r["meta_fails"] + r["raws"]["fails"]), (
            name, r["meta_fails"], r["raws"]["fails"][:3])
    # a stamp written before the option (no key anywhere) reads as deadband 0
    plain_stamp, _ = _height_run(SHL, SB, tmp_path, "stand_net", ckpt=ck["st1"], mask=1, out="plain")
    old = tmp_path / "old_stamp"
    shutil.copytree(plain_stamp, old)
    m = meta(old)
    b = m["p2r_mode"]["stand_height"]
    for rec in (b, b["clip"], b["core"]["clip"]):
        rec.pop("residual_deadband", None)
        rec.pop("residual_deadband_rule", None)
    b.pop("trained_residual_deadband", None)
    (old / "run_meta.json").write_text(json.dumps(m))
    r = RD.read_row(None, old, how="smoke")
    assert r["verdict"] == "PASS" and r["raws"]["residual_deadband"] == 0.0, (r["meta_fails"], r["raws"]["fails"])
    assert r["meta_notes"] == [] and r["trained_residual_deadband"] is None          # nothing recorded: no note
    rp = RD.read_row(None, plain_stamp, how="smoke")                               # trained 0, evaluated 0: no note
    assert rp["verdict"] == "PASS" and rp["meta_notes"] == [] and rp["trained_residual_deadband"] == 0.0
    # a head trained at 0.25 evaluated WITHOUT --sh-deadband (the config-parity break): PASS with the note
    brk = tmp_path / "parity_break"
    shutil.copytree(plain_stamp, brk)
    m = meta(brk)
    m["p2r_mode"]["stand_height"]["trained_residual_deadband"] = d
    (brk / "run_meta.json").write_text(json.dumps(m))
    r = RD.read_row(None, brk, how="smoke")
    assert r["verdict"] == "PASS" and r["trained_residual_deadband"] == d and r["raws"]["residual_deadband"] == 0.0
    assert len(r["meta_notes"]) == 1 and "trained at 0.25" in r["meta_notes"][0], r["meta_notes"]


def test_e2e_scripted_controllers_and_the_bridge_refuse_a_deadband_post_kit(SHL, SB, RD, tmp_path):
    """The mode refuses a deadband on a scripted controller even if handed one past the parser; the reader flags a
    scripted or identity stamp that records one."""
    p2r = SHL.SNL.load_p2r()
    opts = argparse.Namespace(controller="r_h", command="zero", restore="armed", stand_ckpt=None, stand_mask=None,
                              pre_kind=None, height_row=None, harness_manifest=None, harness_scenario=None,
                              deadband=0.25)
    cls = SHL.make_mode_class(p2r, opts)
    env = SB.StubEnv(2)
    with pytest.raises(ValueError, match="deadband"):
        cls("hold", env.cmd, env.unwrapped, priv=(P0, P1), num_prop=NP)
    stamp, _ = _height_run(SHL, SB, tmp_path, "r_h")
    m = meta(stamp)
    m["p2r_mode"]["stand_height"]["residual_deadband"] = 0.25
    (stamp / "run_meta.json").write_text(json.dumps(m))
    assert any("disagrees" in x for x in RD.meta_reasons(stamp))                 # block vs its clip record
    m["p2r_mode"]["stand_height"]["clip"]["residual_deadband"] = 0.25
    (stamp / "run_meta.json").write_text(json.dumps(m))
    assert any("scripted" in x for x in RD.meta_reasons(stamp))


@pytest.mark.skipif(not (SC.S1_CKPT.is_file() and (SC.REFS_SUITE1 / "run_meta.json").is_file()),
                    reason="S1 checkpoint / REFS env stamp not on this machine")
def test_rows_script_deadband_option_and_guards(EC, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("P2R_", "P2_", "SH_", "SN_STAND_"))}
    trained = tmp_path / "model_5000.pt"
    trained.write_bytes(b"x")
    s1_copy = tmp_path / "s1_copy.pt"
    s1_copy.write_bytes(SC.S1_CKPT.read_bytes())
    head = ["SH_P18db", "stand_net", "manifest", "armed", str(trained), "1"]
    rows = ["hwalkstop__omni@1", "hwalkstop__omni@2", "hraise__omni@1", "hraise__omni@2"]
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run", "--deadband", "0.25", *head, *rows], capture_output=True,
                       text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 4
    for line, spec in zip(lines, rows):
        row, es = spec.split("@")
        assert (f"--stand-ckpt\\ {trained}\\ --stand-mask\\ 1\\ --sh-deadband\\ 0.25 " in line
                and f"--row-eval SH_P18db {SC.S1_CKPT} {SC.REFS_SUITE1} {SCEN_SH} {row} {es}" in line), line
    r = subprocess.run(["bash", str(ROWS_SH), "--deadband=0.25", "--dry-run", "SH_P18dbz", "stand_net", "zero", "armed",
                        str(trained), "1", *EC.K5_ROWS], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0 and len(r.stdout.strip().splitlines()) == len(EC.K5_ROWS), r.stderr
    assert all("--sh-deadband\\ 0.25" in ln and "--sh-command\\ zero" in ln for ln in r.stdout.strip().splitlines())
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run", *head], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0 and "--sh-deadband" not in r.stdout                  # off unless asked: the plain path
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run"], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0 and "--sh-deadband" not in r.stdout                  # the identity bridge default
    for args, needle in (
            (["--deadband", "0.25"], "explicit NAME"),
            (["--deadband", "0.25", "SH_ID_s18", "stand_net", "zero", "armed", str(SC.S1_CKPT), "0"], "identity bridge"),
            (["--deadband", "0.25", "SH_ID_x", "stand_net", "zero", "armed", str(trained), "1"], "identity bridge"),
            (["--deadband", "0.25", "SH_A", "stand_net", "zero", "armed", str(s1_copy), "0"],
             "S1 as the stand checkpoint"),
            (["--deadband", "0.25", "X", "r_h", "manifest", "armed", "-", "-"], "stand_net / amask_old"),
            (["--deadband", "0", *head], "0 < X < 1"),
            (["--deadband", "1", *head], "0 < X < 1"),
            (["--deadband", "1.5", *head], "0 < X < 1"),
            (["--deadband", "abc", *head], "plain decimal"),
            (["--deadband", "0.2.5", *head], "plain decimal"),
            (["--deadband", "", *head], "empty"),
            (["--deadband", "0.25", "--deadband", "0.25", *head], "twice"),
            (["--deadband"], "needs a value")):
        r = subprocess.run(["bash", str(ROWS_SH), "--dry-run"] + args, capture_output=True, text=True, timeout=60,
                           env=env)
        assert r.returncode == 2 and not r.stdout and needle in r.stderr, (args, r.stdout, r.stderr)
