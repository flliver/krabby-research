"""Stand-network campaign (``2026-10-11_stand_network``) step 3: the ``stand_net`` launcher variant
``scripts/sn_stand_net_eval_crab_hex_gait.py`` (+ its CPU reader ``sn_bridge_read.py`` and ``sn_eval_rows.sh``).
CPU only: nothing here starts Isaac / Kit or a GPU; the "harness" is the phase-2 tests' fake harness.

Pinned (IMPLEMENTATION_MAP.md section 2.7 items 1-8, risks L1-L11):

* argv / environment refusals fire BEFORE the harness is imported (the harness file is never executed): missing /
  malformed / relative / glob / spaced stand ckpt, missing / bad mask, repeated flags, CLI-env conflicts, leftover
  ``--stand*`` / ``--sn*`` / ``--p2r*``, any ``P2R_*`` variable, STAND_OBS != "0", an env-level stand mask / proxy,
  ``KRABBY_RESTANCE`` (L9, L10); the harness argv keeps its order; the env fallbacks work (run_arm's lr_eval);
* the import and the whole pre-harness path are torch-free (L6); the P2L patches live on a PRIVATE instance (L7);
* ``StandNetMode``: overrides ``__call__`` (the parent relabelled would act S1 everywhere -- L1 control); routes the
  stand net on stand rows and S1 elsewhere with ``torch.where``; never writes the harness obs (L2); S1 sees the
  harness obs; the stand policy sees exactly ``sn_common.stand_net_input(obs, own estimator, mask)`` -- the same call
  the trainer makes -- with ``hist_encoding=True`` (L3, L11); records both actions and R's D3 target / branch on
  stand rows only; refuses a stand estimator that differs from S1's (L5);
* end to end through the fake harness with the REAL S1 networks (a CPU copy of model_19994): S1 as the stand
  checkpoint with mask 0 reproduces the unpatched run BITWISE (the identity-bridge premise; single row and suite
  waves); mask 1 / a perturbed stand net change the stand rows only (pre-entry rows equal REFS); the reader's
  identity / live verdicts; an estimator mismatch never writes a stamp and still closes the app;
* run_meta: variant ``stand_net``, ``stand_mask`` an INT (L4), ``specialist.sha256``, the ``stand_net`` block (L8),
  ``sn_calib.stand_net_reasons`` returns [] on the stamp and rejects a bool mask;
* ``run_arm.sh``: the filename passes ``drivers_idle``'s regex; ``EVAL_EXTRA_ARGS`` reaches the eval and parses;
  ``sn_eval_rows.sh --dry-run`` composes the identity-bridge rows;
* review 2026-10-12: ``bind_s1`` ties a trainer checkpoint to its trained mask (``infos.sn_dagger.stand_mask`` ==
  ``--stand-mask``; a non-trainer checkpoint only if it is S1) -- unit, stub and real-S1 end to end, no stamp on a
  refusal; the stamp records ``trained_stand_mask`` / ``stand_ckpt_kind`` and the reader checks them;
  ``sn_eval_rows.sh`` applies its S1 / mask-0 defaults only to the no-argument identity bridge; the reader's miss rule
  names its own ``bridge_calib.*`` outputs.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import runpy
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
SCRIPTS = TASK / "experiments/2026-10-11_stand_network/scripts"
LAUNCHER = SCRIPTS / "sn_stand_net_eval_crab_hex_gait.py"
READER = SCRIPTS / "sn_bridge_read.py"
ROWS_SH = SCRIPTS / "sn_eval_rows.sh"
SN_CALIB = SCRIPTS / "sn_calib.py"
P2L_PATH = TASK / "experiments/2026-10-07_omni_phase2/scripts/p2r_mode_eval_crab_hex_gait.py"
RUN_ARM = TASK / "experiments/2026-10-07_omni_phase2/scripts/run_arm.sh"
REAL_HARNESS = TASK.parent / "crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
D3 = ("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
      "tilt_max=0.1,eps_step=0.3")
NP, P0, P1, W, NA = 75, 207, 216, 1149, 18


def _load(path: Path, name: str):
    """By path, never writing ``__pycache__`` next to campaign / phase-2 scripts."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


P2T = _load(UNIT / "test_omni_p2r_launcher.py", "_sn_p2r_launcher_tests")   # the phase-2 stubs / fake harness
SC = _load(UNIT / "test_sn_common.py", "_sn_common_tests")                  # real-S1 builders


@pytest.fixture(scope="module")
def SNL():
    return _load(LAUNCHER, "_t_sn_launcher")


@pytest.fixture(scope="module")
def RD():
    return _load(READER, "_t_sn_bridge_read")


@pytest.fixture(scope="module")
def SNC():
    return _load(SN_CALIB, "_t_sn_launcher_calib")


def good_env(SNL, **extra) -> dict:
    env = {"KRABBY_OMNI_STAND_OBS": "0", **SNL.SN.ENV4}
    env.update({k: str(v) for k, v in extra.items()})
    return env


# ===================================================================================== stubs at the real layout
REAL_ADDON = textwrap.dedent('''

    # ---- stand-network tests: the REAL S1 networks behind the stub runner interface (rsl_rl checkpoint format)
    import copy as _copy
    REAL = {"on": False, "crab": None, "est": None, "policy_cfg": None, "est_cfg": None}
    ENV_VARS = {}


    class RealRunner:
        def __init__(self, env, cfg, log_dir=None, device="cpu"):
            cfg["policy"].pop("class_name")
            cfg["estimator"].pop("class_name")
            obs, extras = env.get_observations()
            assert obs.shape[1] == OBS_W and "critic" in extras["observations"]
            StubRunner.builds.append(type(env).__name__)
            self.env = env
            self.alg = types.SimpleNamespace(policy=REAL["crab"](OBS_W, N_ACT, **_copy.deepcopy(REAL["policy_cfg"])),
                                             estimator=REAL["est"](**_copy.deepcopy(REAL["est_cfg"])))
            self.loaded = None

        def load(self, path, load_optimizer=True):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            self.alg.policy.load_state_dict(sd["model_state_dict"], strict=True)
            self.alg.estimator.load_state_dict(sd["estimator_state_dict"], strict=True)
            self.loaded = (str(path), load_optimizer)

        def get_inference_policy(self, device=None):
            self.alg.policy.eval()
            return self.alg.policy.act_inference

        def get_estimator_inference_policy(self, device=None):
            self.alg.estimator.eval()
            return self.alg.estimator


    def make_runner(env, cfg, log_dir=None, device="cpu"):
        if REAL["on"]:
            return RealRunner(env, cfg, log_dir=log_dir, device=device)
        return StubRunner(env, cfg, log_dir=log_dir, device=device)
''')


def _stubs_text() -> str:
    s = P2T.STUBS
    for old, new in (("NUM_PROP, NUM_SCAN, NUM_PRIV = 6, 4, 3", "NUM_PROP, NUM_SCAN, NUM_PRIV = 75, 132, 9"),
                     ("OBS_W, N_ACT = 20, 4", "OBS_W, N_ACT = 1149, 18")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s + REAL_ADDON


def _harness_text() -> str:
    old = '"dt": 0.02}'
    assert P2T.FAKE_HARNESS.count(old) == 1
    return P2T.FAKE_HARNESS.replace(old, '"dt": 0.02, "env_vars": dict(SB.ENV_VARS)}')


@pytest.fixture()
def SB(tmp_path, monkeypatch):
    (tmp_path / "p2r_stubs.py").write_text(_stubs_text())
    (tmp_path / "fake_harness.py").write_text(_harness_text())
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("p2r_stubs", None)
    import p2r_stubs  # noqa: PLC0415

    assert (p2r_stubs.P0, p2r_stubs.P1, p2r_stubs.OBS_W, p2r_stubs.N_ACT) == (P0, P1, W, NA)
    p2r_stubs.StubRunner.builds.clear()
    yield p2r_stubs
    sys.modules.pop("p2r_stubs", None)


@pytest.fixture(scope="module")
def s1_state():
    if not SC.S1_CKPT.is_file():
        pytest.skip("S1 model_19994.pt is not on this machine")
    ck = torch.load(SC.S1_CKPT, map_location="cpu", weights_only=False)
    return {"model_state_dict": {k: v.detach().cpu().clone() for k, v in ck["model_state_dict"].items()},
            "estimator_state_dict": {k: v.detach().cpu().clone() for k, v in ck["estimator_state_dict"].items()},
            "iter": ck["iter"], "infos": ck["infos"]}


@pytest.fixture()
def real(SB, SNL, s1_state, tmp_path):
    """The real-S1 stub world: the CPU copy of S1 + three stand checkpoints (S1 itself, a perturbed head with S1's
    estimator, S1 with a perturbed estimator)."""
    crab_cls, est_cls = SC.rsl_modules()
    pc, ec = SC.s1_cfgs()
    SB.REAL.update(on=True, crab=crab_cls, est=est_cls, policy_cfg=pc, est_cfg=ec)
    SB.ENV_VARS.clear()
    SB.ENV_VARS.update(good_env(SNL))
    paths = {"s1": tmp_path / "s1_cpu.pt", "badest": tmp_path / "st_badest.pt", "head0": tmp_path / "st_head0.pt",
             "head1": tmp_path / "st_head1.pt", "head_noinfo": tmp_path / "st_head_noinfo.pt"}
    torch.save(s1_state, paths["s1"])
    head = copy.deepcopy(s1_state)
    g = torch.Generator().manual_seed(3)
    for k in ("actor.actor_backbone.6.weight", "actor.actor_backbone.6.bias"):
        head["model_state_dict"][k] = head["model_state_dict"][k] + 0.05 * torch.randn(
            head["model_state_dict"][k].shape, generator=g)
    torch.save(head, paths["head_noinfo"])                       # a perturbed head that is not a trainer checkpoint
    for m in (0, 1):                                              # the same weights as a trainer checkpoint per arm
        torch.save(dict(head, infos=sn_infos(m)), paths[f"head{m}"])
    bad = copy.deepcopy(s1_state)
    bad["estimator_state_dict"]["estimator.4.bias"] = bad["estimator_state_dict"]["estimator.4.bias"] + 1e-3
    torch.save(bad, paths["badest"])
    return {k: str(v) for k, v in paths.items()}


def sn_infos(mask: int) -> dict:
    """A trainer checkpoint's provenance (``sn_dagger.py`` ``provenance``: the keys ``trained_arm`` reads)."""
    return {"sn_dagger": {"mode": "dagger", "arm": {1: "mask", 0: "full"}[mask], "stand_mask": mask}}


def stub_ckpts(SB, tmp_path, st_mask: int = 1) -> dict:
    """Stub-format checkpoints: S1, a stand head sharing S1's estimator trained as arm ``st_mask`` (``st``), the same
    head with the other arm's provenance (``st_other``) and without any (``st_noinfo``), one with its own estimator."""
    for name, seed in (("s1", 1), ("other", 2)):
        SB.save_ckpt(tmp_path / f"{name}.pt", seed)
    s1, other = torch.load(tmp_path / "s1.pt"), torch.load(tmp_path / "other.pt")
    torch.save({"policy": other["policy"], "est": s1["est"], "infos": sn_infos(st_mask)}, tmp_path / "st.pt")
    torch.save({"policy": other["policy"], "est": s1["est"], "infos": sn_infos(1 - st_mask)}, tmp_path / "st_other.pt")
    torch.save({"policy": other["policy"], "est": s1["est"]}, tmp_path / "st_noinfo.pt")
    return {"s1": str(tmp_path / "s1.pt"), "st": str(tmp_path / "st.pt"), "badest": str(tmp_path / "other.pt"),
            "st_other": str(tmp_path / "st_other.pt"), "st_noinfo": str(tmp_path / "st_noinfo.pt")}


def run_ref(tmp_path, args, seed=0):
    torch.manual_seed(seed)
    old = sys.argv
    sys.argv = [str(tmp_path / "fake_harness.py")] + args
    try:
        g = runpy.run_path(str(tmp_path / "fake_harness.py"), run_name="p2r_harness")
        g["main"]()
    finally:
        sys.argv = old


def run_sn(SNL, tmp_path, args, seed=0, environ=None):
    torch.manual_seed(seed)
    old = sys.argv
    try:
        SNL.main([str(LAUNCHER)] + args, harness=tmp_path / "fake_harness.py",
                 environ=good_env(SNL) if environ is None else environ)
    finally:
        sys.argv = old


def episodes(stamp: Path):
    return [dict(np.load(p)) for p in sorted((Path(stamp) / "raw").glob("episode_*.npz"))]


def meta(stamp: Path) -> dict:
    return json.loads((Path(stamp) / "run_meta.json").read_text())


# ===================================================================================== argv / environment
def test_parse_keeps_harness_argv_in_order(SNL, tmp_path):
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    argv = ["--headless", "--manifest", "m.yaml", "--stand-ckpt", str(ck), "--scenario", "walkstop__omni",
            "--env-seed", "1", "--stand-mask=1", "--checkpoint", "/s1.pt", "--save-raw", "--no-plot"]
    o, rest = SNL.parse_sn_args(argv, good_env(SNL))
    assert (o.stand_ckpt, o.stand_mask) == (str(ck), 1) and type(o.stand_mask) is int
    assert rest == ["--headless", "--manifest", "m.yaml", "--scenario", "walkstop__omni", "--env-seed", "1",
                    "--checkpoint", "/s1.pt", "--save-raw", "--no-plot"]


def test_env_fallbacks_and_agreement(SNL, tmp_path):
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    o, rest = SNL.parse_sn_args(["--scenario", "x"], good_env(SNL, SN_STAND_CKPT=ck, SN_STAND_MASK="0"))
    assert (o.stand_ckpt, o.stand_mask, rest) == (str(ck), 0, ["--scenario", "x"])
    o, _ = SNL.parse_sn_args(["--stand-ckpt", str(ck), "--stand-mask", "1"],
                             good_env(SNL, SN_STAND_CKPT=ck, SN_STAND_MASK="1"))
    assert o.stand_mask == 1
    o, _ = SNL.parse_sn_args(["--stand-ckpt", str(ck), "--stand-mask", "0"], good_env(SNL, SN_STAND_CKPT=""))
    assert o.stand_mask == 0


def _refusal_cases(ck: str):
    g = {}
    return [
        ([], g, "needs --stand-ckpt"),
        (["--stand-ckpt", ck], g, "needs --stand-mask"),
        (["--stand-ckpt", ck, "--stand-mask", "2"], g, "0 or 1"),
        (["--stand-ckpt", ck, "--stand-mask", "true"], g, "0 or 1"),
        (["--stand-ckpt", "rel/st.pt", "--stand-mask", "0"], g, "absolute"),
        ([], {"SN_STAND_CKPT": ck.replace("st.pt", "s t.pt"), "SN_STAND_MASK": "0"}, "whitespace"),
        (["--stand-ckpt", ck.replace("st.pt", "s*.pt"), "--stand-mask", "0"], g, "glob"),
        (["--stand-ckpt", ck + ".missing", "--stand-mask", "0"], g, "not found"),
        (["--stand-ckpt", ck, "--stand-mask", "0", "--stand-mask", "0"], g, "2 times"),
        (["--stand-ckpt", ck, "--stand-ckpt=" + ck, "--stand-mask", "0"], g, "2 times"),
        (["--stand-ckpt", ck, "--stand-maskx", "0"], g, "forbidden"),
        (["--stand-ckpt", ck, "--stand-mask", "0", "--sn-extra", "1"], g, "forbidden"),
        (["--stand-ckpt", ck, "--stand-mask", "0", "--p2r-mode", "st2"], g, "forbidden"),
        (["--stand-ckpt", ck, "--stand-mask", "0", "--p2r-observe"], g, "forbidden"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"P2R_MODE": "hold"}, "P2R_"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"P2R_BANK_MAX": "5"}, "P2R_"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"SN_STAND_MASK": "1"}, "disagree"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"SN_STAND_CKPT": ck + "x"}, "disagree"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"KRABBY_OMNI_STAND_OBS": "clock"}, "STAND_OBS"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"KRABBY_OMNI_STAND_OBS": None}, "STAND_OBS"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"KRABBY_OMNI_STAND_MASK": "1"}, "STAND_MASK"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"KRABBY_OMNI_STAND_PROXY": "hold"}, "STAND_PROXY"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"KRABBY_RESTANCE": D3}, "KRABBY_RESTANCE"),
        (["--stand-ckpt", ck, "--stand-mask", "0"], {"KRABBY_OMNI_CLOCK_HIDE": None}, "CLOCK_HIDE"),
    ]


def _apply(env: dict, patch: dict) -> dict:
    env = dict(env)
    for k, v in patch.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


def test_parse_refusals(SNL, tmp_path):
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    for argv, patch, needle in _refusal_cases(str(ck)):
        with pytest.raises(SystemExit) as ei:
            SNL.parse_sn_args(argv + ["--scenario", "x"], _apply(good_env(SNL), patch))
        assert needle in str(ei.value), (argv, patch, str(ei.value))


def test_refusals_never_import_the_harness(SNL, tmp_path):
    """Every pre-Kit refusal fires before the harness file is executed (so before AppLauncher / Kit)."""
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    marker = tmp_path / "harness_ran"
    harness = tmp_path / "marker_harness.py"
    harness.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('1')\n")
    for argv, patch, _ in _refusal_cases(str(ck)):
        with pytest.raises(SystemExit):
            SNL.main([str(LAUNCHER)] + argv + ["--checkpoint", str(ck)], harness=harness,
                     environ=_apply(good_env(SNL), patch))
    assert not marker.exists()
    # control: a valid call does reach the harness file (it then fails: the marker harness has no main())
    with pytest.raises((KeyError, SystemExit, TypeError)):
        SNL.main([str(LAUNCHER), "--stand-ckpt", str(ck), "--stand-mask", "0", "--checkpoint", str(ck)],
                 harness=harness, environ=good_env(SNL))
    assert marker.exists()


def test_launcher_name_matches_run_arm_idle_check():
    assert LAUNCHER.name.endswith("eval_crab_hex_gait.py") and " " not in str(LAUNCHER)
    line = f"/home/nickmagus/krabby/isaac_venv/bin/python {LAUNCHER} --headless --manifest m.yaml"
    pat = r"^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"
    assert subprocess.run(["grep", "-qE", pat], input=line, text=True).returncode == 0
    assert f"drivers_idle() {{ while ps -eo args | grep -qE \"{pat}\"" in RUN_ARM.read_text()


def test_import_and_pre_harness_path_are_torch_free(SNL, tmp_path):
    """L6: importing the launcher (sn_common + P2L) and everything up to the harness import loads no torch."""
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    report = tmp_path / "pre_kit.json"
    harness = tmp_path / "probe_harness.py"
    harness.write_text(textwrap.dedent(f"""\
        import json, sys
        json.dump({{"torch": "torch" in sys.modules, "numpy": "numpy" in sys.modules,
                   "argv": sys.argv[1:], "bytecode": sys.dont_write_bytecode}}, open({str(report)!r}, "w"))
        raise SystemExit(0)
        """))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2R_", "SN_"))}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.update(good_env(SNL), CUDA_VISIBLE_DEVICES="")
    code = textwrap.dedent(f"""\
        import importlib.util, sys
        s = importlib.util.spec_from_file_location("_x", {str(LAUNCHER)!r})
        m = importlib.util.module_from_spec(s)
        sys.dont_write_bytecode = True          # no __pycache__ for the launcher file itself (test hygiene) ...
        s.loader.exec_module(m)
        sys.dont_write_bytecode = False         # ... and the process flag the launcher must leave untouched
        assert "torch" not in sys.modules and "numpy" not in sys.modules, "import"
        m.main([{str(LAUNCHER)!r}, "--stand-ckpt", {str(ck)!r}, "--stand-mask", "1", "--checkpoint", "/s1.pt"],
               harness={str(harness)!r})
        """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    got = json.loads(report.read_text())
    assert got == {"torch": False, "numpy": False, "argv": ["--checkpoint", "/s1.pt"], "bytecode": False}


def test_private_p2l_instance_and_single_install(SNL):
    other = _load(P2L_PATH, "_sn_test_other_p2l")
    orig = (other.P2RMode, other.make_factory, other.p2r_block)
    p2r = SNL.load_p2r()
    assert p2r is not other and p2r.HARNESS == REAL_HARNESS and p2r.P2RMode is not other.P2RMode
    base = p2r.P2RMode
    cls = SNL.install(p2r, types.SimpleNamespace(stand_mask=1))
    assert p2r.P2RMode is cls and cls.__mro__[1] is base and cls.STAND_MASK == 1
    assert not any(isinstance(getattr(m, "__dict__", None), dict) and m.__dict__.get("_sn_installed") is True
                   for m in list(sys.modules.values()))                                # never registered
    assert (other.P2RMode, other.make_factory, other.p2r_block) == orig          # L7: nothing leaked
    with pytest.raises(RuntimeError, match="already"):
        SNL.install(p2r, types.SimpleNamespace(stand_mask=0))
    assert SNL.load_p2r() is not p2r                                              # a fresh instance every time
    assert "__call__" in vars(cls) and "bind_s1" in vars(cls)                     # L1: the override exists


def test_mode_class_refusals(SNL, SB):
    p2r = SNL.load_p2r()
    for bad in (True, 2, "1"):
        with pytest.raises(ValueError):
            SNL.make_mode_class(p2r, bad)
    cls = SNL.make_mode_class(p2r, 0)
    env = SB.StubEnv(2)
    with pytest.raises(ValueError):
        cls("hold", env.cmd, env.unwrapped, priv=(P0, P1), num_prop=NP)
    with pytest.raises(ValueError):
        cls("st2", env.cmd, env.unwrapped, priv=(P0, P1), num_prop=NP, stand=None)


# ===================================================================================== StandNetMode (unit, stubs)
S1_SHA_STUB, ST_SHA_STUB = "a" * 64, "b" * 64


def _mode(SNL, SB, flag, tmp_path, n=4, same_est=True, trained="flag", ctx="trainer"):
    """A StandNetMode on stubs, wired as ``install`` / ``make_factory`` wire it: the stand checkpoint path (a file
    carrying ``infos.sn_dagger.stand_mask`` = ``trained``; None = no ``infos.sn_dagger``) and P2L's ctx (``s1_sha``,
    ``specialist.sha256``; ``ctx="s1"`` makes the stand checkpoint S1 itself)."""
    p2r = SNL.load_p2r()
    trained = flag if trained == "flag" else trained
    ck = tmp_path / f"st_{flag}_{trained}.pt"
    torch.save({} if trained is None else {"infos": {"sn_dagger": {"mode": "dagger", "stand_mask": trained}}}, ck)
    cls = SNL.make_mode_class(p2r, flag, str(ck))
    env = SB.StubEnv(n)
    ct = env.cmd
    p2r.arm_slot_latch(ct, n, "cpu")
    s1, s1_est = SB.StubActor(), SB.StubEst()
    st, st_est = SB.StubActor(), SB.StubEst()
    if same_est:
        st_est.load_state_dict(s1_est.state_dict())
    m = cls("st2", ct, env.unwrapped, priv=(P0, P1), num_prop=NP, stand=(st.act_inference, st_est))
    m.s1_runner = types.SimpleNamespace(alg=types.SimpleNamespace(estimator=s1_est))
    m.sn_ctx = types.SimpleNamespace(s1_sha=S1_SHA_STUB,
                                     specialist={"sha256": S1_SHA_STUB if ctx == "s1" else ST_SHA_STUB})
    return p2r, env, ct, (s1, s1_est), (st, st_est), m


def _begin(m, n, T=6):
    m.begin_pass(types.SimpleNamespace(segment_id=np.zeros((T, n), np.int16), hold_labels=["stand"], n_steps=T))


@pytest.mark.parametrize("flag", [0, 1])
def test_mode_routes_records_and_never_writes_obs(SNL, SB, flag, tmp_path):
    p2r, env, ct, (s1, s1_est), (st, st_est), m = _mode(SNL, SB, flag, tmp_path)
    m.bind_s1(s1.act_inference, s1)
    assert m.estimator_equals_s1 is True and m.r_params.to_dict() == SNL.SN.r_params().to_dict()
    assert m.trained_stand_mask == flag and m.stand_ckpt_kind == "sn_dagger"
    assert m.variant == "stand_net" and m.stand_mask == flag and type(m.stand_mask) is int
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 1, 0, 2])                       # stand, move, stand, turn
    _begin(m, 4)
    obs, _ = env.get_observations()
    m.observe(obs)
    obs[:, P0:P1] = s1_est.inference(obs[:, :NP])                  # the harness's in-place S1 estimate
    before = obs.clone()
    with torch.inference_mode():
        a = m(obs, hist_encoding=True)
    assert torch.equal(obs, before)                                # L2: the harness obs is never written
    assert s1.calls[-1][0] is True and torch.equal(s1.calls[-1][1], before)      # S1 sees the harness obs
    x = SNL.SN.stand_net_input(before, st_est, p0=P0, p1=P1, num_prop=NP, mask=flag)
    assert st.calls[-1][0] is True and torch.equal(st.calls[-1][1], x)           # D1 input, hist path (L11)
    if flag:
        rst = SNL.SN.load_rst()
        assert torch.equal(x, rst.stand_mask_obs(before))          # own estimate == S1's here, so: mask(harness obs)
    else:
        assert torch.equal(x, before)
    a_st, a_s1 = torch.tanh(x @ st.w_hist), torch.tanh(before @ s1.w_hist)
    stand = torch.tensor([True, False, True, False])
    assert torch.equal(a[stand], a_st[stand]) and torch.equal(a[~stand], a_s1[~stand])
    rec = m._cur["rst"]
    tgt, br = SNL.SN.r_label(before)
    assert torch.equal(rec["sn_stand_action"][0][stand], a_st[stand]) and not rec["sn_stand_action"][0][~stand].any()
    assert torch.equal(rec["sn_s1_action"][0][stand], a_s1[stand]) and not rec["sn_s1_action"][0][~stand].any()
    assert torch.equal(rec["sn_r_target"][0][stand], tgt[stand]) and not rec["sn_r_target"][0][~stand].any()
    assert torch.equal(rec["sn_r_branch"][0][stand], br[stand].to(torch.int8))
    assert (rec["sn_r_branch"][0][~stand] == -1).all()
    assert m._cur["stand"][0].tolist() == stand.tolist() and m._cur["t"] == 1 and m.calls == 1
    assert m.counts()["stand_rows_alive"] == 2


def test_launcher_and_trainer_make_the_same_input_call(SNL, SB, monkeypatch, tmp_path):
    """Map 2.7 item 3: the input the stand policy receives in StandNetMode is exactly what the trainer's call --
    ``sn_common.stand_net_input(obs, est, p0, p1, num_prop from the estimator cfg, mask=arm)`` -- returns on the same
    obs and estimator (the one shared function; no launcher-private variant)."""
    for flag in (0, 1):
        p2r, env, ct, (s1, s1_est), (st, st_est), m = _mode(SNL, SB, flag, tmp_path)
        m.bind_s1(s1.act_inference, s1)
        seen = []
        real = SNL.SN.stand_net_input

        def spy(obs, estimator, **kw):
            out = real(obs, estimator, **kw)
            seen.append((estimator, dict(kw), out.clone()))
            return out

        monkeypatch.setattr(SNL.SN, "stand_net_input", spy)
        env.unwrapped.episode_length_buf[:] = 3
        ct.slot_id = torch.tensor([0, 0, 1, 0])
        _begin(m, 4)
        obs, _ = env.get_observations()
        m.observe(obs)
        obs[:, P0:P1] = s1_est.inference(obs[:, :NP])
        with torch.inference_mode():
            m(obs, hist_encoding=True)
        monkeypatch.setattr(SNL.SN, "stand_net_input", real)
        assert len(seen) == 1
        est_used, kw, launcher_input = seen[0]
        assert est_used is st_est and kw == {"p0": P0, "p1": P1, "num_prop": NP, "mask": bool(flag)}
        cfg = SB.cfg_dict()["estimator"]                                   # the trainer's source of p0 / p1 / num_prop
        tp0 = cfg["num_prop"] + cfg["num_scan"]
        trainer_input = real(obs, st_est, p0=tp0, p1=tp0 + cfg["num_priv_explicit"], num_prop=cfg["num_prop"],
                             mask=bool(flag))
        assert torch.equal(launcher_input, trainer_input) and torch.equal(st.calls[-1][1], trainer_input)


def test_bind_s1_refuses_a_different_estimator(SNL, SB, tmp_path):
    p2r, env, ct, (s1, s1_est), (st, st_est), m = _mode(SNL, SB, 0, tmp_path, same_est=False)
    with pytest.raises(SystemExit, match="estimator differs"):
        m.bind_s1(s1.act_inference, s1)
    assert m.estimator_equals_s1 is None
    p2r, env, ct, (s1, s1_est), (st, st_est), m = _mode(SNL, SB, 0, tmp_path)
    m.s1_runner = None
    with pytest.raises(RuntimeError, match="runner"):
        m.bind_s1(s1.act_inference, s1)


@pytest.mark.parametrize("flag", [0, 1])
def test_bind_s1_ties_the_checkpoint_to_its_trained_mask(SNL, SB, tmp_path, flag):
    """Review 2026-10-12 (blocking): a trainer checkpoint is evaluated only with the mask it was trained with
    (``infos.sn_dagger.stand_mask``); a checkpoint without ``infos.sn_dagger`` only if it is S1 itself; every refusal
    is a SystemExit in bind_s1 (post-Kit, before the first step) and leaves the mode unbound."""
    # the other arm's checkpoint -> refused
    p2r, env, ct, (s1, _), _, m = _mode(SNL, SB, flag, tmp_path, trained=1 - flag)
    with pytest.raises(SystemExit, match=f"--stand-mask {flag} but the checkpoint was trained with stand_mask "
                                         f"{1 - flag}"):
        m.bind_s1(s1.act_inference, s1)
    assert m.estimator_equals_s1 is None and m.stand_ckpt_kind is None and m.r_params is None
    # malformed provenance -> refused (a bool / a string / out of range / missing)
    for i, bad in enumerate((True, "1", 2, None)):
        (tmp_path / f"b{i}").mkdir()
        p2r, env, ct, (s1, _), _, m = _mode(SNL, SB, flag, tmp_path / f"b{i}", trained=None)
        torch.save({"infos": {"sn_dagger": {"mode": "dagger", "stand_mask": bad}}}, m.stand_ckpt)
        with pytest.raises(SystemExit, match="not an int"):
            m.bind_s1(s1.act_inference, s1)
    # no infos.sn_dagger and not S1 -> refused; S1 itself -> the identity kind, any mask
    p2r, env, ct, (s1, _), _, m = _mode(SNL, SB, flag, tmp_path, trained=None)
    with pytest.raises(SystemExit, match="is not S1 itself"):
        m.bind_s1(s1.act_inference, s1)
    p2r, env, ct, (s1, _), _, m = _mode(SNL, SB, flag, tmp_path, trained=None, ctx="s1")
    m.bind_s1(s1.act_inference, s1)
    assert m.stand_ckpt_kind == "s1_identity" and m.trained_stand_mask is None and m.estimator_equals_s1 is True
    # the matching arm -> bound
    p2r, env, ct, (s1, _), _, m = _mode(SNL, SB, flag, tmp_path)
    m.bind_s1(s1.act_inference, s1)
    assert m.stand_ckpt_kind == "sn_dagger" and m.trained_stand_mask == flag and type(m.trained_stand_mask) is int
    # wiring missing -> RuntimeError (never a silent pass)
    p2r, env, ct, (s1, _), _, m = _mode(SNL, SB, flag, tmp_path, trained=None)
    m.sn_ctx = None
    with pytest.raises(RuntimeError, match="context"):
        m.bind_s1(s1.act_inference, s1)
    p2r, env, ct, (s1, _), _, m = _mode(SNL, SB, flag, tmp_path)
    m.stand_ckpt = None
    with pytest.raises(RuntimeError, match="path"):
        m.bind_s1(s1.act_inference, s1)


def test_control_parent_call_relabelled_acts_s1_everywhere(SNL, SB, tmp_path):
    """L1 control: the parent's __call__ with the stand_net label falls into P2L's probe branch -- S1 on every row
    (a stamp that looks like stand_net but is REFS). The override is load-bearing."""
    p2r, env, ct, (s1, s1_est), (st, st_est), m = _mode(SNL, SB, 0, tmp_path)
    m.bind_s1(s1.act_inference, s1)
    parent = p2r.P2RMode("st2", ct, env.unwrapped, priv=(P0, P1), num_prop=NP, stand=(st.act_inference, st_est))
    parent.bind_s1(s1.act_inference, s1)
    parent.variant = "stand_net"
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 0, 0, 0])
    outs = []
    for mode in (parent, m):
        _begin(mode, 4)
        obs, _ = env.get_observations()
        mode.observe(obs)
        obs[:, P0:P1] = s1_est.inference(obs[:, :NP])
        with torch.inference_mode():
            outs.append((mode(obs, hist_encoding=True), torch.tanh(obs @ s1.w_hist), torch.tanh(obs @ st.w_hist)))
    (pa, ps1, _), (ma, _, mst) = outs
    assert torch.equal(pa, ps1)                    # relabelled parent: S1 on the stand rows
    assert torch.equal(ma, mst) and not torch.equal(ma, ps1)


# ===================================================================================== end to end (fake harness)
def test_e2e_stub_run_meta_contract(SNL, SB, SNC, tmp_path):
    SB.ENV_VARS.clear()
    SB.ENV_VARS.update(good_env(SNL))
    ck = stub_ckpts(SB, tmp_path)
    run_sn(SNL, tmp_path, ["--stand-ckpt", ck["st"], "--stand-mask", "1", "--checkpoint", ck["s1"],
                           "--out", str(tmp_path / "sn")])
    stamp = tmp_path / "sn" / "stamp"
    m = meta(stamp)
    blk = m["p2r_mode"]
    assert blk["variant"] == "stand_net" and blk["enter_rule"] == "latched_stand_slot" and blk["rng_restored"] is True
    assert blk["stand_mask"] == 1 and type(blk["stand_mask"]) is int                   # L4
    assert blk["specialist"] == {"path": ck["st"], "sha256": hashlib.sha256(Path(ck["st"]).read_bytes()).hexdigest()}
    s1_sha = hashlib.sha256(Path(ck["s1"]).read_bytes()).hexdigest()
    assert blk["s1_checkpoint_sha256"] == s1_sha == m["checkpoint_sha256"]
    assert blk["launcher_sha256"] == hashlib.sha256(P2L_PATH.read_bytes()).hexdigest()
    sn = blk["stand_net"]
    assert sn["wrapper_sha256"] == hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()               # L8
    assert sn["common_sha256"] == hashlib.sha256((SCRIPTS / "sn_common.py").read_bytes()).hexdigest()
    assert sn["restance_module_sha256"] == hashlib.sha256(SC.RESTANCE_PY.read_bytes()).hexdigest()
    assert sn["p2r_launcher_sha256"] == blk["launcher_sha256"] and sn["inner_mode"] == "st2"
    assert sn["estimator_equals_s1"] is True and sn["r_knob"] == D3 and sn["hist_encoding"] is True
    assert sn["mask_fn"] == "crab_hex_omni_restance.stand_mask_obs" and sn["mask_rows"] == "all"
    assert sn["mask_after_estimate"] is True and sn["stand_rows_alive"] > 0
    assert sn["trained_stand_mask"] == 1 and type(sn["trained_stand_mask"]) is int and sn["stand_ckpt_kind"] == "sn_dagger"
    assert blk["raw_series"] == ["p2r_stand_mode", "sn_stand_action", "sn_s1_action", "sn_r_target", "sn_r_branch"]
    for ep in episodes(stamp):
        T = ep["actions"].shape[0]
        assert ep["p2r_stand_mode"].shape == (T,) and ep["p2r_stand_mode"].dtype == bool
        for k in ("sn_stand_action", "sn_s1_action", "sn_r_target"):
            assert ep[k].shape == (T, NA) and ep[k].dtype == np.float32
        assert ep["sn_r_branch"].shape == (T, 6) and ep["sn_r_branch"].dtype == np.int8
    # the head-validity contract sn_calib applies to every stand_net stamp
    env = m["env_vars"]
    assert SNC.stand_net_reasons("SN_T", "walkstop__omni@1", env, blk, s1_sha) == []
    assert SNC.stand_net_arm(blk) == (blk["specialist"]["sha256"], 1)
    assert SNC.stand_net_reasons("SN_T", "k", env, dict(blk, stand_mask=True), s1_sha)          # a bool: malformed
    assert SNC.stand_net_reasons("SN_T", "k", env, dict(blk, variant="st2"), s1_sha)
    assert SNC.stand_net_reasons("SN_T", "k", dict(env, KRABBY_OMNI_STAND_OBS="clock"), blk, s1_sha)
    mixed = SNC.stand_net_consistency("SN_T", {"a": blk, "b": dict(blk, stand_mask=0)})
    assert all(mixed.values())
    assert (tmp_path / "sn" / "app_closed").exists()


@pytest.mark.parametrize("which, flag, needle", [("st_other", 0, "trained with stand_mask 1"),
                                                  ("st", 0, "trained with stand_mask 1"),
                                                  ("st_noinfo", 1, "is not S1 itself")])
def test_e2e_stub_wrong_arm_refused_no_stamp_app_closed(SNL, SB, tmp_path, which, flag, needle):
    """Review 2026-10-12 (blocking): through the whole launcher path (P2L's factory, the harness's load +
    get_inference_policy), a checkpoint trained with stand_mask 1 evaluated with --stand-mask 0 (and a non-trainer,
    non-S1 checkpoint) is refused before the first step: no stamp, the app still closes."""
    ck = stub_ckpts(SB, tmp_path, st_mask=1)
    if which == "st_other":
        ck = stub_ckpts(SB, tmp_path, st_mask=0)                 # st_other is then the mask-1 arm
    with pytest.raises(SystemExit, match=needle):
        run_sn(SNL, tmp_path, ["--stand-ckpt", ck[which], "--stand-mask", str(flag), "--checkpoint", ck["s1"],
                               "--out", str(tmp_path / "bad")])
    assert not (tmp_path / "bad" / "stamp").exists()
    assert (tmp_path / "bad" / "app_closed").exists()


def test_e2e_stub_estimator_mismatch_refused_no_stamp_app_closed(SNL, SB, tmp_path):
    ck = stub_ckpts(SB, tmp_path)
    with pytest.raises(SystemExit, match="estimator differs"):
        run_sn(SNL, tmp_path, ["--stand-ckpt", ck["badest"], "--stand-mask", "0", "--checkpoint", ck["s1"],
                               "--out", str(tmp_path / "bad")])
    assert not (tmp_path / "bad" / "stamp").exists()
    assert (tmp_path / "bad" / "app_closed").exists()


@pytest.mark.parametrize("suite", [False, True])
def test_e2e_real_identity_bridge_is_bitwise(SNL, SB, RD, real, tmp_path, suite):
    """S1 (real networks, CPU copy) as the stand checkpoint, mask 0: every raw key of the unpatched run, bitwise."""
    base = ["--checkpoint", real["s1"], "--fall", "9:1"] + (["--suite"] if suite else [])
    run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    run_sn(SNL, tmp_path, ["--stand-ckpt", real["s1"], "--stand-mask", "0"] + base + ["--out", str(tmp_path / "id")])
    ref, sn = episodes(tmp_path / "ref" / "stamp"), episodes(tmp_path / "id" / "stamp")
    assert len(ref) == len(sn) == (6 if suite else 3)
    n_stand = 0
    for a, b in zip(ref, sn):
        for k in a:
            assert np.array_equal(a[k], b[k]), k
        st = b["p2r_stand_mode"]
        n_stand += int(st.sum())
        assert np.array_equal(b["sn_stand_action"][st], b["sn_s1_action"][st])
        assert np.array_equal(b["actions"][st], b["sn_stand_action"][st])
    assert n_stand > 0
    blk = meta(tmp_path / "id" / "stamp")["p2r_mode"]
    assert blk["stand_net"]["stand_rows_action_differs_from_s1"] == 0 and blk["stand_mask"] == 0
    assert blk["stand_net"]["stand_ckpt_kind"] == "s1_identity" and blk["stand_net"]["trained_stand_mask"] is None
    res = RD.read_row(tmp_path / "ref" / "stamp", tmp_path / "id" / "stamp", identity=True)
    assert res["verdict"] == "PASS" and res["bridge"]["verdict"] == "BITWISE", res
    assert res["meta_fails"] == [] and res["swap_live"]["fails"] == []
    live = RD.read_row(tmp_path / "ref" / "stamp", tmp_path / "id" / "stamp", identity=False)
    assert live["verdict"] == "FAIL"                                # an identity stamp is not a live swap


@pytest.mark.parametrize("which, flag", [("s1", 1), ("head", 0), ("head", 1)])
def test_e2e_real_swap_is_live_on_stand_rows_only(SNL, SB, RD, real, tmp_path, which, flag):
    base = ["--checkpoint", real["s1"]]
    run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    ck = real["s1"] if which == "s1" else real[f"head{flag}"]          # a trainer checkpoint of THIS arm
    run_sn(SNL, tmp_path, ["--stand-ckpt", ck, "--stand-mask", str(flag)] + base + ["--out", str(tmp_path / "sn")])
    ref, sn = episodes(tmp_path / "ref" / "stamp"), episodes(tmp_path / "sn" / "stamp")
    differ = 0
    for a, b in zip(ref, sn):
        st = b["p2r_stand_mode"]
        first = int(np.argmax(st))
        assert st.any() and first > 0
        assert np.array_equal(a["actions"][:first], b["actions"][:first])          # walking rows before entry
        assert np.array_equal(b["actions"][st], b["sn_stand_action"][st])
        differ += int((b["sn_stand_action"][st] != b["sn_s1_action"][st]).any(axis=1).sum())
    assert differ > 0
    blk = meta(tmp_path / "sn" / "stamp")["p2r_mode"]
    assert blk["stand_net"]["stand_rows_action_differs_from_s1"] > 0 and blk["stand_mask"] == flag
    res = RD.read_row(tmp_path / "ref" / "stamp", tmp_path / "sn" / "stamp", identity=False)
    assert res["verdict"] == "PASS", res
    assert res["swap_live"]["pre_entry_episodes_checked"] == len(sn)
    assert RD.read_row(tmp_path / "ref" / "stamp", tmp_path / "sn" / "stamp", identity=True)["verdict"] == "FAIL"


def test_e2e_real_estimator_mismatch_refused(SNL, SB, real, tmp_path):
    with pytest.raises(SystemExit, match="estimator differs"):
        run_sn(SNL, tmp_path, ["--stand-ckpt", real["badest"], "--stand-mask", "0", "--checkpoint", real["s1"],
                               "--out", str(tmp_path / "bad")])
    assert not (tmp_path / "bad" / "stamp").exists() and (tmp_path / "bad" / "app_closed").exists()


@pytest.mark.parametrize("which, flag, needle", [("head1", 0, "trained with stand_mask 1"),
                                                  ("head0", 1, "trained with stand_mask 0"),
                                                  ("head_noinfo", 0, "is not S1 itself")])
def test_e2e_real_wrong_arm_refused(SNL, SB, real, tmp_path, which, flag, needle):
    """Review 2026-10-12 (blocking), real S1 networks: the A-mask net is never evaluated unmasked (nor A-full
    masked), a non-trainer head never runs under any mask; no stamp, the app still closes."""
    with pytest.raises(SystemExit, match=needle):
        run_sn(SNL, tmp_path, ["--stand-ckpt", real[which], "--stand-mask", str(flag), "--checkpoint", real["s1"],
                               "--out", str(tmp_path / "bad")])
    assert not (tmp_path / "bad" / "stamp").exists() and (tmp_path / "bad" / "app_closed").exists()


# ===================================================================================== the reader
def _doctor(stamp: Path, fn):
    """Rewrite every raw episode of a stamp through ``fn(dict) -> dict``."""
    for p in sorted((stamp / "raw").glob("episode_*.npz")):
        d = fn(dict(np.load(p)))
        np.savez_compressed(p, **d)


def test_reader_identity_miss_rule_and_cli(SNL, SB, RD, real, tmp_path):
    base = ["--checkpoint", real["s1"]]
    run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    run_sn(SNL, tmp_path, ["--stand-ckpt", real["s1"], "--stand-mask", "0"] + base + ["--out", str(tmp_path / "id")])
    ref, sn = tmp_path / "ref" / "stamp", tmp_path / "id" / "stamp"
    assert RD.main(["--refs-stamp", str(ref), "--sn-stamp", str(sn), "--identity",
                    "--json", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["stamp"]["bridge"]["verdict"] == "BITWISE"

    def nudge(change):
        def fn(d):
            st = d["p2r_stand_mode"]
            t = int(np.nonzero(~st & (np.arange(st.size) > int(np.argmax(st))))[0][0])   # a post-entry walking row
            d["actions"] = d["actions"].copy()
            d["actions"][t, 0] = change(d["actions"][t, 0])
            return d
        return fn

    small = tmp_path / "small"
    import shutil  # noqa: PLC0415
    shutil.copytree(sn, small)
    _doctor(small, nudge(lambda v: np.nextafter(v, np.float32(np.inf))))              # one float32 ulp
    r = RD.read_row(ref, small, identity=True)
    assert 0 < r["bridge"]["max_action_diff"] <= 1e-5
    assert r["bridge"]["verdict"] == "MISS_SMALL" and r["verdict"] == "MISS_SMALL" and r["swap_live"]["fails"] == []
    # review 2026-10-12: the G1-G5 comparison names its OWN outputs, never sn_calib's default calib.json / calib.md
    note = r["bridge"]["note"]
    assert "--head SN_ID_s18 --json " in note and "bridge_calib.json" in note and "bridge_calib.md" in note
    assert "/calib.json" not in note and "/calib.md" not in note
    assert "bridge_calib.json" in RD.__doc__ and "bridge_calib.md" in RD.__doc__
    assert RD.main(["--refs-stamp", str(ref), "--sn-stamp", str(small), "--identity"]) == 3
    big = tmp_path / "big"
    shutil.copytree(sn, big)
    _doctor(big, nudge(lambda v: v + np.float32(1e-3)))
    assert RD.read_row(ref, big, identity=True)["verdict"] == "FAIL"
    assert RD.main(["--refs-stamp", str(ref), "--sn-stamp", str(big), "--identity"]) == 4
    # a stand-row action that is not the stand net's: swap-live (a) fails
    leak = tmp_path / "leak"
    shutil.copytree(sn, leak)

    def stand_leak(d):
        st = d["p2r_stand_mode"]
        d["actions"] = d["actions"].copy()
        d["actions"][int(np.argmax(st)), 1] += np.float32(0.5)
        return d

    _doctor(leak, stand_leak)
    r = RD.read_row(ref, leak, identity=False)
    assert r["verdict"] == "FAIL" and not r["swap_live"]["stand_actions_equal_stand_net"]
    # meta: a bool mask / a wrong env / a missing block fail closed
    bad = tmp_path / "badmeta"
    shutil.copytree(sn, bad)
    m = meta(bad)
    m["p2r_mode"]["stand_mask"] = False
    m["env_vars"]["KRABBY_OMNI_STAND_OBS"] = "clock"
    del m["p2r_mode"]["stand_net"]
    (bad / "run_meta.json").write_text(json.dumps(m))
    fails = RD.meta_reasons(bad, ref, identity=True)
    assert any("stand_mask" in f for f in fails) and any("STAND_OBS" in f for f in fails)
    assert any("estimator_equals_s1" in f for f in fails) and any("r_knob" in f for f in fails)
    assert any("stand_ckpt_kind" in f for f in fails)
    # the trained-arm record (review 2026-10-12): a trainer checkpoint's mask must be the stamp's mask; the identity
    # kind must name S1 itself
    arm = tmp_path / "armmeta"
    shutil.copytree(sn, arm)
    m = meta(arm)
    m["p2r_mode"]["stand_net"].update(stand_ckpt_kind="sn_dagger", trained_stand_mask=1)
    (arm / "run_meta.json").write_text(json.dumps(m))
    assert any("trained_stand_mask 1 != stand_mask 0" in f for f in RD.meta_reasons(arm, ref, identity=False))
    m["p2r_mode"]["stand_net"].update(trained_stand_mask=0)
    (arm / "run_meta.json").write_text(json.dumps(m))
    assert RD.meta_reasons(arm, ref, identity=False) == []
    m["p2r_mode"]["stand_net"].update(stand_ckpt_kind="s1_identity", trained_stand_mask=None)
    m["p2r_mode"]["specialist"]["sha256"] = "c" * 64
    (arm / "run_meta.json").write_text(json.dumps(m))
    assert any("s1_identity but the stand checkpoint" in f for f in RD.meta_reasons(arm, ref, identity=False))
    assert RD.read_row(None, sn, identity=True)["verdict"] == "MISSING"
    assert RD.exit_code({"a": {"verdict": "PASS"}, "b": {"verdict": "MISSING"}}) == 2


# ===================================================================================== run_arm / rows script
SN_EVAL_STUB = textwrap.dedent("""\
    import json, os, sys, pathlib
    a = sys.argv
    sc = a[a.index('--scenario') + 1]
    out = pathlib.Path(a[a.index('--output-root') + 1]) / sc / ('seed%03d' % int(a[a.index('--env-seed') + 1])) / 'ts'
    out.mkdir(parents=True)
    (out / 'complete.json').write_text('{}')
    (out / 'argv.json').write_text(json.dumps({'argv': a, 'env': {k: v for k, v in os.environ.items()
                                                                if k.startswith(('KRABBY_', 'P2R_', 'SN_'))}}))
    """)


def test_run_arm_eval_extra_args_reach_the_launcher_and_parse(SNL, tmp_path):
    ev = tmp_path / "sn_stub_eval_crab_hex_gait.py"
    ev.write_text(SN_EVAL_STUB)
    ck = tmp_path / "st.pt"
    ck.write_bytes(b"x")
    env = P2T._env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1, P2_EVAL=ev,
                   EVAL_EXTRA_ARGS=f"--stand-ckpt {ck} --stand-mask 1")
    env = {k: v for k, v in env.items() if not k.startswith("SN_")}
    st = tmp_path / "stamp"
    st.mkdir()
    (st / "run_meta.json").write_text(json.dumps({"env_vars": good_env(SNL)}))
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
    assert got["argv"][-4:] == ["--stand-ckpt", str(ck), "--stand-mask", "1"]
    assert got["env"] == good_env(SNL)                       # the stamp's env reaches the eval, nothing else
    o, rest = SNL.parse_sn_args(got["argv"][1:], got["env"])
    assert (o.stand_ckpt, o.stand_mask) == (str(ck), 1)
    assert rest[:2] == ["--headless", "--manifest"] and "--checkpoint" in rest and "--save-raw" in rest


@pytest.mark.skipif(not (SC.S1_CKPT.is_file() and (SC.REFS_SUITE1 / "run_meta.json").is_file()),
                    reason="S1 checkpoint / REFS env stamp not on this machine")
def test_rows_script_dry_run_composes_the_identity_bridge(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("P2R_", "P2_"))}
    log = REPO / "parkour/outputs/omni_plan/stand_network/sn_rows.log"
    log_before = log.stat().st_mtime_ns if log.exists() else None
    r = subprocess.run(["bash", str(ROWS_SH), "--dry-run"], capture_output=True, text=True, timeout=60, env=env)
    assert (log.stat().st_mtime_ns if log.exists() else None) == log_before          # a dry run writes no log
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 3
    for line, (row, man) in zip(lines, (("suite__omni", "experiments/eval/scenarios_omni.yaml"),
                                        ("walkstop__omni", "2026-10-07_omni_phase2/scenarios_p2s.yaml"),
                                        ("field_hard__omni", "2026-10-07_omni_phase2/scenarios_p2.yaml"))):
        assert f"P2_EVAL={LAUNCHER}" in line and "P2_GUARD_REF=REFH_S1_s18" in line and "P2_ENV_EXTRA= " in line
        assert f"--stand-ckpt\\ {SC.S1_CKPT}\\ --stand-mask\\ 0" in line
        assert f"--row-eval SN_ID_s18 {SC.S1_CKPT} {SC.REFS_SUITE1} " in line and f"{man} {row} 1" in line
    trained = tmp_path / "model_100.pt"
    trained.write_bytes(b"x")
    s1_copy = tmp_path / "s1_copy.pt"
    s1_copy.write_bytes(SC.S1_CKPT.read_bytes())
    for args, extra, needle in (
            (["X", str(tmp_path / "nope.pt"), "0"], {}, "no stand checkpoint"),
            (["X", "rel.pt", "0"], {}, "absolute"),
            (["X", str(trained), "2"], {}, "MASK"), (["X", str(trained), "0", "ring__omni@1"], {}, "manifest"),
            (["bad-name", str(trained), "0"], {}, "NAME"),
            ([], {"P2_ENV_EXTRA": "KRABBY_OMNI_STAND_OBS=clock"}, "P2_ENV_EXTRA"), ([], {"P2R_MODE": "st2"}, "P2R_"),
            # review 2026-10-12: no S1 / mask-0 default under an explicit NAME
            (["SN_Amask_s18"], {}, "needs STAND_CKPT and MASK"),
            (["SN_Amask_s18", str(trained)], {}, "needs STAND_CKPT and MASK"),
            (["SN_ID_s18"], {}, "needs STAND_CKPT and MASK"),
            # the identity head is S1 with mask 0 only; S1 runs only under an SN_ID* name (any path to the same bytes)
            (["SN_ID_s18", str(trained), "0"], {}, "identity bridge"),
            (["SN_ID_s18", str(SC.S1_CKPT), "1"], {}, "identity bridge"),
            (["SN_Afull_s18", str(SC.S1_CKPT), "0"], {}, "only under an SN_ID"),
            (["SN_Afull_s18", str(s1_copy), "0"], {}, "only under an SN_ID")):
        r = subprocess.run(["bash", str(ROWS_SH), "--dry-run"] + args, capture_output=True, text=True, timeout=60,
                           env={**env, **extra})
        assert r.returncode == 2 and not r.stdout and needle in r.stderr, (args, extra, r.stdout, r.stderr)
    for args, want in ((["SN_ID_s18", str(SC.S1_CKPT), "0", "walkstop__omni@1"], f"{SC.S1_CKPT}\\ --stand-mask\\ 0"),
                       (["SN_IDM_s18", str(SC.S1_CKPT), "1", "walkstop__omni@1"], f"{SC.S1_CKPT}\\ --stand-mask\\ 1"),
                       (["SN_Amask_s18", str(trained), "1", "walkstop__omni@1"], f"{trained}\\ --stand-mask\\ 1")):
        r = subprocess.run(["bash", str(ROWS_SH), "--dry-run"] + args, capture_output=True, text=True, timeout=60,
                           env=env)
        assert r.returncode == 0 and len(r.stdout.strip().splitlines()) == 1 and want in r.stdout, (args, r.stderr)
        assert f"--row-eval {args[0]} " in r.stdout
