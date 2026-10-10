"""Stage p2r Z2 (P2R_Z2_PLAN.md step 1 "Probe" / step 4 "Dose" / "Early checks"; CHARTER 12): the PROBE + CPU-tool
files. CPU only: nothing here starts Isaac/Kit or a GPU.

* ``scripts/p2r_mode_eval_crab_hex_gait.py --p2r-mode restance``: the refusals (env knobs -- since the 2026-10-10
  cams-only amendment only ``STAND_OBS=clock``, no held proxy --, R's constants, the residual JSON, restance-only
  flags); on the stub env, stand rows get ``R(obs)`` (+ the residual bias) and walking
  rows S1's action untouched; the three raw series; the stand bank; end to end through the fake harness of
  ``test_omni_p2r_launcher.py`` (widened to the real obs layout's 89 proprio columns): pre-entry actions bitwise
  equal to the unpatched run, stand actions == ``p2r_target`` + bias, the record block, the bank file, and
  :func:`restance_swap_live` passing (and failing on a tampered stamp);
* ``scripts/p2r_z2_dose.py`` (NOTE(p2r Z2 cams-only, 2026-10-10): R reads cams + hip-pitch positions, no proxy):
  the joint layout vs R's obs columns; isaaclab's ``wrap_to_pi``; the hip default (run_meta, else the linkage); the
  obs rebuild; the dose end to end on a synthetic stamp (mean and sampled actions, the debias constant, the
  'home'-absent STOP, a recorded default); the band verdicts; the recorded-obs cross-check (cams + hips);
* ``scripts/p2r_z2_fidelity.py``: the eval-path replica against the REAL ``Actor`` / ``StateHistoryEncoder`` /
  ``DefaultEstimator`` classes (loaded by path, no Isaac); the score and its line on a tiny fake actor; the CLI's
  rc / ``--dry-run``;
* ``scripts/p2r_early.py --mode z2``: the formation line and the fidelity judge (the module's own ``--self-test``
  carries the bulk).
"""
from __future__ import annotations

import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
CAMP = TASKS / "crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
LAUNCHER = SCRIPTS / "p2r_mode_eval_crab_hex_gait.py"
DOSE = SCRIPTS / "p2r_z2_dose.py"
FIDELITY = SCRIPTS / "p2r_z2_fidelity.py"
EARLY = SCRIPTS / "p2r_early.py"
RESTANCE = TASKS / "crab_hex_omni_task/mdp/crab_hex_omni_restance.py"
FWD_OBS = TASKS / "crab_hex_forward_task/mdp/observations.py"
MODULES = REPO / "parkour/scripts/rsl_rl/modules"
STEP1_TESTS = REPO / "tests/unit/test_omni_p2r_launcher.py"
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
#: NOTE(p2r Z2 cams-only, 2026-10-10): the restance probe needs the stand bit + R's constants only (no held proxy)
ENV_OK = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_RESTANCE": "default"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def L():
    return _load(LAUNCHER, "_t_z2_launcher")


@pytest.fixture(scope="module")
def RST():
    return _load(RESTANCE, "_t_z2_restance")


@pytest.fixture(scope="module")
def D():
    return _load(DOSE, "_t_z2_dose")


@pytest.fixture(scope="module")
def FID():
    return _load(FIDELITY, "_t_z2_fidelity")


@pytest.fixture(scope="module")
def EA():
    return _load(EARLY, "_t_z2_early")


# ===================================================================================== stubs (step-1's, widened)
@pytest.fixture(scope="module")
def STEP1():
    """The step-1 launcher tests' STUBS / FAKE_HARNESS sources (loaded by path; none of its tests is collected
    here)."""
    return _load(STEP1_TESTS, "_t_z2_step1_tests")


def _wide(src: str) -> str:
    """The stub layout widened to the real proprio head: priv_explicit AFTER column 89 (as the real 207), 18
    actions, so R's cam (18..28) and proxy (83..88) columns exist and the harness's priv overwrite never touches
    them."""
    a = src.replace("NUM_PROP, NUM_SCAN, NUM_PRIV = 6, 4, 3", "NUM_PROP, NUM_SCAN, NUM_PRIV = 90, 4, 3")
    a = a.replace("OBS_W, N_ACT = 20, 4", "OBS_W, N_ACT = 100, 18")
    assert a != src
    return a


@pytest.fixture()
def SB(tmp_path, monkeypatch, STEP1):
    (tmp_path / "p2r_stubs.py").write_text(_wide(STEP1.STUBS))
    (tmp_path / "fake_harness.py").write_text(STEP1.FAKE_HARNESS)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("p2r_stubs", None)
    import p2r_stubs  # noqa: PLC0415

    p2r_stubs.StubRunner.builds.clear()
    yield p2r_stubs
    sys.modules.pop("p2r_stubs", None)


def _residual(tmp_path, bias=None, **extra) -> Path:
    p = tmp_path / "residual.json"
    p.write_text(json.dumps(dict({"mu_bias": bias if bias is not None else [0.01 * (i - 9) for i in range(18)]},
                                 **extra)))
    return p


def _opts(L, argv, environ=None, check_files=False):
    o, rest = L.strip_p2r_args(argv, environ or {})
    return L.validate_opts(o, check_files=check_files, environ=environ or {}), rest


# ===================================================================================== launcher: argv / refusals
def test_restance_opts_parse_the_knob_the_bias_and_the_bank(L, RST, tmp_path):
    env = dict(ENV_OK, KRABBY_RESTANCE="lift=-0.6,up_frac=0.7")
    o, rest = _opts(L, ["--p2r-mode", "restance", "--p2r-residual", str(_residual(tmp_path)), "--p2r-stand-bank",
                        "--scenario", "x"], env)
    assert rest == ["--scenario", "x"]
    assert o.restance.lift == -0.6 and o.restance.up_frac == 0.7 and o.stand_bank is True
    assert o.residual["mu_bias"][0] == pytest.approx(-0.09) and len(o.residual["mu_bias"]) == 18
    assert o.residual["sha256"] == L.sha256_file(tmp_path / "residual.json")
    # env fallbacks (run_arm.sh lr_eval passes no extra CLI args)
    o, _ = _opts(L, [], dict(env, P2R_MODE="restance", P2R_RESIDUAL=str(tmp_path / "residual.json"),
                             P2R_STAND_BANK="1"))
    assert o.mode == "restance" and o.stand_bank is True and o.residual["path"] == str(tmp_path / "residual.json")
    o, _ = _opts(L, ["--p2r-mode", "restance"], env)
    assert o.residual is None and o.stand_bank is False


@pytest.mark.parametrize("env", [
    {},                                                                    # nothing
    {"KRABBY_OMNI_STAND_PROXY": "hold", "KRABBY_RESTANCE": "default"},     # no stand bit
    {"KRABBY_RESTANCE": "default"},                                        # no stand bit
    {"KRABBY_OMNI_STAND_OBS": "1", "KRABBY_RESTANCE": "default"},          # the appended head bit, not the clock bit
    {"KRABBY_OMNI_STAND_OBS": "clock"},                                    # no R constants
    {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_RESTANCE": "0"},
    {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_RESTANCE": "lift=0.5"},
    {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_RESTANCE": "nope=1"},
    {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_RESTANCE": "theta_L=0.6"},  # a removed load-sensing constant
])
def test_restance_refused_without_its_environment(L, env):
    with pytest.raises(SystemExit):
        _opts(L, ["--p2r-mode", "restance"], env)


@pytest.mark.parametrize("proxy", [None, "0"])
def test_restance_no_longer_needs_the_held_proxy(L, proxy):
    """NOTE(p2r Z2 cams-only, 2026-10-10): the kinematic R reads no proxy -- the variant validates with the held proxy
    absent or off."""
    env = dict(ENV_OK) if proxy is None else dict(ENV_OK, KRABBY_OMNI_STAND_PROXY=proxy)
    o, _ = _opts(L, ["--p2r-mode", "restance"], env)
    assert o.mode == "restance" and o.restance is not None


def test_restance_refuses_the_held_proxy_on(L):
    """... and refuses it ON: the Z2 reads mark such a stamp INVALID, so the GPU row would be wasted."""
    with pytest.raises(SystemExit, match="KRABBY_OMNI_STAND_PROXY off"):
        _opts(L, ["--p2r-mode", "restance"], dict(ENV_OK, KRABBY_OMNI_STAND_PROXY="hold"))


@pytest.mark.parametrize("payload", [
    {"mu_bias": [0.0] * 17},                                   # wrong length
    {"mu_bias": [0.0] * 17 + [float("nan")]},                  # non-finite
    {"mu_bias": [0.0] * 17 + [True]},                          # a bool is not a bias
    {"mu_bias": [0.0] * 18, "mu_bias_rod": [0.0] * 6},         # a typo'd key must never run as a zero bias
    {"bias": [0.0] * 18},                                      # no mu_bias
    [0.0] * 18,                                                # not an object
])
def test_residual_json_refusals(L, tmp_path, payload):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        _opts(L, ["--p2r-mode", "restance", "--p2r-residual", str(p)], ENV_OK)


def test_residual_missing_file_and_restance_only_flags_refused(L, tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        _opts(L, ["--p2r-mode", "restance", "--p2r-residual", str(tmp_path / "none.json")], ENV_OK)
    with pytest.raises(SystemExit, match="restance"):
        _opts(L, ["--p2r-mode", "hold", "--p2r-residual", str(_residual(tmp_path))], ENV_OK)
    with pytest.raises(SystemExit, match="restance"):
        _opts(L, ["--p2r-mode", "probe", "--p2r-probe", "A=/a.pt", "--p2r-stand-bank"], ENV_OK)
    with pytest.raises(SystemExit):
        _opts(L, ["--p2r-mode", "restance"], dict(ENV_OK, P2R_STAND_BANK="yes"))
    # the step-1 variants never read the KRABBY knobs: hold with an empty env still validates
    o, _ = _opts(L, ["--p2r-mode", "hold"], {})
    assert o.restance is None and o.residual is None and o.stand_bank is False


def test_launcher_loads_the_one_restance_module(L, RST):
    assert L.RESTANCE_MODULE == RESTANCE and RESTANCE.is_file()
    mod = L.load_restance()
    assert mod.BRANCH_NAMES == RST.BRANCH_NAMES and mod.ACTION_DIM == 18 and mod.MIN_OBS_WIDTH == 89


# ===================================================================================== launcher: the mode wrapper
def _mode(L, SB, n=4, params=None, mu_bias=None, stand_bank=False, bank_max=50):
    env = SB.StubEnv(n)
    ct = env.cmd
    L.arm_slot_latch(ct, n, "cpu")
    s1 = SB.StubActor()
    rst = L.load_restance()
    m = L.P2RMode("restance", ct, env.unwrapped, priv=(SB.P0, SB.P1), num_prop=SB.NUM_PROP, bank_max=bank_max,
                  restance=(rst, params or rst.RestanceParams()), mu_bias=mu_bias, stand_bank=stand_bank)
    m.bind_s1(s1.act_inference, s1)
    return env, ct, s1, m, rst


def _begin(m, n, labels=("stand", "fwd"), seg=None, T=5):
    seg = np.zeros((T, n), np.int16) if seg is None else seg
    m.begin_pass(types.SimpleNamespace(segment_id=seg, hold_labels=list(labels), n_steps=T))


def test_restance_variant_constructor_refusals(L, SB):
    env = SB.StubEnv(2)
    rst = L.load_restance()
    kw = dict(priv=(SB.P0, SB.P1), num_prop=SB.NUM_PROP)
    with pytest.raises(ValueError):
        L.P2RMode("restance", env.cmd, env.unwrapped, **kw)                       # no controller
    with pytest.raises(ValueError):
        L.P2RMode("hold", env.cmd, env.unwrapped, restance=(rst, rst.RestanceParams()), **kw)
    with pytest.raises(ValueError):
        L.P2RMode("hold", env.cmd, env.unwrapped, mu_bias=[0.0] * 18, **kw)
    with pytest.raises(ValueError):
        L.P2RMode("hold", env.cmd, env.unwrapped, stand_bank=True, **kw)


def test_restance_stand_rows_get_R_walking_rows_get_s1_untouched(L, SB):
    bias = [0.001 * (i + 1) for i in range(18)]
    env, ct, s1, m, rst = _mode(L, SB, n=4, mu_bias=bias)
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 1, 0, 3])                      # stand, walk, stand, move_turn
    _begin(m, 4)
    obs, _ = env.get_observations()
    m.observe(obs)
    obs[:, SB.P0:SB.P1] = SB.StubEst().inference(obs[:, :SB.NUM_PROP])   # the harness's in-place estimate
    before = obs.clone()
    with torch.inference_mode():
        a = m(obs, hist_encoding=True)
    assert torch.equal(obs, before)                              # the harness's obs is never written
    tgt, br = rst.restance_target(before, rst.RestanceParams())
    want_stand = tgt + torch.tensor(bias).reshape(1, -1)
    ref = s1.act_inference(before, hist_encoding=True)
    assert torch.equal(a[0], want_stand[0]) and torch.equal(a[2], want_stand[2])
    assert torch.equal(a[1], ref[1]) and torch.equal(a[3], ref[3])
    rec = m._cur["rst"]
    assert rec[L.RAW_BRANCH].dtype == torch.int8 and rec[L.RAW_BRANCH].shape == (5, 4, 6)
    assert torch.equal(rec[L.RAW_BRANCH][0, 0], br[0].to(torch.int8)) and (rec[L.RAW_BRANCH][0, 1] == -1).all()
    assert torch.equal(rec[L.RAW_TARGET][0, 2], tgt[2]) and not rec[L.RAW_TARGET][0, 1].any()   # no bias, 0 off
    assert torch.equal(rec[L.RAW_OBS_PROXY][0], before[:, 83:89])                             # every row
    assert m._cur["stand"][0].tolist() == [True, False, True, False]


def test_restance_without_bias_applies_exactly_R(L, SB):
    env, ct, s1, m, rst = _mode(L, SB, n=2)
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 0])
    _begin(m, 2)
    obs, _ = env.get_observations()
    m.observe(obs)
    with torch.inference_mode():
        a = m(obs, hist_encoding=True)
    assert torch.equal(a, rst.restance_target(obs, rst.RestanceParams())[0])


def test_stand_bank_reservoir_alive_stand_rows_only(L, SB):
    n, T, K = 4, 30, 25
    env, ct, s1, m, rst = _mode(L, SB, n=n, stand_bank=True, bank_max=K)
    seg = np.tile(np.array([[0, 0, 1, 1]], np.int16), (T, 1))
    _begin(m, n, labels=("stand", "fwd"), seg=seg, T=T)
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 1, 1, 1])                      # only env 0 stands
    obs, _ = env.get_observations()
    for _t in range(T):
        m.observe(obs)
        with torch.inference_mode():
            m(obs, hist_encoding=True)
        obs = obs + 0.01
    assert m.sbank_seen == T and m.bank_seen == 0                # the probe bank is never touched
    b = m.stand_bank_arrays()
    assert b["obs"].shape == (K, SB.OBS_W) and b["obs"].dtype == np.float32
    assert b["target"].shape == (K, 18) and b["target"].dtype == np.float32
    assert b["branch"].shape == (K, 6) and b["branch"].dtype == np.int8
    assert b["proxy"].shape == (K, 6) and np.array_equal(b["proxy"], b["obs"][:, 83:89])
    assert set(b["hold_label"]) == {"stand"} and set(b["category"]) == {"stand"}
    assert set(b["episode"].tolist()) == {0} and len(set(b["t"].tolist())) == K
    # R recomputed on the banked (true) obs reproduces the banked target / branch
    t2, br2 = rst.restance_target(torch.as_tensor(b["obs"]), rst.RestanceParams())
    assert np.array_equal(t2.numpy(), b["target"]) and np.array_equal(br2.numpy().astype(np.int8), b["branch"])
    assert str(b["restance"]) == rst.format_restance(rst.RestanceParams()) and not bool(b["has_residual"])
    assert json.loads(str(b["restance_json"])) == rst.RestanceParams().to_dict()
    # a dead env is never banked
    m.alive = torch.tensor([False, True, True, True])
    seen = m.sbank_seen
    tgt, br = rst.restance_target(obs, rst.RestanceParams())
    m._stand_bank_add(0, obs, torch.tensor([True, False, False, False]), tgt, br, obs[:, 83:89])
    assert m.sbank_seen == seen


def test_reservoir_pick_matches_algorithm_r(L):
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    pos, slots = L.P2RMode._reservoir_pick(0, 3, 5, rng1)
    assert pos.tolist() == [0, 1, 2] and slots.tolist() == [0, 1, 2]
    assert L.P2RMode._reservoir_pick(10, 0, 5, rng1) == (None, None)
    # late items: the same draws as the inline loop of the step-1 bank
    idx = 7 + np.arange(4)
    j = rng2.integers(0, idx + 1)
    pos, slots = L.P2RMode._reservoir_pick(7, 4, 5, np.random.default_rng(0))
    keep = j < 5
    assert set(slots.tolist()) == set(j[keep].tolist())


def test_restance_stats_counts_recorded_stand_leg_rows(L):
    br = np.full((4, 2, 6), -1, np.int8)
    br[0, 0] = [0, 0, 1, 2, 3, 4]
    br[1, 0] = [3] * 6
    br[3, 1] = [0] * 6                                            # past episode 1's recorded length: ignored
    tgt = np.zeros((4, 2, 18), np.float32)
    tgt[1, 0, 3] = 0.5
    st = br[..., 0] >= 0
    out = L.restance_stats({L.RAW_BRANCH: br, L.RAW_TARGET: tgt, L.RAW_STAND: st}, [4, 3],
                           ("hold", "wait", "lift", "home", "seat"))
    assert out["stand_rows"] == 2
    assert out["branch_shares_on_stand"] == pytest.approx({"hold": 2 / 12, "wait": 1 / 12, "lift": 1 / 12,
                                                           "home": 7 / 12, "seat": 1 / 12})
    assert out["target_nonzero_share"] == 0.5


# ===================================================================================== launcher: end to end
def _run_ref(tmp_path, args, seed=0):
    torch.manual_seed(seed)
    old = sys.argv
    sys.argv = [str(tmp_path / "fake_harness.py")] + args
    try:
        g = runpy.run_path(str(tmp_path / "fake_harness.py"), run_name="p2r_harness")
        g["main"]()
    finally:
        sys.argv = old


def _run_launcher(L, tmp_path, args, environ, seed=0):
    torch.manual_seed(seed)
    old = sys.argv
    try:
        L.main([str(LAUNCHER)] + args, harness=tmp_path / "fake_harness.py", environ=environ)
    finally:
        sys.argv = old


def _episodes(stamp: Path):
    return [dict(np.load(p)) for p in sorted((stamp / "raw").glob("episode_*.npz"))]


@pytest.mark.parametrize("suite", [False, True])
def test_e2e_restance_swap_live_bank_and_record(L, SB, RST, tmp_path, suite):
    SB.save_ckpt(tmp_path / "s1.pt", 1)
    base = ["--checkpoint", str(tmp_path / "s1.pt")] + (["--suite"] if suite else [])
    _run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    bias = [0.002 * (i - 9) for i in range(18)]
    res = _residual(tmp_path, bias, source="Z1 walkstop stand residual (test)")
    _run_launcher(L, tmp_path, ["--p2r-mode", "restance", "--p2r-residual", str(res), "--p2r-stand-bank",
                                "--p2r-bank-max", "7"] + base + ["--out", str(tmp_path / "rst")], dict(ENV_OK))
    ref, rs = _episodes(tmp_path / "ref" / "stamp"), _episodes(tmp_path / "rst" / "stamp")
    assert len(ref) == len(rs) == (6 if suite else 3)
    for a, b in zip(ref, rs):
        st = b["p2r_stand_mode"]
        T = b["actions"].shape[0]
        assert b["p2r_branch"].shape == (T, 6) and b["p2r_branch"].dtype == np.int8
        assert b["p2r_target"].shape == (T, 18) and b["p2r_target"].dtype == np.float32
        assert b["p2r_obs_proxy"].shape == (T, 6) and b["p2r_obs_proxy"].dtype == np.float32
        assert st.any() and (b["p2r_branch"][st] >= 0).all() and (b["p2r_branch"][~st] == -1).all()
        assert not b["p2r_target"][~st].any()
        # applied == R's target + the bias, float32 bitwise (the bias is added in float32 on the wrapper's side)
        want = (b["p2r_target"][st] + np.asarray(bias, np.float32)[None, :]).astype(np.float32)
        assert np.array_equal(b["actions"][st], want)
        first = int(np.argmax(st))
        assert np.array_equal(a["actions"][:first], b["actions"][:first])     # walking before the first stand
    swap = L.restance_swap_live(tmp_path / "ref" / "stamp", tmp_path / "rst" / "stamp")
    assert swap["fails"] == [] and swap["stand_actions_equal_target"] and swap["off_stand_series_clean"]
    assert swap["pre_entry_matches_refs"] and swap["stand_mode_share"] > 0
    assert swap["pre_entry_episodes_checked"] == (0 if suite else 3)          # the suite stands from step 0
    blk = json.loads((tmp_path / "rst" / "stamp" / "run_meta.json").read_text())["p2r_mode"]
    assert blk["variant"] == "restance" and blk["bank"] is None and blk["specialist"] is None
    assert blk["rng_restored"] is True and "probes" not in blk
    assert blk["raw_series"] == ["p2r_stand_mode", "p2r_branch", "p2r_target", "p2r_obs_proxy"]
    r = blk["restance"]
    assert r["params"] == RST.RestanceParams().to_dict() and r["knob"] == RST.format_restance(RST.RestanceParams())
    assert r["module_sha256"] == L.sha256_file(RESTANCE)
    assert r["residual"]["mu_bias"] == pytest.approx(bias) and r["residual"]["source"].startswith("Z1")
    assert sum(r["branch_shares_on_stand"].values()) == pytest.approx(1.0) and r["stand_rows"] > 0
    assert blk["stand_bank"] == "p2r_stand_bank.npz" and 0 < blk["stand_bank_rows"] <= 7
    bank = np.load(tmp_path / "rst" / "stamp" / "p2r_stand_bank.npz")
    assert {"obs", "target", "branch", "proxy", "hold_label", "category", "t", "episode", "s1_sigma", "restance",
            "restance_json", "mu_bias", "has_residual"} <= set(bank.files)
    assert bank["obs"].shape[1] == SB.OBS_W and bool(bank["has_residual"])
    assert np.array_equal(bank["mu_bias"], np.asarray(bias, np.float32))
    assert set(bank["hold_label"]) <= {"stand", "settle", "prologue"}      # stand-mode rows, by the latched slot
    # every banked row IS the raw step it names (target and policy-visible proxy, bitwise)
    for i in range(bank["obs"].shape[0]):
        ep = rs[int(bank["episode"][i])]
        t = int(bank["t"][i])
        assert ep["p2r_stand_mode"][t]
        assert np.array_equal(ep["p2r_target"][t], bank["target"][i])
        assert np.array_equal(ep["p2r_obs_proxy"][t], bank["proxy"][i])
    assert (tmp_path / "rst" / "stamp" / "p2r_stand_bank.npz").stat().st_mtime_ns <= \
        (tmp_path / "rst" / "stamp" / "complete.json").stat().st_mtime_ns
    assert (tmp_path / "rst" / "app_closed").exists()


def test_swap_live_catches_a_tampered_stamp(L, SB, tmp_path):
    SB.save_ckpt(tmp_path / "s1.pt", 1)
    base = ["--checkpoint", str(tmp_path / "s1.pt")]
    _run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    _run_launcher(L, tmp_path, ["--p2r-mode", "restance"] + base + ["--out", str(tmp_path / "rst")], dict(ENV_OK))
    stamp = tmp_path / "rst" / "stamp"
    assert L.restance_swap_live(tmp_path / "ref" / "stamp", stamp)["fails"] == []
    # NOTE(p2r Z2 cams-only, 2026-10-10): the proxy series is recorded but no longer required (R reads no proxy)
    for g in sorted((stamp / "raw").glob("episode_*.npz")):
        d = {k: v for k, v in dict(np.load(g)).items() if k != "p2r_obs_proxy"}
        np.savez(g, **d)
    assert L.restance_swap_live(tmp_path / "ref" / "stamp", stamp)["fails"] == []
    f = sorted((stamp / "raw").glob("episode_*.npz"))[0]
    d = dict(np.load(f))
    st = d["p2r_stand_mode"]
    first = int(np.argmax(st))
    d["actions"] = d["actions"].copy()
    d["actions"][np.nonzero(st)[0][1], 0] += 1e-3                 # a stand action off R
    d["actions"][first - 1, 0] += 1e-3                             # a walking action before the first entry
    np.savez(f, **d)
    swap = L.restance_swap_live(tmp_path / "ref" / "stamp", stamp)
    assert swap["stand_actions_equal_target"] is False and swap["pre_entry_matches_refs"] is False
    assert L.restance_swap_live(None, None)["fails"] == ["no restance stamp"]


def test_e2e_hold_record_unchanged_by_the_restance_additions(L, SB, tmp_path):
    """The hold variant's record carries no restance key and the same raw series as step 1."""
    SB.save_ckpt(tmp_path / "s1.pt", 1)
    _run_launcher(L, tmp_path, ["--p2r-mode", "hold", "--checkpoint", str(tmp_path / "s1.pt"),
                                "--out", str(tmp_path / "hold")], {})
    blk = json.loads((tmp_path / "hold" / "stamp" / "run_meta.json").read_text())["p2r_mode"]
    assert not {"restance", "stand_bank", "stand_bank_rows", "stand_bank_seen"} & set(blk)
    assert blk["raw_series"] == ["p2r_stand_mode"]
    ep = _episodes(tmp_path / "hold" / "stamp")[0]
    assert not {"p2r_branch", "p2r_target", "p2r_obs_proxy"} & set(ep)


# ===================================================================================== dose: the obs rebuild
# NOTE(p2r Z2 cams-only, 2026-10-10): R reads the cams and the hip-pitch POSITIONS (no proxy), so the dose rebuilds
# those two inputs from the raws' joint_pos; the proxy rebuild / sample-and-hold are gone with the load sensing.
def _joint_names():
    names = []
    for leg in LEGS:
        names += [f"{leg}_Body_Hip_RevoluteJoint", f"{leg}_Body_CamShaft_RevoluteJoint"]
    names += [f"{leg}_Hip_Femur_RevoluteJoint" for leg in LEGS]
    names += [f"{leg}_Femur_Tibia_RevoluteJoint" for leg in LEGS]
    return names


def test_joint_layout_matches_the_real_articulation_order(D, RST):
    lay = D.JointLayout(_joint_names(), RST.LEGS)
    assert lay.cam == [1, 3, 5, 7, 9, 11] and lay.pitch == list(range(12, 18)) and lay.knee == list(range(18, 24))
    # the obs columns R reads are those joints' joint_pos_delta columns (the head opens at 17)
    assert [17 + i for i in lay.cam] == list(RST.CAM_OBS_COLS)
    assert [17 + i for i in lay.pitch] == list(RST.HIP_POS_OBS_COLS)
    with pytest.raises(ValueError):
        D.JointLayout(_joint_names()[:-1], RST.LEGS)


def test_the_proxy_rebuild_is_gone(D):
    for name in ("contact_fill", "episode_proxy", "sample_and_hold", "sense_constants", "PROXY_LIVE_MARGIN"):
        assert not hasattr(D, name), name


def test_wrap_to_pi_is_isaaclabs(D):
    x = torch.tensor([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi, 7.0, -7.0], dtype=torch.float32)
    got = D.wrap_to_pi(x)
    w = (x + torch.pi) % (2 * torch.pi)
    assert torch.equal(got, torch.where((w == 0) & (x > 0), torch.pi, w - torch.pi))
    assert (got.abs() <= np.pi + 1e-6).all() and got[1] == pytest.approx(np.pi, abs=1e-6)


def test_hip_default_from_run_meta_else_the_linkage(D, RST):
    lk = D.load_linkage()
    lay = D.JointLayout(_joint_names(), RST.LEGS)
    hd, src = D.hip_default({}, lay, lk)
    assert src == "linkage.hip_default_rad" and hd.dtype == np.float32
    assert np.array_equal(hd, np.full(6, np.float32(lk.hip_default_rad())))
    assert float(hd[0]) == pytest.approx(0.02717, abs=1e-5)       # the scene cfg's init_state, as the obs subtracts
    rec = np.arange(24, dtype=np.float32) * 0.01
    hd, src = D.hip_default({"default_joint_pos": rec.tolist()}, lay, lk)
    assert src == "run_meta.default_joint_pos" and np.array_equal(hd, rec[12:18])
    with pytest.raises(ValueError, match="default_joint_pos"):
        D.hip_default({"default_joint_pos": [0.0] * 23}, lay, lk)


def test_obs_inputs_and_policy_obs(D, RST):
    lay = D.JointLayout(_joint_names(), RST.LEGS)
    g = np.random.default_rng(0)
    jp = g.normal(0, 3, (5, 24)).astype(np.float32)
    hd = np.full(6, np.float32(0.02717))
    cams, hip = D.obs_inputs(jp, lay, hd)
    assert torch.equal(cams, D.wrap_to_pi(torch.as_tensor(jp[:, 1:12:2])))
    assert torch.equal(hip, torch.as_tensor(jp[:, 12:18] - hd[None, :]))       # float32, the obs term's own op
    obs = D.policy_obs(cams, hip, RST)
    assert obs.shape == (5, RST.MIN_OBS_WIDTH)
    assert torch.equal(obs[:, list(RST.CAM_OBS_COLS)], cams) and torch.equal(obs[:, list(RST.HIP_POS_OBS_COLS)], hip)
    others = torch.ones(RST.MIN_OBS_WIDTH, dtype=torch.bool)
    others[list(RST.CAM_OBS_COLS) + list(RST.HIP_POS_OBS_COLS)] = False
    assert not obs[:, others].any()


# ===================================================================================== dose: end to end
def _synthetic_joints(T=14, seed=0):
    g = np.random.default_rng(seed)
    jp = np.zeros((T, 24), np.float32)
    # hip pitch: default + an offset from a little past default down to past the seed lift (-0.125 rad)
    jp[:, 12:18] = (np.float32(0.02717) + g.uniform(-0.15, 0.02, (T, 6))).astype(np.float32)
    jp[:, 18:24] = (np.array([0.31, -0.31] * 3) + g.normal(0, 0.2, (T, 6))).astype(np.float32)
    jp[:, 1:12:2] = g.uniform(-9, 9, (T, 6)).astype(np.float32)      # cams wind past +-pi
    return jp


def _dose_stamp(tmp, D, *, sampled=False, sigma=None, T=14, seed=0, meta_extra=None):
    stamp = tmp / "stamp"
    (stamp / "raw").mkdir(parents=True)
    meta = {"joint_names": _joint_names(), "hold_labels": ["settle", "stand", "fwd"], "dt": 0.02,
            "sample_actions": bool(sampled), "zero_actions": False, **(meta_extra or {})}
    if sampled:
        meta["sample_actions_std"] = [float(s) for s in sigma]
    (stamp / "run_meta.json").write_text(json.dumps(meta))
    (stamp / "complete.json").write_text("{}")
    g = np.random.default_rng(seed)
    segs = [np.array([0] * 4 + [1] * 8 + [2] * (T - 12), np.int16),   # settle, stand, fwd
            np.full(T, 2, np.int16),                                   # never stands: skipped
            np.array([1] * 6 + [2] * (T - 6), np.int16)]               # stands from step 0 (kept: no proxy lag)
    for e, seg in enumerate(segs):
        np.savez(stamp / "raw" / f"episode_{e:02d}.npz", joint_pos=_synthetic_joints(T=T, seed=seed + e),
                 actions=g.normal(0, 0.8, (T, 18)).astype(np.float32), segment_id=seg,
                 joint_vel=g.normal(0, 1, (T, 24)).astype(np.float32))
    return stamp


def _ckpt(tmp, std_raw=0.0):
    p = tmp / "graft.pt"
    torch.save({"model_state_dict": {"std_raw": torch.full((18,), float(std_raw))}}, p)
    return p


def test_dose_end_to_end_mean_actions(D, RST, tmp_path):
    lk = D.load_linkage()
    stamp = _dose_stamp(tmp_path, D)
    ck = _ckpt(tmp_path)
    p = RST.RestanceParams()
    out = D.dose(stamp, ck, p, labels=("stand",), rst=RST, lk=lk)
    assert out["actions"] == "mean" and out["debias_c"] == 0.0
    assert out["episodes"] == 2 and out["rows"] == 8 + 6
    assert out["hip_default_source"] == "linkage.hip_default_rad"
    assert out["hip_default"] == pytest.approx([lk.hip_default_rad()] * 6, abs=1e-7)
    for gone in ("held_equals_live_share", "proxy_live_min", "knee_gate_full_share", "step0_rows_dropped"):
        assert gone not in out, gone
    # independent recomputation: explicit loop over the episodes / steps
    sigma = D.anchor_sigma({"std_raw": torch.zeros(18)})
    hd = np.float32(lk.hip_default_rad())
    rows_r, rows_0, shares, hips = [], [], [], []
    for f in sorted((stamp / "raw").glob("*.npz")):
        z = np.load(f)
        for t in range(z["segment_id"].shape[0]):
            if z["segment_id"][t] != 1:
                continue
            obs = torch.zeros(1, 89)
            obs[0, list(RST.CAM_OBS_COLS)] = D.wrap_to_pi(torch.as_tensor(z["joint_pos"][t, 1:12:2]))
            obs[0, list(RST.HIP_POS_OBS_COLS)] = torch.as_tensor(z["joint_pos"][t, 12:18] - hd)
            hips.append(obs[0, list(RST.HIP_POS_OBS_COLS)])
            tg, br = RST.restance_target(obs, p)
            a = torch.as_tensor(z["actions"][t:t + 1])
            rows_r.append(float(0.5 * (((a - tg) / sigma) ** 2).sum()))
            rows_0.append(float(0.5 * ((a / sigma) ** 2).sum()))
            shares.append(br[0])
    assert out["per_row"]["restance"]["mean"] == pytest.approx(np.mean(rows_r), rel=1e-5)
    assert out["per_row"]["target0"]["mean"] == pytest.approx(np.mean(rows_0), rel=1e-5)
    assert out["branch_shares"] == pytest.approx(RST.branch_shares(torch.stack(shares)))
    hips = torch.stack(hips)
    assert out["leg_down_share"] == pytest.approx(float((hips >= p.p_down).float().mean()))
    assert out["leg_up_share"] == pytest.approx(float((hips <= p.p_up).float().mean()))
    assert 0 < out["leg_down_share"] < 1 and 0 < out["leg_up_share"] < 1           # the draw straddles both
    b = out["bands"]["by_share"]["0.16"]
    assert b["expected_restance"] == pytest.approx(0.16 * np.mean(rows_r), rel=1e-5)
    assert b["band"] == pytest.approx([0.52 * 0.16 * np.mean(rows_r), 2.0 * 0.16 * np.mean(rows_r)], rel=1e-5)
    assert set(out["by_label"]) == {"stand"} and out["rc"] in (D.RC_OK, D.RC_STOP)
    assert [out["band_lo"], out["band_hi"]] == out["bands"]["union_band"]
    md = D.markdown(dict(out, xcheck=None))
    assert "| restance |" in md and "held == live" not in md and "legs DOWN" in md


def test_dose_reads_a_recorded_hip_default(D, RST, tmp_path):
    rec = [0.0] * 24
    rec[12:18] = [0.1] * 6                                          # a different default -> different hip inputs
    out_rec = D.dose(_dose_stamp(tmp_path / "a", D, meta_extra={"default_joint_pos": rec}), _ckpt(tmp_path),
                     RST.RestanceParams(), rst=RST)
    out_lk = D.dose(_dose_stamp(tmp_path / "b", D), _ckpt(tmp_path), RST.RestanceParams(), rst=RST)
    assert out_rec["hip_default_source"] == "run_meta.default_joint_pos"
    assert out_rec["hip_default"] == pytest.approx([0.1] * 6)
    assert out_rec["leg_down_share"] < out_lk["leg_down_share"]     # every hip reads 0.073 lower


def test_dose_sampled_actions_debias_and_home_stop(D, RST, tmp_path):
    lk = D.load_linkage()
    sigma = D.anchor_sigma({"std_raw": torch.zeros(18)}).numpy()
    stamp = _dose_stamp(tmp_path, D, sampled=True, sigma=sigma)
    out = D.dose(stamp, _ckpt(tmp_path), RST.RestanceParams(), rst=RST, lk=lk)
    assert out["actions"] == "sampled" and out["debias_c"] == pytest.approx(9.0, rel=1e-6)
    assert out["sigma_sample_equals_sigma_A"] is True
    assert out["per_row"]["restance"]["mean"] == pytest.approx(out["per_row"]["restance_raw_mean"] - 9.0, rel=1e-6)
    pr = out["per_row"]["restance_minus_target0"]["mean"]
    assert pr == pytest.approx(out["per_row"]["restance_raw_mean"] - out["per_row"]["target0_raw_mean"], rel=1e-5)
    assert out["branch_shares"]["home"] > 0 and out["rc"] == D.RC_OK
    # a DOWN band that swallows every synthetic hip (p_down = 0.7 x 0.25 x -1 = -0.175 < -0.15): every leg reads
    # down -> no in-air branch at all -> STOP (exit 2)
    alldown = RST.RestanceParams(lift=-1.0, rho=0.5, down_frac=0.7, up_frac=0.8)
    out2 = D.dose(stamp, _ckpt(tmp_path), alldown, rst=RST, lk=lk)
    assert out2["leg_down_share"] == 1.0
    assert out2["branch_shares"]["home"] == 0.0 and out2["rc"] == D.RC_STOP and "STOP" in out2["verdict"]


def test_dose_refuses_a_zero_action_stamp_and_missing_labels(D, RST, tmp_path):
    lk = D.load_linkage()
    stamp = _dose_stamp(tmp_path, D)
    with pytest.raises(ValueError, match="hold_labels"):
        D.collect_stand_rows(stamp, ("nope",), RST, lk)
    with pytest.raises(ValueError):
        D.actions_kind({"zero_actions": True})
    with pytest.raises(ValueError):
        D.actions_kind({"sample_actions": True})


def test_band_report_verdicts(D):
    b = D.band_report(20.0, 10.0)
    s12, s16 = b["by_share"]["0.12"], b["by_share"]["0.16"]
    assert s12["expected_restance"] == pytest.approx(2.4) and s12["band"] == pytest.approx([1.248, 4.8])
    assert s12["excludes_own_target0"] is True and s12["excludes_z1_realised"] is False
    assert s16["expected_target0"] == pytest.approx(1.6) and s16["excludes_own_target0"] is True
    assert b["union_band"] == pytest.approx([1.248, 6.4])
    assert b["union_excludes_every_target0"] is False
    assert b["target0_inside_union"] == ["target0@0.16", "z1_realised"]
    far = D.band_report(100.0, 10.0)
    assert far["union_excludes_every_target0"] is True and far["target0_inside_union"] == []


def test_dose_xcheck_on_recorded_obs(D, RST, tmp_path):
    lk = D.load_linkage()
    stamp = _dose_stamp(tmp_path, D)
    hd = np.float32(lk.hip_default_rad())
    rows = []
    for f in sorted((stamp / "raw").glob("*.npz")):
        z = np.load(f)
        for t in (0, 5, 9):                                          # step 0 too: no proxy lag any more
            o = np.zeros(120, np.float32)
            o[41:65] = (z["joint_vel"][t] * np.float32(0.05)).astype(np.float32)
            o[list(RST.CAM_OBS_COLS)] = D.wrap_to_pi(torch.as_tensor(z["joint_pos"][t, 1:12:2])).numpy()
            o[list(RST.HIP_POS_OBS_COLS)] = z["joint_pos"][t, 12:18] - hd
            o[83:89] = 0.7                                           # the proxy columns are not compared
            rows.append(o)
    np.savez(stamp / "p2r_bank.npz", obs=np.stack(rows))
    xc = D.xcheck(stamp, RST, lk)
    assert xc["ok"] and xc["matched"] == 9 and xc["hip_max_abs_diff"] == 0.0 and xc["hip_bitwise_share"] == 1.0
    assert xc["cams_bitwise"] and xc["hip_default_source"] == "linkage.hip_default_rad"
    for col in (30, 20):                                             # a hip column, a cam column
        bad = np.stack(rows)
        bad[0, col] += 1e-3
        np.savez(stamp / "p2r_bank.npz", obs=bad)
        assert D.xcheck(stamp, RST, lk)["ok"] is False, col


# ===================================================================================== fidelity: the replica
def _real_modules():
    """The REAL Actor / StateHistoryEncoder / DefaultEstimator (by path, under a throwaway package name; the
    package ``__init__`` files, which reach Isaac, never run)."""
    pkg, sub = "_t_z2_rslmods", "_t_z2_rslmods.feature_extractors"
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [str(MODULES)]
        sys.modules[pkg] = p
        s = types.ModuleType(sub)
        s.__path__ = [str(MODULES / "feature_extractors")]
        sys.modules[sub] = s
    import importlib  # noqa: PLC0415

    acwe = importlib.import_module(pkg + ".actor_critic_with_encoder")
    est = importlib.import_module(sub + ".estimator")
    return acwe, est


DIMS = dict(num_prop=20, num_scan=5, num_priv_explicit=3, num_priv_latent=4, num_hist=10)


def _tiny_actor(seed=0, n_act=18):
    acwe, est = _real_modules()
    torch.manual_seed(seed)
    actor = acwe.Actor(num_actions=n_act, scan_encoder_dims=[9, 6], actor_hidden_dims=[11, 7],
                       priv_encoder_dims=[7, 5], activation=torch.nn.ELU(), tanh_encoder_output=False,
                       state_history_encoder={"class_name": "StateHistoryEncoder", "channel_size": 4}, **DIMS)
    estimator = est.DefaultEstimator(num_prop=DIMS["num_prop"], num_priv_explicit=DIMS["num_priv_explicit"],
                                     hidden_dims=[10, 8])
    for prm in list(actor.parameters()) + list(estimator.parameters()):
        torch.nn.init.normal_(prm, 0.0, 0.3)
    return actor, estimator


def _save_tiny(tmp, actor, estimator, name="model.pt", std_raw=0.0):
    msd = {f"actor.{k}": v.detach().clone() for k, v in actor.state_dict().items()}
    msd["std_raw"] = torch.full((18,), float(std_raw))
    d = tmp / "run"
    (d / "params").mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": msd, "estimator_state_dict": estimator.state_dict()}, d / name)
    agent = {"policy": {"activation": "elu", "tanh_encoder_output": False,
                        "actor": dict(DIMS, state_history_encoder={"channel_size": 4})}}
    (d / "params" / "agent.yaml").write_text(yaml.safe_dump(agent))
    return d / name


def _obs_width():
    return max(89, DIMS["num_prop"] + DIMS["num_scan"] + DIMS["num_priv_explicit"] + DIMS["num_priv_latent"]) + \
        DIMS["num_hist"] * DIMS["num_prop"]


def test_fidelity_replica_matches_the_real_actor_and_estimator(FID, tmp_path):
    actor, estimator = _tiny_actor()
    ck = _save_tiny(tmp_path, actor, estimator)
    sd, esd = FID.load_ckpt(ck)
    w = FID.widths(ck.parent / "params" / "agent.yaml")
    obs = torch.randn(32, _obs_width(), generator=torch.Generator().manual_seed(1))
    s0 = DIMS["num_prop"] + DIMS["num_scan"]
    with torch.no_grad():
        for hist in (True, False):
            want = actor(obs, hist)
            assert torch.allclose(FID.eval_mu(obs, sd, esd, w, hist_encoding=hist, priv="true"), want, atol=1e-6)
        o2 = obs.clone()
        o2[:, s0:s0 + DIMS["num_priv_explicit"]] = estimator.inference(obs[:, :DIMS["num_prop"]])
        assert torch.allclose(FID.eval_mu(obs, sd, esd, w), actor(o2, True), atol=1e-6)
    assert not torch.equal(FID.eval_mu(obs, sd, esd, w), FID.eval_mu(obs, sd, esd, w, priv="true"))
    with pytest.raises(ValueError):
        FID.eval_mu(obs, sd, {}, w)                                   # the eval path needs the estimator


def _fid_bank(RST, n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, _obs_width(), generator=g)
    obs[:, list(RST.PROXY_OBS_COLS)] = torch.rand(n, 6, generator=g) * 1.4 - 0.5
    obs[:, list(RST.CAM_OBS_COLS)] = (torch.rand(n, 6, generator=g) - 0.5) * 0.3
    tgt, br = RST.restance_target(obs, RST.RestanceParams())
    return {"obs": obs.numpy(), "target": tgt.numpy(), "branch": br.numpy().astype(np.int8),
            "s1_sigma": np.full(18, 0.5, np.float32)}


def test_fidelity_score_and_its_line(FID, RST):
    bank = _fid_bank(RST)
    tgt = torch.as_tensor(bank["target"])
    hold = torch.as_tensor(bank["branch"]).long() == RST.BRANCH_HOLD
    assert hold.any()
    ok = FID.score(bank, tgt.clone(), RST)
    assert ok["rc"] == FID.RC_OK and ok["hold_cam_residual_max_abs"] == 0.0
    assert ok["prior_per_row"]["mean"] == 0.0 and set(ok["by_branch"]) == set(RST.BRANCH_NAMES)
    leg = int(hold.any(dim=0).nonzero()[0])
    mu = tgt.clone()
    mu[:, leg] += 0.02                                               # cam of that leg, every row
    bad = FID.score(bank, mu, RST)
    assert bad["rc"] == FID.RC_NOTIFY and bad["hold_cam_residual_signed_by_leg"][RST.LEGS[leg]] == pytest.approx(0.02)
    mu[:, leg] -= 0.01                                               # +0.01: inside the 0.015 line
    assert FID.score(bank, mu, RST)["rc"] == FID.RC_OK
    mu2 = tgt.clone()
    mu2[:, 6 + leg] += 0.5                                           # a hip residual never trips the CAM line
    assert FID.score(bank, mu2, RST)["rc"] == FID.RC_OK
    nohold = dict(bank, branch=np.full_like(bank["branch"], RST.BRANCH_HOME))
    assert FID.score(nohold, tgt, RST)["rc"] == FID.RC_NO_ROWS


def test_fidelity_cli_rc_and_dry_run(FID, RST, tmp_path):
    actor, estimator = _tiny_actor(seed=3)
    ck = _save_tiny(tmp_path, actor, estimator)
    bank = _fid_bank(RST, n=40)
    np.savez(tmp_path / "p2r_stand_bank.npz", **bank, restance=np.array("default"), has_residual=np.array(False))
    out_json = tmp_path / "fid.json"
    # NOTE(p2r Z2c, 2026-10-10): R is recomputed with the run's constants; the tiny run's agent.yaml carries none
    rc = FID.main(["--bank", str(tmp_path / "p2r_stand_bank.npz"), "--ckpt", str(ck), "--restance", "default",
                   "--json", str(out_json)])
    out = json.loads(out_json.read_text())
    assert rc == out["exit"] == out["rc"] and rc in (FID.RC_OK, FID.RC_NOTIFY)
    assert out["hist_encoding"] is True and out["priv"] == "estimator"
    assert out["agent_yaml"] == str(ck.parent / "params" / "agent.yaml")
    # a random tiny actor is far from R: the line fires; --dry-run reports it and exits 0
    assert rc == FID.RC_NOTIFY
    rc2 = FID.main(["--bank", str(tmp_path / "p2r_stand_bank.npz"), "--ckpt", str(ck), "--dry-run",
                    "--restance", "default", "--json", str(out_json)])
    out2 = json.loads(out_json.read_text())
    assert rc2 == 0 and out2["would_notify"] is True and out2["dry_run"] is True
    assert FID.main(["--bank", str(tmp_path / "nope.npz"), "--ckpt", str(ck)]) == FID.RC_ERROR


# ===================================================================================== p2r_early.py --mode z2
def test_early_z2_formation_line_and_fidelity_judge(EA):
    F0 = 19995
    pts = [(F0 + i, 2.0 if i < 400 else 0.45) for i in range(1100)]   # 0.45 <= 0.25 x 2.0 at +500
    assert EA.judge_formation(pts, F0, at=EA.Z2_FORMATION_AT, frac=EA.Z2_FORMATION_FRAC)["rc"] == EA.RC_OK
    # Z1's line on the same series (0.1x at +200) still flags: the two modes differ
    assert EA.judge_formation(pts, F0)["rc"] == EA.RC_NOTIFY
    assert EA.judge_fidelity(None)["rc"] == EA.RC_WAIT
    assert EA.judge_fidelity({"rc": 3})["rc"] == EA.RC_NOTIFY
    with pytest.raises(ValueError):
        EA.judge_checkpoint("model_20500.pt", 0.1, 1.0, None, {}, F0, mode="z3")


def test_early_z2_needs_the_stand_bank(EA, tmp_path):
    out = EA.run_once(tmp_path, tmp_path / "s1.pt", tmp_path, tmp_path, tmp_path / "b.npz", mode="z2",
                      stand_bank=None)
    assert out["rc"] == EA.RC_ERROR and "stand-bank" in out["why"]
    with pytest.raises(SystemExit):
        EA.main(["--run-dir", str(tmp_path), "--bank", str(tmp_path / "b.npz"), "--mode", "z2"])
