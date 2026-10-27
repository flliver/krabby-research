"""Stand-network campaign (``2026-10-11_stand_network``) step 3: ``scripts/sn_common.py`` on CPU (no Isaac / Kit / GPU).

Pinned here:

* the module is torch-free at import (map risk L6), R is loaded by path only on first use;
* R's D3 constants: the exact string round-trips, differ from ``RestanceParams()`` / ``"default"`` (map T4), and
  ``r_label`` is ``restance_target`` with them, bitwise, on raw / own-estimate / masked obs alike;
* the run_meta field names equal ``sn_calib``'s head-validity contract, ENV4 / the env var names equal ``p2_read``'s;
  ``env_reasons`` passes the REFS_S1_s18 env stamp and refuses every D5 / L10 breach;
* ``stand_net_input`` (the ONE D1 function): a NEW tensor (the caller's obs is never written, L2); the estimate is
  taken on the UNMASKED prop and the mask applied after it (L3); with mask 0 it is bitwise the harness's in-place
  estimate write; with mask 1 it equals ``stand_mask_obs`` on (estimate-then-cat) and differs only in the 264 mask
  columns; bad arguments refuse;
* with the REAL S1 networks (a CPU copy of model_19994): S1 used as the stand network with mask 0 gives S1's actions
  bitwise (the identity-bridge premise on CPU); mask 1 changes them; ``route_actions`` = ``torch.where``;
* inference-mode cache safety of the mask tensors (the trainer masks inside inference mode and trains outside).
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-11_stand_network"
SCRIPTS = CAMP / "scripts"
COMMON = SCRIPTS / "sn_common.py"
SN_CALIB = SCRIPTS / "sn_calib.py"
P2L_PATH = TASK / "experiments/2026-10-07_omni_phase2/scripts/p2r_mode_eval_crab_hex_gait.py"
RESTANCE_PY = TASK / "mdp/crab_hex_omni_restance.py"
RSL_MODULES = REPO / "parkour/scripts/rsl_rl/modules"
S1_DIR = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"
S1_CKPT = S1_DIR / "model_19994.pt"
S1_SHA = "968cc5509899"
REFS_SUITE1 = REPO / "parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24"
D3 = ("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
      "tilt_max=0.1,eps_step=0.3")
W, P0, P1, NP = 1149, 207, 216, 75


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


@pytest.fixture(scope="module")
def SN():
    return _load(COMMON, "_t_sn_common")


@pytest.fixture(scope="module")
def RST():
    return _load(RESTANCE_PY, "_t_sn_restance")


# ===================================================================================== real S1 (CPU copy)
def rsl_modules():
    name = "_sn_test_rslmods"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(RSL_MODULES)]
        sys.modules[name] = pkg
    crab = importlib.import_module(f"{name}.crab_actor_critic_with_encoder")
    est = importlib.import_module(f"{name}.feature_extractors.estimator")
    return crab.CrabHexActorCriticRMA, est.DefaultEstimator


def s1_cfgs():
    agent = yaml.safe_load((S1_DIR / "params/agent.yaml").read_text())
    pc, ec = copy.deepcopy(agent["policy"]), copy.deepcopy(agent["estimator"])
    pc.pop("class_name")
    ec.pop("class_name")
    return pc, ec


def build_real(state: dict):
    """A fresh CrabHexActorCriticRMA + DefaultEstimator from S1's own agent cfg, loaded STRICTLY from ``state``."""
    crab_cls, est_cls = rsl_modules()
    pc, ec = s1_cfgs()
    pol = crab_cls(W, 18, **pc)
    est = est_cls(**ec)
    pol.load_state_dict(state["model_state_dict"], strict=True)
    est.load_state_dict(state["estimator_state_dict"], strict=True)
    return pol.eval(), est.eval()


@pytest.fixture(scope="module")
def s1_state():
    if not S1_CKPT.is_file():
        pytest.skip("S1 model_19994.pt is not on this machine")
    ck = torch.load(S1_CKPT, map_location="cpu", weights_only=False)      # its pickle holds cuda:0 storages
    return {"model_state_dict": {k: v.detach().cpu().clone() for k, v in ck["model_state_dict"].items()},
            "estimator_state_dict": {k: v.detach().cpu().clone() for k, v in ck["estimator_state_dict"].items()},
            "iter": ck["iter"], "infos": ck["infos"]}


def obs_batch(n=48, seed=0) -> torch.Tensor:
    """A plausible 1149-wide batch: small values, non-zero last-action / proxy blocks (so the mask bites), level IMU."""
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(n, W, generator=g) * 0.3
    o[:, 3:5] *= 0.1
    return o


def harness_obs(obs: torch.Tensor, est) -> torch.Tensor:
    """The harness's in-place write (H:1027-1029) on a copy."""
    o = obs.clone()
    o[:, P0:P1] = est.inference(o[:, :NP])
    return o


# ===================================================================================== import hygiene (L6)
def test_import_is_torch_free():
    code = (f"import importlib.util, sys\n"
            f"s = importlib.util.spec_from_file_location('_x', {str(COMMON)!r})\n"
            f"m = importlib.util.module_from_spec(s); sys.dont_write_bytecode = True; s.loader.exec_module(m)\n"
            f"assert m.RESTANCE_D3\n"
            f"bad = [k for k in ('torch', 'numpy', '_sn_restance', 'isaaclab') if k in sys.modules]\n"
            f"print('BAD', bad)\n"
            f"sys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout + r.stderr


def test_paths_and_restance_module_is_the_launchers(SN):
    assert SN.RESTANCE_MODULE == RESTANCE_PY.resolve() and SN.RESTANCE_MODULE.is_file()
    assert SN.CAMPAIGN == CAMP.resolve() and SN.TASK == TASK.resolve()
    p2l = _load(P2L_PATH, "_t_sn_common_p2l")
    assert p2l.RESTANCE_MODULE.resolve() == SN.RESTANCE_MODULE


# ===================================================================================== R (D3)
def test_d3_string_round_trips_and_is_not_the_default(SN, RST):
    assert SN.RESTANCE_D3 == D3
    p = SN.r_params()
    assert RST.format_restance(RST.parse_restance(D3)) == D3
    assert p.to_dict() == RST.parse_restance(D3).to_dict()
    assert p.to_dict() != RST.RestanceParams().to_dict()
    assert p.to_dict() != RST.parse_restance("default").to_dict()
    assert p.p_up == pytest.approx(-0.0625) and p.p_down == pytest.approx(-0.05)
    assert (p.up_frac, p.k_h, p.trim, p.eps_step) == (0.5, 1.0, 0.1, 0.3)


def test_r_label_is_restance_target_with_d3_on_any_view(SN, RST):
    obs = obs_batch(64, seed=3)
    obs[:, list(RST.CAM_OBS_COLS)] = (torch.rand(64, 6) - 0.5) * 0.9
    obs[:, list(RST.HIP_POS_OBS_COLS)] = torch.where(torch.rand(64, 6) < 0.6, torch.zeros(64, 6),
                                                     torch.rand(64, 6) * -0.2)
    want_t, want_b = RST.restance_target(obs, RST.parse_restance(D3))
    est = types.SimpleNamespace(inference=lambda x: torch.tanh(x[:, :9] * 3.0))
    views = [obs, SN.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=False),
             SN.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=True)]
    for v in views:
        t, b = SN.r_label(v)
        assert torch.equal(t, want_t) and torch.equal(b, want_b)
    # the default R labels differently on the same states (the T4 hazard is real)
    t_def, _ = RST.restance_target(obs, RST.RestanceParams())
    assert not torch.equal(t_def, want_t)


# ===================================================================================== contract constants
def test_run_meta_names_match_sn_calib_and_p2_read(SN):
    snc = _load(SN_CALIB, "_t_sn_common_calib")
    assert SN.VARIANT == snc.STAND_NET_VARIANT == "stand_net"
    assert SN.CKPT_KEY == snc.STAND_NET_CKPT_KEY and SN.MASK_KEY == snc.STAND_NET_MASK_KEY
    P = snc.P
    assert SN.ENV4 == P.ENV4
    assert SN.STAND_OBS_VAR == snc.STAND_OBS_VAR
    assert SN.STAND_MASK_VAR == P.STAND_MASK_VAR and SN.STAND_PROXY_VAR == P.STAND_PROXY_VAR
    rst = SN.load_rst()
    assert SN.RESTANCE_VAR == rst.RESTANCE_VAR and SN.ACTION_DIM == rst.ACTION_DIM and SN.NUM_LEGS == len(rst.LEGS)
    assert len(set(SN.RAW_KEYS)) == 4 and all(k.startswith("sn_") for k in SN.RAW_KEYS)


@pytest.mark.skipif(not (REFS_SUITE1 / "run_meta.json").is_file(), reason="REFS_S1_s18 env stamp not on this machine")
def test_env_reasons_pass_the_refs_env_stamp(SN):
    env = json.loads((REFS_SUITE1 / "run_meta.json").read_text())["env_vars"]
    assert env["KRABBY_OMNI_STAND_OBS"] == "0"
    assert SN.env_reasons(env) == []


@pytest.mark.parametrize("patch, needle", [
    ({"KRABBY_OMNI_STAND_OBS": None}, "STAND_OBS"),
    ({"KRABBY_OMNI_STAND_OBS": "clock"}, "STAND_OBS"),
    ({"KRABBY_OMNI_STAND_OBS": "1"}, "STAND_OBS"),
    ({"KRABBY_OMNI_STAND_MASK": "1"}, "STAND_MASK"),
    ({"KRABBY_OMNI_STAND_PROXY": "hold"}, "STAND_PROXY"),
    ({"KRABBY_RESTANCE": D3}, "KRABBY_RESTANCE"),
    ({"KRABBY_RESTANCE": "default"}, "KRABBY_RESTANCE"),
    ({"KRABBY_OMNI_STAND_HOLD": "1"}, "STAND_HOLD"),
    ({"KRABBY_OMNI_STAND_PARK": "1"}, "STAND_PARK"),
    ({"KRABBY_OMNI_CLOCK_HIDE": None}, "CLOCK_HIDE"),
    ({"KRABBY_STD_PARAM": "scalar"}, "STD_PARAM"),
    ({"KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.25:0.25"}, "SPEED_ELLIPSE"),
])
def test_env_reasons_refuse(SN, patch, needle):
    env = {"KRABBY_OMNI_STAND_OBS": "0", "KRABBY_OMNI_STAND_MASK": "0", "KRABBY_OMNI_STAND_HOLD": "0", **SN.ENV4}
    assert SN.env_reasons(env) == []
    for k, v in patch.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    r = SN.env_reasons(env)
    assert r and any(needle in x for x in r), r


def test_env_reasons_env4_optional(SN):
    assert SN.env_reasons({"KRABBY_OMNI_STAND_OBS": "0"}, env4=False) == []
    assert SN.env_reasons({"KRABBY_OMNI_STAND_OBS": "0"})


# ===================================================================================== stand_net_input (synthetic)
class LinEst:
    """A tiny estimator reading the prop, including the masked cols 65-74 (as the real one does)."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(NP, P1 - P0, generator=g) * 0.1
        self.seen = []

    def inference(self, x):
        self.seen.append(x.clone())
        with torch.no_grad():
            return torch.tanh(x @ self.w)


@pytest.mark.parametrize("mask", [False, True, 0, 1])
def test_input_never_writes_obs_and_estimates_unmasked_prop(SN, RST, mask):
    obs = obs_batch(16, seed=1)
    before = obs.clone()
    est = LinEst()
    x = SN.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=mask)
    assert torch.equal(obs, before)                                      # L2: the caller's obs is never written
    assert x.data_ptr() != obs.data_ptr() and x.shape == obs.shape
    assert len(est.seen) == 1 and torch.equal(est.seen[0], before[:, :NP])   # L3: the UNMASKED prop
    recipe = torch.cat([before[:, :P0], torch.tanh(before[:, :NP] @ est.w), before[:, P1:]], dim=1)
    want = RST.stand_mask_obs(recipe) if bool(mask) else recipe
    assert torch.equal(x, want)


def test_mask_after_estimate_matters_and_touches_only_the_mask_columns(SN, RST):
    obs = obs_batch(16, seed=2)
    est = LinEst(seed=4)
    x0 = SN.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=False)
    x1 = SN.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=True)
    cols, vals = RST.stand_mask_columns(W)
    assert len(cols) == 264
    other = torch.ones(W, dtype=torch.bool)
    other[cols] = False
    assert torch.equal(x1[:, other], x0[:, other])                       # estimate cols included: unmasked-prop estimate
    assert torch.equal(x1[:, cols], torch.tensor(vals).expand(16, -1))
    masked_first = RST.stand_mask_obs(obs)
    wrong = torch.tanh(masked_first[:, :NP] @ est.w)                     # mask-before-estimate (L3) would give this
    assert not torch.equal(wrong, x1[:, P0:P1])


def test_mask_zero_is_bitwise_the_harness_in_place_write(SN):
    obs = obs_batch(16, seed=5)
    est = LinEst(seed=1)
    h = obs.clone()
    h[:, P0:P1] = est.inference(h[:, :NP])                               # the harness's write
    assert torch.equal(SN.stand_net_input(h, est, p0=P0, p1=P1, num_prop=NP, mask=False), h)


@pytest.mark.parametrize("kw", [dict(mask=2), dict(mask="1"), dict(mask=None), dict(mask=1.0),
                                dict(num_prop=208), dict(p0=216, p1=207), dict(p1=W + 1)])
def test_input_refuses_bad_arguments(SN, kw):
    args = dict(p0=P0, p1=P1, num_prop=NP, mask=False)
    args.update(kw)
    with pytest.raises((TypeError, ValueError)):
        SN.stand_net_input(obs_batch(4), LinEst(), **args)


def test_input_refuses_a_wrong_width_estimate(SN):
    bad = types.SimpleNamespace(inference=lambda x: x[:, :5])
    with pytest.raises(ValueError, match="estimate"):
        SN.stand_net_input(obs_batch(4), bad, p0=P0, p1=P1, num_prop=NP, mask=False)


def test_route_actions_is_torch_where(SN):
    st = torch.tensor([True, False, True, False])
    a_st, a_s1 = torch.full((4, 18), 2.0), torch.full((4, 18), -3.0)
    out = SN.route_actions(st, a_st, a_s1)
    assert torch.equal(out[st], a_st[st]) and torch.equal(out[~st], a_s1[~st])
    with pytest.raises(ValueError):
        SN.route_actions(st.long(), a_st, a_s1)
    with pytest.raises(ValueError):
        SN.route_actions(st[:3], a_st, a_s1)


def test_state_dicts_equal(SN):
    a = {"w": torch.ones(3), "b": torch.zeros(2)}
    assert SN.state_dicts_equal(a, {k: v.clone() for k, v in a.items()}) == (True, "")
    assert not SN.state_dicts_equal(a, {"w": torch.ones(3)})[0]
    assert not SN.state_dicts_equal(a, {"w": torch.ones(3), "b": torch.tensor([0.0, 1e-7])})[0]
    assert not SN.state_dicts_equal(a, {"w": torch.ones(3, dtype=torch.float64), "b": torch.zeros(2)})[0]


def test_mask_cache_built_in_inference_mode_still_trains_outside(SN):
    """The trainer masks inside inference mode (rollout) and trains on a stored copy outside it."""
    rst = SN.load_rst()
    rst._STAND_MASK_TENSORS.clear()
    est = LinEst()
    with torch.inference_mode():
        x_inf = SN.stand_net_input(obs_batch(4), est, p0=P0, p1=P1, num_prop=NP, mask=True)
    buf = torch.zeros(4, W)
    with torch.inference_mode():
        buf.copy_(x_inf)                                                 # into a pre-allocated normal tensor
    lin = torch.nn.Linear(W, 3)
    lin(buf).sum().backward()
    assert lin.weight.grad is not None
    x = SN.stand_net_input(obs_batch(4), est, p0=P0, p1=P1, num_prop=NP, mask=True)   # cached inference tensors
    lin(x).sum().backward()


# ===================================================================================== real S1 (CPU copy)
def test_real_s1_identity_premise_mask0_bitwise_and_mask1_changes_actions(SN, s1_state):
    s1_pol, s1_est = build_real(s1_state)
    st_pol, st_est = build_real(s1_state)                               # "S1 as the stand checkpoint"
    obs = obs_batch(64, seed=11)
    with torch.inference_mode():
        h = harness_obs(obs, s1_est)
        a_s1 = s1_pol.act_inference(h, hist_encoding=True)
        x0 = SN.stand_net_input(h, st_est, p0=P0, p1=P1, num_prop=NP, mask=False)
        a0 = st_pol.act_inference(x0, hist_encoding=True)
        x1 = SN.stand_net_input(h, st_est, p0=P0, p1=P1, num_prop=NP, mask=True)
        a1 = st_pol.act_inference(x1, hist_encoding=True)
        a1_priv = st_pol.act_inference(x1, hist_encoding=False)
    assert torch.equal(x0, h) and torch.equal(a0, a_s1)                 # identity-bridge premise, bitwise on CPU
    assert not torch.equal(a1, a_s1)                                     # the mask reaches the hist-path actor
    assert (a1 != a_s1).any(dim=1).float().mean() > 0.9
    assert not torch.equal(a1_priv, a1)                                  # L11: the priv path is a different function
    stand = torch.arange(64) % 3 == 0
    routed = SN.route_actions(stand, a1, a_s1)
    assert torch.equal(routed[stand], a1[stand]) and torch.equal(routed[~stand], a_s1[~stand])


def test_real_s1_estimator_reads_the_masked_columns(SN, s1_state):
    """Why the estimate must come first (L3): the real estimator's output moves when cols 65-74 are masked."""
    _, est = build_real(s1_state)
    rst = SN.load_rst()
    obs = obs_batch(32, seed=7)
    with torch.inference_mode():
        e_raw = est.inference(obs[:, :NP])
        e_masked = est.inference(rst.stand_mask_obs(obs)[:, :NP])
        x1 = SN.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=True)
    assert not torch.equal(e_raw, e_masked)
    assert torch.equal(x1[:, P0:P1], e_raw)


def test_real_s1_cpu_copy_is_the_record_checkpoint(s1_state):
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256(S1_CKPT.read_bytes()).hexdigest()
    assert h.startswith(S1_SHA)
    assert s1_state["iter"] == 19994 and "std_raw" in s1_state["model_state_dict"]
    assert s1_state["model_state_dict"]["critic.0.weight"].shape[1] == W
