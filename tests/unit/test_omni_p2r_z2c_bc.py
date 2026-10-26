"""Omni phase-2 stage p2r Z2c, the clean-state cloning (BC) term (``P2R_Z2C_PLAN.md`` step 1; CHARTER 12), on CPU.

Pinned here (no Isaac, no GPU; the tiny real ``CrabHexActorCriticRMA`` + real ``PPOWithExtractor`` of the statue tests):

* knob off (``bc_coef = 0``, no bank) -> nothing is loaded and the update is BITWISE the Z2 update: parameters,
  gradients, optimizer states, the LR, the counter, the global RNG state, the loss-dict keys, ``resume_state``;
* BC on: the bank's training-path columns + R (this run's constants, on the FULL-width rows) are loaded once as
  plain attributes (no ``state_dict``); ``L_bc`` equals the hand formula ``mean_rows 0.5 sum_d ((mu - R)/sigma_A)^2``
  on the rows the PRIVATE generator draws; ``mu`` is ``actor(rows, False)`` bit for bit; the gradient reaches the scan
  encoder and the backbone and NEVER the priv encoder (nor the history encoder, the critic, the std); the hook is
  held only for the call (PPO's own forward still trains the priv encoder);
* the private generator never touches torch's global RNG (an update with BC leaves it where the Z2 update does);
* ``Loss/bc`` (mean over stepped minibatches) / ``Loss/bc_mb0`` (the first minibatch, before any step) /
  ``Loss/grad_norm_bc`` (the 50-iteration split, logging only) and the ``[PPO] construction`` line;
* the generator state rides in ``resume_state`` only when armed and is restored on load;
* bank validation refusals (no ``obs``, width, empty, non-finite, a row whose stand bit is not 1) and the
  constructor's cross rules;
* the three knobs (``KRABBY_BC_W`` / ``KRABBY_BC_BANK`` / ``KRABBY_BC_BATCH``): parsers, cross rules, existence,
  repo-relative resolution, what they write; the ladder: registration, dump checks, the log audit and the dry-run
  composition rule.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("tensordict")
yaml = pytest.importorskip("yaml")
try:  # the training stack these modules import (rsl-rl-lib 5.x); CI pins 3.1.2 for compute/parkour, which has neither
    import rsl_rl.extensions  # noqa: E402,F401
    from rsl_rl.modules import EmpiricalNormalization  # noqa: E402,F401
except ImportError:
    pytest.skip("needs the Isaac training stack (rsl-rl-lib 5.x: rsl_rl.extensions, rsl_rl.modules.EmpiricalNormalization);"
                " CI pins rsl-rl-lib 3.1.2 for compute/parkour", allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
RESTANCE_PY = OMNI / "mdp/crab_hex_omni_restance.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"
RUN_PHASES_PY = OMNI / "experiments/tools/run_phases.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_unregistered(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC = _load(UNIT / "test_omni_p2r_statue_core.py", "p2r_z2c__statue_core")
PH = SC.PH
ppo_mod = PH.ppo_mod
mk = _load(MOTION_KNOBS_PY, "motion_knobs__p2r_z2c_test")
RST = _load_unregistered(RESTANCE_PY, "_t_p2r_z2c_restance")

STAND_COL = SC.STAND_COL
#: The Z2c probe's drift-proof constants (P2R_Z2C_PLAN step 3): the dead band and the trim authority.
P_Z2C = RST.RestanceParams(up_frac=0.5, eps_step=0.3, trim=0.1, k_h=1.0)
#: The tiny actor's training-path width (prop + scan + priv_explicit + priv_latent).
TRAIN_COLS = PH.NUM_PROP + PH.NUM_SCAN + PH.NUM_PRIV_EXPLICIT + PH.NUM_PRIV_LATENT
W = 1149
BC_KNOBS = ("KRABBY_BC_W", "KRABBY_BC_BANK", "KRABBY_BC_BATCH")


# ===================================================================== helpers
def _bank_obs(n: int, seed: int, col: int = STAND_COL, width: int = W) -> np.ndarray:
    """Stand rows straddling R's bands (cams around eps_c .. eps_step, hips around the thresholds, mostly level)."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, width, generator=g)
    obs[:, list(RST.CAM_OBS_COLS)] = (torch.rand(n, 6, generator=g) - 0.5) * 0.9
    obs[:, list(RST.HIP_POS_OBS_COLS)] = torch.rand(n, 6, generator=g) * (1.3 * RST.ROD_SCALE * P_Z2C.lift)
    down = torch.rand(n, 6, generator=g) < 0.6
    obs[:, list(RST.HIP_POS_OBS_COLS)] = torch.where(down, torch.zeros(n, 6), obs[:, list(RST.HIP_POS_OBS_COLS)])
    obs[:, [RST.ROLL_OBS_COL, RST.PITCH_OBS_COL]] = (torch.rand(n, 2, generator=g) - 0.5) * 0.1
    obs[:, col] = 1.0
    return obs.numpy().astype(np.float32)


def _write_bank(path: Path, obs: np.ndarray, **extra) -> Path:
    np.savez(path, obs=obs, **extra)
    return path


def _banks(tmp_path: Path, sizes=(40, 25)) -> list[Path]:
    return [_write_bank(tmp_path / f"bank{i}.npz", _bank_obs(n, seed=100 + i)) for i, n in enumerate(sizes)]


def _z2_kw(restance: RST.RestanceParams = P_Z2C) -> dict:
    return {"stand_prior_target": "restance", "restance": restance.to_dict()}


def _z2(tmp_path, ckpt, seed=0, **kw):
    return SC._statue_alg(tmp_path, seed=seed, ckpt=ckpt, **_z2_kw(), **kw)


def _z2c(tmp_path, ckpt, banks, seed=0, coef=1.0, batch=32, bc_seed=5, **kw):
    return _z2(tmp_path, ckpt, seed=seed, bc_coef=coef, bc_bank=[str(b) for b in banks], bc_batch_size=batch,
               bc_seed=bc_seed, **kw)


def _full_bank(banks) -> torch.Tensor:
    return torch.cat([torch.from_numpy(np.load(b)["obs"]) for b in banks])


def _hand_bc(alg, gen_state: torch.Tensor, banks) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(L_bc, mu, idx)`` by hand: the rows a generator in ``gen_state`` draws, ``actor(rows, False)``, the formula."""
    full = _full_bank(banks)
    target, _ = RST.restance_target(full, P_Z2C)
    g = torch.Generator()
    g.set_state(gen_state.clone())
    idx = torch.randint(full.shape[0], (alg.bc_batch_size,), generator=g)
    mu = alg.policy.actor(full[idx, :TRAIN_COLS], False)
    loss = (0.5 * ((mu - target[idx]) / alg.anchor_sigma) ** 2).sum(-1).mean()
    return loss, mu, idx


@pytest.fixture()
def ckpt(tmp_path):
    return SC._anchor_ckpt(tmp_path, seed=5)            # anchor != policy: every statue term live


# ===================================================================== contract
def test_constants_and_knob_registration():
    assert ppo_mod.BC_BANK_OBS_WIDTH == 1149
    assert mk.BC_KNOBS == BC_KNOBS == (mk.BC_W_VAR, mk.BC_BANK_VAR, mk.BC_BATCH_VAR)
    assert mk.BC_BATCH_DEFAULT == 2048
    assert set(BC_KNOBS) <= set(mk.AGENT_KNOBS) and set(BC_KNOBS) <= set(mk.MOTION_KNOB_PARSERS)
    # runner side, never carried into an eval, not PPO-construction knobs
    assert not set(BC_KNOBS) & set(mk.EVAL_SEMANTIC_KNOBS + mk.EVAL_STRUCTURE_KNOBS + mk.PPO_CONSTRUCTION_KNOBS)
    assert "KRABBY_BC_" not in RUN_PHASES_PY.read_text()               # out of the eval-locked EVAL_CARRY
    # the actor's training path the bank keeps (0:257 on the omni actor) holds R's columns' owner rows only
    assert max(RST.CAM_OBS_COLS + RST.HIP_POS_OBS_COLS) < 75 + 132 + 9 + 41


# ===================================================================== knob off
def test_knob_off_is_the_z2_update_bitwise(tmp_path, ckpt, capsys):
    base = _z2(tmp_path, ckpt)
    line_base = capsys.readouterr().out
    off = _z2(tmp_path, ckpt, bc_coef=0.0, bc_bank=None, bc_batch_size=512, bc_seed=7)
    line_off = capsys.readouterr().out
    assert line_off == line_base and "bc_" not in line_off
    for a in (base, off):
        assert a.bc_obs is None and a.bc_target is None and a.bc_generator is None and a.bc_bank is None
    assert base.resume_state().keys() == off.resume_state().keys() and "bc_generator" not in off.resume_state()
    for it in (0, 1, 50):                                               # the split's cadence never matters when off
        la, lb = SC._update(base, iteration=it), SC._update(off, iteration=it)
        assert la == lb and list(la) == list(lb)
        assert not {"bc", "bc_mb0", "grad_norm_bc"} & set(la)
        PH._assert_same(PH._snapshot(base), PH._snapshot(off))         # incl. the global RNG state
    assert base.policy.state_dict().keys() == off.policy.state_dict().keys()


# ===================================================================== the loaded bank
def test_the_bank_is_loaded_once_as_plain_attributes(tmp_path, ckpt, capsys):
    banks = _banks(tmp_path)
    capsys.readouterr()
    alg = _z2c(tmp_path, ckpt, banks)
    out = capsys.readouterr().out
    full = _full_bank(banks)
    assert torch.equal(alg.bc_obs, full[:, :TRAIN_COLS])               # the training-path columns only
    target, _ = RST.restance_target(full, P_Z2C)                        # R on the FULL-width rows, run's constants
    assert torch.equal(alg.bc_target, target) and not alg.bc_target.requires_grad
    assert bool((target.abs().sum(-1) > 0).any())                       # a live target
    assert alg.bc_obs.dtype == alg.bc_target.dtype == torch.float32
    # in no state_dict, no optimizer
    off = _z2(tmp_path, ckpt)
    assert alg.policy.state_dict().keys() == off.policy.state_dict().keys()
    opt_ids = {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
    assert id(alg.bc_obs) not in opt_ids and id(alg.bc_target) not in opt_ids
    line = [ln for ln in out.splitlines() if ln.startswith("[PPO] construction:")][-1]
    assert f"bc_coef=1.0, bc_bank_rows={full.shape[0]}, bc_batch_size=32, bc_seed=5" in line
    assert "stand_prior_target=restance" in line and f"restance={RST.format_restance(P_Z2C)}" in line


def test_bc_loss_is_the_hand_formula_on_the_private_generators_rows(tmp_path, ckpt):
    banks = _banks(tmp_path)
    alg = _z2c(tmp_path, ckpt, banks)
    state = alg.bc_generator.get_state()
    want, mu_hand, idx = _hand_bc(alg, state, banks)
    got = alg._bc_loss()
    assert torch.allclose(got, want, rtol=1e-6, atol=0)
    # the forward IS actor(rows, False), bit for bit (the hook only detaches)
    rows = alg.bc_obs[idx]
    assert torch.equal(alg._bc_policy_mean(rows), alg.policy.actor(rows, False))
    assert torch.equal(mu_hand, alg._bc_policy_mean(rows))
    # the generator advanced by exactly one draw of bc_batch_size rows
    g = torch.Generator()
    g.set_state(state.clone())
    torch.randint(alg.bc_obs.shape[0], (alg.bc_batch_size,), generator=g)
    assert torch.equal(alg.bc_generator.get_state(), g.get_state())
    # the gradient w.r.t. the backbone is the hand formula's gradient
    params = list(alg.policy.actor.actor_backbone.parameters())
    alg.bc_generator.set_state(state.clone())
    g_got = torch.autograd.grad(alg._bc_loss(), params)
    rows = alg.bc_obs[idx]
    with torch.no_grad():
        latent = alg.policy.actor.infer_priv_latent(rows)
    mu = alg.policy.actor.actor_backbone(torch.cat(
        [rows[:, :PH.NUM_PROP], alg.policy.actor.scan_encoder(rows[:, PH.NUM_PROP:PH.NUM_PROP + PH.NUM_SCAN]),
         rows[:, PH.NUM_PROP + PH.NUM_SCAN:PH.NUM_PROP + PH.NUM_SCAN + PH.NUM_PRIV_EXPLICIT], latent], dim=1))
    target = alg.bc_target[idx]
    g_hand = torch.autograd.grad((0.5 * ((mu - target) / alg.anchor_sigma) ** 2).sum(-1).mean(), params)
    for a, b in zip(g_got, g_hand):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-8)


def test_bc_gradient_reaches_backbone_and_scan_encoder_never_the_priv_encoder(tmp_path, ckpt):
    alg = _z2c(tmp_path, ckpt, _banks(tmp_path))
    actor = alg.policy.actor
    loss = alg._bc_loss()
    groups = {"backbone": list(actor.actor_backbone.parameters()), "scan": list(actor.scan_encoder.parameters()),
              "priv": list(actor.priv_encoder.parameters()), "hist": list(actor.history_encoder.parameters()),
              "critic": list(alg.policy.critic.parameters()), "std": [alg.policy.std_raw]}
    flat = [p for ps in groups.values() for p in ps]
    grads = torch.autograd.grad(loss, flat, allow_unused=True)
    by_group, k = {}, 0
    for name, ps in groups.items():
        by_group[name] = grads[k:k + len(ps)]
        k += len(ps)
    for name in ("backbone", "scan"):
        assert all(g is not None for g in by_group[name]), name
        assert sum(float(g.abs().sum()) for g in by_group[name]) > 0.0, name
    for name in ("priv", "hist", "critic", "std"):
        assert all(g is None for g in by_group[name]), name
    # the hook lives only for the BC call: PPO's own training-path forward still trains the priv encoder
    assert not actor.priv_encoder._forward_hooks
    mu = actor(alg.bc_obs[:8], False)
    g_priv = torch.autograd.grad(mu.square().sum(), groups["priv"], allow_unused=True)
    assert all(g is not None and float(g.abs().sum()) > 0 for g in g_priv)
    # ... and a failing forward removes it too
    with pytest.raises(Exception):
        alg._bc_policy_mean(alg.bc_obs[:, :3])
    assert not actor.priv_encoder._forward_hooks


# ===================================================================== RNG, logging, update
def test_bc_never_touches_the_global_rng(tmp_path, ckpt):
    banks = _banks(tmp_path)
    alg = _z2c(tmp_path, ckpt, banks)
    torch.manual_seed(77)
    before = torch.get_rng_state().clone()
    alg._bc_loss()
    assert torch.equal(torch.get_rng_state(), before)
    # a whole update with BC leaves the global RNG exactly where the Z2 update does (the minibatch permutation
    # draws it; the bank draw does not)
    ref = _z2(tmp_path, ckpt)
    for it in (0, 1):
        SC._update(ref, iteration=it)
        rng_ref = torch.get_rng_state().clone()
        SC._update(alg, iteration=it)
        assert torch.equal(torch.get_rng_state(), rng_ref)
    # BC changed the training (it is live), deterministically (a twin replays it bit for bit)
    twin = _z2c(tmp_path, ckpt, banks)
    twin._bc_loss()                                                     # the same one draw ``alg`` took above
    twin_ref = _z2(tmp_path, ckpt)
    for it in (0, 1):
        SC._update(twin, iteration=it)
        SC._update(twin_ref, iteration=it)
    PH._assert_same(PH._snapshot(alg), PH._snapshot(twin))
    a = PH._snapshot(alg)
    b = PH._snapshot(twin_ref)
    assert any(not torch.equal(a[k], b[k]) for k in a if k.startswith("policy.actor.") and torch.is_tensor(a[k]))


def test_loss_bc_and_bc_mb0(tmp_path, ckpt, monkeypatch):
    banks = _banks(tmp_path)
    alg = _z2c(tmp_path, ckpt, banks)
    before = copy.deepcopy(alg.policy.actor)
    state = alg.bc_generator.get_state()
    seen: list[float] = []
    real = alg._bc_loss

    def _spy():
        out = real()
        seen.append(float(out))
        return out

    monkeypatch.setattr(alg, "_bc_loss", _spy)
    priors: list[float] = []
    real_statue = alg._statue_losses

    def _statue_spy(*a):
        out = real_statue(*a)
        priors.append(float(out["prior"]))
        return out

    monkeypatch.setattr(alg, "_statue_losses", _statue_spy)
    ld = SC._update(alg, iteration=3)
    n_mb = alg.num_learning_epochs * alg.num_mini_batches
    assert len(seen) == n_mb
    assert ld["bc_mb0"] == seen[0]
    assert ld["bc"] == pytest.approx(sum(seen) / n_mb, rel=1e-12)
    assert "grad_norm_bc" not in ld                                     # off-cadence iteration
    # bc_mb0 is the update's FIRST minibatch, before any step: the pre-update actor on the first draw
    g = torch.Generator()
    g.set_state(state.clone())
    idx = torch.randint(alg.bc_obs.shape[0], (alg.bc_batch_size,), generator=g)
    with torch.no_grad():
        mu0 = before(alg.bc_obs[idx], False)
        want = (0.5 * ((mu0 - alg.bc_target[idx]) / alg.anchor_sigma) ** 2).sum(-1).mean()
    assert ld["bc_mb0"] == pytest.approx(float(want), rel=1e-6)
    # the statue terms are logged as before, beside the BC ones -- plus the prior's own first-minibatch value
    assert {"anchor", "stand_prior", "anchor_mb0", "restance_nonzero", "stand_prior_mb0"} <= set(ld)
    assert ld["stand_prior_mb0"] == priors[0] and len(priors) == n_mb


def test_stand_prior_mb0_only_beside_bc(tmp_path, ckpt):
    assert "stand_prior_mb0" not in SC._update(_z2(tmp_path, ckpt), iteration=3)


def test_grad_norm_split_includes_bc_and_moves_nothing(tmp_path, ckpt):
    banks = _banks(tmp_path)
    a, b = _z2c(tmp_path, ckpt, banks), _z2c(tmp_path, ckpt, banks)
    la, lb = SC._update(a, iteration=50), SC._update(b, iteration=51)
    assert {"grad_norm_bc", "grad_norm_statue", "grad_norm_anchor", "grad_norm_rest"} <= set(la)
    assert la["grad_norm_bc"] > 0.0
    assert not {"grad_norm_bc", "grad_norm_rest"} & set(lb)
    PH._assert_same(PH._snapshot(a), PH._snapshot(b))                   # logging only (autograd.grad)
    assert {k: v for k, v in la.items() if not k.startswith("grad_norm")} == lb


def test_grad_norm_bc_is_the_weighted_terms_norm_and_rest_excludes_it(tmp_path, ckpt, monkeypatch):
    alg = _z2c(tmp_path, ckpt, _banks(tmp_path), coef=0.5)
    calls = []
    real = alg._actor_grad_norm

    def _spy(term):
        calls.append(term.detach().clone())
        return real(term)

    monkeypatch.setattr(alg, "_actor_grad_norm", _spy)
    bc_vals = []
    real_bc = alg._bc_loss

    def _bc_spy():
        out = real_bc()
        bc_vals.append(out.detach().clone())
        return out

    monkeypatch.setattr(alg, "_bc_loss", _bc_spy)
    SC._update(alg, iteration=0)
    # the split runs on the first minibatch: statue, anchor, bc, rest -- in that order
    assert len(calls) == 4
    assert torch.allclose(calls[2], 0.5 * bc_vals[0])


def test_resume_state_carries_the_generator_only_when_armed(tmp_path, ckpt):
    banks = _banks(tmp_path)
    alg = _z2c(tmp_path, ckpt, banks)
    SC._update(alg, iteration=0)
    state = alg.resume_state()
    assert torch.equal(state["bc_generator"], alg.bc_generator.get_state())
    nxt = alg._bc_loss().detach()
    fresh = _z2c(tmp_path, ckpt, banks)
    fresh.load_resume_state(state)
    assert torch.equal(fresh.bc_generator.get_state(), state["bc_generator"])
    fresh.policy.load_state_dict(alg.policy.state_dict())
    assert torch.equal(fresh._bc_loss().detach(), nxt)                  # the draw sequence continues
    # a fork from a checkpoint without BC keeps the freshly seeded generator
    fork = _z2c(tmp_path, ckpt, banks)
    seeded = fork.bc_generator.get_state().clone()
    fork.load_resume_state(_z2(tmp_path, ckpt).resume_state())
    assert torch.equal(fork.bc_generator.get_state(), seeded)
    assert torch.equal(seeded, torch.Generator().manual_seed(5).get_state())


# ===================================================================== refusals
def test_bank_validation_refusals(tmp_path, ckpt):
    good = _bank_obs(10, seed=1)
    bad_bit = good.copy()
    bad_bit[3, STAND_COL] = 0.0
    half = good.copy()
    half[:, STAND_COL] = 0.5
    nan = good.copy()
    nan[2, 40] = np.nan
    cases = [(_write_bank(tmp_path / "w.npz", _bank_obs(10, seed=1, width=W - 1)), "1149"),
             (_write_bank(tmp_path / "w2.npz", _bank_obs(10, seed=1, width=W + 89)), "1149"),
             (_write_bank(tmp_path / "e.npz", good[:0]), "no rows"),
             (_write_bank(tmp_path / "b.npz", bad_bit), "stand bit"),
             (_write_bank(tmp_path / "h.npz", half), "10 / 10 rows"),
             (_write_bank(tmp_path / "n.npz", nan), "non-finite")]
    np.savez(tmp_path / "k.npz", target=good)
    cases.append((tmp_path / "k.npz", "no 'obs'"))
    np.savez(tmp_path / "d.npz", obs=good[0])
    cases.append((tmp_path / "d.npz", "1149"))
    for path, match in cases:
        with pytest.raises(ValueError, match=match):
            _z2c(tmp_path, ckpt, [_write_bank(tmp_path / "ok.npz", good), path])
    with pytest.raises(FileNotFoundError):
        _z2c(tmp_path, ckpt, [tmp_path / "missing.npz"])


def test_constructor_refusals(tmp_path, ckpt):
    bank = [str(b) for b in _banks(tmp_path, sizes=(8,))]
    z1_kw = {}
    for kw, base, match in (
            ({"bc_coef": -1.0, "bc_bank": bank}, _z2_kw(), ">= 0"),
            ({"bc_coef": float("nan"), "bc_bank": bank}, _z2_kw(), ">= 0"),
            ({"bc_coef": 1.0, "bc_bank": bank}, z1_kw, "stand_prior_target='restance'"),
            ({"bc_coef": 1.0}, _z2_kw(), "needs bc_bank"),
            ({"bc_coef": 1.0, "bc_bank": []}, _z2_kw(), "needs bc_bank"),
            ({"bc_coef": 1.0, "bc_bank": bank, "bc_batch_size": 0}, _z2_kw(), "bc_batch_size"),
            ({"bc_coef": 1.0, "bc_bank": bank, "bc_batch_size": 1.5}, _z2_kw(), "bc_batch_size"),
            ({"bc_coef": 1.0, "bc_bank": bank, "bc_batch_size": True}, _z2_kw(), "bc_batch_size"),
            ({"bc_coef": 1.0, "bc_bank": bank, "bc_seed": 0.5}, _z2_kw(), "bc_seed"),
            ({"bc_bank": bank}, _z2_kw(), "bc_coef is 0")):
        with pytest.raises(ValueError, match=match):
            SC._statue_alg(tmp_path, seed=0, ckpt=ckpt, **base, **kw)
    # BC needs sigma_A, i.e. the statue checkpoint (the statue rule refuses first, naming anchor_ckpt)
    with pytest.raises(ValueError, match="anchor_ckpt"):
        PH._alg(0, **SC._soft_kw(), stand_prior_coef=1.0, stand_mask_col=STAND_COL, **_z2_kw(), bc_coef=1.0,
                bc_bank=bank)
    # a single path string is accepted as a one-bank list
    alg = _z2(tmp_path, ckpt, bc_coef=1.0, bc_bank=bank[0], bc_batch_size=4)
    assert alg.bc_bank == bank and alg.bc_obs.shape[0] == 8
    # the runner's alg-cfg filter accepts the four keys (and still refuses a typo)
    cfg = {"bc_coef": 1.0, "bc_bank": bank, "bc_batch_size": 2048, "bc_seed": 0}
    assert PH.runner_mod.filter_ppo_alg_cfg(PH.PPO, dict(cfg)) == cfg
    with pytest.raises(ValueError, match="bc_cof"):
        PH.runner_mod.filter_ppo_alg_cfg(PH.PPO, {"bc_cof": 1.0})


# ===================================================================== knobs
def _agent():
    return SC._agent()


def _z2c_env(tmp_path, **over) -> dict:
    ckpt = tmp_path / "graft.pt"
    ckpt.write_bytes(b"x")
    banks = []
    for name in ("es1.npz", "field.npz"):
        (tmp_path / name).write_bytes(b"x")
        banks.append(str(tmp_path / name))
    env = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1.0", "KRABBY_ANCHOR_CKPT": str(ckpt),
           "KRABBY_STAND_PRIOR_W": "1.0", "KRABBY_STAND_PRIOR_TARGET": "restance",
           "KRABBY_RESTANCE": RST.format_restance(P_Z2C), "KRABBY_BC_W": "1.0", "KRABBY_BC_BANK": ",".join(banks)}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def test_bc_parsers():
    p_w, p_bank, p_batch = (mk.MOTION_KNOB_PARSERS[k] for k in BC_KNOBS)
    assert p_w is mk.parse_bc_w and p_bank is mk.parse_bc_bank and p_batch is mk.parse_bc_batch
    for off in (None, "", " ", "0"):
        assert p_w(off) is None and p_bank(off) is None
    assert p_batch(None) is None and p_batch(" ") is None
    assert p_w("1.0") == 1.0 and p_w("0.25") == 0.25
    for bad in ("-0.1", "nan", "inf", "x"):
        with pytest.raises(ValueError, match="KRABBY_BC_W"):
            p_w(bad)
    assert p_bank(" a.npz , b/c.npz ") == ("a.npz", "b/c.npz") and p_bank("a.npz") == ("a.npz",)
    for bad in ("a.npz,", ",a.npz", "a.npz,,b.npz", "a.npz,a.npz"):
        with pytest.raises(ValueError, match="KRABBY_BC_BANK"):
            p_bank(bad)
    assert p_batch("2048") == 2048 and p_batch(" 64 ") == 64
    for bad in ("0", "-5", "1.5", "x", "2e3"):
        with pytest.raises(ValueError, match="KRABBY_BC_BATCH"):
            p_batch(bad)


def test_bc_knobs_reach_the_algorithm_cfg(tmp_path, capsys):
    env = _z2c_env(tmp_path)
    agent = _agent()
    mk.apply_exploration_knobs(agent, env)
    alg = vars(agent.algorithm)
    assert alg["bc_coef"] == 1.0 and alg["bc_batch_size"] == mk.BC_BATCH_DEFAULT
    assert alg["bc_bank"] == env["KRABBY_BC_BANK"].split(",")
    assert alg["stand_prior_target"] == "restance" and alg["restance"] == P_Z2C.to_dict()
    assert alg["stand_mask_col"] == 15
    assert "bc_seed" not in alg                                          # the constructor default (0)
    out = capsys.readouterr().out
    assert f"[KNOB] KRABBY_BC_W=1.0 consumed" in out
    assert f"[KNOB] KRABBY_BC_BANK={env['KRABBY_BC_BANK']} consumed" in out
    assert "KRABBY_BC_BATCH" not in out                                  # blank: the default, no consumed line
    agent = _agent()
    mk.apply_exploration_knobs(agent, {**env, "KRABBY_BC_BATCH": "512"})
    assert agent.algorithm.bc_batch_size == 512
    assert "[KNOB] KRABBY_BC_BATCH=512 consumed" in capsys.readouterr().out
    # repo-relative bank paths resolve against the repo root (like KRABBY_ANCHOR_CKPT)
    rel = "parkour/outputs/__z2c_test_bank.npz"
    assert mk.resolve_bc_bank([rel, "/abs/x.npz"]) == [str(REPO / rel), "/abs/x.npz"]
    # knobs off: nothing written (the Z2 cfg exactly)
    z2_env = {k: v for k, v in env.items() if k not in BC_KNOBS}
    a, b = _agent(), _agent()
    mk.apply_exploration_knobs(a, z2_env)
    mk.apply_exploration_knobs(b, {**z2_env, "KRABBY_BC_W": "0", "KRABBY_BC_BANK": "", "KRABBY_BC_BATCH": ""})
    assert SC._dump(a) == SC._dump(b) and not any(k.startswith("bc_") for k in vars(a.algorithm))


def test_bc_knob_refusals(tmp_path):
    cases = [
        (dict(KRABBY_BC_BANK=None), "needs KRABBY_BC_BANK"),
        (dict(KRABBY_BC_W="0"), "KRABBY_BC_BANK is set but KRABBY_BC_W is off"),
        (dict(KRABBY_BC_W=None), "KRABBY_BC_BANK is set but KRABBY_BC_W is off"),
        (dict(KRABBY_BC_W=None, KRABBY_BC_BANK=None, KRABBY_BC_BATCH="64"), "KRABBY_BC_BATCH is set but"),
        (dict(KRABBY_STAND_PRIOR_TARGET=None, KRABBY_RESTANCE=None), "KRABBY_STAND_PRIOR_TARGET=restance"),
        (dict(KRABBY_RESTANCE="0", KRABBY_STAND_PRIOR_TARGET="0"), "KRABBY_STAND_PRIOR_TARGET=restance"),
        (dict(KRABBY_BC_W="-1"), ">= 0"),
        (dict(KRABBY_BC_BATCH="0"), "positive integer"),
        (dict(KRABBY_BC_BANK="x.npz,,y.npz"), "no empty entry"),
    ]
    for over, match in cases:
        env = _z2c_env(tmp_path, **over)
        agent = _agent()
        before = SC._dump(agent)
        with pytest.raises(ValueError, match=match):
            mk.apply_exploration_knobs(agent, env)
        assert SC._dump(agent) == before, over                          # refused before anything is written
    # a missing bank file is refused at application (the dry run's composition check does not open files)
    env = _z2c_env(tmp_path)
    env["KRABBY_BC_BANK"] += f",{tmp_path / 'missing.npz'}"
    mk.check_bc_composition(env)
    with pytest.raises(ValueError, match="not a file"):
        mk.apply_exploration_knobs(_agent(), env)
    # the pure cross rules on their own (the ladder's dry run calls these): each prerequisite is load-bearing
    base = _z2c_env(tmp_path)
    mk.check_bc_composition(base)
    mk.check_bc_composition({})
    for drop, match in (("KRABBY_OMNI_STAND_OBS", "KRABBY_OMNI_STAND_OBS=clock"),
                        ("KRABBY_STAND_PRIOR_TARGET", "KRABBY_STAND_PRIOR_TARGET=restance"),
                        ("KRABBY_RESTANCE", "needs KRABBY_RESTANCE"), ("KRABBY_ANCHOR_CKPT", "needs KRABBY_ANCHOR_CKPT")):
        with pytest.raises(ValueError, match=match):
            mk.check_bc_composition({k: v for k, v in base.items() if k != drop})
    with pytest.raises(ValueError, match="KRABBY_OMNI_STAND_OBS=clock"):
        mk.check_bc_composition({**base, "KRABBY_OMNI_STAND_OBS": "1"})


# ===================================================================== the ladder
@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES_PY, "run_candidates__p2r_z2c_test")


def test_bc_knobs_are_registered_with_the_ladder(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for var in BC_KNOBS:
        assert var in whitelist and var in parsers and var in rc.OVERLAY_DUMP_CHECKS, var
        assert parsers[var] is rc.knobs()._mk.MOTION_KNOB_PARSERS[var], var
    assert set(BC_KNOBS) <= rc.AGENT_DUMP_KEYS
    assert rc.AGENT_DUMP_KEYS == set(mk.AGENT_KNOBS) | {"KRABBY_OMNI_MIRROR_IMAGES"}
    # NOTE(p2r Z2d, 2026-10-11): + the bank rows' stand-row mask (pinned by test_omni_p2r_z2d_mask.py)
    assert set(rc.BC_ALG_FIELDS) == {"bc_coef", "bc_bank", "bc_batch_size", "bc_seed", "bc_stand_mask"}
    reasons = []
    for var, value in (("KRABBY_BC_W", "1.0"), ("KRABBY_BC_BANK", "a.npz,b.npz"), ("KRABBY_BC_BATCH", "2048")):
        rc._check_knob(var, value, "t", whitelist, parsers, reasons, set())
    assert not reasons, reasons
    for var, value in (("KRABBY_BC_W", "-1"), ("KRABBY_BC_BANK", "a.npz,"), ("KRABBY_BC_BATCH", "0")):
        rc._check_knob(var, value, "t", whitelist, parsers, reasons, set())
    assert len(reasons) == 3, reasons


def _agent_yaml(**alg) -> str:
    base = {"seed": 18, "algorithm": {"class_name": "PPOWithExtractor", "learning_rate": 0.0003,
                                      "stand_prior_coef": 1.0, "stand_mask_col": 15, **alg},
            "policy": {"init_noise_std": 0.8}}
    return yaml.dump(base, default_flow_style=False, sort_keys=False)       # isaaclab's dump_yaml spelling


def test_dump_checks_on_synthetic_dumps(rc, tmp_path):
    w, bank, batch = (rc.OVERLAY_DUMP_CHECKS[k] for k in BC_KNOBS)
    paths = [str(tmp_path / "a.npz"), str(tmp_path / "b.npz")]
    on = _agent_yaml(bc_coef=1.0, bc_bank=paths, bc_batch_size=2048)
    assert f"  bc_bank:\n  - {paths[0]}\n  - {paths[1]}\n" in on               # the format the reader expects
    assert rc._bc_bank_dump(on) == paths
    off = _agent_yaml()
    knob_bank = ",".join(paths)
    assert w(on, "1.0") and not w(on, "0.5") and not w(on, "0") and w(off, "") and w(off, "0") and not w(off, "1")
    assert bank(on, knob_bank) and not bank(on, paths[0]) and not bank(on, f"{paths[1]},{paths[0]}")
    assert bank(off, "") and not bank(off, knob_bank) and not bank(on, "")
    assert batch(on, "") and batch(on, "2048") and not batch(on, "512") and batch(off, "") and not batch(off, "64")
    on512 = _agent_yaml(bc_coef=1.0, bc_bank=paths, bc_batch_size=512)
    assert batch(on512, "512") and not batch(on512, "")                       # blank means the default
    # the weight's check needs the bank and the batch beside it
    assert not w(_agent_yaml(bc_coef=1.0, bc_batch_size=2048), "1.0")
    assert not w(_agent_yaml(bc_coef=1.0, bc_bank=paths), "1.0")
    assert not w(_agent_yaml(bc_bank=paths), "0")                              # a stray bank with the weight off
    assert rc._bc_bank_dump(_agent_yaml(bc_bank=None)) is None
    # repo-relative knob values compare against the dumped absolute (resolved) paths
    rel = "parkour/outputs/__z2c_bank.npz"
    assert bank(_agent_yaml(bc_coef=1.0, bc_bank=[str(REPO / rel)], bc_batch_size=2048), rel)


def test_restance_dump_check_accepts_a_pre_eps_step_dump(rc):
    """NOTE(p2r Z2c): Z1 / Z2 dumps carry no eps_step; their R had none (= eps_c), so they still audit equal --
    but a knob with a dead band does not match them."""
    knob = "lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=0.3,trim=0.03,eps_c=0.1,eps_air=0.05,tilt_max=0.1"
    old = {k: v for k, v in RST.parse_restance(knob).to_dict().items() if k != "eps_step"}
    txt = _agent_yaml(stand_prior_target="restance", restance=old)
    assert rc.OVERLAY_DUMP_CHECKS["KRABBY_RESTANCE"](txt, knob)
    assert not rc.OVERLAY_DUMP_CHECKS["KRABBY_RESTANCE"](txt, knob + ",eps_step=0.3")
    new = _agent_yaml(stand_prior_target="restance", restance=RST.parse_restance(knob + ",eps_step=0.3").to_dict())
    assert rc.OVERLAY_DUMP_CHECKS["KRABBY_RESTANCE"](new, knob + ",eps_step=0.3")
    assert not rc.OVERLAY_DUMP_CHECKS["KRABBY_RESTANCE"](new, knob)


def _armed_log(env: dict, rows: int = 450000, batch: int = 2048) -> str:
    log = "".join(f"[KNOB] {k}={v} consumed\n" for k, v in env.items()
                  if k != "KRABBY_OMNI_STAND_OBS" and str(v).strip())
    log += (f"[PPO] construction: anchor_coef=1.0, stand_prior_coef=1.0, stand_prior_target=restance, "
            f"restance={env['KRABBY_RESTANCE']}, bc_coef={float(env['KRABBY_BC_W'])}, bc_bank_rows={rows}, "
            f"bc_batch_size={batch}, bc_seed=0, learning_rate=0.0003, desired_kl=0.01\n"
            "   Mean anchor: 0.0000\n   Mean stand_prior: 2.3000\n   Mean restance_nonzero: 1.0000\n"
            "   Mean bc: 3.1000\n   Mean bc_mb0: 8.9000\n   Mean anchor_mb0: 0.0000\n")
    return log


def test_bc_log_checks(rc, tmp_path):
    env = _z2c_env(tmp_path)
    log_path = tmp_path / "train.log"
    log_path.write_text(_armed_log(env))
    rec = {"log": str(log_path)}
    got = rc.bc_log_checks(rec, env)
    assert got and all(ok for _n, ok in got), [n for n, ok in got if not ok]
    names = [n for n, _ok in got]
    for want in ("log: [KNOB] KRABBY_BC_W consumed", "log: [KNOB] KRABBY_BC_BANK consumed",
                 "log: no BC field dropped by the runner", "log: [PPO] construction names bc_coef=1.0",
                 "log: [PPO] construction names bc_bank_rows > 0", "log: [PPO] construction names bc_batch_size=2048",
                 "log: Mean bc printed (the BC term ran)", "log: Mean bc_mb0 printed (the BC term ran)"):
        assert want in names, want
    assert "log: [KNOB] KRABBY_BC_BATCH consumed" not in names              # blank batch: no consumed line expected
    # each line is load-bearing
    full = _armed_log(env)
    for drop, name in (("   Mean bc: 3.1000\n", "log: Mean bc printed (the BC term ran)"),
                       ("   Mean bc_mb0: 8.9000\n", "log: Mean bc_mb0 printed (the BC term ran)"),
                       ("bc_coef=1.0, ", "log: [PPO] construction names bc_coef=1.0"),
                       ("bc_bank_rows=450000, ", "log: [PPO] construction names bc_bank_rows > 0"),
                       ("bc_batch_size=2048, ", "log: [PPO] construction names bc_batch_size=2048"),
                       (f"[KNOB] KRABBY_BC_BANK={env['KRABBY_BC_BANK']} consumed\n",
                        "log: [KNOB] KRABBY_BC_BANK consumed")):
        log_path.write_text(full.replace(drop, ""))
        assert dict(rc.bc_log_checks(rec, env))[name] is False, name
    log_path.write_text(full.replace("bc_bank_rows=450000", "bc_bank_rows=0"))
    assert dict(rc.bc_log_checks(rec, env))["log: [PPO] construction names bc_bank_rows > 0"] is False
    log_path.write_text(full + "WARNING Ignoring unsupported PPO config keys for PPOWithExtractor: bc_bank\n")
    assert dict(rc.bc_log_checks(rec, env))["log: no BC field dropped by the runner"] is False
    # an explicit batch is audited against its own value and its consumed line
    env512 = {**env, "KRABBY_BC_BATCH": "512"}
    log_path.write_text(_armed_log(env512, batch=512))
    got = dict(rc.bc_log_checks(rec, env512))
    assert got["log: [PPO] construction names bc_batch_size=512"] and got["log: [KNOB] KRABBY_BC_BATCH consumed"]
    # off / absent: no checks at all; armed without a readable log: one failing check
    assert rc.bc_log_checks(rec, {k: v for k, v in env.items() if k not in BC_KNOBS}) == []
    assert rc.bc_log_checks(rec, {**env, "KRABBY_BC_W": "0", "KRABBY_BC_BANK": ""}) == []
    assert rc.bc_log_checks({}, env) == [(f"training log readable for the BC wiring audit (None)", False)]


def test_overlay_checks_on_an_armed_real_dump(rc, tmp_path):
    env_txt, agent_txt = SC._s1_dumps()
    env = _z2c_env(tmp_path, KRABBY_ANCHOR_CKPT=str(SC.GRAFT))
    env_on, agent_on = SC._armed_dumps(env_txt, agent_txt, str(SC.GRAFT))
    params = RST.parse_restance(env["KRABBY_RESTANCE"])
    block = yaml.dump({"algorithm": {"stand_prior_target": "restance", "restance": params.to_dict(),
                                     "bc_coef": 1.0, "bc_bank": mk.resolve_bc_bank(env["KRABBY_BC_BANK"].split(",")),
                                     "bc_batch_size": 2048}},
                      default_flow_style=False, sort_keys=False).split("\n", 1)[1]
    agent_on = agent_on.replace("  stand_mask_col: 15\n", "  stand_mask_col: 15\n" + block, 1)
    rec = SC._run_dir(tmp_path / "on", env_on, agent_on, _armed_log(env))
    got = rc.overlay_checks(rec, env)
    assert got and all(ok for _n, ok in got), [n for n, ok in got if not ok]
    names = dict(got)
    for key in BC_KNOBS[:2]:
        assert names.get(f"overlay {key}={env[key]} in params/agent.yaml") is True, key
    assert names.get("log: Mean bc_mb0 printed (the BC term ran)") is True
    # the knob-off S1 dump fails the BC claims and passes the off values
    checks = rc.OVERLAY_DUMP_CHECKS
    for key in BC_KNOBS[:2]:
        assert not checks[key](agent_txt, env[key]) and checks[key](agent_txt, ""), key
    assert checks["KRABBY_BC_BATCH"](agent_txt, "")


def test_bc_composition_at_dry_run(rc, tmp_path):
    base = _z2c_env(tmp_path)
    reasons = []
    rc._check_bc_composition(base, "w", reasons)
    rc._check_bc_composition({}, "w", reasons)
    rc._check_bc_composition({k: v for k, v in base.items() if k not in BC_KNOBS}, "w", reasons)
    assert not reasons
    for drop, match in (("KRABBY_BC_BANK", "needs KRABBY_BC_BANK"), ("KRABBY_BC_W", "would train nothing"),
                        ("KRABBY_STAND_PRIOR_TARGET", "KRABBY_STAND_PRIOR_TARGET=restance"),
                        ("KRABBY_ANCHOR_CKPT", "needs KRABBY_ANCHOR_CKPT")):
        reasons = []
        rc._check_bc_composition({k: v for k, v in base.items() if k != drop}, "w", reasons)
        assert len(reasons) == 1 and reasons[0].startswith("w: ") and match in reasons[0], (drop, reasons)
    # a malformed value is left to _check_knob (one reason, not two)
    reasons = []
    rc._check_bc_composition({**base, "KRABBY_BC_W": "-1"}, "w", reasons)
    assert not reasons
