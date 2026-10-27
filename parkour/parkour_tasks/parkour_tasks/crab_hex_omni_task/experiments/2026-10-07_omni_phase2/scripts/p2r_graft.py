#!/usr/bin/env python3
"""Omni phase 2, stage p2r -- the graft tool (P2R_PLAN.md step 4 / P2R_STATUE_PLAN.md step 1 "graft tool"). CPU only,
no Isaac, no CUDA: it loads one checkpoint, zeroes one observation column's wiring into the actor backbone, the
critic and the estimator (+ their Adam moments), and writes the grafted checkpoint plus ``graft.json`` (the SHARED
CONTRACT): ``{source_path, source_sha256, graft_sha256, zeroed: [...], adam_zeroed: [...], identity_cpu: bool,
identity_gpu_stamps: [...]}``.

WHY A HAND-REPLICA for ``--identity-check`` (not an import of ``scripts/rsl_rl/modules``): see ``p2r_drift.py``'s
module docstring -- importing those classes pulls in ``isaaclab`` -> ``pxr`` (USD), CPU-unimportable outside Kit.
This module reuses :mod:`p2r_drift` (same directory, no heavy deps) for the actor's non-history ("anchor path") mu
and extends it with the three pieces :mod:`p2r_drift` never needed: the actor's HISTORY path (``StateHistoryEncoder``,
replicated from ``scripts/rsl_rl/modules/feature_extractors/state_encoder.py``), the critic MLP, and the estimator
MLP.

THE GRAFT (P2R_PLAN.md step 4 / P2R_STATUE_PLAN.md step 1): column 15 of the observation is the grafted stand bit
(``KRABBY_OMNI_STAND_OBS=clock``). Every S1 checkpoint was trained with that column always 0 (``STAND_OBS`` off), so
every weight READING that column is untrained noise on it, not signal -- zeroing it is lossless on S1's own
distribution and makes the column's value (0 at S1, 0 or 1 at the graft) provably inert until a later training step
teaches the network to read it on purpose. Three first-layer weights read observation columns directly:
  * ``model_state_dict["actor.actor_backbone.0.weight"]`` (shape (512, 136) on the X08 S1 head) -- its input is
    ``cat([obs[:, :num_prop], scan_latent, obs_priv_explicit, latent])`` (``Actor.forward``), so column 15 of ITS
    weight IS observation column 15 (column 15 < num_prop = 75, inside the raw prop block, before any encoder);
  * ``model_state_dict["critic.0.weight"]`` (shape (512, 1149)) -- the critic reads its own (wider, privileged)
    observation directly, column 15 at the same raw-prop position (CHARTER 12 "the critic obs group" / "current step
    only" check: P2R_PLAN.md "Left for implementation");
  * ``estimator_state_dict["estimator.0.weight"]`` (shape (128, 75)) -- the estimator's input IS the raw prop block
    (``DefaultEstimator.forward``, ``scripts/rsl_rl/modules/feature_extractors/estimator.py``), column 15 again at
    the same position.
The HISTORY encoder, the scan encoder and the priv encoder never read column 15 at all (their inputs are the scan /
priv_latent blocks, and the trailing HISTORY block of a real actor obs copies PAST prop frames, each with its own
column 15 held at 0 by construction -- "never the history buffer", P2R_STATUE_PLAN.md step 1) -- so they are left
untouched, and the graft is exactly these three weight matrices, nothing else.

ADAM MOMENTS: a parameter that never read a live signal on column 15 also never got a column-15 GRADIENT, so its
Adam ``exp_avg`` / ``exp_avg_sq`` at that column should already read ~0 -- :func:`zero_adam_column` ASSERTS that
(< ``ADAM_PRE_ZERO_EPS``) before zeroing it to an exact 0.0, so a graft never silently papers over a column that
WAS being trained (a sign the column-15 premise, or the state-dict/optimizer index mapping below, is wrong).
Mapping (P2R_PLAN.md "Left for implementation" / P2R_STATUE_PLAN.md step 1): PyTorch's ``Adam.state`` is keyed by
the OPTIMIZER's own 0-based parameter index, which for a single param_group equals the index of that parameter in
``model.state_dict().keys()`` IN ITERATION ORDER (``nn.Module`` registers parameters in definition order, and a
state_dict is an ``OrderedDict`` over exactly that same order) -- so ``list(model_state_dict).index(name)`` IS the
``optimizer_state_dict["state"]`` key for ``name`` (confirmed on the X08 head: index 0 is ``std_raw``, and
``optimizer_state_dict["state"][0]["exp_avg"].shape == (18,) == std_raw.shape``). The estimator has its OWN
optimizer (``alg_state["estimator_optimizer"]``, P2R_PLAN.md step 4's "estimator's Adam state") with its own
0-based indexing over ``estimator_state_dict.keys()`` (index 0 is ``estimator.0.weight`` itself). A parameter that
never received a gradient (the history encoder, since every S1 run trains with ``hist_encoding=False``) has NO
entry in ``state`` at all -- :func:`adam_index` raises on a missing entry rather than silently skipping, so a wrong
index mapping is a loud ``KeyError``, never a silent no-op.

EVERYTHING ELSE BITWISE IDENTICAL: :func:`build_graft` never reads or writes any tensor outside the three weights
above and their two Adam moments; :func:`assert_rest_identical` is a belt-and-suspenders re-check (CPU, before the
file is written) that walks every other leaf tensor of the loaded checkpoint and the about-to-be-written one and
``torch.equal``'s them.

CLI:
    p2r_graft.py --source CKPT.pt --out OUT.pt [--graft-json OUT.json] [--expect-source-sha256 HEX]
        builds the graft; writes OUT.pt and (OUT.json, default OUT.pt with '.json' instead of '.pt').
    p2r_graft.py --identity-check --source CKPT.pt --graft OUT.pt --bank BANK.npz [--agent-yaml PATH]
                 [--graft-json OUT.json]
        the CPU identity check (P2R_STATUE_PLAN.md step 1): asserts torch.equal outputs of S1 (column 15 = 0) vs the
        graft (column 15 in {0, 1}) on the actor (both the hist_encoding=False / anchor path and the history path),
        the estimator and the critic, using rows from a step-1 probe bank. On success (and only then) merges
        ``identity_cpu: true`` into ``--graft-json`` (default alongside ``--graft``).
    p2r_graft.py --self-test
        a tiny synthetic checkpoint (every architecture piece, at toy widths) exercises build + zero + identity
        end-to-end, with no real checkpoint or bank needed. Exit 0 / 1.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import p2r_drift as drift  # noqa: E402  (same directory, CPU-only, no isaaclab -- see module docstring)

#: the grafted observation column (the stand bit, ``KRABBY_OMNI_STAND_OBS=clock`` -- SHARED CONTRACT)
COL = 15
#: the three first-layer weights the graft zeroes column 15 of, and which checkpoint dict each lives in
#: ("model" -> ``d["model_state_dict"]``, "estimator" -> ``d["estimator_state_dict"]``)
GRAFT_TARGETS = (("model", "actor.actor_backbone.0.weight"), ("model", "critic.0.weight"),
                 ("estimator", "estimator.0.weight"))
#: a parameter's Adam moments must already be this small at column 15 before the graft zeroes them (the premise:
#: STAND_OBS was off for every S1 run, so column 15 never carried a trained signal -- P2R_PLAN.md step 4)
ADAM_PRE_ZERO_EPS = 1e-30
#: the X08 S1 head's sha256 (CHARTER 12 / candidates v12's fork source), the graft tool's default expectation
DEFAULT_SOURCE_SHA256 = "968cc550989919cdc3fa36ae047379febabee5659cde0ed4d099195acb13c23c"


# ================================================================================================ checkpoint surgery
def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adam_index(sd_keys: list, name: str, state: dict) -> int:
    """``list(sd_keys).index(name)`` IS the optimizer's state key for ``name`` (module docstring). Raises
    ``KeyError`` (never silently skips) if that index has no recorded Adam state -- a parameter that was never
    stepped (e.g. the history encoder, since S1 trains with ``hist_encoding=False``) is a construction error if
    THIS function is ever asked for it, since every graft target (the backbone / critic / estimator first layers)
    is on the hot forward path and must have been stepped many thousand times by iteration 19994."""
    idx = list(sd_keys).index(name)
    if idx not in state:
        raise KeyError(f"{name} (optimizer param index {idx}) has no Adam state -- never stepped?")
    return idx


def zero_weight_column(weight: torch.Tensor, col: int = COL) -> tuple[torch.Tensor, float]:
    """``(new_weight, pre_norm)`` -- a CLONE with column ``col`` zeroed; ``pre_norm`` is the L2 norm of that column
    before zeroing (reported in ``graft.json``, never hard-asserted: a provenance number, not a gate)."""
    if weight.ndim != 2 or weight.shape[1] <= col:
        raise ValueError(f"weight shape {tuple(weight.shape)} has no column {col}")
    pre_norm = float(weight[:, col].norm())
    out = weight.clone()
    out[:, col] = 0.0
    return out, pre_norm


def zero_adam_column(state_entry: dict, col: int = COL, eps: float = ADAM_PRE_ZERO_EPS) -> float:
    """Mutates ``state_entry["exp_avg"]`` / ``["exp_avg_sq"]`` in place, column ``col`` -> 0.0, after asserting both
    are already ``< eps`` there (the premise: column 15 never carried a trained signal). Returns the pre-zero max
    |value| over both moments at that column (reported). Raises ``AssertionError`` if the premise fails -- the
    whole point of asserting BEFORE zeroing (module docstring)."""
    ea, eas = state_entry["exp_avg"], state_entry["exp_avg_sq"]
    pre_max = float(max(ea[:, col].abs().max(), eas[:, col].abs().max()))
    assert pre_max < eps, (f"Adam moments at column {col} are {pre_max:.3g} (>= {eps:g}): column {col} was being "
                           "trained -- the 'never read, never gradient' premise does not hold for this checkpoint")
    ea[:, col] = 0.0
    eas[:, col] = 0.0
    return pre_max


def _leaf_tensors(obj, prefix=()) -> dict:
    """Every ``torch.Tensor`` leaf of a nested dict/list structure, keyed by its path (recursion over the whole
    checkpoint dict: ``model_state_dict``, ``estimator_state_dict``, ``optimizer_state_dict``, ``alg_state``, ...)."""
    out = {}
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_leaf_tensors(v, prefix + (k,)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_leaf_tensors(v, prefix + (i,)))
    return out


def assert_rest_identical(orig: dict, grafted: dict, exceptions: set) -> None:
    """Every leaf tensor of ``orig`` and ``grafted`` is ``torch.equal`` EXCEPT the paths in ``exceptions`` (the
    graft targets' weight tensors and their Adam-moment tensors) -- the module docstring's belt-and-suspenders
    re-check. Raises ``AssertionError`` naming the first unexpected difference; a path present in one but not the
    other is also a failure (fail closed: the graft must change nothing it did not mean to)."""
    a, b = _leaf_tensors(orig), _leaf_tensors(grafted)
    if set(a) != set(b):
        raise AssertionError(f"checkpoint structure changed: only in orig {set(a) - set(b)}, "
                             f"only in grafted {set(b) - set(a)}")
    for path, ta in a.items():
        if path in exceptions:
            continue
        tb = b[path]
        if not torch.equal(ta, tb):
            raise AssertionError(f"leaf {path} changed outside the graft targets (shape {tuple(ta.shape)})")


def build_graft(source: Path, out: Path, expect_source_sha256: str | None = DEFAULT_SOURCE_SHA256) -> dict:
    """Loads ``source``, zeroes :data:`GRAFT_TARGETS`' column :data:`COL` (weight + Adam moments), asserts every
    OTHER leaf tensor is unchanged, writes ``out``, and returns the ``graft.json`` payload (SHARED CONTRACT) --
    minus ``identity_cpu`` / ``identity_gpu_stamps``, which :func:`run_identity_check` / the GPU bridge fill in.
    Does not write ``graft.json`` itself (the CLI does, after this returns, so a failed write never leaves a stale
    JSON beside a good checkpoint or vice versa)."""
    source = Path(source)
    source_sha = sha256_of(source)
    if expect_source_sha256 and source_sha != expect_source_sha256:
        raise ValueError(f"{source} sha256 {source_sha} != expected {expect_source_sha256}")
    d = torch.load(str(source), map_location="cpu", weights_only=False)
    orig = copy.deepcopy(d)
    msd, esd = d["model_state_dict"], d["estimator_state_dict"]
    opt_state = d["optimizer_state_dict"]["state"]
    est_opt_state = d["alg_state"]["estimator_optimizer"]["state"]
    model_keys = list(msd.keys())
    est_keys = list(esd.keys())

    zeroed, adam_zeroed, real_exceptions = [], [], set()
    for which, name in GRAFT_TARGETS:
        sd, keys, opt = (msd, model_keys, opt_state) if which == "model" else (esd, est_keys, est_opt_state)
        dict_name = "model_state_dict" if which == "model" else "estimator_state_dict"
        new_w, pre_norm = zero_weight_column(sd[name])
        sd[name] = new_w
        real_exceptions.add((dict_name, name))
        zeroed.append({"dict": dict_name, "name": name, "column": COL, "pre_norm": pre_norm})
        idx = adam_index(keys, name, opt)
        pre_max = zero_adam_column(opt[idx])
        if which == "model":
            optimizer, path = "optimizer_state_dict", ("optimizer_state_dict", "state", idx)
        else:
            optimizer, path = "alg_state.estimator_optimizer", ("alg_state", "estimator_optimizer", "state", idx)
        real_exceptions.add(path + ("exp_avg",))
        real_exceptions.add(path + ("exp_avg_sq",))
        adam_zeroed.append({"optimizer": optimizer, "param_index": idx, "name": name, "column": COL,
                            "pre_max_abs": pre_max})
    assert_rest_identical(orig, d, real_exceptions)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, str(out))
    graft_sha = sha256_of(out)
    return {"source_path": str(source), "source_sha256": source_sha, "graft_sha256": graft_sha,
           "zeroed": zeroed, "adam_zeroed": adam_zeroed, "identity_cpu": False, "identity_gpu_stamps": []}


# ================================================================================================ identity-check replica
#: StateHistoryEncoder (scripts/rsl_rl/modules/feature_extractors/state_encoder.py), the conv stride per tsteps --
#: the ONE piece of its architecture not already implicit in the checkpoint's own weight shapes (kernel sizes and
#: channel counts ARE implicit: F.conv1d reads them off the weight tensor it is given).
_HIST_STRIDES = {10: (2, 1), 20: (2, 2), 50: (4, 1, 1)}


def history_latent(hist: torch.Tensor, sd: dict, num_hist: int) -> torch.Tensor:
    """(N, T, num_prop) -> (N, output_size) -- ``StateHistoryEncoder.forward``, replicated: a per-frame Linear+ELU
    projection, ``len(_HIST_STRIDES[num_hist])`` Conv1d+ELU layers over the time axis (kernel sizes read off the
    checkpoint's own ``conv_layers.<i>.weight`` shapes -- never hardcoded, so this is correct at any channel_size),
    Flatten, then a final Linear+ELU. Faithful to the real class for ``tsteps`` 10 / 20 / 50 (the only three it
    supports); any other ``num_hist`` raises, same as the real class."""
    if num_hist not in _HIST_STRIDES:
        raise ValueError(f"StateHistoryEncoder supports tsteps 10/20/50, got num_hist={num_hist}")
    n, t, p = hist.shape
    assert t == num_hist, f"history has T={t}, expected num_hist={num_hist}"
    x = F.elu(F.linear(hist.reshape(n * t, p), sd["actor.history_encoder.encoder.0.weight"],
                       sd["actor.history_encoder.encoder.0.bias"]))
    x = x.reshape(n, t, -1).permute(0, 2, 1)                              # (N, 3*channel_size, T)
    for i, stride in enumerate(_HIST_STRIDES[num_hist]):
        w, b = sd[f"actor.history_encoder.conv_layers.{2 * i}.weight"], sd[f"actor.history_encoder.conv_layers.{2 * i}.bias"]
        x = F.elu(F.conv1d(x, w, b, stride=stride))
    x = x.reshape(n, -1)
    return F.elu(F.linear(x, sd["actor.history_encoder.linear_output.0.weight"],
                          sd["actor.history_encoder.linear_output.0.bias"]))


def actor_mu_history_path(obs_hist: torch.Tensor, sd: dict, w: dict) -> torch.Tensor:
    """mu on the HISTORY path (``hist_encoding=True``): identical to :func:`p2r_drift.actor_mu` except the latent
    comes from :func:`history_latent` on the trailing ``num_prop * num_hist`` columns, not ``infer_priv_latent`` on
    the priv_latent block (``Actor.forward`` / ``infer_hist_latent``). ``obs_hist`` must carry prop/scan/priv_explicit
    (the same prefix as the anchor path) FOLLOWED BY the history block, total width ``l0 + num_prop * num_hist``."""
    p0 = w["num_prop"]
    s0 = p0 + w["num_scan"]
    e0 = s0 + w["num_priv_explicit"]
    l0 = e0 + w["num_priv_latent"]
    prop, scan, priv_explicit = obs_hist[:, :p0], obs_hist[:, p0:s0], obs_hist[:, s0:e0]
    hist = obs_hist[:, -(w["num_prop"] * w["num_hist"]):].reshape(-1, w["num_hist"], w["num_prop"])
    scan_latent = drift._mlp(scan, sd, "actor.scan_encoder.", last_act="tanh")
    hist_latent = history_latent(hist, sd, w["num_hist"])
    backbone_in = torch.cat([prop, scan_latent, priv_explicit, hist_latent], dim=1)
    mu = drift._mlp(backbone_in, sd, "actor.actor_backbone.")
    return torch.tanh(mu) if w.get("tanh_encoder_output") else mu


def critic_value(obs: torch.Tensor, sd: dict) -> torch.Tensor:
    """The critic MLP (``ActorCriticRMA.evaluate``): a plain ``_mlp`` over the critic's OWN (wider, privileged)
    observation -- no encoders, unlike the actor (``critic.0`` reads raw columns directly)."""
    return drift._mlp(obs, sd, "critic.")


def estimator_out(prop: torch.Tensor, sd: dict) -> torch.Tensor:
    """The estimator MLP (``DefaultEstimator.forward``): a plain ``_mlp`` over the prop block only."""
    return drift._mlp(prop, sd, "estimator.")


def full_widths(agent_yaml: Path) -> dict:
    """:func:`p2r_drift.actor_widths` plus ``num_hist`` and ``channel_size`` (the history encoder's two extra
    architecture numbers it never needed)."""
    w = drift.actor_widths(agent_yaml)
    cfg = yaml.safe_load(open(agent_yaml))["policy"]["actor"]
    w["num_hist"] = int(cfg["num_hist"])
    she = cfg.get("state_history_encoder") or {}
    w["channel_size"] = int(she.get("channel_size", w["num_hist"]))
    return w


def load_model_sd(ckpt: Path) -> dict:
    return drift.load_sd(ckpt)           # actor.* + critic.* + std(_raw) -- p2r_drift's own loader


def load_estimator_sd(ckpt: Path) -> dict:
    d = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    return {str(k): v.detach().float() for k, v in d["estimator_state_dict"].items()}


def synth_history_obs(obs: torch.Tensor, w: dict) -> torch.Tensor:
    """``obs`` (N, >= l0) -> (N, l0 + num_prop * num_hist): the anchor-path prefix (prop/scan/priv_explicit/
    priv_latent) followed by a SYNTHETIC history tail built by tiling the prop block ``num_hist`` times. The real
    history buffer is not recoverable from a probe bank captured for the anchor path (P2R_PLAN.md's bank only ever
    needed hist_encoding=False) -- this is a stand-in ONLY for the identity check's invariance-to-column-15
    property, never a claim about real history values. The tiled tail's OWN column 15 (in every historical frame)
    is forced to 0, matching the real contract ("the current step's column 15 only, never the history buffer"):
    the history encoder's weights are untouched by the graft, so a tail that carried the live bit WOULD see a
    different input between bit=0 / bit=1 for a reason that has nothing to do with the graft -- a property of a
    REAL history buffer, enforced here by construction rather than left to chance."""
    p0 = w["num_prop"]
    s0 = p0 + w["num_scan"]
    e0 = s0 + w["num_priv_explicit"]
    l0 = e0 + w["num_priv_latent"]
    prefix = obs[:, :l0]
    tail = obs[:, :p0].clone().repeat(1, w["num_hist"])
    for frame in range(w["num_hist"]):
        tail[:, frame * p0 + COL] = 0.0
    return torch.cat([prefix, tail], dim=1)


def run_identity_check(source: Path, graft: Path, bank_path: Path, agent_yaml: Path | None = None) -> dict:
    """The CPU identity check (module docstring / CLI). Returns ``{"ok": bool, "checks": [(name, ok)]}``."""
    aw = agent_yaml or (Path(source).parent / "params" / "agent.yaml")
    w = full_widths(aw)
    bank = drift.load_bank(bank_path)
    obs = bank["obs"]
    sd_s1, sd_graft = load_model_sd(source), load_model_sd(graft)
    es1, egraft = load_estimator_sd(source), load_estimator_sd(graft)
    obs0, obs1 = obs.clone(), obs.clone()
    obs0[:, COL], obs1[:, COL] = 0.0, 1.0

    checks = []

    def eq(name, a, b):
        checks.append((name, bool(torch.equal(a, b))))

    with torch.no_grad():
        mu_s1 = drift.actor_mu(obs0, sd_s1, w)
        eq("actor anchor-path: graft(bit=0) == S1(bit=0)", drift.actor_mu(obs0, sd_graft, w), mu_s1)
        eq("actor anchor-path: graft(bit=1) == S1(bit=0)", drift.actor_mu(obs1, sd_graft, w), mu_s1)

        oh0, oh1 = synth_history_obs(obs0, w), synth_history_obs(obs1, w)
        mu_s1_h = actor_mu_history_path(oh0, sd_s1, w)
        eq("actor history-path: graft(bit=0) == S1(bit=0)", actor_mu_history_path(oh0, sd_graft, w), mu_s1_h)
        eq("actor history-path: graft(bit=1) == S1(bit=0)", actor_mu_history_path(oh1, sd_graft, w), mu_s1_h)

        v_s1 = critic_value(obs0, sd_s1)
        eq("critic: graft(bit=0) == S1(bit=0)", critic_value(obs0, sd_graft), v_s1)
        eq("critic: graft(bit=1) == S1(bit=0)", critic_value(obs1, sd_graft), v_s1)

        prop0, prop1 = obs0[:, :w["num_prop"]], obs1[:, :w["num_prop"]]
        e_s1 = estimator_out(prop0, es1)
        eq("estimator: graft(bit=0) == S1(bit=0)", estimator_out(prop0, egraft), e_s1)
        eq("estimator: graft(bit=1) == S1(bit=0)", estimator_out(prop1, egraft), e_s1)

    return {"ok": all(ok for _, ok in checks), "checks": checks, "n": int(obs.shape[0]), "widths": w}


# ================================================================================================ self-test
def _tiny_checkpoint(num_prop=20, num_scan=5, num_priv_explicit=3, num_priv_latent=4, num_hist=10, channel_size=4,
                     num_actions=3, num_critic_obs=40, seed=0) -> dict:
    """A toy checkpoint with every architecture piece the real one has (scan/priv encoders, a 2-conv-layer history
    encoder at tsteps=10, actor backbone, critic, estimator, Adam state for every stepped param, estimator Adam
    state) -- small enough for :func:`_self_test` to run in milliseconds, with no real checkpoint needed."""
    g = torch.Generator().manual_seed(seed)

    def lin(n_in, n_out):
        return torch.randn(n_out, n_in, generator=g) * 0.3, torch.randn(n_out, generator=g) * 0.1

    def conv(c_in, c_out, k):
        return torch.randn(c_out, c_in, k, generator=g) * 0.3, torch.randn(c_out, generator=g) * 0.1
    scan_out, priv_out = 6, 5
    backbone_in = num_prop + scan_out + num_priv_explicit + priv_out
    msd = {}
    msd["std_raw"] = torch.zeros(num_actions)
    w, b = lin(num_priv_latent, 7); msd["actor.priv_encoder.0.weight"], msd["actor.priv_encoder.0.bias"] = w, b
    w, b = lin(7, priv_out); msd["actor.priv_encoder.2.weight"], msd["actor.priv_encoder.2.bias"] = w, b
    w, b = lin(num_prop, 3 * channel_size)
    msd["actor.history_encoder.encoder.0.weight"], msd["actor.history_encoder.encoder.0.bias"] = w, b
    w, b = conv(3 * channel_size, 2 * channel_size, 4)
    msd["actor.history_encoder.conv_layers.0.weight"], msd["actor.history_encoder.conv_layers.0.bias"] = w, b
    w, b = conv(2 * channel_size, channel_size, 2)
    msd["actor.history_encoder.conv_layers.2.weight"], msd["actor.history_encoder.conv_layers.2.bias"] = w, b
    w, b = lin(channel_size * 3, priv_out)
    msd["actor.history_encoder.linear_output.0.weight"], msd["actor.history_encoder.linear_output.0.bias"] = w, b
    w, b = lin(num_scan, 9); msd["actor.scan_encoder.0.weight"], msd["actor.scan_encoder.0.bias"] = w, b
    w, b = lin(9, scan_out); msd["actor.scan_encoder.2.weight"], msd["actor.scan_encoder.2.bias"] = w, b
    w, b = lin(backbone_in, 11); msd["actor.actor_backbone.0.weight"], msd["actor.actor_backbone.0.bias"] = w, b
    w, b = lin(11, num_actions); msd["actor.actor_backbone.2.weight"], msd["actor.actor_backbone.2.bias"] = w, b
    w, b = lin(num_critic_obs, 13); msd["critic.0.weight"], msd["critic.0.bias"] = w, b
    w, b = lin(13, 1); msd["critic.2.weight"], msd["critic.2.bias"] = w, b
    esd = {}
    w, b = lin(num_prop, 10); esd["estimator.0.weight"], esd["estimator.0.bias"] = w, b
    w, b = lin(10, num_priv_explicit); esd["estimator.2.weight"], esd["estimator.2.bias"] = w, b

    def adam_state(shape):
        return {"step": torch.tensor(1234.0), "exp_avg": torch.zeros(*shape), "exp_avg_sq": torch.zeros(*shape)}
    hist_names = {"actor.history_encoder.encoder.0.weight", "actor.history_encoder.encoder.0.bias",
                 "actor.history_encoder.conv_layers.0.weight", "actor.history_encoder.conv_layers.0.bias",
                 "actor.history_encoder.conv_layers.2.weight", "actor.history_encoder.conv_layers.2.bias",
                 "actor.history_encoder.linear_output.0.weight", "actor.history_encoder.linear_output.0.bias"}
    opt_state = {i: adam_state(v.shape) for i, (k, v) in enumerate(msd.items()) if k not in hist_names}
    est_state = {i: adam_state(v.shape) for i, v in enumerate(esd.values())}
    agent_yaml_dict = {"policy": {"actor": {"num_prop": num_prop, "num_scan": num_scan,
                                            "num_priv_explicit": num_priv_explicit,
                                            "num_priv_latent": num_priv_latent, "num_hist": num_hist,
                                            "state_history_encoder": {"channel_size": channel_size}},
                                  "tanh_encoder_output": False, "noise_std_type": "soft"}}
    d = {"model_state_dict": msd, "estimator_state_dict": esd,
        "optimizer_state_dict": {"state": opt_state, "param_groups": [{"params": list(range(len(msd)))}]},
        "alg_state": {"estimator_optimizer": {"state": est_state,
                                              "param_groups": [{"params": list(range(len(esd)))}]}},
        "iter": 999, "infos": None}
    return d, agent_yaml_dict, {"num_prop": num_prop, "num_scan": num_scan, "num_priv_explicit": num_priv_explicit,
                               "num_priv_latent": num_priv_latent, "num_hist": num_hist,
                               "num_critic_obs": num_critic_obs}


def _self_test() -> int:
    import tempfile

    torch.manual_seed(0)
    ok = True

    def check_(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        d, agent_yaml_dict, dims = _tiny_checkpoint()
        src = td / "model_19994.pt"
        torch.save(d, str(src))
        (td / "params").mkdir()
        yaml.safe_dump(agent_yaml_dict, open(td / "params" / "agent.yaml", "w"))
        src_sha = sha256_of(src)

        out = td / "graft" / "model_19994.pt"
        payload = build_graft(src, out, expect_source_sha256=src_sha)
        check_("build_graft: writes a readable checkpoint", out.is_file())
        check_("build_graft: source_sha256 matches", payload["source_sha256"] == src_sha)
        check_("build_graft: three targets zeroed", len(payload["zeroed"]) == 3)
        check_("build_graft: three Adam entries zeroed", len(payload["adam_zeroed"]) == 3)
        check_("build_graft: identity_cpu starts false", payload["identity_cpu"] is False)
        try:
            build_graft(src, td / "bad.pt", expect_source_sha256="deadbeef" * 8)
            check_("build_graft: a wrong expected sha256 raises", False)
        except ValueError:
            check_("build_graft: a wrong expected sha256 raises", True)

        g = torch.load(str(out), map_location="cpu", weights_only=False)
        check_(f"graft: actor_backbone.0.weight column {COL} is exactly 0",
              bool((g["model_state_dict"]["actor.actor_backbone.0.weight"][:, COL] == 0).all()))
        check_(f"graft: critic.0.weight column {COL} is exactly 0",
              bool((g["model_state_dict"]["critic.0.weight"][:, COL] == 0).all()))
        check_(f"graft: estimator.0.weight column {COL} is exactly 0",
              bool((g["estimator_state_dict"]["estimator.0.weight"][:, COL] == 0).all()))
        other_w_same = torch.equal(g["model_state_dict"]["critic.2.weight"], d["model_state_dict"]["critic.2.weight"])
        check_("graft: an untouched weight is bitwise identical", other_w_same)
        hist_same = torch.equal(g["model_state_dict"]["actor.history_encoder.encoder.0.weight"],
                                d["model_state_dict"]["actor.history_encoder.encoder.0.weight"])
        check_("graft: the history encoder is untouched (no Adam state, never a target)", hist_same)

        # a column NOT pre-zero in Adam state must refuse (the premise check)
        d2, ay2, _ = _tiny_checkpoint(seed=1)
        d2["optimizer_state_dict"]["state"][list(d2["model_state_dict"]).index("critic.0.weight")]["exp_avg"][:, COL] = 5.0
        src2 = td / "model_bad_adam.pt"
        torch.save(d2, str(src2))
        try:
            build_graft(src2, td / "graft_bad.pt", expect_source_sha256=sha256_of(src2))
            check_("build_graft: a non-zero Adam moment at column 15 refuses", False)
        except AssertionError:
            check_("build_graft: a non-zero Adam moment at column 15 refuses", True)

        # identity check, with a synthetic bank matching the toy widths (critic obs width = num_critic_obs)
        n = 32
        bank_obs = torch.randn(n, dims["num_critic_obs"], generator=torch.Generator().manual_seed(2))
        bank_obs[:, COL] = 0.0
        bank_path = td / "bank.npz"
        import numpy as np
        np.savez(bank_path, obs=bank_obs.numpy(), s1_sigma=np.full(3, 0.3, dtype="float32"),
                hold_label=np.array(["fwd"] * n), category=np.array(["flat"] * n))
        res = run_identity_check(src, out, bank_path, agent_yaml=td / "params" / "agent.yaml")
        check_("identity-check: every check passes on a valid graft", res["ok"])
        for name, good in res["checks"]:
            check_(f"  identity: {name}", good)

        # a graft that does NOT zero critic.0 column 15 must fail the critic check (a true negative)
        bad_graft = copy.deepcopy(d)
        bad_graft["model_state_dict"]["actor.actor_backbone.0.weight"] = \
            zero_weight_column(d["model_state_dict"]["actor.actor_backbone.0.weight"])[0]
        bad_graft["estimator_state_dict"]["estimator.0.weight"] = \
            zero_weight_column(d["estimator_state_dict"]["estimator.0.weight"])[0]
        bad_path = td / "bad_graft.pt"
        torch.save(bad_graft, str(bad_path))
        res_bad = run_identity_check(src, bad_path, bank_path, agent_yaml=td / "params" / "agent.yaml")
        check_("identity-check: an un-grafted critic is caught (true negative)", not res_bad["ok"])

    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--graft-json", type=Path, default=None)
    ap.add_argument("--expect-source-sha256", default=DEFAULT_SOURCE_SHA256)
    ap.add_argument("--identity-check", action="store_true")
    ap.add_argument("--graft", type=Path)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--agent-yaml", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    if a.identity_check:
        if not (a.source and a.graft and a.bank):
            ap.error("--identity-check needs --source, --graft and --bank")
        res = run_identity_check(a.source, a.graft, a.bank, agent_yaml=a.agent_yaml)
        print(json.dumps({k: v for k, v in res.items() if k != "widths"}, default=str, indent=1))
        gj = a.graft_json or a.graft.with_suffix(".json")
        if res["ok"] and gj.is_file():
            payload = json.loads(gj.read_text())
            payload["identity_cpu"] = True
            gj.write_text(json.dumps(payload, indent=1))
            print(f"identity_cpu: true written to {gj}")
        return 0 if res["ok"] else 1
    if not (a.source and a.out):
        ap.error("--source and --out are required (or --identity-check / --self-test)")
    sha = a.expect_source_sha256 or None
    payload = build_graft(a.source, a.out, expect_source_sha256=sha)
    gj = a.graft_json or a.out.with_suffix(".json")
    gj.parent.mkdir(parents=True, exist_ok=True)
    gj.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
