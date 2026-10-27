#!/usr/bin/env python3
"""Omni phase 2, stage p2r -- the CPU drift scorer (P2R_PLAN.md step 1 probe variant / step 6 "anchor dose" / step 7
"early anchor-strength check"). CPU only, read-only: no Isaac, no CUDA, no training.

WHY A HAND-REPLICA, NOT AN IMPORT of ``scripts/rsl_rl/modules``. ``import``-ing those classes transitively pulls in
``scripts/rsl_rl/__init__.py`` -> ``exporter`` -> ``isaaclab_rl`` -> ``isaaclab.utils.mesh`` -> ``pxr`` (USD), which is
only importable inside the Kit/Isaac process. Every existing CPU-only drift / distillation tool in this repo (see
``2026-09-30_rl_algorithm_review/artifacts/*/recon.py``, ``2026-09-28_side_bisect/.../rep/recon.py``) works around
this the same way: the architecture is REPLICATED in plain ``torch``, with every weight read directly off the
checkpoint's own ``model_state_dict`` (never re-derived from scratch), so a drift in the checkpoint's own numbers can
never hide behind a drift in some OTHER, hand-typed copy of the numbers. This module follows that precedent.

ARCHITECTURE (``scripts/rsl_rl/modules/actor_critic_with_encoder.py`` ``Actor.forward`` / ``infer_priv_latent``,
mirrored exactly; the ``+ the S1 env width`` the plan asks for are the four slice widths below, read from the run's
own ``params/agent.yaml`` -- ``policy.actor.num_prop`` / ``num_scan`` / ``num_priv_explicit`` / ``num_priv_latent``):
  obs (N, W) -- prop = obs[:, :num_prop]; scan = obs[:, num_prop:num_prop+num_scan]; priv_explicit =
  obs[:, p+s:p+s+num_priv_explicit] (the TRUE priv_explicit the bank was written with, never the estimator's own
  output -- that is what makes this "the anchor path"); priv_latent_raw = obs[:, p+s+e:p+s+e+num_priv_latent].
  ``hist_encoding=False`` (the anchor path, P2R_PLAN.md) means the actor never touches the trailing history slice at
  all: ``infer_priv_latent`` runs on ``priv_latent_raw`` instead of the history encoder. mu =
  actor_backbone(cat[prop, scan_encoder(scan), priv_explicit, priv_encoder(priv_latent_raw)]), with the final
  backbone layer's activation (none, unless ``tanh_encoder_output``) and every encoder's own layer count and widths
  read off the checkpoint's state dict itself (:func:`_mlp`), never off ``agent.yaml`` -- so a graft that only
  changes ONE observation column (P2R_PLAN.md step 4's clock-column bit) never needs this file touched.

D (P2R_PLAN.md "the anchor dose" / "the early anchor-strength check"): for a bank row with true obs ``o`` and two
checkpoints A (a probed / trained checkpoint) and S1 (the frozen anchor target), ``D_i = 1/2 sum_d ((muA_i,d -
muS1_i,d) / sigma_S1,d)^2``; ``D(name)`` is the mean of ``D_i`` over every bank row (the bank already holds only
NON-STAND steps -- P2R_PLAN.md's shared contract -- so this mean already IS "D over non-stand steps"; there is no
further masking to do here). ``sigma_S1`` is never recomputed here: it is read verbatim from the bank's own
``s1_sigma`` (18,) -- the GPU harness's own read of S1's ``effective_std()`` at capture time, so a drift tool and the
harness that wrote the bank can never disagree about what S1's sigma was.

CLI:
    p2r_drift.py --bank BANK.npz --ref S1_CKPT --ckpt NAME=PATH [NAME=PATH ...] [--agent-yaml PATH] [--json OUT]
        D(name) for every --ckpt, plus the non-stand mean |mu - mu_S1| magnitude (reported) and a by-hold-label
        breakdown when the bank carries ``hold_label``. ``--agent-yaml`` overrides the default
        ``<ref ckpt>/../params/agent.yaml`` (the "S1 env width" source).
    p2r_drift.py --self-test
        a tiny synthetic bank + a 1-layer actor: two copies of the same checkpoint read D == 0 exactly; a perturbed
        copy reads D > 0. Exit 0 / 1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent

#: Actor.update_distribution's three noise_std_type kernels (crab_actor_critic_with_encoder.py), reimplemented here
#: in plain torch for the same reason as the rest of this module -- importing that file pulls in the package's own
#: __init__.py (-> isaaclab -> pxr). ``soft`` is what every B-recipe head of record trains with (KRABBY_STD_PARAM
#: soft); ``scalar`` / ``log`` are supported too since a pre-B checkpoint (or REFS's own replicate seeds) may not be.
SOFT_STD_MIN, SOFT_STD_MAX = 0.05, 2.0
SOFT_LOG_LO, SOFT_LOG_HI = math.log(SOFT_STD_MIN), math.log(SOFT_STD_MAX)


def soft_std(x: torch.Tensor) -> torch.Tensor:
    span = SOFT_LOG_HI - SOFT_LOG_LO
    log_std = torch.where(x >= 0, SOFT_LOG_HI - span * torch.sigmoid(-2.0 * x),
                          SOFT_LOG_LO + span * torch.sigmoid(2.0 * x))
    return torch.exp(log_std)


def effective_std(sd: dict, noise_std_type: str = "soft") -> torch.Tensor | None:
    """(num_actions,) -- reported only (:func:`p2r_dose`); the drift score itself always uses the BANK's own
    ``s1_sigma``, never this (see module docstring)."""
    if noise_std_type == "soft" and "std_raw" in sd:
        return soft_std(sd["std_raw"])
    if noise_std_type == "scalar" and "std" in sd:
        return sd["std"].clamp(SOFT_STD_MIN, SOFT_STD_MAX)
    if noise_std_type == "log" and "log_std" in sd:
        return torch.exp(sd["log_std"]).clamp(SOFT_STD_MIN, SOFT_STD_MAX)
    return None


# ================================================================================================ architecture replica
def actor_widths(agent_yaml: Path) -> dict:
    """``policy.actor``'s four slice widths plus ``tanh_encoder_output`` and ``noise_std_type`` -- "the S1 env
    width" (P2R_PLAN.md step 6/7). Every checkpoint scored against the same bank must share these (the bank is one
    run's own observation layout); this file never re-derives them from a checkpoint's tensor shapes, so a width
    mismatch is a loud ``KeyError`` / ``ValueError`` at load time, not a silently wrong slice."""
    cfg = yaml.safe_load(open(agent_yaml))["policy"]
    a = cfg["actor"]
    return {"num_prop": int(a["num_prop"]), "num_scan": int(a["num_scan"]),
            "num_priv_explicit": int(a["num_priv_explicit"]), "num_priv_latent": int(a["num_priv_latent"]),
            "tanh_encoder_output": bool(cfg.get("tanh_encoder_output", False)),
            "noise_std_type": str(cfg.get("noise_std_type", "soft"))}


def load_sd(ckpt: Path) -> dict:
    """The checkpoint's ``model_state_dict`` (actor + critic + std; never the estimator -- the anchor path never
    calls it), as float32 CPU tensors."""
    d = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    return {str(k): v.detach().float() for k, v in d["model_state_dict"].items()}


def _mlp(x: torch.Tensor, sd: dict, pre: str, last_act: str | None = None) -> torch.Tensor:
    """One ``nn.Sequential`` of alternating ``Linear`` + activation, with its layer count and widths read off
    ``sd``'s own keys (``<pre><i>.weight`` for integer ``i``) rather than off ``agent.yaml`` -- so this is correct
    whether the encoder is a real MLP or (``priv_encoder_dims == []`` / no scan encoder) an ``nn.Identity``, which
    this reads as "no layers" and returns ``x`` unchanged. ELU between every pair of layers (the construction code's
    own convention -- ``actor_critic_with_encoder.py``'s ``Actor.__init__``); ``last_act`` ("tanh" / "elu" / None)
    is the activation after the FINAL layer only (scan encoder: tanh; priv encoder: elu; actor backbone: none,
    unless ``tanh_encoder_output``, handled by the caller)."""
    idx = sorted({int(k[len(pre):].split(".")[0]) for k in sd if k.startswith(pre) and k.endswith("weight")})
    for j, i in enumerate(idx):
        x = F.linear(x, sd[f"{pre}{i}.weight"], sd[f"{pre}{i}.bias"])
        if j < len(idx) - 1:
            x = F.elu(x)
        elif last_act == "tanh":
            x = torch.tanh(x)
        elif last_act == "elu":
            x = F.elu(x)
    return x


def actor_mu(obs: torch.Tensor, sd: dict, w: dict) -> torch.Tensor:
    """mu on the ANCHOR path (``hist_encoding=False``): the TRUE priv_explicit columns are taken straight from
    ``obs`` (never the estimator's), and ``infer_priv_latent`` (not the history encoder) fills the latent slot.
    Faithful to ``Actor.forward`` / ``infer_priv_latent`` -- see the module docstring for why this is a replica,
    not an import."""
    p0 = w["num_prop"]
    s0 = p0 + w["num_scan"]
    e0 = s0 + w["num_priv_explicit"]
    l0 = e0 + w["num_priv_latent"]
    prop, scan, priv_explicit, priv_raw = obs[:, :p0], obs[:, p0:s0], obs[:, s0:e0], obs[:, e0:l0]
    scan_latent = _mlp(scan, sd, "actor.scan_encoder.", last_act="tanh")
    priv_latent = _mlp(priv_raw, sd, "actor.priv_encoder.", last_act="elu")
    backbone_in = torch.cat([prop, scan_latent, priv_explicit, priv_latent], dim=1)
    mu = _mlp(backbone_in, sd, "actor.actor_backbone.")
    return torch.tanh(mu) if w.get("tanh_encoder_output") else mu


# ================================================================================================ the bank / D
def load_bank(path: Path) -> dict:
    """The P2R_PLAN.md shared-contract probe bank: ``obs`` (N, W) float32 (TRUE priv columns), ``hold_label`` (N,)
    str, ``category`` (N,) str, ``s1_sigma`` (18,) float32. Every bank row is already a NON-STAND step (the hold
    label is not "stand" / "settle") -- the harness filters that at capture time, so :func:`drift` needs no further
    masking to read "D over non-stand steps"."""
    d = np.load(path, allow_pickle=False)
    out = {"obs": torch.as_tensor(np.asarray(d["obs"]), dtype=torch.float32),
          "s1_sigma": torch.as_tensor(np.asarray(d["s1_sigma"]), dtype=torch.float32)}
    for k in ("hold_label", "category"):
        if k in d.files:
            out[k] = np.asarray(d[k])
    return out


def drift(bank: dict, sd_a: dict, sd_s1: dict, w: dict) -> dict:
    """D(A vs S1) over the whole bank (= over non-stand steps, by the bank's own construction): mean of
    ``1/2 sum_d ((muA - muS1) / s1_sigma)^2``; plus, when the bank carries ``hold_label``, the per-label mean (
    reported only). Also reported: the mean |muA - muS1| magnitude (uninterpreted; 0 would still be D == 0)."""
    with torch.no_grad():
        obs = bank["obs"]
        mu_a, mu_s1 = actor_mu(obs, sd_a, w), actor_mu(obs, sd_s1, w)
        sigma = bank["s1_sigma"].clamp_min(1e-9)
        d_i = 0.5 * (((mu_a - mu_s1) / sigma) ** 2).sum(dim=1)
        mag = (mu_a - mu_s1).abs().mean(dim=1)
    out = {"D": float(d_i.mean()), "n": int(d_i.shape[0]), "mean_abs_dmu": float(mag.mean())}
    if "hold_label" in bank:
        labels = bank["hold_label"]
        by = {}
        for lab in sorted(set(labels.tolist())):
            m = labels == lab
            by[str(lab)] = {"D": float(d_i[torch.as_tensor(m)].mean()), "n": int(m.sum())}
        out["by_label"] = by
    return out


def parse_ckpt_args(items: list) -> dict:
    """``NAME=PATH`` CLI tokens -> ``{NAME: Path(PATH)}``; a token with no ``=`` or an empty name is refused."""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--ckpt expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if not name:
            raise ValueError(f"--ckpt expects NAME=PATH, got {item!r}")
        out[name] = Path(path)
    return out


def run(bank_path: Path, ref_ckpt: Path, ckpts: dict, agent_yaml: Path | None = None) -> dict:
    """Every ``--ckpt`` scored against ``ref_ckpt`` (S1) on ``bank_path``; also the reported-only sigma / dose
    context (``effective_std`` of each scored checkpoint, median |mu| -- never used by :func:`drift` itself)."""
    aw = agent_yaml or (ref_ckpt.parent / "params" / "agent.yaml")
    w = actor_widths(aw)
    bank = load_bank(bank_path)
    sd_s1 = load_sd(ref_ckpt)
    out = {"ref": str(ref_ckpt), "agent_yaml": str(aw), "bank": str(bank_path), "n": int(bank["obs"].shape[0]),
          "widths": w, "ckpts": {}}
    for name, path in ckpts.items():
        sd_a = load_sd(path)
        r = drift(bank, sd_a, sd_s1, w)
        r["path"] = str(path)
        out["ckpts"][name] = r
    return out


# ================================================================================================ self-test
def _tiny_sd(num_prop=4, num_scan=3, num_priv_explicit=2, num_priv_latent=2, hidden=5, num_actions=2,
            seed=0, scale=1.0) -> dict:
    """A 1-layer scan / priv encoder + 1-hidden-layer backbone, random weights (reproducible by ``seed``), scaled
    by ``scale`` so two checkpoints at ``scale=1`` are bitwise-identical copies and a third at a different scale
    reads a nonzero drift against them."""
    g = torch.Generator().manual_seed(seed)

    def lin(n_in, n_out):
        return (torch.randn(n_out, n_in, generator=g) * scale, torch.randn(n_out, generator=g) * scale)
    w1, b1 = lin(num_scan, 6)
    w2, b2 = lin(num_priv_latent, 6)
    wb, bb = lin(num_prop + 6 + num_priv_explicit + 6, hidden)
    wo, bo = lin(hidden, num_actions)
    return {"actor.scan_encoder.0.weight": w1, "actor.scan_encoder.0.bias": b1,
            "actor.priv_encoder.0.weight": w2, "actor.priv_encoder.0.bias": b2,
            "actor.actor_backbone.0.weight": wb, "actor.actor_backbone.0.bias": bb,
            "actor.actor_backbone.2.weight": wo, "actor.actor_backbone.2.bias": bo}


def _self_test() -> int:
    torch.manual_seed(0)
    w = {"num_prop": 4, "num_scan": 3, "num_priv_explicit": 2, "num_priv_latent": 2, "tanh_encoder_output": False}
    n = 32
    obs = torch.randn(n, w["num_prop"] + w["num_scan"] + w["num_priv_explicit"] + w["num_priv_latent"] + 20)
    bank = {"obs": obs, "s1_sigma": torch.full((2,), 0.3),
           "hold_label": np.array((["fwd"] * (n // 2)) + (["diag"] * (n - n // 2)))}
    sd_s1 = _tiny_sd(seed=1, scale=1.0)
    sd_copy = _tiny_sd(seed=1, scale=1.0)
    sd_pert = _tiny_sd(seed=1, scale=1.3)
    ok = True
    r_same = drift(bank, sd_copy, sd_s1, w)
    print(f"two copies of S1: D = {r_same['D']!r} (want exactly 0.0)")
    if r_same["D"] != 0.0:
        ok = False
    r_pert = drift(bank, sd_pert, sd_s1, w)
    print(f"a perturbed copy: D = {r_pert['D']!r} (want > 0)")
    if not r_pert["D"] > 0.0:
        ok = False
    if "fwd" not in r_pert.get("by_label", {}) or "diag" not in r_pert.get("by_label", {}):
        print("by_label breakdown missing fwd/diag")
        ok = False
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--agent-yaml", type=Path, default=None)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    if not (a.bank and a.ref and a.ckpt):
        ap.error("--bank, --ref and at least one --ckpt NAME=PATH are required (or --self-test)")
    try:
        ckpts = parse_ckpt_args(a.ckpt)
    except ValueError as exc:
        ap.error(str(exc))
        return 2
    out = run(a.bank, a.ref, ckpts, agent_yaml=a.agent_yaml)
    text = json.dumps(out, indent=1, default=str)
    if a.json:
        a.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
