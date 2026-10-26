#!/usr/bin/env python3
"""Omni phase 2, stage p2r Z2 -- the CPU stand-fidelity check (P2R_Z2_PLAN.md step 4 "Early checks: stand
fidelity", the user's 2026-10-10 decision; CHARTER 12). CPU only, read-only: no Isaac, no CUDA, no training.
NOTE(p2r Z2, 2026-10-10).

WHAT IT READS. A checkpoint's action mean mu on the restance probe's banked stand states
(``p2r_stand_bank.npz``, written by ``p2r_mode_eval_crab_hex_gait.py --p2r-mode restance --p2r-stand-bank``: the TRUE
obs, R's target and branch, the policy-visible proxy columns), against R:
  * per branch (``BRANCH_NAMES``; leg-rows in that branch): |mu - R| of that leg's cam / hip / knee dims (mean,
    median, p90);
  * per row: mean |mu - R| over all 18 dims, and the prior ``0.5 sum ((mu - R) / sigma)^2`` (sigma = the bank's
    ``s1_sigma`` = sigma_A);
  * THE LINE: per leg, the mean SIGNED loaded-leg cam residual ``mu_cam - R_cam`` over that leg's HOLD-branch rows
    (down and inside the home band -- "loaded" before the 2026-10-10 cams-only amendment, whose kinematic R reads the
    leg phase from the hip-pitch position --, where R's cam command is the +-0.03 restoring trim); NOTIFY (rc 3) when the max
    over legs of |mean| exceeds 0.015 -- half the trim's authority (P2R_Z2_PLAN step 4; the 0.006 line applied only
    to a hold with no trim).

THE EVAL PATH (P2R_Z2_PLAN "Left for implementation": "Use the eval path's hist_encoding setting (the drift scorer
and the anchor use False)"). mu is computed exactly as the gait-eval harness computes the action: the priv_explicit
columns are overwritten with the scored checkpoint's OWN estimator on ``obs[:, :num_prop]`` (the harness's in-place
write) and the actor runs with ``hist_encoding=True`` (the history encoder on the trailing ``num_hist x num_prop``
columns, never ``priv_latent``). ``--priv true`` / ``--no-hist-encoding`` switch either half off (diagnostics). The
architecture is a plain-torch replica (as :mod:`p2r_drift`, whose ``_mlp`` / ``load_sd`` it reuses; every weight read
off the checkpoint, the history encoder's conv strides from the class's own ``tsteps`` table) -- importing the
modules pulls in Isaac. ``--xcheck-stamp`` verifies the replica end to end: a probe stamp's ``p2r_bank.npz`` rows
(true obs) matched to their raw step by joint velocity, the replica's eval-path mu against the RECORDED action of
the checkpoint that ran the stamp (``--xcheck-ckpt``, default the scored one).

R RECOMPUTED, NOT STORED (NOTE(p2r Z2c, 2026-10-10); P2R_Z2C_PLAN.md "Left for implementation": "Fidelity scorer:
it must recompute R(obs) with the run's KRABBY_RESTANCE (or refuse a bank whose `restance` differs); today it uses
the bank's stored target"). The target and branch every number above is read against are ``restance_target(obs, p)``
on the bank's own obs, with ``p`` = ``--restance K=V,...`` or, else, the scored run's ``params/agent.yaml``
``algorithm.restance`` (the explicit ``--agent-yaml`` first, then the checkpoint's own run dir); neither -> refused
(rc 1), never a silent fall-back to the bank's string. ``--stored-target`` keeps the old behaviour (the bank's stored
``target`` / ``branch``). ``target_vs_stored_max_abs`` reports how far the recomputed R sits from the stored one (a
residual-pass bank's stored target, or a bank recorded under other constants).

BOTH PATHS + THE ALL-HOLD STATISTIC (NOTE(p2r Z2c, 2026-10-10); P2R_Z2C_PLAN step 1 "Fidelity scorer" / step 5
"FIDELITY GATE"). ``paths`` scores the bank twice, whatever the primary flags:
  * ``eval``  -- the harness's action (own-estimator priv_explicit, ``hist_encoding=True``): what the gate judges;
  * ``train`` -- the PPO / BC training-path forward (true priv columns, ``hist_encoding=False``): reported.
Each carries the ALL-HOLD statistic: on the rows whose EVERY leg's (recomputed) branch is HOLD (R commands rods 0 and
the cam trim), the per-leg mean SIGNED ``mu - R`` of the hip (action cols 6..11), knee (12..17) and cam (0..5) dims and
the max ``|.|`` over legs (``p2r_z2c_gate.py`` holds hip and cam to 0.05; Z2 read hip 0.13-0.18). The primary path's
own HOLD-branch cam line (below) and every earlier output are unchanged.

THE STAND-ROW INPUT MASK (NOTE(p2r Z2d, 2026-10-11); P2R_Z2D_PLAN.md step 1 "The gate"). ``--stand-mask`` scores
the bank as a masked env feeds the policy: AFTER R is recomputed (R reads columns 3, 4, 18..34, none of them masked),
every bank row goes through ``crab_hex_omni_restance.stand_mask_obs`` (the env's and the cloning bank loader's own
function: last action 0, proxy -0.5, current step + the real history frames on a full-width row) before any path is
scored. The JSON records ``masked``. Off (the default), nothing changes.

``score_paths(..., by_branch=True)`` (NOTE(p2r Z2d, 2026-10-11); the gate's LIFT / HOME report) adds each path's
``by_branch``: per R branch, the leg-rows in it and ``|mu - R|`` of that leg's cam / hip / knee (mean, median, p90)
-- :func:`score`'s table, on that path.

Exit codes: 0 = within the line; 3 = NOTIFY (the loaded-leg cam residual line); 2 = the bank has no HOLD-branch row
(cannot judge); 1 = unreadable input / no R constants / the replica cross-check failed. ``--dry-run`` (the pre-launch
run on GRAFT_S1 against the bank) computes everything and reports ``would_notify`` but exits 0 unless an input is
unreadable.

CLI:
    p2r_z2_fidelity.py --bank p2r_stand_bank.npz --ckpt MODEL.pt [--agent-yaml PATH] [--priv estimator|true]
                       [--no-hist-encoding] [--restance K=V,... | --stored-target] [--xcheck-stamp DIR]
                       [--xcheck-ckpt PATH] [--stand-mask] [--dry-run] [--json OUT]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import p2r_drift as drift  # noqa: E402  (same directory, CPU only: _mlp / load_sd)

# scripts -> 2026-10-07_omni_phase2 -> experiments -> crab_hex_omni_task -> parkour_tasks (inner)
_TASKS = HERE.parents[3]
RESTANCE_MODULE = _TASKS / "crab_hex_omni_task" / "mdp" / "crab_hex_omni_restance.py"
R = Path(__file__).resolve().parents[7]  # this checkout's root (scripts -> campaign -> experiments -> ... -> repo)
#: the S1 run's own params (the graft dir has none; every Z fork shares its widths)
S1_AGENT_YAML = R / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/params/agent.yaml"
CAM_LINE = 0.015                   # notify: max over legs of |mean signed HOLD-branch mu_cam - R_cam|
XCHECK_TOL = 1e-4                  # replica vs recorded GPU action, max abs
OBS_JOINT_VEL = slice(41, 65)
JOINT_VEL_SCALE = np.float32(0.05)
#: StateHistoryEncoder's conv stack per tsteps: (kernel, stride) per Conv1d (feature_extractors/state_encoder.py)
HIST_CONV = {10: ((4, 2), (2, 1)), 20: ((6, 2), (4, 2)), 50: ((8, 4), (5, 1), (5, 1))}
RC_OK, RC_ERROR, RC_NO_ROWS, RC_NOTIFY = 0, 1, 2, 3
#: NOTE(p2r Z2c, 2026-10-10): the two forwards every bank is scored on -- name: (hist_encoding, priv)
PATHS = {"eval": (True, "estimator"), "train": (False, "true")}
#: the obs column carrying the grafted stand bit (KRABBY_OMNI_STAND_OBS=clock; every stand-bank row reads 1)
STAND_BIT_COL = 15


def load_restance(path: Path = RESTANCE_MODULE):
    spec = importlib.util.spec_from_file_location("_p2r_z2f_restance", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================================================ the actor replica
def widths(agent_yaml: Path) -> dict:
    """``policy.actor``'s slice widths + ``num_hist`` and the history channel size, ``tanh_encoder_output``, and the
    activation (ELU is the only one replicated: anything else refused)."""
    cfg = yaml.safe_load(open(agent_yaml))["policy"]
    a = cfg["actor"]
    act = str(cfg.get("activation", "elu"))
    if act != "elu":
        raise ValueError(f"{agent_yaml}: activation {act!r}; only elu is replicated")
    return {"num_prop": int(a["num_prop"]), "num_scan": int(a["num_scan"]),
            "num_priv_explicit": int(a["num_priv_explicit"]), "num_priv_latent": int(a["num_priv_latent"]),
            "num_hist": int(a["num_hist"]), "tanh_encoder_output": bool(cfg.get("tanh_encoder_output", False))}


def load_ckpt(path: Path) -> tuple[dict, dict]:
    """``(model_state_dict, estimator_state_dict)`` as float32 CPU tensors."""
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    msd = {str(k): v.detach().float() for k, v in d["model_state_dict"].items()}
    esd = {str(k): v.detach().float() for k, v in (d.get("estimator_state_dict") or {}).items()}
    return msd, esd


def history_latent(hist: torch.Tensor, sd: dict, num_hist: int, num_prop: int) -> torch.Tensor:
    """``StateHistoryEncoder.forward``: per-step Linear+ELU projection, the tsteps conv stack (ELU after each),
    flatten, Linear+ELU."""
    if num_hist not in HIST_CONV:
        raise ValueError(f"num_hist {num_hist}: StateHistoryEncoder supports {sorted(HIST_CONV)}")
    pre = "actor.history_encoder."
    nd = hist.shape[0]
    x = hist.reshape(nd * num_hist, num_prop)
    x = F.elu(F.linear(x, sd[pre + "encoder.0.weight"], sd[pre + "encoder.0.bias"]))
    x = x.reshape(nd, num_hist, -1).permute(0, 2, 1)
    for j, (_k, stride) in enumerate(HIST_CONV[num_hist]):
        x = F.elu(F.conv1d(x, sd[f"{pre}conv_layers.{2 * j}.weight"], sd[f"{pre}conv_layers.{2 * j}.bias"],
                           stride=stride))
    x = x.flatten(1)
    return F.elu(F.linear(x, sd[pre + "linear_output.0.weight"], sd[pre + "linear_output.0.bias"]))


def eval_mu(obs: torch.Tensor, sd: dict, esd: dict, w: dict, *, hist_encoding: bool = True,
            priv: str = "estimator") -> torch.Tensor:
    """The harness's action on ``obs`` (true priv columns): ``priv == "estimator"`` overwrites priv_explicit with
    the checkpoint's own estimator first (the harness's in-place write), then ``Actor.forward(obs, hist_encoding)``."""
    p0 = w["num_prop"]
    s0 = p0 + w["num_scan"]
    e0 = s0 + w["num_priv_explicit"]
    obs = obs.clone()
    if priv == "estimator":
        if not esd:
            raise ValueError("checkpoint has no estimator_state_dict (--priv true to score on the true priv columns)")
        obs[:, s0:e0] = drift._mlp(obs[:, :p0], esd, "estimator.")
    elif priv != "true":
        raise ValueError(f"priv must be 'estimator' or 'true', got {priv!r}")
    scan_latent = drift._mlp(obs[:, p0:s0], sd, "actor.scan_encoder.", last_act="tanh")
    if hist_encoding:
        latent = history_latent(obs[:, -w["num_hist"] * p0:], sd, w["num_hist"], p0)
    else:
        latent = drift._mlp(obs[:, e0:e0 + w["num_priv_latent"]], sd, "actor.priv_encoder.", last_act="elu")
    mu = drift._mlp(torch.cat([obs[:, :p0], scan_latent, obs[:, s0:e0], latent], dim=1), sd, "actor.actor_backbone.")
    return torch.tanh(mu) if w["tanh_encoder_output"] else mu


# ================================================================================================ the score
def _stats(x: torch.Tensor) -> dict:
    if x.numel() == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    x = x.double().reshape(-1)
    return {"n": int(x.numel()), "mean": float(x.mean()), "median": float(x.median()),
            "p90": float(torch.quantile(x, 0.9))}


def branch_table(mu: torch.Tensor, target, branch, rst) -> dict:
    """Per R branch (pure): the leg-rows in it and ``|mu - R|`` of that leg's cam / hip / knee (:func:`_stats`).
    NOTE(p2r Z2d, 2026-10-11): factored out of :func:`score` (unchanged) so :func:`score_paths` reports it per path."""
    tgt = torch.as_tensor(np.asarray(target), dtype=torch.float32)
    br = torch.as_tensor(np.asarray(branch).astype(np.int64))
    d = mu - tgt
    cam, hip, knee = d[:, rst.CAM_ACT], d[:, rst.HIP_ACT], d[:, rst.KNEE_ACT]
    by_branch = {}
    for k, name in enumerate(rst.BRANCH_NAMES):
        m = br == k
        by_branch[name] = {"leg_rows": int(m.sum()), "cam_abs": _stats(cam.abs()[m]), "hip_abs": _stats(hip.abs()[m]),
                           "knee_abs": _stats(knee.abs()[m])}
    return by_branch


def score(bank: dict, mu: torch.Tensor, rst, line: float = CAM_LINE) -> dict:
    """The fidelity table and the line (pure): ``bank`` carries ``target`` (N, 18), ``branch`` (N, 6), ``s1_sigma``;
    ``mu`` (N, 18) the checkpoint's eval-path mean on the bank's obs."""
    tgt = torch.as_tensor(np.asarray(bank["target"]), dtype=torch.float32)
    br = torch.as_tensor(np.asarray(bank["branch"]).astype(np.int64))
    sigma = torch.as_tensor(np.asarray(bank["s1_sigma"]), dtype=torch.float32).clamp_min(1e-9)
    d = mu - tgt
    cam = d[:, rst.CAM_ACT]
    by_branch = branch_table(mu, bank["target"], bank["branch"], rst)
    hold = br == rst.BRANCH_HOLD
    per_leg = []
    for i in range(len(rst.LEGS)):
        m = hold[:, i]
        per_leg.append(float(cam[m, i].double().mean()) if bool(m.any()) else None)
    vals = [abs(v) for v in per_leg if v is not None]
    worst = max(vals) if vals else None
    out = {"rows": int(mu.shape[0]), "by_branch": by_branch,
           "row_mean_abs": _stats(d.abs().mean(dim=1)),
           "prior_per_row": _stats(0.5 * (d / sigma.reshape(1, -1)).square().sum(dim=1)),
           "hold_cam_residual_signed_by_leg": dict(zip(rst.LEGS, per_leg)),
           "hold_cam_residual_max_abs": worst, "line": line}
    if worst is None:
        out.update(rc=RC_NO_ROWS, why="no HOLD-branch leg-row in the bank: the loaded-cam line cannot be read")
    elif worst > line:
        out.update(rc=RC_NOTIFY, why=f"max over legs |mean HOLD mu_cam - R_cam| {worst:.4g} > {line:g}")
    else:
        out.update(rc=RC_OK, why=f"max over legs |mean HOLD mu_cam - R_cam| {worst:.4g} <= {line:g}")
    return out


# ================================================================================================ R's constants
def restance_from_agent_yaml(path: Path, rst):
    """``algorithm.restance`` of a run's ``params/agent.yaml`` (a dict of R's constants, or a knob string) ->
    ``RestanceParams``, or None when the file / key is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        cfg = yaml.safe_load(open(path)) or {}
    except yaml.YAMLError:
        # a dumped cfg may carry python tags (tuples); read it tolerantly
        class _L(yaml.SafeLoader):
            pass

        _L.add_multi_constructor("", lambda ld, suffix, node: None)
        cfg = yaml.load(open(path), Loader=_L) or {}  # noqa: S506
    r = (cfg.get("algorithm") or {}).get("restance") if isinstance(cfg, dict) else None
    if r is None:
        return None
    if isinstance(r, dict):
        return rst.RestanceParams.from_dict(r)
    return rst.parse_restance(str(r))


def resolve_restance(rst, restance: str | None = None, agent_yamls=()) -> tuple:
    """``(params, source)`` -- NOTE(p2r Z2c, 2026-10-10): ``restance`` (a ``KRABBY_RESTANCE`` string) first, then the
    first of ``agent_yamls`` carrying ``algorithm.restance``. Neither -> ValueError (the scorer REFUSES rather than
    read the bank's stored target under unknown constants)."""
    if restance is not None:
        p = rst.parse_restance(restance)
        if p is None:
            raise ValueError(f"--restance {restance!r} reads OFF (blank / 0): no R constants")
        return p, "--restance"
    for ay in agent_yamls:
        if ay is None:
            continue
        p = restance_from_agent_yaml(Path(ay), rst)
        if p is not None:
            return p, f"{ay}: algorithm.restance"
    raise ValueError("no R constants: pass --restance K=V,... or score a checkpoint whose run dir's params/agent.yaml "
                     "carries algorithm.restance (--stored-target reads the bank's stored target instead)")


def recompute_targets(bank: dict, rst, params) -> dict:
    """``bank`` with ``target`` / ``branch`` replaced by ``restance_target(obs, params)`` (the stored ones kept as
    ``stored_target`` / ``stored_branch``)."""
    obs = torch.as_tensor(np.asarray(bank["obs"]), dtype=torch.float32)
    tgt, br = rst.restance_target(obs, params)
    out = dict(bank, target=tgt.numpy(), branch=br.numpy())
    if "target" in bank:
        out["stored_target"] = bank["target"]
    if "branch" in bank:
        out["stored_branch"] = bank["branch"]
    return out


def mask_bank(bank: dict, rst) -> dict:
    """NOTE(p2r Z2d, 2026-10-11): ``bank`` with every obs row through ``rst.stand_mask_obs`` (the env's stand-row
    mask; full width = current step + the real history frames). Called AFTER :func:`recompute_targets` (R reads none
    of the masked columns, so the targets are the same either way)."""
    obs = torch.as_tensor(np.asarray(bank["obs"]), dtype=torch.float32)
    return dict(bank, obs=rst.stand_mask_obs(obs).numpy())


def allhold_bias(mu: torch.Tensor, target, branch, rst) -> dict:
    """NOTE(p2r Z2c, 2026-10-10): the gate statistic (pure). On ALL-HOLD rows (every leg's branch == HOLD), per leg
    the mean SIGNED ``mu - target`` of cam / hip / knee, and per joint group the max ``|.|`` over legs."""
    tgt = torch.as_tensor(np.asarray(target), dtype=torch.float32)
    br = torch.as_tensor(np.asarray(branch).astype(np.int64))
    ah = (br == rst.BRANCH_HOLD).all(dim=1)
    n = int(ah.sum())
    out = {"rows": int(br.shape[0]), "allhold_rows": n, "allhold_share": (n / br.shape[0] if br.shape[0] else None)}
    d = (mu - tgt)[ah].double()
    for name, sl in (("cam", rst.CAM_ACT), ("hip", rst.HIP_ACT), ("knee", rst.KNEE_ACT)):
        if n == 0:
            out[f"{name}_by_leg"], out[f"{name}_max_abs"] = None, None
            continue
        per = d[:, sl].mean(dim=0)
        out[f"{name}_by_leg"] = dict(zip(rst.LEGS, (float(v) for v in per)))
        out[f"{name}_max_abs"] = float(per.abs().max())
    return out


def score_paths(bank: dict, sd: dict, esd: dict, w: dict, rst, paths=tuple(PATHS), by_branch: bool = False) -> dict:
    """``{path: allhold_bias(...)}`` for each of :data:`PATHS` named in ``paths`` (+ the per-row prior on that path).
    NOTE(p2r Z2d, 2026-10-11): ``by_branch`` adds each path's :func:`branch_table` (the gate's LIFT / HOME report)."""
    obs = torch.as_tensor(np.asarray(bank["obs"]), dtype=torch.float32)
    sigma = torch.as_tensor(np.asarray(bank["s1_sigma"]), dtype=torch.float32).clamp_min(1e-9) \
        if "s1_sigma" in bank else None
    out = {}
    for name in paths:
        he, pv = PATHS[name]
        with torch.no_grad():
            mu = eval_mu(obs, sd, esd, w, hist_encoding=he, priv=pv)
        r = allhold_bias(mu, bank["target"], bank["branch"], rst)
        r.update(hist_encoding=he, priv=pv)
        if by_branch:                              # NOTE(p2r Z2d, 2026-10-11)
            r["by_branch"] = branch_table(mu, bank["target"], bank["branch"], rst)
        if sigma is not None:
            tgt = torch.as_tensor(np.asarray(bank["target"]), dtype=torch.float32)
            r["prior_per_row_mean"] = float((0.5 * ((mu - tgt) / sigma.reshape(1, -1)).square().sum(dim=1)).mean())
        out[name] = r
    return out


def load_bank(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    need = {"obs", "target", "branch", "s1_sigma"}
    if not need <= set(d.files):
        raise ValueError(f"{path}: not a p2r stand bank (missing {sorted(need - set(d.files))})")
    return {k: np.asarray(d[k]) for k in d.files}


def xcheck(stamp: Path, sd: dict, esd: dict, w: dict, max_rows: int = 2000) -> dict:
    """The replica end to end: ``stamp``'s ``p2r_bank.npz`` (true obs) rows matched to their raw step by obs 41..64
    == joint_vel * 0.05 (bitwise); eval-path mu vs the RECORDED action there (the stamp's own checkpoint)."""
    obs = np.asarray(np.load(Path(stamp) / "p2r_bank.npz")["obs"])[:max_rows]
    key = {obs[i, OBS_JOINT_VEL].tobytes(): i for i in range(obs.shape[0])}
    rows, acts = [], []
    for f in sorted((Path(stamp) / "raw").glob("episode_*.npz")):
        if not key:
            break
        z = np.load(f)
        jv = (np.asarray(z["joint_vel"]) * JOINT_VEL_SCALE).astype(np.float32)
        a = None
        for t in range(jv.shape[0]):
            i = key.pop(jv[t].tobytes(), None)
            if i is not None:
                a = np.asarray(z["actions"]) if a is None else a
                rows.append(i)
                acts.append(a[t])
    if not rows:
        return {"stamp": str(stamp), "matched": 0, "ok": False}
    with torch.no_grad():
        mu = eval_mu(torch.as_tensor(obs[rows]), sd, esd, w)
    diff = (mu - torch.as_tensor(np.stack(acts))).abs()
    mx = float(diff.max())
    return {"stamp": str(stamp), "matched": len(rows), "max_abs_diff": mx, "mean_abs_diff": float(diff.mean()),
            "tol": XCHECK_TOL, "ok": mx <= XCHECK_TOL}


def run(bank_path: Path, ckpt: Path, agent_yaml: Path | None = None, *, hist_encoding: bool = True,
        priv: str = "estimator", line: float = CAM_LINE, restance: str | None = None,
        stored_target: bool = False, stand_mask: bool = False) -> dict:
    """The full read. NOTE(p2r Z2c, 2026-10-10): R is RECOMPUTED on the bank's obs with ``restance`` or the run's
    ``params/agent.yaml`` ``algorithm.restance`` (ValueError when neither: refused), unless ``stored_target``; the
    ``paths`` block scores both the eval and the training path with the all-HOLD statistic. NOTE(p2r Z2d,
    2026-10-11): ``stand_mask`` masks every bank row (:func:`mask_bank`) after R, before scoring."""
    rst = load_restance()
    ckpt = Path(ckpt)
    run_yaml = ckpt.parent / "params" / "agent.yaml"
    aw = agent_yaml or run_yaml
    if not Path(aw).is_file():
        aw = S1_AGENT_YAML
    w = widths(aw)
    bank = load_bank(bank_path)
    if stored_target:
        params, src = None, "the bank's stored target (--stored-target)"
    else:
        params, src = resolve_restance(rst, restance, (agent_yaml, run_yaml))
        bank = recompute_targets(bank, rst, params)
    if stand_mask:
        bank = mask_bank(bank, rst)                # NOTE(p2r Z2d, 2026-10-11): after R, before any path
    sd, esd = load_ckpt(ckpt)
    with torch.no_grad():
        mu = eval_mu(torch.as_tensor(bank["obs"], dtype=torch.float32), sd, esd, w, hist_encoding=hist_encoding,
                     priv=priv)
    out = score(bank, mu, rst, line=line)
    out.update(bank=str(bank_path), ckpt=str(ckpt), agent_yaml=str(aw), hist_encoding=hist_encoding, priv=priv,
               restance=(str(bank["restance"]) if "restance" in bank else None),
               bank_has_residual=(bool(bank["has_residual"]) if "has_residual" in bank else None))
    out["target_source"] = src
    out["restance_scored"] = None if params is None else rst.format_restance(params)
    if "stored_target" in bank:
        out["target_vs_stored_max_abs"] = float(np.abs(np.asarray(bank["target"], dtype=np.float32)
                                                       - np.asarray(bank["stored_target"], dtype=np.float32)).max())
        out["branch_vs_stored_share"] = (float((np.asarray(bank["branch"]) != np.asarray(bank["stored_branch"])).mean())
                                         if "stored_branch" in bank else None)
    out["allhold"] = allhold_bias(mu, bank["target"], bank["branch"], rst)
    out["paths"] = score_paths(bank, sd, esd, w, rst)
    if stand_mask:
        out["masked"] = True                       # NOTE(p2r Z2d, 2026-10-11)
    return out


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--agent-yaml", type=Path, default=None)
    ap.add_argument("--priv", choices=("estimator", "true"), default="estimator")
    ap.add_argument("--no-hist-encoding", action="store_true")
    ap.add_argument("--line", type=float, default=CAM_LINE)
    ap.add_argument("--restance", default=None,
                    help="R's constants (KRABBY_RESTANCE spelling); default the run's params/agent.yaml "
                         "algorithm.restance; neither -> refused")
    ap.add_argument("--stored-target", action="store_true",
                    help="read the bank's stored target / branch (the pre-Z2c behaviour) instead of recomputing R")
    ap.add_argument("--xcheck-stamp", type=Path, default=None)
    ap.add_argument("--xcheck-ckpt", type=Path, default=None)
    ap.add_argument("--stand-mask", action="store_true",
                    help="score the bank through the env's stand-row input mask (after R; NOTE(p2r Z2d, 2026-10-11))")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.stored_target and a.restance is not None:
            raise ValueError("--stored-target and --restance are exclusive")
        out = run(a.bank, a.ckpt, a.agent_yaml, hist_encoding=not a.no_hist_encoding, priv=a.priv, line=a.line,
                  restance=a.restance, stored_target=a.stored_target, stand_mask=a.stand_mask)
        if a.xcheck_stamp is not None:
            xc_ckpt = a.xcheck_ckpt or a.ckpt
            sd, esd = load_ckpt(xc_ckpt)
            out["xcheck"] = dict(xcheck(a.xcheck_stamp, sd, esd, widths(Path(out["agent_yaml"]))), ckpt=str(xc_ckpt))
    except (OSError, KeyError, ValueError) as exc:
        print(f"p2r_z2_fidelity: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return RC_ERROR
    rc = int(out["rc"])
    if out.get("xcheck") is not None and not out["xcheck"]["ok"]:
        rc = RC_ERROR
        out["why"] = f"the replica cross-check FAILS: {out['xcheck']}"
    if a.dry_run:
        out["dry_run"] = True
        out["would_notify"] = rc == RC_NOTIFY
        rc = RC_ERROR if rc == RC_ERROR else RC_OK
    out["exit"] = rc
    text = json.dumps(out, indent=1, default=str)
    if a.json:
        a.json.write_text(text)
    print(text)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
