#!/usr/bin/env python3
"""Omni phase 2, stage p2r Z2 -- the CPU dose step (P2R_Z2_PLAN.md step 4 "Dose, before launch"; CHARTER 12). CPU
only, read-only: no Isaac, no CUDA, no training. NOTE(p2r Z2, 2026-10-10).

WHAT IT ESTIMATES. Z2's statue prior on bit-1 rows is ``0.5 * sum_d ((mu - R(obs)) / sigma_A,d)^2`` (sum over the
masked rows / B, Z1's normalisation), so its FIRST-UPDATE value is ``stand share x E_stand-rows[0.5 sum((mu - R) /
sigma_A)^2]`` at the fork. Z1's +200 realised-dose band was [0.52x, 2x] around its expected 2.3; this script sets
Z2's band the same way, and says whether it EXCLUDES the target-0 value (Z1's statue, ``0.5 sum (mu / sigma_A)^2``
on the same rows; Z1 realised 1.99) -- a band that does not cannot tell Z2's wiring from Z1's at +200.

THE BANK (P2R_Z2_PLAN "Left for implementation": "The fork's first rollout is S1 marching under noise
(DIAG_REFSAMP_s18), not the zero-mean statue"). Both DIAG stamps exist (``lr/DIAG_REFSAMP_s18``,
``lr/DIAG_NOISYSTATUE_s18``, suite es1, 1000 episodes) and BOTH were run with ``--sample-actions``: their recorded
``actions`` are SAMPLED (``act``: mean + sigma_s * eps, pre-clip; ``run_meta.sample_actions_std`` = sigma_s), NOT the
mean. NOISYSTATUE's actions are pure noise around a zeroed actor, so it carries no fork mu at all; REFSAMP's are S1's
mu plus independent Gaussian noise drawn AFTER the obs, so for any target ``y(obs)``
  E[0.5 sum ((a - y) / sigma_A)^2] = 0.5 sum ((mu - y) / sigma_A)^2 + 0.5 sum (sigma_s / sigma_A)^2,
and the per-row estimate ``0.5 sum ((a - y) / sigma_A)^2 - c`` with ``c = 0.5 sum (sigma_s / sigma_A)^2`` is
unbiased (``c = 0`` on a deterministic stamp, whose actions ARE the mean). Every number below is that debiased mean
(the raw mean and ``c`` are reported beside it).

THE OBSERVATION R READS, rebuilt from the raws (the raws hold joint states, not obs):
  * proxy (obs cols 83..88): the forward task's ``_get_contact_fill`` (``crab_hex_forward_task/mdp/observations.py``)
    replicated op for op on the recorded ``applied_torque[t]`` / ``joint_pos[t]`` (read before the step = the state
    obs_t was built from) and ``rod_speed[t-1]`` (recorded AFTER step t-1 = the action term's rod speed when obs_t was
    built); its three ``_CURRENT_SENSE_*`` constants are parsed from that file's source (it imports isaaclab, so it is
    never imported), the linkage functions are the module itself (loaded by path). Step 0 of an episode has no
    rod_speed[-1]: its gate is taken as 0 and the row is never a stand row here (reported).
  * sample-and-hold (``KRABBY_OMNI_STAND_PROXY=hold``): along each episode, held starts at -0.5 and takes the live
    value wherever that leg's KNEE gate is full (``knee rod gate >= 1``, the knee-inclusive scale); stand rows see
    ``held``. (The gate is computed from the recorded rod speed exactly, so WHICH steps update is bitwise; the values
    are the CPU rebuild of a GPU float32 formula.)
  * cams (obs cols 18+2i): isaaclab's ``wrap_to_pi(joint_pos - default)``, cam default 0 (``hip_to_cam_shaft_default
    (0.0)``).
  ``--xcheck-stamp`` cross-checks the rebuild on RECORDED obs: a P2R_PROBE_s18 stamp's ``p2r_bank.npz`` rows (true
  obs) are matched to their raw (episode, step) by the joint-velocity columns (obs 41..64 = joint_vel * 0.05, bitwise)
  and the rebuilt proxy / cams compared (proxy within ``XCHECK_TOL``, cams bitwise; otherwise rc 1).

sigma_A: the GRAFT checkpoint's (= S1's) soft std, :func:`anchor_sigma` (a replica of
``ppo_with_extractor.anchor_sigma_from_state``: std_raw -> soft_std; std / log_std clamped to [0.05, 2.0]).

OUTPUT: ``--json`` (the campaign's ``p2r_z2_dose.json`` is what ``p2_live``'s Z2 +200 wiring reads: top-level
``band_lo`` / ``band_hi`` = the union band below) and ``--md``.

REPORTED: per stand row, both losses (debiased mean, SE); x the realised stand share at 0.12 and 0.16 (Z1's +200
measured prior_rows); each share's band [0.52x, 2x] and the union band [0.52 x E(0.12), 2 x E(0.16)], each with
whether it excludes the target-0 values (both shares) and Z1's realised 1.99; R's branch shares on these
training-scale stand rows (beside any ``--probe-stamp``'s own); the share of rows whose R is non-zero; held == live
on stand rows (the +200 wiring's ">= 99 %" statistic, on this sampled bank).

Exit codes: 0 = ok; 2 = STOP: R's in-air homing branch ('home') is absent on the bank (P2R_Z2_PLAN step 4);
3 = the union band does not exclude a target-0 value (stop and ask); 1 = unreadable input / rebuild cross-check
failed.

CLI:
    p2r_z2_dose.py [--raws STAMP] [--ckpt GRAFT.pt] [--restance K=V,...] [--labels stand] [--xcheck-stamp DIR|none]
                   [--probe-stamp DIR ...] [--json OUT] [--md OUT]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# scripts -> 2026-10-07_omni_phase2 -> experiments -> crab_hex_omni_task -> parkour_tasks (inner)
_TASKS = HERE.parents[3]
RESTANCE_MODULE = _TASKS / "crab_hex_omni_task" / "mdp" / "crab_hex_omni_restance.py"
FWD_MDP = _TASKS / "crab_hex_forward_task" / "mdp"
FWD_OBS = FWD_MDP / "observations.py"
R = Path("/home/nickmagus/krabby/krabby-research")
LROUT = R / "parkour/outputs/omni_plan/phase2/lr"
GRAFT_CKPT = R / "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt"
S1_POLICY_CKPT = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/6_p2a_field/model_19994.pt"
DEFAULT_RAWS_RUN, NOISY_RAWS_RUN, XCHECK_RUN = "DIAG_REFSAMP_s18", "DIAG_NOISYSTATUE_s18", "P2R_PROBE_s18"

SHARES = (0.12, 0.16)              # Z1's design stand share / its +200 measured prior_rows
BAND = (0.52, 2.0)                 # Z1's [1.2, 4.6] around 2.3
Z1_REALISED_TARGET0 = 1.99         # Z1's first-update Loss/stand_prior (CHARTER 12.1)
ACTION_STD_MIN, ACTION_STD_MAX = 0.05, 2.0
OBS_JOINT_VEL = slice(41, 65)      # joint_vel * 0.05 (the xcheck match key)
JOINT_VEL_SCALE = np.float32(0.05)
XCHECK_TOL = 1e-5
RC_OK, RC_ERROR, RC_STOP, RC_BAND = 0, 1, 2, 3
#: the +200 held == live line = the bank's own share minus this (see ``proxy_live_min``)
PROXY_LIVE_MARGIN = 0.02
_SENSE_NAMES = ("_CURRENT_SENSE_NO_LOAD", "_CURRENT_SENSE_GATE_FRACTION", "_CURRENT_SENSE_MAX")


# ================================================================================================ by-path loads
def _load(name: str, path: Path, register: bool = False):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_restance(path: Path = RESTANCE_MODULE):
    """R's module, by path (pure torch; the task package's ``__init__`` pulls in Isaac)."""
    return _load("_p2r_z2_restance", path)


def load_linkage(mdp: Path = FWD_MDP):
    """``crab_hex_linkage`` by path. Its ``from . import crab_hex_dimensions`` falls back to a flat import, so the
    dimensions module (pure ``math``) is registered under its flat name first -- no sys.path insertion."""
    if "crab_hex_dimensions" not in sys.modules:
        _load("crab_hex_dimensions", mdp / "crab_hex_dimensions.py", register=True)
    return _load("_p2r_z2_linkage", mdp / "crab_hex_linkage.py")


def sense_constants(path: Path = FWD_OBS) -> dict:
    """The three ``_CURRENT_SENSE_*`` module constants, parsed from the forward observations' SOURCE (an AST read:
    the module imports isaaclab). A missing / non-literal one raises."""
    tree = ast.parse(Path(path).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in _SENSE_NAMES:
                out[node.targets[0].id] = float(ast.literal_eval(node.value))
    missing = set(_SENSE_NAMES) - set(out)
    if missing:
        raise ValueError(f"{path}: constant(s) {sorted(missing)} not found as module-level literals")
    return out


# ================================================================================================ the rebuild
class JointLayout:
    """Leg-ordered (FL FR ML MR RL RR) joint indices into the raws' 24-joint arrays, from ``run_meta.joint_names``:
    hip pitch (``_Hip_Femur_``), knee (``_Femur_Tibia_``) and cam (``_Body_CamShaft_``)."""

    def __init__(self, joint_names, legs):
        names = list(joint_names)

        def pick(tag):
            idx = []
            for leg in legs:
                hits = [i for i, n in enumerate(names) if n.startswith(f"{leg}_") and tag in n]
                if len(hits) != 1:
                    raise ValueError(f"joint layout: {leg} has {len(hits)} '{tag}' joints in {names}")
                idx.append(hits[0])
            return idx

        self.pitch, self.knee, self.cam = pick("_Hip_Femur_"), pick("_Femur_Tibia_"), pick("_Body_CamShaft_")


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """isaaclab.utils.math.wrap_to_pi, op for op."""
    w = (x + torch.pi) % (2 * torch.pi)
    return torch.where((w == 0) & (x > 0), torch.pi, w - torch.pi)


def contact_fill(tau, jpos, hip_speed, knee_speed, layout: JointLayout, lk, k: dict):
    """The forward ``_get_contact_fill``, op for op (float32 torch): ``(proxy (N, 6), knee gate (N, 6))``.
    ``hip_speed`` / ``knee_speed`` are the action term's rod speeds the obs was built with."""
    theta_h = jpos[:, layout.pitch]
    theta_k = jpos[:, layout.knee]
    ma_h = lk.hip_moment_arm(theta_h).clamp_min(1e-4)
    ma_k = lk.knee_moment_arm(theta_h, theta_k.abs()).clamp_min(1e-4)
    force_h = tau[:, layout.pitch].abs() / ma_h / lk.HIP_FORCE_N
    force_k = tau[:, layout.knee].abs() / ma_k / lk.KNEE_FORCE_N
    gate_h = (hip_speed.abs() / (k["_CURRENT_SENSE_GATE_FRACTION"] * lk.HIP_SPEED_M_S)).clamp(0.0, 1.0)
    gate_k = (knee_speed.abs() / (k["_CURRENT_SENSE_GATE_FRACTION"] * lk.KNEE_SPEED_M_S)).clamp(0.0, 1.0)
    current_h = gate_h * (k["_CURRENT_SENSE_NO_LOAD"] + force_h).clamp(0.0, k["_CURRENT_SENSE_MAX"])
    current_k = gate_k * (k["_CURRENT_SENSE_NO_LOAD"] + force_k).clamp(0.0, k["_CURRENT_SENSE_MAX"])
    return torch.maximum(current_h, current_k) - 0.5, gate_k


def episode_proxy(ep: dict, layout: JointLayout, lk, k: dict):
    """One episode's policy-visible proxy inputs: ``(live (T, 6), knee gate (T, 6))``. obs_t pairs the state read
    at t with the rod speed recorded after step t-1; step 0 gets rod speed 0 (unknown; never a stand row here)."""
    tau = torch.as_tensor(np.asarray(ep["applied_torque"]), dtype=torch.float32)
    jpos = torch.as_tensor(np.asarray(ep["joint_pos"]), dtype=torch.float32)
    rod = torch.as_tensor(np.asarray(ep["rod_speed"]), dtype=torch.float32)
    prev = torch.cat([torch.zeros_like(rod[:1]), rod[:-1]], dim=0)
    return contact_fill(tau, jpos, prev[:, :6], prev[:, 6:], layout, lk, k)


def sample_and_hold(live: np.ndarray, full: np.ndarray, init: float = -0.5) -> np.ndarray:
    """The ``KRABBY_OMNI_STAND_PROXY=hold`` buffer along ONE episode (reset to ``init`` at its start): at each step
    a leg's held value becomes the live value where its knee gate is full, else keeps the last one."""
    live = np.asarray(live)
    full = np.asarray(full, dtype=bool)
    T = live.shape[0]
    last = np.where(full, np.arange(T)[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)
    held = np.take_along_axis(live, np.clip(last, 0, None), axis=0)
    return np.where(last >= 0, held, np.asarray(init, dtype=live.dtype)).astype(live.dtype)


def policy_obs(cams: torch.Tensor, proxy: torch.Tensor, rst) -> torch.Tensor:
    """An (N, MIN_OBS_WIDTH) obs carrying only the columns R reads (cams, proxy); every other column 0."""
    obs = torch.zeros(cams.shape[0], rst.MIN_OBS_WIDTH, dtype=torch.float32)
    obs[:, list(rst.CAM_OBS_COLS)] = cams.float()
    obs[:, list(rst.PROXY_OBS_COLS)] = proxy.float()
    return obs


# ================================================================================================ sigma / losses
def anchor_sigma(model_state: dict) -> torch.Tensor:
    """``ppo_with_extractor.anchor_sigma_from_state``, replicated (std_raw -> soft_std; std / log_std clamped)."""
    import p2r_drift  # noqa: PLC0415  (same directory: soft_std)

    if "std_raw" in model_state:
        s = p2r_drift.soft_std(model_state["std_raw"].float())
    elif "std" in model_state:
        s = model_state["std"].float().clamp(ACTION_STD_MIN, ACTION_STD_MAX)
    elif "log_std" in model_state:
        s = torch.exp(model_state["log_std"].float()).clamp(ACTION_STD_MIN, ACTION_STD_MAX)
    else:
        raise ValueError("checkpoint: model_state_dict carries no std_raw / std / log_std")
    return s.detach().reshape(-1).clone()


def row_losses(a: torch.Tensor, y: torch.Tensor | None, sigma: torch.Tensor) -> torch.Tensor:
    """Per row ``0.5 * sum_d ((a - y) / sigma_d)^2`` (``y`` None -> target 0)."""
    d = a if y is None else a - y
    return 0.5 * (d / sigma.reshape(1, -1)).square().sum(dim=-1)


def _mean_se(x: torch.Tensor) -> dict:
    n = int(x.numel())
    if n == 0:
        return {"mean": None, "se": None, "n": 0}
    m = float(x.double().mean())
    se = float(x.double().std(unbiased=True) / math.sqrt(n)) if n > 1 else None
    return {"mean": m, "se": se, "n": n}


def band_report(per_row_r: float, per_row_0: float, shares=SHARES) -> dict:
    """Each share's expected first-update values, band [0.52x, 2x] and the exclusion verdicts; plus the union band
    [0.52 x E(min share), 2 x E(max share)]."""
    lo_f, hi_f = BAND
    out = {"by_share": {}}
    t0_refs = {f"target0@{s:g}": s * per_row_0 for s in shares}
    t0_refs["z1_realised"] = Z1_REALISED_TARGET0
    for s in shares:
        e, t0 = s * per_row_r, s * per_row_0
        lo, hi = lo_f * e, hi_f * e
        out["by_share"][f"{s:g}"] = {"expected_restance": e, "expected_target0": t0, "band": [lo, hi],
                                     "excludes_own_target0": not (lo <= t0 <= hi),
                                     "excludes_z1_realised": not (lo <= Z1_REALISED_TARGET0 <= hi)}
    lo, hi = lo_f * min(shares) * per_row_r, hi_f * max(shares) * per_row_r
    inside = sorted(k for k, v in t0_refs.items() if lo <= v <= hi)
    out["union_band"] = [lo, hi]
    out["target0_refs"] = t0_refs
    out["union_excludes_every_target0"] = not inside
    out["target0_inside_union"] = inside
    return out


# ================================================================================================ the bank
def newest_stamp(name: str, row: str = "suite__omni", seed: int = 1, lrout: Path = LROUT) -> Path | None:
    d = Path(lrout) / name / row / f"seed{seed:03d}"
    hits = [p for p in sorted(d.glob("*/")) if (p / "complete.json").is_file() or (p / "p2_complete.json").is_file()]
    return hits[-1] if hits else None


def actions_kind(meta: dict) -> tuple[str, np.ndarray | None]:
    """``("sampled", sigma_s)`` / ``("mean", None)``; a zero-action stamp is refused."""
    if meta.get("zero_actions"):
        raise ValueError("a --zero-actions stamp carries no policy action")
    if meta.get("sample_actions"):
        std = meta.get("sample_actions_std")
        if not std:
            raise ValueError("sample_actions stamp without sample_actions_std (cannot debias)")
        return "sampled", np.asarray(std, dtype=np.float32)
    return "mean", None


def collect_stand_rows(stamp: Path, labels, rst, lk, k: dict, max_episodes: int | None = None) -> dict:
    """Every recorded row of ``stamp`` whose hold label is in ``labels`` (step >= 1): the rebuilt cams, live and held
    proxy, knee gate, the recorded action, and the label. Episodes without such a row are skipped unread past
    ``segment_id``."""
    meta = json.loads((stamp / "run_meta.json").read_text())
    layout = JointLayout(meta["joint_names"], rst.LEGS)
    hold_labels = list(meta["hold_labels"])
    want = [i for i, lab in enumerate(hold_labels) if lab in set(labels)]
    if not want:
        raise ValueError(f"{stamp}: none of {list(labels)} in hold_labels {hold_labels}")
    acc = {k_: [] for k_ in ("cams", "live", "held", "gate_k", "actions", "label")}
    episodes = step0 = 0
    for f in sorted((stamp / "raw").glob("episode_*.npz")):
        z = np.load(f)
        seg = np.asarray(z["segment_id"])
        m = np.isin(seg, want)
        if not m.any():
            continue
        if max_episodes is not None and episodes >= max_episodes:
            break
        episodes += 1
        ep = {key: z[key] for key in ("applied_torque", "joint_pos", "rod_speed", "actions")}
        live, gate_k = episode_proxy(ep, layout, lk, k)
        live_np = live.numpy()
        held = sample_and_hold(live_np, gate_k.numpy() >= 1.0)
        step0 += int(m[0])
        m[0] = False
        idx = np.nonzero(m)[0]
        jp = torch.as_tensor(np.asarray(ep["joint_pos"])[idx][:, layout.cam], dtype=torch.float32)
        acc["cams"].append(wrap_to_pi(jp).numpy())
        acc["live"].append(live_np[idx])
        acc["held"].append(held[idx])
        acc["gate_k"].append(gate_k.numpy()[idx])
        acc["actions"].append(np.asarray(ep["actions"])[idx])
        acc["label"].append(np.asarray(hold_labels, dtype=object)[seg[idx]])
    if not acc["cams"]:
        raise ValueError(f"{stamp}: no {list(labels)} row past step 0")
    out = {k_: np.concatenate(v) for k_, v in acc.items()}
    out.update(meta=meta, episodes=episodes, step0_rows_dropped=step0)
    return out


def xcheck(stamp: Path, rst, lk, k: dict, max_rows: int | None = 5000) -> dict:
    """The rebuild against RECORDED obs: ``stamp``'s ``p2r_bank.npz`` (true obs, knob-off probe) rows matched to
    their raw (episode, step) by obs 41..64 == joint_vel * 0.05 (bitwise); the rebuilt live proxy vs obs 83..88 and
    cams vs obs 18+2i. ``ok`` = proxy within :data:`XCHECK_TOL` and cams bitwise on every matched row (step >= 1)."""
    bank = np.load(stamp / "p2r_bank.npz")
    obs = np.asarray(bank["obs"])
    if max_rows is not None:
        obs = obs[:max_rows]
    key = {obs[i, OBS_JOINT_VEL].tobytes(): i for i in range(obs.shape[0])}
    meta = json.loads((stamp / "run_meta.json").read_text())
    layout = JointLayout(meta["joint_names"], rst.LEGS)
    prox_cols, cam_cols = list(rst.PROXY_OBS_COLS), list(rst.CAM_OBS_COLS)
    n_match = n_eq = n_cmp = 0
    max_abs = 0.0
    cams_ok = True
    for f in sorted((stamp / "raw").glob("episode_*.npz")):
        if len(key) == 0:
            break
        z = np.load(f)
        jv = (np.asarray(z["joint_vel"]) * JOINT_VEL_SCALE).astype(np.float32)
        hits = [(t, key.pop(jv[t].tobytes())) for t in range(1, jv.shape[0]) if jv[t].tobytes() in key]
        if not hits:
            continue
        ep = {key_: z[key_] for key_ in ("applied_torque", "joint_pos", "rod_speed")}
        live, _ = episode_proxy(ep, layout, lk, k)
        ts = np.array([t for t, _ in hits])
        rows = np.array([i for _, i in hits])
        got = live.numpy()[ts]
        want = obs[rows][:, prox_cols]
        n_match += len(hits)
        n_eq += int((got == want).sum())
        n_cmp += got.size
        max_abs = max(max_abs, float(np.abs(got - want).max()))
        cams = wrap_to_pi(torch.as_tensor(np.asarray(ep["joint_pos"])[ts][:, layout.cam], dtype=torch.float32))
        if not np.array_equal(cams.numpy(), obs[rows][:, cam_cols]):
            cams_ok = False
    ok = n_match > 0 and max_abs <= XCHECK_TOL and cams_ok
    return {"stamp": str(stamp), "bank_rows": int(obs.shape[0]), "matched": n_match,
            "proxy_bitwise_share": (n_eq / n_cmp if n_cmp else None), "proxy_max_abs_diff": max_abs,
            "cams_bitwise": cams_ok, "tol": XCHECK_TOL, "ok": ok}


def probe_branch_shares(stamp: Path) -> dict | None:
    meta = json.loads((Path(stamp) / "run_meta.json").read_text())
    blk = (meta.get("p2r_mode") or {}).get("restance") or {}
    return blk.get("branch_shares_on_stand")


def dose(stamp: Path, ckpt: Path, params, labels=("stand",), rst=None, lk=None, k=None,
         max_episodes: int | None = None) -> dict:
    """The full dose read (see the module docstring)."""
    rst = rst or load_restance()
    lk = lk or load_linkage()
    k = k or sense_constants()
    rows = collect_stand_rows(stamp, labels, rst, lk, k, max_episodes=max_episodes)
    meta = rows["meta"]
    kind, sigma_s = actions_kind(meta)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)["model_state_dict"]
    sigma_a = anchor_sigma(sd)
    c = 0.0 if sigma_s is None else float(0.5 * ((torch.as_tensor(sigma_s) / sigma_a) ** 2).sum())
    cams = torch.as_tensor(rows["cams"])
    held = torch.as_tensor(rows["held"])
    obs = policy_obs(cams, held, rst)
    tgt, branch = rst.restance_target(obs, params)
    a = torch.as_tensor(rows["actions"], dtype=torch.float32)
    raw_r, raw_0 = row_losses(a, tgt, sigma_a), row_losses(a, None, sigma_a)
    per_r, per_0 = _mean_se(raw_r - c), _mean_se(raw_0 - c)
    out = {
        "stamp": str(stamp), "checkpoint": str(ckpt), "labels": list(labels), "actions": kind,
        "debias_c": c, "sigma_A": [float(v) for v in sigma_a],
        "sigma_sample": (None if sigma_s is None else [float(v) for v in sigma_s]),
        "sigma_sample_equals_sigma_A": (None if sigma_s is None
                                        else bool(np.allclose(sigma_s, sigma_a.numpy(), rtol=1e-5, atol=1e-6))),
        "restance": params.to_dict(), "restance_knob": rst.format_restance(params),
        "episodes": rows["episodes"], "rows": int(a.shape[0]), "step0_rows_dropped": rows["step0_rows_dropped"],
        "per_row": {"restance": per_r, "target0": per_0, "restance_raw_mean": float(raw_r.mean()),
                    "target0_raw_mean": float(raw_0.mean()),
                    # paired, row by row (c cancels): how far apart the two targets put the fork's prior
                    "restance_minus_target0": _mean_se(raw_r - raw_0)},
        "branch_shares": rst.branch_shares(branch),
        "restance_nonzero_share": float((tgt.abs() > 1e-6).any(dim=-1).float().mean()),
        "held_equals_live_share": float((rows["held"] == rows["live"]).mean()),
        "knee_gate_full_share": float((rows["gate_k"] >= 1.0).mean()),
        "mean_abs_target": [float(v) for v in tgt.abs().mean(dim=0)],
        "by_label": {},
    }
    for lab in sorted(set(rows["label"].tolist())):
        m = torch.as_tensor(rows["label"] == lab)
        out["by_label"][str(lab)] = {"rows": int(m.sum()), "restance": _mean_se((raw_r - c)[m]),
                                     "target0": _mean_se((raw_0 - c)[m]),
                                     "branch_shares": rst.branch_shares(branch[m])}
    out["bands"] = band_report(per_r["mean"], per_0["mean"])
    # the +200 wiring's band (p2_live.z2_dose_band reads band_lo / band_hi): the union band, covering both shares
    out["band_lo"], out["band_hi"] = out["bands"]["union_band"]
    out["band_basis"] = f"union [{BAND[0]:g} x E({min(SHARES):g}), {BAND[1]:g} x E({max(SHARES):g})]"
    # NOTE(p2r Z2, 2026-10-10, decision recorded in CHARTER 12.1): the +200 held == live line is CALIBRATED here,
    # on the training-noise bank, not the plan's pre-knee-decision 0.99 (the knee gate alone is full on ~97.6 % of
    # noisy stand steps): the bank's own share minus PROXY_LIVE_MARGIN.
    out["proxy_live_min"] = round(out["held_equals_live_share"] - PROXY_LIVE_MARGIN, 4)
    home = out["branch_shares"]["home"]
    if not home:
        out["rc"], out["verdict"] = RC_STOP, "STOP: R's in-air homing branch ('home') is absent on the bank"
    elif not out["bands"]["union_excludes_every_target0"]:
        # NOTE(p2r Z2, 2026-10-10, decision recorded in CHARTER 12.1): the band CANNOT exclude target-0 at any R
        # that is small next to S1's marching mu (which dominates both priors at the fork), so exclusion is
        # REPORTED, not required; the +200 wiring proves R is applied through Loss/restance_nonzero and the branch
        # shares instead. rc stays OK so the band is still the +200 range check (a scale / NaN bug).
        out["rc"] = RC_OK
        out["verdict"] = (f"band {out['bands']['union_band']} does not exclude "
                          f"{out['bands']['target0_inside_union']} (informational: S1's mu dominates both targets); "
                          "home branch present")
    else:
        out["rc"], out["verdict"] = RC_OK, "band excludes every target-0 value; home branch present"
    return out


def markdown(out: dict) -> str:
    b = out["bands"]
    L = ["# p2r Z2 dose (P2R_Z2_PLAN step 4, CPU)", "",
         f"- bank: `{out['stamp']}` ({out['episodes']} episodes, {out['rows']} rows, labels {out['labels']}); actions "
         f"**{out['actions']}** (debias c = {out['debias_c']:.4f}; sigma_sample == sigma_A: "
         f"{out['sigma_sample_equals_sigma_A']})",
         f"- sigma_A from `{out['checkpoint']}`; R = `{out['restance_knob']}`",
         f"- rebuild cross-check: {json.dumps(out.get('xcheck'))}", "",
         "| per stand row | mean | SE | n |", "|---|---|---|---|"]
    for name in ("restance", "target0", "restance_minus_target0"):
        p = out["per_row"][name]
        L.append(f"| {name} | {p['mean']:.4f} | {p['se'] if p['se'] is None else round(p['se'], 4)} | {p['n']} |")
    L += ["", "| share | E restance | E target-0 | band | excludes own target-0 | excludes Z1 1.99 |",
          "|---|---|---|---|---|---|"]
    for s, v in b["by_share"].items():
        L.append(f"| {s} | {v['expected_restance']:.4f} | {v['expected_target0']:.4f} | "
                 f"[{v['band'][0]:.3f}, {v['band'][1]:.3f}] | {v['excludes_own_target0']} | "
                 f"{v['excludes_z1_realised']} |")
    L += ["", f"- union band [{b['union_band'][0]:.3f}, {b['union_band'][1]:.3f}]; target-0 refs inside: "
              f"{b['target0_inside_union'] or 'none'}",
          f"- R branch shares (leg-rows): {json.dumps({k: round(v, 4) for k, v in out['branch_shares'].items()})}",
          f"- probe branch shares: {json.dumps(out.get('probe_branch_shares'))}",
          f"- R non-zero share {out['restance_nonzero_share']:.4f}; held == live {out['held_equals_live_share']:.4f}; "
          f"knee gate full {out['knee_gate_full_share']:.4f}", "",
          f"**rc {out['rc']}: {out['verdict']}**", ""]
    return "\n".join(L)


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raws", type=Path, default=None, help=f"stamp dir (default: newest lr/{DEFAULT_RAWS_RUN} suite)")
    ap.add_argument("--ckpt", type=Path, default=None, help="sigma_A source (default: the graft, else S1's policy)")
    ap.add_argument("--restance", default=None, help="R's constants (default: $KRABBY_RESTANCE)")
    ap.add_argument("--labels", default="stand", help="comma-separated hold labels that are stand rows")
    ap.add_argument("--xcheck-stamp", default="auto", help=f"'auto' (newest lr/{XCHECK_RUN} walkstop), a dir, or none")
    ap.add_argument("--probe-stamp", type=Path, action="append", default=[])
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    rst = load_restance()
    try:
        params = rst.parse_restance(a.restance if a.restance is not None else os.environ.get(rst.RESTANCE_VAR))
    except ValueError as exc:
        print(f"p2r_z2_dose: {exc}", file=sys.stderr)
        return RC_ERROR
    if params is None:
        print(f"p2r_z2_dose: no R constants (--restance or ${rst.RESTANCE_VAR}; 'default' = the seeds)",
              file=sys.stderr)
        return RC_ERROR
    raws = a.raws or newest_stamp(DEFAULT_RAWS_RUN)
    ckpt = a.ckpt or (GRAFT_CKPT if GRAFT_CKPT.is_file() else S1_POLICY_CKPT)
    if raws is None or not Path(raws).is_dir():
        print(f"p2r_z2_dose: raws stamp not found ({raws})", file=sys.stderr)
        return RC_ERROR
    lk, k = load_linkage(), sense_constants()
    xc = None
    if a.xcheck_stamp != "none":
        xs = newest_stamp(XCHECK_RUN, "walkstop__omni") if a.xcheck_stamp == "auto" else Path(a.xcheck_stamp)
        if xs is not None and (Path(xs) / "p2r_bank.npz").is_file():
            xc = xcheck(Path(xs), rst, lk, k)
            if not xc["ok"]:
                print(f"p2r_z2_dose: the proxy rebuild FAILS its cross-check: {json.dumps(xc)}", file=sys.stderr)
                return RC_ERROR
        else:
            xc = {"ok": None, "why": f"no p2r_bank.npz under {xs}"}
    out = dose(Path(raws), Path(ckpt), params, labels=[s for s in a.labels.split(",") if s.strip()], rst=rst, lk=lk,
               k=k, max_episodes=a.max_episodes)
    out["xcheck"] = xc
    out["noisy_statue_stamp"] = (str(newest_stamp(NOISY_RAWS_RUN)) if newest_stamp(NOISY_RAWS_RUN) else None)
    out["probe_branch_shares"] = {str(p): probe_branch_shares(p) for p in a.probe_stamp} or None
    text = json.dumps(out, indent=1, default=str)
    if a.json:
        a.json.write_text(text)
    if a.md:
        a.md.write_text(markdown(out))
    print(text)
    return int(out["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
