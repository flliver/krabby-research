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

THE OBSERVATION R READS, rebuilt from the raws (the raws hold joint states, not obs). NOTE(p2r Z2 cams-only,
2026-10-10): R reads NO load signal any more (the kinematic re-step, P2R_Z2_PLAN "Amendment 2026-10-10"), so the
held contact proxy is no longer rebuilt and the +200 held == live line is gone; R's two inputs are
  * cams (obs cols 18+2i): isaaclab's ``wrap_to_pi(joint_pos - default)``, cam default 0 (``hip_to_cam_shaft_default
    (0.0)``);
  * hip-pitch positions (obs cols 29+i): ``joint_pos - default`` of the ``_Hip_Femur_`` joints (float32), the
    default taken from ``run_meta.default_joint_pos`` when a stamp records it, else the linkage's
    ``hip_default_rad()`` (0.02717 rad, the scene cfg's init_state; measured as the obs offset on P2R_PROBE_s18's
    walkstop bank: 0.027169 on all six legs).
  ``joint_pos[t]`` is read before step t, i.e. the state obs_t was built from, so every row (step 0 included) is
  rebuilt. ``--xcheck-stamp`` cross-checks the rebuild on RECORDED obs: a P2R_PROBE_s18 stamp's ``p2r_bank.npz``
  rows (true obs) are matched to their raw (episode, step) by the joint-velocity columns (obs 41..64 = joint_vel *
  0.05, bitwise) and the rebuilt cams / hip positions compared (cams bitwise, hip within ``XCHECK_TOL``; otherwise
  rc 1).

sigma_A: the GRAFT checkpoint's (= S1's) soft std, :func:`anchor_sigma` (a replica of
``ppo_with_extractor.anchor_sigma_from_state``: std_raw -> soft_std; std / log_std clamped to [0.05, 2.0]).

OUTPUT: ``--json`` (the campaign's ``p2r_z2_dose.json`` is what ``p2_live``'s Z2 +200 wiring reads: top-level
``band_lo`` / ``band_hi`` = the union band below) and ``--md``.

REPORTED: per stand row, both losses (debiased mean, SE); x the realised stand share at 0.12 and 0.16 (Z1's +200
measured prior_rows); each share's band [0.52x, 2x] and the union band [0.52 x E(0.12), 2 x E(0.16)], each with
whether it excludes the target-0 values (both shares) and Z1's realised 1.99; R's branch shares on these
training-scale stand rows (beside any ``--probe-stamp``'s own); the share of rows whose R is non-zero; the share of
stand leg-rows DOWN / UP by R's own hip-position thresholds.

THE BC DOSE (NOTE(p2r Z2c, 2026-10-10); P2R_Z2C_PLAN.md step 4 "BC dose"). ``--bc-bank NPZ`` (repeatable: the
chain's KRABBY_BC_BANK banks) adds the clean-state cloning term's expected FIRST-MINIBATCH value: the fork
checkpoint's (``--ckpt``, the GRAFT) TRAINING-PATH action mean (true priv columns, ``hist_encoding=False`` -- the
forward BC runs) on every bank row, against R RECOMPUTED on that row's obs with these constants (``--restance``), per
row ``0.5 * sum_d ((mu_fork - R(obs)) / sigma_A,d)^2`` (sigma_A as above), averaged over the POOLED rows (the BC
minibatch draws rows uniformly from the pooled bank). That mean is ``bc_mb0_expected`` (top level); ``p2_live``'s
Z2c +200 wiring bands the first-update ``Loss/bc_mb0`` at [0.52, 2] x it. Each bank must read the stand bit on every
row (obs col 15 == 1, as the trainer refuses otherwise); the per-bank means, the banks' resolved paths and the
recomputed-vs-stored target difference are reported under ``bc``. The actor's widths come from ``--agent-yaml``
(default: the checkpoint's run dir, else the S1 run's).

Exit codes: 0 = ok; 2 = STOP: R's in-air homing branch ('home') is absent on the bank (P2R_Z2_PLAN step 4);
(``RC_BAND`` = 3 is reserved; band exclusion is reported, not required -- CHARTER 12.1); 1 = unreadable input /
rebuild cross-check failed.

CLI:
    p2r_z2_dose.py [--raws STAMP] [--ckpt GRAFT.pt] [--restance K=V,...] [--labels stand] [--xcheck-stamp DIR|none]
                   [--probe-stamp DIR ...] [--bc-bank NPZ ...] [--agent-yaml PATH] [--json OUT] [--md OUT]
"""
from __future__ import annotations

import argparse
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
#: NOTE(p2r Z2c, 2026-10-10): the stand-bit obs column every BC bank row must read 1 (the trainer's own refusal)
STAND_BIT_COL = 15
BC_CHUNK = 8192


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

        self.names = names
        self.pitch, self.knee, self.cam = pick("_Hip_Femur_"), pick("_Femur_Tibia_"), pick("_Body_CamShaft_")


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """isaaclab.utils.math.wrap_to_pi, op for op."""
    w = (x + torch.pi) % (2 * torch.pi)
    return torch.where((w == 0) & (x > 0), torch.pi, w - torch.pi)


def hip_default(meta: dict, layout: JointLayout, lk) -> tuple[np.ndarray, str]:
    """The hip-pitch defaults the obs subtracts (float32, leg order) and their source: ``run_meta.default_joint_pos``
    (aligned with ``joint_names``) when the stamp records it, else the linkage's ``hip_default_rad()`` on every leg
    (the scene cfg's init_state; NOTE(p2r Z2 cams-only, 2026-10-10))."""
    rec = meta.get("default_joint_pos")
    if rec is not None:
        rec = np.asarray(rec, dtype=np.float32).reshape(-1)
        if rec.shape[0] != len(layout.names):
            raise ValueError(f"run_meta.default_joint_pos has {rec.shape[0]} entries for {len(layout.names)} joints")
        return rec[layout.pitch].astype(np.float32), "run_meta.default_joint_pos"
    return np.full(len(layout.pitch), lk.hip_default_rad(), dtype=np.float32), "linkage.hip_default_rad"


def obs_inputs(jpos, layout: JointLayout, hip_def) -> tuple[torch.Tensor, torch.Tensor]:
    """R's two policy-obs inputs rebuilt from joint positions ``jpos`` (N, 24): ``(cams (N, 6), hip (N, 6))`` --
    ``wrap_to_pi(cam joint_pos - 0)`` and ``hip-pitch joint_pos - default`` (float32, the obs term's own ops)."""
    jp = torch.as_tensor(np.asarray(jpos), dtype=torch.float32)
    cams = wrap_to_pi(jp[:, layout.cam])
    hip = jp[:, layout.pitch] - torch.as_tensor(np.asarray(hip_def, dtype=np.float32)).reshape(1, -1)
    return cams, hip


def policy_obs(cams: torch.Tensor, hip: torch.Tensor, rst) -> torch.Tensor:
    """An (N, MIN_OBS_WIDTH) obs carrying only the columns R reads (cams, hip-pitch positions); every other col 0."""
    obs = torch.zeros(cams.shape[0], rst.MIN_OBS_WIDTH, dtype=torch.float32)
    obs[:, list(rst.CAM_OBS_COLS)] = cams.float()
    obs[:, list(rst.HIP_POS_OBS_COLS)] = hip.float()
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


def collect_stand_rows(stamp: Path, labels, rst, lk, max_episodes: int | None = None) -> dict:
    """Every recorded row of ``stamp`` whose hold label is in ``labels``: the rebuilt cams and hip-pitch positions,
    the recorded action, and the label. Episodes without such a row are skipped unread past ``segment_id``."""
    meta = json.loads((stamp / "run_meta.json").read_text())
    layout = JointLayout(meta["joint_names"], rst.LEGS)
    hip_def, hip_src = hip_default(meta, layout, lk)
    hold_labels = list(meta["hold_labels"])
    want = [i for i, lab in enumerate(hold_labels) if lab in set(labels)]
    if not want:
        raise ValueError(f"{stamp}: none of {list(labels)} in hold_labels {hold_labels}")
    acc = {k_: [] for k_ in ("cams", "hip", "actions", "label")}
    episodes = 0
    for f in sorted((stamp / "raw").glob("episode_*.npz")):
        z = np.load(f)
        seg = np.asarray(z["segment_id"])
        m = np.isin(seg, want)
        if not m.any():
            continue
        if max_episodes is not None and episodes >= max_episodes:
            break
        episodes += 1
        idx = np.nonzero(m)[0]
        cams, hip = obs_inputs(np.asarray(z["joint_pos"])[idx], layout, hip_def)
        acc["cams"].append(cams.numpy())
        acc["hip"].append(hip.numpy())
        acc["actions"].append(np.asarray(z["actions"])[idx])
        acc["label"].append(np.asarray(hold_labels, dtype=object)[seg[idx]])
    if not acc["cams"]:
        raise ValueError(f"{stamp}: no {list(labels)} row")
    out = {k_: np.concatenate(v) for k_, v in acc.items()}
    out.update(meta=meta, episodes=episodes, hip_default=[float(v) for v in hip_def], hip_default_source=hip_src)
    return out


def xcheck(stamp: Path, rst, lk, max_rows: int | None = 5000) -> dict:
    """The rebuild against RECORDED obs: ``stamp``'s ``p2r_bank.npz`` (true obs, knob-off probe) rows matched to
    their raw (episode, step) by obs 41..64 == joint_vel * 0.05 (bitwise); the rebuilt cams vs obs 18+2i and hip-pitch
    positions vs obs 29..34. ``ok`` = cams bitwise and hip within :data:`XCHECK_TOL` on every matched row."""
    bank = np.load(stamp / "p2r_bank.npz")
    obs = np.asarray(bank["obs"])
    if max_rows is not None:
        obs = obs[:max_rows]
    key = {obs[i, OBS_JOINT_VEL].tobytes(): i for i in range(obs.shape[0])}
    meta = json.loads((stamp / "run_meta.json").read_text())
    layout = JointLayout(meta["joint_names"], rst.LEGS)
    hip_def, hip_src = hip_default(meta, layout, lk)
    cam_cols, hip_cols = list(rst.CAM_OBS_COLS), list(rst.HIP_POS_OBS_COLS)
    n_match = n_eq = n_cmp = 0
    max_abs = 0.0
    cams_ok = True
    for f in sorted((stamp / "raw").glob("episode_*.npz")):
        if len(key) == 0:
            break
        z = np.load(f)
        jv = (np.asarray(z["joint_vel"]) * JOINT_VEL_SCALE).astype(np.float32)
        hits = [(t, key.pop(jv[t].tobytes())) for t in range(jv.shape[0]) if jv[t].tobytes() in key]
        if not hits:
            continue
        ts = np.array([t for t, _ in hits])
        rows = np.array([i for _, i in hits])
        cams, hip = obs_inputs(np.asarray(z["joint_pos"])[ts], layout, hip_def)
        got, want = hip.numpy(), obs[rows][:, hip_cols]
        n_match += len(hits)
        n_eq += int((got == want).sum())
        n_cmp += got.size
        max_abs = max(max_abs, float(np.abs(got - want).max()))
        if not np.array_equal(cams.numpy(), obs[rows][:, cam_cols]):
            cams_ok = False
    ok = n_match > 0 and max_abs <= XCHECK_TOL and cams_ok
    return {"stamp": str(stamp), "bank_rows": int(obs.shape[0]), "matched": n_match,
            "hip_bitwise_share": (n_eq / n_cmp if n_cmp else None), "hip_max_abs_diff": max_abs,
            "hip_default_source": hip_src, "cams_bitwise": cams_ok, "tol": XCHECK_TOL, "ok": ok}


def probe_branch_shares(stamp: Path) -> dict | None:
    meta = json.loads((Path(stamp) / "run_meta.json").read_text())
    blk = (meta.get("p2r_mode") or {}).get("restance") or {}
    return blk.get("branch_shares_on_stand")


def dose(stamp: Path, ckpt: Path, params, labels=("stand",), rst=None, lk=None,
         max_episodes: int | None = None) -> dict:
    """The full dose read (see the module docstring)."""
    rst = rst or load_restance()
    lk = lk or load_linkage()
    rows = collect_stand_rows(stamp, labels, rst, lk, max_episodes=max_episodes)
    meta = rows["meta"]
    kind, sigma_s = actions_kind(meta)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)["model_state_dict"]
    sigma_a = anchor_sigma(sd)
    c = 0.0 if sigma_s is None else float(0.5 * ((torch.as_tensor(sigma_s) / sigma_a) ** 2).sum())
    cams = torch.as_tensor(rows["cams"])
    hip = torch.as_tensor(rows["hip"])
    obs = policy_obs(cams, hip, rst)
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
        "episodes": rows["episodes"], "rows": int(a.shape[0]),
        "hip_default": rows["hip_default"], "hip_default_source": rows["hip_default_source"],
        "per_row": {"restance": per_r, "target0": per_0, "restance_raw_mean": float(raw_r.mean()),
                    "target0_raw_mean": float(raw_0.mean()),
                    # paired, row by row (c cancels): how far apart the two targets put the fork's prior
                    "restance_minus_target0": _mean_se(raw_r - raw_0)},
        "branch_shares": rst.branch_shares(branch),
        "restance_nonzero_share": float((tgt.abs() > 1e-6).any(dim=-1).float().mean()),
        # R's own leg phase on these rows (NOTE(p2r Z2 cams-only, 2026-10-10): hip-position DOWN / UP thresholds)
        "leg_down_share": float((hip >= params.p_down).float().mean()),
        "leg_up_share": float((hip <= params.p_up).float().mean()),
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
    # NOTE(p2r Z2 cams-only, 2026-10-10): no held == live line any more (R reads no proxy; p2_live dropped it).
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


def bc_dose(bank_paths, ckpt: Path, params, rst=None, agent_yaml: Path | None = None) -> dict:
    """NOTE(p2r Z2c, 2026-10-10): the BC term's expected first-minibatch value (module docstring "THE BC DOSE")."""
    import p2r_z2_fidelity as fid  # noqa: PLC0415  (same directory: the actor replica)

    rst = rst or load_restance()
    ckpt = Path(ckpt)
    aw = agent_yaml or (ckpt.parent / "params" / "agent.yaml")
    if not Path(aw).is_file():
        aw = fid.S1_AGENT_YAML
    w = fid.widths(aw)
    sd, esd = fid.load_ckpt(ckpt)
    sigma = anchor_sigma(sd)
    banks, rows_all = [], []
    for bp in bank_paths:
        bp = Path(bp)
        d = np.load(bp, allow_pickle=False)
        if "obs" not in d.files:
            raise ValueError(f"{bp}: no 'obs' array (not a stand bank)")
        obs = np.asarray(d["obs"], dtype=np.float32)
        if obs.ndim != 2 or obs.shape[0] == 0:
            raise ValueError(f"{bp}: obs has shape {obs.shape}")
        bad = int((obs[:, STAND_BIT_COL] != 1).sum())
        if bad:
            raise ValueError(f"{bp}: {bad} row(s) do not read the stand bit (obs col {STAND_BIT_COL} != 1)")
        stored = np.asarray(d["target"], dtype=np.float32) if "target" in d.files else None
        per = []
        tdiff = 0.0
        for i in range(0, obs.shape[0], BC_CHUNK):
            o = torch.as_tensor(obs[i:i + BC_CHUNK])
            with torch.no_grad():
                mu = fid.eval_mu(o, sd, esd, w, hist_encoding=False, priv="true")
                tgt, _br = rst.restance_target(o, params)
            per.append(row_losses(mu, tgt, sigma))
            if stored is not None:
                tdiff = max(tdiff, float((tgt - torch.as_tensor(stored[i:i + BC_CHUNK])).abs().max()))
        per = torch.cat(per)
        rows_all.append(per)
        banks.append({"path": str(bp.resolve()), "rows": int(per.numel()), "obs_width": int(obs.shape[1]),
                      "per_row": _mean_se(per),
                      "stored_restance": (str(d["restance"]) if "restance" in d.files else None),
                      "target_vs_stored_max_abs": (tdiff if stored is not None else None)})
    if not banks:
        raise ValueError("no --bc-bank given")
    pooled = _mean_se(torch.cat(rows_all))
    exp = pooled["mean"]
    if exp is None or not math.isfinite(exp) or not exp > 0:
        raise ValueError(f"the pooled BC loss per row {exp!r} is not a finite positive number")
    return {"checkpoint": str(ckpt), "agent_yaml": str(aw), "path": "train (true priv, hist_encoding=False)",
            "restance_knob": rst.format_restance(params), "banks": banks, "rows": int(sum(b["rows"] for b in banks)),
            "per_row_pooled": pooled, "bc_mb0_expected": exp, "bc_mb0_band": [BAND[0] * exp, BAND[1] * exp]}


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
          f"- R non-zero share {out['restance_nonzero_share']:.4f}; legs DOWN {out['leg_down_share']:.4f} / UP "
          f"{out['leg_up_share']:.4f} (hip default from {out['hip_default_source']})", "",
          f"**rc {out['rc']}: {out['verdict']}**", ""]
    bc = out.get("bc")
    if bc:                                         # NOTE(p2r Z2c, 2026-10-10)
        L += ["## BC dose (Z2c)", "",
              f"- fork `{bc['checkpoint']}`, training path; R = `{bc['restance_knob']}`; {bc['rows']} pooled rows",
              f"- **bc_mb0_expected {bc['bc_mb0_expected']:.4f}** (pooled SE {bc['per_row_pooled']['se']}); +200 band "
              f"[{bc['bc_mb0_band'][0]:.3f}, {bc['bc_mb0_band'][1]:.3f}]", "",
              "| bank | rows | BC loss per row | SE | R vs stored max abs |", "|---|---|---|---|---|"]
        for b in bc["banks"]:
            L.append(f"| `{b['path']}` | {b['rows']} | {b['per_row']['mean']:.4f} | {b['per_row']['se']} | "
                     f"{b['target_vs_stored_max_abs']} |")
        L.append("")
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
    ap.add_argument("--bc-bank", type=Path, action="append", default=[],
                    help="a BC training bank (repeatable): adds bc_mb0_expected (NOTE(p2r Z2c, 2026-10-10))")
    ap.add_argument("--agent-yaml", type=Path, default=None, help="the fork's actor widths for the BC dose")
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
    lk = load_linkage()
    xc = None
    if a.xcheck_stamp != "none":
        xs = newest_stamp(XCHECK_RUN, "walkstop__omni") if a.xcheck_stamp == "auto" else Path(a.xcheck_stamp)
        if xs is not None and (Path(xs) / "p2r_bank.npz").is_file():
            xc = xcheck(Path(xs), rst, lk)
            if not xc["ok"]:
                print(f"p2r_z2_dose: the obs rebuild FAILS its cross-check: {json.dumps(xc)}", file=sys.stderr)
                return RC_ERROR
        else:
            xc = {"ok": None, "why": f"no p2r_bank.npz under {xs}"}
    out = dose(Path(raws), Path(ckpt), params, labels=[s for s in a.labels.split(",") if s.strip()], rst=rst, lk=lk,
               max_episodes=a.max_episodes)
    out["xcheck"] = xc
    out["noisy_statue_stamp"] = (str(newest_stamp(NOISY_RAWS_RUN)) if newest_stamp(NOISY_RAWS_RUN) else None)
    out["probe_branch_shares"] = {str(p): probe_branch_shares(p) for p in a.probe_stamp} or None
    if a.bc_bank:
        try:
            out["bc"] = bc_dose(a.bc_bank, Path(ckpt), params, rst=rst, agent_yaml=a.agent_yaml)
        except (OSError, KeyError, ValueError) as exc:
            print(f"p2r_z2_dose: the BC dose fails: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            return RC_ERROR
        out["bc_mb0_expected"] = out["bc"]["bc_mb0_expected"]
    text = json.dumps(out, indent=1, default=str)
    if a.json:
        a.json.write_text(text)
    if a.md:
        a.md.write_text(markdown(out))
    print(text)
    return int(out["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
