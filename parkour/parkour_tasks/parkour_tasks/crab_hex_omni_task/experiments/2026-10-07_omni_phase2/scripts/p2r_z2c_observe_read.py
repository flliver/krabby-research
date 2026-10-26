#!/usr/bin/env python3
"""Z2c diagnosis (2026-10-10): read the closed-loop observe banks of P2__Z2c_s18 model_24993.

For each observe bank (suite es1, walkstop es1): R's branch shares on the closed-loop stand states (vs the probe's
own banks), the applied action mu - R by branch and dim group, mu - R on all-HOLD rows, the CPU eval-path mu on the
banked obs vs the recorded applied mu (the scorer reproduces the harness), and the distance of the closed-loop obs
from the training banks on the columns R and the actor read (rod positions, proxy, last action, body height proxy).

ANY OBSERVE RUN (NOTE(p2r Z2d, 2026-10-11); P2R_Z2D_PLAN.md step 1 "The observe read" / step 2b): ``--name`` (the
lr/<NAME> observe dir, default P2R_OBSZ2c_s18), ``--ckpt`` (default Z2c's model_24993), ``--restance`` (R's constants,
default Z2c's) and ``--stand-mask``: the head ran MASKED, so every banked STAND row (obs col 15 == 1) must read the
mask constants in columns 65..88 EXACTLY (last action 0, proxy -0.5: the closed-loop proof the mask was live) -- else
the read is REFUSED (exit 3, nothing scored) -- and the CPU eval-path mu is computed on the bank's obs through
``crab_hex_omni_restance.stand_mask_obs`` on the stand rows (the env's own mask; idempotent on a proven bank). Each
row's JSON block carries ``masked``. Without ``--stand-mask`` everything reads as before.

CLI: p2r_z2c_observe_read.py [OUT.json] [--name NAME] [--ckpt MODEL.pt] [--restance K=V,...] [--stand-mask]
Exit codes: 0 read; 3 REFUSED (``--stand-mask`` and a stand row not masked).
"""
import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(16)
REPO = Path(__file__).resolve().parents[7]  # this checkout's root (scripts -> campaign -> experiments -> ... -> repo)
C = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
spec = importlib.util.spec_from_file_location("_fid", str(C / "scripts/p2r_z2_fidelity.py"))
fid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fid)
rst = fid.load_restance()
W = fid.widths(fid.S1_AGENT_YAML)
CK = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-10_19-01-45_seed18_X18_p2r_A15pB/model_24993.pt"
LR = REPO / "parkour/outputs/omni_plan/phase2/lr"
#: Z2c's R constants (the --restance default)
Z2C_RESTANCE = ("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,"
                "eps_step=0.3")
PARAMS = rst.parse_restance(Z2C_RESTANCE)
#: NOTE(p2r Z2d, 2026-10-11): the observe run read by default, the stand-bit column, the refusal's exit code
DEFAULT_NAME = "P2R_OBSZ2c_s18"
STAND_BIT_COL = 15
RC_OK, RC_REFUSED = 0, 3
GROUPS = {"cam": rst.CAM_ACT, "hip": rst.HIP_ACT, "knee": rst.KNEE_ACT}
COLS = {"hip_pos(29:35)": slice(29, 35), "cam(18:30:2)": slice(18, 30, 2), "proxy(83:89)": slice(83, 89),
        "last_act_rods(71:83)": slice(71, 83), "roll_pitch(3:5)": slice(3, 5), "joint_vel(41:65)": slice(41, 65)}


def load(p):
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def mask_proof(obs) -> dict:
    """NOTE(p2r Z2d, 2026-10-11): do the bank's STAND rows (obs col 15 == 1) read the mask constants EXACTLY in every
    column the mask writes (:func:`stand_mask_columns`: 65..88, plus each real history frame's on a full-width row)? ``ok``
    is False with no stand row at all (nothing proves the mask)."""
    obs = np.asarray(obs)
    stand = obs[:, STAND_BIT_COL] == 1
    # every column the mask writes at this width: the current block and, full-width, the 10 real history frames
    cols, vals = rst.stand_mask_columns(obs.shape[1])
    want = np.asarray(vals, dtype=obs.dtype)
    bad = int((obs[stand][:, cols] != want[None, :]).any(axis=1).sum())
    return {"stand_rows": int(stand.sum()), "unmasked_stand_rows": bad, "ok": bool(stand.any()) and bad == 0}


def summarize(name, b, sd, esd, ref_banks, params=None, stand_mask=False, w=None):
    params = PARAMS if params is None else params
    w = W if w is None else w
    obs = torch.as_tensor(b["obs"])
    tgt, br = rst.restance_target(obs, params)
    br = br.numpy()
    mu_app = torch.as_tensor(b["mu"])
    # NOTE(p2r Z2d, 2026-10-11): a masked head is scored through the env's own mask on its stand rows
    obs_in = rst.stand_mask_obs(obs, obs[:, STAND_BIT_COL] == 1) if stand_mask else obs
    with torch.no_grad():
        mu_cpu = fid.eval_mu(obs_in, sd, esd, w)
    out = {"rows": int(obs.shape[0]), "masked": bool(stand_mask),
           "cpu_vs_applied_mu_max_abs": float((mu_cpu - mu_app).abs().max()),
           "cpu_vs_applied_mu_mean_abs": float((mu_cpu - mu_app).abs().mean())}
    shares = {nm: float((br == k).mean()) for k, nm in enumerate(rst.BRANCH_NAMES)}
    out["branch_leg_shares"] = shares
    ah = (br == 0).all(1)
    out["allhold_share"] = float(ah.mean())
    d = (mu_app - tgt).numpy()
    out["allhold_mean_mu_minus_R"] = {g: d[ah][:, s].mean(0).round(4).tolist() for g, s in GROUPS.items()} if ah.any() else None
    byb = {}
    for k, nm in enumerate(rst.BRANCH_NAMES):
        m = br == k
        if not m.any():
            continue
        byb[nm] = {"leg_rows": int(m.sum())}
        for g, s in GROUPS.items():
            byb[nm][g + "_abs"] = round(float(np.abs(d[:, s])[m].mean()), 4)
            byb[nm][g + "_signed"] = round(float(d[:, s][m].mean()), 4)
    out["by_branch"] = byb
    # how far the closed-loop states are from the training banks, per column group (mean / p90 of |x|, both sides)
    dist = {}
    for cn, sl in COLS.items():
        a = b["obs"][:, sl]
        if not ref_banks:                          # NOTE(p2r Z2d, 2026-10-11): no training bank to compare with
            dist[cn] = {"observe_mean_abs": round(float(np.abs(a).mean()), 4),
                        "observe_p90_abs": round(float(np.quantile(np.abs(a), 0.9)), 4),
                        "train_mean_abs": None, "train_p90_abs": None}
            continue
        r = np.concatenate([rb["obs"][:, sl] for rb in ref_banks])
        dist[cn] = {"observe_mean_abs": round(float(np.abs(a).mean()), 4), "observe_p90_abs": round(float(np.quantile(np.abs(a), 0.9)), 4),
                    "train_mean_abs": round(float(np.abs(r).mean()), 4), "train_p90_abs": round(float(np.quantile(np.abs(r), 0.9)), 4)}
    out["column_scale_vs_train_banks"] = dist
    out["proxy_parked_share"] = {"observe": float((np.abs(b["obs"][:, 83:89] + 0.5) < 1e-4).mean()),
                                 "train": (float(np.mean([(np.abs(rb["obs"][:, 83:89] + 0.5) < 1e-4).mean()
                                                          for rb in ref_banks])) if ref_banks else None)}
    # per step since episode start (suite statue from spawn): when does the hip drift begin
    t = b["t"]
    prof = {}
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 10 ** 6)):
        m = (t >= lo) & (t < hi)
        if m.sum():
            prof[f"{lo}-{hi}"] = {"n": int(m.sum()), "mu_hip_mean": round(float(b["mu"][m][:, 6:12].mean()), 4),
                                  "R_hold_leg_share": round(float((br[m] == 0).mean()), 3),
                                  "R_lift_or_seat_share": round(float(((br[m] == 2) | (br[m] == 4)).mean()), 3)}
    out["by_step"] = prof
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", type=Path, default=None, help="also write the JSON here")
    # NOTE(p2r Z2d, 2026-10-11): any observe run
    ap.add_argument("--name", default=DEFAULT_NAME, help=f"the lr/<NAME> observe dir (default {DEFAULT_NAME})")
    ap.add_argument("--ckpt", type=Path, default=CK, help="the observed checkpoint (default Z2c's model_24993)")
    ap.add_argument("--restance", default=Z2C_RESTANCE, help="R's constants (default Z2c's)")
    ap.add_argument("--stand-mask", action="store_true",
                    help="the head ran masked: refuse a bank whose stand rows' cols 65..88 are not the mask "
                         "constants, and score through the mask")
    ap.add_argument("--lr", type=Path, default=LR, help=argparse.SUPPRESS)
    ap.add_argument("--agent-yaml", type=Path, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    params = rst.parse_restance(a.restance)
    if params is None:
        print(f"p2r_z2c_observe_read: --restance {a.restance!r} reads OFF", file=sys.stderr)
        return 1
    w = fid.widths(a.agent_yaml) if a.agent_yaml else W
    banks = {}
    for row in ("suite", "walkstop"):
        ps = sorted(glob.glob(str(a.lr / f"{a.name}/{row}__omni/seed001/*/p2r_stand_bank.npz")))
        if not ps:
            print(row, "no observe bank yet")
            continue
        b = load(ps[-1])
        assert bool(b.get("observe", False)), ps[-1]
        if a.stand_mask:
            proof = mask_proof(b["obs"])
            if not proof["ok"]:
                print(f"p2r_z2c_observe_read: REFUSED -- {ps[-1]}: {proof['unmasked_stand_rows']} of "
                      f"{proof['stand_rows']} stand row(s) do not read the mask constants in cols 65..88 (the mask "
                      "was not live in the closed loop)", file=sys.stderr)
                return RC_REFUSED
        banks[row] = (ps[-1], b)
    sd, esd = fid.load_ckpt(a.ckpt)
    train = [load(p) for p in sorted(glob.glob(str(a.lr / "P2R_RST*g_s18/*__omni/seed001/*/p2r_stand_bank.npz")))]
    res = {}
    for row, (path, b) in banks.items():
        res[row] = summarize(row, b, sd, esd, train, params=params, stand_mask=a.stand_mask, w=w)
        res[row]["bank"] = path
    print(json.dumps(res, indent=1))
    if a.out is not None:
        a.out.write_text(json.dumps(res, indent=1))
    return RC_OK


if __name__ == "__main__":
    raise SystemExit(main())
