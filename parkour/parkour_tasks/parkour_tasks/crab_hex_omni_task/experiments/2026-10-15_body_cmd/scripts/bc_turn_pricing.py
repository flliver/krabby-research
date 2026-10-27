#!/usr/bin/env python3
"""Step 1a (CPU): price the body-mode yaw-term set on S1's measured per-step yaw-rate traces.

PLAN D4 "turn-pricing check" + CHARTER section 3 (rule pinned 2026-10-15 before this read). For each
command level c, three traces per S1 hold:

* healthy tracker  = wobble + c   (the measured trace with its mean removed, shifted to the command)
* step-in-place    = wobble       (mean removed, not shifted: steps, does not rotate)
* still stander    = 0

The yaw-term income per step is 0.2 * exp(-|dyaw_1s - c*1s| / sigma) + 1.0 * tent(p, c), with p = sign(c) *
dyaw_1s / 1 s and tent(p, c) = min(p, 2|c| - p); track_ang_vel_z_exp pays 0 on turning rows in body mode. The
1-s window needs 50 samples, so only steps with a full window are scored.

    bc_turn_pricing.py [--stamp DIR] [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

R = Path("/home/nickmagus/krabby/krabby-research")
HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
STAMP = R / "parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24"

W_TRACK, W_TENT, W_ANG = 0.2, 1.0, 1.0          # S1 p2a realised weights (CHARTER section 3)
SIGMA = 0.05                                     # rad (= w_min x the 1-s window)
ANG_STD = 0.5                                    # track_ang_vel_z_exp std (straight rows only)
WIN_S = 1.0
SKIP_S = 2.0                                     # first 2 s of each hold are not scored
LEVELS = {"w_min": 0.05, "0.08": 0.08, "ceiling": 0.167, "w_max": 0.21}
B_WMIN = 0.40                                    # pinned before the read (CHARTER section 3)
CEIL_SHARE = 0.10
TRANSLATE = ("fwd", "lat_l", "lat_r", "back", "diag")
MOVE_TURN = ("move_turn", "move_turn_r")


def hold_traces(stamp: Path) -> tuple[dict[str, list[np.ndarray]], float]:
    meta = json.loads((stamp / "run_meta.json").read_text())
    dt = float(meta["dt"])
    out: dict[str, list[np.ndarray]] = {k: [] for k in TRANSLATE + MOVE_TURN}
    for row in meta["env_index_map"]:
        label = row["label"]
        if label not in out:
            continue
        z = np.load(stamp / "raw" / f"episode_{row['env_index']:02d}.npz")
        seg = z["segment_id"]
        hold = seg >= 1                                   # -1 pre-roll, 0 settle, >= 1 the hold
        if z["done"][hold].any():
            continue                                      # a terminated hold is not a healthy trace
        wz = z["root_ang_vel_b"][hold, 2].astype(np.float64)
        wz = wz[int(round(SKIP_S / dt)):]
        out[label].append(wz)
    return out, dt


def window_sum(x: np.ndarray, n: int, dt: float) -> np.ndarray:
    """Delta-yaw over each full trailing window of n samples (rad)."""
    c = np.concatenate([[0.0], np.cumsum(x)]) * dt
    return c[n:] - c[:-n]


def income(dyaw: np.ndarray, c: float) -> np.ndarray:
    """Per-step yaw-term income on a turning row commanding c (rad/s), from the window integrals."""
    win = W_TRACK * np.exp(-np.abs(dyaw - c * WIN_S) / SIGMA)
    p = math.copysign(1.0, c) * dyaw / WIN_S
    tent = np.minimum(p, 2.0 * abs(c) - p)
    return win + W_TENT * tent


def straight_income(wz: np.ndarray, dyaw: np.ndarray, n: int) -> np.ndarray:
    """c = 0 rows: heading-hold window term + track_ang_vel_z_exp (tent pays 0 at c = 0)."""
    return W_TRACK * np.exp(-np.abs(dyaw) / SIGMA) + W_ANG * np.exp(-np.square(wz[n - 1:]) / ANG_STD ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default=str(STAMP))
    ap.add_argument("--json", default=str(CAMPAIGN / "step1a_turn_pricing.json"))
    ap.add_argument("--md", default=str(CAMPAIGN / "step1a_turn_pricing.md"))
    ap.add_argument("--win", type=float, default=None, help="window seconds (default WIN_S)")
    ap.add_argument("--sigma", type=float, default=None, help="window kernel width, rad (default SIGMA)")
    ap.add_argument("--w-tent", type=float, default=None, help="tent weight (default W_TENT)")
    a = ap.parse_args()
    global WIN_S, SIGMA, W_TENT
    WIN_S = WIN_S if a.win is None else a.win
    SIGMA = SIGMA if a.sigma is None else a.sigma
    W_TENT = W_TENT if a.w_tent is None else a.w_tent
    stamp = Path(a.stamp)
    traces, dt = hold_traces(stamp)
    n = int(round(WIN_S / dt))

    res: dict = {"stamp": str(stamp), "dt": dt, "window_steps": n, "window_s": WIN_S, "sigma": SIGMA,
                 "weights": {"track": W_TRACK, "tent": W_TENT, "ang": W_ANG}, "levels": LEVELS,
                 "B_wmin": B_WMIN, "ceil_share_max": CEIL_SHARE, "families": {}}
    ideal = {k: (W_TRACK * math.exp(-c * WIN_S / SIGMA)) / (W_TRACK + W_TENT * c) for k, c in LEVELS.items()}
    res["idealised_still_share"] = ideal
    fails = []
    for fam, labels in (("translation", TRANSLATE), ("move_turn", MOVE_TURN)):
        wob_all = []
        for lab in labels:
            for wz in traces[lab]:
                wob_all.append(wz - wz.mean())
        if not wob_all:
            fails.append(f"{fam}: no traces")
            continue
        dyaw_w = np.concatenate([window_sum(w, n, dt) for w in wob_all])
        fam_out = {"n_traces": len(wob_all), "n_windows": int(dyaw_w.size),
                   "window_dyaw_std_rad": float(dyaw_w.std()),
                   "window_dyaw_abs_p50_rad": float(np.median(np.abs(dyaw_w))),
                   "window_dyaw_abs_p90_rad": float(np.quantile(np.abs(dyaw_w), 0.9)),
                   "per_step_wz_std": float(np.concatenate(wob_all).std()),
                   "levels": {}}
        st = np.concatenate([straight_income(w, window_sum(w, n, dt), n) for w in wob_all])
        fam_out["straight_wobble_income"] = float(st.mean())
        fam_out["straight_still_income"] = W_TRACK + W_ANG
        for key, c in LEVELS.items():
            trk = float(income(dyaw_w + c * WIN_S, c).mean())
            sip = float(income(dyaw_w, c).mean())
            still = float(income(np.zeros(1), c).mean())
            lvl = {"c": c, "tracker": trk, "step_in_place": sip, "still": still,
                   "share_step_in_place": sip / trk, "share_still": still / trk,
                   "premium_vs_still_per_step": trk - still, "premium_vs_still_per_s": (trk - still) / dt,
                   "premium_vs_step_in_place_per_step": trk - sip}
            if key in ("ceiling", "w_max"):
                for nm in ("share_step_in_place", "share_still"):
                    if lvl[nm] > CEIL_SHARE:
                        fails.append(f"{fam} {key}: {nm} {lvl[nm]:.3f} > {CEIL_SHARE}")
            if key == "w_min":
                for nm in ("share_step_in_place", "share_still"):
                    if lvl[nm] > B_WMIN:
                        fails.append(f"{fam} w_min: {nm} {lvl[nm]:.3f} > B {B_WMIN}")
            fam_out["levels"][key] = lvl
        res["families"][fam] = fam_out
    res["fails"] = fails
    res["verdict"] = "PASS" if not fails else "FAIL"
    Path(a.json).write_text(json.dumps(res, indent=2) + "\n")

    L = ["# Step 1a — body-mode turn pricing on S1's measured yaw traces", "",
         f"Rule: CHARTER §3 (pinned 2026-10-15 before this read). Stamp `{stamp.relative_to(R)}`; window "
         f"{n} steps ({WIN_S} s); σ {SIGMA} rad; weights track {W_TRACK} / tent {W_TENT} (ang-vel term 0 on turning rows).", "",
         f"**Verdict: {res['verdict']}**" + ("" if not fails else " — " + "; ".join(fails)), ""]
    L.append("Idealised still-stander share (wobble-free tracker): " +
             ", ".join(f"{k} {v:.3f}" for k, v in ideal.items()) + f"; pinned w_min bound B = {B_WMIN}.")
    L.append("")
    for fam, fo in res["families"].items():
        L += [f"## {fam} traces (n = {fo['n_traces']}, windows {fo['n_windows']})", "",
              f"{WIN_S:g}-s Δyaw: std {fo['window_dyaw_std_rad']:.4f} rad, |·| p50 {fo['window_dyaw_abs_p50_rad']:.4f}, "
              f"p90 {fo['window_dyaw_abs_p90_rad']:.4f}; per-step wz std {fo['per_step_wz_std']:.3f} rad/s. "
              f"Straight rows (c = 0): wobble income {fo['straight_wobble_income']:.4f}/step vs still "
              f"{fo['straight_still_income']:.4f}.", "",
              "| level | c | tracker | step-in-place | still | share SIP | share still | premium vs still /step | "
              "premium /s |", "|---|---|---|---|---|---|---|---|---|"]
        for key, lv in fo["levels"].items():
            L.append(f"| {key} | {lv['c']:.3f} | {lv['tracker']:.4f} | {lv['step_in_place']:.4f} | {lv['still']:.4f} | "
                     f"{lv['share_step_in_place']:.3f} | {lv['share_still']:.3f} | "
                     f"{lv['premium_vs_still_per_step']:.4f} | {lv['premium_vs_still_per_s']:.2f} |")
        L.append("")
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
