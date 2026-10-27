#!/usr/bin/env python3
"""Rung 0b read-out (CPU, read-only): the [0, 2) s fall share per row and the pre-registered staging decision
(CHARTER section 5; plan step 2 item 4).

Usage: rung0b_read.py [--root DIR] [--json OUT] [--assume-verified] [ROW=STAMP_DIR ...]
  --root: the rung0b.sh output root (<root>/<row>/<row>/seed00N/<ts>/ and <root>/<row>.log); explicit ROW=STAMP
          pairs override it. Row ids: rung0b_{frac0,rsi}_{off,on}_e{1,2}__omni.

PER ROW (from the raw npz, one episode per env -- a schedule row never re-spawns a done env):
  fall share [0, 2) s = episodes whose FIRST done step is < 2 s after reset (step 100 at dt 0.02) AND is a fall
  (run_meta termination_reason "fall", else the npz crab_failure at that step) / all episodes.
  RSI rows, descriptive: the frame's own speed (|root_lin_vel_b xy| at row 0, the spawned state) and lateral sign,
  and the [0, 2) s speed along the command (world +y here: dir pi/2, face 0) up to the first done step.

BEFORE DECIDING (CHARTER 5)
  * staging verification from each row's log: ONLY the env's tagged lines (parkour_actions.py
    ``_report_rsi_cam_stage``, one per reset with RSI envs; never the "[KNOB] KRABBY_RSI_CAM_STAGE..." lines)
        RSI_CAM_STAGE_DEBUG n_rsi=<n> stage=<on|off> target_err=<e> hist_err=<e> target_absmax=<a>
            hist_absmax=<a> bank_absmax=<a>
    computed PER ENV by the env. Staging-ON RSI rows: every line stage=on, n_rsi > 0, target_err and hist_err
    <= 1e-4 (applied cam target - clamp(jv, +-pi) rad/s; history cam columns, every slot, - clip(jv / pi)).
    Staging-OFF RSI rows: every line stage=off, n_rsi > 0, target_absmax and hist_absmax == 0 and bank_absmax > 0
    (the frames do move the cams). FRAC-0 rows: no RSI env, so no line (OK; a line there is a FAIL). Only the
    two RSI conditions gate the decision. A log without a tagged line reads UNVERIFIED (--assume-verified only
    after a manual check).
  * the RSI staging-off and staging-on rows of a seed must differ (identical raw arrays -> NO DECISION: fix and
    rerun). The FRAC-0 pairs are identical by construction and are not tested.

THE RULE (both conditions, in BOTH seeds; normal = the FRAC-0 staging-off row)
  (a) deficit = off - normal > 0.04 AND > the seed-pair spread (the largest |seed1 - seed2| over the four
      conditions);
  (b) (on - normal) <= 0.5 x (off - normal).
  Both hold -> STAGING ON: C_lr_rsi runs as chains C_lr_rsi_st__*; otherwise C_lr_rsi__* (X16's definition).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

DT = 0.02
WINDOW_S = 2.0
DEFICIT_MIN = 0.04
HALF = 0.5
VERIFY_TOL = 1e-4
CONDS = ("frac0_off", "frac0_on", "rsi_off", "rsi_on")
SEEDS = (1, 2)


def row_id(cond: str, seed: int) -> str:
    return f"rung0b_{cond}_e{seed}__omni"


def stamp_complete(d: Path) -> bool:
    """A finished stamp. ``complete.json`` exists only on CATEGORY-SUITE rows (``report.write_run`` writes it
    last); the rung-0b rows are plain ``schedule`` rows, whose last write is the raw npz set (one file per
    episode, after run_meta/metrics/summary). So: ``complete.json``, or a run_meta.json whose
    ``n_steps_per_env`` count equals the number of raw ``episode_*.npz`` files (fix 2026-10-01, SMOKES.md)."""
    if (d / "complete.json").is_file():
        return True
    try:
        n = len(json.loads((d / "run_meta.json").read_text())["n_steps_per_env"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return n > 0 and len(list((d / "raw").glob("episode_*.npz"))) == n


def find_stamps(root: Path) -> dict:
    out = {}
    for cond in CONDS:
        for s in SEEDS:
            rid = row_id(cond, s)
            ds = sorted(d for d in (root / rid / rid).glob("seed*/*/") if stamp_complete(d))
            if ds:
                out[rid] = ds[-1]
    return out


def read_row(stamp: Path) -> dict:
    meta = json.loads((stamp / "run_meta.json").read_text())
    reasons = meta.get("termination_reason") or []
    files = sorted((stamp / "raw").glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    w = int(round(WINDOW_S / DT))
    falls = 0
    rows = []
    h = hashlib.sha256()
    for p in files:
        idx = int(p.stem.split("_")[1])
        z = np.load(p)
        done = z["done"].astype(bool)
        td = int(np.argmax(done)) if done.any() else len(done)
        reason = reasons[idx] if idx < len(reasons) else ""
        is_fall = reason == "fall" or (reason == "" and "crab_failure" in z.files and td < len(done)
                                       and bool(z["crab_failure"][td]))
        fell = bool(td < w and is_fall)
        falls += fell
        rp = z["root_pos_w"][:, :2].astype(float)
        v0 = z["root_lin_vel_b"][0, :2].astype(float)
        hi = max(1, min(w, td))
        along = float((rp[hi] - rp[0])[1] / (hi * DT)) if hi < len(rp) else float("nan")
        rows.append({"frame_speed": float(np.linalg.norm(v0)), "frame_vy": float(v0[1]), "along_0_2": along,
                     "fell_0_2": fell})
        h.update(np.ascontiguousarray(z["root_pos_w"][: w + 1]).tobytes())
    n = len(files)
    out = {"stamp": str(stamp), "n": n, "falls_0_2": falls, "fall_share": falls / n if n else None,
           "digest": h.hexdigest(), "env_vars": meta.get("env_vars", {})}
    if rows:
        fs = np.array([r["frame_speed"] for r in rows])
        vy = np.array([r["frame_vy"] for r in rows])
        al = np.array([r["along_0_2"] for r in rows])
        sp = {"frame_speed_median": float(np.median(fs)), "along_0_2_median": float(np.nanmedian(al))}
        for name, m in (("frame_+y", vy > 0.02), ("frame_-y", vy < -0.02), ("frame_still", np.abs(vy) <= 0.02)):
            if m.any():
                sp[name] = {"n": int(m.sum()), "frame_speed_median": float(np.median(fs[m])),
                            "along_0_2_median": float(np.nanmedian(al[m])),
                            "fall_share": float(np.mean([r["fell_0_2"] for r, k in zip(rows, m) if k]))}
        out["speed"] = sp
    return out


#: The env's rung-0b line (parkour_actions.py ``_report_rsi_cam_stage``): the tag as a whole token, followed by
#: ``n_rsi=`` -- the "[KNOB] KRABBY_RSI_CAM_STAGE_DEBUG=1 consumed" line contains the tag only as a substring.
DEBUG_LINE = re.compile(r"(?:^|\s)RSI_CAM_STAGE_DEBUG\s+(n_rsi=.*)$")


def verify_log(log: Path | None, cond: str) -> dict:
    """Staging verification from the env's ``RSI_CAM_STAGE_DEBUG`` lines (see the module docstring)."""
    if log is None or not log.is_file():
        return {"status": "UNVERIFIED", "why": "no log"}
    lines, recs = [], []
    for raw in log.read_text(errors="replace").splitlines():
        m = DEBUG_LINE.search(raw.rstrip())
        if not m:
            continue
        lines.append(raw.rstrip())
        kv = dict(re.findall(r"([A-Za-z_]+)=(\S+)", m.group(1)))
        try:
            recs.append({"stage": kv.get("stage"), **{k: float(kv[k]) for k in (
                "n_rsi", "target_err", "hist_err", "target_absmax", "hist_absmax", "bank_absmax")}})
        except (KeyError, ValueError) as exc:
            return {"lines": lines[:6], "n_lines": len(lines), "status": "UNVERIFIED",
                    "why": f"malformed RSI_CAM_STAGE_DEBUG line ({exc!r}): {raw.strip()[:160]}"}
    out = {"lines": lines[:6], "n_lines": len(lines)}
    if cond in ("frac0_off", "frac0_on"):
        if recs:
            return {**out, "status": "FAIL", "why": "a FRAC-0 row logged RSI envs"}
        return {**out, "status": "OK", "why": "no RSI env at FRAC 0 (nothing to stage)"}
    if not recs:
        return {**out, "status": "UNVERIFIED", "why": "no RSI_CAM_STAGE_DEBUG line in the log"}
    mx = {k: max(r[k] for r in recs) for k in ("target_err", "hist_err", "target_absmax", "hist_absmax", "bank_absmax")}
    mx["n_rsi_min"] = min(r["n_rsi"] for r in recs)
    stages = sorted({r["stage"] for r in recs})
    if cond == "rsi_on":
        ok = (stages == ["on"] and mx["n_rsi_min"] > 0 and mx["target_err"] <= VERIFY_TOL
              and mx["hist_err"] <= VERIFY_TOL)
    elif cond == "rsi_off":
        ok = (stages == ["off"] and mx["n_rsi_min"] > 0 and mx["target_absmax"] == 0.0 and mx["hist_absmax"] == 0.0
              and mx["bank_absmax"] > 0.0)
    else:
        return {**out, "status": "n/a"}
    return {**out, "status": "OK" if ok else "FAIL", "stages": stages, "max": mx}


def decide(shares: dict, verified: dict, identical: dict, assume_verified: bool = False) -> dict:
    """``shares[cond][seed]`` fall shares; ``verified[cond][seed]`` statuses; ``identical[seed]`` rsi off == on."""
    missing = [(c, s) for c in CONDS for s in SEEDS if shares.get(c, {}).get(s) is None]
    if missing:
        return {"decision": "NO DECISION", "why": f"rows missing: {missing}"}
    if any(identical.get(s) for s in SEEDS):
        return {"decision": "NO DECISION", "why": "RSI staging-on and staging-off rows are identical: fix and rerun"}
    bad = [(c, s, verified[c][s]) for c in ("rsi_on", "rsi_off") for s in SEEDS
           if verified.get(c, {}).get(s) not in ("OK",)]
    if bad and not assume_verified:
        return {"decision": "NO DECISION", "why": f"staging not verified: {bad} (check the logs; --assume-verified)"}
    spread = max(abs(shares[c][1] - shares[c][2]) for c in CONDS)
    per_seed = {}
    for s in SEEDS:
        normal, off, on = shares["frac0_off"][s], shares["rsi_off"][s], shares["rsi_on"][s]
        deficit = off - normal
        a = deficit > DEFICIT_MIN and deficit > spread
        b = (on - normal) <= HALF * deficit
        per_seed[s] = {"normal": normal, "off": off, "on": on, "deficit": deficit, "a_deficit_exists": a,
                       "b_staging_halves_it": b}
    on = all(v["a_deficit_exists"] and v["b_staging_halves_it"] for v in per_seed.values())
    return {"decision": "STAGING ON" if on else "STAGING OFF", "seed_pair_spread": spread, "per_seed": per_seed,
            "chains": "C_lr_rsi_st__C0/__A/__B" if on else "C_lr_rsi__C0/__A/__B (X16's definition)",
            "assumed_verified": bool(bad and assume_verified)}


def self_test() -> int:
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got!r}")

    V = {c: {1: "OK", 2: "OK"} for c in CONDS}
    sh = {"frac0_off": {1: 0.05, 2: 0.06}, "frac0_on": {1: 0.05, 2: 0.06}, "rsi_off": {1: 0.20, 2: 0.22},
          "rsi_on": {1: 0.08, 2: 0.09}}
    chk("deficit + halved -> ON", decide(sh, V, {1: False, 2: False})["decision"], "STAGING ON")
    sh2 = dict(sh, rsi_on={1: 0.15, 2: 0.16})
    chk("not halved -> OFF", decide(sh2, V, {1: False, 2: False})["decision"], "STAGING OFF")
    sh3 = dict(sh, rsi_off={1: 0.08, 2: 0.09})
    chk("deficit <= 0.04 -> OFF", decide(sh3, V, {1: False, 2: False})["decision"], "STAGING OFF")
    sh4 = dict(sh, rsi_off={1: 0.20, 2: 0.40}, rsi_on={1: 0.06, 2: 0.07})
    chk("deficit below the seed-pair spread in seed 1 -> OFF", decide(sh4, V, {1: False, 2: False})["decision"], "STAGING OFF")
    chk("identical rows -> NO DECISION", decide(sh, V, {1: True, 2: False})["decision"], "NO DECISION")
    V2 = dict(V, rsi_on={1: "UNVERIFIED", 2: "OK"})
    chk("unverified -> NO DECISION", decide(sh, V2, {1: False, 2: False})["decision"], "NO DECISION")
    chk("unverified + --assume-verified -> decides", decide(sh, V2, {1: False, 2: False}, True)["decision"], "STAGING ON")
    print(f"self-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction/rung0b"))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--assume-verified", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("rows", nargs="*", help="ROW=STAMP_DIR overrides")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    stamps = find_stamps(a.root)
    for spec in a.rows:
        rid, d = spec.split("=", 1)
        stamps[rid] = Path(d)
    res, shares, verified, digests = {}, {}, {}, {}
    for cond in CONDS:
        for s in SEEDS:
            rid = row_id(cond, s)
            if rid not in stamps:
                continue
            r = read_row(Path(stamps[rid]))
            r["verify"] = verify_log(a.root / f"{rid}.log", cond)
            res[rid] = r
            shares.setdefault(cond, {})[s] = r["fall_share"]
            verified.setdefault(cond, {})[s] = r["verify"]["status"]
            digests[(cond, s)] = r["digest"]
    identical = {s: (("rsi_off", s) in digests and digests.get(("rsi_off", s)) == digests.get(("rsi_on", s)))
                 for s in SEEDS}
    dec = decide(shares, verified, identical, a.assume_verified)
    print("| row | n | falls [0,2) s | fall share | verify | frame speed med | along [0,2) med |")
    print("|---|---|---|---|---|---|---|")
    for rid, r in res.items():
        sp = r.get("speed") or {}
        print(f"| {rid} | {r['n']} | {r['falls_0_2']} | {r['fall_share']:.3f} | {r['verify']['status']} | "
              f"{sp.get('frame_speed_median', float('nan')):.3f} | {sp.get('along_0_2_median', float('nan')):+.3f} |")
    for rid, r in res.items():
        if r["verify"].get("lines"):
            print(f"  {rid} staging log: {r['verify']['lines'][0][:200]}")
    print(f"RUNG0B {dec['decision']}: {dec.get('why') or dec.get('chains')}")
    if a.json:
        a.json.write_text(json.dumps({"rows": res, "decision": dec,
                                      "identical": {str(k): v for k, v in identical.items()}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
