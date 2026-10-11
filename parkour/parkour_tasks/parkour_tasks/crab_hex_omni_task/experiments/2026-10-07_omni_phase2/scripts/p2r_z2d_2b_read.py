#!/usr/bin/env python3
"""Omni phase 2, stage p2r Z2d -- step 2b's continue / STOP read (P2R_Z2D_PLAN.md step 2b; NOTE(p2r Z2d, 2026-10-11)).

The mask-only closed-loop control: Z2c model_24993 ran with KRABBY_OMNI_STAND_MASK=1 through the observe launcher
(``lr/P2R_OBSZ2cM_s18``, suite es1 + walkstop es1). Continue to candidates v17 only if ALL hold:

* suite es1 is STILL (p2_read's six STILL clauses: steps <= 0.5/s, clearance <= 0.05 m, off-ground <= 0.1,
  hover <= 5/100, taps <= 1/s, KS <= 0.5);
* the suite stand body height is within 0.01 m of the drift-proof probe's (P2R_RSTg_s18 suite es1);
* the observe all-HOLD per-leg mean hip mu - R is <= 0.02 in |.| on both rows;
* the banked stand rows read the mask constants in every masked column (p2r_z2c_observe_read's refusal).

Reported: the walkstop homed share / lifts / settle (the mask-only baseline step 5 reports Z2d against), the
observe per-branch table. Exit 0 CONTINUE, 4 STOP, 2 PENDING (a row missing).

    p2r_z2d_2b_read.py [--json OUT]
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
NAME = "P2R_OBSZ2cM_s18"
PROBE = "P2R_RSTg_s18"
Z2C_CK = "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-10_19-01-45_seed18_X18_p2r_A15pB/model_24993.pt"
HEIGHT_TOL, HIP_LINE = 0.01, 0.02
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", str(HERE / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--name", default=NAME, help="the observe row-eval name (default the step-2b control)")
    ap.add_argument("--no-mask", action="store_true", help="dry-run only: read an UNMASKED observe set")
    a = ap.parse_args()
    rd = _load("p2_read")
    pt, probe = rd.z2_light_point(a.name), rd.z2_light_point(PROBE)
    out = {"name": a.name, "probe": PROBE, "clauses": {}, "reported": {}, "pending": []}
    still = (pt.get("still") or {}).get(1)
    if not still or not (pt.get("stamps") or {}).get("suite@1"):
        out["pending"].append("suite es1 not run")
    else:
        g = rd.z2_stand_gates(pt.get("z2") or {}, pt.get("still") or {}, None, None, seeds=(1,), relative=False)
        for k, v in g["clauses"].items():
            if k.startswith("STILL"):
                out["clauses"][k] = v
            else:
                out["reported"][k] = v
        h, hp = still.get("body_height_m"), ((probe.get("still") or {}).get(1) or {}).get("body_height_m")
        d = None if h is None or hp is None else abs(h - hp)
        out["clauses"][f"suite es1 |body height - {PROBE}| <= {HEIGHT_TOL}"] = (d, HEIGHT_TOL,
                                                                                  d is not None and d <= HEIGHT_TOL)
        out["body_height_m"] = {"masked_Z2c": h, "probe": hp}
    ws = ((pt.get("z2") or {}).get("walkstop") or {}).get(1)
    out["reported"]["walkstop es1 mask-only baseline"] = ws
    # the observe read: scored through the mask, refusing an unmasked bank (rc 3)
    obs = subprocess.run([PY, str(HERE / "p2r_z2c_observe_read.py"), "--name", a.name, "--ckpt", str(REPO / Z2C_CK)]
                         + ([] if a.no_mask else ["--stand-mask"]), capture_output=True, text=True, cwd=str(REPO),
                         env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    out["observe_rc"] = obs.returncode
    if obs.returncode == 3:
        out["clauses"]["observe banks read the mask constants on every stand row"] = (False, True, False)
    elif obs.returncode != 0:
        out["pending"].append(f"observe read rc {obs.returncode}: {obs.stderr[-400:]}")
    else:
        txt = obs.stdout
        rows = json.loads(txt[txt.index("{"):])
        out["clauses"]["observe banks read the mask constants on every stand row"] = (True, True, True)
        for row in ("suite", "walkstop"):
            r = rows.get(row)
            if not r:
                out["pending"].append(f"observe {row} bank missing")
                continue
            hip = (r.get("allhold_mean_mu_minus_R") or {}).get("hip")
            m = None if hip is None else max(abs(x) for x in hip)
            out["clauses"][f"observe {row} es1 all-HOLD |mean hip mu - R| <= {HIP_LINE}"] = (m, HIP_LINE,
                                                                                         m is not None and m <= HIP_LINE)
            out["reported"][f"observe {row} by_branch"] = r.get("by_branch")
            out["reported"][f"observe {row} branch_leg_shares"] = r.get("branch_leg_shares")
    bad = [k for k, (_v, _t, ok) in out["clauses"].items() if not ok]
    out["failed"] = bad
    out["verdict"] = "PENDING" if out["pending"] else ("STOP" if bad else "CONTINUE")
    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if a.json:
        a.json.write_text(txt)
    return {"CONTINUE": 0, "STOP": 4, "PENDING": 2}[out["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
