"""Rod odd RMS and joint-space odd RMS per hand of eval stamps (CPU, read-only; CHARTER 4.9, descriptive).

Usage: odd_rms.py [--cache DIR] [--json OUT] [--reuse] [--jobs N] NAME=STAMP_DIR [NAME=STAMP_DIR ...]

For each NAME the stamp is extracted into the sym cache (extract.py's one(), unless --reuse finds it cached), then
for every lat_* hold label present:
  rod odd RMS    = amp * sqrt(odd_pi) of the clipped rod-action pattern (``act_rod``: Hip_Femur + Femur_Tibia),
  joint odd RMS  = the same on the rod joint positions (``jp_rod``),
exactly the side bisect's final.py statistic (``a["amp"] * np.sqrt(a["odd_pi"])``, symlib.selfmirror at the
M_y "pi" mirror). The phase is chosen as sym.py's row() does: the gait clock when it is logged and the rod joints
lock to it at least 0.8x as well as to the intrinsic (PCA + Hilbert) phase, else the intrinsic phase. A hand's
value is the mean over its speeds (lat_l_022/025 -> L, lat_r_022/025 -> R). The side bisect read odd RMS >= 1.4
on the FAST one-direction strokes and <= 0.9 on the two-direction shuffles (ROOT_CAUSE).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract  # noqa: E402
import symlib  # noqa: E402

LABELS = ("lat_l_022", "lat_l_025", "lat_r_022", "lat_r_025")


def phase_for(d: dict):
    """(phase (E,300), 'clock'|'intrinsic') -- sym.py row()'s rule, verbatim."""
    th, _f0, _ = symlib.intrinsic_phase(d)
    has_clock = bool(np.isfinite(d["ph"]).all())
    r2i = symlib.pattern(symlib.sub(d, "jp_rod"), th)[1]
    r2c = symlib.pattern(symlib.sub(d, "jp_rod"), d["ph"])[1] if has_clock else 0.0
    use_clock = has_clock and r2c >= 0.8 * r2i
    return (d["ph"] if use_clock else th), ("clock" if use_clock else "intrinsic")


def odd_rms_label(d: dict) -> dict:
    ph, which = phase_for(d)
    out = {"n": int(len(d["act"])), "phase": which, "feet_ss": float(np.median(d["feet_ss"]))}
    for rep, key in (("act_rod", "rod_odd_rms"), ("jp_rod", "joint_odd_rms")):
        o, _A = symlib.selfmirror(symlib.sub(d, rep), ph, rep)
        out[key] = float(o["amp"] * np.sqrt(o["odd_pi"]))
        out[key.replace("odd_rms", "odd_frac")] = float(o["odd_pi"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=None, help=f"sym cache (default {symlib.CACHE})")
    ap.add_argument("--json", default=None)
    ap.add_argument("--reuse", action="store_true", help="skip extraction when NAME is already cached")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("stamps", nargs="+", help="NAME=STAMP_DIR")
    a = ap.parse_args(argv)
    cache = symlib.set_cache(a.cache) if a.cache else symlib.set_cache(symlib.CACHE)
    items = extract.parse_specs(a.stamps)
    todo = [i for i in items if not (a.reuse and any(cache.glob(f"{i[0]}__lat_*.npz")))]
    if todo:
        extract.main(["--cache", str(cache), "--jobs", str(a.jobs)]
                     + [f"{n}:{g}={p}" for n, g, p in todo])
    res = {}
    for name, _g, path in items:
        r = {"stamp": path, "labels": {}}
        for lab in LABELS:
            d = symlib.load(name, lab)
            if d is None or len(d["act"]) < 4:
                continue
            r["labels"][lab] = odd_rms_label(d)
        for hand in ("l", "r"):
            ls = [v for k, v in r["labels"].items() if k.startswith(f"lat_{hand}_")]
            if ls:
                r[f"rod_odd_rms_{hand}"] = float(np.mean([v["rod_odd_rms"] for v in ls]))
                r[f"joint_odd_rms_{hand}"] = float(np.mean([v["joint_odd_rms"] for v in ls]))
        res[name] = r
        print(f"== {name}  {path}")
        for lab, v in r["labels"].items():
            print(f"   {lab:10s} n {v['n']:3d}  phase {v['phase']:9s}  feet_ss {v['feet_ss']:+.3f}  "
                  f"rod odd RMS {v['rod_odd_rms']:.3f} (frac {v['rod_odd_frac']:.2f})  "
                  f"joint odd RMS {v['joint_odd_rms']:.3f} (frac {v['joint_odd_frac']:.2f})")
        for hand in ("l", "r"):
            if f"rod_odd_rms_{hand}" in r:
                print(f"   HAND {hand.upper()}  rod odd RMS {r[f'rod_odd_rms_{hand}']:.3f}  "
                      f"joint odd RMS {r[f'joint_odd_rms_{hand}']:.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
