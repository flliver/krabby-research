"""Per-gate loss breakdown (share of ungated credit) per run and category, with the eval's own foot_slip
(loaded-foot planar speed) for context. usage: breakdown.py [k=v ...]"""
import json
import sys
from pathlib import Path
import importlib.util

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diag import RSP, RUNS  # noqa: E402

B = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval"
RUNS = dict(RUNS)
RUNS.update({
    "Z_t3": f"{B}/omni_phases/2026-09-23_clock_deprecation/seed3_X05_t3_A15pB/suite__omni/seed001/2026-09-25_04-33-44",
    "Z_c6": f"{B}/omni_phases/2026-09-23_clock_deprecation/seed3_X05_c6_A15pB/suite__omni/seed001/2026-09-25_06-52-47",
    "A4500": f"{B}/omni_phases/2026-09-23_clock_deprecation/seed3_X04_c6_A15pB/suite__omni/seed001/2026-09-24_16-39-45",
    "E6": f"{B}/omni_phases/2026-09-23_clock_deprecation/step0_E6_statue/suite__omni/seed001/2026-09-23_21-59-41",
    "E7": f"{B}/omni_phases/2026-09-23_clock_deprecation/step0_E7_creeper-1c/suite__omni/seed001/2026-09-23_22-07-41",
})


def main(names, params):
    out = {}
    for name in names:
        st = RSP.load_stamp(RUNS[name])
        rp = RSP.replay(st, params)
        sd = rp.stances
        catv = np.array([e["category"] for e in st.episodes])
        cats = ("side",) if st.kind == "side" else RSP.MOVING
        for c in cats:
            k = st.own[sd["env"], sd["step"]] & rp.gate[sd["env"], sd["step"]] & (catv[sd["env"]] == c)
            pk = k & sd["payable"]
            U = sd["ungated"][k].sum()
            fs = []
            for i in np.flatnonzero(catv == c):
                j = st.json_eps[i] or {}
                q = ((j.get("holds") or {}).get(st.episodes[i]["label"]) or {}).get("quality") or {}
                if q.get("foot_slip") is not None:
                    fs.append(q["foot_slip"])
            back_slide = np.maximum(sd["B"] - sd["Ba"], 0.0)
            row = {"U": float(U), "kept": float(sd["pay"][pk].sum() / U) if U else None,
                   "lost_n": float(sd["ungated"][k & ~sd["payable"]].sum() / U) if U else None,
                   "lost_anchor": float((sd["ungated"] - sd["after_anchor"])[pk].sum() / U) if U else None,
                   "lost_slip": float((sd["after_anchor"] - sd["after_slip"])[pk].sum() / U) if U else None,
                   "lost_prog": float((sd["after_slip"] - sd["pay"])[pk].sum() / U) if U else None,
                   "kept_of_anchored": float(sd["pay"][pk].sum() / sd["after_anchor"][pk].sum()) if pk.any() else None,
                   "back_slide_over_B": float(back_slide[pk].sum() / np.maximum(sd["B"][pk], 0).sum()) if pk.any() else None,
                   "median_rho": float(np.median(sd["rho"][pk])) if pk.any() else None,
                   "eval_foot_slip": float(np.median(fs)) if fs else None,
                   "per_s": RSP.cat_income(RSP.GateRun(name, "x", st, rp), c)}
            out[f"{name}.{c}"] = row
            print(f"{name:10s} {c:9s} " + " ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                                                   for a, b in row.items()), flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    names = [a for a in args if "=" not in a] or ["W", "X0x_s3", "X0x_s2", "E1", "side_slow", "side_step"]
    params = RSP.parse_params([a for a in args if "=" in a])
    res = main(names, params)
    (Path.cwd() / "breakdown.json").write_text(json.dumps(RSP._jsonable(res), indent=1))
