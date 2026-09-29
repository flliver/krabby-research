import sys, numpy as np, glob
sys.path.insert(0, "/tmp/claude-1000/sidespeed")
import kin
src = open("/tmp/claude-1000/sidespeed/kin3.py").read().split("res = {}")[0]
g = {}; exec(src, g)
base = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/"
stamps = {p.split("omni_phases/")[1].split("/suite")[0]: p.rsplit("/complete.json", 1)[0] for p in glob.glob(base + "2026-09-2[67]_stroke_*/*/suite__omni/seed001/*/complete.json")}
from pathlib import Path
for name, p in sorted(stamps.items()):
    kin.STAMPS[name] = Path(p)
    r = g["run"](name, "lat_l")
    print(f"{name:45s} speed {r['speed']:.3f} trail knee -rail {r['trail.knee_rail_neg']:.2f} trail u {r['trail.u_p50']:.3f} lead u {r['lead.u_p50']:.3f} lead knee +rail {r['lead.knee_rail_pos']:.2f} stance du trail {r['trail.stance_du']:.3f} lead {r['lead.stance_du']:.3f}")
