"""Later starts of one move under every reconstruction (frame=sample, resampled 50 ms x phase x offset, dither)."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen
name = sys.argv[2]
m = [x for x in common.load(sys.argv[1]) if x["name"] == name][0]
drv = m["drv"]; n = len(drv)
with open(sys.argv[3], "w", newline="\n") as f:
    for s in range(0, n - 22):
        sub = dict(m); sub["drv"] = drv[s:]; sub["c0"] = drv[s - 1][1] if s else m["c0"]
        for tag, (E, I, R) in gen.variants(sub, mc=20, seed=s).items():
            if tag in ("S+1", "S-1"): continue
            R = [0] * 10 + [1] * (len(E) - 10)
            f.write(f"{s}|{tag}|{E[-1]} {1 if m['retract'] else 0} {0 if m['curl'] else 1} 1000 {len(E)}\n")
            f.write(" ".join(f"{a} {b} {c}" for a, b, c in zip(E, I, R)) + "\n")
