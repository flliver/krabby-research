"""Truncation grid: the running level and the end level each carry an unknown sub-count offset 0..9 tenths."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen, analyze_lib
moves = common.load(sys.argv[1])
with open(sys.argv[2], "w", newline="\n") as f:
    for m in moves:
        if m["yaw"] or m["peak"] < 120 or "quicktest" in m["name"]: continue
        tr, still = analyze_lib.truth(m)
        if tr not in ("switch", "mech", "stall"): continue
        E, I, R = gen.base(m); n = len(E); mx = E[0]; last = 0
        for k in range(1, n):
            if E[k] > mx: mx = E[k]; last = k
        # the end level = samples from 2 before the last forward edge on (the current is already moving there)
        for a in range(0, 10, 1):
            for b in range(0, 10, 1):
                C = [10 * v + (b if k > last else a) for k, v in enumerate(I)]
                f.write(f"{m['name']}|{a}_{b}|{tr}|{m['peak']}|{m['joint']} {1 if m['retract'] else 0} {0 if m['curl'] else 1} 1000 {n}\n")
                f.write(" ".join(f"{x} {y} {z}" for x, y, z in zip(E, C, R)) + "\n")
