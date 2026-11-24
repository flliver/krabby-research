import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen, analyze_lib
def variants(E, I, R):
    n = len(E); mx = E[0]; last = 0
    for k in range(1, n):
        if E[k] > mx: mx = E[k]; last = k
    C = [10 * v + 5 for v in I]
    out = {"base": (E, C, R)}
    for d in (10, -10, 5, -5):
        out[f"endI{d:+d}"] = (E, [c + d if k > last else c for k, c in enumerate(C)], R)
        out[f"refI{d:+d}"] = (E, [c + d if k <= last else c for k, c in enumerate(C)], R)
    out["E>>1"] = ([0] + E[:-1], C, R)
    out["E<<1"] = (E[1:] + [E[-1]], C, R)
    for late in (1, 2, 3):
        e2 = list(E)
        for q in range(late):
            if last + q < n: e2[last + q] = E[last - 1] if last > 0 else 0
        if last + late < n: e2[last + late] = E[last]
        out[f"lastEdge+{late}"] = (e2, C, R)
    # remove one-sample blips above both neighbours
    e3 = list(E)
    for k in range(1, n - 1):
        if E[k] > E[k - 1] and E[k + 1] < E[k]: e3[k] = E[k + 1]
    out["noBlip"] = (e3, C, R)
    out["noBlipA0"] = (e3, [10 * v for v in I], R)
    return out
if __name__ == "__main__":
    moves = common.load(sys.argv[1])
    with open(sys.argv[2], "w", newline="\n") as f:
        for m in moves:
            if m["yaw"]: continue
            tr, still = analyze_lib.truth(m)
            if tr not in ("switch", "mech", "stall"): continue
            E, I, R = gen.base(m)
            for tag, (e, c, r) in variants(E, I, R).items():
                f.write(f"{m['name']}|{tag}|{tr}|{m['peak']}|{m['joint']} {1 if m['retract'] else 0} {0 if m['curl'] else 1} 1000 {len(e)}\n")
                f.write(" ".join(f"{a} {max(0,b)} {x}" for a, b, x in zip(e, c, r)) + "\n")
