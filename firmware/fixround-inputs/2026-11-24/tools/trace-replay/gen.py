"""Generate harness input for every move and reconstruction variant."""
import sys, os, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

def base(m):
    """per-frame lists from the first nonzero-PWM frame: edges, cur(int), ramp."""
    s = m["sign"]; c0 = m["c0"]; pk = m["peak"]
    E = [-s * (x[1] - c0) for x in m["drv"]]
    I = [x[3] for x in m["drv"]]
    R = []; done = False
    for x in m["drv"]:
        done = done or abs(x[2]) >= pk
        R.append(1 if done else 0)
    return E, I, R

def variants(m, mc=0, seed=1):
    E, I, R = base(m)
    n = len(E)
    out = {}
    out["A0"] = (E, [10 * v for v in I], R)
    out["A5"] = (E, [10 * v + 5 for v in I], R)
    out["A9"] = (E, [10 * v + 9 for v in I], R)
    prevI = m["drv"][0][3]
    # trapezoid: boxcar of avgIS over the 50 ms before the frame ~ mean of the two frame values (+0.5 truncation bias)
    T = [5 * (I[k] + (I[k - 1] if k else I[0])) + 5 for k in range(n)]
    out["T5"] = (E, T, R)
    out["T0"] = (E, [t - 5 for t in T], R)
    # timing: drop first frame / one extra leading sample
    out["S+1"] = (E[1:], [10 * v + 5 for v in I[1:]], R[1:])
    out["S-1"] = ([0] + E, [10 * I[0] + 5] + [10 * v + 5 for v in I], [0] + R)
    # resample on a 50 ms grid; frames are p ms apart (first frame at p*U after the drive start, U=0.5)
    drv = m["drv"]
    p = 1000.0 * (drv[-1][0] - drv[0][0]) / max(1, n - 1)
    if not (49.0 <= p <= 54.0): p = 51.4
    for ph10 in range(10):
        ph = ph10 / 10.0
        Er, Cr, Rr = [], [], []
        k = 1
        while True:
            t = 50.0 * k
            f = t / p - ph
            if f > n - 1: break
            j = int(math.floor(f)); a = f - j
            if j < 0:
                e = 0; cur = float(I[0]); r = 0
            else:
                e0 = E[j]; e1 = E[min(j + 1, n - 1)]
                e = int(math.floor(e0 + a * (e1 - e0)))
                fb = max(0.0, f - 25.0 / p); jb = int(math.floor(fb)); ab = fb - jb
                cur = I[jb] + ab * (I[min(jb + 1, n - 1)] - I[jb])
                r = R[j]
            Er.append(e); Cr.append(cur); Rr.append(r)
            k += 1
        for off in (0, 5, 9):
            out["R%d_%d" % (ph10, off)] = (Er, [int(round(10 * c + off)) for c in Cr], Rr)
    rnd = random.Random(seed)
    for q in range(mc):
        out["MC%02d" % q] = (E, [int(10 * (v + rnd.random())) for v in I], R)
    return out

if __name__ == "__main__":
    moves = common.load(sys.argv[1]); mc = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    with open(sys.argv[2], "w", newline="\n") as f:
        for m in moves:
            if m["yaw"]: continue
            for tag, (E, I, R) in variants(m, mc, seed=hash(m["name"]) & 0xffff).items():
                f.write(f"{m['name']}|{tag} {1 if m['retract'] else 0} {0 if m['curl'] else 1} 1000 {len(E)}\n")
                f.write(" ".join(f"{e} {i} {r}" for e, i, r in zip(E, I, R)) + "\n")
