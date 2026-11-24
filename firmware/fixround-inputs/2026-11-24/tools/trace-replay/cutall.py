"""Later-start replays of every real end. usage: cutall.py moves.json out.txt [as_curl]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen, analyze_lib
want = sys.argv[3].split(",") if len(sys.argv) > 3 else ["switch", "mech"]
force = sys.argv[4] if len(sys.argv) > 4 else ""   # "" | "hipcurl" | "kneecurl"
minpwm = int(sys.argv[5]) if len(sys.argv) > 5 else 120
with open(sys.argv[2], "w", newline="\n") as f:
    for m in common.load(sys.argv[1]):
        if m["yaw"] or m["peak"] < minpwm: continue
        tr, still = analyze_lib.truth(m)
        if tr not in want: continue
        E, I, R = gen.base(m); n = len(E); ramp = R.index(1)
        retract, leaving = (1 if m["retract"] else 0), (0 if m["curl"] else 1)
        if force == "hipcurl": retract, leaving = 1, 0
        if force == "kneecurl": retract, leaving = 0, 0
        total = E[-1]
        def emit(tag, e, i, r):
            for off in (0, 5, 9):
                f.write(f"{m['name']}|{tag}|{off}|{tr}|{m['peak']}|{m['joint']}|{total - (total - e[-1])} {retract} {leaving} 1000 {len(e)}\n")
                f.write(" ".join(f"{a} {10 * b + off} {c}" for a, b, c in zip(e, i, r)) + "\n")
        for N in range(0, max(0, n - ramp - 12)):
            off = E[ramp + N - 1] - E[ramp - 1]
            emit(f"M1_{N:03d}", E[:ramp] + [x - off for x in E[ramp + N:]], I[:ramp] + I[ramp + N:], R[:ramp] + R[ramp + N:])
        for s in range(0, max(0, n - 22)):
            base = E[s - 1] if s else 0
            e = [x - base for x in E[s:]]
            emit(f"M2_{s:03d}", e, I[s:], [0] * 10 + [1] * (len(e) - 10))
