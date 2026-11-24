"""Later-start replays of one real move. usage: cut.py moves.json name out.txt [mc]"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen
name = sys.argv[2]; mc = int(sys.argv[4]) if len(sys.argv) > 4 else 0
m = [x for x in common.load(sys.argv[1]) if x["name"] == name][0]
E, I, R = gen.base(m); n = len(E)
ramp = R.index(1)
rnd = random.Random(7)
def curs(I):
    yield "A0", [10 * v for v in I]
    yield "A5", [10 * v + 5 for v in I]
    yield "A9", [10 * v + 9 for v in I]
    for q in range(mc):
        yield "MC%02d" % q, [int(10 * (v + rnd.random())) for v in I]
with open(sys.argv[3], "w", newline="\n") as f:
    def emit(tag, e, i, r):
        for ct, c in curs(i):
            f.write(f"{tag}|{ct} {1 if m['retract'] else 0} {0 if m['curl'] else 1} 1000 {len(e)}\n")
            f.write(" ".join(f"{a} {b} {c2}" for a, b, c2 in zip(e, c, r)) + "\n")
    # M1: ramp kept, N full-PWM samples removed after it
    for N in range(0, n - ramp - 12):
        off = E[ramp + N - 1] - E[ramp - 1]
        e = E[:ramp] + [x - off for x in E[ramp + N:]]
        emit(f"M1_{N:03d}", e, I[:ramp] + I[ramp + N:], R[:ramp] + R[ramp + N:])
    # M2: the drive starts at frame s; the firmware's 0.52 s ramp = 10 samples
    for s in range(0, n - 22):
        base = E[s - 1] if s else 0
        e = [x - base for x in E[s:]]
        emit(f"M2_{s:03d}", e, I[s:], [0] * 10 + [1] * (len(e) - 10))
