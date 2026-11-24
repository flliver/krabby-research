import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen
rows = collections.defaultdict(list)
for m in common.load(sys.argv[1]):
    if m["yaw"] or not m["retract"]: continue
    E, I, R = gen.base(m); n = len(E)
    mx = E[0]; last = 0
    for k in range(1, n):
        if E[k] > mx: mx = E[k]; last = k
    still = n - 1 - last
    if still < 12 or R.count(1) < 12: continue
    # worst spread over any 8-sample window starting 3+ samples after the last forward edge
    seg = I[last + 3:]
    win = [max(seg[a:a + 8]) - min(seg[a:a + 8]) for a in range(0, len(seg) - 7)]
    best = min(win) if win else None; worst = max(win) if win else None
    kind = ("hip" if m["hip"] else "knee") + ("-press" if E[-1] <= 3 else "-end")
    rows[kind].append((worst, best, m["name"], m["joint"], m["peak"], E[-1], min(seg), max(seg)))
for kind, rs in rows.items():
    print(kind, len(rs), "moves; worst 8-sample spread histogram:", sorted(collections.Counter(r[0] for r in rs).items()))
    for r in rs:
        if r[0] is not None and r[0] >= 2: print("    spread>=2:", r)
