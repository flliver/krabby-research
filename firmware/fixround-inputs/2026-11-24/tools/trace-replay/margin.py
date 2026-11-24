import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen
for m in common.load(sys.argv[1]):
    if m["yaw"] or "quicktest" in m["name"]: continue
    E, I, R = gen.base(m)
    n = len(E)
    # forward-edge sample indices (new maxima)
    mx = E[0]; fw = [0]
    for k in range(1, n):
        if E[k] > mx: mx = E[k]; fw.append(k)
    if len(fw) < 10 or E[-1] < 50: continue
    last = fw[-1]
    gaps = [(fw[i+1]-fw[i], fw[i+1]) for i in range(len(fw)-1) if fw[i] > 25]
    if not gaps: continue
    g = max(gaps)
    # min progress over 20 frames while moving (after frame 40, up to last edge)
    mp = min((E[k]-E[k-20], k) for k in range(max(40,20), last+1)) if last > 40 else (None, None)
    tailE = [E[k] for k in range(max(0,last-30), min(n, last+4))]
    print(f"{m['name']:36s} {m['joint']} {'-' if m['retract'] else '+'}{m['peak']:3d} E={E[-1]:5d} lastFwd={last:3d}/{n} maxGapFr={g[0]} at {g[1]} minProg20={mp[0]} at {mp[1]}  last30: {' '.join(str(E[-1]-e) for e in tailE)}")
