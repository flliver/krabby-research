import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
SW_OPEN = {"FL": 8, "FR": 11}

def truth(m):
    drv = m["drv"]; s = m["sign"]; c0 = m["c0"]
    E = [-s * (x[1] - c0) for x in drv]
    lastchg = max((i for i in range(1, len(drv)) if drv[i][1] != drv[i - 1][1]), default=0)
    still = len(drv) - 1 - lastchg
    travel = E[-1]
    tail = [x[3] for x in drv[-6:]]
    leg = m["joint"][:2]
    if still < 6:
        return "running", still
    if abs(travel) <= 3:
        return "press", still
    flat = max(tail) - min(tail) <= 1
    if m["retract"]:
        # a switch: count dead, current flat at the joint's switch-open level
        lvl = SW_OPEN.get(leg, 99) if m["hip"] else 8
        if flat and max(tail) <= lvl:
            return "switch", still
        return "stall", still
    else:
        if m["hip"]:
            if flat and max(tail) <= SW_OPEN.get(leg, 99): return "ext-switch", still
            return "obstacle", still
        return "mech", still   # knee extend: every such move here ends at the curl stop (checked by count below)

def main():
    moves = {m["name"]: m for m in common.load(sys.argv[1])}
    res = collections.defaultdict(dict)
    for line in open(sys.argv[2]):
        p = line.split()
        name, tag = p[0].split("|")
        d = dict(kind=p[1]); 
        for kv in p[2:]:
            k, v = kv.split("="); d[k] = int(v)
        res[name][tag] = d
    flt = sys.argv[3] if len(sys.argv) > 3 else ""
    for name, m in moves.items():
        if m["yaw"] or name not in res: continue
        if flt and not re.search(flt, name): continue
        tr, still = truth(m)
        r = res[name]
        drv = m["drv"]; s = m["sign"]; c0 = m["c0"]
        E = [-s * (x[1] - c0) for x in drv]
        det = ["A0", "A5", "A9", "T0", "T5", "S+1", "S-1"]
        kinds = [r[t]["kind"] for t in det]
        mc = collections.Counter(r[t]["kind"] for t in r if t.startswith("MC"))
        rs = collections.Counter(r[t]["kind"] for t in r if t.startswith("R"))
        a = r["A0"]
        flip = len(set(kinds)) > 1 or len(mc) > 1 or len(rs) > 1 or (set(rs) != set(kinds))
        exp = {"running": "None", "press": "Still", "switch": "Switch", "stall": "Stuck", "mech": "Stop", "ext-switch": "?", "obstacle": "?"}[tr]
        print(f"{name:36s} {m['joint']} {'-' if m['retract'] else '+'}{m['peak']:3d} {'curl  ' if m['curl'] else 'uncurl'} n={len(drv):3d} E={E[-1]:5d} c={c0:5d}>{drv[-1][1]:5d} still={still:2d} truth={tr:10s} "
              f"A0={a['kind']:6s}k={a['k']:3d} e={a['edges']:4d} arm={a['kArm']:3d} i={a['iArm']:3d}>{a['iRef']:3d} v={a['vRef']:3d} cur={a['cur']}/{a['ref']} nrm={a['normal']:2d} | "
              f"{'FLIP ' if flip else ''}{' '.join(t + ':' + k[:4] for t, k in zip(det, kinds)) if len(set(kinds)) > 1 else ''} R={dict(rs)} MC={dict(mc)}")
main()
