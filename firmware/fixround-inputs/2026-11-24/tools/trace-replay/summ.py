import sys, statistics, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
flt = sys.argv[2] if len(sys.argv) > 2 else ""
for m in common.load(sys.argv[1]):
    if flt and flt not in m["log"]: continue
    drv = m["drv"]; sign = m["sign"]; peak = m["peak"]; c0 = m["c0"]
    full = [x for x in drv if abs(x[2]) == peak]
    cend = drv[-1][1]; cfin = m["post"][-1][1] if m["post"] else cend
    per = (drv[-1][0]-drv[0][0])/max(1,len(drv)-1)
    dts = sorted(round((drv[i+1][0]-drv[i][0])*1000) for i in range(len(drv)-1))
    big = sum(1 for d in dts if d > 80)
    lastchg = max((i for i in range(1,len(drv)) if drv[i][1] != drv[i-1][1]), default=0)
    still = len(drv)-1-lastchg
    cur_run = [x[3] for x in full]
    tail = [x[3] for x in drv[-6:]]
    mid = cur_run[len(cur_run)//3: 2*len(cur_run)//3] or cur_run or [0]
    print(f"{m['name'][:26]:26s} {m['joint']} {'+' if sign>0 else '-'}{peak:3d} {'curl  ' if m['curl'] else 'uncurl'} n={len(drv):4d} nfull={len(full):4d} per={per*1000:5.1f} gaps>80={big:2d} "
          f"c0={c0:6d} cend={cend:6d} cfin={cfin:6d} drive={-sign*(cend-c0):5d} stillFr={still:3d} midI={statistics.median(mid):4.0f} tailI={tail}")
