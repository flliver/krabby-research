import sys, os
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detector as det, parse_moves as pm
from replay import run_move
for log in sys.argv[1:]:
    try:
        mvs = pm.moves(log)
    except Exception as ex:
        print(log, 'ERR', ex); continue
    print('==', os.path.basename(log), len(mvs), 'moves')
    for idx, mv in enumerate(mvs):
        sp = [x for x in mv['trace'] if x[2] != 0]
        if not sp: continue
        mv['direction'] = pm.direction(mv['joint'], sp[0][2])
        r = run_move(mv, det.Params())
        if r is None: continue
        travel = abs(sp[-1][1] - sp[0][1])
        peak = max(abs(x[2]) for x in sp)
        line = f"#{idx:<3} {mv['joint']} {mv['direction']:<6} pwm={peak:3d} n={len(sp):4d} travel={travel:4d} start={sp[0][1]:5d} final={r['final_c']:5d} -> {r['kind']:<6}"
        if r['kind'] != 'none':
            line += f" t={r['fire_t']:5.2f}s ({r['after_stop']:+.2f}s vs last edge) rec={r['fire_c']} short={r['remaining']:+d} {r.get('info',{}).get('ratio','')}"
        else:
            line += f" host stop {r['host_stop_t']}s last edge {r['stop_t']}s"
        line += f" armed={r['armed']} iRef={r['i_ref']} vRef={r['v_ref']}"
        print(line)
