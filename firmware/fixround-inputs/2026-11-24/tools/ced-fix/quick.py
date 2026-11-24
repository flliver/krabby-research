import glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import detector as det, parse_moves as pm, evaluate as ev
sys.path.insert(0, os.path.join(HERE, '..', 'ced-design'))
def expected(mv):
    tr = mv['trace']; drv = [x for x in tr if x[2] != 0]
    travel = abs(drv[-1][1] - drv[0][1])
    j, d = mv['joint'], mv['direction']
    moving_at_stop = any(drv[n][1] != drv[n - 1][1] for n in range(max(1, len(drv) - 6), len(drv)))
    if travel <= 3: return {'static', 'none'}
    if j.endswith('HL') and d == 'uncurl' and travel > 900: return {'stuck', 'mech'}
    if moving_at_stop: return {'none'}
    if j.endswith('HL') and d == 'uncurl': return {'stuck', 'mech'}
    if j.endswith('HL') and d == 'curl': return {'switch'}
    if j.endswith('KL') and d == 'uncurl': return {'stuck'}
    if j.endswith('KL') and d == 'curl': return {'mech', 'static'}
    return {'?'}
params = det.Params()
for a in sys.argv[1:]:
    n, v = a.split('='); cur = getattr(params, n)
    setattr(params, n, (v.lower() in ('1','true')) if isinstance(cur, bool) else type(cur)(v))
ok = bad = 0; ratios = []
for log in sorted(glob.glob(os.path.join(HERE, '..', 'quicktest-*.log'))):
    for idx, mv in enumerate(pm.moves(log)):
        if mv['joint'].endswith('HY') or not mv['trace']: continue
        sp = [x for x in mv['trace'] if x[2] != 0]
        if not sp: continue
        mv['direction'] = pm.direction(mv['joint'], sp[0][2])
        r = ev.move_samples(mv)
        if r is None: continue
        s, retract, curl = r
        g = ev.judge(s, retract, curl, params)
        exp = expected(mv)
        good = g[0] in exp; ok += good; bad += not good
        if g[0] == 'switch': ratios.append((g[3]['fast'], g[3]['ratio'], mv['joint']))
        if not good: print('BAD', os.path.basename(log), idx, mv['joint'], mv['direction'], exp, g)
print(ok, 'OK', bad, 'BAD'); print(sorted(ratios))
