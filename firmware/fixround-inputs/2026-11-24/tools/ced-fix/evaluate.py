"""Review-fix harness: every robot move (2026-11-24 endcur goldens + today's PWM 128/150 logs),
short-drive variants of every real curl-end approach, and shortened variants of the F1 stall,
judged by a detector variant. Usage: python evaluate.py [name=value ...] [-v]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detector as det  # noqa: E402
import parse_moves as pm  # noqa: E402

SP = os.path.join(HERE, '..')
TODAY = ['clearance-FL-115032.log', 'clearance-FL-121754.log', 'clearance-FL-140513.log', 'clearance-FR-115750.log',
         'fr-recover-120231.log', 'hipend-FL-115546.log', 'rodscale-hip_back-114914.log',
         'rodscale-knee_back_hip_out-114339.log', 'rodscale-knee_out-114217.log']
F1 = ('clearance-FR-115750.log', 7)


def move_samples(mv):
    """(samples [(e, i10, r)], retract, curl) from the first nonzero PWM to the host stop."""
    tr = mv['trace']
    start = next((n for n, x in enumerate(tr) if x[2] != 0), None)
    if start is None:
        return None
    sign = 1 if tr[start][2] > 0 else -1
    hip = mv['joint'].endswith('HL')
    curl = (sign < 0) if hip else (sign > 0)
    c0 = tr[start - 1][1] if start > 0 else tr[start][1]
    out = []
    for n in range(start, len(tr)):
        t, c, spwm, cur, en = tr[n]
        if spwm == 0:
            break
        e = (c0 - c) if sign > 0 else (c - c0)
        out.append((e, int(round(cur * 10)), 1 if abs(spwm) >= mv['pwm'] else 0))
    return out, sign < 0, curl


def judge(samples, retract, curl, params):
    d = det.EndDetector(1, retract, curl, params)
    for k, (e, i, r) in enumerate(samples):
        ev = d.step(e, i, 1 if r else 0)
        if ev:
            return ev.kind, k, e, ev.info
    return 'none', -1, 0, {}


def cut_variant(samples, cut):
    """The same approach, but the drive starts `cut` full-PWM samples later (ramp kept)."""
    full = next(n for n, s in enumerate(samples) if s[2])
    if full + cut >= len(samples):
        return None
    ramp = samples[:full]
    base = ramp[-1][0] if ramp else 0
    prev = samples[full + cut - 1][0] if full + cut > 0 else 0
    tail = [(base + e - prev, i, r) for e, i, r in samples[full + cut:]]
    return pad(ramp + tail)


def pad(v):
    """Hold the final count for 3 s more, cycling the last 8 currents (the host stopped some presses early)."""
    last = v[-8:]
    return v + [(v[-1][0], last[n % len(last)][1], 1) for n in range(60)]


def load_all():
    moves = []   # (name, samples, retract, curl, expected set, category)
    golds = json.load(open(os.path.join(SP, 'ced-design', 'goldens.json')))
    for g in golds:
        moves.append((g['move'], g['joint'], g['pwm'], [tuple(s) for s in g['samples']], g['retract'], g['curl'],
                      {g['verdict']['kind']}, 'golden'))
    for log in ['endcur-111808.log', 'endcur-112315.log'] + TODAY:
        for idx, mv in enumerate(pm.moves(os.path.join(SP, log))):
            r = move_samples(mv)
            if r is None:
                continue
            s, retract, curl = r
            name = f"{log[:-4]}#{idx}"
            moves.append((name, mv['joint'], mv['pwm'], s, retract, curl, None,
                          'endcur' if log.startswith('endcur') else 'today'))
    return moves


def main():
    args = [a for a in sys.argv[1:] if a != '-v']
    verbose = '-v' in sys.argv
    base = det.Params()
    params = det.Params()
    for a in args:
        n, v = a.split('=')
        cur = getattr(params, n)
        setattr(params, n, (v.lower() in ('1', 'true')) if isinstance(cur, bool) else type(cur)(v))
    moves = load_all()
    bad = {}
    rows = []
    for name, joint, pwm, s, retract, curl, exp, cat in moves:
        got = judge(s, retract, curl, params)
        if exp is None:
            b = judge(s, retract, curl, base)
            exp = {b[0]}
            if (name.split('#')[0] + '.log', int(name.split('#')[1])) == F1:
                exp = {'stuck'}
        ok = got[0] in exp
        bad.setdefault(cat, [0, 0])[0 if ok else 1] += 1
        if not ok or verbose:
            rows.append(f"{cat:7} {name:40} {joint} {pwm:3} {'R' if retract else 'E'} {'curl' if curl else 'uncurl'} "
                        f"exp={'/'.join(sorted(exp))} got={got[0]} k={got[1]} e={got[2]} {got[3]}")
        # short-drive variants of every real end approach at PWM >= 120 (a curl that ended at its end)
        if cat != 'golden' and curl and pwm >= 120 and got[0] in ('mech', 'switch') and \
                (name.split('#')[0] + '.log', int(name.split('#')[1]) if '#' in name else -1) != F1:
            full = next(n for n, x in enumerate(s) if x[2])
            endE = max(x[0] for x in s)
            band = []
            for cut in range(0, 400):
                v = cut_variant(s, cut)
                if v is None:
                    break
                total = max(x[0] for x in v)
                if total < 14:
                    break
                g = judge(v, retract, curl, params)
                good = g[0] in ('mech', 'switch', 'static')
                band.append('M' if g[0] == 'mech' else 'W' if g[0] == 'switch' else 's' if g[0] == 'static'
                            else 'X' if g[0] == 'stuck' else '-')
                cat2 = 'short'
                bad.setdefault(cat2, [0, 0])[0 if good else 1] += 1
                if not good:
                    rows.append(f"short   {name:40} cut={cut} total={total} got={g[0]} k={g[1]} e={g[2]} {g[3]}")
            if verbose:
                rows.append(f"band    {name:40} {''.join(band)[::-1]}")
    # F1 shortened: the stall after fewer edges of curl (a later start on the same run)
    f1 = next(m for m in moves if m[0] == 'clearance-FR-115750#7')
    s = f1[3]
    band = []
    for cut in range(0, 400):
        v = cut_variant(s, cut)
        if v is None or max(x[0] for x in v) < 14:
            break
        g = judge(v, True, True, params)
        good = g[0] == 'stuck'
        band.append('M' if g[0] == 'mech' else 'W' if g[0] == 'switch' else 's' if g[0] == 'static'
                    else 'X' if g[0] == 'stuck' else '-')
        bad.setdefault('f1short', [0, 0])[0 if good else 1] += 1
    rows.append(f"f1band  {''.join(band)[::-1]}  (left = shortest run)")
    for r in rows:
        print(r)
    for cat, (o, b) in bad.items():
        print(f"{cat:8} OK {o:4d} BAD {b:4d}")


if __name__ == '__main__':
    main()
