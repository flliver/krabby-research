"""Replay the end detector over every jog move in the endcur logs (and the JSON traces).

Usage: python -I replay.py [--json] [--param name=value ...] LOG...
Each telemetry frame of the moving joint is one 50 ms detector sample (the firmware
emits telemetry every >= 50 ms, 51.5 ms on average). Samples start at the first
frame with a nonzero PWM and stop at the host's J 0 (the motor stops there).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detector as det  # noqa: E402
import parse_moves as pm  # noqa: E402


def expected(mv, idx_in_log, log):
    """Ground truth from the 2026-11-24 observations."""
    j, d, pwm = mv["joint"], mv["direction"], mv["pwm"]
    tr = mv["trace"]
    driven = [x for x in tr if x[2] != 0]
    travel = abs(driven[-1][1] - driven[0][1]) if driven else 0
    if travel <= 3:
        return "static"
    if d == "uncurl" and j.endswith("HL"):
        return "none"                       # host stopped at 150 counts, still running
    if j == "FRKL" and d == "uncurl" and pwm == 80 and travel < 200:
        return "none"                       # FRKL out 150 at 80 reached 150
    if j == "FLKL" and d == "uncurl" and pwm == 80:
        return "stuck"                      # stalls ~100 out, no contact
    if j == "FRKL" and d == "uncurl" and pwm == 80:
        return "stuck"                      # crawled to 771, 18 short of the switch
    if j.endswith("HL") and d == "curl":
        return "switch"
    if j.endswith("KL") and d == "uncurl":
        return "switch"
    return "mech"


def run_move(mv, params):
    tr = mv["trace"]
    start = next((n for n, x in enumerate(tr) if x[2] != 0), None)
    if start is None:
        return None
    sign = 1 if tr[start][2] > 0 else -1          # +PWM extends (count falls)
    retract = sign < 0
    hip = mv["joint"].endswith("HL")
    curl = (sign < 0) if hip else (sign > 0)
    c0 = tr[start - 1][1] if start > 0 else tr[start][1]
    d = det.EndDetector(mv["pwm"], retract, curl, params)
    fired = None
    driven = []
    for n in range(start, len(tr)):
        t, c, spwm, cur, en = tr[n]
        if spwm == 0:
            break
        edges = (c0 - c) if sign > 0 else (c - c0)
        driven.append((t, c, edges, cur))
        ev = d.step(edges, int(round(cur * 10)), abs(spwm))
        if ev and not fired:
            fired = (ev, t, c)
            break
    # truth from the full trace while driven
    full = [x for x in tr[start:] if x[2] != 0]
    final_c = full[-1][1]
    last_change = max((n for n in range(1, len(full)) if full[n][1] != full[n - 1][1]), default=0)
    t_last_change = full[last_change][0]
    t0 = tr[start][0]
    res = dict(joint=mv["joint"], dir="curl" if curl else "uncurl", pwm=mv["pwm"], n=len(full),
               start_c=c0, final_c=final_c, stop_t=round(t_last_change - t0, 2),
               host_stop_t=round(full[-1][0] - t0, 2), armed=d.armed,
               i_ref=d.i_ref, v_ref=d.v_ref)
    if fired:
        ev, t, c = fired
        remaining = (c - final_c) if sign > 0 else (final_c - c)
        res.update(kind=ev.kind, fire_t=round(t - t0, 2), fire_k=ev.k, fire_c=c, remaining=remaining,
                   after_stop=round(t - t_last_change, 2), info=ev.info)
    else:
        res.update(kind="none")
    return res


def main():
    args = sys.argv[1:]
    params = det.Params()
    logs = []
    use_json = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--param":
            name, val = args[i + 1].split("=")
            cur = getattr(params, name)
            setattr(params, name, type(cur)(val))
            i += 2
            continue
        if a == "--json":
            use_json = True
            i += 1
            continue
        logs.append(a)
        i += 1
    ok = bad = 0
    rows = []
    for log in logs:
        mvs = pm.moves(log)
        for idx, mv in enumerate(mvs):
            sp = [x for x in mv["trace"] if x[2] != 0]
            mv["direction"] = pm.direction(mv["joint"], sp[0][2]) if sp else "?"
            r = run_move(mv, params)
            if r is None:
                continue
            exp = expected(mv, idx, log)
            r["expected"] = exp
            r["log"] = os.path.basename(log)[7:13]
            r["idx"] = idx
            good = r["kind"] == exp
            r["ok"] = good
            ok += good
            bad += not good
            rows.append(r)
    for r in rows:
        line = (f"{r['log']}#{r['idx']:<2} {r['joint']} {r['dir']:<6} {r['pwm']:3d} exp={r['expected']:<6} "
                f"got={r['kind']:<6} {'OK ' if r['ok'] else 'BAD'} ")
        if r["kind"] != "none":
            line += (f"t={r['fire_t']:5.2f}s ({r['after_stop']:+.2f}s vs last edge) rec={r['fire_c']:4d} "
                     f"final={r['final_c']:4d} short={r['remaining']:+3d} ")
            info = r.get("info") or {}
            if "ratio" in info:
                line += f"ratio={info['ratio']} "
        else:
            line += f"(host stop at {r['host_stop_t']}s, last edge {r['stop_t']}s, final={r['final_c']})"
        line += f" iRef={r['i_ref']} vRef={r['v_ref']}"
        print(line)
    print(f"{ok} OK, {bad} BAD of {ok + bad}")
    if use_json:
        print(json.dumps(rows, default=str))


if __name__ == "__main__":
    main()
