"""Extract every jog move (traced or not) from the endcur .log files.

A move is the span from the first nonzero "J<joint> <pwm>" to the "J<joint> 0"
that ends it. Per FRONT telemetry frame of that joint inside the span (plus
0.5 s after the stop) we keep (host_t, count, signed_pwm, current, en).
Fields of a joint segment: name pos pot current enL enR pwmL pwmR hallEdges conn.
"""
import json
import re
import sys

LINE = re.compile(r"\s*([\d.]+) (TX|RX) (.*)$")


def parse_log(path):
    frames = []   # (t, {name: (count, spwm, cur, en)})
    cmds = []     # (t, joint, pwm)
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            t, d, body = float(m.group(1)), m.group(2), m.group(3)
            if d == "TX":
                j = re.match(r"J([A-Z]+) (-?\d+)", body)
                if j:
                    cmds.append((t, j.group(1), int(j.group(2))))
                elif body.strip() == "H":
                    cmds.append((t, "*", 0))
                continue
            if not (body.startswith("FRONT;") or body.startswith("LEFT ;") or body.startswith("RIGHT;")):
                continue
            seg = {}
            for s in body.split(";")[1:]:
                p = s.split()
                if len(p) >= 10 and len(p[0]) == 4:
                    try:
                        seg[p[0]] = (int(p[8]), int(p[7]) - int(p[6]), float(p[3]), int(p[4]))
                    except ValueError:
                        pass
            if seg:
                frames.append((t, seg))
    return frames, cmds


def moves(path):
    frames, cmds = parse_log(path)
    out = []
    active = {}
    for t, joint, pwm in cmds:
        if joint == "*":
            for j, st in list(active.items()):
                out.append((j, st[0], t, st[1]))
            active.clear()
            continue
        if pwm != 0:
            if joint not in active:
                active[joint] = [t, pwm]
            else:
                active[joint][1] = pwm if abs(pwm) > abs(active[joint][1]) else active[joint][1]
        elif joint in active:
            st = active.pop(joint)
            out.append((joint, st[0], t, st[1]))
    result = []
    for joint, t0, t1, peak in sorted(out, key=lambda x: x[1]):
        tr = [(round(t, 3),) + fr[joint] for t, fr in frames if t0 - 0.12 <= t <= t1 + 0.5 and joint in fr]
        result.append(dict(joint=joint, t_start=t0, t_stop=t1, pwm=abs(peak),
                           direction=None, trace=tr))
    return result


def direction(joint, signed_pwm):
    # hips curl with -PWM (retract), knees with +PWM (extend)
    hip = joint.endswith("HL")
    curl_sign = -1 if hip else 1
    return "curl" if (signed_pwm > 0) == (curl_sign > 0) else "uncurl"


if __name__ == "__main__":
    for path in sys.argv[1:]:
        for i, mv in enumerate(moves(path)):
            tr = mv["trace"]
            sp = [x[2] for x in tr if x[2] != 0]
            mv["direction"] = direction(mv["joint"], sp[0]) if sp else "?"
            print(i, mv["joint"], mv["direction"], mv["pwm"], round(mv["t_start"], 2), round(mv["t_stop"], 2),
                  len(tr), tr[0][1] if tr else None, "->", tr[-1][1] if tr else None)
