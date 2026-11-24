"""Independent parser: extract every jog move from the robot logs.
A move = first nonzero 'J<joint> <pwm>' until 'J<joint> 0' or 'H'.
Frame fields per joint: name pos pot current enL enR pwmL pwmR hallEdges conn.
"""
import re, sys, json, os
LINE = re.compile(r"\s*([\d.]+) (TX|RX) (.*)$")
ROLE = {"F": "FRONT;", "M": None, "R": None}

def parse(path):
    cmds = []; frames = {}   # joint -> list of (t, count, spwm, cur, en, lineNo)
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            m = LINE.match(raw.rstrip("\n"))
            if not m: continue
            t = float(m.group(1)); body = m.group(3)
            if m.group(2) == "TX":
                j = re.match(r"J([A-Z]{4}) (-?\d+)\s*$", body)
                if j: cmds.append((t, j.group(1), int(j.group(2)), ln))
                elif body.strip() == "H": cmds.append((t, "*", 0, ln))
                continue
            if not (body.startswith("FRONT;") or body.startswith("LEFT ;") or body.startswith("RIGHT;")):
                continue
            for s in body.split(";")[1:]:
                p = s.split()
                if len(p) == 10 and len(p[0]) == 4 and p[0][0] in "FMR":
                    try:
                        frames.setdefault(p[0], []).append(
                            (t, int(p[8]), int(p[7]) - int(p[6]), int(float(p[3])), int(p[4]) | int(p[5]), ln))
                    except ValueError:
                        pass
    return cmds, frames

def moves(path):
    cmds, frames = parse(path)
    spans = []; active = {}
    for t, joint, pwm, ln in cmds:
        if joint == "*":
            for j, st in active.items(): spans.append((j, st[0], t, st[1], st[2]))
            active.clear(); continue
        if pwm != 0:
            if joint not in active: active[joint] = [t, [pwm], ln]
            else: active[joint][1].append(pwm)
        elif joint in active:
            st = active.pop(joint); spans.append((joint, st[0], t, st[1], st[2]))
    for j, st in active.items(): spans.append((j, st[0], None, st[1], st[2]))
    out = []
    for idx, (joint, t0, t1, pwms, ln) in enumerate(sorted(spans, key=lambda x: x[1])):
        fr = frames.get(joint, [])
        t_end = (t1 if t1 is not None else t0 + 1e9)
        pre = [x for x in fr if t0 - 0.6 <= x[0] < t0 - 0.02]
        tr = [x for x in fr if t0 - 0.02 <= x[0] <= t_end + 1.5]
        out.append(dict(log=os.path.basename(path), idx=idx, joint=joint, t0=t0, t1=t1, pwms=pwms, line=ln,
                        pre=pre[-3:], trace=tr))
    return out

if __name__ == "__main__":
    allm = []
    for p in sys.argv[2:]:
        allm += moves(p)
    json.dump(allm, open(sys.argv[1], "w"))
    print(len(allm), "moves")
