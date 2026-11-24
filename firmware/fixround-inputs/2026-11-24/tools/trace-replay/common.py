import json
def load(path):
    mv = json.load(open(path))
    out = []
    for m in mv:
        tr = m["trace"]
        i0 = next((i for i, x in enumerate(tr) if x[2] != 0), None)
        if i0 is None: continue
        sign = 1 if tr[i0][2] > 0 else -1
        i1 = i0
        while i1 < len(tr) and tr[i1][2] != 0 and (tr[i1][2] > 0) == (sign > 0): i1 += 1
        drv = tr[i0:i1]
        post = tr[i1:i1 + 25]
        # post must not include the next drive
        pz = []
        for x in post:
            if x[2] != 0: break
            pz.append(x)
        peak = max(abs(x[2]) for x in drv)
        c0 = tr[i0 - 1][1] if i0 > 0 else (m["pre"][-1][1] if m["pre"] else drv[0][1])
        m2 = dict(m); m2.update(sign=sign, drv=drv, post=pz, peak=peak, c0=c0,
                               hip=m["joint"].endswith("HL"), yaw=m["joint"].endswith("HY"))
        m2["retract"] = sign < 0
        m2["curl"] = (sign < 0) if m2["hip"] else (sign > 0)
        m2["name"] = m["log"].replace(".log", "") + "#" + str(m["idx"])
        out.append(m2)
    return out
