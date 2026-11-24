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

