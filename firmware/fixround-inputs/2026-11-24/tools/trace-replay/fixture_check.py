import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen
src = open("C:/Users/nickm/krabby/krabby-research/tests/native/firmware/arduino/src/actuator/robot_end_current_replay.hpp", encoding="utf-8").read()
arrays = {}
for m in re.finditer(r"static const RobotEndSample (kRobotEnd\d+)\[\] = \{(.*?)\n\};", src, re.S):
    arrays[m.group(1)] = [tuple(int(x) for x in t.split(",")) for t in re.findall(r"\{(-?\d+, \d+, \d)\}", m.group(2))]
tab = src[src.index("kRobotEndMoves[]"):]
rows = re.findall(r'\{"([^"]+)", RobotEndGroup::(\w+), "(\w+)", (\d+), (true|false), (true|false), CalibrationEnd::(\w+), (-?\d+), (-?\d+), (kRobotEnd\d+),', tab)
print(len(arrays), "arrays", len(rows), "rows")
moves = {}
for m in common.load("out/moves.json"):
    nm = m["name"]
    short = nm.replace("endcur-", "") if nm.startswith("endcur") else nm
    moves[short] = m
ok = bad = miss = 0
import collections
kinds = collections.Counter()
for name, group, joint, pwm, retract, curl, exp, k, edges, arr in rows:
    kinds[(group, exp)] += 1
    if group == "Short": continue
    m = moves.get(name)
    if m is None:
        miss += 1; print("no move for", name); continue
    E, I, R = gen.base(m)
    mine = list(zip(E, [10 * v for v in I], R))
    fx = arrays[arr]
    same = mine[:len(fx)] == fx
    meta = (m["joint"] == joint and m["peak"] == int(pwm) and m["retract"] == (retract == "true") and m["curl"] == (curl == "true"))
    if same and meta: ok += 1
    else:
        bad += 1
        d = next((i for i, (a, b) in enumerate(zip(mine, fx)) if a != b), None)
        print("DIFF", name, joint, pwm, "len fx", len(fx), "mine", len(mine), "first diff at", d, mine[d] if d is not None and d < len(mine) else None, fx[d] if d is not None else None, "meta", meta)
print("identical:", ok, "different:", bad, "missing:", miss)
print(sorted(kinds.items()))
