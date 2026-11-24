import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen
name = sys.argv[2]; lo = int(sys.argv[3]); hi = int(sys.argv[4]); tags = sys.argv[5].split(",") if len(sys.argv) > 5 else ["A0"]
for m in common.load(sys.argv[1]):
    if m["name"] != name: continue
    v = gen.variants(m)
    for tag in tags:
        E, I, R = v[tag]
        print(tag, "n=", len(E))
        print(" ".join(f"{k}:{E[k]}/{I[k]}{'' if R[k] else 'r'}" for k in range(max(0, lo), min(hi, len(E)))))
    print("t:", " ".join(f"{x[0]-m['drv'][0][0]:.2f}" for x in m["drv"][max(0,lo):hi]))
