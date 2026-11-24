import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, gen
want = {"f1": "clearance-FR-115750#7", "hipout": "clearance-FR-115750#3", "knee": "fr-recover-120231#2",
        "realsw150": "fr-recover-120231#1", "realsw128": "rodscale-hip_back-114914#1", "realsw128b": "hipend-FL-115546#1"}
mv = {m["name"]: m for m in common.load(sys.argv[1])}
with open(sys.argv[2], "w", newline="\n") as f:
    for k, nm in want.items():
        E, I, R = gen.base(mv[nm])
        f.write(f"{k} {len(E)}\n" + " ".join(f"{e} {i}" for e, i in zip(E, I)) + "\n")
