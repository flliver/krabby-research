import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); ea.Reload()
tags = ea.Tags()["scalars"]
import re
pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else ".*")
for t in tags:
    if pat.search(t):
        v = ea.Scalars(t)
        pts = {e.step: e.value for e in v}
        sel = [s for s in (0, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 2999) if s in pts]
        print(t, len(v), " ".join(f"{s}:{pts[s]:.4g}" for s in sel))
