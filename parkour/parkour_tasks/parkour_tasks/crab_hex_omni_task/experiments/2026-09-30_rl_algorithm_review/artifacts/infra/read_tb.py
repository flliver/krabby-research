import sys
from tensorboard.backend.event_processing import event_accumulator

path = sys.argv[1]
tags = sys.argv[2:]
ea = event_accumulator.EventAccumulator(path, size_guidance={'scalars': 0})
ea.Reload()
available = ea.Tags().get('scalars', [])
for t in tags:
    if t not in available:
        print(f"{t}: NOT FOUND (available tags sample: {available[:10]})")
        continue
    events = ea.Scalars(t)
    n = len(events)
    # print first 5, and some spaced samples
    idxs = sorted(set([0,1,2,3,4, n//4, n//2, 3*n//4, n-1]))
    idxs = [i for i in idxs if 0 <= i < n]
    print(f"=== {t} (n={n}) ===")
    for i in idxs:
        e = events[i]
        print(f"  step={e.step} value={e.value}")
