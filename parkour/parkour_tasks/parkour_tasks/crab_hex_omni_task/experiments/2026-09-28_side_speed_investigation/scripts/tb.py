import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    ea=EventAccumulator(d,size_guidance={'scalars':0}); ea.Reload()
    tags=[t for t in ea.Tags()['scalars'] if t.startswith('Episode_Reward/') or 'mean_reward' in t or 'episode_length' in t.lower()]
    print(d.split('/')[-1])
    for t in sorted(tags):
        s=ea.Scalars(t); v=[x.value for x in s[-100:]]
        print(f"  {t:60s} {sum(v)/len(v):9.4f}  (n={len(s)})")
