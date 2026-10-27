import glob, numpy as np, sys
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
R="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/"
for run in sys.argv[1:]:
    f=glob.glob(R+run+"/events*")[0]; s={}
    for raw in RawEventFileLoader(f).Load():
        ev=event_pb2.Event.FromString(raw)
        for v in ev.summary.value:
            if v.tag in ("Train/mean_reward","Train/mean_episode_length") or "terrain_levels" in v.tag or v.tag.startswith("Episode_Termination") or v.tag.startswith("Metrics/base_velocity/error_vel") or v.tag.startswith("Episode_Reward/"):
                s.setdefault(v.tag,[]).append(v.simple_value)
    n=len(s["Train/mean_reward"]); q=[int(n*x) for x in (0.02,0.1,0.25,0.5,0.75,0.99)]
    print("==",run,n)
    def m(t,i): a=np.array(s[t]); return a[max(0,i-25):i+25].mean()
    for t in ["Train/mean_reward","Train/mean_episode_length"]+[t for t in s if "terrain" in t or "Termination" in t or "error_vel" in t]:
        print("  %-55s"%t[-55:], " ".join("%9.3f"%m(t,i) for i in q))
    # biggest movers among reward terms between 25% and 99%
    mv=sorted(((m(t,q[-1])-m(t,q[2]),t) for t in s if t.startswith("Episode_Reward/")), key=lambda x:x[0])
    print("  top reward-term drops (25%->end):", [(t.split('/')[-1],round(d,3)) for d,t in mv[:4]])
