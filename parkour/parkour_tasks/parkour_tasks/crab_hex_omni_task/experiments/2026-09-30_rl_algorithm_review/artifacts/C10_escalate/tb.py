import glob, os, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
runs={"X03_M1":"2026-09-23_04-25-38_seed3_X03_c6ft_A15pB","X01_ctl":"2026-09-23_02-38-36_seed3_X01_c6ft_A15pB","X02_ctl":"2026-09-23_03-32-07_seed2_X02_c6ft_A15pB","X08c6_parent":"2026-09-22_11-58-07_seed3_X08_c6_A15pB"}
for k,r in runs.items():
    ea=EventAccumulator(base+r,size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    print("==",k,r); print("  tags:",sorted(tags))
    lr=ea.Scalars('Loss/learning_rate'); v=np.array([s.value for s in lr]); st=np.array([s.step for s in lr])
    off=(v>1.0e-5*1.001)
    print(f"  LR n={len(v)} steps {st[0]}-{st[-1]} off_floor={off.sum()} ({off.mean():.4f}) mean={v.mean():.3e} max={v.max():.3e} first_val={v[0]:.3e}")
    u,c=np.unique(np.round(v/1e-5,3),return_counts=True); print("  LR levels(x1e-5):",dict(zip(u,c)))
    std=ea.Scalars('Policy/mean_noise_std'); sv=np.array([s.value for s in std])
    print(f"  std first={sv[0]:.4f} last={sv[-1]:.5f} min={sv.min():.4f} max={sv.max():.4f}; first idx>=0.4999: {int(np.argmax(sv>=0.4999)) if (sv>=0.4999).any() else None}")
    for t in ['Train/mean_episode_length','Train/mean_reward']:
        if t in tags:
            x=np.array([s.value for s in ea.Scalars(t)]); print(f"  {t}: last500 mean={x[-500:].mean():.1f} first500 mean={x[:500].mean():.1f}")
    kl=[t for t in tags if 'kl' in t.lower()]; print("  kl tags:",kl)
