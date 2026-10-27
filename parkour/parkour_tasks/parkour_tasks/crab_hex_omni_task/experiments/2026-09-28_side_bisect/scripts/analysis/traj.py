import glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
root='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
names={'X01':'O','X02':'C','X03':'R','X04':'C_mo','X05':'C_band','X06':'C_rsi','X07':'C_s2','X08':'C_mo_s2','X09':'C_mo_s1','X10':'LRfa','X11':'LR4','X12':'LR4_dwell','X14':'C_lr','X15':'C_lr_10'}
bins=[(0,200),(200,400),(400,600),(600,800),(800,1000),(1000,1500),(1500,2000),(2000,2500),(2500,3000)]
print('arm        '+' '.join(f'{a//100:>2}-{b//100:<2}' for a,b in bins), ' | symm loss 800-1000, 2500-3000')
for d in sorted(glob.glob(root+'2026-09-29_*_s1_A15pB')+glob.glob(root+'2026-09-30_*_s1_A15pB')):
    tag=[k for k in names if k in d]
    if not tag: continue
    ea=E(glob.glob(d+'/events*')[0],size_guidance={'scalars':0});ea.Reload()
    ev=ea.Scalars('Metrics/base_parkour/track_ratio_move_mean')
    s=np.array([e.step for e in ev]);v=np.array([e.value for e in ev])
    row=[np.mean(v[(s>=a)&(s<b)]) if ((s>=a)&(s<b)).any() else np.nan for a,b in bins]
    sy=ea.Scalars('Loss/symmetry') if 'Loss/symmetry' in ea.Tags()['scalars'] else []
    ss=np.array([e.step for e in sy]);sv=np.array([e.value for e in sy])
    sym=[np.mean(sv[(ss>=a)&(ss<b)]) if len(sv) else np.nan for a,b in [(800,1000),(2500,3000)]]
    print(f'{names[tag[0]]:10s} '+' '.join(f'{x:5.3f}' for x in row), f' | {sym[0]:.4f} {sym[1]:.4f}')
