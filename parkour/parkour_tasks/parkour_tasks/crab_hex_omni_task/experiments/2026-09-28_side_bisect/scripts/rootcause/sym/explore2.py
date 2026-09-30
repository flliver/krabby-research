import json,numpy as np
from pathlib import Path
d=Path('/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect/seed3_X02_s1_A15pB/side22_25__omni/seed001/2026-09-29_07-00-57')
m=json.loads((d/'run_meta.json').read_text()); labels=m['hold_labels']
z=np.load(d/'raw'/'episode_103.npz')
seg=z['segment_id']; sid=labels.index('lat_l_022'); idx=np.where(seg==sid)[0]; t0=idx[0]
ph=np.unwrap(z['clock_phase']); print('phase advance over hold [4,10):',(ph[t0+500]-ph[t0+200])/(2*np.pi),'cycles in 6 s; clock_cmd_speed',z['clock_cmd_speed'][t0+300])
F=(z['foot_force_norm']>50).astype(int); h=z['foot_pos_w'][:,:,2]-z['terrain_z']
np.set_printoptions(linewidth=250)
for t in range(t0+200,t0+320,3):
    print(t-t0, f"{z['clock_phase'][t]:.2f}", F[t], np.round(h[t]*100).astype(int), np.round(np.clip(z['actions'][t],-1,1)[6:]).astype(int))
