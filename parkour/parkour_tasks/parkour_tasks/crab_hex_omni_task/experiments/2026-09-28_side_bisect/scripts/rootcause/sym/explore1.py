import json,numpy as np,glob,sys
from pathlib import Path
from scipy.signal import welch
def load(d):
    d=Path(d); m=json.loads((d/'run_meta.json').read_text()); return d,m
for name,d in [('C','/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect/seed3_X02_s1_A15pB/side22_25__omni/seed001/2026-09-29_07-00-57'),
               ('O','/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect/seed3_X01_s1_A15pB/side22_25__omni/seed001/2026-09-29_05-50-19'),
               ('R','/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect/seed3_X03_s1_A15pB/side22_25__omni/seed001/2026-09-29_13-36-31'),
               ('W','/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect/harness_W/side22_25__omni/seed001/2026-09-29_04-23-49')]:
    d,m=load(d); labels=m['hold_labels']
    print('==',name)
    cnt=0
    for e in m['env_index_map']:
        if e['label']!='lat_l_022': continue
        z=np.load(d/'raw'/f"episode_{e['env_index']:02d}.npz")
        done=z['done'].astype(bool); S=int(np.argmax(done)) if done.any() else len(done)
        seg=z['segment_id']; sid=labels.index('lat_l_022'); idx=np.where((seg==sid)&(np.arange(len(seg))<S))[0]
        if len(idx)<500: continue
        t0=idx[0]
        sl=slice(t0+200,t0+500)
        a=z['actions'][sl]; ph=z['clock_phase'][sl]; cr=z['clock_running'][sl]
        dph=np.diff(np.unwrap(ph))/0.02/(2*np.pi)
        h=z['foot_pos_w'][sl,:,2]-z['terrain_z'][sl]
        f,P=welch(h-h.mean(0),fs=50,nperseg=256,axis=0); pk=f[np.argmax(P.sum(1))]
        F=(z['foot_force_norm'][sl]>50).astype(float)
        f2,P2=welch(F-F.mean(0),fs=50,nperseg=256,axis=0); pk2=f2[np.argmax(P2.sum(1))]
        print(f" ep{e['env_index']} clock_rate {np.median(dph):.3f}Hz running {cr.mean():.2f} footh_peak {pk:.2f} contact_peak {pk2:.2f}Hz |a|max {np.abs(a).max():.1f} frac>1 {(np.abs(a)>1).mean():.2f} ph [{ph.min():.2f},{ph.max():.2f}] obs0 {z['clock_obs'][t0+200]} sin/cos(ph) {np.sin(ph[0]):.2f},{np.cos(ph[0]):.2f}")
        cnt+=1
        if cnt>=4: break
