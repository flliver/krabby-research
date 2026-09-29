import numpy as np, glob
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
f=sorted(glob.glob(S+'/raw/*.npz'))[3]; z=np.load(f); c=z['foot_force_norm']>0.1; seg=z['segment_id']
print('segments', [(int(s),int((seg==s).sum())) for s in np.unique(seg)])
m=seg==1
print('contact frac in creep', c[m].mean(0).round(2)); print('contact frac in stand', c[seg==0].mean(0).round(2))
idle=np.zeros(6); mx=np.zeros(6)
for t in range(len(c)):
    idle=np.where(c[t],0,idle+1)
    if m[t]: mx=np.maximum(mx,idle)
print('max idle in creep per foot', mx)
print('steady in creep', z['steady_mask'][m].mean())
