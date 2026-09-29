import numpy as np, glob
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
carry=[];reset=[]
for f in sorted(glob.glob(S+'/raw/*.npz')):
    z=np.load(f); c=z['foot_force_norm']>0.1; seg=z['segment_id']; m=(seg==1)&z['steady_mask']
    if z['done'].any(): m[np.argmax(z['done']):]=False
    T=len(c); idle=np.zeros(6); idle2=np.zeros(6); pc=np.zeros(T); pr=np.zeros(T)
    for t in range(T):
        if t>0 and seg[t]!=seg[t-1]: idle2[:]=0
        idle=np.where(c[t],0,idle+1); idle2=np.where(c[t],0,idle2+1)
        pc[t]=np.maximum(idle-90,0).sum(); pr[t]=np.maximum(idle2-90,0).sum()
    if m.sum()<50: continue
    carry.append(-0.12*pc[m].mean()); reset.append(-0.12*pr[m].mean())
carry=np.array(carry);reset=np.array(reset)
print('side head creep foot_idle/s: carry mean %.3f median %.3f p90 %.3f | reset-at-hold mean %.3f median %.3f; frac eps < -0.05: %.2f'%(carry.mean(),np.median(carry),np.percentile(carry,10),reset.mean(),np.median(reset),(carry<-0.05).mean()))
