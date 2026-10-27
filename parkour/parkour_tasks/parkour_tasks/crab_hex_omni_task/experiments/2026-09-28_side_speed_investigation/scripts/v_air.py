import numpy as np, glob
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
for lo in (0,150):
  runs=[];eps90=0;n=0
  for f in sorted(glob.glob(S+'/raw/*.npz')):
    z=np.load(f); c=z['foot_force_norm']>0.1; seg=z['segment_id']; idx=np.where((seg==1)&z['steady_mask'])[0]
    if len(idx)<200: continue
    n+=1; a0=idx[lo]; a1=idx[-1]; big=False
    for j in range(6):
        air=~c[:,j]; t=a0
        # runs that START within window
        k=a0
        while k<=a1:
            if air[k] and (k==0 or not air[k-1]):
                e=k
                while e<len(air) and air[e]: e+=1
                runs.append(e-k); big|= (e-k)>90; k=e
            else: k+=1
    eps90+=big
  r=np.array(runs); print('skip',lo,'nruns',len(r),'p50 %d p90 %d frac>90 %.3f frac60-79 %.2f eps_with>90 %.2f'%(np.median(r),np.percentile(r,90),(r>90).mean(),((r>=60)&(r<80)).mean(),eps90/n))
