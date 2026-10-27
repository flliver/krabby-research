import sys, json, importlib.util, glob
import numpy as np
T='/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/replay_stroke_posture.py'
spec=importlib.util.spec_from_file_location('rsp',T); RSP=importlib.util.module_from_spec(spec); sys.modules['rsp']=RSP; spec.loader.exec_module(RSP)
REC=RSP.REC
from dataclasses import replace
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
meta=json.load(open(S+'/run_meta.json')); jn=meta['joint_names']; ty=meta['travel_yaw']
econW,info=RSP.load_economy(str(RSP.W_ENV_YAML))
# side head's own economy: std sqrt(.1), l1 -1, clock 1/1 sum
econS=replace(econW,clock=(1.0,1.0),combine='product')
res={}
for cmd_mag in (0.22,):
  for ename,econ in (('W',econW),('S',econS)):
    acc=[]
    for f in sorted(glob.glob(S+'/raw/episode_*.npz')):
        z=dict(np.load(f))
        c=z['cmd_applied'].astype(np.float64).copy()
        cb=c.copy(); cs,sn=np.cos(ty),np.sin(ty)
        cb[:,0]=cs*c[:,0]-sn*c[:,1]; cb[:,1]=sn*c[:,0]+cs*c[:,1]
        if cmd_mag is not None:
            n=np.linalg.norm(cb[:,:2],axis=1); k=np.where(n>0.12,cmd_mag/np.maximum(n,1e-9),1.0); cb[:,:2]*=k[:,None]
        z['cmd_applied']=cb.astype(np.float32)
        z['heading_err']=np.zeros(len(c),np.float32)
        p=REC.episode_primitives(z,action_clip=1.0,hold_labels=meta['hold_labels'])
        p['tibia']=RSP.tibia_deviation(z,jn,info)
        m=REC.hold_mask(p,1)  # creep
        if m.sum()<50: continue
        with RSP.extra_terms(): vals=REC.term_values(p,econ)
        tot=sum(vals.values())
        v=z['root_lin_vel_b'][:,:2].astype(float); d=cb[:,:2]/np.maximum(np.linalg.norm(cb[:,:2],axis=1),1e-9)[:,None]
        r={'speed':float((v*d).sum(1)[m].mean()),'cmd':float(np.linalg.norm(cb[:,:2],axis=1)[m].mean()),'pre':float(tot[m].mean()),'post':float(np.clip(tot[m],0,None).mean()),'bind':float((tot[m]<0).mean())}
        for k,vv in vals.items(): r[k]=float(vv[m].mean())
        acc.append(r)
    res[(cmd_mag,ename)]={k:float(np.mean([a[k] for a in acc])) for k in acc[0]}
    print('cmd',cmd_mag,'econ',ename,'n',len(acc))
keys=list(next(iter(res.values())).keys())
print(f"{'key':14s}"+''.join(f"{str(k[0])+'/'+k[1]:>12s}" for k in res))
for k in keys: print(f"{k:14s}"+''.join(f"{res[kk][k]:12.4f}" for kk in res))
