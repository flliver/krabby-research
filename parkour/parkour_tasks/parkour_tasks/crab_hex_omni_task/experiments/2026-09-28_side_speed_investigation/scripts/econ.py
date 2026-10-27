import sys, json, importlib.util, math
from pathlib import Path
import numpy as np
T='/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/replay_stroke_posture.py'
spec=importlib.util.spec_from_file_location('rsp',T); RSP=importlib.util.module_from_spec(spec); sys.modules['rsp']=RSP; spec.loader.exec_module(RSP)
REC=RSP.REC
stamp=sys.argv[1]; econ_yaml=sys.argv[2] if len(sys.argv)>2 else str(RSP.W_ENV_YAML)
out=sys.argv[3] if len(sys.argv)>3 else None
econ,info=RSP.load_economy(econ_yaml)
roll=REC.load_suite_rollout(stamp)
labels=list(roll['meta']['hold_labels']); jn=list(roll['meta']['joint_names'])
CATS=('fwd','side','back','diag')
rows=[]
for ep in roll['episodes']:
    if ep['category'] not in CATS: continue
    z=ep['z']
    p=REC.episode_primitives(z,action_clip=roll['action_clip'],hold_labels=labels)
    p['tibia']=RSP.tibia_deviation(z,jn,info)
    m=REC.hold_mask(p,ep['hold_idx'])
    if m.sum()<50: continue
    with RSP.extra_terms():
        vals=REC.term_values(p,econ)
    tot=sum(vals.values())
    cmd=z['cmd_applied'][:,:2].astype(float); v=z['root_lin_vel_b'][:,:2].astype(float)
    sp=np.linalg.norm(cmd,axis=1); d=cmd/np.maximum(sp,1e-6)[:,None]
    along=(v*d).sum(1); cross=v[:,0]*(-d[:,1])+v[:,1]*d[:,0]
    r={'cat':ep['category'],'hold':ep['hold'],'n':int(m.sum()),'cmd':float(sp[m].mean()),'speed':float(along[m].mean()),
       'cross_rms':float(np.sqrt((cross[m]**2).mean())),
       'pre':float(tot[m].mean()),'post':float(np.clip(tot[m],0,None).mean()),'bind':float((tot[m]<0).mean())}
    for k,vv in vals.items(): r[k]=float(vv[m].mean())
    # counterfactual +dv along command, tracking terms only; delivered with clip
    for dv in (0.05,):
        v2=v+dv*d
        e2=cmd-v2; esq=(e2**2).sum(1); el1=np.sqrt(esq)
        lin2=econ.track_lin_w*np.exp(-esq/econ.track_lin_std**2)
        l12=econ.l1_w*el1
        prog2=econ.progress_w*np.clip((v2*d).sum(1),0,1.05)*(sp>0.12)
        dtrack=(lin2-vals['track_lin_exp'])+(l12-vals['L1'])+(prog2-vals['progress'])
        tot2=tot+dtrack
        r['cf_lin']=float((lin2-vals['track_lin_exp'])[m].mean()); r['cf_l1']=float((l12-vals['L1'])[m].mean()); r['cf_prog']=float((prog2-vals['progress'])[m].mean())
        r['cf_nominal']=float(dtrack[m].mean())
        r['cf_delivered']=float((np.clip(tot2,0,None)-np.clip(tot,0,None))[m].mean())
    rows.append(r)
if out: json.dump(rows,open(out,'w'))
keys=[k for k in rows[0] if k not in('cat','hold')]
print('econ',econ_yaml.split('/')[-3], 'stamp',stamp.split('/')[-4:])
print(f"{'key':16s}"+''.join(f"{c:>10s}" for c in CATS))
for k in keys:
    print(f"{k:16s}"+''.join(f"{np.mean([r[k] for r in rows if r['cat']==c]):10.4f}" if any(r['cat']==c for r in rows) else f"{'-':>10s}" for c in CATS))
# within-category regression of each term vs speed
print('\nwithin-category slope d(term)/d(speed) per (m/s), [r]:')
for c in ('fwd','side'):
    R=[r for r in rows if r['cat']==c]; s=np.array([r['speed'] for r in R])
    line=[]
    for k in keys:
        if k in('speed','n','cmd'): continue
        y=np.array([r[k] for r in R])
        if np.std(y)<1e-9: continue
        b=np.polyfit(s,y,1)[0]; rr=np.corrcoef(s,y)[0,1]
        line.append(f"{k}:{b:+.3f}[{rr:+.2f}]")
    print(c,f"n={len(R)} speed sd={s.std():.3f}"); print('  '+'  '.join(line))
