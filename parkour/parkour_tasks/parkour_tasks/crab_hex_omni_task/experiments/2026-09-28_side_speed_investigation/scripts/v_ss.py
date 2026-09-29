import sys, json, importlib.util, glob
import numpy as np
T='/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/replay_stroke_posture.py'
spec=importlib.util.spec_from_file_location('rsp',T); RSP=importlib.util.module_from_spec(spec); sys.modules['rsp']=RSP; spec.loader.exec_module(RSP)
REC=RSP.REC
SKIP=int(sys.argv[1])
econW,info=RSP.load_economy(str(RSP.W_ENV_YAML))
def idle_reset(z):
    c=z['foot_force_norm']>0.1; seg=z['segment_id']; T=len(c); idle=np.zeros(6); pr=np.zeros(T)
    for t in range(T):
        if t>0 and seg[t]!=seg[t-1]: idle[:]=0
        idle=np.where(c[t],0,idle+1); pr[t]=np.maximum(idle-90,0).sum()
    return -0.12*pr
def skipmask(m,skip):
    idx=np.where(m)[0]; m2=m.copy(); m2[idx[:skip]]=False; return m2
def price(z,labels,jn,hold_ids,skip,reset_idle):
    p=REC.episode_primitives(z,action_clip=1.0,hold_labels=labels); p['tibia']=RSP.tibia_deviation(z,jn,info)
    with RSP.extra_terms(): vals=REC.term_values(p,econW)
    if reset_idle: vals['foot_idle']=idle_reset(z)
    vals['facing']=np.zeros(p['T']) ; 
    out=[]
    for h in hold_ids:
        m=REC.hold_mask(p,h); m=skipmask(m,skip)
        if m.sum()<100: continue
        cmd=z['cmd_applied'][:,:2].astype(float); v=z['root_lin_vel_b'][:,:2].astype(float); d=cmd/np.linalg.norm(cmd,axis=1,keepdims=True).clip(1e-9)
        r={k:float(vv[m].mean()) for k,vv in vals.items()}; r['speed']=float((v*d).sum(1)[m].mean()); tot=sum(vals.values()); r['pre']=float(tot[m].mean()); out.append(r)
    return out
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
meta=json.load(open(S+'/run_meta.json')); ty=meta['travel_yaw']
SH=[]
for f in sorted(glob.glob(S+'/raw/episode_*.npz')):
    z=dict(np.load(f)); c=z['cmd_applied'].astype(np.float64).copy(); cb=c.copy(); cs,sn=np.cos(ty),np.sin(ty)
    cb[:,0]=cs*c[:,0]-sn*c[:,1]; cb[:,1]=sn*c[:,0]+cs*c[:,1]
    n=np.linalg.norm(cb[:,:2],axis=1); k=np.where(n>0.12,0.22/np.maximum(n,1e-9),1.0); cb[:,:2]*=k[:,None]
    z['cmd_applied']=cb.astype(np.float32); z['heading_err']=np.zeros(len(c),np.float32)
    SH+=price(z,meta['hold_labels'],meta['joint_names'],[1],SKIP,True)
W='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29'
roll=REC.load_suite_rollout(W); labels=list(roll['meta']['hold_labels']); jn=list(roll['meta']['joint_names'])
WS=[]
for ep in roll['episodes']:
    if ep['category']!='side': continue
    WS+=price(ep['z'],labels,jn,[ep['hold_idx']] if not isinstance(ep['hold_idx'],list) else ep['hold_idx'],SKIP,False)
a={k:np.mean([r[k] for r in SH]) for k in SH[0]}; b={k:np.mean([r[k] for r in WS]) for k in WS[0]}
print('skip',SKIP,'nSH',len(SH),'nW',len(WS))
for k in a: print(f"{k:16s} SH {a[k]:8.4f}  W {b[k]:8.4f}  diff {a[k]-b[k]:+8.4f}")
