import sys, json, glob, ast
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/scripts')
from side22_classify import hold_stats
ARMS='/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect/arms/'
CAN='/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect/canon/'
stamps={}
for n in ['O','R','C_band','C_rsi','C','C_s2','C_mo','C_mo_s2','C_mo_s1']:
    d=json.load(open(ARMS+n+'.json'))[n]; stamps[n]=d['stamp']
for n in ['LRfa','LR4','LR4_dwell','C_lr','C_lr_10','C_lr_rsi','C_lr_10_rsi']:
    d=json.load(open(ARMS+n+'_lr.json'))[n]; stamps[n+'|lr']=d['stamp']
for h in ['C','C_s2','C_mo_s2','C_mo_s1']:
    stamps['canon_'+h]=glob.glob(CAN+h+'/wave/side22_25lr__omni/seed001/*')[0]
rng=np.random.default_rng(0)
def quat_grav(q):
    w,x,y,z=[q[:,i] for i in range(4)]
    # projected gravity in body frame: R^T * (0,0,-1)
    gx=-2*(x*z - w*y); gy=-2*(y*z + w*x); return gx,gy
def yaw(q):
    w,x,y,z=[q[:,i] for i in range(4)]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
out={}
for name,st in stamps.items():
    st=st if st.startswith('/') else '/home/nickmagus/krabby/krabby-research/'+st
    d=Path(st); meta=json.loads((d/'run_meta.json').read_text())
    labels=meta['hold_labels']; dirs=meta['hold_dirs']; speeds=meta['hold_speeds']
    jn=ast.literal_eval(meta['joint_names']) if isinstance(meta['joint_names'],str) else meta['joint_names']
    legs=['FL','FR','ML','MR','RL','RR']
    tib=[jn.index(f'{l}_Femur_Tibia_RevoluteJoint') for l in legs]; q0=np.array([0.3102061292,-0.3102061292]*3)
    emap=ast.literal_eval(meta['env_index_map']) if isinstance(meta['env_index_map'],str) else meta['env_index_map']
    res={}
    for e in emap:
        lab=e['label']
        if not lab.startswith('lat_'): continue
        z=np.load(d/'raw'/f"episode_{e['env_index']:02d}.npz")
        done=z['done'].astype(bool); S=int(np.argmax(done)) if done.any() else len(z['segment_id'])
        sid=labels.index(lab); hs=hold_stats(z,sid,S)
        if hs is None: continue
        seg=z['segment_id']; idx=np.where((seg==sid)&(np.arange(len(seg))<S))[0]; t0=idx[0]; tend=idx[-1]
        a4=t0+200; hi=min(t0+500,tend); W=slice(a4,hi)
        q=z['root_quat_w'].astype(float); psi=yaw(q)
        vb=z['root_lin_vel_b'].astype(float)
        vw=np.stack([np.cos(psi)*vb[:,0]-np.sin(psi)*vb[:,1], np.sin(psi)*vb[:,0]+np.cos(psi)*vb[:,1]],1)
        dr=dirs[sid]; sp=speeds[sid]; u=np.array([np.cos(dr),np.sin(dr)]); cw=sp*u
        err=vw-cw; e2=(err**2).sum(1)
        r={}
        r['track_exp']=1.25*np.mean(np.exp(-e2[W]/0.04))
        r['l1']=-2.0*np.mean(np.sqrt(e2[W]))
        r['prog']=0.6*np.mean(np.clip(vw[W]@u,0,2))
        nrm=np.array([-u[1],u[0]]); r['cross']=-3.0*np.mean((vw[W]@nrm)**2)
        wz=z['root_ang_vel_b'][:,2].astype(float); wzc=z['cmd_applied'][:,2].astype(float)
        r['angz']=1.0*np.mean(np.exp(-(wzc[W]-wz[W])**2/0.25))
        F=z['foot_force_norm'].astype(float)>0.1
        idle=np.zeros(6); ex=[]; streak=np.zeros(6); mx=np.zeros(6)
        for t in range(hi):
            idle=np.where(F[t],0,idle+1)
            if t>=a4: ex.append(np.maximum(idle-90,0).sum()); mx=np.maximum(mx,idle)
        r['foot_idle']=-0.12*np.mean(ex); r['max_air_s']=float(mx.max()*0.02)
        r['excess']=-0.2*np.mean(np.maximum(F[W].sum(1)-4,0))
        r['n_contact']=float(F[W].sum(1).mean())
        a=z['actions'].astype(float); ac=np.clip(a,-1,1)
        r['ar_det']=-0.3*np.mean(np.linalg.norm(ac[a4:hi]-ac[a4-1:hi-1],axis=1))
        acc=[]
        for k in range(8):
            n1=np.clip(a[a4-1:hi]+2.0*rng.standard_normal(a[a4-1:hi].shape),-1,1)
            acc.append(np.mean(np.linalg.norm(n1[1:]-n1[:-1],axis=1)))
        r['ar_sto2']=-0.3*np.mean(acc)
        r['sat']=float(np.mean(np.abs(a[W])>1)); r['sat3']=float(np.mean(np.abs(a[W])>3))
        # expected executed action under noise (mean of clip(mu+2e))
        tau=z['applied_torque'].astype(float); r['dtorq']=-1e-6*np.mean(((tau[a4:hi]-tau[a4-1:hi-1])**2).sum(1))
        jp=z['joint_pos'].astype(float)[:,tib]; ff=F[:,:]
        r['tibia']=-0.28*np.mean((((jp-q0)**2)*ff)[W].sum(1))
        gx,gy=quat_grav(q); r['orient']=-0.7*np.mean((gx**2+gy**2)[W])
        r['linz']=-0.15*np.mean(vb[W,2]**2)
        r['feet_ss']=hs['feet_ss']; r['sweep']=hs['sweep']; r['lean']=hs['lean_abs']
        hand='L' if 'lat_l' in lab else 'R'
        res.setdefault(hand,[]).append(r)
    out[name]={h:{k:float(np.nanmedian([r[k] for r in rs])) for k in rs[0]}|{'n':len(rs)} for h,rs in res.items()}
    print(name, {h:v['n'] for h,v in out[name].items()}, flush=True)
json.dump(out,open('replay.json','w'),indent=1)
