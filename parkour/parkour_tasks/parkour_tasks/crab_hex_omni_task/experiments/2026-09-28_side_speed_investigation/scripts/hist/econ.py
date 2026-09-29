# CPU-only partial reward-economy replay on raw eval npz: prices the translation-tracking and
# common penalty terms of W's omni economy (weights from W params/env.yaml) on side holds of
# omni stamps and on the sideways head's walking holds (rotated into the travel frame).
import sys, json, os, glob, numpy as np
W = dict(track=1.25, sig2=0.04, l1=-2.0, prog=0.6, cross=-3.0, arate=-0.3, dtau=-1e-6, vz=-0.15, orient=-0.7,
         airtime=0.8, air_thr=None, excess=-0.2, idle=-0.12)
def gravity_xy(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    # projected gravity in body frame = R^T (0,0,-1)
    gx = -2*(x*z - w*y); gy = -2*(y*z + w*x)
    return gx, gy
def load(stamp, labels_keep):
    meta=json.load(open(os.path.join(stamp,'run_meta.json')))
    labs=meta['hold_labels']; ty=float(meta.get('travel_yaw',0.0)); side_task = abs(ty-np.pi/2)<1e-3
    out={}
    for f in sorted(glob.glob(os.path.join(stamp,'raw','episode_*.npz'))):
        a=np.load(f)
        seg=a['segment_id']; st=a['steady_mask'].astype(bool)
        vb=a['root_lin_vel_b'][:,:2].astype(np.float64); cmd=a['cmd_applied'][:,:2].astype(np.float64)
        if side_task:
            v=np.stack([vb[:,1],-vb[:,0]],1)  # travel frame
            cmd=np.stack([cmd[:,0],np.zeros(len(cmd))],1)
        else:
            v=vb
        act=a['actions'].astype(np.float64); tau=a['applied_torque'].astype(np.float64)
        dact=np.zeros(len(act)); dact[1:]=np.linalg.norm(np.diff(act,axis=0),axis=1)
        dtau=np.zeros(len(act)); dtau[1:]=np.sum(np.diff(tau,axis=0)**2,axis=1)
        vz=a['root_lin_vel_b'][:,2]; gx,gy=gravity_xy(a['root_quat_w'].astype(np.float64))
        F=a['foot_force_norm']; ncont=(F>0.1).sum(1)
        for sid,lab in enumerate(labs):
            if lab not in labels_keep: continue
            m=(seg==sid)&st
            if m.sum()<50: continue
            c=cmd[m]; vv=v[m]; sp=np.linalg.norm(c,axis=1); d=c/np.maximum(sp,1e-8)[:,None]
            err=np.linalg.norm(c-vv,axis=1)
            along=(vv*d).sum(1); cr=vv[:,0]*(-d[:,1])+vv[:,1]*d[:,0]
            r=dict(speed=along.mean(), cmd=sp.mean(),
                   track=W['track']*np.exp(-err**2/W['sig2']).mean(), l1=W['l1']*err.mean(),
                   prog=W['prog']*np.clip(along,0,2).mean(), cross=W['cross']*(cr**2).mean(),
                   arate=W['arate']*dact[m].mean(), dtau=W['dtau']*dtau[m].mean(),
                   vz=W['vz']*(vz[m]**2).mean(), orient=W['orient']*(gx[m]**2+gy[m]**2).mean(),
                   excess=W['excess']*np.maximum(ncont[m]-3,0).mean())
            out.setdefault(lab,[]).append(r)
    return out
def summ(name,out):
    for lab,rs in out.items():
        keys=list(rs[0].keys()); med={k:float(np.median([r[k] for r in rs])) for k in keys}
        tot=sum(med[k] for k in keys if k not in('speed','cmd'))
        print(f"{name:10s} {lab:6s} n={len(rs):3d} "+' '.join(f"{k}={med[k]:+.3f}" for k in keys)+f"  SUM={tot:+.3f}")
for name,stamp,labs in [(n,s,l.split(',')) for n,s,l in (x.split('|') for x in sys.argv[1:])]:
    summ(name, load(stamp, labs))
