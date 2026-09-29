import numpy as np, glob, json
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
W='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29'
T3='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation/step0_E3_X08-t3_visible/suite__omni/seed001/2026-09-23_21-38-11'
def ep_arrays(f, segs, ty=0.0, cmd_mag=None):
    z=np.load(f); m=np.isin(z['segment_id'],segs)&z['steady_mask']
    if 'done' in z.files and z['done'].any(): m[np.argmax(z['done']):]=False
    if m.sum()<50: return None
    c=z['cmd_applied'][:,:2].astype(float); cs,sn=np.cos(ty),np.sin(ty)
    cb=np.stack([cs*c[:,0]-sn*c[:,1], sn*c[:,0]+cs*c[:,1]],1)
    if cmd_mag: cb=cb/np.maximum(np.linalg.norm(cb,axis=1),1e-9)[:,None]*cmd_mag
    v=z['root_lin_vel_b'][:,:2].astype(float)
    return cb[m],v[m]
def summarize(name, eps):
    rows=[]
    for cb,v in eps:
        sp=np.linalg.norm(cb,axis=1); d=cb/sp[:,None]
        al=(v*d).sum(1); cr=v[:,0]*(-d[:,1])+v[:,1]*d[:,0]
        e=cb-v; esq=(e**2).sum(1); el1=np.sqrt(esq)
        r=dict(cmd=sp.mean(),speed=al.mean(),sd_along=al.std(),cross_rms=np.sqrt((cr**2).mean()),
          linW=1.25*np.exp(-esq/0.04).mean(), l1W=-2*el1.mean(), linS=1.25*np.exp(-esq/0.1).mean(), l1S=-1*el1.mean(),
          prog=0.6*np.clip(al,0,1.05).mean(), cross=-3*(cr**2).mean())
        # same gait but with velocity fluctuation removed (mean velocity only): kernel ceiling of smoothness
        vm=v.mean(0); e0=cb-vm; r['linW_smooth']=1.25*np.exp(-(e0**2).sum(1)/0.04).mean()
        rows.append(r)
    out={k:np.mean([r[k] for r in rows]) for k in rows[0]}
    out['trackW']=out['linW']+out['l1W']+out['prog']+out['cross']; out['trackS']=out['linS']+out['l1S']+out['prog']+out['cross']
    print(f"{name:22s} n={len(rows):3d} "+' '.join(f"{k}={v:.3f}" for k,v in out.items()))
    return out
def omni(stamp,cat,labs):
    m=json.load(open(stamp+'/run_meta.json')); lab=m['hold_labels']; ids=[lab.index(l) for l in labs]
    eps=[ep_arrays(stamp+f"/raw/episode_{e['env_index']:02d}.npz",ids) for e in m['env_index_map'] if e['category']==cat]
    return [e for e in eps if e is not None]
summarize('W fwd',omni(W,'fwd',['fwd']))
summarize('W side',omni(W,'side',['lat_l','lat_r']))
summarize('W back',omni(W,'back',['back']))
summarize('t3 fwd',omni(T3,'fwd',['fwd']))
summarize('t3 side',omni(T3,'side',['lat_l','lat_r']))
files=sorted(glob.glob(S+'/raw/*.npz'))
summarize('sidehead creep@0.25',[e for e in (ep_arrays(f,[1],np.pi/2) for f in files) if e is not None])
summarize('sidehead creep@0.22cf',[e for e in (ep_arrays(f,[1],np.pi/2,0.22) for f in files) if e is not None])
summarize('sidehead low@0.35',[e for e in (ep_arrays(f,[2],np.pi/2) for f in files) if e is not None])
print('--- window analysis (3 s windows) ---')
def windows(eps, L=150):
    out=[]
    for cb,v in eps:
        for s in range(0,len(v)-L+1,L):
            c=cb[s:s+L]; vv=v[s:s+L]; sp=np.linalg.norm(c,axis=1); d=c/sp[:,None]
            al=(vv*d).sum(1); cr=vv[:,0]*(-d[:,1])+vv[:,1]*d[:,0]; e=c-vv; esq=(e**2).sum(1)
            out.append((al.mean(), al.std(), np.sqrt((cr**2).mean()), 1.25*np.exp(-esq/0.04).mean()-2*np.sqrt(esq).mean()+0.6*np.clip(al,0,1.05).mean()-3*(cr**2).mean()))
    return np.array(out)
for name,eps in (('W side',omni(W,'side',['lat_l','lat_r'])),('W fwd',omni(W,'fwd',['fwd'])),('sidehead',[e for e in (ep_arrays(f,[1],np.pi/2,0.22) for f in files) if e is not None])):
    a=windows(eps); sp,sd,cr,tr=a.T
    X=np.c_[np.ones_like(sp),sp,sd,cr]; b=np.linalg.lstsq(X,tr,rcond=None)[0]
    print(f"{name:10s} nwin={len(a)} speed mean={sp.mean():.3f} sd={sp.std():.3f} | corr(speed,sd_along)={np.corrcoef(sp,sd)[0,1]:+.2f} corr(speed,cross)={np.corrcoef(sp,cr)[0,1]:+.2f} | track ~ {b[1]:+.2f}*speed {b[2]:+.2f}*sd {b[3]:+.2f}*cross | raw slope track/speed={np.polyfit(sp,tr,1)[0]:+.2f}")
    # binned
    q=np.quantile(sp,[0,.25,.5,.75,1.0])
    for i in range(4):
        mm=(sp>=q[i])&(sp<=q[i+1]); print(f"   speed[{q[i]:.3f},{q[i+1]:.3f}] sd_along={sd[mm].mean():.3f} cross={cr[mm].mean():.3f} track={tr[mm].mean():.3f}")
print('--- filtered-velocity tracking (moving average) ---')
def filt(eps, L):
    rows=[]
    k=np.ones(L)/L
    for cb,v in eps:
        if len(v)<L+5: continue
        vf=np.stack([np.convolve(v[:,i],k,mode='valid') for i in range(2)],1); c=cb[L-1:]
        sp=np.linalg.norm(c,axis=1); d=c/sp[:,None]; al=(vf*d).sum(1); cr=vf[:,0]*(-d[:,1])+vf[:,1]*d[:,0]
        e=c-vf; esq=(e**2).sum(1)
        rows.append([1.25*np.exp(-esq/0.04).mean(), -2*np.sqrt(esq).mean(), 0.6*np.clip(al,0,1.05).mean(), -3*(cr**2).mean()])
    r=np.mean(rows,0); return r, r.sum()
for L in (1,25,50,100):
    line=[]
    for name,eps in (('W fwd',omni(W,'fwd',['fwd'])),('W side',omni(W,'side',['lat_l','lat_r'])),('t3 side',omni(T3,'side',['lat_l','lat_r'])),('sidehead@.22',[e for e in (ep_arrays(f,[1],np.pi/2,0.22) for f in files) if e is not None])):
        r,s=filt(eps,L); line.append(f"{name}: lin={r[0]:.3f} L1={r[1]:.3f} tot={s:.3f}")
    print(f"window {L*0.02:.1f}s | "+' | '.join(line))
