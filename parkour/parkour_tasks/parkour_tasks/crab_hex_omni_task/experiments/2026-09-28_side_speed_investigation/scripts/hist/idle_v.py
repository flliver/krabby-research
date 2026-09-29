import sys,json,os,glob,numpy as np
for x in sys.argv[1:]:
    n,s,l=x.split('|'); keep=l.split(',')
    meta=json.load(open(os.path.join(s,'run_meta.json'))); labs=meta['hold_labels']; res={}
    for f in sorted(glob.glob(os.path.join(s,'raw','episode_*.npz')))[:300]:
        a=np.load(f); seg=a['segment_id']; st=a['steady_mask'].astype(bool); c=a['foot_force_norm']>0.1
        idle=np.zeros(6); ex=np.zeros(len(c))
        for t in range(len(c)):
            idle=np.where(c[t],0,idle+1); ex[t]=np.maximum(idle-90,0).sum()
        vy=a['root_lin_vel_b'][:,:2]
        for sid,lab in enumerate(labs):
            if lab not in keep: continue
            m=(seg==sid)&st
            if m.sum()<50: continue
            sp=np.linalg.norm(vy[m].mean(0))
            vstd=np.linalg.norm(vy[m]-vy[m].mean(0),axis=1).mean()
            res.setdefault(lab,[]).append((-0.12*ex[m].mean(), sp, vstd, (c[m].all(0)).sum()))
    for lab,v in res.items():
        v=np.array(v); print(n,lab,len(v),'idle/s=%.3f mean|v|=%.3f dev=%.3f alwaysdown_feet=%.2f'%tuple(np.median(v,0)), 'frac eps idle<0: %.2f'%(v[:,0]<0).mean())
