import numpy as np, glob, json
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
meta=json.load(open(S+'/run_meta.json')); print(meta['hold_labels'], meta.get('travel_yaw'))
files=sorted(glob.glob(S+'/raw/*.npz'))
z=np.load(files[0]); print(z.files)
seg=z['segment_id']; st=z['steady_mask']
for s in np.unique(seg): print(s,(seg==s).sum(),(st&(seg==s)).sum())
ty=np.pi/2
res={0:[],1:[],2:[],3:[]}
allb=[]
for f in files:
    z=np.load(f); m=(z['segment_id']==1)&z['steady_mask']
    if 'done' in z.files and z['done'].any(): m[np.argmax(z['done']):]=False
    idx=np.where(m)[0]
    if len(idx)<150: continue
    c=z['cmd_applied'][:,:2].astype(float); cb=np.stack([np.cos(ty)*c[:,0]-np.sin(ty)*c[:,1], np.sin(ty)*c[:,0]+np.cos(ty)*c[:,1]],1)
    cb=cb/np.linalg.norm(cb,axis=1,keepdims=True).clip(1e-9)*0.22
    v=z['root_lin_vel_b'][:,:2].astype(float)
    for w in range(len(idx)//150):
        ii=idx[w*150:(w+1)*150]; d=cb[ii]/0.22; al=(v[ii]*d).sum(1); e=cb[ii]-v[ii]; esq=(e**2).sum(1); cr=v[ii,0]*(-d[:,1])+v[ii,1]*d[:,0]
        tr=1.25*np.exp(-esq/0.04).mean()-2*np.sqrt(esq).mean()+0.6*np.clip(al,0,1.05).mean()-3*(cr**2).mean()
        res.setdefault(w,[]).append((al.mean(),al.std(),tr))
for w,r in res.items():
    if r: a=np.array(r); print('window',w,len(a),'speed %.3f sd %.3f track %.3f'%tuple(a.mean(0)))
