import numpy as np, json
W='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29'
m=json.load(open(W+'/run_meta.json')); lab=m['hold_labels']; print(lab)
for cat,labs in (('side',['lat_l','lat_r']),('fwd',['fwd'])):
  ids=[lab.index(l) for l in labs]; res={}
  for e in m['env_index_map']:
    if e['category']!=cat: continue
    z=np.load(W+f"/raw/episode_{e['env_index']:02d}.npz")
    for sid in ids:
      mm=(z['segment_id']==sid)&z['steady_mask']
      if z['done'].any(): mm[np.argmax(z['done']):]=False
      idx=np.where(mm)[0]
      c=z['cmd_applied'][:,:2].astype(float); v=z['root_lin_vel_b'][:,:2].astype(float)
      for w in range(len(idx)//150):
        ii=idx[w*150:(w+1)*150]; cc=c[ii]; sp=np.linalg.norm(cc,axis=1); d=cc/sp[:,None]; al=(v[ii]*d).sum(1); e2=cc-v[ii]; esq=(e2**2).sum(1); cr=v[ii,0]*(-d[:,1])+v[ii,1]*d[:,0]
        tr=1.25*np.exp(-esq/0.04).mean()-2*np.sqrt(esq).mean()+0.6*np.clip(al,0,1.05).mean()-3*(cr**2).mean()
        res.setdefault(w,[]).append((al.mean(),al.std(),tr))
  for w,r in sorted(res.items()):
    a=np.array(r); print(cat,'window',w,len(a),'speed %.3f sd %.3f track %.3f'%tuple(a.mean(0)))
z=np.load(W+'/raw/episode_00.npz'); print([(s,(z['segment_id']==s).sum()) for s in np.unique(z['segment_id'])])
