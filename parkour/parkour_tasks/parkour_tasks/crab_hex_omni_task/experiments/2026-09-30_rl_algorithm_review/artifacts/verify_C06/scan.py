import pickle, glob, csv, numpy as np, os
rows={r['run_id']:r for r in csv.DictReader(open('telemetry/runs.csv')) if r['alg_class']=='PPOWithExtractor'}
dds=[]; emin=(1e9,None); emax=(-1e9,None); ent_out=[]
for rid,r in rows.items():
    d=pickle.load(open(f'telemetry/series/{rid}.pkl','rb'))['series']
    s=d.get('Policy/mean_noise_std'); 
    if not s: continue
    s=sorted(s); v=np.array([x[2] for x in s])
    dd=np.max(np.maximum.accumulate(v)-v); dds.append((dd,r['experiment'],r['run'],v[0],v.max(),v[-1],r['resume']))
    e=d.get('Loss/entropy')
    if e:
        ev=np.array([x[2] for x in e])
        if ev.min()<emin[0]: emin=(ev.min(),r['run'])
        if ev.max()>emax[0]: emax=(ev.max(),r['run'])
        if ev.min()<32.8 or ev.max()>38.05: ent_out.append((r['experiment'],r['run'],round(ev.min(),2),round(ev.max(),2)))
dds.sort(reverse=True)
print('runs',len(dds))
for x in dds[:12]: print(x)
print('drawdown>0.01:',sum(x[0]>0.01 for x in dds),' >0.005:',sum(x[0]>0.005 for x in dds))
print('entropy min',emin,'max',emax)
print('runs with entropy outside [32.8,38.05]:',len(ent_out)); 
for x in ent_out: print(x)
