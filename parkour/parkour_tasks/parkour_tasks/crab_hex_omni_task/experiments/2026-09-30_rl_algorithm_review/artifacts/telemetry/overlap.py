import pickle, glob, numpy as np, collections, os
D=pickle.load(open('runs.pkl','rb')); rows={r['run_id']:r for r in D['rows']}
iv={}
for p in glob.glob('series/*.pkl'):
    d=pickle.load(open(p,'rb')); rid=os.path.basename(p)[:-4]
    s=d['series'].get('Perf/total_fps')
    if not s: continue
    a=np.array(s)  # step, wall, val
    iv[rid]=a
# per-iteration concurrency: for each run, each logged iteration's wall time, count other runs active
spans={k:(v[0,1],v[-1,1]) for k,v in iv.items()}
res=collections.defaultdict(list)
alone_fps=collections.defaultdict(list); conc_fps=collections.defaultdict(list)
for k,a in iv.items():
    r=rows[k]
    others=[s for kk,s in spans.items() if kk!=k and s[1]>a[0,1] and s[0]<a[-1,1]]
    w=a[:,1]
    busy=np.zeros(len(w),bool)
    for s in others: busy|=(w>=s[0])&(w<=s[1])
    r['conc_frac']=busy.mean()
    key=(r['experiment'].replace('crab_hex_',''), r['num_envs'])
    alone_fps[key]+=list(a[~busy,2]); conc_fps[key]+=list(a[busy,2])
for key in sorted(alone_fps, key=str):
    al=alone_fps[key]; co=conc_fps[key]
    print(key, 'alone its',len(al),'fps p50 %.0f'%(np.median(al) if al else float('nan')), '| concurrent its',len(co),'fps p50 %.0f'%(np.median(co) if co else float('nan')))
# 64 vs 256 collection time alone
ct=collections.defaultdict(list)
for p in glob.glob('series/*.pkl'):
    d=pickle.load(open(p,'rb')); rid=os.path.basename(p)[:-4]; r=rows[rid]
    c=d['series'].get('Perf/collection time'); l=d['series'].get('Perf/learning_time')
    if not c: continue
    ct[r['num_envs']].append((np.median([x[2] for x in c]), np.median([x[2] for x in l]), r['experiment'], r['run'], r['iters_logged'], round(r['conc_frac'],2)))
for n in ct:
    v=ct[n]; print(n,'runs',len(v),'collect p50 %.3f learn p50 %.3f'%(np.median([x[0] for x in v]),np.median([x[1] for x in v])))
for x in sorted(ct[64], key=lambda x:x[3])[:40]: print('  64env',x)
pickle.dump({k:rows[k].get('conc_frac') for k in rows}, open('conc.pkl','wb'))
print('runs with >50% concurrent its', sum(1 for r in rows.values() if (r.get('conc_frac') or 0)>0.5))
