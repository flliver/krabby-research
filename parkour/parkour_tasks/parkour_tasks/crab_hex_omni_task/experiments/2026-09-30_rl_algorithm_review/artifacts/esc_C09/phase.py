import sys, numpy as np, glob, datetime
from tb import load
# (A) two-process test: per-iteration timing vs wall clock for L1/L3/L5 bundles
E=sys.argv[1]
runs={}
for d in sorted(glob.glob(E+'/L*/')):
    evs=glob.glob(d+'**/events.out.tfevents.*', recursive=True)
    if not evs: continue
    s=load(evs[0].rsplit('/',1)[0]); runs[d.split('/')[-2]]=s
    c=s['Perf/collection time']; print(d.split('/')[-2], evs[0].rsplit('/',2)[-2], 'its',len(c),'wall',datetime.datetime.fromtimestamp(c[0,1]).strftime('%H:%M'),'-',datetime.datetime.fromtimestamp(c[-1,1]).strftime('%H:%M'))
spans={k:(v['Perf/collection time'][0,1],v['Perf/collection time'][-1,1]) for k,v in runs.items()}
for k,s in runs.items():
    c=s['Perf/collection time']; l=s['Perf/learning_time']; f=s['Perf/total_fps']; w=c[:,1]
    busy=np.zeros(len(w),bool)
    for kk,sp in spans.items():
        if kk!=k: busy|=(w>=sp[0])&(w<=sp[1])
    for lab,m in (('solo',~busy),('overlap',busy)):
        if m.sum()>5: print('  %s %-8s its %5d collect %.3f learn %.3f fps %.0f'%(k[:3],lab,m.sum(),np.median(c[m,2]),np.median(l[m,2]),np.median(f[m,2])))
# (B) iteration-phase confound inside long 256-env runs
for d in sys.argv[2:]:
    s=load(d); c=s['Perf/collection time']; it=c[:,0]-c[0,0]; el=s.get('Train/mean_episode_length')
    def med(m): return np.median(c[m,2]) if m.sum() else float('nan')
    print(d.split('/')[-1][:40],'its',len(c),'coll its1-19 %.3f | its20-499 %.3f | last500 %.3f | whole %.3f'%(med((it>=1)&(it<20)),med((it>=20)&(it<500)),med(it>=len(c)-500),np.median(c[:,2])), 'eplen its1-19 %.0f last500 %.0f'%(np.median(el[1:20,2]),np.median(el[-500:,2])) if el is not None else '')
