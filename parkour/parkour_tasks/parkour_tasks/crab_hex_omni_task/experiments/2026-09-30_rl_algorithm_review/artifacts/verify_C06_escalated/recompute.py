import pickle, csv, numpy as np, os, math
SP=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows=[r for r in csv.DictReader(open(SP+'/telemetry/runs.csv'))]
print('total rows',len(rows),'PPO',sum(r['alg_class']=='PPOWithExtractor' for r in rows),'other',{r['alg_class'] for r in rows if r['alg_class']!='PPOWithExtractor'})
ppo=[r for r in rows if r['alg_class']=='PPOWithExtractor']
C=0.5*math.log(2*math.pi*math.e)
stats=[]; tot_it=0; tot_ge=0
for r in ppo:
    d=pickle.load(open(f"{SP}/telemetry/series/{r['run_id']}.pkl",'rb'))['series']
    s=d.get('Policy/mean_noise_std')
    if not s: print('NO STD',r['run']); continue
    s=sorted(s); st=np.array([x[0] for x in s]); v=np.array([x[2] for x in s])
    e=d.get('Loss/entropy'); ev=np.array([x[2] for x in sorted(e)]) if e else None
    cap=float(r['action_std_max']) if r['action_std_max'] not in ('',None) else 2.0
    idx=np.where(v>=1.99)[0]
    first199=int(st[idx[0]]-st[0]) if len(idx) else None
    after_min=float(v[idx[0]:].min()) if len(idx) else None
    dd=float(np.max(np.maximum.accumulate(v)-v))
    # drawdown before pin
    pre=v[:idx[0]] if len(idx) else v
    dd_pre=float(np.max(np.maximum.accumulate(pre)-pre)) if len(pre) else 0.0
    # entropy identity residual
    res=None
    if ev is not None and len(ev)==len(v):
        res=float(np.max(np.abs(ev-18*(C+np.log(np.clip(v,0.05,cap))))))
    stats.append(dict(run=r['run'],exp=r['experiment'],resume=r['resume'],cap=cap,n=len(v),first=v[0],last=v[-1],vmin=v.min(),vmax=v.max(),
        frac199=(v>=1.99).mean(),first199=first199,after_min=after_min,dd=dd,dd_pre=dd_pre,below_start=float(v[0]-v.min()),
        emin=None if ev is None else ev.min(),emax=None if ev is None else ev.max(),res=res,iters=int(r['iters_logged'])))
    tot_it+=len(v); tot_ge+=int((v>=1.99).sum())
n=len(stats)
print('runs with std',n)
print('end>=1.99:',sum(s['last']>=1.99 for s in stats))
print('iteration-weighted frac>=1.99:',tot_ge/tot_it,'over',tot_it)
print('start==2.0 (>=1.99):',sum(s['first']>=1.99 for s in stats),'start ~1.5:',sum(abs(s['first']-1.5)<0.01 for s in stats),'other starts:',[(s['run'],round(s['first'],3)) for s in stats if s['first']<1.99 and abs(s['first']-1.5)>=0.01])
fresh=[s for s in stats if s['resume']=='False']
print('fresh',len(fresh),'resume',n-len(fresh),'resume values',set(s['resume'] for s in stats))
fr=[s for s in fresh if s['first199'] is not None]
print('fresh reaching 1.99:',len(fr),'percentiles p10/50/90:',np.percentile([s['first199'] for s in fr],[10,50,90]))
fr2=[s for s in fresh if s['first199'] is not None and s['cap']==2.0]
print('fresh cap=2.0 reaching 1.99:',len(fr2),np.percentile([s['first199'] for s in fr2],[10,50,90]))
print('fresh cap!=2:',[(s['run'],s['cap'],s['n'],round(s['first'],3),round(s['last'],3),s['first199'],s['emin'],s['emax']) for s in fresh if s['cap']!=2.0])
print('all cap!=2:',[(s['run'],s['cap'],s['resume']) for s in stats if s['cap']!=2.0])
for L in (2000,2500,3000,3500,5000):
    f=[s for s in fresh if s['n']>=L]; print(f'fresh n>={L}:',len(f),'pinned',sum(s['first199'] is not None for s in f),'unpinned:',[(s['run'][:45],s['n'],round(s['last'],3)) for s in f if s['first199'] is None][:8])
print('below start >0.01:',[(s['run'],round(s['first'],4),round(s['vmin'],4)) for s in stats if s['below_start']>0.01])
print('drawdown from peak >0.01:',sorted([(round(s['dd'],4),s['exp'],s['run']) for s in stats if s['dd']>0.01],reverse=True))
print('drawdown before pin >0.01:',sorted([(round(s['dd_pre'],4),s['run']) for s in stats if s['dd_pre']>0.01],reverse=True))
pinned=[s for s in stats if s['first199'] is not None]
print('pinned runs',len(pinned),'min after first 1.99:',min(s['after_min'] for s in pinned),[s['run'] for s in pinned if s['after_min']==min(x['after_min'] for x in pinned)])
print('any run leaves cap by >0.01 after pin:',[(s['run'],s['after_min']) for s in pinned if s['after_min']<1.98])
print('global std max',max(s['vmax'] for s in stats))
ents=[s for s in stats if s['emin'] is not None]
print('entropy: runs',len(ents),'min',min(s['emin'] for s in ents),'max',max(s['emax'] for s in ents))
print('entropy outside [32.8,38.05]:',[(s['run'],round(s['emin'],3),round(s['emax'],3)) for s in ents if s['emin']<32.8 or s['emax']>38.05])
print('entropy identity residual max',max(s['res'] for s in stats if s['res'] is not None),'median',np.median([s['res'] for s in stats if s['res'] is not None]),'n',sum(s['res'] is not None for s in stats))
print('18*ent@1.5=',18*(C+math.log(1.5)),'@2.0=',18*(C+math.log(2.0)))
# per experiment
from collections import defaultdict
pe=defaultdict(lambda:[0,0,0,0])
for s in stats:
    p=pe[s['exp']]; p[0]+=1; p[1]+=int(s['last']>=1.99); p[2]+=int(s['n']*s['frac199']); p[3]+=s['n']
for k,p in sorted(pe.items()): print(f'{k:40s} runs {p[0]:4d} end>=1.99 {p[1]:4d} iterfrac {p[2]/p[3]:.3f}')
