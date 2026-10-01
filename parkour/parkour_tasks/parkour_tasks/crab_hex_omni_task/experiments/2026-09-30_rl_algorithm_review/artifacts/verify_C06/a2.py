import pickle, numpy as np, collections, math
D=pickle.load(open('../telemetry/runs.pkl','rb')); rows=D['rows']; comp=D['compact']
ppo=[r for r in rows if r.get('alg_class')=='PPOWithExtractor' and (r.get('iters_logged') or 0)>0]
fresh=[r for r in ppo if not r.get('resume') and abs((r['std_first'] or 0)-(r.get('init_noise_std') or 0))<0.05]
for th in [1000,2000,2500,3000,3500,4000,5000]:
    rs=[r for r in fresh if r['iters_logged']>=th]
    c=sum(r['std_first_iter_ge_cap'] is not None for r in rs)
    c2=sum(r['std_first_iter_ge_cap'] is not None and r['std_first_iter_ge_cap']<th for r in rs)
    print('fresh iters>=',th,'n',len(rs),'ever cap',c,'cap before',th, c2)
print('fresh iters dist', np.percentile([r['iters_logged'] for r in fresh],[10,25,50,75,90]))
print('fresh by exp', collections.Counter(r['experiment'] for r in fresh))
print('resume flag counts', collections.Counter(str(r.get('resume')) for r in ppo))
# non-fresh non-resume?
odd=[(r['experiment'],r['run'],r['std_first'],r.get('init_noise_std')) for r in ppo if not r.get('resume') and r not in fresh]
print('not resume but not fresh', len(odd), odd[:10])
# resumed runs starting below cap -> do they reach cap
res=[r for r in ppo if r.get('resume') and r['std_first']<1.99]
print('resumed below 1.99 at start', len(res), 'end >=1.99', sum(r['std_last']>=1.99 for r in res))
# ending not >=1.99 breakdown
notend=[r for r in ppo if r['std_last']<1.99]
print('not ending >=1.99', len(notend), 'fresh', sum(r in fresh for r in notend), 'iters pct', np.percentile([r['iters_logged'] for r in notend],[10,50,90]))
# per-experiment frac and per-run median
by=collections.defaultdict(list)
for r in ppo: by[r['experiment']].append(r)
for e,rs in by.items():
    s=np.concatenate([comp[r['run_id']]['std'] for r in rs])
    print(e, len(rs), 'iterfrac>=1.99 %.3f'%(s>=1.99).mean(), 'end>=1.99', sum(r['std_last']>=1.99 for r in rs))
# entropy range
emin=[];emax=[]
for r in ppo:
    e=comp[r['run_id']].get('ent')
    if e is None: print('no ent',r['run']); continue
    e=e[np.isfinite(e)]; emin.append((float(e.min()),r['run'])); emax.append((float(e.max()),r['run']))
print('ent min lowest', sorted(emin)[:5]); print('ent max highest', sorted(emax)[-5:])
# check ent vs std relation
C=0.5*math.log(2*math.pi*math.e)
res=[]
for r in ppo:
    c=comp[r['run_id']]; n=min(len(c['std']),len(c['ent']))
    s=np.clip(c['std'][:n],0.05,r.get('action_std_max') and float(r['action_std_max']) or 2.0); pred=18*(C+np.log(s))
    res.append(np.nanmax(np.abs(pred-c['ent'][:n])))
print('max |ent - 18*(C+ln clamp(std))| p50/p99/max', np.percentile(res,[50,99,100]))
print('18*H(1.5), 18*H(2.0)', 18*(C+math.log(1.5)), 18*(C+math.log(2.0)))
for run in ['2026-08-20_16-23-35','2026-08-20_19-01-38','2026-08-20_22-40-59']:
    r=[x for x in ppo if x['run']==run][0]; s=comp[r['run_id']]['std']
    print(run, r['iters_logged'], r.get('resume'), 'argmax', int(np.argmax(s)), float(s.max()), 'last', float(s[-1]), 'min after max', float(s[np.argmax(s):].min()))
