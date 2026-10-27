import pickle, numpy as np, collections, math
D=pickle.load(open('../telemetry/runs.pkl','rb')); rows=D['rows']; comp=D['compact']
print(len(rows), collections.Counter(r.get('alg_class') for r in rows))
ppo=[r for r in rows if (r.get('alg_class')=='PPOWithExtractor' or (r.get('alg_class') is None and 'lr' in comp[r['run_id']] and r['experiment']!='crab_hex_student')) and (r.get('iters_logged') or 0)>0]
print('ppo',len(ppo))
nostd=[r['run'] for r in ppo if 'std' not in comp[r['run_id']]]; print('no std series', nostd)
end199=[r for r in ppo if r['std_last']>=1.99]; print('end>=1.99', len(end199))
tot=sum(len(comp[r['run_id']]['std']) for r in ppo); ge=sum((comp[r['run_id']]['std']>=1.99).sum() for r in ppo)
print('iter frac >=1.99', ge/tot, tot, 'iters_logged sum', sum(r['iters_logged'] for r in ppo))
# dips
for r in ppo:
    s=comp[r['run_id']]['std']
    if s.min()<s[0]-0.01: print('DIP', r['experiment'], r['run'], s[0], s.min(), s[-1], r.get('action_std_max'), r.get('action_std_reset'), r.get('resume'))
# raw drawdown from running max
dd=[(float(np.max(np.maximum.accumulate(comp[r['run_id']]['std'])-comp[r['run_id']]['std'])), r['experiment'], r['run']) for r in ppo]
dd.sort(reverse=True); print('top drawdowns', dd[:8])
# std min/max overall
allmin=min(comp[r['run_id']]['std'].min() for r in ppo); allmax=max(comp[r['run_id']]['std'].max() for r in ppo); print('std range', allmin, allmax)
# non-default caps
for r in ppo:
    if r.get('action_std_max') not in (None,'') or r.get('action_std_reset') not in (None,''):
        s=comp[r['run_id']]['std']; print('KNOB', r['experiment'], r['run'], r.get('action_std_max'), r.get('action_std_reset'), s[0], s.min(), s.max(), s[-1], len(s))
# init std
print('init_noise_std', collections.Counter(str(r.get('init_noise_std')) for r in ppo))
print('std_first dist', collections.Counter(round(float(r['std_first']),2) for r in ppo).most_common(15))
# fresh
fresh=[r for r in ppo if not r.get('resume') and abs((r['std_first'] or 0)-(r.get('init_noise_std') or 0))<0.05]
fi=[r['std_first_iter_ge_cap'] for r in fresh if r['std_first_iter_ge_cap'] is not None]
print('fresh', len(fresh), 'reach cap', len(fi), 'pct', np.percentile(fi,[10,50,90]) if fi else None)
fi199=[]
for r in fresh:
    s=comp[r['run_id']]['std']; w=np.where(s>=1.99)[0]
    if len(w): fi199.append(w[0])
print('fresh reach 1.99 (index)', len(fi199), np.percentile(fi199,[0,10,50,90,100]))
# fresh runs never capping and long
lng=[(r['experiment'],r['run'],r['iters_logged'],float(comp[r['run_id']]['std'][-1])) for r in fresh if r['std_first_iter_ge_cap'] is None and r['iters_logged']>=1500]
print('fresh long never capped', lng)
# entropy
ents=[(float(np.nanmin(comp[r['run_id']]['ent'])) if 'ent' in comp[r['run_id']] else None, float(np.nanmax(comp[r['run_id']]['ent'])) if 'ent' in comp[r['run_id']] else None, r['run']) for r in ppo]
print('ent keys sample', list(comp[ppo[0]['run_id']].keys()))
