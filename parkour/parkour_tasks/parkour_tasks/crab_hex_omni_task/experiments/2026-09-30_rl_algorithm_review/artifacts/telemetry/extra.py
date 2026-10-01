import pickle, numpy as np, collections, math
D=pickle.load(open('runs.pkl','rb')); rows=D['rows']; comp=D['compact']
ppo=[r for r in rows if r.get('alg_class')=='PPOWithExtractor' and r['iters_logged']>0]
F=1e-5*1.01
# env steps
steps=sum(r['iters_logged']*r['num_envs']*24 for r in ppo); print('PPO env steps', steps/1e9,'B')
st=[r for r in rows if r['experiment']=='crab_hex_student']
print('student steps', sum(r['iters_logged']*(r['num_envs'] or 0)*120 for r in st)/1e9, 'B (num_steps_per_env 120 assumed from cfg)')
# LR distribution of logged values
allv=np.concatenate([comp[r['run_id']]['lr'] for r in ppo if r['num_envs']==256])
print('logged LR pct (256-env runs) p50/p75/p90/p95/p99', np.percentile(allv,[50,75,90,95,99]))
off=allv[allv>F]; print('off-floor logged LR: n',len(off),'p50',np.median(off),'p90',np.percentile(off,90),'max',off.max())
print('time-mean logged LR over all 256-env its', allv.mean(), 'geo-mean', np.exp(np.log(allv).mean()))
# per run mean LR by experiment
by=collections.defaultdict(list)
for r in ppo:
    if r['num_envs']!=256: continue
    by[r['experiment']].append(comp[r['run_id']]['lr'].mean())
for e,v in by.items(): print(' mean logged LR', e, 'runs',len(v),'p50 %.2e p90 %.2e'%(np.median(v),np.percentile(v,90)))
# sigma ever decreases?
dec=[(r['experiment'],r['run'],r['std_first'],r['std_min'],r['std_last']) for r in ppo if r['std_min']<r['std_first']-0.01]
print('runs where sigma dips >0.01 below its first value:', len(dec), dec[:10])
drops=[]
for r in ppo:
    s=comp[r['run_id']]['std']; 
    if len(s)>1: drops.append(np.max(np.maximum.accumulate(s)-s))
print('max drawdown of sigma within a run p50/p90/max', np.percentile(drops,[50,90,100]))
# fresh runs: pre-pin vs post-pin LR off-floor
pre=[0,0];post=[0,0]
for r in ppo:
    if r['resume'] or r.get('std_first_iter_ge_cap') in (None,0) or r['num_envs']!=256: continue
    k=r['std_first_iter_ge_cap']; lr=comp[r['run_id']]['lr']
    pre[0]+=(lr[:k]>F).sum(); pre[1]+=k; post[0]+=(lr[k:]>F).sum(); post[1]+=len(lr)-k
print('fresh runs that pin: off-floor frac pre-pin %.3f (%d its) post-pin %.3f (%d its)'%(pre[0]/pre[1],pre[1],post[0]/post[1],post[1]))
# fresh 3000-it omni runs: off-floor by 500-it window
w=collections.defaultdict(lambda:[0,0])
for r in ppo:
    if r['experiment']!='crab_hex_omni_flat_walk' or r['resume'] or r['iters_logged']<3000: continue
    lr=comp[r['run_id']]['lr']
    for i in range(0,3000,500): w[i][0]+=(lr[i:i+500]>F).sum(); w[i][1]+=500
print('omni fresh >=3000its off-floor by window', {k:round(a/b,3) for k,(a,b) in sorted(w.items())})
# sym vs no-sym among fresh flat-walk
for s in (0.0,0.5):
    rs=[r for r in ppo if r['experiment']=='crab_hex_flat_walk' and not r['resume'] and r['sym_coef']==s and r['iters_logged']>=1000]
    print('flat fresh sym',s,len(rs),[ (r['run'],r['iters_logged'],round(r['lr_frac_floor'],3),r.get('std_first_iter_ge_cap')) for r in rs][:12])
# resume w/ alg state
ra=[r for r in ppo if r.get('resume_alg_state')]
print('resume_alg_state runs', len(ra), 'resume True among them', sum(bool(r['resume']) for r in ra), 'first dates', min(r['start_time'] for r in ra))
# eplen
print('time per iteration by num_envs', {n:np.median([r['s_per_iter_median'] for r in ppo if r['num_envs']==n]) for n in (64,256)})
print('fps by num_envs', {n:np.median([r['fps_median'] for r in ppo if r['num_envs']==n]) for n in (64,256)})
# by month GPU-h
m=collections.defaultdict(float)
for r in rows: m[r['start_time'][:7]]+= (r.get('train_gpu_h') or 0)
print('gpu-h by month', dict(m))
# run length histogram
print('run length buckets', collections.Counter(('<=30' if r['iters_logged']<=30 else '31-500' if r['iters_logged']<=500 else '501-2000' if r['iters_logged']<=2000 else '2001-5000' if r['iters_logged']<=5000 else '>5000') for r in ppo))
print('iters in runs <=30:', sum(r['iters_logged'] for r in ppo if r['iters_logged']<=30))
