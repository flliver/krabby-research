import pickle,numpy as np,collections
D=pickle.load(open('runs.pkl','rb')); rows=D['rows']; comp=D['compact']
ppo=[r for r in rows if r.get('alg_class')=='PPOWithExtractor' and r['iters_logged']>0 and r['num_envs']==256]
def sm(v,w=50):
    c=np.cumsum(np.insert(v,0,0)); out=np.array([(c[i+1]-c[max(0,i-w+1)])/(i+1-max(0,i-w+1)) for i in range(len(v))]); return out
print('== fresh runs >=2000 its: reward milestones')
fam=collections.defaultdict(list)
for r in ppo:
    if r['resume'] or r['iters_logged']<2000: continue
    v=comp[r['run_id']]['rew']; s=sm(v); r0=s[:50].mean(); rf=s[-50:].mean(); g=rf-r0
    if g<=0: continue
    its={q:int(np.argmax((s-r0)/g>=q)) for q in (0.5,0.9)}
    n=len(s); last_third=(rf - s[int(2*n/3)])/g
    key=(r['experiment'].replace('crab_hex_',''), r['iters_logged'] if r['iters_logged'] in (2000,3000,5000,8000,20000) else 'other')
    fam[key].append((its[0.5],its[0.9],last_third,r['iters_logged']))
for k,v in sorted(fam.items(), key=str):
    a=np.array(v)
    print(k,'n',len(v),'its50 p50 %d its90 p50 %d (p25 %d p75 %d); gain in last third p50 %.2f (p10 %.2f p90 %.2f)'%(np.median(a[:,0]),np.median(a[:,1]),np.percentile(a[:,1],25),np.percentile(a[:,1],75),np.median(a[:,2]),np.percentile(a[:,2],10),np.percentile(a[:,2],90)))
print('== value function relative scale (last 100 its): sqrt(vf) / (r_step/(1-gamma)), r_step = rew_final/eplen_final')
rel=collections.defaultdict(list)
for r in ppo:
    if not r.get('rew_final') or r['rew_final']<=0 or not r.get('eplen_final'): continue
    vscale=(r['rew_final']/r['eplen_final'])/(1-0.99)
    rel[r['experiment']].append(np.sqrt(r['vf_med_last'])/vscale)
for e,v in rel.items(): print(e,len(v),'p10/50/90',np.round(np.percentile(v,[10,50,90]),3))
print('== vf trend within fresh >=2000 runs: vf last-100 / vf 100-200')
t=[]
for r in ppo:
    if r['resume'] or r['iters_logged']<2000: continue
    vf=comp[r['run_id']]['vf']; t.append(np.median(vf[-100:])/np.median(vf[100:200]))
print(np.round(np.percentile(t,[10,50,90]),2), 'n',len(t))
