import pickle, numpy as np
D=pickle.load(open('../telemetry/runs.pkl','rb')); rows=D['rows']; comp=D['compact']
ppo=[r for r in rows if r.get('alg_class')=='PPOWithExtractor']
m=[]
for r in ppo:
    s=comp[r['run_id']]['std']; w=np.where(s>=1.99)[0]
    if len(w): m.append((float(1.99 - s[w[0]:].min()) if s[w[0]:].min()<1.99 else 0.0, float(s[w[0]:].min()), r['run']))
m.sort(reverse=True); print('runs that hit 1.99:', len(m), 'worst later min', m[:5])
