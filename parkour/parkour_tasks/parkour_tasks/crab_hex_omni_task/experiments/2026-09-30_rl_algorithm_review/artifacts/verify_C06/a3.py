import pickle, numpy as np
D=pickle.load(open('../telemetry/runs.pkl','rb')); rows=D['rows']; comp=D['compact']
for r in rows:
    if r['run'].startswith('2026-08-20') and r.get('alg_class')=='PPOWithExtractor':
        s=comp[r['run_id']]['std']; rm=np.maximum.accumulate(s); dd=rm-s; i=int(np.argmax(dd))
        pk=int(np.argmax(s[:i+1]))
        print(r['experiment'], r['run'], len(s), 'first %.4f'%s[0], 'peak@%d %.4f'%(pk,s[pk]), 'trough@%d %.4f dd %.4f'%(i,s[i],dd[i]), 'last %.4f'%s[-1], 's[::200]', np.round(s[::250],3))
