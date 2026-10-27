import pickle, numpy as np, collections
res=pickle.load(open('outside.pkl','rb'))
has=[r for r in res if len(r[1])]
print('files', len(res), 'with std', len(has), 'errors', sum(r[3] is not None for r in res))
n_end=sum(r[1][-1]>=1.99 for r in has); tot=sum(len(r[1]) for r in has); ge=sum((r[1]>=1.99).sum() for r in has)
print('end>=1.99', n_end, 'iterfrac', ge/tot, 'iters', tot)
print('std global min/max', min(r[1].min() for r in has), max(r[1].max() for r in has))
print('std_first dist', collections.Counter(round(float(r[1][0]),2) for r in has).most_common(12))
dips=[(r[0].split('experiments/')[1][:90], round(float(r[1][0]),3), round(float(r[1].min()),3), round(float(r[1][-1]),3), len(r[1])) for r in has if r[1].min()<r[1][0]-0.01]
print('dips below start >0.01:', len(dips)); [print(' ',d) for d in dips]
dd=sorted([(float(np.max(np.maximum.accumulate(r[1])-r[1])), r[0].split('experiments/')[1][:90]) for r in has], reverse=True)[:12]
print('top drawdowns'); [print(' ',d) for d in dd]
above=[(r[0].split('experiments/')[1][:90], float(r[1].max())) for r in has if r[1].max()>2.001]
print('max >2.001', len(above), above[:10])
ents=[(float(np.nanmin(r[2])),float(np.nanmax(r[2])), r[0].split('experiments/')[1][:70]) for r in has if len(r[2])]
print('ent min', sorted(ents)[:4]); print('ent max', sorted(ents,key=lambda x:x[1])[-3:])
for r in res:
    if 'sigma_fix' in r[0] or len(r[1])==0: print('SPECIAL', r[0].split('experiments/')[1], len(r[1]), r[1][:1], r[1][-1:] if len(r[1]) else None, r[1].min() if len(r[1]) else None, r[3])
