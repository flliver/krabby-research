import pickle, numpy as np
res=pickle.load(open('outside.pkl','rb'))
for r in res:
    s=r[1]
    if not len(s): continue
    rm=np.maximum.accumulate(s); dd=rm-s
    if dd.max()>0.05:
        i=int(np.argmax(dd)); pk=int(np.argmax(s[:i+1]))
        w=np.where(s>=1.99)[0]
        print(r[0].split('experiments/')[1].split('/events')[0], 'n',len(s), 'first %.3f peak@%d %.4f trough@%d %.4f last %.4f'%(s[0],pk,s[pk],i,s[i],s[-1]), 'first>=1.99', w[0] if len(w) else None, 'min after first cap', round(float(s[w[0]:].min()),4) if len(w) else None)
        print('   ', np.round(s[::max(1,len(s)//16)],3))
