import pickle, numpy as np, collections, csv
res=pickle.load(open("lr.pkl","rb"))
rows={r['events_path']:r for r in csv.DictReader(open("../telemetry/runs.csv"))}
F=1e-5*(1+1e-5)
A=[];G=collections.defaultdict(list); fs=collections.Counter(); ff=0;n=0
for p,cls,lr,out,err in res:
    if cls!="PPOWithExtractor": continue
    if p not in rows: print("NEW",p,len(out)); continue
    r=rows[p]; k=int(r['iters_logged'])
    o=sorted(out)[:k]; v=np.array([x for _,x in o]); A.append(v); G[lr].append(v)
    fs[o[0][0]==0]+=1; n+=1; ff+= v[0]<=F
    if abs(len(out)-k)>0: print("grew",p.split('rsl_rl/')[1],k,len(out))
a=np.concatenate(A)
print("runs",n,"iters",len(a),"floor",np.mean(a<=F),"<=3.4e-5",np.mean(a<=3.4e-5),">1e-4",np.mean(a>1e-4),"mean",a.mean(),"first floor",ff,"first step==0",fs)
for k,vs in G.items():
    c=np.concatenate(vs); print(k,"runs",len(vs),"iters",len(c),"iterwt mean LR %.3g median %.3g floor %.3f"%(c.mean(),np.median(c),np.mean(c<=F)), "per-run median floor", np.median([np.mean(v<=F) for v in vs]))
