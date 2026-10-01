import pickle, numpy as np, collections, os
res=pickle.load(open("lr.pkl","rb"))
F=1e-5*(1+1e-5)
ppo=[r for r in res if r[1]=="PPOWithExtractor"]
print("files",len(res),"ppo",len(ppo),"classes",collections.Counter(r[1] for r in res))
print("errs",[ (r[0],r[4]) for r in res if r[4]])
allv=[]; first_floor=0; per=collections.defaultdict(list); dupruns=0; nolr=0
for p,cls,lr,out,err in ppo:
    if not out: nolr+=1; continue
    steps=[s for s,_ in out]
    if len(set(steps))!=len(steps): dupruns+=1
    v=np.array([x for _,x in sorted(out)])
    allv.append(v)
    out_s=sorted(out)
    if out_s[0][1]<=F: first_floor+=1
    per[lr].append(np.mean(v<=F))
a=np.concatenate(allv)
print("n runs with lr",len(allv),"nolr",nolr,"dup-step runs",dupruns)
print("total iters",len(a))
print("floor frac",np.mean(a<=F))
print("<=3.4e-5 frac",np.mean(a<=3.4e-5*(1+1e-5)), "<=3.5e-5", np.mean(a<=3.5e-5))
print(">1e-4 frac",np.mean(a>1e-4))
print("mean",a.mean())
print("first at floor",first_floor,"of",len(allv))
for k,v in per.items(): print("lr_cfg",k,"n",len(v),"median floor share",np.median(v),"iter-wt?")
# distinct values
u,c=np.unique(np.round(a,9),return_counts=True); idx=np.argsort(-c)[:12]
print([(u[i],c[i]) for i in idx])
