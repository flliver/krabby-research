import pickle, os, re, numpy as np, yaml, collections
def algclass(path):
    a=os.path.join(os.path.dirname(path),"params","agent.yaml")
    if not os.path.exists(a): return None, None
    txt=open(a).read()
    m=re.search(r"class_name:\s*(\S+)", txt[txt.find("algorithm:"):]) if "algorithm:" in txt else None
    lr=re.search(r"learning_rate:\s*(\S+)", txt)
    sch=re.search(r"schedule:\s*(\S+)", txt)
    return (m.group(1) if m else None), (float(lr.group(1)) if lr else None, sch.group(1) if sch else None)
def summarize(pk, label):
    d=pickle.load(open(pk,"rb"))
    rows=[]
    for p,(lr,err) in d.items():
        if err: print("ERR",p,err)
        ac,cfg=algclass(p)
        if not lr: continue
        rows.append((p,ac,cfg,np.array([v for s,v in lr]),[s for s,v in lr]))
    print("==",label,"files",len(d),"with lr",len(rows), collections.Counter(r[1] for r in rows))
    return rows
FLOOR=1.00001e-5
def stats(rows,label):
    if not rows: print(label,"none"); return
    allv=np.concatenate([r[3] for r in rows])
    n=len(allv)
    print(f"{label}: runs {len(rows)} iters {n} floor {np.mean(allv<=FLOOR):.4f} le3.4e-5 {np.mean(allv<=3.4e-5+1e-9):.4f} gt1e-4 {np.mean(allv>1e-4):.4f} mean {allv.mean():.3e}  first-at-floor {sum(r[3][0]<=FLOOR for r in rows)}")
    by=collections.defaultdict(list)
    for r in rows: by[(r[2] or (None,None))[0]].append(np.mean(r[3]<=FLOOR))
    for k,v in sorted(by.items(), key=lambda x:str(x[0])): print("   lr_cfg",k,"n",len(v),"median floor share",round(float(np.median(v)),3))
    sch=collections.Counter((r[2] or (None,None))[1] for r in rows); print("   schedules",sch)
main=summarize(D+"/main_lr.pkl","main") if (D:=os.path.dirname(os.path.abspath(__file__))) else None
uniq=summarize(D+"/unique_lr.pkl","unique")
ppo=lambda rows:[r for r in rows if r[1]=="PPOWithExtractor"]
stud=lambda rows:[r for r in rows if r[1]!="PPOWithExtractor"]
print(collections.Counter(r[1] for r in main), collections.Counter(r[1] for r in uniq))
stats(ppo(main),"MAIN PPO")
stats(ppo(uniq),"UNIQUE PPO")
stats(ppo(main)+ppo(uniq),"ALL PPO")
stats([r for r in uniq if r[1]!="PPOWithExtractor"],"UNIQUE nonPPO")
pickle.dump({"main":[(r[0],r[1],r[2]) for r in main],"uniq":[(r[0],r[1],r[2]) for r in uniq]},open(D+"/meta.pkl","wb"))
