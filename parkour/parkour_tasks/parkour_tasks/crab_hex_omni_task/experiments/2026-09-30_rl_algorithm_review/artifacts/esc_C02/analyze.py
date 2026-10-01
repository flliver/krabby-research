import pickle, numpy as np, collections, csv, sys
F=1e-5*(1+1e-5)
res=pickle.load(open("all.pkl","rb"))
print("files",len(res),"errors",[(r["path"],r["err"]) for r in res if r["err"]])
print("classes",collections.Counter(r["cls"] for r in res))
print("no agent.yaml",sum(r["cls"] is None for r in res), "ppo with no lr tag", sum(r["cls"]=="PPOWithExtractor" and not r["lr"] for r in res))
ppo=[r for r in res if r["cls"]=="PPOWithExtractor" and r["lr"]]
for r in ppo:
    o=sorted(r["lr"]); r["steps"]=[s for s,_ in o]; r["v"]=np.array([v for _,v in o]); r["dup"]=len(r["steps"])-len(set(r["steps"]))
print("dup-step runs",sum(r["dup"]>0 for r in ppo), "schedules",collections.Counter(r["sched"] for r in ppo))
main=[r for r in ppo if "/parkour/logs/rsl_rl/" in r["path"]]
camp=[r for r in ppo if "/experiments/" in r["path"]]
census={row["events_path"]:int(row["iters_logged"]) for row in csv.DictReader(open("../telemetry/runs.csv"))}
def block(rows,label):
    a=np.concatenate([r["v"] for r in rows]); n=len(a)
    print(f"\n== {label}: runs {len(rows)} iters {n}")
    print(f"  floor(<=1e-5) {np.mean(a<=F):.4f}  <=3.4e-5 {np.mean(a<=3.4e-5*(1+1e-6)):.4f}  >1e-4 {np.mean(a>1e-4):.5f}  mean {a.mean():.3e}  median {np.median(a):.2e}")
    ff=sum(r["v"][0]<=F for r in rows); s0=[r for r in rows if r["steps"][0]==0]; s0f=sum(r["v"][0]<=F for r in s0)
    print(f"  first-logged-at-floor {ff}/{len(rows)};  first step==0: {len(s0)} runs, at floor {s0f};  first step>0 (resume): {len(rows)-len(s0)}, at floor {ff-s0f}")
    fs1=[r for r in rows if r["steps"][0]!=0]; print("   resume runs' first step dist sample",sorted(collections.Counter(r['steps'][0] for r in fs1).items())[:8])
    g=collections.defaultdict(list)
    for r in rows: g[r["lr_cfg"]].append(r)
    for k,rs in sorted(g.items(), key=lambda x:-x[0]):
        c=np.concatenate([r["v"] for r in rs]); f=[np.mean(r["v"]<=F) for r in rs]
        print(f"  lr_cfg {k:g}: runs {len(rs)} iters {len(c)} per-run-median-floor {np.median(f):.3f} iterwt-floor {np.mean(c<=F):.3f} mean-lr {c.mean():.3g} median-lr {np.median(c):.2g} p90 {np.percentile(c,90):.2g} first-at-floor {sum(r['v'][0]<=F for r in rs)} resume {collections.Counter(r['resume'] for r in rs)}")
    # resume x cfg
    g2=collections.defaultdict(list)
    for r in rows: g2[(r["lr_cfg"],r["resume"])].append(r)
    for k,rs in sorted(g2.items(), key=lambda x:str(x[0])):
        c=np.concatenate([r["v"] for r in rs]); print(f"    cfg{k[0]:g} resume={k[1]}: runs {len(rs)} iters {len(c)} floor {np.mean(c<=F):.3f} mean {c.mean():.3g} first-at-floor {sum(r['v'][0]<=F for r in rs)}")
    u,cn=np.unique(np.round(a,10),return_counts=True); idx=np.argsort(-cn)[:6]; print("  top values",[(float(u[i]),int(cn[i])) for i in idx])
    return a
# census-truncated main
tr=[]
for r in main:
    k=census.get(r["path"])
    if k is None: print("  NOT IN CENSUS:",r["path"].split("rsl_rl/")[-1],len(r["v"])); continue
    rr=dict(r); rr["v"]=r["v"][:k]; rr["steps"]=r["steps"][:k]; tr.append(rr)
    if len(r["v"])!=k: print("  grew:",r["path"].split("rsl_rl/")[-1].split("/events")[0],k,"->",len(r["v"]))
block(tr,"MAIN census-truncated")
block(main,"MAIN now")
block(camp,"CAMPAIGN copies (forward_task/experiments)")
block(main+camp,"ALL (main + campaign)")
# fresh-run in-iteration-0 collapse: how many fresh runs at floor at step 0 given cfg
fresh=[r for r in main+camp if r["steps"][0]==0]
print("\nfresh runs step0 value by cfg:")
for k in sorted(set(r["lr_cfg"] for r in fresh), reverse=True):
    rs=[r for r in fresh if r["lr_cfg"]==k]; print(f"  cfg {k:g}: n {len(rs)} at floor at step0 {sum(r['v'][0]<=F for r in rs)}; step0 values {collections.Counter(float('%.2g' % r['v'][0]) for r in rs).most_common(5)}")
# min halvings needed
import math; print("halvings 3e-4->1e-5:",math.log(30)/math.log(1.5), " 1e-4:",math.log(10)/math.log(1.5)," 3e-5:",math.log(3)/math.log(1.5)," 2e-4:",math.log(20)/math.log(1.5))
