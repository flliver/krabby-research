import pickle, os, numpy as np, collections, re
D=os.path.dirname(os.path.abspath(__file__))
FLOOR=1.00001e-5
meta=pickle.load(open(D+"/meta.pkl","rb"))
def load(pk,m):
    d=pickle.load(open(pk,"rb")); mm={p:(a,c) for p,a,c in m}
    out=[]
    for p,(lr,e) in d.items():
        if p in mm and mm[p][0]=="PPOWithExtractor" and lr:
            s=[x for x,_ in lr]; v=np.array([y for _,y in lr])
            out.append(dict(p=p,cfg=mm[p][1][0],v=v,s=s,dup=len(s)-len(set(s))))
    return out
main=load(D+"/main_lr.pkl",meta["main"]); uniq=load(D+"/unique_lr.pkl",meta["uniq"])
print("dup steps main",sum(r["dup"] for r in main),"uniq",sum(r["dup"] for r in uniq))
for lab,rows in [("MAIN",main),("UNIQ",uniq)]:
    print("==",lab)
    g=collections.defaultdict(list)
    for r in rows: g[r["cfg"]].append(r)
    for k,rs in sorted(g.items()):
        allv=np.concatenate([r["v"] for r in rs])
        fs=[np.mean(r["v"]<=FLOOR) for r in rs]
        print(f" cfg {k}: runs {len(rs)} iters {len(allv)} iterw-floor {np.mean(allv<=FLOOR):.3f} median-run floor {np.median(fs):.3f} (IQR {np.percentile(fs,25):.2f}-{np.percentile(fs,75):.2f}) mean lr {allv.mean():.2e} median lr {np.median(allv):.2e} p90 {np.percentile(allv,90):.2e}  first-at-floor {sum(r['v'][0]<=FLOOR for r in rs)}  runs floor<0.5 {sum(f<0.5 for f in fs)}")
    # runs with lowest floor shares & long
    rs=sorted([r for r in rows if len(r["v"])>=500], key=lambda r: np.mean(r["v"]<=FLOOR))
    print(" lowest floor-share runs >=500 its:")
    for r in rs[:5]: print("   ",r["p"].split("rsl_rl/")[-2][-60:] if False else r["p"].replace("/home/nickmagus/krabby/krabby-research/parkour/",""), len(r["v"]), round(float(np.mean(r["v"]<=FLOOR)),3), f"{r['v'].mean():.2e}")
    fs=np.array([np.mean(r["v"]<=FLOOR) for r in rows])
    print(" run floor-share hist", np.histogram(fs,bins=[0,0.1,0.2,0.3,0.5,0.7,0.9,0.99,1.0001])[0])
    lens=np.array([len(r["v"]) for r in rows]); print(" min floor-share run", fs.min(), "runs with floor share <0.2:", sum(fs<0.2), "their lens", sorted(lens[fs<0.2])[:20])
    first=collections.Counter(float(f"{r['v'][0]:.2g}") for r in rows); print(" lr_first",first.most_common(8))
