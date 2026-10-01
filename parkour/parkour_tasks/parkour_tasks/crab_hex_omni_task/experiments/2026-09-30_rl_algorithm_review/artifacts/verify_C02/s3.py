import pickle,numpy as np,collections
exec(open("stats2.py").read().split("for lab,rows")[0])
u=[r for r in uniq if "/Trash/" not in r["p"]]; A=main+u
F=1.00001e-5
for lab,rows in [("UNIQ-noTrash",u),("ALL",A)]:
    v=np.concatenate([r["v"] for r in rows])
    print(lab,len(rows),len(v),round(np.mean(v<=F),4),round(np.mean(v<=3.4e-5+1e-9),4),round(np.mean(v>1e-4),4),f"{v.mean():.3e}",sum(r["v"][0]<=F for r in rows))
    g=collections.defaultdict(list)
    for r in rows: g[r["cfg"]].append(np.mean(r["v"]<=F))
    print("  ",{k:(len(x),round(float(np.median(x)),3)) for k,x in sorted(g.items())})
print(sorted(set(r["p"].split("/experiments/")[1].split("/logs/")[0].split("/")[0] for r in u)))
