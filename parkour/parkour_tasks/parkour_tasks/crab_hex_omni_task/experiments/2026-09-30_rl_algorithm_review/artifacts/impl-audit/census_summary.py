import json, numpy as np, collections
D = "/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/impl-audit/"
tb = json.load(open(D + "tb_census.json")); cfg = json.load(open(D + "cfg_census.json"))
print("runs", len(tb), "errors", sum(1 for r in tb if "err" in r))
rl = [r for r in tb if r.get("n", 0) >= 50 and not r["run"].startswith("crab_hex_student") and "lr_med" in r]
print("PPO runs with >=50 its:", len(rl))
def q(a):
    a = np.array([x for x in a if x is not None], dtype=float)
    return f"n={len(a)} min {a.min():.3g} p25 {np.percentile(a,25):.3g} med {np.median(a):.3g} p75 {np.percentile(a,75):.3g} max {a.max():.3g}"
print("lr_first:", collections.Counter(round(r["lr_first"], 7) for r in rl).most_common(6))
print("lr_med:", q([r["lr_med"] for r in rl]))
print("lr_med_after100:", q([r["lr_med_after100"] for r in rl]))
print("lr_floor_frac:", q([r["lr_floor_frac"] for r in rl]))
print("lr_le_3.4e-5 frac:", q([r["lr_le_3p4e5"] for r in rl]))
print("lr_max:", q([r["lr_max"] for r in rl]))
w = np.array([r["n"] for r in rl]); print("total PPO iterations logged:", int(w.sum()))
print("iteration-weighted floor frac:", float(np.sum(w*np.array([r['lr_floor_frac'] for r in rl]))/w.sum()), "iteration-weighted frac<=3.4e-5:", float(np.sum(w*np.array([r['lr_le_3p4e5'] for r in rl]))/w.sum()))
std_last = [r.get("std_last") for r in rl if r.get("std_last") is not None]
print("std_last:", q(std_last)); print("runs ending std>=1.999:", sum(1 for s in std_last if s >= 1.999), "/", len(std_last))
st2 = [r for r in rl if r.get("std_first") is not None and abs(r["std_first"]-1.5) < 0.01]
print("runs starting at std 1.5:", len(st2), "first it reaching 2.0:", q([r["std_first_it_2"] for r in st2 if r.get("std_first_it_2") is not None]), "never reached:", sum(1 for r in st2 if r.get("std_first_it_2") is None), "their n its:", q([r['n'] for r in st2 if r.get('std_first_it_2') is None]))
print("runs starting at std>=1.999:", sum(1 for r in rl if r.get('std_first',0) >= 1.999))
print("fps med:", q([r.get("fps_med") for r in rl if r.get("fps_med")]))
print("tcol med:", q([r.get("tcol_med") for r in rl if r.get("tcol_med")]), "tlearn med:", q([r.get("tlearn_med") for r in rl if r.get("tlearn_med")]))
print("prc_last:", q([r.get("prc_last") for r in rl if r.get("prc_last") is not None]))
print("sym_med:", q([r.get("sym_med") for r in rl if r.get("sym_med") is not None]))
print("surr_med:", q([r.get("surr_med") for r in rl if r.get("surr_med") is not None]))
print("vf_med:", q([r.get("vf_med") for r in rl if r.get("vf_med") is not None]))
print("hl_last:", q([r.get("hl_last") for r in rl if r.get("hl_last") is not None]))
print("est_last:", q([r.get("est_last") for r in rl if r.get("est_last") is not None]))
print("ent_last:", q([r.get("ent_last") for r in rl if r.get("ent_last") is not None]))
by = collections.defaultdict(list)
for r in rl: by[r["run"].split("/")[0]].append(r)
for k, v in by.items():
    print(k, len(v), "lr_med med %.3g" % np.median([r["lr_med"] for r in v]), "floor frac med %.3f" % np.median([r["lr_floor_frac"] for r in v]), "std_last med %.4f" % np.nanmedian([r.get("std_last", np.nan) for r in v]), "its", sum(r["n"] for r in v))
st = [r for r in tb if r.get("run","").startswith("crab_hex_student") and r.get("n",0) >= 50]
for r in st: print("student", r["run"], r["n"], "lr", r.get("lr_med"))
