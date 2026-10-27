"""Stage 3: census analysis over runs.pkl -> printed tables (analysis.txt)."""
import os, pickle, math, collections
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(os.path.join(OUT, "runs.pkl"), "rb"))
rows, comp = D["rows"], D["compact"]
FLOOR = 1e-5


def g(r, k, default=None):
    v = r.get(k, default)
    return default if v is None or (isinstance(v, float) and math.isnan(v)) else v


def is_ppo(r):
    return r.get("alg_class") == "PPOWithExtractor" or (r.get("alg_class") is None and "lr" in comp[r["run_id"]]
                                                          and r["experiment"] != "crab_hex_student")


ppo = [r for r in rows if is_ppo(r) and g(r, "iters_logged", 0) > 0]
stud = [r for r in rows if r["experiment"] == "crab_hex_student"]
print("rows", len(rows), "ppo", len(ppo), "student", len(stud))
print("alg classes", collections.Counter(r.get("alg_class") for r in rows))
print("no agent yaml", [(r["experiment"], r["run"]) for r in rows if not r.get("has_agent_yaml")])
print("parse errors", [(r["experiment"], r["run"], r["parse_error"]) for r in rows if r["parse_error"]])
print("zero iters", [(r["experiment"], r["run"]) for r in rows if not g(r, "iters_logged", 0)])

# ---------------------------------------------------------------- per-experiment overview
print("\n== OVERVIEW per experiment (PPO runs)")
by = collections.defaultdict(list)
for r in ppo:
    by[r["experiment"]].append(r)
hdr = "exp n iters_total iters_med iters_max gpu_h wall_h fps_med s_it_med learnfrac num_envs"
print(hdr)
tot_g = tot_w = tot_i = 0
for e, rs in sorted(by.items()):
    it = [g(r, "iters_logged", 0) for r in rs]
    gh = sum(g(r, "train_gpu_h", 0) for r in rs); wh = sum(g(r, "wall_clock_h", 0) for r in rs)
    tot_g += gh; tot_w += wh; tot_i += sum(it)
    print(e, len(rs), sum(it), int(np.median(it)), max(it), round(gh, 1), round(wh, 1),
          int(np.median([g(r, "fps_median", 0) for r in rs])),
          round(float(np.median([g(r, "s_per_iter_median", 0) for r in rs])), 3),
          round(float(np.median([g(r, "learn_frac", 0) for r in rs])), 3),
          collections.Counter(r.get("num_envs") for r in rs))
print("TOTAL ppo", len(ppo), tot_i, round(tot_g, 1), round(tot_w, 1))
sg = sum(g(r, "train_gpu_h", 0) for r in stud); sw = sum(g(r, "wall_clock_h", 0) for r in stud)
print("student", len(stud), sum(g(r, "iters_logged", 0) for r in stud), round(sg, 1), round(sw, 1),
      "fps_med", np.median([g(r, "fps_median", 0) for r in stud]), "s_it", np.median([g(r, "s_per_iter_median", 0) for r in stud]))

# ---------------------------------------------------------------- config census
print("\n== CONFIG census (PPO)")
for k in ["lr_cfg", "schedule", "desired_kl", "entropy_coef", "init_noise_std", "action_std_max", "action_std_reset",
          "clip_param", "epochs", "minibatches", "gamma", "lam", "steps_per_env", "clip_actions", "sym_coef", "sym_func",
          "resume", "resume_alg_state", "num_envs", "max_grad_norm", "noise_std_type", "policy_class"]:
    print(k, dict(collections.Counter(str(r.get(k)) for r in ppo)))

# ---------------------------------------------------------------- (1) LR
print("\n== (1) LR floor")
ff = np.array([g(r, "lr_frac_floor", 0) for r in ppo])
print("frac of iterations at floor: all-iter weighted",
      round(sum(g(r, "lr_frac_floor", 0) * r["iters_logged"] for r in ppo) / sum(r["iters_logged"] for r in ppo), 4))
print("runs by frac-floor bucket", {b: int(((ff >= lo) & (ff < hi)).sum()) for b, (lo, hi) in
                                     {"<0.5": (0, .5), "0.5-0.9": (.5, .9), "0.9-0.99": (.9, .99), "0.99-1": (.99, 1.0001)}.items()})
ffi = [g(r, "lr_first_floor_iter") for r in ppo]
print("first floor iter: n None", sum(x is None for x in ffi), "dist",
      collections.Counter(("0" if x == 0 else "1-9" if x < 10 else "10-99" if x < 100 else ">=100") for x in ffi if x is not None))
print("lr_first (logged at iter0) dist", collections.Counter(f"{g(r,'lr_first'):.2g}" for r in ppo))
print("runs whose LR is ever > 1e-4:", sum(g(r, "lr_max", 0) > 1e-4 for r in ppo), "ever > 2e-5:", sum(g(r, "lr_max", 0) > 2e-5 for r in ppo))
print("iteration-weighted frac > 1e-4:",
      round(sum(g(r, "lr_frac_gt_1e-4", 0) * r["iters_logged"] for r in ppo) / sum(r["iters_logged"] for r in ppo), 5))
for e, rs in sorted(by.items()):
    f = np.array([g(r, "lr_frac_floor", 0) for r in rs]); it = np.array([r["iters_logged"] for r in rs])
    print(f"  {e}: iter-weighted floor frac {np.sum(f*it)/np.sum(it):.4f}; runs <0.9 floor: {int((f<0.9).sum())}/{len(rs)}; median {np.median(f):.4f}")

# off-floor time course (by decile of run)
print("\n LR off-floor fraction by run decile (iteration-weighted over PPO runs with >=500 its)")
dec = np.zeros(10); cnt = np.zeros(10)
for r in ppo:
    v = comp[r["run_id"]].get("lr")
    if v is None or len(v) < 500:
        continue
    off = v > FLOOR * 1.01
    for i, ch in enumerate(np.array_split(off, 10)):
        dec[i] += ch.sum(); cnt[i] += len(ch)
print(" ", np.round(dec / cnt, 4))

# group by sym coef / std pinned
print("\n LR off-floor vs config groups (iteration-weighted)")
def grp(keyf, rs=ppo):
    out = collections.defaultdict(lambda: [0, 0, 0])
    for r in rs:
        k = keyf(r)
        out[k][0] += g(r, "lr_frac_floor", 0) * r["iters_logged"]; out[k][1] += r["iters_logged"]; out[k][2] += 1
    for k, (a, b, n) in sorted(out.items(), key=lambda x: str(x[0])):
        print(f"   {k}: runs {n}, iters {b}, floor frac {a/b:.4f}")
grp(lambda r: ("sym", r.get("sym_coef")))
grp(lambda r: ("lr_cfg", r.get("lr_cfg")))
grp(lambda r: ("resume", r.get("resume")))
grp(lambda r: ("std_cap", r.get("action_std_max")))
grp(lambda r: ("ent", r.get("entropy_coef")))

# LR off floor while sigma below cap vs at cap
print("\n LR off-floor rate conditional on sigma state (iteration-weighted)")
a = collections.Counter()
for r in ppo:
    c = comp[r["run_id"]]
    if "lr" not in c or "std" not in c:
        continue
    n = min(len(c["lr"]), len(c["std"]))
    cap = g(r, "std_cap", 2.0)
    pinned = c["std"][:n] >= cap - 0.01
    off = c["lr"][:n] > FLOOR * 1.01
    a["pinned_n"] += pinned.sum(); a["pinned_off"] += (pinned & off).sum()
    a["free_n"] += (~pinned).sum(); a["free_off"] += (~pinned & off).sum()
print("  sigma<cap: off-floor", a["free_off"] / max(a["free_n"], 1), "of", a["free_n"],
      "| sigma at cap: off-floor", a["pinned_off"] / max(a["pinned_n"], 1), "of", a["pinned_n"])

print("\n runs with LR off floor >= 10% of iterations:")
for r in sorted(ppo, key=lambda r: g(r, "lr_frac_floor", 0)):
    if g(r, "lr_frac_floor", 0) < 0.9:
        print(f"   {r['experiment']}/{r['run']} its={r['iters_logged']} floor={g(r,'lr_frac_floor',0):.3f} lrmax={g(r,'lr_max',0):.2e} "
              f"exc={r.get('lr_n_excursions')} longest={r.get('lr_longest_excursion')} sym={r.get('sym_coef')} lr_cfg={r.get('lr_cfg')} "
              f"std_first={g(r,'std_first',0):.2f} std_pin_it={r.get('std_first_iter_ge_cap')} resume={r.get('resume')} init_std={r.get('init_noise_std')}")

# ---------------------------------------------------------------- (2) sigma
print("\n== (2) sigma pinning")
sp = np.array([g(r, "std_frac_ge_cap", 0) for r in ppo])
print("runs ever at cap:", int(sum(g(r, "std_max", 0) >= g(r, "std_cap", 2.0) - 0.01 for r in ppo)), "/", len(ppo))
print("runs pinned at iter 0 (std_first>=cap-0.01):", int(sum(g(r, "std_first", 0) >= g(r, "std_cap", 2.0) - 0.01 for r in ppo)))
print("iteration-weighted frac at cap:", round(sum(g(r, "std_frac_ge_cap", 0) * r["iters_logged"] for r in ppo) / sum(r["iters_logged"] for r in ppo), 4))
print("std_last dist:", collections.Counter(f"{g(r,'std_last',0):.1f}" for r in ppo))
print("std_max (raw param) dist:", collections.Counter(f"{g(r,'std_max',0):.2f}" for r in ppo).most_common(12))
fresh = [r for r in ppo if not r.get("resume") and abs(g(r, "std_first", 0) - g(r, "init_noise_std", 0) or 0) < 0.05]
fi = [r.get("std_first_iter_ge_cap") for r in fresh]
print("from-scratch-like runs (std_first == init):", len(fresh), "pinned eventually:", sum(x is not None for x in fi),
      "first-pin iter percentiles:", np.percentile([x for x in fi if x is not None], [0, 10, 25, 50, 75, 90, 100]).round() if any(x is not None for x in fi) else None)
print("never pinned fresh runs (its):", sorted([r["iters_logged"] for r in fresh if r.get("std_first_iter_ge_cap") is None]))
for e, rs in sorted(by.items()):
    it = np.array([r["iters_logged"] for r in rs]); f = np.array([g(r, "std_frac_ge_cap", 0) for r in rs])
    print(f"  {e}: iter-weighted frac at cap {np.sum(f*it)/np.sum(it):.3f}; runs ending at cap {sum(g(r,'std_last',0)>=g(r,'std_cap',2)-0.01 for r in rs)}/{len(rs)}")

# std trajectory for fresh long runs: std at iteration 0, 500, 1000, 2000, 5000
print("\n std trajectory percentiles for fresh runs (value at iteration k):")
for k in [0, 100, 300, 500, 1000, 2000, 3000, 5000, 10000]:
    vals = [comp[r["run_id"]]["std"][k] for r in fresh if "std" in comp[r["run_id"]] and len(comp[r["run_id"]]["std"]) > k]
    if vals:
        print(f"   it {k}: n={len(vals)} p10/50/90 = {np.percentile(vals,[10,50,90]).round(3)}")

# ---------------------------------------------------------------- (3) non-default knobs
print("\n== (3) non-default knob runs")
def nd(r):
    tags = []
    if r.get("action_std_max") not in (None, ""): tags.append(f"std_max={r['action_std_max']}")
    if r.get("action_std_reset") not in (None, ""): tags.append(f"std_reset={r['action_std_reset']}")
    if r.get("entropy_coef") not in (None, 0.01): tags.append(f"ent={r['entropy_coef']}")
    if r.get("lr_cfg") not in (None, 3e-4, 2e-4): tags.append(f"lr={r['lr_cfg']}")
    if r.get("init_noise_std") not in (None, 1.5): tags.append(f"init_std={r['init_noise_std']}")
    if r.get("sym_coef") not in (None, 0.5): tags.append(f"sym={r['sym_coef']}")
    if r.get("desired_kl") not in (None, 0.01): tags.append(f"kl={r['desired_kl']}")
    if r.get("schedule") not in (None, "adaptive"): tags.append(f"sched={r['schedule']}")
    return tags
for r in ppo:
    t = nd(r)
    if t:
        print(f"  {r['experiment']}/{r['run']} {t} its={r['iters_logged']} lr first/last/min/max={g(r,'lr_first',0):.2e}/{g(r,'lr_last',0):.2e}/{g(r,'lr_min',0):.2e}/{g(r,'lr_max',0):.2e} "
              f"floorfrac={g(r,'lr_frac_floor',0):.3f} std first/last/max={g(r,'std_first',0):.3f}/{g(r,'std_last',0):.3f}/{g(r,'std_max',0):.3f} "
              f"pinfrac={g(r,'std_frac_ge_cap',0):.3f} rew {g(r,'rew_first',0):.2f}->{g(r,'rew_final',0):.2f} eplen {g(r,'eplen_first',0):.0f}->{g(r,'eplen_final',0):.0f} resume={r.get('resume')}")

# ---------------------------------------------------------------- (4) value loss
print("\n== (4) value loss")
for e, rs in sorted(by.items()):
    vf0 = [g(r, "vf_med_first") for r in rs if g(r, "vf_med_first") is not None]
    vf1 = [g(r, "vf_med_last") for r in rs if g(r, "vf_med_last") is not None]
    rat = [g(r, "vf_med_last") / g(r, "vf_med_first") for r in rs if g(r, "vf_med_first")]
    vmax = [g(r, "vf_max") for r in rs if g(r, "vf_max") is not None]
    print(f"  {e}: vf first-100 median p50={np.median(vf0):.4g} [p10 {np.percentile(vf0,10):.3g}, p90 {np.percentile(vf0,90):.3g}]; "
          f"last-100 p50={np.median(vf1):.4g} [p10 {np.percentile(vf1,10):.3g}, p90 {np.percentile(vf1,90):.3g}]; last/first p50={np.median(rat):.3g}; "
          f"max over runs {max(vmax):.4g}; runs with last>first {sum(x>1 for x in rat)}/{len(rat)}")
# vf vs reward scale
print("  vf_last / rew_final^2 (fresh-ish runs) p50:",
      np.median([g(r, "vf_med_last") / max(g(r, "rew_final", 1) ** 2, 1e-9) for r in ppo if g(r, "vf_med_last") and g(r, "rew_final")]))
print("  surrogate last-100 median p10/50/90:", np.percentile([g(r, "surr_med_last") for r in ppo if g(r, "surr_med_last") is not None], [10, 50, 90]))
print("  entropy first/last p50:", np.median([g(r, "ent_first") for r in ppo if g(r, "ent_first") is not None]),
      np.median([g(r, "ent_last") for r in ppo if g(r, "ent_last") is not None]))
print("  symmetry loss last-100 median p10/50/90:", np.percentile([g(r, "sym_med_last") for r in ppo if g(r, "sym_med_last") is not None], [10, 50, 90]),
      "first-100:", np.percentile([g(r, "sym_med_first") for r in ppo if g(r, "sym_med_first") is not None], [10, 50, 90]))

# ---------------------------------------------------------------- (6) reward curves
print("\n== (6) reward curves")
for e, rs in sorted(by.items()):
    fr = [r for r in rs if not r.get("resume") and g(r, "rew_iter_90pct") is not None]
    rr = [r for r in rs if r.get("resume") and g(r, "rew_iter_90pct") is not None]
    if fr:
        print(f"  {e} fresh n={len(fr)}: its-to-90% p25/50/75={np.percentile([r['rew_iter_90pct'] for r in fr],[25,50,75]).round()} "
              f"as frac of run p50={np.median([r['rew_frac_run_90pct'] for r in fr]):.2f}; run length p50={np.median([r['iters_logged'] for r in fr])}")
    if rr:
        print(f"  {e} resumed n={len(rr)}: its-to-90% p50={np.median([r['rew_iter_90pct'] for r in rr])}; frac p50={np.median([r['rew_frac_run_90pct'] for r in rr]):.2f}")
print(" late decline (smoothed max - final)/(max - start) > 0.2, runs >= 1000 its:")
for r in ppo:
    ld = g(r, "rew_late_decline_frac")
    if ld is not None and ld > 0.2 and r["iters_logged"] >= 1000:
        print(f"   {r['experiment']}/{r['run']} its={r['iters_logged']} decline={ld:.2f} rew start/max/final={g(r,'rew_first',0):.2f}/{g(r,'rew_max_smooth',0):.2f}/{g(r,'rew_final',0):.2f} "
              f"max@{r.get('rew_iter_max_smooth')} eplen {g(r,'eplen_first',0):.0f}->{g(r,'eplen_final',0):.0f} resume={r.get('resume')}")

# ---------------------------------------------------------------- (7) anomalies
print("\n== (7) anomalies")
print(" non-finite scalar values:", [(r["experiment"], r["run"], r["nonfinite_any"]) for r in rows if g(r, "nonfinite_any", 0)])
print(" LR jump ratio >1.6 between consecutive iterations (max possible per-iteration x1.5^80):")
for r in ppo:
    j = g(r, "lr_max_jump_ratio")
    if j and j > 10:
        print(f"   {r['experiment']}/{r['run']} max jump x{j:.1f}")
print(" vf_max > 100:", [(r["experiment"], r["run"], round(g(r, "vf_max"), 1)) for r in ppo if g(r, "vf_max", 0) > 100])
print(" surrogate max > 1:", [(r["experiment"], r["run"], round(g(r, "surr_max"), 2)) for r in ppo if g(r, "surr_max", 0) > 1])
print(" num_envs != 256:", [(r["experiment"], r["run"], r.get("num_envs")) for r in rows if r.get("num_envs") not in (256,)])
print(" iters_logged != last-first+1:", [(r["experiment"], r["run"], r["iters_logged"], r["first_iter"], r["last_iter"]) for r in rows
                                          if r["iters_logged"] and r["last_iter"] - r["first_iter"] + 1 != r["iters_logged"]][:20])
