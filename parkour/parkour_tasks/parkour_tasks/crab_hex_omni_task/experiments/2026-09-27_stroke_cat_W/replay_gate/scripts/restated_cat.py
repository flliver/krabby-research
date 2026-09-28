"""Restated R3 / R5 in-plant / R7 / R8 (stroke_posture SA0 decisions 6-7) for a per-category gate run.
usage: restated_cat.py <config.json> <gate_out_dir>   -> <gate_out_dir>/restated_cat.json
R3  short-stance share <= 2 % (the tool's R3 short_share, every healthy + sideways row) AND tap test:
    Spearman(pay per scored stance, contact_switch_hz) >= 0 on side (each healthy head; point, CI reported).
R5  in-plant: the tool's no_flip clause (rg1 over healthy+posture vs degenerate, at the table) AND creeper
    term-alone <= 0.5 x healthy on translation rows; cross-plant clause not run (reported as known FAIL).
    Plus the creeper's move_turn omega x r credit at the table's move_turn weight.
R7  clip shift <= 5 pp and absorbed <= 15 % (the tool's R7 rows, at the table).
R8  hand gap: FAIL only if |gap| > 10 % AND its 95 % episode-bootstrap CI excludes 0 (restated.r8r).
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
T=("/home/nickmagus/krabby/krabby-research/.claude/worktrees/stroke-posture/parkour/parkour_tasks/parkour_tasks/"
   "crab_hex_omni_task/experiments/tools/replay_stroke_posture.py")
spec=importlib.util.spec_from_file_location("rsp",T); R=importlib.util.module_from_spec(spec); sys.modules["rsp"]=R; spec.loader.exec_module(R)
sys.path.insert(0,"/home/nickmagus/krabby/krabby-research/.claude/worktrees/stroke-posture/parkour/parkour_tasks/parkour_tasks/"
                  "crab_hex_omni_task/experiments/2026-09-26_stroke_posture/replay_gate_calib")
import restated as RS

cfg=json.load(open(sys.argv[1])); out_dir=Path(sys.argv[2])
rep=json.load(open(out_dir/"replay_gate.json")); crit=rep["criteria"]
params=R.parse_params([f"{k}={v}" for k,v in (cfg.get("params") or {}).items()])
econ_yaml=cfg.get("economy_env_yaml") or rep["economy"]["source"]
econ,_=R.load_economy(econ_yaml)
table=rep["stroke_cat_w"]["table"]; w_S=rep["w_S"]; w_eff=rep["stroke_cat_w"]["w_eff"]
runs={}
for key in ("healthy","degenerate"):
    for n,p in cfg[key].items():
        st=R.load_stamp(p); rp=R.replay(st,params); runs[n]=(key,st,rp)
res={}
# ---- R3
short=crit["R3"]["short_share"]
ok3a=all(v is None or v<=0.02 for v in short.values())
rng=np.random.default_rng(7); tap={}
for n,(role,st,rp) in runs.items():
    if role!="healthy": continue
    sd=rp.stances
    idx=[i for i,e in enumerate(st.episodes) if e["category"]=="side"]
    sw=np.array([(((st.json_eps[i] or {}).get("holds") or {}).get(st.episodes[i]["label"],{}).get("quality") or {})
                 .get("contact_switch_hz",np.nan) for i in idx],float)
    pps=[]
    for i in idx:
        kk=(sd["env"]==i)&sd["payable"]&st.own[sd["env"],sd["step"]]&rp.gate[sd["env"],sd["step"]]
        pps.append(sd["pay"][kk].mean() if kk.any() else np.nan)
    pps=np.array(pps,float)
    rho=RS.spearman(pps,sw)
    m=np.flatnonzero(np.isfinite(pps)&np.isfinite(sw))
    bs=np.array([x for x in (RS.spearman(pps[b],sw[b]) for b in (rng.choice(m,m.size) for _ in range(1000))) if x is not None])
    tap[f"{n}.side"]={"rho_pay_per_stance_vs_switch":rho,"ci95":[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],"n":int(m.size)}
ok3b=all(v["rho_pay_per_stance_vs_switch"] is not None and v["rho_pay_per_stance_vs_switch"]>=0 for v in tap.values())
mx=max(((k,v) for k,v in short.items() if v is not None),key=lambda kv:kv[1])
res["R3"]={"pass":bool(ok3a and ok3b),"short_share_max":mx,"short_ok":ok3a,"tap":tap,"tap_ok":ok3b,
           "scale_invariant":True}
# ---- R5 in-plant
r5=crit["R5"]
flips=[r for r in r5["rg1"]["rows"] if r["pass"] is False]
alone={k:v for k,v in r5["term_alone"].items() if v["gated"]}
mxa=max(((k,v["ratio"]) for k,v in alone.items() if v["ratio"] is not None),key=lambda kv:kv[1])
rot={k:v["ratio"] for k,v in r5["term_alone"].items() if not v["gated"]}
# creeper move_turn omega x r credit at the table's move_turn weight, as a share of the healthy head's
# move_turn tracking income (nominal, w_mt x stroke/s at w=1 over the gated own move_turn steps)
w_mt=w_eff["move_turn"]
cr=cfg.get("creeper"); mt={}
if cr in runs:
    _,stc,rpc=runs[cr]
    gc=R.GateRun(name=cr,role="degenerate",st=stc); gc.rp=rpc
    ci=R.cat_income(gc,"move_turn")
    trc=R.tracking_rate(stc,econ)
    for n,(role,st,rp) in runs.items():
        if role!="healthy": continue
        g=R.GateRun(name=n,role=role,st=st); g.rp=rp
        hi=R.cat_income(g,"move_turn")
        m_ep=np.array([e["category"]=="move_turn" for e in st.episodes])
        msk=st.own&m_ep[:,None]&rp.gate
        track=float(R.tracking_rate(st,econ)[msk].sum()/msk.sum())
        mt[f"{cr}/{n}"]={"creeper_S_per_s":ci,"healthy_S_per_s":hi,"ratio":(ci/hi if hi else None),"w_move_turn":w_mt,
                         "healthy_track_per_s":track,
                         "creeper_credit_share_of_healthy_track":w_mt*ci/track,
                         "healthy_credit_share_of_own_track":w_mt*hi/track,
                         "creeper_excess_share_of_healthy_track":w_mt*(ci-hi)/track}
res["R5"]={"pass":bool(r5["clauses"]["no_flip"] is not False and all(v["ratio"] is None or v["ratio"]<=0.5 for v in alone.values())),
           "no_flip":r5["clauses"]["no_flip"],"n_rows":len(r5["rg1"]["rows"]),"flips":flips,
           "term_alone_translation_max":mxa,"term_alone_rotational":rot,"creeper_move_turn":mt,
           "cross_plant":"not run (forward-era stamps not loadable); known FAIL (lunge 3.0x at p=1.5), mitigated by the unison flag; not counted"}
# ---- R7
r7=crit["R7"]["rows"]
mxs=max(r7.items(),key=lambda kv:abs(kv[1]["shift_pp"])); mxab=max(((k,v["absorbed"]) for k,v in r7.items() if v["absorbed"] is not None),key=lambda kv:kv[1])
res["R7"]={"pass":bool(all(abs(v["shift_pp"])<=5 and (v["absorbed"] is None or v["absorbed"]<=0.15) for v in r7.values())),
           "max_shift":[mxs[0],mxs[1]["shift_pp"]],"max_absorbed":list(mxab),"rows":r7}
# ---- R8
r8=RS.r8r(R,{n:v for n,v in runs.items() if v[0]=="healthy"})
res["R8"]={**r8,"scale_invariant":True}
res["table"]={"w_S":w_S,"spec":rep["stroke_cat_w"]["spec"],"w_eff":w_eff}
(out_dir/"restated_cat.json").write_text(json.dumps(R._jsonable(res),indent=1))
for k in ("R3","R5","R7","R8"):
    v=res[k]; print(k,"PASS" if v["pass"] else "FAIL")
print(json.dumps(R._jsonable({k:{kk:vv for kk,vv in v.items() if kk not in ("rows","flips")} for k,v in res.items() if k!="table"}),indent=1)[:6000])
print(json.dumps(R._jsonable(r8["rows"]),indent=0))
