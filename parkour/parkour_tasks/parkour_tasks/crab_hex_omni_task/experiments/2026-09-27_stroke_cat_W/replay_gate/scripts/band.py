"""stride_per_speed seed band (plan step 4): per setting, the B pair's side stride_per_speed (MEAN key),
the seed delta, and a POOLED MDE proxy built from the tool's own paired_bootstrap draws.

Pooled MDE proxy: step 8's pooled MDE is the centred 95 % half-width of (d_s3 + d_s2) / 2, two independent
seed-matched paired bootstraps (arm - B at each seed). No arm exists yet, so each seed's paired draw is
stood in for by the B pair's own paired bootstrap (B_s3 - B_s2, env-clustered on the shared suite),
drawn TWICE with independent bootstrap seeds; the two draws are averaged element-wise and the centred
half-width taken (resolution.half_width). Also reported: the single-pair MDE (the conservative bound).
Also the tool's own pooled_delta on arm=base-of-other-seed (arm_s3=B_s2 vs base_s3=B_s3, arm_s2=B_s3 vs
base_s2=B_s2) to show that the literal arm=base form degenerates (draws cancel)."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
T=("/home/nickmagus/krabby/krabby-research/.claude/worktrees/stroke-posture/parkour/parkour_tasks/parkour_tasks/"
   "crab_hex_omni_task/experiments/tools/replay_stroke_posture.py")
spec=importlib.util.spec_from_file_location("rsp",T); R=importlib.util.module_from_spec(spec); sys.modules["rsp"]=R; spec.loader.exec_module(R)
L="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/"
P={"W":{"s3":L+"2026-09-26_stroke_posture/seed3_X01_c6y_A15pB/suite__omni/seed001/2026-09-26_08-39-30",
        "s2":L+"2026-09-26_stroke_posture/seed2_X02_c6y_A15pB/suite__omni/seed001/2026-09-26_10-25-08"},
   "t3":{"s3":L+"2026-09-26_stroke_formation/seed3_X01_t3_A15pB/suite__omni/seed001/2026-09-26_22-56-51",
         "s2":L+"2026-09-26_stroke_formation/seed2_X02_t3_A15pB/suite__omni/seed001/2026-09-27_04-08-09"}}
KEYS=("stride_per_speed","stride_along","achieved_speed")
RES=R.res()
out={}
for setting,pp in P.items():
    st3=R.load_stamp(pp["s3"]); st2=R.load_stamp(pp["s2"])
    v3=R.episode_values(st3,R.replay(st3)); v2=R.episode_values(st2,R.replay(st2))
    ba=R.paired_bootstrap(st3,v3,st2,v2,keys=KEYS,return_raw=True,seed=R.BOOT_SEED)
    bb=R.paired_bootstrap(st3,v3,st2,v2,keys=KEYS,return_raw=True,seed=R.BOOT_SEED+1)
    lit=R.pooled_delta(st2,v2,st3,v3,st3,v3,st2,v2,keys=KEYS)
    rows={}
    for row in ("side","side/l","side/r"):
        m=dict(R._row_masks(st3))[row]
        r={}
        for k in KEYS:
            a3=R.aggregate({k:v3[k]},m,k); a2=R.aggregate({k:v2[k]},dict(R._row_masks(st2))[row],k)
            comb=(ba[row][k]["boot"]+bb[row][k]["boot"])/2
            pooled=RES.half_width(comb[np.isfinite(comb)],0.975)
            r[k]={"s3":a3,"s2":a2,"seed_delta":a3-a2,"paired_delta":ba[row][k]["delta"],
                  "single_pair_mde":ba[row][k]["mde"],"pooled_mde_proxy":pooled,
                  "band":max(abs(a3-a2),pooled),
                  "literal_pooled_arm_eq_base":{"mean_delta":lit[row][k]["mean_delta"],"pooled_mde":lit[row][k]["pooled_mde"]}}
        rows[row]=r
    excl={"s3":R._sps_excluded(st3,v3).get("side"),"s2":R._sps_excluded(st2,v2).get("side")}
    out[setting]={"stamps":pp,"rows":rows,"n_sps_excluded_side":excl}
    s=rows["side"]["stride_per_speed"]
    print(f"{setting}: side stride_per_speed s3 {s['s3']:.4f} s2 {s['s2']:.4f} seed d {s['seed_delta']:+.4f} "
          f"single MDE {s['single_pair_mde']:.4f} pooled MDE {s['pooled_mde_proxy']:.4f} band {s['band']:.4f} "
          f"(literal arm=base pooled: {s['literal_pooled_arm_eq_base']}) excl {excl}")
    for k in ("stride_along","achieved_speed"):
        s=rows["side"][k]; print(f"    {k}: s3 {s['s3']:.4f} s2 {s['s2']:.4f} d {s['seed_delta']:+.4f} pooled MDE {s['pooled_mde_proxy']:.4f}")
Path(sys.argv[1]).write_text(json.dumps(R._jsonable(out),indent=1))
