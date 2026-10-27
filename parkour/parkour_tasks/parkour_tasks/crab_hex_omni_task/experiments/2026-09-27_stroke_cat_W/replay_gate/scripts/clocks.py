import json, sys, importlib.util
from types import SimpleNamespace
from pathlib import Path
T="/home/nickmagus/krabby/krabby-research/.claude/worktrees/stroke-posture/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/replay_stroke_posture.py"
spec=importlib.util.spec_from_file_location("rsp",T); R=importlib.util.module_from_spec(spec); sys.modules["rsp"]=R; spec.loader.exec_module(R)
cfg=json.load(open("/home/nickmagus/krabby/krabby-research/.claude/worktrees/stroke-posture/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-26_stroke_posture/replay_gate_config_p15.json"))
L="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/"
extra={"E3":L+"2026-09-23_clock_deprecation/step0_E3_X08-t3_visible/suite__omni/seed001/2026-09-23_21-38-11",
"Bf_s3":L+"2026-09-26_stroke_formation/seed3_X01_t3_A15pB/suite__omni/seed001/2026-09-26_22-56-51",
"Bf_s2":L+"2026-09-26_stroke_formation/seed2_X02_t3_A15pB/suite__omni/seed001/2026-09-27_04-08-09",
"B_s3":L+"2026-09-26_stroke_posture/seed3_X01_c6y_A15pB/suite__omni/seed001/2026-09-26_08-39-30",
"B_s2":L+"2026-09-26_stroke_posture/seed2_X02_c6y_A15pB/suite__omni/seed001/2026-09-26_10-25-08"}
allr={}
for k in ("healthy","sideways","posture","degenerate"):
    for n,p in cfg[k].items(): allr[f"{k}:{n}"]=p
for n,p in extra.items(): allr[f"new:{n}"]=p
import glob
# other step0 stamps in clock_deprecation (possible clock-held degenerates)
for d in sorted(glob.glob(L+"2026-09-23_clock_deprecation/step0_*/suite__omni/seed001/*")):
    allr.setdefault("step0:"+d.split("/step0_")[1].split("/")[0], d)
for n,p in allr.items():
    meta=json.load(open(Path(p)/"run_meta.json"))
    r=R.stamp_clock(SimpleNamespace(meta=meta))
    print(f"{n:32s} clock={r[0] if r else None}  ck={meta.get('checkpoint')}")
econ="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_09-10-29_seed3_X08_t3_A15pB/params/env.yaml"
rw=R.RMT._yaml_load_tolerant(Path(econ)).get("rewards")
print("t3 econ clock", rw["reward_clock_schedule"]["weight"], rw["reward_clock_swing_apex"]["weight"])
