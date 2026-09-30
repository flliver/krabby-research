import importlib.util, sys
from pathlib import Path
R=Path("/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments")
spec=importlib.util.spec_from_file_location("rc", R/"tools/run_candidates.py")
rc=importlib.util.module_from_spec(spec); sys.modules["rc"]=rc; spec.loader.exec_module(rc)
cd=rc.load_candidates(R/"2026-09-23_clock_deprecation/candidates.yaml")
rec=rc.load_candidates(R/"2026-09-19_facing_tracking_ladder/candidates.yaml")
t={s["stage"]:s for s in cd["chains"]["tail"]["stages"]}
r={s["stage"]:s for s in rec["chains"]["curriculum_turnfirst"]["stages"]}
for k in t:
    a,b=t[k],r.get(k)
    print(k, a["iters"], a["resume_from"], a["eval"], "overlay equal:", b is not None and {x:str(y) for x,y in a["overlay"].items()}=={x:str(y) for x,y in b["overlay"].items()})
