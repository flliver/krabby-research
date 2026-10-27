import json, numpy as np
D = "/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/impl-audit/"
tb = {r["run"]: r for r in json.load(open(D + "tb_census.json"))}
cfg = {c["exp"] + "/" + c["run"]: c for c in json.load(open(D + "cfg_census.json"))}
keys = ["crab_hex_flat_walk/2026-09-07_04-38-50", "crab_hex_side_teacher/2026-09-12_03-04-37", "crab_hex_omni_flat_walk/2026-09-25_14-53-27_seed3_X03_c6x_A15pB",
        "crab_hex_omni_flat_walk/2026-09-30_12-27-45_seed3_X19_s1_A15pB", "crab_hex_student/2026-09-08_05-54-01"]
m1 = [k for k, c in cfg.items() if c.get("stdmax") not in (None, "None")]
for k in keys + m1:
    r = tb.get(k, {}); c = cfg.get(k, {})
    print(k, {x: (round(r[x], 6) if isinstance(r.get(x), float) else r.get(x)) for x in ["n","it0","itN","lr_first","lr_med","lr_floor_frac","lr_le_3p4e5","lr_max","std_first","std_last","std_first_it_2","prc_first","prc_last","pr_last","sym_med","ent_last","vf_med","fps_med","eplen_last","rew_last"] if x in r}, "stdmax", c.get("stdmax"), "stdreset", c.get("stdreset"), "ent", c.get("ent"), "envs", c.get("num_envs"))
