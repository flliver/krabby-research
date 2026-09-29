import torch
for p in ["/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt","/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9998.pt"]:
    sd=torch.load(p,map_location="cpu",weights_only=False)["model_state_dict"]
    for k,v in sd.items():
        if "std" in k.lower() and v.numel()<50: print(p[-40:],k,v)
