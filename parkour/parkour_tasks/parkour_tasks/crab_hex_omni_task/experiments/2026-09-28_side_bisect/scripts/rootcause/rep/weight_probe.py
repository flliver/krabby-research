"""CPU weight probe over checkpoints: first-layer input-column norms of the actor backbone (read-only).

Actor backbone input (136) = obs[0:75] (true prop dims 0..74: head 0-16, joint_pos 17-40, joint_vel 41-64,
last_action[0:10] 65-74) | scan-encoder latent (32) | estimated 'priv_explicit' slice (9) | priv/hist latent (20).
(agent.yaml num_prop=75 vs the live 89-dim step: crab_hex_rl_cfg.py:12-19.)
"""
import glob, json, sys
import numpy as np, torch
R = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
ARMS = [("C", "2026-09-29_01-52-56", "F", "+y fa scalar"), ("C_s2", "2026-09-29_13-23-08", "F", "+y fa scalar"),
        ("C_mo_s2", "2026-09-29_14-36-09", "F", "+y fa scalar"), ("C_mo_s1", "2026-09-29_18-09-45", "F", "+y fa scalar"),
        ("C_mo", "2026-09-29_09-39-22", "I", "+y fa scalar"),
        ("O", "2026-09-29_00-40-18", "S", "+y 4img vis"), ("R", "2026-09-29_08-28-30", "S", "+y 4img vis"),
        ("C_band", "2026-09-29_10-52-23", "S", "+y 4img vis"),
        ("LRfa", "2026-09-29_19-18-51", "S", "±y fa vis"), ("LR4", "2026-09-29_20-33-42", "S", "±y 4img vis"),
        ("C_lr", "2026-09-30_00-00-38", "S", "±y fa vis"), ("C_lr_10", "2026-09-30_01-15-33", "S", "±y fa vis"),
        ("C_lr_rsi", "2026-09-30_09-05-43", "S", "±y fa vis")]
ITS = [0, 100, 300, 600, 900, 1200, 1500, 2000, 2500, 2999]
G = {"wz_cmd5": [5], "facing6_7": [6, 7], "cmd_vx8": [8], "cmd_vy9": [9], "speed10": [10], "linv13_14": [13, 14],
     "clock15_16": [15, 16], "angvel0_2": [0, 1, 2], "rp3_4": [3, 4], "envidx11_12": [11, 12],
     "jpos": list(range(17, 41)), "jvel": list(range(41, 65)), "lastact": list(range(65, 75)),
     "scanlat": list(range(75, 107)), "est": list(range(107, 116)), "latent": list(range(116, 136))}
def load(d, it):
    p = f"{d}/model_{it}.pt"
    try:
        return torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
    except FileNotFoundError:
        return None
out = {}
for lab, pre, cls, desc in ARMS:
    d = glob.glob(R + pre + "*")[0]
    rows = {}
    for it in ITS:
        sd = load(d, it)
        if sd is None: continue
        W1 = sd["actor.actor_backbone.0.weight"].numpy()           # 512 x 136
        W2 = sd["actor.actor_backbone.2.weight"].numpy(); W3 = sd["actor.actor_backbone.4.weight"].numpy()
        W4 = sd["actor.actor_backbone.6.weight"].numpy(); b4 = sd["actor.actor_backbone.6.bias"].numpy()
        cn = np.linalg.norm(W1, axis=0)                              # per-input column norm
        ref = np.median(cn[:75])
        # absolute path-norm proxy to the rod outputs (6..17) and the cam outputs (0..5)
        Pabs = np.abs(W4) @ np.abs(W3) @ np.abs(W2) @ np.abs(W1)    # 18 x 136
        rows[it] = dict(ref=float(ref), std=float(sd["std"].mean()),
                        col={g: float(np.sqrt(np.mean(cn[ix]**2))) for g, ix in G.items()},
                        path_rod={g: float(Pabs[6:, ix].mean()) for g, ix in G.items()},
                        W4n=float(np.linalg.norm(W4)), b4=float(np.abs(b4).mean()),
                        W1n=float(np.linalg.norm(W1)), W2n=float(np.linalg.norm(W2)), W3n=float(np.linalg.norm(W3)))
    out[lab] = dict(cls=cls, desc=desc, rows=rows)
    print("done", lab, sorted(rows))
json.dump(out, open("weight_probe.json", "w"), indent=1)
