import sys, numpy as np
sys.argv = [sys.argv[0]]
exec(open("probe_hand.py").read().split("res = {}")[0])
arms = ["C","C_s2","C_mo_s2","C_mo_s1","C_mo","O","R","C_band","C_rsi","LR4","LRfa","C_lr","C_lr_10","LR4_dwell"]
CM = {}
for arm in arms:
    meta, W = windows(stamp(arm))
    an = meta["action_joint_names"]; Pa, Sa = mymap(an); Pr = Pa[6:]-6; Sr = Sa[6:]
    for lab in ("lat_l_022","lat_l_025","lat_r_022","lat_r_025"):
        if lab not in W: continue
        eps = W[lab]
        mats = [corrmat(np.clip(e["a"],-1,1)[:,6:]) for e in eps]
        C = np.mean(mats,0); h1 = np.mean(mats[0::2],0); h2 = np.mean(mats[1::2],0)
        Cm = (Sr[:,None]*Sr[None,:]) * C[np.ix_(Pr,Pr)]
        A = np.clip(np.stack([e["a"] for e in eps]),-1,1)
        am = A.mean(axis=(0,1)); amm = Sa*am[Pa]
        cam = am[:6]
        # per-channel oscillation amplitude (std over time, mean over eps) rods
        osc = A[:,:,6:].std(axis=1).mean(0)
        CM[(arm,lab)] = C
        print(f"{arm:9s} {lab:9s} self-hand coord {np.linalg.norm(C-Cm):.2f} (noise {np.linalg.norm(h1-h2):.2f}) | mean-self-hand {np.linalg.norm(am-amm):.2f} | cam mean {np.round(cam,2)} | rod osc med {np.median(osc):.2f}")
ref = CM[("C","lat_l_022")]
print("coord distance to C lat_l_022:")
for k,v in CM.items():
    print(f"  {k[0]:9s} {k[1]:9s} {np.linalg.norm(v-ref):.2f}")
