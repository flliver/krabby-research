import json, numpy as np
R=json.load(open("sym_results.json"))
print("stamp         spd | rep     | pattern: |R-L| |R-M_pi L| |R-M_0 L| noise | static: |mR-mL| |mR-M mL| noise | static share of L-R diff")
for c in R["cross"]:
    for rep in ("act","act_rod","jp_rod","foot_y"):
        x=c[rep]
        print(f"{c['stamp']:13s} {c['speed']} | {rep:7s} | {x['pat_agn']:.2f} {x['pat_equi']:.2f} {x['pat_equi0']:.2f} {x['pat_noise']:.2f} | {x['st_agn']:.3f} {x['st_equi']:.3f} {x['st_noise']:.3f} | {x['static_share_of_LR_diff']:.2f}")
