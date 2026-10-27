import json, numpy as np
R=json.load(open("sym_results.json"))
print("stamp          hand spd cls  n   feet  swp  lean ph  | act_rod: R2  odd_pi (h1/h2) corr  [other] odd0 | act_cam oddpi share | jp_rod oddpi | foot_y oddpi foot_z oddpi | static odd: rod  cam(camv)  lean_y | share_static(rod) | J lead/trail")
for r in R["rows"]:
    a=r["act_rod"]; c=r["act_cam"]; j=r["jp_rod"]; fy=r["foot_y"]; fz=r["foot_z"]; st=r["static"]
    print(f"{r['stamp']:14s} {r['hand']}   {r['label'][-3:]} {r['cls']:4s} {r['n']:3d} {r['feet_ss']:.3f} {r['sweep']:.3f} {r['lean_abs']:+.2f} {r['phase'][:3]} | {a['lockR2']:.2f} {a['odd_pi']:.2f} ({a['odd_pi_h1']:.2f}/{a['odd_pi_h2']:.2f}) {a['odd_pi_corr']:.2f} [{a.get('odd_pi_otherphase',float('nan')):.2f}] {a['odd_0']:.2f} | {c['odd_pi']:.2f} {r['act']['energy_share']['cam']:.2f} | {j['odd_pi']:.2f} | {fy['odd_pi']:.2f} {fz['odd_pi']:.2f} | {st['act_rod']['odd_norm']:.2f}±{st['act_rod']['odd_noise']:.2f} {st['camv']['odd_norm']:.2f} {st['body']['odd'][1]:+.3f} | {a['static_share_of_odd']:.2f} | {r['prop']['J_lead']*1e3:+.1f}/{r['prop']['J_trail']*1e3:+.1f}")
