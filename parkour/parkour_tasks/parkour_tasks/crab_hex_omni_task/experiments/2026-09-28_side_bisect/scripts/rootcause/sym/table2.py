import json, numpy as np
R=json.load(open("sym_results.json"))
print("stamp          hand spd cls  feet  | act_rod amp oddrms evenrms  harm(sym_oddk anti_oddk sym_evk anti_evk) pairodd f/m/r lead_share | act_cam amp | foot_y amp oddrms p2p lead/trail | foot_yz harm sym_oddk anti_oddk | jp_rod amp odd")
for r in R["rows"]:
    a=r["act_rod"]; h=a["harm"]; fy=r["foot_y"]; fh=r["foot_yz"]["harm"]; j=r["jp_rod"]
    po=a["pair_odd"]
    print(f"{r['stamp']:14s} {r['hand']}   {r['label'][-3:]} {r['cls']:4s} {r['feet_ss']:.3f} | {a['amp']:.2f} {a['amp']*np.sqrt(a['odd_pi']):.2f} {a['amp']*np.sqrt(1-a['odd_pi']):.2f}  ({h['sym_oddk']:.2f} {h['anti_oddk']:.2f} {h['sym_evenk']:.2f} {h['anti_evenk']:.2f}) {po['front']:.2f}/{po['mid']:.2f}/{po['rear']:.2f} {a['lead_share']:.2f} | {r['act_cam']['amp']:.2f} | {fy['amp']*100:.1f}cm {fy['amp']*np.sqrt(fy['odd_pi'])*100:.1f} {fy['p2p_lead']*100:.1f}/{fy['p2p_trail']*100:.1f} | {fh['sym_oddk']:.2f} {fh['anti_oddk']:.2f} | {j['amp']:.3f} {j['odd_pi']:.2f}")
