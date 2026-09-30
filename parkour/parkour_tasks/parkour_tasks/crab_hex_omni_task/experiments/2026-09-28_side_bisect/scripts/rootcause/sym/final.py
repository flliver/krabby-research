"""Final tables + separation summary + sym_summary.json (reads sym_results.json and the cache)."""
import json
from symlib import *
R = json.load(open("sym_results.json"))
rows = R["rows"]
summ = {"rows": [], "cross": [], "separation": {}}
def half_odd_rms(name, lab, phase):
    d = load(name, lab); X = sub(d, "act_rod")
    ph = d["ph"] if phase == "clock" else intrinsic_phase(d)[0]
    h1, h2 = split_idx(len(X)); out = []
    for h in (h1, h2):
        A = pattern(X, ph, h)[0]; O = (A - mir(A, "act_rod", NB // 2)) / 2; out.append(rms(O))
    return out
print("| stamp | grp | hand | spd | class | n | feet_ss | sweep | lean | phase | rod lockR2 | rod odd frac pi [h1/h2] | rod odd frac 0 | rod ODD rms [h1/h2] | rod EVEN rms | sym@odd-k | jp_rod odd frac | static odd rod (±half-noise) | cam-spin static odd | lean_y static | static share of odd | J lead/trail (1e-3 m2/s) |")
print("|" + "---|" * 22)
for r in rows:
    a = r["act_rod"]; st = r["static"]
    h = half_odd_rms(r["stamp"], r["label"], r["phase"])
    odd_rms = a["amp"] * np.sqrt(a["odd_pi"]); even_rms = a["amp"] * np.sqrt(1 - a["odd_pi"])
    rec = dict(stamp=r["stamp"], group=r["group"], hand=r["hand"], speed=r["label"][-3:], cls=r["cls"], n=r["n"],
               feet_ss=r["feet_ss"], sweep=r["sweep"], lean_abs=r["lean_abs"], phase=r["phase"], rod_lockR2=a["lockR2"],
               rod_odd_frac_pi=a["odd_pi"], rod_odd_frac_pi_halves=[a["odd_pi_h1"], a["odd_pi_h2"]], rod_odd_frac_0=a["odd_0"],
               rod_odd_frac_pi_otherphase=a.get("odd_pi_otherphase"),
               rod_odd_rms=odd_rms, rod_odd_rms_halves=h, rod_even_rms=even_rms, rod_harm=a["harm"], rod_pair_odd=a["pair_odd"],
               rod_lead_share=a["lead_share"], act_full_odd_frac_pi=r["act"]["odd_pi"], act_cam_odd_frac_pi=r["act_cam"]["odd_pi"],
               jp_rod_odd_frac_pi=r["jp_rod"]["odd_pi"], jp_rod_amp=r["jp_rod"]["amp"], foot_y_odd_frac_pi=r["foot_y"]["odd_pi"],
               foot_y_odd_rms_m=r["foot_y"]["amp"] * np.sqrt(r["foot_y"]["odd_pi"]), foot_y_p2p_lead_trail_m=[r["foot_y"]["p2p_lead"], r["foot_y"]["p2p_trail"]],
               static_odd_rod=st["act_rod"]["odd_norm"], static_odd_rod_noise=st["act_rod"]["odd_noise"],
               static_odd_camspin=st["camv"]["odd_norm"], static_lean_y=st["body"]["odd"][1], static_roll=st["body"]["odd"][2],
               static_share_of_odd_rod=a["static_share_of_odd"], J_lead=r["prop"]["J_lead"], J_trail=r["prop"]["J_trail"],
               f0_intrinsic_hz=r["f0_intrinsic_hz"], clock_hz=r.get("clock_hz"))
    summ["rows"].append(rec)
    print(f"| {r['stamp']} | {r['group']} | {r['hand']} | {r['label'][-3:]} | {r['cls']} | {r['n']} | {r['feet_ss']:.3f} | {r['sweep']:.3f} | {r['lean_abs']:+.2f} | {r['phase'][:5]} | {a['lockR2']:.2f} | {a['odd_pi']:.2f} [{a['odd_pi_h1']:.2f}/{a['odd_pi_h2']:.2f}] | {a['odd_0']:.2f} | **{odd_rms:.2f}** [{h[0]:.2f}/{h[1]:.2f}] | {even_rms:.2f} | {a['harm']['sym_oddk']:.2f} | {r['jp_rod']['odd_pi']:.2f} | {st['act_rod']['odd_norm']:.2f} ±{st['act_rod']['odd_noise']:.2f} | {st['camv']['odd_norm']:.2f} | {st['body']['odd'][1]:+.3f} | {a['static_share_of_odd']:.2f} | {r['prop']['J_lead']*1e3:+.1f}/{r['prop']['J_trail']*1e3:+.1f} |")

# cross-hand
def cosine(a, b): return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
print("\n| stamp | spd | rod pattern abs(R-L) | abs(R-M_pi L) | noise | cos(odd_R,odd_L) | cos(even_R,even_L) | static rod abs(mR-mL) | abs(mR-M mL) | noise | cos(static-odd R,L) | cos(cam-spin odd R,L) | static share of L-R diff (rod) |")
print("|" + "---|" * 13)
for c in R["cross"]:
    L = load(c["stamp"], f"lat_l_{c['speed']}"); Rr = load(c["stamp"], f"lat_r_{c['speed']}")
    AL = pattern(sub(L, "act_rod"), L["ph"])[0]; AR = pattern(sub(Rr, "act_rod"), Rr["ph"])[0]
    OL = (AL - mir(AL, "act_rod", 12)) / 2; OR = (AR - mir(AR, "act_rod", 12)) / 2
    mL = sub(L, "act_rod").mean(axis=(0, 1)); mR = sub(Rr, "act_rod").mean(axis=(0, 1))
    oL = (mL - mir_static(mL, "act_rod")) / 2; oR = (mR - mir_static(mR, "act_rod")) / 2
    cL = L["camv"].astype(float).mean(axis=(0, 1)); cR = Rr["camv"].astype(float).mean(axis=(0, 1))
    coL = (cL - mir_static(cL, "camv")) / 2; coR = (cR - mir_static(cR, "camv")) / 2
    x = c["act_rod"]
    rec = dict(stamp=c["stamp"], speed=c["speed"], group=c["group"], pat_RminusL=x["pat_agn"], pat_RminusMpiL=x["pat_equi"], pat_noise=x["pat_noise"],
               cos_odd=cosine(OR, OL), cos_even=cosine(AR - OR, AL - OL), st_RminusL=x["st_agn"], st_RminusML=x["st_equi"], st_noise=x["st_noise"],
               cos_static_odd=cosine(oR, oL), cos_camspin_odd=cosine(coR, coL), static_share_LR=x["static_share_of_LR_diff"],
               full_act=c["act"], jp_rod=c["jp_rod"], foot_y=c["foot_y"])
    summ["cross"].append(rec)
    print(f"| {c['stamp']} | {c['speed']} | {x['pat_agn']:.2f} | {x['pat_equi']:.2f} | {x['pat_noise']:.2f} | {rec['cos_odd']:+.2f} | {rec['cos_even']:+.2f} | {x['st_agn']:.2f} | {x['st_equi']:.2f} | {x['st_noise']:.2f} | {rec['cos_static_odd']:+.2f} | {rec['cos_camspin_odd']:+.2f} | {x['static_share_of_LR_diff']:.2f} |")

# separation summary over rows (canon rows = the heads; lat_r canon rows included as FAST by construction)
def sep(key, higher_is_fast=True, excl=()):
    F = [r[key] for r in summ["rows"] if r["cls"] == "FAST" and r["stamp"] not in excl]
    I = [r[key] for r in summ["rows"] if r["cls"] == "INT" and r["stamp"] not in excl]
    S = [r[key] for r in summ["rows"] if r["cls"] in ("SLOW", "DEAD") and r["stamp"] not in excl]
    if higher_is_fast:
        gap = min(F) - max(S)
    else:
        gap = min(S) - max(F)
    return dict(fast=[float(min(F)), float(max(F))], int=[float(min(I)), float(max(I))] if I else None, slow=[float(min(S)), float(max(S))], gap=float(gap), separates=bool(gap > 0))
for key, hi in (("sweep", True), ("rod_odd_rms", True), ("rod_odd_frac_pi", True), ("foot_y_odd_rms_m", True),
                ("rod_even_rms", True), ("static_odd_rod", False), ("static_share_of_odd_rod", False), ("static_odd_camspin", False),
                ("lean_abs", False), ("jp_rod_odd_frac_pi", True)):
    s = sep(key, hi); summ["separation"][key] = s
    s2 = sep(key, hi, excl=("R",)); summ["separation"][key + "__exclR"] = s2
    print(f"{key:26s} FAST {s['fast'][0]:.3f}-{s['fast'][1]:.3f}  INT {s['int'][0]:.3f}-{s['int'][1]:.3f}  SLOW/DEAD {s['slow'][0]:.3f}-{s['slow'][1]:.3f}  gap {s['gap']:+.3f} {'SEPARATES' if s['separates'] else 'overlap'} | excl R gap {s2['gap']:+.3f}")
json.dump(summ, open("sym_summary.json", "w"), indent=1)
