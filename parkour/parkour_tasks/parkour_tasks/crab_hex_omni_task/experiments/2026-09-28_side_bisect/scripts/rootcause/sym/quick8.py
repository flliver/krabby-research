from symlib import *
import json
reg=json.load(open(CACHE/"registry.json"))
def cosine(a,b): return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
print("stamp        spd | act_rod: cos(O_R,O_L) cos(E_R,E_L) |O_L| |O_R| |E_L| |E_R| | static odd rod cos(oR,oL) |oL| |oR| | camv static odd cos |oL| |oR| | act_cam O cos")
for name,group,path in reg["stamps"]:
    for spd in ("022","025"):
        L=load(name,f"lat_l_{spd}"); R=load(name,f"lat_r_{spd}")
        if L is None or R is None: continue
        out=[]
        AL=pattern(sub(L,"act_rod"),L["ph"])[0]; AR=pattern(sub(R,"act_rod"),R["ph"])[0]
        OL=(AL-mir(AL,"act_rod",12))/2; OR=(AR-mir(AR,"act_rod",12))/2
        EL=(AL+mir(AL,"act_rod",12))/2; ER=(AR+mir(AR,"act_rod",12))/2
        mL=sub(L,"act_rod").mean(axis=(0,1)); mR=sub(R,"act_rod").mean(axis=(0,1))
        oL=(mL-mir_static(mL,"act_rod"))/2; oR=(mR-mir_static(mR,"act_rod"))/2
        cL=L["camv"].astype(float).mean(axis=(0,1)); cR=R["camv"].astype(float).mean(axis=(0,1))
        coL=(cL-mir_static(cL,"camv"))/2; coR=(cR-mir_static(cR,"camv"))/2
        CL=pattern(sub(L,"act_cam"),L["ph"])[0]; CR=pattern(sub(R,"act_cam"),R["ph"])[0]
        OCL=(CL-mir(CL,"act_cam",12))/2; OCR=(CR-mir(CR,"act_cam",12))/2
        print(f"{name:12s} {spd} | {cosine(OR,OL):+.2f} {cosine(ER,EL):+.2f} {rms(OL):.2f} {rms(OR):.2f} {rms(EL):.2f} {rms(ER):.2f} | {cosine(oR,oL):+.2f} {np.linalg.norm(oL):.2f} {np.linalg.norm(oR):.2f} | {cosine(coR,coL):+.2f} {np.linalg.norm(coL):.2f} {np.linalg.norm(coR):.2f} | {cosine(OCR,OCL):+.2f}")
