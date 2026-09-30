from symlib import *
np.set_printoptions(precision=3, suppress=True)
for name in ["canon_C","canon_C_mo_s1"]:
    for spd in ("022","025"):
        L=load(name,f"lat_l_{spd}"); R=load(name,f"lat_r_{spd}")
        for rep in ("act","jp","foot"):
            AL,r2l,_=pattern(L[rep],L["ph"]); AR,r2r,_=pattern(R[rep],R["ph"])
            h1,h2=split_idx(len(L[rep])); A1,_,_=pattern(L[rep],L["ph"],h1); A2,_,_=pattern(L[rep],L["ph"],h2)
            n=np.sqrt(rms(AL)*rms(AR))
            ds=[rms(AR-mir(AL,rep,k))/n for k in range(NB)]
            print(name,spd,rep,f"R2 {r2l:.2f}/{r2r:.2f} |R-M_pi L| {rms(AR-mir(AL,rep,12))/n:.3f} |R-M_0 L| {rms(AR-mir(AL,rep,0))/n:.3f} min {min(ds):.3f}@{np.argmin(ds)*15} |R-L| {rms(AR-AL)/n:.3f} halfnoise {rms(A1-A2)/rms(AL):.3f}")
