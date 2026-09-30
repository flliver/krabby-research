exec(open("mytb.py").read().split("tags = sorted")[0])
tags = sorted(data["O"].keys())
import numpy as np
print("std trajectory (Policy/mean_noise_std) at its:")
its=[(0,50),(200,300),(300,400),(500,600),(900,1000),(1400,1500),(1900,2000),(2400,2500),(2900,3000)]
print("%-9s c "%"arm"+" ".join("%9s"%f"{a}-{b}" for a,b in its))
for a in RUNS: print("%-9s %s "%(a,CLS[a])+" ".join("%9.3f"%w(data[a],"Policy/mean_noise_std",lo,hi) for lo,hi in its))
print("\nper-term Episode_Reward means over its 2500-3000: O vs C_mo seeds vs C")
terms=[t for t in tags if t.startswith("Episode_Reward/")]
print("%-52s %8s %8s %8s %8s %8s"%("term","O","C_mo","C_mo_s2","C_mo_s1","C"))
tot={k:0 for k in ["O","C_mo","C_mo_s2","C_mo_s1","C"]}
for t in terms:
    vals=[w(data[a],t,2500,3000) for a in ["O","C_mo","C_mo_s2","C_mo_s1","C"]]
    for k,v in zip(tot,vals): tot[k]+=v
    if max(abs(v) for v in vals)>0.05: print("%-52s "%t.split("/")[1]+" ".join("%8.3f"%v for v in vals))
print("%-52s "%"SUM of terms"+" ".join("%8.3f"%tot[k] for k in tot))
print("%-52s "%"Train/mean_reward"+" ".join("%8.2f"%w(data[a],"Train/mean_reward",2500,3000) for a in tot))
print("%-52s "%"Train/mean_episode_length"+" ".join("%8.1f"%w(data[a],"Train/mean_episode_length",2500,3000) for a in tot))
print("\noptimizer / termination curves by class (2500-3000 and 300-600):")
for t in ["Loss/entropy","Loss/learning_rate","Loss/surrogate","Loss/priv_reg","Loss/priv_reg_coef","Loss/value_function","Loss/symmetry","Episode_Termination/total_terminates","Train/mean_episode_length"]:
    for lo,hi in [(300,600),(2500,3000)]:
        F=[w(data[a],t,lo,hi) for a in RUNS if CLS[a]=="F"]; S=[w(data[a],t,lo,hi) for a in RUNS if CLS[a] in "SD"]
        print("%-40s %s FAST [%.4g, %.4g]  SLOW [%.4g, %.4g]"%(t,f"{lo}-{hi}",min(F),max(F),min(S),max(S)))
