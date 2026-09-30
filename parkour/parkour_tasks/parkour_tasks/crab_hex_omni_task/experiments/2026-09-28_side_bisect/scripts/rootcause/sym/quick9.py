from symlib import *
names=["C","C_s2","C_mo_s2","C_mo_s1","C_mo","O","C_band","C_rsi","R","LRfa","LR4","C_lr","C_lr_10","C_lr_rsi","C_lr_10_rsi","LR4_dwell"]
E={};O={}
for n in names:
    d=load(n,"lat_l_022"); A=pattern(sub(d,"act_rod"),d["ph"])[0]
    o=(A-mir(A,"act_rod",12))/2; E[n]=A-o; O[n]=o
def cs(a,b): return np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))
# allow best circular shift (phase origin could differ by run): max over shifts of cosine
def best(a,b):
    return max(cs(a,np.roll(b,k,0)) for k in range(NB))
print("even-part cosine (best shift) of rod-command pattern, lat_l_022")
print(" "*12+" ".join(f"{n[:7]:>7s}" for n in names))
for a in names:
    print(f"{a:12s}"+" ".join(f"{best(E[a],E[b]):7.2f}" for b in names))
print("odd-part cosine (best shift)")
for a in names:
    print(f"{a:12s}"+" ".join(f"{best(O[a],O[b]):7.2f}" for b in names))
