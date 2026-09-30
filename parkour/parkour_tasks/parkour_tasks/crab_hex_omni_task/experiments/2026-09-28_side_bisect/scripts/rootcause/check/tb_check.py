"""Re-derive the tfevents numbers that carry weight in the draft (CPU, read-only)."""
import glob, json, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
L="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk"
RUNS={"O":"X01","C":"X02","R":"X03","C_mo":"X04","C_band":"X05","C_rsi":"X06","C_s2":"X07","C_mo_s2":"X08","C_mo_s1":"X09",
      "LRfa":"X10","LR4":"X11","LR4_dwell":"X12","C_lr":"X14","C_lr_10":"X15","C_lr_rsi":"X16","C_lr_10_rsi":"X17","C_lr_rsi33":"X19"}
CLS={"O":"S","C":"F","R":"S","C_mo":"I","C_band":"S","C_rsi":"S","C_s2":"F","C_mo_s2":"F","C_mo_s1":"F","LRfa":"S","LR4":"S","LR4_dwell":"D","C_lr":"S","C_lr_10":"S","C_lr_rsi":"S","C_lr_10_rsi":"S","C_lr_rsi33":"S"}
TAGS=["Metrics/base_parkour/track_ratio_mean","Loss/symmetry","Train/mean_reward","Episode_Reward/track_lin_vel_xy_exp","Episode_Reward/reward_action_rate","Episode_Reward/reward_delta_torques","Episode_Reward/penalty_foot_idle_when_forward","Policy/mean_noise_std"]
out={}
for name,x in RUNS.items():
    d=sorted(glob.glob(f"{L}/*_{x}_s1_A15pB"))[-1]
    f=sorted(glob.glob(d+"/events.out.tfevents.*"))[0]
    ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
    S={}
    for t in TAGS:
        if t in ea.Tags()['scalars']:
            ev=ea.Scalars(t); S[t]=(np.array([e.step for e in ev]),np.array([e.value for e in ev]))
    out[name]=S
def sm(y,k=25):
    return np.convolve(y,np.ones(k)/k,mode='same')
print("run        cls  n_it  tr_final(2900-3000)  early(800-1000)  it95%  it>0.53  it>0.60  sym(0-100) sym(2900-end)  O-vs-Cmo n/a")
res={}
for name,S in out.items():
    st,tr=S["Metrics/base_parkour/track_ratio_mean"]; trs=sm(tr)
    fin=tr[(st>=2900)].mean() if st.max()>=2900 else tr[-50:].mean()
    early=tr[(st>=800)&(st<1000)].mean()
    i95=int(st[np.argmax(trs>=0.95*fin)]) if (trs>=0.95*fin).any() else -1
    i53=int(st[np.argmax(trs>=0.53)]) if (trs>=0.53).any() else -1
    i60=int(st[np.argmax(trs>=0.60)]) if (trs>=0.60).any() else -1
    if "Loss/symmetry" in S:
        ss,sy=S["Loss/symmetry"]; sy0=sy[ss<100].mean(); sy1=sy[ss>=2900].mean() if ss.max()>=2900 else sy[-50:].mean()
    else: sy0=sy1=np.nan
    res[name]=dict(fin=fin,early=early,i95=i95,i53=i53,i60=i60,sy0=sy0,sy1=sy1,n=int(st.max()))
    print(f"{name:10s} {CLS[name]:3s} {int(st.max()):5d}  {fin:.3f}               {early:.3f}          {i95:5d}  {i53:5d}   {i60:5d}   {sy0:.4f}     {sy1:.4f}")
# separation iteration: lowest fast vs highest slow (smoothed)
fast=[n for n in res if CLS[n]=="F"]; slow=[n for n in res if CLS[n] in "SD" and res[n]["n"]>=2999]
grid=np.arange(0,3000,10)
def interp(name):
    st,tr=out[name]["Metrics/base_parkour/track_ratio_mean"]; return np.interp(grid,st,sm(tr))
F=np.min([interp(n) for n in fast],0); Sx=np.max([interp(n) for n in slow],0)
sep=grid[np.argmax((F>Sx)&(np.cumsum((F>Sx)[::-1])[::-1]==np.arange(len(grid),0,-1)))] if (F>Sx).any() else -1
# first grid it from which min-fast > max-slow for the rest
ok=F>Sx; k=len(ok)
while k>0 and ok[k-1]: k-=1
print("first it after which min(FAST) > max(SLOW) permanently:", grid[k] if k<len(grid) else None)
# O vs C_mo over 100-350
o=interp("O"); c=interp("C_mo"); m=(grid>=100)&(grid<=350); print("O - C_mo over 100-350: mean %.3f  min %.3f max %.3f"%((o-c)[m].mean(),(o-c)[m].min(),(o-c)[m].max()))
# smoothing share of later reward gain (after it 500): d(action_rate+delta_torques)/d(mean_reward)
print("\nlater-gain share: run, d_total(500->end), d_smooth, share, d_track")
for name,S in out.items():
    st,mr=S["Train/mean_reward"]; sel=lambda a,b,s: a[(s>=b[0])&(s<b[1])].mean()
    def term(t,w):
        s,v=S[t]; return sel(v,w,s)
    w0=(450,550); w1=(2900,3001)
    if st.max()<2999: continue
    dtot=sel(mr,w1,st)-sel(mr,w0,st)
    dsm=(term("Episode_Reward/reward_action_rate",w1)-term("Episode_Reward/reward_action_rate",w0))+(term("Episode_Reward/reward_delta_torques",w1)-term("Episode_Reward/reward_delta_torques",w0))
    dtr=term("Episode_Reward/track_lin_vel_xy_exp",w1)-term("Episode_Reward/track_lin_vel_xy_exp",w0)
    # note Episode_Reward is per-episode sum; mean_reward is per-step mean * ... compare with the sum of all terms instead
    terms=[t for t in S if t.startswith("Episode_Reward/")]
    dall=sum(term(t,w1)-term(t,w0) for t in terms)
    print(f"{name:10s} {CLS[name]} dtot_ep {dall:7.2f} d_smooth {dsm:7.2f} share {dsm/dall if dall else np.nan:5.2f}  d_track {dtr:6.2f}  (only tracked terms: track/ar/dt/fi)")
json.dump({k:{kk:(float(vv) if not isinstance(vv,int) else vv) for kk,vv in v.items()} for k,v in res.items()},open("/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/4b0248a2-13ef-464e-81a9-16e93543e283/scratchpad/check/tb_check.json","w"),indent=1)
