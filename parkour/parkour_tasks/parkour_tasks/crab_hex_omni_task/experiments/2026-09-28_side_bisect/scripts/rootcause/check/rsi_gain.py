import glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
L="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk"
def load(x):
    d=sorted(glob.glob(f"{L}/*_{x}_s1_A15pB"))[-1]; f=sorted(glob.glob(d+"/events.out.tfevents.*"))[0]
    ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload(); return ea
def win(ea,t,a,b):
    ev=ea.Scalars(t); s=np.array([e.step for e in ev]); v=np.array([e.value for e in ev]); m=(s>=a)&(s<b); return v[m].mean() if m.any() else np.nan
print("== RSI claims")
for x,nm in (("X16","C_lr_rsi"),("X17","C_lr_10_rsi"),("X19","C_lr_rsi33"),("X04","C_mo"),("X01","O")):
    ea=load(x); tags=ea.Tags()['scalars']
    for t in ["Metrics/base_parkour/reach_edge_frac","Metrics/base_parkour/reach_edge_frac_rsi","Metrics/base_parkour/reach_obst_frac","Metrics/base_parkour/reach_obst_frac_rsi","Metrics/base_parkour/crab_failure_obst","Metrics/base_parkour/crab_failure_obst_rsi","Metrics/base_parkour/crab_failure_flat","Metrics/base_parkour/rsi_frac_actual"]:
        if t in tags:
            print(f"{nm:12s} {t.split('/')[-1]:24s} 0-300 {win(ea,t,0,300):.3f}  1000-1500 {win(ea,t,1000,1500):.3f}  2500-3000 {win(ea,t,2500,3000):.3f}  all {win(ea,t,0,3001):.3f}")
print("\n== smoothing share with gain.py's 12 terms, [450,550)->[2500,3000)")
T=['track_lin_vel_xy_exp','penalty_tracking_error_l1','reward_forward_progress_along_command','track_ang_vel_z_exp','reward_clock_schedule','reward_clock_swing_apex','penalty_foot_idle_when_forward','reward_action_rate','reward_delta_torques','penalty_tibia_deviation_in_stance','penalty_excess_feet_contact_forward','penalty_lin_vel_y']
for x,nm,c in (("X01","O","S"),("X05","C_band","S"),("X06","C_rsi","S"),("X10","LRfa","S"),("X11","LR4","S"),("X14","C_lr","S"),("X16","C_lr_rsi","S"),("X19","C_lr_rsi33","S"),("X02","C","F"),("X07","C_s2","F"),("X08","C_mo_s2","F"),("X09","C_mo_s1","F"),("X04","C_mo","I")):
    ea=load(x); d={t:win(ea,'Episode_Reward/'+t,2500,3000)-win(ea,'Episode_Reward/'+t,450,550) for t in T}
    tot=sum(d.values()); sm=d['reward_action_rate']+d['reward_delta_torques']; tr=d['track_lin_vel_xy_exp']+d['penalty_tracking_error_l1']+d['reward_forward_progress_along_command']+d['penalty_lin_vel_y']
    print(f"{nm:10s} {c} total12 {tot:+.3f} smoothing {sm:+.3f} share {sm/tot if tot else np.nan:.2f} tracking4 {tr:+.3f} foot_idle {d['penalty_foot_idle_when_forward']:+.3f}")
