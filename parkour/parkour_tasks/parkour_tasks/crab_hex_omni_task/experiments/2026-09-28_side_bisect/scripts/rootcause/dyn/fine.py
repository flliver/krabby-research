import sys; sys.path.insert(0,'.')
from common import *
G={'F':FAST,'S1':['X01','X05','X06'],'S2':['X10','X11','X14','X16','X19']}
T=['Metrics/base_parkour/track_ratio_mean','Episode_Reward/track_lin_vel_xy_exp','Episode_Reward/penalty_tracking_error_l1','Episode_Reward/reward_forward_progress_along_command','Episode_Reward/reward_action_rate','Episode_Reward/penalty_tibia_deviation_in_stance','Episode_Reward/penalty_excess_feet_contact_forward','Episode_Reward/reward_clock_schedule','Episode_Reward/reward_clock_swing_apex','Episode_Reward/penalty_foot_idle_when_forward','Episode_Reward/reward_delta_torques','Metrics/base_parkour/crab_failure_flat','Train/mean_episode_length','Policy/mean_noise_std','Loss/symmetry','Loss/value_function','Loss/entropy','Metrics/base_velocity/error_vel_xy']
wins=[(i,i+50) for i in range(0,1000,50)]+[(1000,1500),(2500,3000)]
for t in T:
    print('==',t)
    print('   '+' '.join('%6d'%a for a,b in wins))
    for g,xs in G.items():
        row=[np.nanmean([win(x,t,a,b) for x in xs]) for a,b in wins]
        print('%-3s'%g+' '.join('%6.3f'%r if abs(r)<100 else '%6.0f'%r for r in row))
