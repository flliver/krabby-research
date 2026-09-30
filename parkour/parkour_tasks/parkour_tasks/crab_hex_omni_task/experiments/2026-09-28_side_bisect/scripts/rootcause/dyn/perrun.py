import sys; sys.path.insert(0,'.')
from common import *
T=['track_lin_vel_xy_exp','penalty_tracking_error_l1','reward_forward_progress_along_command','penalty_lin_vel_y','track_ang_vel_z_exp','reward_clock_schedule','reward_clock_swing_apex','penalty_foot_idle_when_forward','reward_action_rate','reward_delta_torques','penalty_tibia_deviation_in_stance','penalty_excess_feet_contact_forward','reward_orientation','reward_yaw_progress','reward_lin_vel_z','reward_feet_air_time_positive']
a,b=int(sys.argv[1]),int(sys.argv[2])
print('%-12s'%'arm'+''.join('%9s'%t.replace('penalty_','p_').replace('reward_','r_')[:8] for t in T)+'   SUM*40  meanRew  epLen')
for x in FAST+SLOW:
    v=[win(x,'Episode_Reward/'+t,a,b) for t in T]
    tot=0
    d=load(x)
    for t in d:
        if t.startswith('Episode_Reward/'): tot+=win(x,t,a,b)
    print('%-12s'%ARM[x]+''.join('%9.3f'%u for u in v)+'  %7.2f %7.2f %7.0f'%(tot*40,win(x,'Train/mean_reward',a,b),win(x,'Train/mean_episode_length',a,b)))
