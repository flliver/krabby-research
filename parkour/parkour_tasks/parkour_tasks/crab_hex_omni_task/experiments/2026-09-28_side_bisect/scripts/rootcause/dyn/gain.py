import sys; sys.path.insert(0,'.')
from common import *
import json
G={'F':FAST,'S1':['X01','X05','X06'],'S2':['X10','X11','X14','X16']}
T=['track_lin_vel_xy_exp','penalty_tracking_error_l1','reward_forward_progress_along_command','track_ang_vel_z_exp','reward_clock_schedule','reward_clock_swing_apex','penalty_foot_idle_when_forward','reward_action_rate','reward_delta_torques','penalty_tibia_deviation_in_stance','penalty_excess_feet_contact_forward','penalty_lin_vel_y']
def rows(a0,a1,b0,b1):
    print(f'-- change in per-second rate from [{a0},{a1}) to [{b0},{b1})')
    for g,xs in G.items():
        d={t:np.median([win(x,'Episode_Reward/'+t,b0,b1)-win(x,'Episode_Reward/'+t,a0,a1) for x in xs]) for t in T}
        track=d['track_lin_vel_xy_exp']+d['penalty_tracking_error_l1']+d['reward_forward_progress_along_command']+d['penalty_lin_vel_y']
        smooth=d['reward_action_rate']+d['reward_delta_torques']
        print('  %-3s tracking(4) %+.3f | smoothing(ar+dtq) %+.3f | foot_idle %+.3f | clock+apex %+.3f | tibia+excess %+.3f | angz %+.3f'%(g,track,smooth,d['penalty_foot_idle_when_forward'],d['reward_clock_schedule']+d['reward_clock_swing_apex'],d['penalty_tibia_deviation_in_stance']+d['penalty_excess_feet_contact_forward'],d['track_ang_vel_z_exp']))
rows(100,200,400,500); rows(450,550,2500,3000)
