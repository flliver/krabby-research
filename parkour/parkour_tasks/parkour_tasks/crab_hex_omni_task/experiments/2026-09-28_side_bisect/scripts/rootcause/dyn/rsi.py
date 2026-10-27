import sys; sys.path.insert(0,'.')
from common import *
P='Metrics/base_parkour/'
T=['rsi_frac_actual','reach_edge_frac','reach_edge_frac_rsi','field_frac_mean','field_frac_mean_rsi','goals_passed_mean','goals_passed_mean_rsi','crab_failure_obst','crab_failure_obst_rsi','crab_failure_obst_spread','spread_frac_actual','ep_steps_obst','r_max_mean']
wins=[(0,100),(100,200),(200,300),(300,400),(400,500),(500,700),(700,1000),(1000,1500),(1500,2000),(2500,3000)]
for x in ['X17','X18','X15','X16','X19','X14','X01','X04','X08','X09','X10','X11','X02']:
    print('==',x,ARM[x])
    for t in T:
        print('  %-26s'%t+' '.join('%7.3f'%win(x,P+t,a,b) for a,b in wins))
