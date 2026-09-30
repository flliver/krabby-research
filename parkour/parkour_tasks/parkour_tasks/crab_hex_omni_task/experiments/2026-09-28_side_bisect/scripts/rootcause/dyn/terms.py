import sys; sys.path.insert(0,'.')
from common import *
G={'F':FAST,'S1':['X01','X05','X06'],'S2':['X10','X11','X14','X16','X19']}
tags=sorted(t for t in load('X10') if t.startswith(('Episode_Reward','Loss','Policy','Train','Episode_Term','Metrics/base_velocity/error','Metrics/base_parkour/crab_failure_f','Metrics/base_parkour/track')))
W=[(100,200),(250,350),(400,500),(550,650),(900,1000),(1400,1500),(2500,3000)]
print('%-44s'%'term', ' | '.join('%-24s'%f'{a}-{b}' for a,b in W))
for t in tags:
    row=[]
    for a,b in W:
        cells=[]
        for g,xs in G.items():
            vals=[win(x,t,a,b) for x in xs]; vals=[v for v in vals if not np.isnan(v)]
            cells.append('%7.3f'%np.mean(vals) if vals else '   nan ')
        row.append(' '.join(cells))
    print('%-44s'%t.replace('Episode_Reward/','R/').replace('Metrics/base_parkour/','').replace('Metrics/base_velocity/',''), ' | '.join(row))
