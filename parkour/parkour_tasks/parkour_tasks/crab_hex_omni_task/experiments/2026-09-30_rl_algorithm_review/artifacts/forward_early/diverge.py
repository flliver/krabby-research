import sys
sys.path.insert(0,'/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/forward_early')
from parse_logs import parse, sig
a=parse(sys.argv[1]); b=parse(sys.argv[2])
for i in [0,1,2,5,10,20,30,50,100,150,200,300,500,1000,2000,2999]:
    if i<len(a) and i<len(b):
        x,y=a[i],b[i]
        print(i, 'A rew %.3f ep %.0f fail %.3f vf %.5f ent %.3f'%(x.get('rew',0),x.get('eplen',0),x.get('fail',0),x.get('vf',0),x.get('ent',0)), '| B rew %.3f ep %.0f fail %.3f vf %.5f ent %.3f'%(y.get('rew',0),y.get('eplen',0),y.get('fail',0),y.get('vf',0),y.get('ent',0)))
