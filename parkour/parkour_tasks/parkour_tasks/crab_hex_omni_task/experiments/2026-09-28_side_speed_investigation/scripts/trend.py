import sys,math
sys.path.insert(0,'/tmp/claude-1000/sidespeed')
from parse_cat import parse,avg
cats=['fwd','side','back','diag','move_turn']
# mean commanded speed per pure category (U(0.21,cap)) approx
cmd={'fwd':0.28,'side':0.23,'back':0.23,'diag':0.249,'move_turn':None}
for f in sys.argv[1:]:
    its=parse(f); n=len(its); W=int(max(50,n//10))
    print(f.split('/')[-1], n, 'iters', its[0]['it'], its[-1]['it'])
    print('   window        '+' '.join('%14s'%c for c in cats)+'   (progress/0.6 -> m/s ; ratio=prog/cmd ; L1err=-l1/2)')
    for lo in list(range(0,n,W)):
        sub=its[lo:lo+W]
        row=[]
        for c in cats:
            p=avg(sub,(c,'reward_forward_progress_along_command'),(c,'dwell_s'))/0.6
            l1=-avg(sub,(c,'penalty_tracking_error_l1'),(c,'dwell_s'))/2
            r=p/cmd[c] if cmd[c] else float('nan')
            row.append('%5.3f %4.2f %4.3f'%(p,r,l1))
        print('   %5d-%5d '%(sub[0]['it'],sub[-1]['it'])+' '.join('%14s'%x for x in row))
