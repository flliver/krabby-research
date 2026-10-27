import sys,re,collections,math
cats=['stand','fwd','side','back','diag','turn','move_turn']
def parse(f):
    its=[];cur=None
    for l in open(f,errors='ignore'):
        if 'Learning iteration' in l:
            m=re.search(r'Learning iteration (\d+)/',l); cur={'it':int(m.group(1))}; its.append(cur); continue
        if cur is None: continue
        m=re.search(r'Episode_RewardCat/(\w+)/(\S+): (\S+)',l)
        if m: cur[(m.group(1),m.group(2))]=float(m.group(3)); continue
        m=re.search(r'(Metrics/\S+|Episode_Reward/\S+|Mean reward|Mean episode length): (\S+)',l)
        if m:
            try: cur[m.group(1)]=float(m.group(2))
            except: pass
    return its
def avg(its,key,wkey=None):
    s=0;n=0
    for it in its:
        v=it.get(key)
        if v is None or math.isnan(v): continue
        w=1.0
        if wkey: 
            w=it.get(wkey,0.0)
            if not w or math.isnan(w): continue
        s+=v*w;n+=w
    return s/n if n else float('nan')
if __name__=='__main__':
    f=sys.argv[1]; last=int(sys.argv[2]) if len(sys.argv)>2 else 300
    its=parse(f)
    print(f, 'iters',len(its), its[0]['it'], its[-1]['it'])
    for lo in ([0,len(its)//4,len(its)//2,3*len(its)//4,len(its)-last] if len(sys.argv)>3 else [len(its)-last]):
        sub=its[lo:lo+last] if lo!=len(its)-last else its[lo:]
        print('--- iters %d..%d'%(sub[0]['it'],sub[-1]['it']))
        for t in ['track_lin_vel_xy_exp','penalty_tracking_error_l1','reward_forward_progress_along_command','penalty_lin_vel_y','penalty_foot_idle_when_forward','penalty_excess_feet_contact_forward','reward_action_rate','reward_delta_torques','penalty_tibia_deviation_in_stance','reward_stroke','reward_stroke_length','pre_clip_total','post_clip_total','clip_bind_frac','dwell_s']:
            row=[avg(sub,(c,t),(c,'dwell_s') if t!='dwell_s' else None) for c in cats]
            if all(math.isnan(x) for x in row): continue
            print('%-40s'%t,' '.join('%8.4f'%x for x in row))
        for k in ['Metrics/base_parkour/track_ratio_mean','Metrics/base_parkour/track_ratio_move_mean','Metrics/base_parkour/rsi_frac_actual','Mean episode length','Metrics/base_velocity/recap_frac']:
            print(k, '%.4f'%avg(sub,k))
