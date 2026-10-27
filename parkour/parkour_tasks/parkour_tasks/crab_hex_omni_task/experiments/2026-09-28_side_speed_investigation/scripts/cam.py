import json,numpy as np,glob
def stats(stamp, want):
    meta=json.load(open(stamp+'/run_meta.json'))
    jn=meta['joint_names']; hl=meta['hold_labels']
    cam=[i for i,n in enumerate(jn) if 'CamShaft' in n]
    hip=[i for i,n in enumerate(jn) if 'Hip_Femur' in n]
    tib=[i for i,n in enumerate(jn) if 'Femur_Tibia' in n]
    res={}
    for f in sorted(glob.glob(stamp+'/raw/*.npz')):
        d=np.load(f)
        seg=d['segment_id']; m=d['steady_mask']&~d['done']
        for sid in np.unique(seg[m]):
            lab=hl[sid] if isinstance(hl,list) else hl[str(sid)]
            if lab not in want: continue
            mm=m&(seg==sid)
            if mm.sum()<50: continue
            jv=d['joint_vel'][mm]; jp=d['joint_pos'][mm]; a=d['actions'][mm]
            v=d['root_lin_vel_b'][mm]
            r=res.setdefault(lab,[])
            r.append([np.abs(jv[:,cam]).mean(), np.abs(jv[:,hip]).mean(), np.abs(jv[:,tib]).mean(),
                      jp[:,tib].std(0).mean(), jp[:,hip].std(0).mean(),
                      np.abs(a[:,:6]).mean(), np.abs(a[:,6:12]).mean(), np.abs(a[:,12:]).mean(), (np.abs(a)>0.98).mean(),
                      v[:,0].mean(), v[:,1].mean()])
    return jn,res
hdr='|cam_w| |hip_w| |tib_w| tib_pos_sd hip_pos_sd |a_cam| |a_hip| |a_tib| sat_frac vx vy'
for stamp,want in [('/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39',['creep','low']),
                   ('/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29',['lat_l','lat_r','fwd','back'])]:
    jn,res=stats(stamp,want)
    print(stamp.split('/')[-5]); print('   ',hdr)
    for k,v in res.items():
        v=np.array(v); print('   %-6s n=%3d '%(k,len(v))+' '.join('%7.3f'%x for x in np.median(v,0)))
