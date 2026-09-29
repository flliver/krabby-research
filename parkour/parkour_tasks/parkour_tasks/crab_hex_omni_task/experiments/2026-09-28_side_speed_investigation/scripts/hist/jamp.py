import sys, json, os, glob, ast, numpy as np
def run(name, stamp, keep, nmax=40):
    meta=json.load(open(os.path.join(stamp,'run_meta.json')))
    jn=meta['joint_names']; jn=ast.literal_eval(jn) if isinstance(jn,str) else jn
    an=meta['action_joint_names']; an=ast.literal_eval(an) if isinstance(an,str) else an
    labs=meta['hold_labels']; res={}
    files=sorted(glob.glob(os.path.join(stamp,'raw','episode_*.npz')))
    for f in files:
        a=np.load(f); seg=a['segment_id']; st=a['steady_mask'].astype(bool)
        for sid,lab in enumerate(labs):
            if lab not in keep: continue
            m=(seg==sid)&st
            if m.sum()<100: continue
            if len(res.get(lab,[]))>=nmax: continue
            q=a['joint_pos'][m]; amp=np.percentile(q,95,0)-np.percentile(q,5,0)
            act=a['actions'][m]; amu=np.abs(act).mean(0); sat=(np.abs(act)>=0.99).mean(0)
            res.setdefault(lab,[]).append((amp,amu,sat))
    types=['Body_Hip','Body_CamShaft','Hip_Femur','Femur_Tibia']
    for lab,v in res.items():
        amp=np.median([x[0] for x in v],0)
        s=f"{name:9s} {lab:6s} n={len(v)} jointpos p5-95 range (rad) by type, legs FL FR ML MR RL RR:"
        print(s)
        for t in types:
            idx=[jn.index(f'{L}_{t}_RevoluteJoint') for L in ['FL','FR','ML','MR','RL','RR']]
            print(f"     {t:14s}"+' '.join(f"{amp[i]:6.3f}" for i in idx))
        amu=np.median([x[1] for x in v],0); sat=np.median([x[2] for x in v],0)
        print(f"     action |a| mean by action-joint group: "+'; '.join(f"{g}:{amu[[i for i,n in enumerate(an) if g in n]].mean():.2f} sat{sat[[i for i,n in enumerate(an) if g in n]].mean():.2f}" for g in ['CamShaft','Hip_Femur','Femur_Tibia','Body_Hip']))
for x in sys.argv[1:]:
    n,s,l=x.split('|'); run(n,s,l.split(','))
