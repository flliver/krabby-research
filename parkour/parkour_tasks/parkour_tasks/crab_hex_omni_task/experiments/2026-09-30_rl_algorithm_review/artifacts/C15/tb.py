import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    fs=sorted(glob.glob(d+'/events.out.tfevents.*'))
    for f in fs:
        ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
        tags=ea.Tags()['scalars']
        out=[d.split('/')[-1], f.split('.')[-2][-6:], 'n_tags',len(tags)]
        for t in ['Loss/symmetry','Train/mean_reward','Policy/mean_noise_std','Loss/learning_rate']:
            if t in tags:
                s=ea.Scalars(t); 
                out.append(f"{t}: first({s[0].step})={s[0].value:.4g} last({s[-1].step})={s[-1].value:.4g}")
            else: out.append(f"{t}: ABSENT")
        print(' | '.join(map(str,out)))
