import sys, glob, numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util
def load(run_dir, tags=('Perf/collection time','Perf/learning_time','Perf/total_fps','Train/mean_episode_length')):
    out={t:[] for t in tags}
    for f in sorted(glob.glob(run_dir+'/events.out.tfevents.*')):
        for ev in EventFileLoader(f).Load():
            if not ev.HasField('summary'): continue
            for v in ev.summary.value:
                if v.tag in out:
                    val = v.simple_value if v.HasField("simple_value") else float(tensor_util.make_ndarray(v.tensor).ravel()[0])
                    out[v.tag].append((ev.step, ev.wall_time, val))
    return {t:np.array(v) for t,v in out.items() if v}
if __name__=='__main__':
    for d in sys.argv[1:]:
        s=load(d)
        c=s['Perf/collection time']; l=s['Perf/learning_time']; f=s['Perf/total_fps']
        w=np.diff(c[:,1]); 
        print(d.split('/')[-2][:22], d.split('/')[-1][:45], 'its',len(c),'steps %d-%d'%(c[0,0],c[-1,0]),
              'collect med %.3f'%np.median(c[:,2]),'learn med %.3f'%np.median(l[:,2]),'fps med %.0f'%np.median(f[:,2]),
              'wall/iter med %.3f'%np.median(w), 'eplen %.0f->%.0f'%(s['Train/mean_episode_length'][0,2],s['Train/mean_episode_length'][-1,2]) if 'Train/mean_episode_length' in s else '')
