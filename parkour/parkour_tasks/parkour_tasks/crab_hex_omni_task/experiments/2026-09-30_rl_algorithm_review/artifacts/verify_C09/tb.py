import sys, glob, numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
TAGS=['Perf/collection time','Perf/learning_time','Perf/total_fps']
def load(run):
    f=sorted(glob.glob(run+'/events.out.tfevents.*'))
    out={t:[] for t in TAGS}; walls=[]
    for fn in f:
        for ev in EventFileLoader(fn).Load():
            for v in ev.summary.value:
                if v.tag in out:
                    val=v.simple_value if v.HasField('simple_value') else float(np.frombuffer(v.tensor.tensor_content,dtype=np.float32)[0]) if v.tensor.tensor_content else v.tensor.float_val[0]
                    out[v.tag].append((ev.step,val,ev.wall_time))
    return out
for run in sys.argv[1:]:
    o=load(run)
    c=np.array([x[1] for x in o['Perf/collection time']]); l=np.array([x[1] for x in o['Perf/learning_time']]); fps=np.array([x[1] for x in o['Perf/total_fps']])
    k=min(20,len(c))
    w=[x[2] for x in o['Perf/total_fps']]
    print(run.split('rsl_rl/')[1], 'n',len(c),'collect med %.3f (it0-19 %.3f, it0 %.3f) learn med %.3f fps med %.0f (0-19 %.0f) wall/iter %.3f'%(np.median(c),np.median(c[:k]),c[0],np.median(l),np.median(fps),np.median(fps[:k]), (w[-1]-w[0])/max(1,len(w)-1)))
