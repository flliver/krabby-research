import sys, glob, numpy as np, datetime
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
TAGS=['Perf/collection time','Perf/learning_time','Perf/total_fps']
for run in sys.argv[1:]:
    out={t:[] for t in TAGS}
    for fn in sorted(glob.glob(run+'/events.out.tfevents.*')):
        for ev in EventFileLoader(fn).Load():
            for v in ev.summary.value:
                if v.tag in out:
                    val=v.simple_value if v.HasField('simple_value') else v.tensor.float_val[0]
                    out[v.tag].append((ev.step,val,ev.wall_time))
    c=np.array(out['Perf/collection time']); l=np.array(out['Perf/learning_time']); f=np.array(out['Perf/total_fps'])
    if len(c)==0: print(run,'empty'); continue
    t0=datetime.datetime.fromtimestamp(c[0,2]).strftime('%H:%M'); t1=datetime.datetime.fromtimestamp(c[-1,2]).strftime('%H:%M')
    print(run.split('experiments/')[-1][:70], t0,'-',t1,'n',len(c))
    # rolling windows of 50 its
    for i in range(0,len(c),max(50,len(c)//8)):
        s=slice(i,i+50)
        print('   it %5d %s collect %.3f learn %.3f fps %.0f'%(c[i,0], datetime.datetime.fromtimestamp(c[i,2]).strftime('%H:%M'), np.median(c[s,1]), np.median(l[s,1]), np.median(f[s,1])))
