import sys,glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    ea=EventAccumulator(glob.glob(d+'/events*')[0], size_guidance={'scalars':0}); ea.Reload()
    s=[x.value for x in ea.Scalars('Policy/mean_noise_std')]
    i=next((k for k,x in enumerate(s) if x>=1.99),-1)
    print(d.split('/')[-1],'first sigma>=1.99 at it',i,'of',len(s),'sigma@500',round(s[min(500,len(s)-1)],3),'@1000',round(s[min(1000,len(s)-1)],3))
