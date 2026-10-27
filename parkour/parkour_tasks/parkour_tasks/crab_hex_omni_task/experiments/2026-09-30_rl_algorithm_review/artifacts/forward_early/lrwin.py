import sys, statistics as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    ea=EventAccumulator(d,size_guidance={'scalars':0}); ea.Reload()
    lr=[e.value for e in ea.Scalars('Loss/learning_rate')]
    ns=[e.value for e in ea.Scalars('Policy/mean_noise_std')]
    n=len(lr); w=max(1,n//10)
    print('==',d.split('rsl_rl/')[-1], 'n',n,'overall frac<=1.01e-5 %.3f mean %.2e median %.2e'%(sum(v<=1.01e-5 for v in lr)/n, st.mean(lr), st.median(lr)))
    print('   window mean LR:',' '.join('%.1e'%st.mean(lr[i:i+w]) for i in range(0,n,w)))
    print('   window floor frac:',' '.join('%.2f'%(sum(v<=1.01e-5 for v in lr[i:i+w])/len(lr[i:i+w])) for i in range(0,n,w)))
    print('   noise std window end:',' '.join('%.2f'%ns[min(i+w-1,n-1)] for i in range(0,n,w)))
