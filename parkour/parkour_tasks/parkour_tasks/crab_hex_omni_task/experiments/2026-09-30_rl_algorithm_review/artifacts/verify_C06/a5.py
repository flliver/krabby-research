import glob, numpy as np
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
R='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/'
for run in ['crab_hex_flat_walk/2026-08-20_19-01-38','crab_hex_omni_flat_walk/2026-09-23_04-25-38_seed3_X03_c6ft_A15pB','crab_hex_flat_walk/2026-08-04_11-55-56']:
    f=glob.glob(R+run+'/events.out.tfevents*')
    S={};E={}
    for p in f:
        for raw in RawEventFileLoader(p).Load():
            ev=event_pb2.Event.FromString(raw)
            for v in ev.summary.value:
                if v.tag=='Policy/mean_noise_std': S[ev.step]=v.simple_value if v.HasField('simple_value') else v.tensor.float_val[0]
                if v.tag=='Loss/entropy': E[ev.step]=v.simple_value if v.HasField('simple_value') else v.tensor.float_val[0]
    s=np.array([S[k] for k in sorted(S)]); e=np.array([E[k] for k in sorted(E)])
    print(run, len(f), len(s), 'first %.4f min %.4f max %.5f last %.5f'%(s[0],s.min(),s.max(),s[-1]), 'ent min %.3f max %.3f'%(e.min(),e.max()))
