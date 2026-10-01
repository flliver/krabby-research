import numpy as np, pickle, sys
from multiprocessing import Pool
def parse(p):
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto import event_pb2
    S={};E={};err=None
    try:
        for raw in RawEventFileLoader('/home/nickmagus/krabby/krabby-research/'+p[2:]).Load():
            ev=event_pb2.Event.FromString(raw)
            for v in ev.summary.value:
                if v.tag in ('Policy/mean_noise_std','Loss/entropy'):
                    val=v.simple_value if v.HasField('simple_value') else (v.tensor.float_val[0] if v.tensor.float_val else None)
                    (S if v.tag[0]=='P' else E)[ev.step]=val
    except Exception as e: err=repr(e)[:200]
    return p, np.array([S[k] for k in sorted(S)]), np.array([E[k] for k in sorted(E)]), err
if __name__=='__main__':
    ps=[l.strip() for l in open('outside.txt')]
    with Pool(4) as pool: res=pool.map(parse, ps)
    pickle.dump(res, open('outside.pkl','wb'))
