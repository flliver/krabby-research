import sys, os, glob, pickle
from multiprocessing import Pool
def parse(path):
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto import event_pb2
    lr=[]; has_sym=False; err=None
    try:
        for raw in RawEventFileLoader(path).Load():
            ev=event_pb2.Event.FromString(raw)
            if not ev.HasField("summary"): continue
            for v in ev.summary.value:
                if v.tag=="Loss/learning_rate":
                    val=v.simple_value if v.HasField("simple_value") else (v.tensor.float_val[0] if v.tensor.float_val else None)
                    if val is not None: lr.append((ev.step,val))
    except Exception as e: err=repr(e)[:200]
    return path, lr, err
if __name__=="__main__":
    files=[l.strip() for l in open(sys.argv[1]) if l.strip()]
    out={}
    with Pool(4) as p:
        for path,lr,err in p.imap_unordered(parse, files):
            out[path]=(lr,err)
    pickle.dump(out, open(sys.argv[2],"wb"))
    print(len(out))
