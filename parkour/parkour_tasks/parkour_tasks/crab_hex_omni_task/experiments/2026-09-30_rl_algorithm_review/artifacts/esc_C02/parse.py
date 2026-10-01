import sys, os, pickle, re
from multiprocessing import Pool
def parse(path):
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto import event_pb2
    lr=[]; err=None; ntags=set()
    try:
        for raw in RawEventFileLoader(path).Load():
            ev=event_pb2.Event.FromString(raw)
            if not ev.HasField("summary"): continue
            for v in ev.summary.value:
                ntags.add(v.tag)
                if v.tag=="Loss/learning_rate":
                    val=v.simple_value if v.HasField("simple_value") else (v.tensor.float_val[0] if v.tensor.float_val else None)
                    if val is not None: lr.append((int(ev.step),float(val)))
    except Exception as e: err=repr(e)[:200]
    d=os.path.dirname(path); ay=os.path.join(d,"params","agent.yaml")
    cls=None; lrc=None; sch=None; res=None; ras=None
    if os.path.exists(ay):
        t=open(ay).read()
        i=t.find("algorithm:")
        m=re.search(r"class_name:\s*(\S+)", t[i:]) if i>=0 else None
        cls=m.group(1) if m else None
        m=re.search(r"^\s*learning_rate:\s*(\S+)", t[i:], re.M) if i>=0 else None
        lrc=float(m.group(1)) if m else None
        m=re.search(r"^\s*schedule:\s*(\S+)", t, re.M); sch=m.group(1) if m else None
        m=re.search(r"^resume:\s*(\S+)", t, re.M); res=m.group(1) if m else None
        m=re.search(r"^resume_alg_state:\s*(\S+)", t, re.M); ras=m.group(1) if m else None
    return dict(path=path,lr=lr,err=err,cls=cls,lr_cfg=lrc,sched=sch,resume=res,ras=ras,has_tag="Loss/learning_rate" in ntags)
if __name__=="__main__":
    files=[l.strip() for l in open(sys.argv[1]) if l.strip()]
    out=[]
    with Pool(4) as p:
        for r in p.imap_unordered(parse, files, chunksize=4): out.append(r)
    pickle.dump(out, open(sys.argv[2],"wb")); print("done",len(out))
