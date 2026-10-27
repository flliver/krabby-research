import sys, os, glob, json
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util

def read_scalars(run_dir, tags=None):
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents*"))):
        for ev in EventFileLoader(f).Load():
            if not ev.HasField("summary"):
                continue
            for v in ev.summary.value:
                if tags is not None and v.tag not in tags:
                    continue
                if v.HasField("simple_value"):
                    val = v.simple_value
                elif v.HasField("tensor"):
                    try:
                        val = float(tensor_util.make_ndarray(v.tensor))
                    except Exception:
                        continue
                else:
                    continue
                out.setdefault(v.tag, []).append((ev.step, val))
    return out

if __name__ == "__main__":
    d = read_scalars(sys.argv[1])
    for k in sorted(d):
        print(k, len(d[k]), d[k][:2], d[k][-1:])
