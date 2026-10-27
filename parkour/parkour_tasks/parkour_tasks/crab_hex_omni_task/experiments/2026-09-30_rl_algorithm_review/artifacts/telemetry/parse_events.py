"""CPU-only TensorBoard census of rsl_rl training runs (read-only on the repo).

Stage 1: parse every events file -> per-run pickled series (Loss/*, Loss_depth/*, Policy/*, Perf/*,
Train/mean_reward, Train/mean_episode_length) with steps + wall times, into series/<id>.pkl.
"""
import os, sys, glob, pickle, hashlib, time, traceback
from multiprocessing import Pool

ROOT = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl"
OUT = os.path.dirname(os.path.abspath(__file__))
SER = os.path.join(OUT, "series")
os.makedirs(SER, exist_ok=True)
EXPS = ["crab_hex_flat_walk", "crab_hex_teacher", "crab_hex_student", "crab_hex_side_flat_walk",
        "crab_hex_side_teacher", "crab_hex_omni_flat_walk", "crab_hex_omni_teacher", "_stopped_partial_runs"]
KEEP_PREFIX = ("Loss/", "Loss_depth/", "Policy/", "Perf/")
KEEP_EXACT = {"Train/mean_reward", "Train/mean_episode_length", "Episode_Termination/total_terminates",
              "Episode_Termination/crab_failure", "Episode_Termination/time_out"}


def rid(path):
    return hashlib.md5(path.encode()).hexdigest()[:12]


def parse(path):
    out = os.path.join(SER, rid(path) + ".pkl")
    st = os.stat(path)
    if os.path.exists(out):
        try:
            with open(out, "rb") as f:
                d = pickle.load(f)
            if d.get("size") == st.st_size:
                return path, "cached"
        except Exception:
            pass
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto import event_pb2
    series = {}
    other_tags = set()
    first_wall = None
    last_wall = None
    err = None
    n = 0
    try:
        for raw in RawEventFileLoader(path).Load():
            ev = event_pb2.Event.FromString(raw)
            n += 1
            if first_wall is None:
                first_wall = ev.wall_time
            last_wall = ev.wall_time
            if not ev.HasField("summary"):
                continue
            for v in ev.summary.value:
                tag = v.tag
                if tag.startswith(KEEP_PREFIX) or tag in KEEP_EXACT:
                    if v.HasField("simple_value"):
                        val = v.simple_value
                    elif v.HasField("tensor") and v.tensor.float_val:
                        val = v.tensor.float_val[0]
                    else:
                        continue
                    series.setdefault(tag, []).append((ev.step, ev.wall_time, val))
                else:
                    other_tags.add(tag.split("/")[0])
    except Exception as e:  # tolerate partial/actively-written files
        err = repr(e)[:300]
    d = {"path": path, "size": st.st_size, "mtime": st.st_mtime, "n_events": n, "first_wall": first_wall,
         "last_wall": last_wall, "series": series, "other_tag_groups": sorted(other_tags), "error": err}
    with open(out, "wb") as f:
        pickle.dump(d, f)
    return path, "ok" if err is None else "err:" + err


def main():
    files = []
    for e in EXPS:
        files += glob.glob(os.path.join(ROOT, e, "**", "events.out.tfevents*"), recursive=True)
    files.sort(key=lambda p: -os.path.getsize(p))
    print(len(files), "files", sum(os.path.getsize(p) for p in files) / 1e9, "GB", flush=True)
    t = time.time()
    with Pool(4) as pool:
        for i, (p, s) in enumerate(pool.imap_unordered(parse, files)):
            if i % 25 == 0 or not s.startswith(("ok", "cached")):
                print(i, round(time.time() - t), s, p[len(ROOT):], flush=True)
    print("done", time.time() - t)


if __name__ == "__main__":
    main()
