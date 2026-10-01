import sys, glob, statistics as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root = "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/"
files = sorted(glob.glob(root+"2026-08-1[2-3]*/**/events.out.tfevents*", recursive=True))
for f in files:
    rel = f.replace(root,"").split("/logs/")[0]
    ea = EventAccumulator(f, size_guidance={"scalars":0}); ea.Reload()
    tags = ea.Tags()["scalars"]
    def s(k):
        return [e.value for e in ea.Scalars(k)] if k in tags else None
    def st_(k):
        return [e.step for e in ea.Scalars(k)] if k in tags else None
    lr=s("Loss/learning_rate"); sg=s("Policy/mean_noise_std"); vf=s("Loss/value_function"); fps=s("Perf/total_fps"); rw=s("Train/mean_reward"); sym=s("Loss/symmetry"); ent=s("Loss/entropy"); steps=st_("Train/mean_reward") or st_("Loss/learning_rate")
    line=[rel, f"it {steps[0]}-{steps[-1]} n={len(steps)}" if steps else "no-steps"]
    if lr: line.append(f"LR med={st.median(lr):.1e} max={max(lr):.1e} frac@<=1.01e-5={sum(x<=1.01e-5 for x in lr)/len(lr):.2f} first5={[f'{x:.0e}' for x in lr[:5]]}")
    if sg:
        first_pin = next((i for i,x in enumerate(sg) if x>=1.99), None)
        line.append(f"sigma first={sg[0]:.3f} last={sg[-1]:.3f} min={min(sg):.3f} max={max(sg):.3f} frac>=1.99={sum(x>=1.99 for x in sg)/len(sg):.2f} first_idx>=1.99={first_pin}")
    if ent: line.append(f"ent first={ent[0]:.2f} last={ent[-1]:.2f}")
    if vf: line.append(f"VF first={vf[0]:.4f} med={st.median(vf):.4f} last={vf[-1]:.4f} max={max(vf):.3f}")
    if sym: line.append(f"sym first={sym[0]:.4f} last={sym[-1]:.4f}")
    if fps: line.append(f"fps med={st.median(fps):.0f}")
    if rw:
        def at(q):
            i=min(range(len(steps)), key=lambda j:abs(steps[j]-(steps[0]+q)))
            return rw[i]
        line.append(f"rew first={rw[0]:.2f} @500={at(500):.1f} @1000={at(1000):.1f} @2999={at(2999):.1f} last={rw[-1]:.2f} max={max(rw):.2f}")
    print(" | ".join(line))
