import sys, os, glob, json
import numpy as np
from multiprocessing import Pool
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl"
RUNS = {
 # morph P1 (seed 3, 0->5k from scratch)
 "P1_base": "crab_hex_flat_walk/2026-09-04_13-18-07",
 "P1_B": "crab_hex_flat_walk/2026-09-04_15-49-15",
 "P1_A10": "crab_hex_flat_walk/2026-09-04_19-33-30",
 "P1_A15": "crab_hex_flat_walk/2026-09-04_22-02-27",
 "P1_A20": "crab_hex_flat_walk/2026-09-05_00-31-35",
 "P1_A10pB": "crab_hex_flat_walk/2026-09-05_03-00-33",
 "P1_A15pB": "crab_hex_flat_walk/2026-09-05_05-27-19",
 "P1_A20pB": "crab_hex_flat_walk/2026-09-05_07-56-31",
 # morph P2 s1 / s2 seed3
 "P2s1_base": "crab_hex_flat_walk/2026-09-05_12-03-33",
 "P2s2_base": "crab_hex_flat_walk/2026-09-05_14-32-24",
 "P2s1_A15pB_s3": "crab_hex_flat_walk/2026-09-05_16-51-20",
 "P2s2_A15pB_s3": "crab_hex_flat_walk/2026-09-05_19-18-29",
 "P2s1_B_s3": "crab_hex_flat_walk/2026-09-05_21-37-35",
 "P2s2_B_s3": "crab_hex_flat_walk/2026-09-06_00-09-00",
 "P2s1_B_s2": "crab_hex_flat_walk/2026-09-06_11-13-20",
 "P2s2_B_s2": "crab_hex_flat_walk/2026-09-06_13-42-22",
 "P2s1_A15pB_s2": "crab_hex_flat_walk/2026-09-06_16-03-30",
 "P2s2_A15pB_s2": "crab_hex_flat_walk/2026-09-06_18-32-32",
 # lineage seed3
 "L_w0": "crab_hex_flat_walk/2026-09-06_21-32-46",
 "L_w1": "crab_hex_flat_walk/2026-09-06_23-59-54",
 "L_w2": "crab_hex_flat_walk/2026-09-07_02-19-49",
 "L_w3": "crab_hex_flat_walk/2026-09-07_04-38-50",
 "L_w4": "crab_hex_flat_walk/2026-09-07_06-57-52",
 "L_w5": "crab_hex_flat_walk/2026-09-07_09-16-39",
 # student
 "S_3a": "crab_hex_student/2026-09-08_05-54-01",
 "S_3b": "crab_hex_student/2026-09-09_02-06-51",
 # sideways
 "SD_s3_1a": "crab_hex_side_flat_walk/2026-09-11_10-57-55",
 "SD_s3_2a": "crab_hex_side_teacher/2026-09-11_14-50-01",
 "SD_s3_2b": "crab_hex_side_teacher/2026-09-11_17-19-40",
 "SD_s3_2c": "crab_hex_side_teacher/2026-09-11_19-43-19",
 "SD_s2_1a": "crab_hex_side_flat_walk/2026-09-12_00-31-09",
 "SD_s2_2a": "crab_hex_side_teacher/2026-09-12_03-04-37",
 "SD_s2_2b": "crab_hex_side_teacher/2026-09-12_05-24-07",
 "SD_s2_2c": "crab_hex_side_teacher/2026-09-12_07-45-40",
}
TAGS = ["Loss/learning_rate", "Policy/mean_noise_std", "Loss/entropy", "Loss/value_function",
        "Loss/surrogate", "Loss/symmetry", "Loss/priv_reg_loss", "Loss/priv_reg_coef", "Perf/total_fps",
        "Train/mean_reward", "Train/mean_episode_length", "Loss/depth_actor_loss", "Loss/kl",
        "Perf/collection time", "Perf/learning_time", "Loss/hist_latent_loss", "Loss/estimator"]


def summarize(item):
    name, rel = item
    d = os.path.join(ROOT, rel)
    fs = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
    if not fs:
        return name, {"error": "no events"}
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    avail = ea.Tags()["scalars"]
    out = {"rel": rel, "n_tags": len(avail)}
    wanted = [t for t in avail if t in TAGS or t.startswith("Loss/") or t.startswith("Policy/") or t.startswith("Perf/")]
    for t in wanted:
        ev = ea.Scalars(t)
        steps = np.array([e.step for e in ev]); vals = np.array([e.value for e in ev], dtype=float)
        if len(vals) == 0:
            continue
        s0 = steps[0]
        def at(frac_or_abs):
            idx = int(np.argmin(np.abs(steps - frac_or_abs)))
            return float(vals[idx])
        first50 = float(np.mean(vals[:50])); last50 = float(np.mean(vals[-50:]))
        rec = {"n": int(len(vals)), "step0": int(s0), "stepN": int(steps[-1]),
               "v0": float(vals[0]), "first50": first50, "last50": last50,
               "min": float(vals.min()), "max": float(vals.max()),
               "at+100": at(s0 + 100), "at+1000": at(s0 + 1000), "at+2500": at(s0 + 2500)}
        if t == "Loss/learning_rate":
            rec["frac_at_floor_1e-5"] = float(np.mean(vals <= 1.0001e-5))
            rec["first_step_at_floor"] = int(steps[np.argmax(vals <= 1.0001e-5)]) if np.any(vals <= 1.0001e-5) else None
            rec["frac_ge_1e-4"] = float(np.mean(vals >= 1e-4))
        if t == "Policy/mean_noise_std":
            rec["frac_ge_1.99"] = float(np.mean(vals >= 1.99))
            rec["first_step_ge_1.99"] = int(steps[np.argmax(vals >= 1.99)]) if np.any(vals >= 1.99) else None
        out[t] = rec
    return name, out


if __name__ == "__main__":
    os.nice(10)
    with Pool(4) as p:
        res = dict(p.map(summarize, list(RUNS.items())))
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "tb_summary.json"), "w"), indent=1)
    print("done", len(res))
