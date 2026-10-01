"""Stage 2: per-run summary -> runs.csv (+ runs.pkl with compact arrays for stage-3 analysis)."""
import os, glob, pickle, re, math, csv, datetime
import numpy as np
import yaml

OUT = os.path.dirname(os.path.abspath(__file__))
SER = os.path.join(OUT, "series")
ROOT = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl"
FLOOR = 1e-5


class L(yaml.SafeLoader):
    pass


def _any(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


L.add_multi_constructor("", _any)
L.add_multi_constructor("tag:yaml.org,2002:python/", _any)


def load_yaml(p):
    try:
        with open(p) as f:
            return yaml.load(f, Loader=L)
    except Exception as e:
        return {"_err": repr(e)[:200]}


def env_num_envs(p):
    try:
        with open(p) as f:
            for line in f:
                m = re.match(r"^\s{0,4}num_envs:\s*(\d+)", line)
                if m:
                    return int(m.group(1))
    except Exception:
        return None
    return None


def arr(series, tag):
    s = series.get(tag)
    if not s:
        return None
    a = np.array(s, dtype=np.float64)  # step, wall, val
    # de-dup by step keep last
    _, idx = np.unique(a[:, 0][::-1], return_index=True)
    idx = len(a) - 1 - idx
    a = a[np.sort(idx)]
    a = a[np.argsort(a[:, 0], kind="stable")]
    return a


def smooth(v, w=50):
    if len(v) < 2:
        return v
    w = max(1, min(w, len(v) // 5 or 1))
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty_like(v)
    for i in range(len(v)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def fnum(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}g}"
    return x


def main():
    rows = []
    compact = {}
    for p in sorted(glob.glob(os.path.join(SER, "*.pkl"))):
        with open(p, "rb") as f:
            d = pickle.load(f)
        path = d["path"]
        rel = os.path.relpath(path, ROOT)
        parts = rel.split("/")
        exp = parts[0]
        run_dir = os.path.dirname(path)
        run = os.path.relpath(run_dir, os.path.join(ROOT, exp))
        ser = d["series"]
        r = {"run_id": os.path.basename(p)[:-4], "experiment": exp, "run": run, "events_path": path,
             "events_MB": round(d["size"] / 1e6, 2), "parse_error": d["error"] or ""}
        r["start_time"] = (datetime.datetime.fromtimestamp(d["first_wall"]).strftime("%Y-%m-%d %H:%M:%S")
                           if d["first_wall"] else "")
        r["wall_clock_h"] = round((d["last_wall"] - d["first_wall"]) / 3600, 3) if d["first_wall"] else None
        ag = load_yaml(os.path.join(run_dir, "params", "agent.yaml")) if os.path.exists(
            os.path.join(run_dir, "params", "agent.yaml")) else {}
        alg = (ag or {}).get("algorithm", {}) or {}
        pol = (ag or {}).get("policy", {}) or {}
        sym = alg.get("symmetry_cfg") or {}
        r.update({
            "has_agent_yaml": bool(ag),
            "alg_class": alg.get("class_name"), "runner_class": ag.get("runner_class_name") if ag else None,
            "lr_cfg": alg.get("learning_rate"), "schedule": alg.get("schedule"), "desired_kl": alg.get("desired_kl"),
            "entropy_coef": alg.get("entropy_coef"), "init_noise_std": pol.get("init_noise_std"),
            "action_std_max": pol.get("action_std_max"), "action_std_reset": pol.get("action_std_reset"),
            "clip_param": alg.get("clip_param"), "epochs": alg.get("num_learning_epochs"),
            "minibatches": alg.get("num_mini_batches"), "gamma": alg.get("gamma"), "lam": alg.get("lam"),
            "max_grad_norm": alg.get("max_grad_norm"), "steps_per_env": ag.get("num_steps_per_env") if ag else None,
            "clip_actions": ag.get("clip_actions") if ag else None,
            "sym_coef": sym.get("mirror_loss_coeff") if sym.get("use_mirror_loss") else (0.0 if ag else None),
            "sym_func": (sym.get("data_augmentation_func") or "").split(":")[-1],
            "resume": ag.get("resume") if ag else None, "load_run": ag.get("load_run") if ag else None,
            "load_checkpoint": ag.get("load_checkpoint") if ag else None,
            "resume_alg_state": ag.get("resume_alg_state") if ag else None,
            "max_iterations": ag.get("max_iterations") if ag else None,
            "noise_std_type": pol.get("noise_std_type"), "policy_class": pol.get("class_name"),
            "num_envs": env_num_envs(os.path.join(run_dir, "params", "env.yaml")),
        })
        # iteration span
        lr = arr(ser, "Loss/learning_rate")
        anyloss = lr if lr is not None else next((arr(ser, t) for t in ser if t.startswith("Perf/")), None)
        r["iters_logged"] = int(len(anyloss)) if anyloss is not None else 0
        r["first_iter"] = int(anyloss[0, 0]) if anyloss is not None else None
        r["last_iter"] = int(anyloss[-1, 0]) if anyloss is not None else None
        comp = {"exp": exp, "run": run}
        # LR
        if lr is not None:
            v = lr[:, 2]
            st = lr[:, 0] - lr[0, 0]
            fl = v <= FLOOR * 1.01
            r.update({"lr_first": v[0], "lr_last": v[-1], "lr_min": v.min(), "lr_max": v.max(),
                      "lr_frac_floor": fl.mean(), "lr_frac_gt_1e-4": (v > 1e-4).mean(),
                      "lr_first_floor_iter": int(st[np.argmax(fl)]) if fl.any() else None})
            if fl.any():
                i0 = np.argmax(fl)
                after = v[i0:]
                off = after > FLOOR * 1.01
                # excursions
                runs_, cur = [], 0
                for o in off:
                    if o:
                        cur += 1
                    elif cur:
                        runs_.append(cur); cur = 0
                if cur:
                    runs_.append(cur)
                r.update({"lr_after_floor_frac_off": off.mean(), "lr_after_floor_max": after.max(),
                          "lr_n_excursions": len(runs_), "lr_longest_excursion": max(runs_) if runs_ else 0})
            # spikes: consecutive ratio
            ratio = v[1:] / np.maximum(v[:-1], 1e-12)
            r["lr_max_jump_ratio"] = ratio.max() if len(ratio) else None
            comp["lr"] = v.astype(np.float32)
        # sigma
        sd = arr(ser, "Policy/mean_noise_std")
        if sd is not None:
            v = sd[:, 2]; st = sd[:, 0] - sd[0, 0]
            cap = r["action_std_max"] if r["action_std_max"] not in (None, "") else 2.0
            try:
                cap = float(cap)
            except Exception:
                cap = 2.0
            pin = v >= cap - 0.01
            r.update({"std_cap": cap, "std_first": v[0], "std_last": v[-1], "std_max": v.max(), "std_min": v.min(),
                      "std_frac_ge_cap": pin.mean(), "std_first_iter_ge_cap": int(st[np.argmax(pin)]) if pin.any() else None,
                      "std_frac_ge_1.99": (v >= 1.99).mean()})
            comp["std"] = v.astype(np.float32)
        for tag, key in [("Loss/entropy", "ent"), ("Loss/value_function", "vf"), ("Loss/surrogate", "surr"),
                         ("Loss/symmetry", "sym"), ("Loss/priv_reg", "privreg"), ("Loss/priv_reg_coef", "privcoef"),
                         ("Loss/estimator", "est"), ("Loss/hist_latent", "hist")]:
            a = arr(ser, tag)
            if a is None:
                continue
            v = a[:, 2]
            n = len(v); k = max(1, min(100, n // 10))
            r[f"{key}_first"] = v[0]; r[f"{key}_last"] = v[-1]
            r[f"{key}_med_first"] = float(np.nanmedian(v[:k])); r[f"{key}_med_last"] = float(np.nanmedian(v[-k:]))
            r[f"{key}_max"] = float(np.nanmax(v)); r[f"{key}_nonfinite"] = int((~np.isfinite(v)).sum())
            if key in ("vf", "surr", "ent", "sym"):
                comp[key] = v.astype(np.float32)
        # perf
        fps = arr(ser, "Perf/total_fps"); ct = arr(ser, "Perf/collection time"); lt = arr(ser, "Perf/learning_time")
        if fps is not None:
            r["fps_median"] = float(np.median(fps[:, 2]))
        if ct is not None and lt is not None:
            n = min(len(ct), len(lt))
            tot = ct[:n, 2] + lt[:n, 2]
            r["collect_s_median"] = float(np.median(ct[:n, 2])); r["learn_s_median"] = float(np.median(lt[:n, 2]))
            r["s_per_iter_median"] = float(np.median(tot)); r["train_gpu_h"] = float(tot.sum() / 3600)
            r["learn_frac"] = float(lt[:n, 2].sum() / max(tot.sum(), 1e-9))
        # reward
        rw = arr(ser, "Train/mean_reward")
        if rw is not None:
            v = rw[:, 2]; st = rw[:, 0] - rw[0, 0]
            s = smooth(v, 50)
            k = max(1, min(50, len(v) // 10))
            fin = float(np.mean(v[-k:])); r0 = float(np.mean(v[:k]))
            r.update({"rew_first": r0, "rew_final": fin, "rew_max_smooth": float(s.max()),
                      "rew_iter_max_smooth": int(st[np.argmax(s)])})
            span = fin - r0
            if abs(span) > 1e-9:
                prog = (s - r0) / span
                hit = prog >= 0.9
                r["rew_iter_90pct"] = int(st[np.argmax(hit)]) if hit.any() else None
                r["rew_frac_run_90pct"] = r["rew_iter_90pct"] / max(st[-1], 1) if hit.any() else None
            gain = s.max() - r0
            r["rew_late_decline_frac"] = float((s.max() - s[-1]) / gain) if gain > 1e-9 else None
            r["rew_nonfinite"] = int((~np.isfinite(v)).sum())
            comp["rew"] = v.astype(np.float32)
        el = arr(ser, "Train/mean_episode_length")
        if el is not None:
            k = max(1, min(50, len(el) // 10))
            r["eplen_first"] = float(np.mean(el[:k, 2])); r["eplen_final"] = float(np.mean(el[-k:, 2]))
        r["loss_policy_tags"] = ";".join(sorted(t for t in ser if t.startswith(("Loss", "Policy"))))
        r["nonfinite_any"] = int(sum(int((~np.isfinite(np.array([x[2] for x in s]))).sum()) for s in ser.values()))
        rows.append(r)
        compact[r["run_id"]] = comp
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    rows.sort(key=lambda r: (r["experiment"], r["run"]))
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fnum(r.get(k)) for k in keys})
    with open(os.path.join(OUT, "runs.pkl"), "wb") as f:
        pickle.dump({"rows": rows, "compact": compact}, f)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
