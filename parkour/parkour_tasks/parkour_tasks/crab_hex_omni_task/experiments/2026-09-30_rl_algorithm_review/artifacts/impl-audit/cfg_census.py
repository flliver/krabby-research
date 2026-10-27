import glob, os, re, json, collections, yaml
ROOT = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl"
class L(yaml.SafeLoader): pass
def _any(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode): return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)
L.add_multi_constructor("tag:yaml.org,2002:python/", _any)
L.add_multi_constructor("!", _any)
rows = []
for a in sorted(glob.glob(f"{ROOT}/*/*/params/agent.yaml")):
    run = os.path.dirname(os.path.dirname(a)); exp = os.path.basename(os.path.dirname(run))
    if exp in ("gait_eval",): continue
    try:
        ag = yaml.load(open(a), Loader=L)
    except Exception as e:
        print("ERR", a, e); continue
    envf = os.path.join(os.path.dirname(a), "env.yaml")
    txt = open(envf).read() if os.path.exists(envf) else ""
    def rx(p, cast=float):
        m = re.search(p, txt, re.M)
        return cast(m.group(1)) if m else None
    alg = ag.get("algorithm", {}) or {}
    pol = ag.get("policy", {}) or {}
    est = ag.get("estimator", {}) or {}
    sym = alg.get("symmetry_cfg") or {}
    de = ag.get("depth_encoder")
    rows.append(dict(exp=exp, run=os.path.basename(run), num_envs=rx(r"^\s{2}num_envs: (\d+)", int),
        dt=rx(r"^\s{2}dt: ([\d.e-]+)"), decimation=rx(r"^decimation: (\d+)", int), ep_s=rx(r"^episode_length_s: ([\d.e-]+)"),
        steps=ag.get("num_steps_per_env"), epochs=alg.get("num_learning_epochs"), mb=alg.get("num_mini_batches"),
        lr=alg.get("learning_rate"), sched=alg.get("schedule"), kl=alg.get("desired_kl"), clip=alg.get("clip_param"),
        ent=alg.get("entropy_coef"), vcoef=alg.get("value_loss_coef"), vclip=alg.get("use_clipped_value_loss"),
        gamma=alg.get("gamma"), lam=alg.get("lam"), gn=alg.get("max_grad_norm"), cls=alg.get("class_name"),
        dagger=alg.get("dagger_update_freq"), prs=str(alg.get("priv_reg_coef_schedual")), advmb=alg.get("normalize_advantage_per_mini_batch"),
        noise=pol.get("init_noise_std"), stdtype=pol.get("noise_std_type"), stdmax=pol.get("action_std_max"), stdreset=pol.get("action_std_reset"),
        pcls=pol.get("class_name"), ahid=str(pol.get("actor_hidden_dims")), act=pol.get("activation"),
        clip_actions=ag.get("clip_actions"), empnorm=ag.get("empirical_normalization"), maxit=ag.get("max_iterations"),
        resume=ag.get("resume"), load_run=ag.get("load_run"), ckpt=ag.get("load_checkpoint"), ras=ag.get("resume_alg_state"),
        sym_coef=sym.get("mirror_loss_coeff"), sym_func=(sym.get("data_augmentation_func") or "").split(":")[-1], sym_aug=sym.get("use_data_augmentation"),
        est_lr=est.get("learning_rate"), tw_est=est.get("train_with_estimated_states"), depth_lr=(de or {}).get("learning_rate") if isinstance(de, dict) else None,
        depth_steps=(de or {}).get("num_steps_per_env") if isinstance(de, dict) else None, seed=ag.get("seed")))
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "cfg_census.json"), "w"), indent=0, default=str)
print("runs", len(rows))
keys = ["exp","num_envs","dt","decimation","ep_s","steps","epochs","mb","lr","sched","kl","clip","ent","gamma","lam","gn","dagger","prs","noise","stdtype","stdmax","stdreset","clip_actions","empnorm","sym_coef","sym_func","est_lr","tw_est","vcoef","vclip","advmb","act","ahid","ras"]
combo = collections.Counter(tuple((k, str(r[k])) for k in keys) for r in rows)
for c, n in combo.most_common():
    print(n, dict(c))
