"""PLAN step 5 (GPU, Kit): the TRAINING-env smoke on the full rough mix. WRITTEN, NOT YET RUN (the queue runs it:
``rt_queue.py --block smoke`` -> ``rt_kit_run.sh <env file> rt_kit_train_smoke.py ...``, one Isaac process at a time,
MemoryMax 48G, SIGTERM-only timeout, the env composed ONLY from the env file, every parent KRABBY_* unset).

Builds ``--task`` (Isaac-Crab-Hex-Omni-Teacher-v0) from the launch env (``--mode rt``: envs/RT_smoke_train.env =
RT_train.env + the smoke-only KRABBY_EPISODE_S; ``--mode ref``: envs/RT_smoke_ref.env, the knobs-off reference =
TDR's training env + the same KRABBY_EPISODE_S), ``--num-envs`` 400 (20 columns x 20), a FIXED ``env.cfg.seed``,
loads TDR's weights through the repo's runner (``make_on_policy_runner`` + ``runner.load``; the teacher's inference
path: estimator into the priv_explicit slice, ``policy(obs, hist_encoding=True)``) and steps
``--episodes-mult`` (2.5) x the max episode length, so every column resets at least twice. Then a contact-level slide
test with the policy idle. Records (each check with its numbers; ``pass`` = their conjunction; mode ref asserts only
the mode-independent ones):

* modules / env: PYTHONPATH = this worktree, parkour_tasks / parkour_isaaclab / runner_factory under it, the process
  KRABBY_* = the launch env file's;
* rt: every kind builds -- the realised column count per kind = the 20-column mix (rt_spec.TRAIN_MIX), env share per
  kind, every column >= 2 envs;
* rt: every shape's mu per env: static = dynamic, restitution 0, one value per env, = env._krabby_rt_mu_eff = the
  contract's keyed draw (rt_friction.contract_mu recomputed); gravel envs U(0.2, 0.5), the others (b + 1) / 2 of the
  64 buckets of U(0.6, 2.0); ref: TDR's shipped event, one bucket value b per env in [0.6, 2.0];
* the privileged friction input (policy obs col 210 = prop 89 + scan 108 + priv_explicit 9 + mass/com 4) = 2 mu_eff -
  1 (rt) / the raw shape mu (ref), cross-checked against the obs term's own ``_get_priv_latent()`` slice;
* rt: a SLIDE test (measured, not recomputed: the PhysX combine priority ground multiply over the robot's average):
  after a full reset and a 1 s settle with zero actions (joints held at the default pose), the gravel env with the
  lowest mu and the non-gravel env with the highest mu get a horizontal root velocity (1.0 / 2.0 m/s); the speed's
  least-squares deceleration over the sliding phase gives mu_meas = -a / g (rt_smoke_check.slide_verdict);
* rt: per-kind telemetry keys in ``extras["log"]`` (site_level / site_cnt_end / site_cnt_failed per kind present in
  the grid; the ledger's site_fail_frac once a kind ended an episode);
* the policy obs width 1149;
* per kind: root height above the local surface (site_raster.surface_under) at the first step after each reset within
  +-0.05 m of 1.085 m (plus the placement right after the reset, report), and the count of episodes terminating
  within their first 2 s (``early_by_kind``; rt_smoke_check.py train compares it with the ref run's);
* every column saw >= 1 reset (resets per column recorded).

    <isaac python> rt_kit_train_smoke.py --headless --mode rt|ref --checkpoint TDR [--num-envs 400] [--seed 19]
        [--episodes-mult 2.5] --out JSON
Writes ``{..., "pass": bool, "complete": true}`` (complete LAST); exit 0 when written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
W = Path(__file__).resolve().parents[7]
parser = argparse.ArgumentParser(description="Rough-terrain PLAN step 5: the training-env smoke.")
parser.add_argument("--task", default="Isaac-Crab-Hex-Omni-Teacher-v0")
parser.add_argument("--mode", choices=("rt", "ref"), required=True)
parser.add_argument("--num-envs", type=int, default=400)
parser.add_argument("--seed", type=int, default=19)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes-mult", type=float, default=2.5)
parser.add_argument("--settle-steps", type=int, default=50)
parser.add_argument("--slide-steps", type=int, default=60)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
LAUNCH_KRABBY = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
LAUNCH_PYTHONPATH = os.environ.get("PYTHONPATH", "")          # before Kit (it may extend it)
app = AppLauncher(args).app

# the harness's sys.path surgery: parkour/scripts must not shadow pip rsl_rl; `scripts.rsl_rl.*` from parkour/
for _p in (str(W / "parkour" / "scripts"), str(W / "parkour" / "scripts" / "rsl_rl")):
    while _p in sys.path:
        sys.path.remove(_p)
for _p in (str(W), str(W / "parkour"), str(W / "parkour" / "parkour_tasks"), str(HERE)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import parkour_isaaclab  # noqa: E402
import parkour_tasks  # noqa: E402,F401  (registers the gym ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from parkour_tasks.crab_hex_omni_task.mdp import rt_friction, site_raster  # noqa: E402
from scripts.rsl_rl import runner_factory  # noqa: E402
from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner  # noqa: E402
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402

import rt_machinery as M  # noqa: E402
import rt_smoke_check as C  # noqa: E402

OBS_WIDTH = 1149
FRICTION_COL = 89 + 108 + 9 + 4
PRIV_LATENT = slice(89 + 108 + 9, 89 + 108 + 9 + 53)
MU_TOL = 1e-5
LOG_PREFIX = "Metrics/base_parkour/"


def out_write(out: dict) -> None:
    out.pop("complete", None)
    out["complete"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"[rt smoke] wrote {args.out}", flush=True)


def check(out: dict, name: str, ok: bool, **info) -> None:
    out["checks"][name] = {"pass": bool(ok), **info}
    print(f"[rt smoke] {'PASS' if ok else 'FAIL'} {name} {json.dumps(info, default=float)[:600]}", flush=True)


def under_w(p) -> bool:
    return bool(p) and str(Path(p).resolve()).startswith(str(W) + os.sep)


def tile_kind_names(u) -> tuple[np.ndarray, list]:
    """``(per-column kind name array (C,), per-env kind names (N,))``: the site kinds by sub-terrain key (rt), else
    the generator's sub-terrain names (ref)."""
    gen = u.scene.terrain.terrain_generator_class
    names = np.asarray(gen.terrain_names)
    names = names[..., 0] if names.ndim == 3 else names
    lut = dict(zip(site_raster.TILE_SUBTERRAIN_KEYS, site_raster.TILE_KINDS))
    col = []
    for c in range(names.shape[1]):
        uniq = sorted(set(str(x) for x in names[:, c]))
        col.append(lut.get(uniq[0], uniq[0]) if len(uniq) == 1 else "+".join(uniq))
    types = u.scene.terrain.terrain_types.detach().cpu().numpy().astype(int)
    return np.asarray(col), [col[t] for t in types]


def main() -> int:
    t0 = time.time()
    out = {"tool": "rt_kit_train_smoke.py", "mode": args.mode, "task": args.task, "num_envs": args.num_envs,
           "seed": args.seed, "checkpoint": args.checkpoint, "launch_krabby": LAUNCH_KRABBY, "checks": {}}
    mods = {"parkour_tasks": parkour_tasks.__file__, "parkour_isaaclab": parkour_isaaclab.__file__,
            "runner_factory": runner_factory.__file__, "rt_friction": rt_friction.__file__}
    pp = LAUNCH_PYTHONPATH
    check(out, "modules_and_pythonpath_under_worktree", all(under_w(v) for v in mods.values()) and pp == M.PYTHONPATH_W,
          modules=mods, pythonpath=pp)
    want_file = M.krabby_of(M.SMOKE_TRAIN_ENV if args.mode == "rt" else M.SMOKE_REF_ENV)
    check(out, "process_krabby_env_is_the_env_file", LAUNCH_KRABBY == want_file,
          differ=sorted(k for k in set(LAUNCH_KRABBY) | set(want_file) if LAUNCH_KRABBY.get(k) != want_file.get(k)))

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = ParkourRslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped
    dev = u.device
    n = int(u.num_envs)
    runner = make_on_policy_runner(env, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=dev)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=dev)
    estimator = runner.get_estimator_inference_policy(device=dev)
    est = agent_cfg.to_dict()["estimator"]
    n_prop, n_scan, n_pe = int(est["num_prop"]), int(est["num_scan"]), int(est["num_priv_explicit"])
    asset = u.scene["robot"]
    dt = float(u.step_dt)
    max_len = int(u.max_episode_length)
    out.update(step_dt=dt, max_episode_length=max_len, episode_length_s=float(env_cfg.episode_length_s))

    # ---- grid: kinds, columns, env share
    col_kind, env_kind = tile_kind_names(u)
    types = u.scene.terrain.terrain_types.detach().cpu().numpy().astype(int)
    per_col = np.bincount(types, minlength=len(col_kind))
    out["columns"] = {"kinds": col_kind.tolist(), "envs_per_column": per_col.tolist()}
    if args.mode == "rt":
        mix = M.rt_spec().TRAIN_MIX
        got = {k: int((col_kind == k).sum()) for k in sorted(set(col_kind.tolist()))}
        check(out, "every_kind_builds_and_column_count_is_the_mix", got == dict(sorted(mix.items())) and
              len(col_kind) == sum(mix.values()), got=got, want=mix)
        share = {k: sum(1 for e in env_kind if e == k) / n for k in sorted(set(env_kind))}
        want_share = {k: v / sum(mix.values()) for k, v in mix.items()}
        check(out, "env_share_per_kind", all(abs(share.get(k, 0.0) - want_share[k]) <= 1.0 / n * len(col_kind)
                                             for k in want_share), share=share, want=want_share)
    check(out, "every_column_has_two_envs", int(per_col.min()) >= 2, min=int(per_col.min()))

    # ---- materials and mu
    mats = asset.root_physx_view.get_material_properties().detach().cpu().double()          # (N, S, 3)
    mu_s, mu_d, rest = mats[..., 0], mats[..., 1], mats[..., 2]
    one = bool(torch.equal(mu_s, mu_s[:, :1].expand_as(mu_s)))
    check(out, "every_shape_one_mu_static_eq_dynamic_restitution0", one and bool(torch.equal(mu_s, mu_d))
          and bool((rest == 0).all()), n_shapes=int(mats.shape[1]))
    mu = mu_s[:, 0].numpy()
    pe = getattr(env_cfg.events, "physics_material", None)
    params = dict(getattr(pe, "params", {}) or {})
    fr = tuple(float(x) for x in params.get("friction_range", params.get("static_friction_range", (0.6, 2.0))))
    nb = int(params.get("num_buckets", 64))
    out["material_event"] = {"func": getattr(getattr(pe, "func", None), "__name__", str(getattr(pe, "func", None))),
                             "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()
                                        if k != "asset_cfg"}}
    if args.mode == "rt":
        attr = getattr(u, rt_friction.RT_MU_ATTR, None)
        grav = getattr(u, rt_friction.RT_GRAVEL_ATTR, None)
        gr = tuple(float(x) for x in params.get("gravel_range", (0.2, 0.5)))
        gmask = np.asarray([k == "gravel" for k in env_kind])
        check(out, "contract_event_armed", out["material_event"]["func"] == "rt_friction_contract" and attr is not None
              and grav is not None and bool(np.array_equal(np.asarray(grav), gmask)), func=out["material_event"]["func"])
        want = rt_friction.contract_mu(env_cfg.seed, np.arange(n), gmask, fr, nb, gr)
        buckets = 0.5 * (rt_friction.bucket_values(env_cfg.seed, fr, nb) + 1.0)
        nong = ~gmask
        in_b = np.min(np.abs(mu[nong, None] - buckets[None, :]), axis=1) <= MU_TOL if nong.any() else np.array([True])
        check(out, "mu_equals_contract_draw", attr is not None and float(np.abs(mu - want).max()) <= MU_TOL
              and float(np.abs(np.asarray(attr) - want).max()) <= 1e-9, max_err=float(np.abs(mu - want).max()))
        check(out, "gravel_mu_in_range", bool(gmask.any()) and float(mu[gmask].min()) >= gr[0] - MU_TOL
              and float(mu[gmask].max()) <= gr[1] + MU_TOL, n=int(gmask.sum()), min=float(mu[gmask].min()),
              max=float(mu[gmask].max()), range=gr)
        check(out, "others_mu_are_tdr_buckets", bool(in_b.all()), n=int(nong.sum()),
              min=float(mu[nong].min()), max=float(mu[nong].max()), mean=float(mu[nong].mean()),
              want_range=[(fr[0] + 1) / 2, (fr[1] + 1) / 2])
        out["mu_by_kind"] = {k: [float(mu[[e == k for e in env_kind]].min()), float(mu[[e == k for e in env_kind]].max())]
                             for k in sorted(set(env_kind))}
    else:
        check(out, "ref_shipped_event_mu_in_range", float(mu.min()) >= fr[0] - MU_TOL and float(mu.max()) <= fr[1] + MU_TOL,
              min=float(mu.min()), max=float(mu.max()), range=fr)

    # ---- the rollout
    obs, _ = env.get_observations()
    check(out, "obs_width_1149", int(obs.shape[-1]) == OBS_WIDTH, width=int(obs.shape[-1]))
    want_obs = (2 * mu - 1) if args.mode == "rt" else mu
    err = float(np.abs(obs[:, FRICTION_COL].detach().cpu().double().numpy() - want_obs).max())
    term_err = None
    for tc in u.observation_manager._group_obs_term_cfgs["policy"]:                       # smoke-only private
        if hasattr(tc.func, "_get_priv_latent") and not isinstance(tc.func, type):
            pl = tc.func._get_priv_latent().detach().cpu().double()
            term_err = float((pl[:, 4] - torch.as_tensor(want_obs)).abs().max())
            break
    check(out, "friction_input", err <= MU_TOL and (term_err is None or term_err <= MU_TOL), obs_col=FRICTION_COL,
          max_err=err, term_err=term_err, mapping="2 mu_eff - 1" if args.mode == "rt" else "raw shape mu")

    steps = int(args.episodes_mult * max_len) + 5
    ep_start = np.zeros(n, dtype=np.int64)
    ep_kind = list(env_kind)
    pending = np.zeros(n, dtype=bool)               # a reset happened last step: read the first step after it
    z_reset, z_first, early = {}, {}, {}
    resets_per_col = np.zeros(len(col_kind), dtype=np.int64)
    log_keys = set()

    def height_above(ids) -> np.ndarray:
        """Root z above the nominal surface under the root (the lookups take every env: own-tile frame)."""
        p = asset.data.root_pos_w
        s = site_raster.surface_under(u, p[:, 0], p[:, 1])
        return (p[:, 2] - s).detach().cpu().double().numpy()[np.asarray(ids, dtype=np.int64)]

    pending[:] = True                               # the startup reset: its first step is read too

    torch.set_grad_enabled(False)
    for t in range(steps):
        obs[:, n_prop + n_scan:n_prop + n_scan + n_pe] = estimator.inference(obs[:, :n_prop])
        act = policy(obs, hist_encoding=True)
        obs, _rew, dones, extras = env.step(act)
        log_keys |= set((extras.get("log") or {}).keys())
        if pending.any():
            ids = np.nonzero(pending)[0]
            for i, z in zip(ids, height_above(ids)):
                z_first.setdefault(ep_kind[i], []).append(float(z))
            pending[:] = False
        d = dones.detach().cpu().numpy().astype(bool)
        if d.any():
            to = extras.get("time_outs")
            to = to.detach().cpu().numpy().astype(bool) if to is not None else np.zeros(n, bool)
            ids = np.nonzero(d)[0]
            for i in ids:
                k = ep_kind[i]
                rec = early.setdefault(k, {"episodes": 0, "early": 0, "time_outs": 0})
                rec["episodes"] += 1
                rec["time_outs"] += int(to[i])
                if not to[i] and (t + 1 - ep_start[i]) * dt < M.EARLY_S:
                    rec["early"] += 1
            types_now = u.scene.terrain.terrain_types.detach().cpu().numpy().astype(int)
            for i, z in zip(ids, height_above(ids)):
                resets_per_col[types_now[i]] += 1
                ep_start[i] = t + 1
                ep_kind[i] = col_kind[types_now[i]]
                z_reset.setdefault(ep_kind[i], []).append(float(z))
            pending[ids] = True
    out["early_by_kind"] = early
    out["resets_per_column"] = resets_per_col.tolist()
    check(out, "every_column_resets", int(resets_per_col.min()) >= 1, min=int(resets_per_col.min()))
    spawn = {k: C.within_spawn(v) for k, v in sorted(z_first.items())}
    out["spawn_reset_placement"] = {k: C.within_spawn(v) for k, v in sorted(z_reset.items())}
    check(out, "spawn_height_first_step_per_kind", bool(spawn) and all(s["pass"] for s in spawn.values()),
          per_kind={k: {kk: s[kk] for kk in ("n", "n_out", "min", "max")} for k, s in spawn.items()})
    if args.mode == "rt":
        present = sorted(set(col_kind.tolist()))
        need = [f"{LOG_PREFIX}{stem}/{k}" for k in present for stem in ("site_level", "site_cnt_end", "site_cnt_failed")]
        need += [f"{LOG_PREFIX}site_fail_frac/{k}" for k in present if (early.get(k) or {}).get("episodes")]
        miss = [k for k in need if k not in log_keys]
        check(out, "per_kind_telemetry_keys_logged", not miss, missing=miss[:20], n_need=len(need))
        out["site_log_keys"] = sorted(k for k in log_keys if k.startswith(LOG_PREFIX + "site_"))

        # ---- the slide test (policy idle, joints held at the default pose)
        env.reset()
        zero = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
        for _ in range(args.settle_steps):
            env.step(zero)
        col_kind2, env_kind2 = tile_kind_names(u)
        g_ids = [i for i in range(n) if env_kind2[i] == "gravel"]
        o_ids = [i for i in range(n) if env_kind2[i] != "gravel"]
        lo_env = min(g_ids, key=lambda i: mu[i]) if g_ids else None
        hi_env = max(o_ids, key=lambda i: mu[i]) if o_ids else None
        slide = {}
        for tag, i, v0 in (("low_mu_gravel", lo_env, 1.0), ("high_mu", hi_env, 2.0)):
            if i is None:
                slide[tag] = {"pass": False, "why": "no env"}
                continue
            ids = torch.tensor([i], device=dev)
            vel = asset.data.root_vel_w[ids].clone()
            yaw_q = asset.data.root_quat_w[ids]
            vel[:, :] = 0.0
            vel[:, 0] = v0
            asset.write_root_velocity_to_sim(vel, env_ids=ids)
            ts, vs = [], []
            for k in range(args.slide_steps):
                env.step(zero)
                v = asset.data.root_lin_vel_w[i, :2]
                ts.append((k + 1) * dt)
                vs.append(float(torch.linalg.norm(v)))
            fit = C.fit_decel(ts, vs)
            slide[tag] = {"env": int(i), "kind": env_kind2[i], "v0": v0, "fit": fit, "speeds": vs[:20],
                          "quat0": yaw_q.detach().cpu().tolist(),
                          **C.slide_verdict(fit.get("mu_meas"), float(mu[i]), low=(tag == "low_mu_gravel"))}
        out["slide"] = slide
        check(out, "slide_contact_mu", all(s.get("pass") for s in slide.values()),
              **{k: {kk: s.get(kk) for kk in ("env", "mu_eff", "mu_meas", "average_combine_would_read")}
                 for k, s in slide.items()})
    out["pass"] = all(c["pass"] for c in out["checks"].values())
    out["wall_s"] = time.time() - t0
    out_write(out)
    print(f"[rt smoke] mode={args.mode} pass={out['pass']}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        app.close()
    sys.exit(rc)
