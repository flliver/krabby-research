"""CPU-only: reconstruct omni policy observations from a suite npz, load W on CPU, and measure
(1) reconstruction fidelity vs the recorded deterministic action, (2) the KL between the
rollout-time policy (estimator-substituted priv slice, hist or priv latent) and the update-time
policy (true priv slice, priv latent) that PPOWithExtractor compares.

No Isaac, no CUDA. Architecture replicated from scripts/rsl_rl/modules/*.py and loaded with
strict=True (a mismatch would raise).
"""
import glob, json, math, os, sys, re
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(4)
REPO = "/home/nickmagus/krabby/krabby-research/parkour"
sys.path.insert(0, REPO + "/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp")
import crab_hex_linkage as linkage  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else REPO + "/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt"
EVAL = sys.argv[2] if len(sys.argv) > 2 else REPO + "/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29"
N_EP = int(sys.argv[3]) if len(sys.argv) > 3 else 120
ON_FLAT = int(os.environ.get("ON_FLAT", "1"))

NUM_PROP, NUM_SCAN, NUM_PE, NUM_PL, NUM_HIST = 75, 132, 9, 41, 10
P = 89  # true per-step proprio width (omni, no extra head)
act = nn.ELU()


class StateHistoryEncoder(nn.Module):
    def __init__(self, input_size, tsteps, output_size, channel_size):
        super().__init__()
        self.tsteps = tsteps
        self.encoder = nn.Sequential(nn.Linear(input_size, 3 * channel_size), nn.ELU())
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3 * channel_size, 2 * channel_size, 4, 2), nn.ELU(),
            nn.Conv1d(2 * channel_size, channel_size, 2, 1), nn.ELU(), nn.Flatten())
        self.linear_output = nn.Sequential(nn.Linear(channel_size * 3, output_size), nn.ELU())

    def forward(self, obs):
        nd = obs.shape[0]; T = self.tsteps
        proj = self.encoder(obs.reshape([nd * T, -1]))
        out = self.conv_layers(proj.reshape([nd, T, -1]).permute((0, 2, 1)))
        return self.linear_output(out)


def mlp(dims, out_act_last=False, tanh_last=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
        elif tanh_last:
            layers.append(nn.Tanh())
        elif out_act_last:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.priv_encoder = mlp([NUM_PL, 64, 20], out_act_last=True)
        self.history_encoder = StateHistoryEncoder(NUM_PROP, NUM_HIST, 20, 10)
        self.scan_encoder = mlp([NUM_SCAN, 128, 64, 32], tanh_last=True)
        self.actor_backbone = mlp([NUM_PROP + 32 + NUM_PE + 20, 512, 256, 128, 18])

    def forward(self, obs, hist_encoding):
        scan_latent = self.scan_encoder(obs[:, NUM_PROP:NUM_PROP + NUM_SCAN])
        pe = obs[:, NUM_PROP + NUM_SCAN:NUM_PROP + NUM_SCAN + NUM_PE]
        if hist_encoding:
            lat = self.history_encoder(obs[:, -NUM_HIST * NUM_PROP:].reshape(-1, NUM_HIST, NUM_PROP))
        else:
            s = NUM_PROP + NUM_SCAN + NUM_PE
            lat = self.priv_encoder(obs[:, s:s + NUM_PL])
        return self.actor_backbone(torch.cat([obs[:, :NUM_PROP], scan_latent, pe, lat], dim=1))


class Wrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = mlp([1149, 512, 256, 128, 1])
        self.std = nn.Parameter(torch.ones(18))


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
pol = Wrap(); pol.load_state_dict(ck["model_state_dict"], strict=True); pol.eval()
est = nn.Module(); est.estimator = mlp([NUM_PROP, 128, 64, 9]); est.load_state_dict(ck["estimator_state_dict"], strict=True); est.eval()
std_param = ck["model_state_dict"]["std"]
sigma = std_param.clamp(0.05, 2.0)
print("ckpt", CKPT, "iter", ck["iter"], "std param min/max", float(std_param.min()), float(std_param.max()), "clamped", float(sigma.mean()))

meta = json.load(open(os.path.join(EVAL, "run_meta.json")))
jn = meta["joint_names"]
mass0 = meta["body_masses_kg"][0]


def default_pos(n):
    if "Hip_Femur" in n: return 0.027169998105984527
    if "Femur_Tibia" in n: return 0.31020612922871676 if n[1] == "L" else -0.31020612922871676
    return 0.0

dpos = np.array([default_pos(n) for n in jn], dtype=np.float32)
cam_ids = [i for i, n in enumerate(jn) if "CamShaft" in n]
legs = ["FL", "FR", "ML", "MR", "RL", "RR"]
pitch_ids = [jn.index(f"{l}_Hip_Femur_RevoluteJoint") for l in legs]
knee_ids = [jn.index(f"{l}_Femur_Tibia_RevoluteJoint") for l in legs]


def quat_to_rp(q):  # (w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def contact_fill(jp, tau, rod_prev):
    th = torch.tensor(jp[:, pitch_ids]); tk = torch.tensor(jp[:, knee_ids])
    ma_h = linkage.hip_moment_arm(th).clamp_min(1e-4)
    ma_k = linkage.knee_moment_arm(th, tk.abs()).clamp_min(1e-4)
    fh = torch.tensor(np.abs(tau[:, pitch_ids])) / ma_h / linkage.HIP_FORCE_N
    fk = torch.tensor(np.abs(tau[:, knee_ids])) / ma_k / linkage.KNEE_FORCE_N
    gh = (torch.tensor(np.abs(rod_prev[:, :6])) / (0.3 * linkage.HIP_SPEED_M_S)).clamp(0, 1)
    gk = (torch.tensor(np.abs(rod_prev[:, 6:])) / (0.3 * linkage.KNEE_SPEED_M_S)).clamp(0, 1)
    ch = gh * (0.1 + fh).clamp(0, 1.5); ck_ = gk * (0.1 + fk).clamp(0, 1.5)
    return (torch.maximum(ch, ck_) - 0.5).numpy()


def build_episode(d):
    T = d["actions"].shape[0]
    roll, pitch = quat_to_rp(d["root_quat_w"])
    cmd = d["cmd_applied"]
    jpd = d["joint_pos"] - dpos[None]
    jpd[:, cam_ids] = (jpd[:, cam_ids] + np.pi) % (2 * np.pi) - np.pi
    last_act = np.zeros_like(d["actions"]); last_act[1:] = np.clip(d["actions"][:-1], -1, 1)
    rod_prev = np.zeros_like(d["rod_speed"]); rod_prev[1:] = d["rod_speed"][:-1]
    cf = contact_fill(d["joint_pos"], d["applied_torque"], rod_prev)
    fe = np.clip(d["heading_err"], -0.6, 0.6)
    ob = np.concatenate([
        d["root_ang_vel_b"] * 0.25, roll[:, None], pitch[:, None], cmd[:, 2:3], fe[:, None], fe[:, None],
        cmd[:, 0:2], np.linalg.norm(cmd[:, 0:2], axis=1, keepdims=True),
        np.full((T, 1), 1.0 - ON_FLAT), np.full((T, 1), float(ON_FLAT)),
        d["root_lin_vel_b"][:, :2] * 2.0, d["clock_obs"], jpd, d["joint_vel"] * 0.05, last_act, cf], axis=1).astype(np.float32)
    assert ob.shape[1] == P, ob.shape
    hz = np.clip(d["root_pos_w"][:, 2] - d["terrain_z"].mean(1) - 0.3, -1, 1)
    heights = np.repeat(hz[:, None], 108, axis=1)
    pe = np.concatenate([d["root_lin_vel_b"] * 2.0, np.zeros((T, 6))], axis=1)
    pl = np.zeros((T, 53), dtype=np.float32); pl[:, 0] = mass0  # com filled later from estimator; friction sampled
    hist = np.zeros((T, 10 * P), dtype=np.float32)
    for t in range(T):
        rows = [ob[max(0, t - 10 + k)] if t - 10 + k >= 0 else ob[0] for k in range(10)]
        hist[t] = np.concatenate(rows)
    full = np.concatenate([ob, heights, pe, pl, hist], axis=1).astype(np.float32)
    assert full.shape[1] == 1149, full.shape
    return full


files = sorted(glob.glob(os.path.join(EVAL, "raw", "episode_*.npz")))
rng = np.random.default_rng(0)
rng.shuffle(files)
obs_l, act_l = [], []
for f in files[:N_EP]:
    d = np.load(f)
    alive = ~np.cumsum(d["done"]).astype(bool)
    full = build_episode(d)
    idx = np.arange(12, len(full))
    idx = idx[alive[idx]]
    if len(idx) == 0: continue
    idx = idx[:: 3]
    obs_l.append(full[idx]); act_l.append(d["actions"][idx])
obs = torch.tensor(np.concatenate(obs_l)); acts = torch.tensor(np.concatenate(act_l))
print("states", obs.shape[0], "from", len(obs_l), "episodes")


PE0 = NUM_PROP + NUM_SCAN  # 207
from scipy.stats import norm as _n  # noqa
with torch.no_grad():
    e = est.estimator(obs[:, :NUM_PROP])
    print("estimator out mean", [round(x, 4) for x in e.mean(0).tolist()])
    print("estimator out std ", [round(x, 4) for x in e.std(0).tolist()])
    # obs[207:216] = com(3), friction, stiff_ratio[0:5]. The constant dims: take the estimator's
    # (converged) prediction as truth; friction ~ U-bucket[0.6, 2.0] (64 buckets, startup event).
    true_pe = e.clone()
    true_pe[:, 0:3] = e[:, 0:3].mean(0)
    true_pe[:, 4:] = torch.round(e[:, 4:].mean(0))
    buckets = np.linspace(0.6, 2.0, 64)
    fr = torch.tensor(rng.choice(buckets, size=obs.shape[0]), dtype=torch.float32)
    true_pe[:, 3] = fr
    obs_true = obs.clone(); obs_true[:, PE0:PE0 + 9] = true_pe
    obs_est = obs.clone(); obs_est[:, PE0:PE0 + 9] = e
    print(f"friction: true var {fr.var():.4f}; est-vs-true MSE {((e[:,3]-fr)**2).mean():.4f}; per-9-dim MSE {((e-true_pe)**2).mean():.4f} (TB Loss/estimator ~0.017)")

    # latent: the true priv-latent input slice is unknown here (stiff/damp ratios); the history
    # encoder is trained to regress priv_encoder(true input) (Loss/hist_latent ~3e-4), so use it.
    A = pol.actor
    def mu_with(o, latent):
        sl = A.scan_encoder(o[:, NUM_PROP:NUM_PROP + NUM_SCAN])
        return A.actor_backbone(torch.cat([o[:, :NUM_PROP], sl, o[:, PE0:PE0 + 9], latent], 1))
    hl = A.history_encoder(obs[:, -NUM_HIST * NUM_PROP:].reshape(-1, NUM_HIST, NUM_PROP))
    # search the priv-latent input among {-1,0} group patterns for the one that reproduces hl
    groups = {"BH": [i for i, n in enumerate(jn) if "Body_Hip" in n], "CS": cam_ids,
              "HF": pitch_ids, "FT": knee_ids}
    best = None
    import itertools
    for sv in itertools.product([-1.0, 0.0], repeat=4):
        for dv in itertools.product([-1.0, 0.0], repeat=4):
            st = torch.zeros(24); dp = torch.zeros(24)
            for (g, ids), v in zip(groups.items(), sv): st[ids] = v
            for (g, ids), v in zip(groups.items(), dv): dp[ids] = v
            x = torch.cat([st[5:24], dp[0:22]])[None]
            pl = A.priv_encoder(x)
            r = (pl - hl).norm(dim=1).mean().item()
            if best is None or r < best[0]: best = (r, sv, dv, pl)
    print(f"priv-latent input search: best |priv(x)-hist| {best[0]:.4f} stiff(BH,CS,HF,FT)={best[1]} damp={best[2]}; |hist| mean {hl.norm(dim=1).mean():.3f}")
    pl_true = best[3].expand_as(hl)
    # stiff_ratio[0:5] implied by the best pattern (check vs estimator constants)
    mu_upd = mu_with(obs_true, pl_true)          # update(): true priv slice, priv latent
    mu_roll = mu_with(obs_est, pl_true)           # act(), non-dagger iteration
    mu_dag = mu_with(obs_est, hl)                 # act(), dagger iteration (it % 20 == 0) == deployed policy
    acts_ = acts
    err = (mu_dag - acts_)
    corr = np.corrcoef(mu_dag.flatten().numpy(), acts_.flatten().numpy())[0, 1]
    print(f"VALIDATION deployed-mu vs recorded action: corr {corr:.4f}, sign agree {(mu_dag.sign()==acts_.sign()).float().mean():.4f}, median |err| {err.abs().median():.3f} (median |a| {acts_.abs().median():.2f})")
    def kl(m1, m2, s): return ((m1 - m2) ** 2 / (2 * s ** 2)).sum(-1)
    for name, a, b in [("act(non-dagger) vs update", mu_roll, mu_upd), ("act(dagger it) vs update", mu_dag, mu_upd),
                       ("hist vs priv latent only", mu_dag, mu_roll)]:
        k = kl(a, b, sigma)
        print(f"KL {name}: mean {k.mean():.5f} median {k.median():.5f} p90 {k.quantile(0.9):.5f} p99 {k.quantile(0.99):.5f} frac>0.02 {(k>0.02).float().mean():.4f} | mean|dmu| {(a-b).abs().mean():.4f}")
    for name, mroll in [("non-dagger", mu_roll), ("dagger", mu_dag)]:
        g = torch.Generator().manual_seed(1)
        a = mroll + sigma * torch.randn(mroll.shape, generator=g)
        lr_ = torch.distributions.Normal(mu_upd, sigma).log_prob(a).sum(-1) - torch.distributions.Normal(mroll, sigma).log_prob(a).sum(-1)
        r = lr_.exp()
        print(f"ratio before any update ({name}): median {r.median():.4f} frac outside [0.8,1.2] {((r<0.8)|(r>1.2)).float().mean():.4f} p1 {r.quantile(0.01):.3f} p99 {r.quantile(0.99):.3f}")
    # Behavioural vs Gaussian KL: Fisher information of mu for the Gaussian vs the +-1 censored Gaussian
    m = mu_dag.numpy(); s = float(sigma[0])
    zl = (-1 - m) / s; zu = (1 - m) / s
    Pl = _n.cdf(zl); Pu = _n.sf(zu); phl = _n.pdf(zl); phu = _n.pdf(zu)
    # interior part: E[((a-mu)/s^2)^2 ; a in (-1,1)] = (1/s^2) * [ (Phi(zu)-Phi(zl)) - (zu*phu - zl*phl) ]
    interior = ((_n.cdf(zu) - _n.cdf(zl)) - (zu * phu - zl * phl)) / s**2
    atoms = (phl / s) ** 2 / np.maximum(Pl, 1e-300) + (phu / s) ** 2 / np.maximum(Pu, 1e-300)
    I_c = interior + atoms; I_g = 1 / s**2
    ratio = I_c / I_g
    print(f"Fisher(mu) censored/Gaussian: mean {ratio.mean():.4f} median {np.median(ratio):.5f} frac dims<0.05 {(ratio<0.05).mean():.3f}  -> behaviourally relevant share of measured KL for an isotropic dmu")
    pin = (_n.cdf(zu) - _n.cdf(zl))
    print(f"P(sampled action inside +-1): mean {pin.mean():.4f}; frac |mu|>1 {(np.abs(m)>1).mean():.4f}; median |mu| {np.median(np.abs(m)):.2f}; p90 |mu| {np.quantile(np.abs(m),0.9):.2f}")
    # effect of the action scale on KL: a KL of 0.02 at sigma 2 == sum dmu^2 = 0.16 -> rms dmu per dim
    print("KL 0.02 at sigma=2 <-> rms dmu/dim", math.sqrt(0.02 * 2 * 4 / 18))
    W0 = A.actor_backbone[0].weight
    cn = W0.norm(dim=0)
    print("backbone L0 col-norm: prop mean", round(cn[:75].mean().item(), 3), "scan-lat mean", round(cn[75:107].mean().item(), 3),
          "priv_explicit cols", [round(x, 3) for x in cn[107:116].tolist()], "latent mean", round(cn[116:].mean().item(), 3))
    o2 = obs_true.clone(); o2[:, PE0 + 3] += 0.1
    dmu = (mu_with(o2, pl_true) - mu_upd) / 0.1
    print(f"d mu / d friction: mean |.| per dim {dmu.abs().mean():.4f}; rms-norm over dims {dmu.pow(2).sum(-1).sqrt().mean():.4f}")
    # critic input scale
    print("obs abs max per block: prop", obs[:, :89].abs().max().item(), "heights", obs[:, 89:197].abs().max().item(), "mass col 206", obs[0, 206].item())
