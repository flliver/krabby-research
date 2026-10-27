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

CKPT = os.environ.get("CKPT") or REPO + "/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt"
EVAL = os.environ.get("EVAL") or REPO + "/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29"
N_EP = int(os.environ.get("N_EP", "120"))
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


