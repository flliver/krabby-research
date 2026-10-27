"""Reconstruct the 1149-dim omni policy observation from eval raw npz and run the actor on CPU (read-only).

Layout (live env): prop89 = head17 | joint_pos-default 24 | joint_vel*0.05 24 | last action 18 | current-sense 6;
then scan 108 (flat: clip(20.7) = 1.0), priv_explicit 9 (lin_vel_b*2, 0*6), priv_latent 53 (mass, com3, friction,
stiff ratio 24, damp ratio 24), history 10 x 89 (obs_buf rows t-10..t-1).
Network slicing (agent.yaml): prop=obs[:75], 'scan'=obs[75:207] -> scan encoder, 'priv_explicit'=obs[207:216]
(replaced by estimator(obs[:75])), 'priv_latent'=obs[216:257], hist=obs[-750:].view(10,75).
"""
import sys, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp")
import crab_hex_linkage as lk  # noqa: E402

HIP_DEF = lk.hip_default_rad(); KNEE_L = lk.knee_default_left_rad(); KNEE_R = lk.knee_default_right_rad()
NO_LOAD, GATE_FRAC, CUR_MAX = 0.1, 0.3, 1.5

def default_pos(jn):
    d = np.zeros(len(jn))
    for i, n in enumerate(jn):
        if "_Hip_Femur_" in n: d[i] = HIP_DEF
        elif "_Femur_Tibia_" in n: d[i] = KNEE_L if n[1] == "L" else KNEE_R
    return d

def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi

def euler_rp(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = np.clip(2 * (w * y - z * x), -1, 1); pitch = np.arcsin(sinp)
    return wrap(roll), wrap(pitch)

def prop_series(z, meta, env_flags=(0.0, 1.0), cmd_scalar=False, facing_clip=0.6):
    jn = meta["joint_names"]; T = z["actions"].shape[0]
    q = z["joint_pos"].astype(np.float64); dq = z["joint_vel"].astype(np.float64)
    jd = q - default_pos(jn)[None]
    cam = [i for i, n in enumerate(jn) if "_Body_CamShaft_" in n]
    jd[:, cam] = wrap(jd[:, cam])
    roll, pitch = euler_rp(z["root_quat_w"].astype(np.float64))
    cmd = z["cmd_applied"].astype(np.float64)
    head = np.zeros((T, 17))
    head[:, 0:3] = z["root_ang_vel_b"] * 0.25
    head[:, 3] = roll; head[:, 4] = pitch
    head[:, 5] = cmd[:, 2]
    fe = np.clip(z["heading_err"].astype(np.float64), -facing_clip, facing_clip)
    head[:, 6] = fe; head[:, 7] = fe
    if not cmd_scalar: head[:, 8:10] = cmd[:, 0:2]
    head[:, 10] = z["obs_cmd_vx"]
    head[:, 11] = env_flags[0]; head[:, 12] = env_flags[1]
    head[:, 13:15] = z["root_lin_vel_b"][:, :2] * 2.0
    head[:, 15:17] = z["clock_obs"]
    act = np.clip(z["actions"].astype(np.float64), -1, 1)
    last = np.vstack([np.zeros((1, 18)), act[:-1]])
    # current-sense fill: tau(t), joint_pos(t), rod speeds from the previous step's action processing
    pid = [i for i, n in enumerate(jn) if "_Hip_Femur_" in n]
    kid = [jn.index(jn[i].replace("_Hip_Femur_", "_Femur_Tibia_")) for i in pid]
    tau = z["applied_torque"].astype(np.float64)
    th = torch.tensor(q[:, pid]); tk = torch.tensor(q[:, kid])
    ma_h = lk.hip_moment_arm(th).clamp_min(1e-4).numpy(); ma_k = lk.knee_moment_arm(th, tk.abs()).clamp_min(1e-4).numpy()
    fh = np.abs(tau[:, pid]) / ma_h / lk.HIP_FORCE_N; fk = np.abs(tau[:, kid]) / ma_k / lk.KNEE_FORCE_N
    rs = z["rod_speed"].astype(np.float64); rs = np.vstack([np.zeros((1, 12)), rs[:-1]])
    gh = np.clip(np.abs(rs[:, :6]) / (GATE_FRAC * lk.HIP_SPEED_M_S), 0, 1)
    gk = np.clip(np.abs(rs[:, 6:]) / (GATE_FRAC * lk.KNEE_SPEED_M_S), 0, 1)
    ch = gh * np.clip(NO_LOAD + fh, 0, CUR_MAX); ck = gk * np.clip(NO_LOAD + fk, 0, CUR_MAX)
    fill = np.maximum(ch, ck) - 0.5
    return np.hstack([head, jd, dq * 0.05, last, fill])  # T x 89

def obs_at(P, lin_b, mass, t_idx):
    """Full obs rows for time indices t_idx from the prop series P (T x 89)."""
    n = len(t_idx); o = np.zeros((n, 1149))
    o[:, 0:89] = P[t_idx]
    o[:, 89:197] = 1.0
    o[:, 197:200] = lin_b[t_idx] * 2.0
    o[:, 206] = mass
    o[:, 211:259] = 1.0          # stiffness / damping ratios (nominal)
    for i in range(10):
        src = np.maximum(t_idx - 10 + i, 0)
        o[:, 259 + 89 * i: 259 + 89 * (i + 1)] = P[src]
    return o

class Net:
    def __init__(self, ckpt):
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.sd = {k: v.float() for k, v in d["model_state_dict"].items()}
        self.est = {k: v.float() for k, v in d["estimator_state_dict"].items()}
        self.std = self.sd["std"]
    def _mlp(self, x, pre, n, last_act=None, sd=None):
        sd = sd or self.sd
        idx = sorted({int(k[len(pre):].split(".")[0]) for k in sd if k.startswith(pre) and k.endswith("weight")})
        for j, i in enumerate(idx):
            x = F.linear(x, sd[f"{pre}{i}.weight"], sd[f"{pre}{i}.bias"])
            if j < len(idx) - 1: x = F.elu(x)
            elif last_act == "tanh": x = torch.tanh(x)
            elif last_act == "elu": x = F.elu(x)
        return x
    def estimator(self, prop75):
        return self._mlp(prop75, "estimator.", 0, sd=self.est)
    def hist_latent(self, obs):
        h = obs[:, -750:].reshape(-1, 10, 75)
        sd = self.sd; nd = h.shape[0]
        p = F.elu(F.linear(h.reshape(nd * 10, 75), sd["actor.history_encoder.encoder.0.weight"], sd["actor.history_encoder.encoder.0.bias"]))
        p = p.reshape(nd, 10, -1).permute(0, 2, 1)
        c = F.elu(F.conv1d(p, sd["actor.history_encoder.conv_layers.0.weight"], sd["actor.history_encoder.conv_layers.0.bias"], stride=2))
        c = F.elu(F.conv1d(c, sd["actor.history_encoder.conv_layers.2.weight"], sd["actor.history_encoder.conv_layers.2.bias"], stride=1))
        c = c.flatten(1)
        return F.elu(F.linear(c, sd["actor.history_encoder.linear_output.0.weight"], sd["actor.history_encoder.linear_output.0.bias"]))
    def priv_latent(self, obs):
        return self._mlp(obs[:, 216:257], "actor.priv_encoder.", 0, last_act="elu")
    def backbone_input(self, obs, hist=True, use_est=True):
        obs = obs.clone()
        if use_est: obs[:, 207:216] = self.estimator(obs[:, :75])
        scan = self._mlp(obs[:, 75:207], "actor.scan_encoder.", 0, last_act="tanh")
        lat = self.hist_latent(obs) if hist else self.priv_latent(obs)
        return torch.cat([obs[:, :75], scan, obs[:, 207:216], lat], 1)
    def mu(self, obs, hist=True, use_est=True):
        x = self.backbone_input(obs, hist, use_est)
        return self._mlp(x, "actor.actor_backbone.", 0)

def load_stamp(stamp):
    d = Path(stamp); meta = json.loads((d / "run_meta.json").read_text())
    return d, meta

def episodes(d, meta, labels, win=(200, 500)):
    hl = meta["hold_labels"]; out = []
    for e in meta["env_index_map"]:
        if e["label"] not in labels: continue
        z = np.load(d / "raw" / f"episode_{e['env_index']:02d}.npz")
        done = z["done"].astype(bool); S = int(np.argmax(done)) if done.any() else len(done)
        idx = np.where((z["segment_id"] == hl.index(e["label"])) & (np.arange(len(done)) < S))[0]
        if len(idx) < win[1]: continue
        out.append((e["label"], z, idx[0] + np.arange(win[0], win[1])))
    return out
