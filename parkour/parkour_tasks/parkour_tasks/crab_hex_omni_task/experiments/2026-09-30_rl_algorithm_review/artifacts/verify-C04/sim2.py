"""CPU re-enactment of ONE PPOWithExtractor.update() on reconstructed W states (no env, no GPU).

Question: at the LR the adaptive schedule sits at (1e-5 .. 4e-5), how large is the per-minibatch
KL(old || current) that ppo_with_extractor.py:377-404 feeds to the controller, and how much of it
is behaviourally real (KL of the +-1 clipped action distribution the env actually executes)?

Faithful to the code: same minibatch partition reused across the 5 epochs (rsl_rl
mini_batch_generator draws one randperm), KL measured on each minibatch BEFORE its step with the
code's formula, clipped surrogate (0.2), entropy 0.01 with the [0.05, 2.0] std clamp, global grad
clip 1.0, Adam restored from the checkpoint's optimizer_state_dict. NOT included: value loss (the
critic has separate weights, it only enters through the global grad-norm clip), the 0.5 mirror
loss (needs the env's mirror tables), priv_reg (coef ~0 at W's counter -> 0.1, latent ~0).
Advantages are synthetic: 'noise' (no information) or 'signal' (a linear function of the
EXECUTED, clipped action plus noise).
"""
import os, sys, math, itertools
import numpy as np
import torch, torch.nn as nn
from scipy.stats import norm as N
sys.path.insert(0, os.path.dirname(__file__))
from recon import *  # noqa: F401,F403  (obs, acts, pol, est, sigma, ck, rng, jn, cam_ids, pitch_ids, knee_ids, mass0)

torch.manual_seed(0)
PE0 = 207
with torch.no_grad():
    e = est.estimator(obs[:, :75])
    fr = torch.tensor(rng.choice(np.linspace(0.6, 2.0, 64), size=obs.shape[0]), dtype=torch.float32)
    st = torch.zeros(24); dp = torch.zeros(24)
    bh = [i for i, n in enumerate(jn) if "Body_Hip" in n]
    st[bh] = -1.0; dp[bh] = -1.0; dp[cam_ids] = -1.0  # pattern recovered in mismatch_kl.py (|priv-hist| 3e-4)
    pl = torch.cat([torch.full((obs.shape[0], 1), mass0), e[:, 0:3].mean(0).expand(obs.shape[0], 3), fr[:, None],
                    st.expand(obs.shape[0], 24), dp.expand(obs.shape[0], 24)], 1)
    obs_t = obs.clone(); obs_t[:, 206:259] = pl
    lat_err = (pol.actor.priv_encoder(obs_t[:, 216:257]) - pol.actor.history_encoder(obs_t[:, -750:].reshape(-1, 10, 75))).norm(dim=1).mean()
print("check |priv latent - hist latent| on rebuilt obs:", float(lat_err))

B = 6144
idx = torch.randperm(obs_t.shape[0])[:B]
O = obs_t[idx]
init_state = {k: v.clone() for k, v in pol.state_dict().items()}


def cens_kl(mo, mn, s):
    """KL(old || new) of N(mu, s) censored to [-1, 1] (the env executes clip(a, -1, 1)), per element."""
    zl_o, zu_o = (-1 - mo) / s, (1 - mo) / s
    zl_n, zu_n = (-1 - mn) / s, (1 - mn) / s
    Plo_o, Phi_o = N.cdf(zl_o), N.sf(zu_o)
    Plo_n, Phi_n = N.cdf(zl_n), N.sf(zu_n)
    Pin_o = np.clip(1 - Plo_o - Phi_o, 0, 1)
    Ea = mo * Pin_o + s * (N.pdf(zl_o) - N.pdf(zu_o))  # E[a ; a in (-1,1)] under old
    interior = (mo - mn) / (2 * s * s) * (2 * Ea - (mo + mn) * Pin_o)
    t1 = np.where(Plo_o > 0, Plo_o * (np.log(np.maximum(Plo_o, 1e-300)) - np.log(np.maximum(Plo_n, 1e-300))), 0.0)
    t2 = np.where(Phi_o > 0, Phi_o * (np.log(np.maximum(Phi_o, 1e-300)) - np.log(np.maximum(Phi_n, 1e-300))), 0.0)
    return t1 + t2 + interior


def run(lr, n_mb, adv_kind, epochs=5, seed=0):
    pol.load_state_dict(init_state)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    opt.load_state_dict(ck["optimizer_state_dict"])
    for g in opt.param_groups: g["lr"] = lr
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mu_old = pol.actor(O, False)
        s_old = pol.std.clamp(0.05, 2.0).expand_as(mu_old)
        a = mu_old + s_old * torch.randn(mu_old.shape, generator=g)
        lp_old = torch.distributions.Normal(mu_old, s_old).log_prob(a).sum(-1)
        if adv_kind == "noise":
            A = torch.randn(B, generator=g)
        else:
            w = torch.randn(18, generator=g); w /= w.norm()
            sig = (a.clamp(-1, 1) * w).sum(-1)
            A = (sig - sig.mean()) / (sig.std() + 1e-8) + 0.5 * torch.randn(B, generator=g)
        A = (A - A.mean()) / (A.std() + 1e-8)
    mbs = B // n_mb
    SIG = s_old[0].numpy().astype(np.float64)[None, :]
    perm = torch.randperm(B, generator=g)
    kls, ckls, ckls2, ups, downs = [], [], [], 0, 0
    for ep in range(epochs):
        for i in range(n_mb):
            b = perm[i * mbs:(i + 1) * mbs]
            mu = pol.actor(O[b], False)
            s = pol.std.clamp(0.05, 2.0).expand_as(mu)
            with torch.no_grad():
                kl = torch.sum(torch.log(s / s_old[b] + 1e-5) + (s_old[b] ** 2 + (mu_old[b] - mu) ** 2) / (2 * s ** 2) - 0.5, -1).mean().item()
                ck_ = cens_kl(mu_old[b].numpy().astype(np.float64), mu.detach().numpy().astype(np.float64), SIG).sum(-1).mean()
                ck2 = cens_kl(mu_old[b].numpy().astype(np.float64), mu.detach().numpy().astype(np.float64), 2.0).sum(-1).mean()
            kls.append(kl); ckls.append(ck_); ckls2.append(ck2)
            if kl > 0.02: downs += 1
            elif 0 < kl < 0.005: ups += 1
            dist = torch.distributions.Normal(mu, s)
            ratio = torch.exp((dist.log_prob(a[b]).sum(-1) - lp_old[b]).clamp(-20, 20))
            surr = torch.max(-A[b] * ratio, -A[b] * ratio.clamp(0.8, 1.2)).mean()
            loss = surr - 0.01 * dist.entropy().sum(-1).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
    kls = np.array(kls); ckls = np.array(ckls)
    with torch.no_grad():
        mu_new = pol.actor(O, False)
        exec_change = (mu_new.clamp(-1, 1) - mu_old.clamp(-1, 1)).abs().mean().item()
        std_after = pol.std.detach().clone()
        s_new = pol.std.clamp(0.05, 2.0).expand_as(mu_new)
        gk_full = torch.sum(torch.log(s_new / s_old + 1e-5) + (s_old ** 2 + (mu_old - mu_new) ** 2) / (2 * s_new ** 2) - 0.5, -1).mean().item()
        gk_mean_only = torch.sum((mu_old - mu_new) ** 2 / (2 * s_old ** 2), -1).mean().item()
        ck_full = cens_kl(mu_old.numpy().astype(np.float64), mu_new.numpy().astype(np.float64), SIG).sum(-1).mean()
        # stochastic executed-action mean change: E[clip(mu+s*eps)] closed form
        def emean(m, sg):
            m = m.numpy().astype(np.float64); zl = (-1 - m) / sg; zu = (1 - m) / sg
            return -N.cdf(zl) + N.sf(zu) + m * (N.cdf(zu) - N.cdf(zl)) + sg * (N.pdf(zl) - N.pdf(zu))
        stoch_exec = float(np.abs(emean(mu_new, SIG) - emean(mu_old, SIG)).mean())
    return dict(lr=lr, n_mb=n_mb, mbs=mbs, adv=adv_kind, epochs=epochs,
                kl_mb_first_epoch_end=float(kls[n_mb - 1]), kl_last=float(kls[-1]), kl_mean=float(kls.mean()),
                frac_gt_0p02=float((kls > 0.02).mean()), frac_lt_0p005=float((kls < 0.005).mean()),
                net_lr_steps=int(ups - downs), cens_kl_last=float(ckls[-1]), cens_share_last=float(ckls[-1] / max(kls[-1], 1e-12)), cens_share_last_sig2=float(ckls2[-1] / max(kls[-1], 1e-12)), cens_share_mbmean=float(np.mean(np.array(ckls)[kls>1e-4] / kls[kls>1e-4])), full_gauss_kl=gk_full, full_gauss_meanonly=gk_mean_only, full_cens_kl=float(ck_full), full_cens_share=float(ck_full / max(gk_mean_only, 1e-12)), stoch_exec_change=stoch_exec, seed=seed, sigma_mean=float(SIG.mean()),
                exec_action_change=exec_change, mean_abs_dmu=float((mu_new - mu_old).abs().mean()),
                std_param_after=float(std_after.mean()))


rows = []
ADVS = os.environ.get("ADVS", "noise,signal").split(",")
MBS = [int(x) for x in os.environ.get("MBS", "16,4").split(",")]
LRS = [float(x) for x in os.environ.get("LRS", "1e-5,3e-5,1e-4,3e-4").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
for adv in ADVS:
  for n_mb in MBS:
    for lr in LRS:
        for sd in SEEDS:
            r = run(lr, n_mb, adv, seed=sd)
            rows.append(r)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
import json
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("OUT", "sim2.json")), "w"), indent=1)
