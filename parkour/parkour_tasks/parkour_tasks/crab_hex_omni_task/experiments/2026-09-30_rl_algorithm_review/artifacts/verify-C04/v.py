"""Verify C04: re-run the impl-audit PPO re-enactment with (a) per-dim true sigma in the censored KL,
(b) several seeds, (c) the ADAPTIVE LR rule (not fixed LR), (d) alternative executed-change metrics
(stochastic E[clip(a)] change, censored TV per dim)."""
import os, sys, json
import numpy as np, torch, torch.nn as nn
from scipy.stats import norm as N
sys.path.insert(0, "/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/impl-audit")
from recon import *  # noqa
torch.manual_seed(0)
with torch.no_grad():
    e = est.estimator(obs[:, :75])
    fr = torch.tensor(rng.choice(np.linspace(0.6, 2.0, 64), size=obs.shape[0]), dtype=torch.float32)
    st = torch.zeros(24); dp = torch.zeros(24)
    bh = [i for i, n in enumerate(jn) if "Body_Hip" in n]
    st[bh] = -1.0; dp[bh] = -1.0; dp[cam_ids] = -1.0
    pl = torch.cat([torch.full((obs.shape[0], 1), mass0), e[:, 0:3].mean(0).expand(obs.shape[0], 3), fr[:, None],
                    st.expand(obs.shape[0], 24), dp.expand(obs.shape[0], 24)], 1)
    obs_t = obs.clone(); obs_t[:, 206:259] = pl
print("std raw per dim:", [round(x, 3) for x in ck["model_state_dict"]["std"].tolist()])
B = 6144
idx = torch.randperm(obs_t.shape[0])[:B]
O = obs_t[idx]
init_state = {k: v.clone() for k, v in pol.state_dict().items()}

def cens_kl(mo, mn, so, sn):
    zl_o, zu_o = (-1 - mo) / so, (1 - mo) / so
    zl_n, zu_n = (-1 - mn) / sn, (1 - mn) / sn
    Plo_o, Phi_o = N.cdf(zl_o), N.sf(zu_o); Plo_n, Phi_n = N.cdf(zl_n), N.sf(zu_n)
    Pin_o = np.clip(1 - Plo_o - Phi_o, 0, 1)
    Ea = mo * Pin_o + so * (N.pdf(zl_o) - N.pdf(zu_o))
    Ea2 = (mo**2 + so**2) * Pin_o + so * ((mo - 1) * N.pdf(zl_o) * 0 + 0)  # placeholder, general form below
    # general interior: int_{-1}^{1} p_o log(p_o/p_n) = log(sn/so)Pin + E[(a-mn)^2/(2sn^2) - (a-mo)^2/(2so^2); in]
    # E[a^2; in] = (mo^2+so^2)Pin + so*((mo-1)*pdf(zu)... ) use: E[(a-mo)^2;in] = so^2*(Pin - (zu*pdf(zu)-zl*pdf(zl)))
    Em2 = so**2 * (Pin_o - (zu_o * N.pdf(zu_o) - zl_o * N.pdf(zl_o)))  # E[(a-mo)^2; in]
    Em1 = Ea - mo * Pin_o  # E[(a-mo); in]
    En2 = Em2 + 2 * (mo - mn) * Em1 + (mo - mn) ** 2 * Pin_o  # E[(a-mn)^2; in]
    interior = np.log(sn / so) * Pin_o + En2 / (2 * sn**2) - Em2 / (2 * so**2)
    t1 = np.where(Plo_o > 0, Plo_o * (np.log(np.maximum(Plo_o, 1e-300)) - np.log(np.maximum(Plo_n, 1e-300))), 0.0)
    t2 = np.where(Phi_o > 0, Phi_o * (np.log(np.maximum(Phi_o, 1e-300)) - np.log(np.maximum(Phi_n, 1e-300))), 0.0)
    return t1 + t2 + interior

def eclip(m, s):  # E[clip(mu + s eps, -1, 1)]
    zl, zu = (-1 - m) / s, (1 - m) / s
    return -N.cdf(zl) + N.sf(zu) + m * (N.cdf(zu) - N.cdf(zl)) + s * (N.pdf(zl) - N.pdf(zu))

# (1) Monte Carlo check of cens_kl on a few (mo, mn, s)
g0 = np.random.default_rng(1)
for mo, mn, so, sn in [(0.3, 0.5, 2.0, 2.0), (9.0, 9.3, 2.0, 2.0), (1.5, 1.2, 1.6, 1.7), (-2.0, -1.0, 2.0, 1.8)]:
    a = np.clip(mo + so * g0.standard_normal(4_000_000), -1, 1)
    def logp(a, m, s):
        lo = N.logcdf((-1 - m) / s); hi = N.logsf((1 - m) / s)
        return np.where(a <= -1, lo, np.where(a >= 1, hi, N.logpdf(a, m, s)))
    mc = (logp(a, mo, so) - logp(a, mn, sn)).mean()
    print(f"MC check mo={mo} mn={mn} so={so} sn={sn}: MC {mc:.6f} formula {float(cens_kl(np.array(mo), np.array(mn), so, sn)):.6f}")

def run(lr0, n_mb, adv_kind, seed, adaptive, iters=1, epochs=5):
    pol.load_state_dict(init_state)
    opt = torch.optim.Adam(pol.parameters(), lr=lr0)
    opt.load_state_dict(ck["optimizer_state_dict"])
    lr = lr0
    for gg in opt.param_groups: gg["lr"] = lr
    g = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        mu_start = pol.actor(O, False); s_start = pol.std.clamp(0.05, 2.0).clone()
    for it in range(iters):
        with torch.no_grad():
            mu_old = pol.actor(O, False)
            s_old = pol.std.clamp(0.05, 2.0).expand_as(mu_old).clone()
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
        perm = torch.randperm(B, generator=g)
        kls, lrs = [], []
        for ep in range(epochs):
            for i in range(n_mb):
                b = perm[i * mbs:(i + 1) * mbs]
                mu = pol.actor(O[b], False)
                s = pol.std.clamp(0.05, 2.0).expand_as(mu)
                with torch.no_grad():
                    kl = torch.sum(torch.log(s / s_old[b] + 1e-5) + (s_old[b] ** 2 + (mu_old[b] - mu) ** 2) / (2 * s ** 2) - 0.5, -1).mean().item()
                kls.append(kl)
                if adaptive:
                    if kl > 0.02: lr = max(1e-5, lr / 1.5)
                    elif 0 < kl < 0.005: lr = min(1e-2, lr * 1.5)
                    for gg in opt.param_groups: gg["lr"] = lr
                lrs.append(lr)
                dist = torch.distributions.Normal(mu, s)
                ratio = torch.exp((dist.log_prob(a[b]).sum(-1) - lp_old[b]).clamp(-20, 20))
                surr = torch.max(-A[b] * ratio, -A[b] * ratio.clamp(0.8, 1.2)).mean()
                loss = surr - 0.01 * dist.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        with torch.no_grad():
            mu_new = pol.actor(O, False); s_new = pol.std.clamp(0.05, 2.0).expand_as(mu_new)
            mo = mu_old.numpy().astype(np.float64); mn = mu_new.numpy().astype(np.float64)
            so = s_old.numpy().astype(np.float64); sn = s_new.numpy().astype(np.float64)
            gk = (np.log(sn / so) + (so**2 + (mo - mn) ** 2) / (2 * sn**2) - 0.5).sum(-1).mean()
            ck_true = cens_kl(mo, mn, so, sn).sum(-1).mean()
            ck_s2 = cens_kl(mo, mn, 2.0, 2.0).sum(-1).mean()
            det = np.abs(np.clip(mn, -1, 1) - np.clip(mo, -1, 1)).mean()
            sto = np.abs(eclip(mn, sn) - eclip(mo, so)).mean()
            sign_flip = (np.sign(mn) != np.sign(mo)).mean()
        out.append(dict(it=it, kl_last_mb=kls[-1], kl_mean_mb=float(np.mean(kls)), gauss_kl_full=float(gk),
                        cens_kl_true_sigma=float(ck_true), cens_kl_sigma2=float(ck_s2),
                        share_true=float(ck_true / gk), share_s2=float(ck_s2 / gk), det_exec=float(det), stoch_exec=float(sto),
                        sign_flip=float(sign_flip), lr_mean=float(np.mean(lrs)), lr_end=lr, lr_geo=float(np.exp(np.mean(np.log(lrs)))),
                        sigma_mean=float(so.mean())))
    return out

rows = []
mode = os.environ.get("MODE", "fixed")
seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
if mode == "fixed":
    for adv in ["noise", "signal"]:
        for n_mb in [int(x) for x in os.environ.get("MBS", "16,4").split(",")]:
            for lr in [float(x) for x in os.environ.get("LRS", "1e-5,3e-5,1e-4").split(",")]:
                for sd in seeds:
                    r = run(lr, n_mb, adv, sd, False)[0]; r.update(adv=adv, n_mb=n_mb, lr0=lr, seed=sd, mode=mode)
                    rows.append(r); print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
else:
    for adv in ["noise", "signal"]:
        for n_mb in [16, 4]:
            for sd in seeds:
                rs = run(1e-5, n_mb, adv, sd, True, iters=int(os.environ.get("ITERS", "4")))
                for r in rs:
                    r.update(adv=adv, n_mb=n_mb, lr0=1e-5, seed=sd, mode=mode)
                    rows.append(r); print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(rows, open(os.environ["OUT"], "w"), indent=1)
