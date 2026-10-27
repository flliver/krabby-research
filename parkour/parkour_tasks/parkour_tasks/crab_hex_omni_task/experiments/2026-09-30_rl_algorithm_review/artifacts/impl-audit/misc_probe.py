import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from recon import *  # noqa
with torch.no_grad():
    o = obs.clone(); o[:, 207:216] = est.estimator(o[:, :75])
    A = pol.actor
    sl = A.scan_encoder(o[:, 75:207])
    hl = A.history_encoder(o[:, -750:].reshape(-1, 10, 75))
    def mu_lat(lat): return A.actor_backbone(torch.cat([o[:, :75], sl, o[:, 207:216], lat], 1))
    m1 = mu_lat(hl); m0 = mu_lat(torch.zeros_like(hl))
    s = 2.0
    kl = ((m1 - m0) ** 2 / (2 * s * s)).sum(-1)
    print(f"latent ablation (hist latent -> 0): mean|dmu| {(m1-m0).abs().mean():.4f}, KL mean {kl.mean():.5f}, sign agreement {(m1.sign()==m0.sign()).float().mean():.4f}; |hist latent| mean {hl.norm(dim=1).mean():.4f}")
    # scan-slice ablation: which parts of the 'scan' slice matter: last-8 actions+contact (75:89), heights (89:197), lin vel (197:200), mass(206)
    for name, sl_ in [("actions8+contact6 [75:89]", slice(75, 89)), ("heights [89:197]", slice(89, 197)), ("true base lin vel [197:200]", slice(197, 200))]:
        o2 = o.clone(); o2[:, sl_] = o2[:, sl_].mean(0)
        m2 = A.actor_backbone(torch.cat([o2[:, :75], A.scan_encoder(o2[:, 75:207]), o2[:, 207:216], hl], 1))
        print(f"replace {name} by its mean: mean|dmu| {(m2-m1).abs().mean():.3f}, sign agreement {(m2.sign()==m1.sign()).float().mean():.4f}")
    V = pol.critic(o).squeeze(-1)
    print(f"critic V on W eval states: mean {V.mean():.3f} std {V.std():.3f} p5 {V.quantile(0.05):.3f} p95 {V.quantile(0.95):.3f}")
    W0 = pol.critic[0].weight
    pre_mass = (W0[:, 206] * 158.757).abs()
    pre_all = (o @ W0.T).abs()
    print(f"critic L0: |w_mass*158.76| mean {pre_mass.mean():.3f} vs |total preact| mean {pre_all.mean():.3f}; |bias| mean {pol.critic[0].bias.abs().mean():.3f}")
    Wa = A.scan_encoder[0].weight
    print(f"scan-encoder L0: |w_mass*158.76| mean {(Wa[:, 131] * 158.757).abs().mean():.3f} vs |preact| mean {(o[:, 75:207] @ Wa.T).abs().mean():.3f}")
