import os, sys, torch
sys.path.insert(0, "/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/impl-audit")
from recon import *  # noqa
with torch.no_grad():
    A = pol.actor
    o = obs.clone(); e = est.estimator(o[:, :75]); o[:, 207:216] = e
    print("estimator output std per dim", e.std(0).numpy().round(4))
    print("true obs[207:216] std per dim (recon)", obs[:, 207:216].std(0).numpy().round(4))
    sl = A.scan_encoder(o[:, 75:207])
    hl = A.history_encoder(o[:, -750:].reshape(-1, 10, 75))
    pl_ = A.priv_encoder(o[:, 216:257])
    print("|hist lat|", hl.norm(dim=1).mean().item(), "hist lat std over states", hl.std(0).mean().item(), "|priv lat|", pl_.norm(dim=1).mean().item(), "priv lat std", pl_.std(0).mean().item())
    def mu(pe, lat): return A.actor_backbone(torch.cat([o[:, :75], sl, pe, lat], 1))
    m1 = mu(e, hl)
    for name, m2 in [("hist->0", mu(e, torch.zeros_like(hl))), ("hist->mean", mu(e, hl.mean(0, keepdim=True).expand_as(hl))), ("est->mean", mu(e.mean(0, keepdim=True).expand_as(e), hl)), ("est->0", mu(torch.zeros_like(e), hl))]:
        d = m2 - m1
        c1 = m1.clamp(-1, 1); c2 = m2.clamp(-1, 1)
        print(f"{name}: mean|dmu| {d.abs().mean():.5f} KL@2.0 {((d**2)/(8)).sum(-1).mean():.2e} KL@0.3 {((d**2)/(2*0.09)).sum(-1).mean():.2e} mean|d clipped action| {(c2-c1).abs().mean():.5f}")
