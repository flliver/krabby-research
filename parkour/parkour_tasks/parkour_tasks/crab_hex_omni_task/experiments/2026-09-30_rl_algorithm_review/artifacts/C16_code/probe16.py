import os, sys, torch
sys.path.insert(0, "/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/impl-audit")
from recon import *  # noqa
torch.manual_seed(0)
s = 2.0
def kl(a, b): return ((a - b) ** 2 / (2 * s * s)).sum(-1).mean().item()
def rep(tag, a, b):
    print(f"{tag}: mean|dmu| {(a-b).abs().mean():.4f}  KL(s=2) {kl(a,b):.2e}  sign-agree {(a.sign()==b.sign()).float().mean():.4f}  clipped-diff {(a.clamp(-1,1)-b.clamp(-1,1)).abs().mean():.4f}")
with torch.no_grad():
    A = pol.actor
    o = obs.clone()
    est_out = est.estimator(o[:, :75])
    print("estimator out std per dim:", [round(x, 4) for x in est_out.std(0).tolist()])
    print("estimator out mean per dim:", [round(x, 4) for x in est_out.mean(0).tolist()])
    oe = o.clone(); oe[:, 207:216] = est_out
    base = A(oe, True)   # deploy path
    # 1 history latent -> 0
    hl = A.history_encoder(oe[:, -750:].reshape(-1, 10, 75))
    print("|hist latent| mean", hl.norm(dim=1).mean().item(), "std across states (mean over dims)", hl.std(0).mean().item())
    sl = A.scan_encoder(oe[:, 75:207])
    m0 = A.actor_backbone(torch.cat([oe[:, :75], sl, oe[:, 207:216], torch.zeros_like(hl)], 1))
    rep("deploy: hist latent -> 0", base, m0)
    mH = A.actor_backbone(torch.cat([oe[:, :75], sl, oe[:, 207:216], hl.mean(0, keepdim=True).expand_as(hl)], 1))
    rep("deploy: hist latent -> its mean", base, mH)
    # 2 estimator output -> its mean / -> zero
    o2 = o.clone(); o2[:, 207:216] = est_out.mean(0)
    rep("deploy: estimator out -> its mean", base, A(o2, True))
    o3 = o.clone(); o3[:, 207:216] = 0
    rep("deploy: estimator out -> 0", base, A(o3, True))
    # 3 training-rollout path: hist_encoding=False (priv latent from slice 216:257)
    pl_ = A.priv_encoder(oe[:, 216:257])
    print("|priv latent| mean", pl_.norm(dim=1).mean().item(), "std across states", pl_.std(0).mean().item(), " |priv - hist| mean", (pl_-hl).norm(dim=1).mean().item())
    tr = A(oe, False)
    rep("training path (priv latent) vs deploy path (hist latent)", base, tr)
    mp0 = A.actor_backbone(torch.cat([oe[:, :75], sl, oe[:, 207:216], torch.zeros_like(pl_)], 1))
    rep("training path: priv latent -> 0", tr, mp0)
    print("priv slice 216:257 std (recon, mean over dims)", oe[:, 216:257].std(0).mean().item())
