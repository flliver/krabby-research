import os, sys, torch
sys.path.insert(0, "/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/impl-audit")
os.environ.setdefault("N_EP", "60")
from recon import *  # noqa
s = 2.0
def rep(tag, a, b):
    print(f"{tag}: mean|dmu| {(a-b).abs().mean():.4f}  KL(s=2) {((a-b)**2/(2*s*s)).sum(-1).mean():.2e}  sign-agree {(a.sign()==b.sign()).float().mean():.4f}  clipped-diff {(a.clamp(-1,1)-b.clamp(-1,1)).abs().mean():.4f}")
with torch.no_grad():
    A = pol.actor; o = obs.clone(); e = est.estimator(o[:, :75]); o[:, 207:216] = e
    for path in (True, False):
        lo = o.clone(); lo[:, 210] = 0.6; hi = o.clone(); hi[:, 210] = 2.0
        rep(f"hist_encoding={path}: friction slot 0.6 vs 2.0", A(lo, path), A(hi, path))
