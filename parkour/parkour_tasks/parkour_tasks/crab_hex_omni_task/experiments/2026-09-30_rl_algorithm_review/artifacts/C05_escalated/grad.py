import torch
from torch.distributions import Normal
torch.manual_seed(0)
for raw in [1.5, 1.9, 2.0, 2.0002, 2.1]:
    std = torch.nn.Parameter(torch.full((18,), raw))
    mean = torch.zeros(4,18)
    d = Normal(mean, std.expand_as(mean).clamp(0.05, 2.0))
    a = d.sample()
    ent = d.entropy().sum(-1).mean()
    lp = d.log_prob(a).sum(-1).mean()
    g_ent = torch.autograd.grad(-0.01*ent, std, retain_graph=True)[0].mean().item()
    g_lp = torch.autograd.grad(lp, std)[0].mean().item()
    print(f"raw={raw}: d(-0.01*H)/dstd={g_ent:+.5f}  d(logprob)/dstd={g_lp:+.5f}")
