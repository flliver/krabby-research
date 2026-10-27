import torch
from torch.distributions import Normal
for v in [1.9, 2.0, 2.1]:
    s=torch.nn.Parameter(torch.tensor([v]))
    d=Normal(torch.zeros(1), s.clamp(0.05,2.0))
    (-0.01*d.entropy().sum()).backward()
    print(v, "grad of -0.01*entropy wrt raw std:", s.grad.item())
# Adam drift after gradient goes to zero
s=torch.nn.Parameter(torch.tensor([1.5])); opt=torch.optim.Adam([s],lr=3e-4)
for i in range(20000):
    opt.zero_grad(); d=Normal(torch.zeros(1), s.clamp(0.05,2.0)); (-0.01*d.entropy().sum()).backward(); opt.step()
    if i in (0,100,1000,1666,1700,2000,5000,19999): print(i, s.item())
