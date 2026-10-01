import os, sys, glob, numpy as np, torch
from scipy.stats import norm as N
sys.path.insert(0, os.path.dirname(__file__))
from recon import *  # noqa
RUN = os.environ["RUN"]
its = [int(x) for x in os.environ.get("ITS", "0,100,200,300,500,700,1000,1500,2000,2999").split(",")]
with torch.no_grad():
    for it in its:
        f = os.path.join(RUN, f"model_{it}.pt")
        if not os.path.exists(f): continue
        c = torch.load(f, map_location="cpu", weights_only=False)
        pol.load_state_dict(c["model_state_dict"]); est.load_state_dict(c["estimator_state_dict"])
        o = obs.clone(); o[:, 207:216] = est.estimator(o[:, :75])
        mu = pol.actor(o, True).numpy()
        s = float(c["model_state_dict"]["std"].clamp(0.05, 2.0).mean()); sraw = float(c["model_state_dict"]["std"].mean())
        zl, zu = (-1 - mu) / s, (1 - mu) / s
        pin = N.cdf(zu) - N.cdf(zl)
        interior = ((N.cdf(zu) - N.cdf(zl)) - (zu * N.pdf(zu) - zl * N.pdf(zl))) / s**2
        atoms = (N.pdf(zl) / s) ** 2 / np.maximum(N.cdf(zl), 1e-300) + (N.pdf(zu) / s) ** 2 / np.maximum(N.sf(zu), 1e-300)
        fr = (interior + atoms) * s**2
        pflip = N.cdf(-np.abs(mu) / s)
        print(f"it {it:5d} std_raw {sraw:.4f} sigma {s:.3f} | frac|mu|>1 {(np.abs(mu)>1).mean():.3f} median|mu| {np.median(np.abs(mu)):.2f} p90 {np.quantile(np.abs(mu),0.9):.2f} | P(a in +-1) {pin.mean():.3f} | Fisher cens/gauss mean {fr.mean():.3f} frac<0.05 {(fr<0.05).mean():.3f} | P(sampled sign != sign mu) mean {pflip.mean():.3f} frac<0.01 {(pflip<0.01).mean():.3f}")
