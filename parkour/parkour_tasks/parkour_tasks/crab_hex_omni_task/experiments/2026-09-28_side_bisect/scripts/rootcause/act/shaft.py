"""Cam as a hardware sign-multiplier: where on the crank circle does each shaft sit? gain g = dyaw/dshaft = K(K+cos)/(1+2Kcos+K^2) flips sign when cos < -K."""
import numpy as np
from lib import *
K = np.sin(np.radians(25.0))
def g(th): c = np.cos(th); return K * (K + c) / (1 + 2*K*c + K*K)
for key, lab in [("W","fwd"),("W","back"),("W","lat_l"),("W","lat_r"),("canon:C","lat_l_022"),("canon:C","lat_r_022"),("lr:LR4","lat_l_022"),("lr:LR4","lat_r_022")]:
    m, jn, an, E = episodes(key, [lab]); ji = {n:i for i,n in enumerate(jn)}
    TH = np.concatenate([np.stack([e["q"][:, ji[l+"_Body_CamShaft_RevoluteJoint"]] for l in LEGS], 1) for e in E])
    QD = np.concatenate([np.stack([e["qd"][:, ji[l+"_Body_CamShaft_RevoluteJoint"]] for l in LEGS], 1) for e in E])
    z = np.exp(1j * TH).mean(0); gp = (g(TH) > 0).mean(0)
    print(f"{key:9s} {lab:10s} circ-mean angle(deg) {np.round(np.degrees(np.angle(z))).astype(int)} R {np.round(np.abs(z),2)} | frac gain>0 {np.round(gp,2)} | mean spin {np.round(QD.mean(0),1)}")
