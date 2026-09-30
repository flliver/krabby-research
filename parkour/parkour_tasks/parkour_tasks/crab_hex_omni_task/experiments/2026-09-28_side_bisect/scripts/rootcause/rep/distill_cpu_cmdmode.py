"""CPU representability test: can a FRESH actor of the exact deployed architecture (default PyTorch init, direction
visible in slots 8-9) fit the fast two-hand policy pi*(s) = C(s) for +y, M_y C(M_y s) for -y, by plain supervised
regression on the canonical-C eval states of both hands?  (Offline, open-loop; no sim, no GPU.)

Tracks during training, on held-out episodes: executed-action sign agreement per hand, and the student's response
to a command-only flip (rod 'mod' = phase-structured part; pi* itself scores ~0.31-0.34, the RL two-direction nets
0.07-0.17), to see whether the direction x phase interaction is learned and whether it lags the additive part.
"""
import glob, json, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from recon import *
from mirrors import build
torch.manual_seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0); np.random.seed(0)
torch.set_num_threads(8)
OUT = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
st = sorted(glob.glob(f"{OUT}/canon/C/cmd/side22_25lr__omni/seed001/*/"))[-1]
d, meta = load_stamp(st); T = Net(meta["checkpoint"]); lr, fa = build(meta)
LABS = ["lat_l_022", "lat_r_022", "lat_l_025", "lat_r_025"]
Xv, Y, H, PH, EP, RG, FL = [], [], [], [], [], [], []
for li, lab in enumerate(LABS):
    eps = episodes(d, meta, [lab], win=(-100, 500))
    for ei, (_, z, tix) in enumerate(eps):
        ar = np.arange(-100, 500); reg = np.where(ar < 0, 2, np.where(ar < 25, 3, np.where(ar < 200, 1, 0)))[::2]
        tix = tix[::2]; RG.append(reg)
        Pv = prop_series(z, meta, env_flags=(1.0, 0.0), cmd_scalar=False)
        Ps = prop_series(z, meta, env_flags=(1.0, 0.0), cmd_scalar=True)
        lin = z["root_lin_vel_b"].astype(np.float64); mass = meta["body_masses_kg"][0]
        ov = torch.tensor(obs_at(Pv, lin, mass, tix), dtype=torch.float32)
        os_ = torch.tensor(obs_at(Ps, lin, mass, tix), dtype=torch.float32)
        flag = torch.tensor(z["canon_flag"][tix].astype(bool))[:, None]; FL.append(z['canon_flag'][tix])
        with torch.no_grad():
            xc = torch.where(flag, lr.mirror_obs(os_), os_)
            mt = T.mu(xc, hist=True); mt = torch.where(flag, lr.mirror_actions(mt), mt)
            ov[:, 207:216] = T.estimator(ov[:, :75])
        Xv.append(ov); Y.append(mt); H.append(np.full(len(tix), li % 2)); PH.append(z["clock_phase"][tix]); EP.append(np.full(len(tix), ei))
Xv = torch.cat(Xv); Y = torch.cat(Y); H = np.concatenate(H); PH = np.concatenate(PH); EP = np.concatenate(EP); RG = np.concatenate(RG)
test = EP % 5 == 4; tr = ~test
print("samples", len(Y), "train", tr.sum(), "test", test.sum(), "teacher med|mu|", float(Y.abs().median()), flush=True)

class Student(nn.Module):  # same modules / sizes / default init as ActorCriticRMA's Actor (agent.yaml)
    def __init__(self):
        super().__init__()
        self.scan = nn.Sequential(nn.Linear(132, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 32), nn.Tanh())
        self.enc = nn.Sequential(nn.Linear(75, 30), nn.ELU())
        self.conv = nn.Sequential(nn.Conv1d(30, 20, 4, stride=2), nn.ELU(), nn.Conv1d(20, 10, 2, stride=1), nn.ELU(), nn.Flatten())
        self.hout = nn.Sequential(nn.Linear(30, 20), nn.ELU())
        self.bb = nn.Sequential(nn.Linear(136, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 18))
    def forward(self, o):
        n = o.shape[0]
        h = self.enc(o[:, -750:].reshape(n * 10, 75)).reshape(n, 10, -1).permute(0, 2, 1)
        lat = self.hout(self.conv(h))
        return self.bb(torch.cat([o[:, :75], self.scan(o[:, 75:207]), o[:, 207:216], lat], 1))

def cflip(o):
    of = o.clone()
    for r in range(11):
        base = 0 if r == 0 else 259 + 89 * (r - 1)
        of[:, base + 9] = -of[:, base + 9]
    return of

NB = 16
def mod_static(D, ph):
    b = np.minimum((np.mod(ph, 2 * np.pi) / (2 * np.pi) * NB).astype(int), NB - 1)
    Pm = np.stack([D[b == i].mean(0) for i in range(NB)]); st = Pm.mean(0)
    return float(np.sqrt(np.mean((Pm - st) ** 2))), float(np.sqrt(np.mean(st ** 2)))

def evaluate(model, tag, region=0):
    idx = np.where(test & (RG == region))[0]
    with torch.no_grad():
        o = Xv[idx]; m = model(o); mf = model(cflip(o))
    a = m.clamp(-1, 1).numpy(); at = Y[idx].clamp(-1, 1).numpy(); h = H[idx]
    agree = [float(np.mean(np.sign(a[h == k]) == np.sign(at[h == k]))) for k in (0, 1)]
    D = ((m.clamp(-1, 1) - mf.clamp(-1, 1)) / 2).numpy()
    rm, rs = mod_static(D[:, 6:], PH[idx])
    # odd-in-hand phase-locked component of the executed rods: student vs teacher (behavioural O1 on held-out states)
    def O1(A):
        out = []
        for k in (0, 1):
            b = np.minimum((np.mod(PH[idx][h == k], 2 * np.pi) / (2 * np.pi) * NB).astype(int), NB - 1)
            out.append(np.stack([A[h == k][b == i, 6:].mean(0) for i in range(NB)]))
        Od = (out[0] - out[1]) / 2; Od = Od - Od.mean(0); Ev = (out[0] + out[1]) / 2; Ev = Ev - Ev.mean(0)
        return float(np.sqrt(np.mean(Od ** 2))), float(np.sqrt(np.mean(Ev ** 2)))
    so1, se1 = O1(a); to1, te1 = O1(at)
    print(f"{tag}: sign agree L {agree[0]:.3f} R {agree[1]:.3f} | cmd-flip rod mod {rm:.3f} static {rs:.3f} | "
          f"rod O1 student {so1:.3f} (teacher {to1:.3f}) E1 {se1:.3f} ({te1:.3f})", flush=True)
    return dict(agree=agree, flip_mod=rm, flip_static=rs, O1=so1, E1=se1, O1_t=to1, E1_t=te1)

model = Student(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
hist = [dict(ep=0, **evaluate(model, "init"))]
tri = np.where(tr)[0]; B = 2048; t0 = time.time()
for ep in range(1, int(sys.argv[2]) + 1 if len(sys.argv) > 2 else 41):
    np.random.shuffle(tri); tot = 0
    for i in range(0, len(tri), B):
        b = tri[i:i + B]
        loss = F.mse_loss(model(Xv[b]), Y[b])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += float(loss) * len(b)
    if ep in (1, 2, 3, 5, 8, 12, 20, 30, 40, 60, 80) :
        r = evaluate(model, f"ep{ep:3d} STEADY train mse {tot / len(tri):.3f} ({time.time() - t0:.0f}s)"); r2 = evaluate(model, "      ONSET(hold 0.5-4 s)", 1); r3 = evaluate(model, "      FIRST 0.5 s of hold", 3)
        hist.append(dict(ep=ep, steady=r, onset=r2, first=r3))
json.dump(hist, open(f"distill_cpu_cmdmode_s{sys.argv[1] if len(sys.argv) > 1 else 0}.json", "w"), indent=1)
torch.save(model.state_dict(), "distilled_student_cmdmode.pt")
# gating stats of the distilled student, same metric as probe_gate.py, on held-out steady handed states
idx = np.where(test & (RG == 0))[0]
with torch.no_grad():
    o = Xv[idx]; n = o.shape[0]
    h = model.enc(o[:, -750:].reshape(n * 10, 75)).reshape(n, 10, -1).permute(0, 2, 1); lat = model.hout(model.conv(h))
    x = torch.cat([o[:, :75], model.scan(o[:, 75:207]), o[:, 207:216], lat], 1)
    z1 = model.bb[0](x).numpy(); W1 = model.bb[0].weight.detach().numpy()
Hh = H[idx]; wsd = 0.5 * (z1[Hh == 0].std(0) + z1[Hh == 1].std(0)) + 1e-9
cd = 2 * np.abs(W1[:, 9]) * 0.22; hd = np.abs(z1[Hh == 0].mean(0) - z1[Hh == 1].mean(0))
g = np.mean(((z1[Hh == 0] < 0).mean(0) > 0.8) & ((z1[Hh == 1] < 0).mean(0) < 0.2) | ((z1[Hh == 1] < 0).mean(0) > 0.8) & ((z1[Hh == 0] < 0).mean(0) < 0.2))
print("DISTILLED STUDENT gating: c-swing/spread median %.2f frac>1 %.3f | hand mean-diff/spread median %.2f frac>1 %.3f | one-hand-negative units %.3f" % (
    np.median(cd / wsd), np.mean(cd / wsd > 1), np.median(hd / wsd), np.mean(hd / wsd > 1), g))
