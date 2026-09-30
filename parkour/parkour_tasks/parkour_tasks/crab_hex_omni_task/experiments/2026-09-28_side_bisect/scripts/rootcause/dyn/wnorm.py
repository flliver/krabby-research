import torch, glob, re, json, sys
sys.path.insert(0,'.')
from common import ARM, FAST, SLOW
R="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
out={}
for x in FAST+SLOW:
    d=glob.glob(R+f"2026-09-*_{x}_s1_A15pB")[0]
    res={}
    for it in [0,100,200,300,400,500,600,800,1000,1500,2000,2999]:
        p=f"{d}/model_{it}.pt"
        try: sd=torch.load(p,map_location='cpu',weights_only=False)['model_state_dict']
        except Exception: continue
        W=sd['actor.actor_backbone.6.weight']; b=sd['actor.actor_backbone.6.bias']
        # also dims: 18 x 128
        res[it]=dict(Wfro=float(W.norm()), rowmean=float(W.norm(dim=1).mean()), bias_abs=float(b.abs().mean()),
                     W0=float(sd['actor.actor_backbone.0.weight'].norm()), W2=float(sd['actor.actor_backbone.2.weight'].norm()), W4=float(sd['actor.actor_backbone.4.weight'].norm()))
    out[x]=res
    print(ARM[x], ' '.join(f"{it}:{v['rowmean']:.2f}" for it,v in res.items()), flush=True)
json.dump(out,open('wnorm.json','w'))
