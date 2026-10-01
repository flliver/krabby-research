import json, numpy as np, collections
import sys
for f in sys.argv[1:]:
    d=json.load(open(f+'.json')); g=collections.defaultdict(list)
    for r in d: g[(r['adv'],r['n_mb'],r['lr'])].append(r)
    print('=====',f, 'sigma', round(d[0]['sigma_mean'],3))
    print('adv  nmb  lr   | votes(mean,min,max) | csh_last(sig)[min,max] csh_last(2.0) csh_mbmean full_csh | exec_det[min,max]  exec_stoch | kl_mean')
    for k,rs in g.items():
        a=lambda x: np.array([r[x] for r in rs])
        v=a('net_lr_steps')
        print(f"{k[0]:6s} {k[1]:2d} {k[2]:.0e} | {v.mean():6.1f} {v.min():4d} {v.max():4d} | {a('cens_share_last').mean():.3f} [{a('cens_share_last').min():.3f},{a('cens_share_last').max():.3f}] {a('cens_share_last_sig2').mean():.3f} {a('cens_share_mbmean').mean():.3f} {a('full_cens_share').mean():.3f} | {a('exec_action_change').mean():.5f} [{a('exec_action_change').min():.5f},{a('exec_action_change').max():.5f}] {a('stoch_exec_change').mean():.5f} | {a('kl_mean').mean():.4f}")
