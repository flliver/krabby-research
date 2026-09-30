import json, numpy as np, pickle
a = json.load(open('all.json'))
cls = {'O':'S','C':'F','C_s2':'F','R':'S','C_mo':'I','C_mo_s2':'F','C_mo_s1':'F','C_band':'S','C_rsi':'S','LRfa':'S','LR4':'S','LR4_dwell':'D','C_lr':'S','C_lr_10':'S',
       'W_suite':'S','W_harness':'S','side_s3_3000':'F','side_s2_3000':'F','side_s3_4999':'F','side_s2_4999':'F'}
keys = ['sweep','lean_abs','lean_absabs','frac_lean015','cad','stall','fall_hold','feet_lean_rho','lean_sweep_rho']
rows = []
for k, v in a.items():
    if k not in cls: continue
    for l, r in v['labels'].items():
        r = dict(r); r['ratio'] = r['feet_ss'] / (r['cad'] * r['sweep']); r['arm'] = k; r['lab'] = l; r['cls'] = cls[k]; rows.append(r)
keys.append('ratio')
for key in keys:
    F = [r[key] for r in rows if r['cls']=='F']; S = [r[key] for r in rows if r['cls'] in 'SD']; I=[r[key] for r in rows if r['cls']=='I']
    F=np.array(F,float); S=np.array(S,float)
    print(f"{key:15s} FAST [{np.nanmin(F):.3f},{np.nanmax(F):.3f}] med {np.nanmedian(F):.3f}  SLOW/DEAD [{np.nanmin(S):.3f},{np.nanmax(S):.3f}] med {np.nanmedian(S):.3f}  INT {np.round(I,3)}")
print()
for r in rows: print(f"{r['arm']:13s} {r['lab']:10s} {r['cls']} ratio {r['ratio']:.2f} rho(lean,fss) {r['feet_lean_rho']:+.2f} rho(lean,sweep) {r['lean_sweep_rho']:+.2f} walkers {r['walkers_feet_ss']:.3f}")
