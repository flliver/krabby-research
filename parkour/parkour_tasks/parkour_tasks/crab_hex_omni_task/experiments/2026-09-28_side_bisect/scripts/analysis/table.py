import json
a=json.load(open('all.json'))
meta={'O':('3','SLOW','X01'),'C':('3','FAST','X02'),'C_s2':('2','FAST','X07'),'R':('3','SLOW','X03'),'C_mo':('3','INTERMEDIATE','X04'),
'C_mo_s2':('2','FAST','X08'),'C_mo_s1':('1','FAST','X09'),'C_band':('3','SLOW','X05'),'C_rsi':('3','SLOW','X06'),'LRfa':('3','SLOW','X10'),
'LR4':('3','SLOW','X11'),'LR4_dwell':('3','DEAD','X12'),'C_lr':('3','SLOW','X14'),'C_lr_10':('3','SLOW','X15'),
'W_harness':('-','SLOW (ref)',None),'W_suite':('-','SLOW (ref)',None),'side_s3_3000':('3','FAST (ref)',None),'side_s2_3000':('2','FAST (ref)',None),
'side_s3_4999':('3','FAST (ref)',None),'side_s2_4999':('2','FAST (ref)',None)}
tr={'X01':(0.564,0.568),'X02':(0.671,0.798),'X03':(0.473,0.462),'X04':(0.654,0.773),'X05':(0.561,0.579),'X06':(0.498,0.536),'X07':(0.702,0.746),
'X08':(0.700,0.870),'X09':(0.765,0.853),'X10':(0.536,0.523),'X11':(0.571,0.598),'X12':(0.491,0.482),'X14':(0.524,0.528),'X15':(0.538,0.570)}
print("| Arm | seed | hand | class | n 0.22/0.25 | feet_ss 0.22 [IQR] | feet_ss 0.25 [IQR] | lean_abs (m) | lean>0.15 | sweep (m) | cadence (Hz) | stall | fall in hold | early read | final tr |")
print("|"+"---|"*15)
for k,(seed,cls,x) in meta.items():
    L=a[k]['labels']
    hands=['l','r'] if any(l.startswith('lat_r') for l in L) else ['l']
    for h in hands:
        if k=='W_suite':
            ls=[f'lat_{h}']
        else:
            ls=[f'lat_{h}_022',f'lat_{h}_025']
        rs=[L[l] for l in ls]
        j=lambda f: " / ".join(f(r) for r in rs)
        fss=[f"{r['feet_ss']:.3f} [{r['q25']:.3f}, {r['q75']:.3f}]" for r in rs]+["—"]
        e,fn=(tr[x] if x else (None,None))
        def J(key, fmt): return " / ".join(fmt(r[key]) for r in rs)
        cells=[k,seed,'L' if h=='l' else 'R',cls,J('n',str),fss[0],fss[1],
               J('lean_abs',lambda v:f"{v:+.3f}"),J('frac_lean015',lambda v:f"{100*v:.0f}%"),J('sweep',lambda v:f"{v:.3f}"),
               J('cad',lambda v:f"{v:.2f}"),J('stall',lambda v:f"{100*v:.0f}%"),J('fall_hold',lambda v:f"{100*v:.0f}%"),
               '—' if e is None else f"{e:.3f}",'—' if fn is None else f"{fn:.3f}"]
        print("| "+" | ".join(cells)+" |")
