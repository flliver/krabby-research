import sys,re
def terms(path):
    lines=open(path).read().split('\n')
    # find top-level 'rewards:' section
    out={}; sec=None; cur=None; func=None
    for l in lines:
        if re.match(r'^\S',l): sec=l.rstrip(':'); cur=None; continue
        if sec!='rewards': continue
        m=re.match(r'^  (\w+):\s*(null)?$',l)
        if m: cur=m.group(1); out[cur]={'w':None,'f':None,'null':bool(m.group(2))}; continue
        m=re.match(r'^    weight:\s*(\S+)',l)
        if m and cur: out[cur]['w']=m.group(1)
        m=re.match(r'^    func:\s*(\S+)',l)
        if m and cur: out[cur]['f']=m.group(1).split(':')[-1]
    return out
a=terms(sys.argv[1]); b=terms(sys.argv[2])
for k in sorted(set(a)|set(b)):
    x=a.get(k,{}); y=b.get(k,{})
    fx=x.get('f'); fy=y.get('f'); 
    flag='' if (x.get('w')==y.get('w') and fx==fy) else '  <<'
    print(f"{k:45s} {str(x.get('w')):10s} {str(y.get('w')):10s} {fx if fx==fy else str(fx)+' | '+str(fy)}{flag}")
