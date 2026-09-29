import re,sys
def rewards(p):
    lines=open(p).read().split('\n')
    start=[i for i,l in enumerate(lines) if l.startswith('rewards:')][0]
    out={};cur=None
    for l in lines[start+1:]:
        if l and not l.startswith(' '): break
        m=re.match(r'^  ([a-zA-Z_0-9]+):',l)
        if m: cur=m.group(1); out[cur]={'p':{}}; continue
        m=re.match(r'^    func: (.*)',l)
        if m: out[cur]['func']=m.group(1).split(':')[-1]
        m=re.match(r'^    weight: (.*)',l)
        if m: out[cur]['w']=m.group(1)
        m=re.match(r'^      ([a-z_0-9]+): ([^\s].*)$',l)
        if m and cur: out[cur]['p'][m.group(1)]=m.group(2)
        m=re.match(r'^        joint_names: (.*)',l)
        if m: out[cur]['p']['joint_names']=m.group(1)
    return out
a=rewards(sys.argv[1]); b=rewards(sys.argv[2])
for k in sorted(set(a)|set(b)):
    x=a.get(k,{});y=b.get(k,{})
    fx=(x.get('func'),x.get('w'));fy=(y.get('func'),y.get('w'))
    px={kk:v for kk,v in x.get('p',{}).items() if kk not in('command_name','parkour_name')}
    py={kk:v for kk,v in y.get('p',{}).items() if kk not in('command_name','parkour_name')}
    if fx!=fy or px!=py:
        print(k, fx, fy)
        if px!=py: print('    A',px,'\n    B',py)
