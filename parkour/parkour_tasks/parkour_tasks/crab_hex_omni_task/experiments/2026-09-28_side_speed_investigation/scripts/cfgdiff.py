import re,sys
def grab(p):
    txt=open(p).read().split('\n')
    out={}
    # commands block
    s=[i for i,l in enumerate(txt) if l.startswith('commands:')][0]
    block=[]
    for l in txt[s+1:]:
        if l and not l.startswith(' '): break
        block.append(l)
    b='\n'.join(block)
    for key in ['resampling_time_range','slot_probs','speed_ellipse','facing_delta','dir_band','cmd_pool','ang_vel_max','heading_control_stiffness','lin_vel_x','heading_offset','stand_frac','rsi_source','relay_hyst','stand_hold']:
        m=re.search(r'\n\s+'+key+r': (.*)((\n\s+- .*)*)',b)
        if m: out['cmd.'+key]=(m.group(1)+' '+' '.join(x.strip('- ').strip() for x in m.group(2).split('\n') if x.strip())).strip()
    m=re.search(r'ang_vel_clip: (\S+)',b); out['cmd.ang_vel_clip']=m.group(1) if m else None
    # scan whole file for a few other keys
    full='\n'.join(txt)
    for key in ['rsi_frac','rsi_source','episode_length_s','facing_obs_clip','clock_hide','source','bank_path']:
        ms=re.findall(r'\n\s+'+key+r': (.*)',full)
        if ms: out['any.'+key]=ms[:3]
    return out
runs=sys.argv[1:]
res=[grab(r) for r in runs]
keys=sorted(set().union(*res))
for k in keys:
    print(k)
    for r,x in zip(runs,res): print('   %-40s %s'%(r.split('/')[-3][:40], x.get(k)))
