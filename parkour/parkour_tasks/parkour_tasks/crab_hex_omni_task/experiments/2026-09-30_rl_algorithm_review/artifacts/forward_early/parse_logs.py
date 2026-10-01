import re, sys, math, glob, os
ITER=re.compile(r'Learning iteration (\d+)/(\d+)')
KEYS={'Mean reward':'rew','Mean value_function':'vf','Mean entropy':'ent','Mean surrogate':'surr','Mean symmetry':'sym','Mean episode length':'eplen','Computation':'sps','Episode_Termination/crab_failure':'fail'}
def parse(path):
    rows=[];cur=None
    with open(path,errors='ignore') as f:
        for line in f:
            m=ITER.search(line)
            if m:
                if cur: rows.append(cur)
                cur={'it':int(m.group(1)),'n':int(m.group(2))};continue
            if cur is None: continue
            s=line.strip()
            for k,v in KEYS.items():
                if s.startswith(k+':'):
                    val=s.split(':',1)[1].strip().split()[0]
                    try: cur[v]=float(val)
                    except: pass
    if cur: rows.append(cur)
    return rows
def sig(ent,d=18): return math.exp(ent/d-0.5*math.log(2*math.pi*math.e))
def main():
    for p in sys.argv[1:]:
      r=parse(p)
      if not r: print(os.path.basename(p),'no rows');continue
      n=len(r)
      pick=[0,min(50,n-1),min(200,n-1),min(1000,n-1),n//2,n-1]
      out=[]
      for i in pick:
          x=r[i]; e=x.get('ent',float('nan'))
          out.append(f"it{x['it']}:ent={e:.2f}(sig~{sig(e):.2f}) vf={x.get('vf',float('nan')):.4f} rew={x.get('rew',float('nan')):.2f} ep={x.get('eplen',float('nan')):.0f} fail={x.get('fail',float('nan')):.3f}")
      # first iter where sigma>=1.99
      first=next((x['it'] for x in r if 'ent' in x and sig(x['ent'])>=1.99),None)
      sps=[x['sps'] for x in r if 'sps' in x]
      print(os.path.basename(os.path.dirname(p))[:30]+'/'+os.path.basename(p), 'rows',n,'first_sig>=1.99 at',first,'median sps',sorted(sps)[len(sps)//2] if sps else None)
      for o in out: print('   ',o)

if __name__=="__main__": main()
