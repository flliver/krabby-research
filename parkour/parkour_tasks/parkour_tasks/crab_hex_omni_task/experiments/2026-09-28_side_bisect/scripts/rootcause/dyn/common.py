import pickle, numpy as np
ARM={'X01':'O','X02':'C','X03':'R','X04':'C_mo','X05':'C_band','X06':'C_rsi','X07':'C_s2','X08':'C_mo_s2','X09':'C_mo_s1',
 'X10':'LRfa','X11':'LR4','X12':'LR4_dwell','X14':'C_lr','X15':'C_lr_10','X16':'C_lr_rsi','X17':'C_lr_10_rsi','X19':'C_lr_rsi33','X18':'C_lr_10_rsi_s2','X13':'LR4_dwell_s2'}
FAST=['X02','X04','X07','X08','X09']; SLOW=['X01','X03','X05','X06','X10','X11','X12','X14','X15','X16','X17','X19']
D={}
def load(x):
    if x not in D: D[x]=pickle.load(open(f'/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/4b0248a2-13ef-464e-81a9-16e93543e283/scratchpad/dyn/tb/{x}.pkl','rb'))
    return D[x]
def ser(x,t):
    d=load(x)
    if t not in d: return None,None
    a=np.array(d[t]); return a[:,0],a[:,1]
def win(x,t,a,b):
    s,v=ser(x,t)
    if s is None: return np.nan
    m=(s>=a)&(s<b); return float(np.nanmean(v[m])) if m.any() else np.nan
def smooth(v,k=25):
    k=min(k,len(v)); c=np.convolve(v,np.ones(k)/k,mode='valid'); return np.concatenate([np.full(k-1,np.nan),c])
