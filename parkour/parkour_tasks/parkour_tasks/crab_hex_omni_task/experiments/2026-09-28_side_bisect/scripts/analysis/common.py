import pickle, numpy as np
ARMS=[('X01','O','S'),('X03','R','S'),('X05','C_band','S'),('X06','C_rsi','S'),('X02','C','F'),('X07','C_s2','F'),('X04','C_mo','I'),('X08','C_mo_s2','F'),('X09','C_mo_s1','F'),('X10','LRfa','S'),('X11','LR4','S'),('X12','LR4_dw','D'),('X13','LR4dw_s2','p'),('X14','C_lr','S'),('X15','C_lr_10','S')]
D={x:pickle.load(open(f'tb/{x}.pkl','rb')) for x,_,_ in ARMS}
def ser(x,t):
    a=np.array(D[x].get(t,[]));
    if len(a)==0: return None
    s=np.full(int(a[:,0].max())+1,np.nan); s[a[:,0].astype(int)]=a[:,1]; return s
def win(s,lo,hi):
    if s is None or len(s)<=lo: return np.nan
    return np.nanmean(s[lo:hi])
WINS=[(0,50),(50,100),(100,200),(200,300),(300,400),(400,600),(600,900),(900,1200),(1200,1600),(1600,2000),(2000,2500),(2500,3000)]
def table(t,wins=WINS,fmt='%7.3f',arms=ARMS):
    print('==',t); print('%-9s c '%'arm'+' '.join('%7s'%f'{a}-{b}' for a,b in wins))
    for x,n,c in arms:
        s=ser(x,t); print('%-9s %s '%(n,c)+' '.join(fmt%win(s,a,b) for a,b in wins))
