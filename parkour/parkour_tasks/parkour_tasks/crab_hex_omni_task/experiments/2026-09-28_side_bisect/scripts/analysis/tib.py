from common import *
import numpy as np
def roll(s,w=50):
    s=np.nan_to_num(s,nan=np.nanmean(s)); c=np.convolve(s,np.ones(w)/w,mode='valid'); return np.concatenate([np.full(w-1,np.nan),c])
for x,n,c in ARMS:
    if x=='X13': continue
    t=-roll(ser(x,'Episode_Reward/penalty_tibia_deviation_in_stance'))
    first=int(np.argmax(t<=0.020)) if (t<=0.020).any() else -1
    frac=np.mean(t[700:]<=0.020)
    mn=np.nanmin(t[100:]); amn=int(np.nanargmin(t[100:]))+100
    # bin 100 its min/argmin before 1000 and value at 1500
    print('%-8s %s first|tib|<=.020: %5d  frac(700-3000)<=.020: %.2f  min %.4f@%4d  |tib|@700 %.4f @1500 %.4f @2999 %.4f'%(n,c,first,frac,mn,amn,t[699],t[1499],t[2999]))
