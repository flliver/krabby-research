from common import *
import numpy as np
def roll(s,w=50):
    s=np.nan_to_num(s,nan=np.nanmean(s)); c=np.convolve(s,np.ones(w)/w,mode='valid'); return np.concatenate([np.full(w-1,np.nan),c])
print('%-9s c  tr200  tr300 tr_pk(it) trFin  plat95 slope500-1500/1k  far_fin far_plat95 far_slope/1k  std>=1.99  symL300 symLfin  vfFin'%'arm')
for x,n,c in ARMS:
    if x=='X13': continue
    tr=roll(ser(x,'Metrics/base_parkour/track_ratio_mean')); fin=np.nanmean(ser(x,'Metrics/base_parkour/track_ratio_mean')[2500:])
    p95=int(np.argmax(tr>=0.95*fin)); pk=int(np.nanargmax(tr[:1500]))
    sl=(tr[1499]-tr[499])
    far=roll(ser(x,'Metrics/base_parkour/how_far_from_start_point')); ff=np.nanmean(ser(x,'Metrics/base_parkour/how_far_from_start_point')[2500:])
    fp=int(np.argmax(far>=0.95*ff)); fs=far[1499]-far[499]
    std=ser(x,'Policy/mean_noise_std'); sc=int(np.argmax(std>=1.99)) if (std>=1.99).any() else -1
    sy=ser(x,'Loss/symmetry'); vf=ser(x,'Loss/value_function')
    print('%-9s %s  %.3f  %.3f %.3f(%4d) %.3f  %5d   %+.3f            %.2f   %5d     %+.2f        %5d     %.4f  %.4f  %.4f'%(n,c,tr[199],tr[299],tr[pk],pk,fin,p95,sl,ff,fp,fs,sc,np.nanmean(sy[250:350]),np.nanmean(sy[2500:]),np.nanmean(vf[2500:])))
print(max(np.nanmax(ser(x,'Policy/mean_noise_std')) for x,_,_ in ARMS))
