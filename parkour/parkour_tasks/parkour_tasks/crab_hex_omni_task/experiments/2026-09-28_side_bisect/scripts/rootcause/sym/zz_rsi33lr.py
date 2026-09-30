import sys, json, numpy as np
sys.argv=[sys.argv[0]]
import extract
item=("C_lr_rsi33lr","twoway","/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect/lr/C_lr_rsi33/side22_25lr__omni/seed001/2026-09-30_17-42-44")
extract.one(item)
import sym
sym.CLASS["C_lr_rsi33lr"]=("SLOW","SLOW")
from symlib import load
for lab in ("lat_l_022","lat_l_025","lat_r_022","lat_r_025"):
    d=load("C_lr_rsi33lr",lab)
    if d is None: print("no",lab); continue
    o,p,ph=sym.row("C_lr_rsi33lr",lab,d)
    print(lab,"n",o["n"],"feet",round(o["feet_ss"],3),"sweep",round(o["sweep"],3),"phase",o["phase"],
          "rod odd frac",round(o["act_rod"]["odd_pi"],3),"rod ODD rms",round(o["act_rod"]["odd_pattern_rms"],3),
          "rod amp",round(o["act_rod"]["amp"],3),"even",round(o["act_rod"]["amp"]*(1-o["act_rod"]["odd_pi"])**0.5,3),"static odd rod",round(o["act_rod"]["static_odd"],3),
          "static share",round(o["act_rod"]["static_share_of_odd"],3),"lean",round(o["lean_abs"],3))
