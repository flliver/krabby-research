```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_01-37-43/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4799332158744442  p25=0.4522100650724278  p75=0.49851816959665546
tippy_tap_fraction median=0.25641447368421055
slip_ratio         median=0.25617698003380107
tracking_ratio     median=0.47401105119015774  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.024 m/s | tripod median=0.25373755075618043 (n=100)
     creep: cmd 0.25 -> achieved 0.129 m/s | tripod median=0.5735390862603276 (n=99)
       low: cmd 0.35 -> achieved 0.151 m/s | tripod median=0.618302926401942 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
