```
=== crab-hex gait eval ===
scenario   : fwd__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5868265421868413  p25=0.5634450373125238  p75=0.6065053903706419
tippy_tap_fraction median=0.27712418300653596
slip_ratio         median=0.244157960574531
tracking_ratio     median=0.3106797761429265  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
       low: cmd 0.30 -> achieved 0.122 m/s | tripod median=0.6458317229190276 (n=100)
       mid: cmd 0.47 -> achieved 0.139 m/s | tripod median=0.5906454720361722 (n=100)
      high: cmd 0.65 -> achieved 0.149 m/s | tripod median=0.5284131190952845 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
