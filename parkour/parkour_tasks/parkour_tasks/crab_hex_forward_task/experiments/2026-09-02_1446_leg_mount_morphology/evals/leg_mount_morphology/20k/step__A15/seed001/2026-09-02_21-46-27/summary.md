```
=== crab-hex gait eval ===
scenario   : step__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6198327399864874  p25=0.5553522220654815  p75=0.6717922518935509
tippy_tap_fraction median=0.23353293413173654
slip_ratio         median=0.21280401931033005
tracking_ratio     median=0.39515912047965107  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}

by hold:
     stand: cmd 0.00 -> achieved 0.067 m/s | tripod median=0.6778903990937069 (n=100)
     creep: cmd 0.25 -> achieved 0.110 m/s | tripod median=0.6143211587130024 (n=100)
       low: cmd 0.35 -> achieved 0.129 m/s | tripod median=0.6045885790598013 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
