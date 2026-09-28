```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-28_02-04-13/model_4998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.31460909578858265  p25=0.2570705617449872  p75=0.35200166944875233
tippy_tap_fraction median=0.3269230769230769
slip_ratio         median=0.3440174837029028
tracking_ratio     median=0.4866472706055771  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.67
terminations={'fall': 33, 'schedule_complete': 67}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.0 (n=88)
     creep: cmd 0.25 -> achieved 0.126 m/s | tripod median=0.49569786910991365 (n=98)
       low: cmd 0.35 -> achieved 0.157 m/s | tripod median=0.44630268844163 (n=77)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
