```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_10-45-00/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5782434500555218  p25=0.5358605101538322  p75=0.6218883901092394
tippy_tap_fraction median=0.2530883025773982
slip_ratio         median=0.21213536161291482
tracking_ratio     median=0.5458609351280314  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.55
terminations={'schedule_complete': 55, 'fall': 45}

by hold:
     stand: cmd 0.00 -> achieved 0.062 m/s | tripod median=0.5886312752272541 (n=100)
     creep: cmd 0.25 -> achieved 0.149 m/s | tripod median=0.6078130713182837 (n=98)
       low: cmd 0.35 -> achieved 0.161 m/s | tripod median=0.5476712389051835 (n=68)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
