```
=== crab-hex gait eval ===
scenario   : slow__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_00-09-00/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.45490667995024536  p25=0.4049505530830941  p75=0.49069069461481196
tippy_tap_fraction median=0.3377221856484529
slip_ratio         median=0.5220255559507456
tracking_ratio     median=0.6218145059892313  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.0 (n=96)
     creep: cmd 0.25 -> achieved 0.170 m/s | tripod median=0.5804009501193894 (n=100)
       low: cmd 0.35 -> achieved 0.194 m/s | tripod median=0.744334762447385 (n=91)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
