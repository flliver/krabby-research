```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.46008588920517757  p25=0.39532783078889466  p75=0.49468502728808383
tippy_tap_fraction median=0.2512626262626263
slip_ratio         median=0.20104162831709577
tracking_ratio     median=0.5645079679388212  (achieved/commanded vx, walking holds; n=91)
schedule_completion_rate=0.51
terminations={'schedule_complete': 51, 'fall': 49}

by hold:
     stand: cmd 0.00 -> achieved 0.111 m/s | tripod median=0.4458457192537589 (n=100)
     creep: cmd 0.25 -> achieved 0.155 m/s | tripod median=0.4450122469573952 (n=91)
       low: cmd 0.35 -> achieved 0.163 m/s | tripod median=0.4524939467387216 (n=61)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
