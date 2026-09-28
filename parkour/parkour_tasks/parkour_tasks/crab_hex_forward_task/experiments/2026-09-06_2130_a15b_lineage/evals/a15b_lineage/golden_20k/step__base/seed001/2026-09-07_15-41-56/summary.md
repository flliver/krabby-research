```
=== crab-hex gait eval ===
scenario   : step__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_15-10-31/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5404353322779754  p25=0.4877356054829898  p75=0.5971377226324235
tippy_tap_fraction median=0.25499671268902036
slip_ratio         median=0.23987589867530323
tracking_ratio     median=0.46847585449456397  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.58
terminations={'schedule_complete': 58, 'fall': 42}

by hold:
     stand: cmd 0.00 -> achieved 0.058 m/s | tripod median=0.579722176080712 (n=100)
     creep: cmd 0.25 -> achieved 0.126 m/s | tripod median=0.5664079897063292 (n=100)
       low: cmd 0.35 -> achieved 0.143 m/s | tripod median=0.5320418181314275 (n=72)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
