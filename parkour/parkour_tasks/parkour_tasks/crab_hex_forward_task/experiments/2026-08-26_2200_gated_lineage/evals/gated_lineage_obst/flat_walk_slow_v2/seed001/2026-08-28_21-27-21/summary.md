```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-27_01-06-53/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.46465813720365595  p25=0.404935957251922  p75=0.511057354126361
tippy_tap_fraction median=0.2547237076648841
slip_ratio         median=0.30719417893854195
tracking_ratio     median=0.4411127672577341  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.65
terminations={'schedule_complete': 65, 'fall': 35}

by hold:
     stand: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.2909017192804434 (n=100)
     creep: cmd 0.25 -> achieved 0.127 m/s | tripod median=0.6036356031981014 (n=100)
       low: cmd 0.35 -> achieved 0.129 m/s | tripod median=0.5053047310320372 (n=83)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
