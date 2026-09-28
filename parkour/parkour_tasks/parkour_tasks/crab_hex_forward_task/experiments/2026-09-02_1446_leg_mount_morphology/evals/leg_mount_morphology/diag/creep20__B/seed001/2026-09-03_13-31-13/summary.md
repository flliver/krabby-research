```
=== crab-hex gait eval ===
scenario   : creep20__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-09-02_23-43-36/model_4999.pt
episodes   : 100  unscored: 6
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5105004800908759  p25=0.4477422564987459  p75=0.552194375964689
tippy_tap_fraction median=0.3372140246378773
slip_ratio         median=0.26546116977571577
tracking_ratio     median=0.7094599593796704  (achieved/commanded vx, walking holds; n=94)
schedule_completion_rate=0.57
terminations={'schedule_complete': 57, 'fall': 43}

by hold:
   creep_a: cmd 0.25 -> achieved 0.179 m/s | tripod median=0.5095967181964032 (n=94)
   creep_b: cmd 0.25 -> achieved 0.178 m/s | tripod median=0.519824134456874 (n=69)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
