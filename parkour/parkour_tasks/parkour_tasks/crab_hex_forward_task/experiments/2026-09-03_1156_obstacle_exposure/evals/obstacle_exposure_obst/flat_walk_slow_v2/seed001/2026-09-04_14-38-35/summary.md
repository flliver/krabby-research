```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_08-18-59/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5421039245380114  p25=0.48350466370366807  p75=0.5838198983716987
tippy_tap_fraction median=0.26239754098360657
slip_ratio         median=0.22454554516530395
tracking_ratio     median=0.49638232852555964  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.46
terminations={'schedule_complete': 46, 'fall': 54}

by hold:
     stand: cmd 0.00 -> achieved 0.085 m/s | tripod median=0.5857288233210207 (n=100)
     creep: cmd 0.25 -> achieved 0.139 m/s | tripod median=0.5856490627313693 (n=99)
       low: cmd 0.35 -> achieved 0.146 m/s | tripod median=0.38029163313697734 (n=76)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
